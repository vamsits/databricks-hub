"""
update_site.py
──────────────
Scrapes Databricks release notes, blog, and changelog sources,
then uses Gemini API to intelligently update index.html.

Outputs GitHub Actions step output:
  updated=true  → site was changed, ready to commit
  updated=false → no meaningful changes, skip commit
"""

import os
import sys
import json
import hashlib
import datetime
import requests
import feedparser
from bs4 import BeautifulSoup
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
SITE_FILE       = "index.html"
CACHE_FILE      = ".github/scripts/.content_cache.json"
MAX_FEED_ITEMS  = 10   # max items to pull from each RSS feed
GEMINI_MODEL    = "gemini-1.5-flash"

# Sources to scrape
SOURCES = {
    # ── Core Databricks ───────────────────────────────────────────────────────
    "release_notes": {
        "url": "https://docs.databricks.com/release-notes/product/index.html",
        "type": "html",
        "description": "Official Databricks product release notes"
    },
    "databricks_blog": {
        "url": "https://www.databricks.com/feed",
        "type": "rss",
        "description": "Databricks official blog feed"
    },
    "runtime_releases": {
        "url": "https://docs.databricks.com/release-notes/runtime/index.html",
        "type": "html",
        "description": "Databricks Runtime version release notes"
    },
    "delta_releases": {
        "url": "https://github.com/delta-io/delta/releases.atom",
        "type": "rss",
        "description": "Delta Lake GitHub releases"
    },
    "mlflow_releases": {
        "url": "https://github.com/mlflow/mlflow/releases.atom",
        "type": "rss",
        "description": "MLflow GitHub releases"
    },

    # ── Databricks SQL ────────────────────────────────────────────────────────
    "dbsql_release_notes": {
        "url": "https://docs.databricks.com/aws/en/sql/release-notes/2025",
        "type": "html",
        "description": "Databricks SQL release notes 2025 — new SQL features, warehouse changes, behavioral updates"
    },
    "dbsql_query_optimization": {
        "url": "https://docs.databricks.com/aws/en/optimizations/",
        "type": "html",
        "description": "Databricks query optimization — predicate pushdown, dynamic file pruning, AQE, low-shuffle merge, Z-order, Liquid Clustering"
    },
    "dbsql_performance_best_practices": {
        "url": "https://docs.databricks.com/aws/en/lakehouse-architecture/performance-efficiency/best-practices",
        "type": "html",
        "description": "Databricks SQL performance best practices — caching, Photon, serverless warehouses, managed tables, UDF guidance"
    },
    "dbsql_query_federation_perf": {
        "url": "https://docs.databricks.com/gcp/en/query-federation/performance-recommendations",
        "type": "html",
        "description": "Databricks SQL query federation — join pushdown, predicate pushdown for Redshift/Snowflake/BigQuery/PostgreSQL"
    },

    # ── PySpark ───────────────────────────────────────────────────────────────
    "pyspark_best_practices": {
        "url": "https://spark.apache.org/docs/latest/api/python/tutorial/pandas_on_spark/best_practices.html",
        "type": "html",
        "description": "Official PySpark best practices — Arrow optimization, pandas API on Spark, memory config, checkpointing"
    },
    "spark_releases": {
        "url": "https://spark.apache.org/news/",
        "type": "html",
        "description": "Apache Spark release announcements — latest Spark versions, new PySpark features, SQL improvements"
    },
    "pyspark_github_releases": {
        "url": "https://github.com/apache/spark/releases.atom",
        "type": "rss",
        "description": "Apache Spark GitHub releases — version tags and release notes for Spark/PySpark"
    },
    "databricks_pyspark_migration": {
        "url": "https://docs.databricks.com/aws/en/migration/",
        "type": "html",
        "description": "Databricks migration and upgrade guides — PySpark version compatibility, Spark 3.x to 4.x migration"
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[update_site] {msg}", flush=True)


def set_output(key: str, value: str):
    """Write a GitHub Actions step output."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        # Local testing fallback
        print(f"OUTPUT: {key}={value}")


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def hash_content(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Scrapers ──────────────────────────────────────────────────────────────────

def scrape_html(url: str, description: str) -> str:
    """Scrape key text content from an HTML page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DatabricksHubBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove nav, footer, scripts, styles for cleaner text
        for tag in soup(["nav", "footer", "script", "style", "aside"]):
            tag.decompose()

        # Extract headings + first paragraph under each for structure
        sections = []
        for heading in soup.find_all(["h1", "h2", "h3"])[:30]:
            text = heading.get_text(strip=True)
            if text:
                next_p = heading.find_next_sibling("p")
                excerpt = next_p.get_text(strip=True)[:300] if next_p else ""
                sections.append(f"## {text}\n{excerpt}")

        result = f"### {description} ({url})\n\n" + "\n\n".join(sections)
        log(f"  ✓ HTML scraped: {url} ({len(sections)} sections)")
        return result

    except Exception as e:
        log(f"  ⚠ Failed to scrape {url}: {e}")
        return f"### {description}\n(Could not fetch — {e})"


def scrape_rss(url: str, description: str) -> str:
    """Parse an RSS/Atom feed and return recent items."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:MAX_FEED_ITEMS]:
            title   = entry.get("title", "Untitled")
            summary = entry.get("summary", entry.get("content", [{}])[0].get("value", ""))
            # Strip HTML tags from summary
            summary = BeautifulSoup(summary, "html.parser").get_text()[:400]
            date    = entry.get("published", entry.get("updated", ""))
            items.append(f"- **{title}** ({date})\n  {summary}")

        result = f"### {description} ({url})\n\n" + "\n\n".join(items)
        log(f"  ✓ RSS scraped: {url} ({len(items)} items)")
        return result

    except Exception as e:
        log(f"  ⚠ Failed to parse RSS {url}: {e}")
        return f"### {description}\n(Could not fetch — {e})"


def gather_all_content() -> str:
    """Scrape all sources and combine into one context block."""
    log("Scraping Databricks sources...")
    parts = [
        f"# Databricks Content Refresh — {datetime.date.today().isoformat()}\n"
    ]
    for key, source in SOURCES.items():
        log(f"  Fetching: {key}")
        if source["type"] == "rss":
            parts.append(scrape_rss(source["url"], source["description"]))
        else:
            parts.append(scrape_html(source["url"], source["description"]))

    return "\n\n---\n\n".join(parts)


# ── Gemini ────────────────────────────────────────────────────────────────────

def call_gemini(current_html: str, new_content: str) -> str:
    """
    Send the current site HTML + scraped content to Gemini.
    Gemini returns the updated HTML with the latest information woven in.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    today = datetime.date.today().isoformat()

    prompt = f"""You are an expert Databricks technical writer and web developer.

I have a single-file Databricks Knowledge Hub website (HTML/CSS/JS) that needs to be updated
with the latest information from official Databricks and Apache Spark sources.

## YOUR TASK

Update the provided HTML file by applying ALL of the following changes:

1. **What's New / Latest Updates section** — Add or refresh a "Latest Updates" section near
   the top of the page with the most important recent Databricks announcements from the
   scraped release notes. Use a card or accordion style matching the existing design.
   Include the version number and date for each item where available.

2. **Databricks SQL best practices** — Using the scraped content from the SQL optimization
   and performance best practices sources, update or expand the SQL sections with:
   - Latest predicate pushdown behavior and when it applies
   - Dynamic file pruning updates
   - New SQL warehouse types or serverless improvements
   - Query result caching guidance
   - Any new DBSQL version-specific features (e.g. 2025.xx releases)
   - New SQL syntax or functions added recently

3. **PySpark best practices** — Using the scraped Apache Spark and PySpark sources, update
   or expand the PySpark/Spark tuning sections with:
   - Latest Spark version (currently 4.x series — note any new Spark 4.x PySpark features)
   - Arrow optimization updates
   - pandas API on Spark improvements
   - Any new PySpark functions, optimizations, or deprecations
   - Spark 4.x migration notes if relevant

4. **Glossary** — Add any new important terms from the scraped content not already present
   (e.g. new feature names, new runtime versions, new SQL functions, new Spark concepts).

5. **Version numbers** — Update any outdated version numbers for Databricks Runtime,
   Spark, Delta Lake, MLflow, or PySpark found in the existing HTML.

6. **Blog highlights** — In the Resources section, refresh with 2–3 recent blog post
   titles and brief descriptions from the scraped Databricks blog feed.

7. **Footer / meta** — Update the "Last updated" date to {today}.

## STRICT RULES

- Return ONLY the complete, valid HTML file. No markdown, no explanation, no code fences.
- Preserve ALL existing CSS, JavaScript, design tokens, dark/light mode logic, and interactive features.
- Do NOT remove any existing sections or content — only add or update.
- Do NOT change the overall design, color scheme, fonts, or layout.
- Keep the file self-contained (no new external dependencies).
- If the scraped content has no meaningful new information, return the original HTML unchanged
  except for updating the last-updated date in the footer.
- The output must be a complete, working HTML file from <!DOCTYPE html> to </html>.
- For code snippets, use the existing syntax highlighting CSS classes: .kw .str .cm .fn .num .var .op

## SCRAPED CONTENT (latest from Databricks SQL, PySpark, and official sources)

{new_content[:18000]}

## CURRENT SITE HTML

{current_html}
"""

    log("Calling Gemini API...")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,        # Low temp = more faithful, less creative drift
            max_output_tokens=8192, # Gemini 1.5 Pro supports up to 8192 output tokens
        )
    )

    result = response.text.strip()

    # Strip any accidental markdown fences Gemini might add
    if result.startswith("```html"):
        result = result[7:]
    if result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]

    return result.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Validate API key
    if not GEMINI_API_KEY:
        log("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    # Load current site
    if not os.path.exists(SITE_FILE):
        log(f"ERROR: {SITE_FILE} not found. Make sure it's in the repo root.")
        sys.exit(1)

    with open(SITE_FILE, encoding="utf-8") as f:
        current_html = f.read()

    log(f"Loaded site: {SITE_FILE} ({len(current_html):,} chars)")

    # Scrape all sources
    new_content = gather_all_content()
    content_hash = hash_content(new_content)

    # Check cache — skip Gemini call if scraped content hasn't changed
    cache = load_cache()
    force = os.environ.get("INPUT_FORCE_UPDATE", "false").lower() == "true"

    if not force and cache.get("content_hash") == content_hash:
        log("No new content detected since last run. Skipping Gemini update.")
        set_output("updated", "false")
        return

    log(f"New content detected (hash: {content_hash[:8]}). Calling Gemini...")

    # Call Gemini to update the site
    updated_html = call_gemini(current_html, new_content)

    # Sanity check — make sure we got valid HTML back
    if not updated_html.startswith("<!DOCTYPE") and not updated_html.startswith("<html"):
        log("ERROR: Gemini returned invalid HTML. Aborting update.")
        log(f"First 200 chars of response: {updated_html[:200]}")
        set_output("updated", "false")
        sys.exit(1)

    # Check if anything actually changed
    if hash_content(updated_html) == hash_content(current_html):
        log("Gemini returned identical content. No update needed.")
        set_output("updated", "false")
        # Still save the content hash so we don't re-scrape next time
        cache["content_hash"] = content_hash
        cache["last_run"] = datetime.date.today().isoformat()
        save_cache(cache)
        return

    # Write updated site
    with open(SITE_FILE, "w", encoding="utf-8") as f:
        f.write(updated_html)

    log(f"✅ Site updated successfully ({len(updated_html):,} chars)")

    # Save cache
    cache["content_hash"] = content_hash
    cache["last_run"] = datetime.date.today().isoformat()
    save_cache(cache)

    set_output("updated", "true")


if __name__ == "__main__":
    main()

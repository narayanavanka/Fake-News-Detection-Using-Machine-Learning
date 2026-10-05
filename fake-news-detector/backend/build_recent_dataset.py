import argparse
import calendar
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import feedparser
import requests
from dotenv import load_dotenv

from backend.feeds import FEEDS
from backend.scraper import ScraperError, scrape_article
from backend.verify import FACTCHECK_URL, _rate

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "recent_test_set.csv"
TIMEOUT = 10
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
DEFAULT_QUERIES = [
    "india", "world", "politics", "sports", "health", "technology",
    "economy", "war", "election", "climate", "celebrity", "viral",
]


def _iso_from_struct(struct_time) -> str:
    if not struct_time:
        return ""
    return datetime.fromtimestamp(calendar.timegm(struct_time), tz=timezone.utc).isoformat()


def _iso_or_blank(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return value


def fetch_recent_feed_entries(per_outlet: int = 2) -> List[Dict]:
    entries, seen = [], set()
    for outlet, _, urls in FEEDS:
        picked = 0
        for feed_url in urls:
            if picked >= per_outlet:
                break
            try:
                resp = requests.get(feed_url, headers=USER_AGENT, timeout=TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException:
                continue
            parsed = feedparser.parse(resp.content)
            for item in parsed.entries:
                url = (item.get("link") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                entries.append(
                    {
                        "outlet": outlet,
                        "url": url,
                        "title": (item.get("title") or "").strip(),
                        "published_at": _iso_from_struct(item.get("published_parsed") or item.get("updated_parsed")),
                    }
                )
                picked += 1
                if picked >= per_outlet:
                    break
    return entries


def _scrape_real(entry: Dict) -> Optional[Dict]:
    try:
        article = scrape_article(entry["url"])
    except ScraperError:
        return None
    return {
        "expected_label": "Real",
        "sample_type": "real_article",
        "input_mode": "manual",
        "title": article["title"],
        "text": article["text"],
        "url": article["url"],
        "publisher": entry["outlet"],
        "published_at": article.get("publish_date") or entry["published_at"],
        "notes": f"Trusted newsroom feed: {entry['outlet']}",
    }


def fetch_real_samples(limit: int) -> List[Dict]:
    entries = fetch_recent_feed_entries(per_outlet=max(2, limit // max(1, len(FEEDS)) + 1))
    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_scrape_real, entry) for entry in entries]
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows[:limit]


def _factcheck_search(query: str) -> List[Dict]:
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_FACTCHECK_API_KEY is not set.")
    resp = requests.get(
        FACTCHECK_URL,
        timeout=TIMEOUT,
        params={"query": query, "key": api_key, "languageCode": "en", "pageSize": 10},
    )
    resp.raise_for_status()
    return resp.json().get("claims", [])


def fetch_fake_samples(limit: int, real_rows: List[Dict]) -> List[Dict]:
    topical_queries = [row["title"] for row in real_rows[: min(8, len(real_rows))] if row.get("title")]
    queries = list(dict.fromkeys(DEFAULT_QUERIES + topical_queries))
    rows, seen = [], set()
    for query in queries:
        try:
            claims = _factcheck_search(query[:200])
        except requests.RequestException:
            continue
        for claim in claims:
            for review in claim.get("claimReview", []):
                rating = _rate(review.get("textualRating", ""))
                if rating != "false":
                    continue
                claim_text = (claim.get("text") or review.get("title") or "").strip()
                if len(claim_text.split()) < 5:
                    continue
                url = (review.get("url") or "").strip()
                key = (claim_text.lower(), url)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "expected_label": "Fake",
                        "sample_type": "fake_claim",
                        "input_mode": "manual",
                        "title": claim_text,
                        "text": (review.get("title") or claim_text).strip(),
                        "url": url,
                        "publisher": (review.get("publisher") or {}).get("name", ""),
                        "published_at": _iso_or_blank(review.get("reviewDate")),
                        "notes": f"Fact-check rating: {review.get('textualRating', '')}",
                    }
                )
                if len(rows) >= limit:
                    return rows
    return rows


def write_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(exist_ok=True)
    fieldnames = [
        "expected_label",
        "sample_type",
        "input_mode",
        "title",
        "text",
        "url",
        "publisher",
        "published_at",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Build a recent evaluation CSV from live sources.")
    ap.add_argument("--real-limit", type=int, default=20, help="Number of recent real articles to scrape.")
    ap.add_argument("--fake-limit", type=int, default=20, help="Number of recent debunked claims to collect.")
    ap.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path.")
    args = ap.parse_args()

    real_rows = fetch_real_samples(args.real_limit)
    fake_rows = fetch_fake_samples(args.fake_limit, real_rows)
    rows = real_rows + fake_rows
    if not rows:
        raise SystemExit("No recent samples were collected.")

    write_csv(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Real rows: {len(real_rows)}")
    print(f"Fake rows: {len(fake_rows)}")


if __name__ == "__main__":
    main()

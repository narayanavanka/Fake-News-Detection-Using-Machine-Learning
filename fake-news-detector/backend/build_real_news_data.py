import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import feedparser
import pandas as pd
import requests

from backend.scraper import scrape_article
from backend.train_model import clean_text

log = logging.getLogger("fakenews.realdata")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# outlet, domain, feed urls  (all public RSS; broad topical coverage)
EXTRA_FEEDS = [
    ("BBC News", "bbc.com", ["https://feeds.bbci.co.uk/sport/rss.xml",
                             "https://feeds.bbci.co.uk/news/politics/rss.xml",
                             "https://feeds.bbci.co.uk/news/education/rss.xml",
                             "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"]),
    ("The Guardian", "theguardian.com", ["https://www.theguardian.com/uk/sport/rss",
                                         "https://www.theguardian.com/uk/business/rss",
                                         "https://www.theguardian.com/uk/politics/rss",
                                         "https://www.theguardian.com/uk/technology/rss",
                                         "https://www.theguardian.com/uk/media/rss"]),
    ("NPR", "npr.org", ["https://feeds.npr.org/1006/rss.xml",   # business
                        "https://feeds.npr.org/1015/rss.xml",   # national
                        "https://feeds.npr.org/1007/rss.xml"]),  # world
    ("Al Jazeera", "aljazeera.com", ["https://www.aljazeera.com/xml/rss/all.xml"]),
    ("CNN", "cnn.com", ["http://rss.cnn.com/rss/edition.rss", "http://rss.cnn.com/rss/edition_world.rss",
                        "http://rss.cnn.com/rss/edition_sport.rss", "http://rss.cnn.com/rss/money_news.rss"]),
    ("Sky News", "skynews.com sky.com", ["https://feeds.skynews.com/feeds/rss/world.xml",
                                         "https://feeds.skynews.com/feeds/rss/uk.xml",
                                         "https://feeds.skynews.com/feeds/rss/business.xml"]),
    ("CBS News", "cbsnews.com", ["https://www.cbsnews.com/latest/rss/politics",
                                 "https://www.cbsnews.com/latest/rss/moneywatch",
                                 "https://www.cbsnews.com/latest/rss/sports"]),
    ("The Hindu", "thehindu.com", ["https://www.thehindu.com/sport/feeder/default.rss",
                                   "https://www.thehindu.com/business/feeder/default.rss",
                                   "https://www.thehindu.com/sci-tech/feeder/default.rss"]),
    ("Indian Express", "indianexpress.com", ["https://indianexpress.com/section/sports/feed/",
                                             "https://indianexpress.com/section/business/feed/"]),
    ("NDTV", "ndtv.com", ["https://feeds.feedburner.com/ndtvsports-latest",
                          "https://feeds.feedburner.com/ndtvnews-india-news"]),
    ("Hindustan Times", "hindustantimes.com", ["https://www.hindustantimes.com/feeds/rss/sports/rssfeed.xml",
                                               "https://www.hindustantimes.com/feeds/rss/business/rssfeed.xml"]),
    ("Times of India", "timesofindia.indiatimes.com", ["https://timesofindia.indiatimes.com/rssfeeds/3945318.cms",   # sports
                                                       "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms"]),  # business
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}


def fetch_feed_urls(feed_url, cap=40):
    r = requests.get(feed_url, timeout=10, headers=HEADERS)
    r.raise_for_status()
    out = []
    for e in feedparser.parse(r.content).entries[:cap]:
        link = e.get("link")
        if link:
            out.append(link)
    return out


def collect(limit):
    jobs = [url for _, _, feeds in EXTRA_FEEDS for url in feeds]
    log.info("Fetching %d feeds for URLs...", len(jobs))
    urls = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(fetch_feed_urls, u): u for u in jobs}
        for fut in as_completed(futs):
            try:
                urls += fut.result()
            except Exception as exc:
                log.info("feed fetch failed: %s (%s)", futs[fut], exc)
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    log.info("%d unique article URLs; scraping up to %d...", len(urls), limit)

    rows = []

    def work(u):
        try:
            d = scrape_article(u)
            text = (d.get("text") or "").strip()
            if len(text.split()) < 80:
                return None
            return {"title": d.get("title") or "", "text": text, "url": u}
        except Exception:
            return None

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(work, u) for u in urls[: limit * 2]]
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if done % 50 == 0:
                log.info("scraped %d/%d...", done, min(len(urls), limit * 2))
            if r:
                rows.append(r)
                if len(rows) >= limit:
                    break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["content"] = (df["title"] + " " + df["text"]).map(clean_text)
        df = df[df["content"].str.split().str.len() >= 40]
        df = df.drop_duplicates(subset="content")
    return df


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "real_news_modern.csv"))
    ap.add_argument("--limit", type=int, default=1500)
    args = ap.parse_args()
    df = collect(args.limit)
    if df.empty:
        raise SystemExit("No articles could be scraped.")
    # merge with any previous file to accumulate across runs
    out = Path(args.out)
    if out.is_file():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["url"])
        if "content" not in df.columns:
            df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)
        df = df[df["content"].str.split().str.len() >= 40]
    df[["title", "text", "outlet", "url"] if "outlet" in df.columns else ["title", "text", "url"]].to_csv(out, index=False)
    print(f"Saved {len(df)} total modern real articles to {out}")


if __name__ == "__main__":
    main()

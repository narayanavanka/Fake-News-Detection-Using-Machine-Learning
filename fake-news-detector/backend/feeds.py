import calendar
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import feedparser
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

log = logging.getLogger("fakenews.feeds")
REFRESH_SECONDS = 300
TIMEOUT = 8
SKIP = ENGLISH_STOP_WORDS | {"breaking", "update", "exclusive", "watch", "video", "live"}

# (outlet, its website domains, feed URLs). Edit freely: add feeds, remove dead ones.
# Reuters and AP have no official feeds, and CNN's feeds stopped updating in 2023.
FEEDS = [
    ("BBC News", "bbc.com bbc.co.uk", ["https://feeds.bbci.co.uk/news/rss.xml", "https://feeds.bbci.co.uk/news/world/rss.xml",
                                       "https://feeds.bbci.co.uk/news/business/rss.xml", "https://feeds.bbci.co.uk/news/technology/rss.xml",
                                       "https://feeds.bbci.co.uk/news/health/rss.xml"]),
    ("The Guardian", "theguardian.com", ["https://www.theguardian.com/world/rss"]),
    ("NPR", "npr.org", ["https://feeds.npr.org/1001/rss.xml", "https://feeds.npr.org/1004/rss.xml"]),
    ("Al Jazeera", "aljazeera.com", ["https://www.aljazeera.com/xml/rss/all.xml"]),
    ("DW", "dw.com", ["https://rss.dw.com/rdf/rss-en-all"]),
    ("France 24", "france24.com", ["https://www.france24.com/en/rss"]),
    ("The Washington Post", "washingtonpost.com", ["https://feeds.washingtonpost.com/rss/world", "https://feeds.washingtonpost.com/rss/national"]),
    ("The New York Times", "nytimes.com", ["https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"]),
    ("CBS News", "cbsnews.com", ["https://www.cbsnews.com/latest/rss/main"]),
    ("NBC News", "nbcnews.com", ["https://feeds.nbcnews.com/nbcnews/public/news"]),
    ("ABC News (Australia)", "abc.net.au", ["https://www.abc.net.au/news/feed/51120/rss.xml"]),
    ("The Hindu", "thehindu.com", ["https://www.thehindu.com/news/national/feeder/default.rss", "https://www.thehindu.com/news/international/feeder/default.rss"]),
    ("Indian Express", "indianexpress.com", ["https://indianexpress.com/feed/"]),
    ("NDTV", "ndtv.com", ["https://feeds.feedburner.com/ndtvnews-top-stories"]),
    ("Times of India", "timesofindia.indiatimes.com", ["https://timesofindia.indiatimes.com/rssfeedstopstories.cms"]),
    ("Hindustan Times", "hindustantimes.com", ["https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"]),
]

_state = {"items": [], "updated": 0.0, "ok": 0, "total": 0}
_lock = threading.Lock()


def tokens(text, limit=None):
    """Lowercase keywords without stop words; a trailing 's' is trimmed (railways -> railway)."""
    out = []
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(w) > 2 and w not in SKIP:
            w = w[:-1] if len(w) > 3 and w.endswith("s") else w
            if w not in out:
                out.append(w)
    return out[:limit] if limit else out


def _fetch(outlet, domains, url):
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"})
    r.raise_for_status()
    items = []
    for e in feedparser.parse(r.content).entries[:60]:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        summary = re.sub(r"<[^>]+>", " ", e.get("summary") or "")
        stamp = e.get("published_parsed") or e.get("updated_parsed")
        items.append({"outlet": outlet, "domains": domains.split(), "title": title, "url": e.get("link", ""),
                      "time": calendar.timegm(stamp) if stamp else None, "words": set(tokens(f"{title} {summary}"))})
    return items


def refresh():
    """Reload every feed (in parallel) and swap in the new headlines."""
    jobs = [(o, d, u) for o, d, urls in FEEDS for u in urls]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [(job, pool.submit(_fetch, *job)) for job in jobs]
    items, ok = [], 0
    for job, fut in futures:
        try:
            items += fut.result()
            ok += 1
        except Exception as exc:
            log.info("Feed skipped (%s): %s", job[2], exc)
    with _lock:
        if ok:  # if everything failed (e.g. offline), keep the previous headlines
            _state.update(items=items, updated=time.time())
        _state.update(ok=ok, total=len(jobs))
    log.info("RSS refresh: %d/%d feeds loaded, %d headlines", ok, len(jobs), len(items))


def start_background_refresh():
    def loop():
        while True:
            try:
                refresh()
            except Exception:
                log.exception("RSS refresh failed")
            time.sleep(REFRESH_SECONDS)
    threading.Thread(target=loop, daemon=True, name="rss-refresh").start()


def find_matches(words, own_host=None, limit=5):
    """Headlines sharing most keywords with the article. One per outlet; the article's own outlet is skipped."""
    with _lock:
        items, meta = _state["items"], dict(_state)
    meta.pop("items")
    need = max(2, math.ceil(0.6 * len(words)))
    best = {}
    for it in items:
        if own_host and any(own_host == d or own_host.endswith("." + d) for d in it["domains"]):
            continue
        score = len(words & it["words"])
        key = (score, it["time"] or 0)
        if score >= need and (it["outlet"] not in best or key > best[it["outlet"]][0]):
            best[it["outlet"]] = (key, it)
    ranked = sorted(best.values(), key=lambda b: b[0], reverse=True)
    return [it for _, it in ranked[:limit]], meta

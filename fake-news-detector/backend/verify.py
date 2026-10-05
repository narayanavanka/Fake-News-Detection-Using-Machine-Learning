import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from backend.anthropic_verify import anthropic_ready, web_verify as anthropic_web_verify
from backend.feeds import SKIP as _SKIP, find_matches, tokens
from backend.gemini_verify import gemini_ready, web_verify as gemini_web_verify

load_dotenv()

log = logging.getLogger("fakenews.verify")
TIMEOUT = 10
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GNEWS_URL = "https://gnews.io/api/v4/search"

# Well-known news organisations. Edit freely. Being listed is not a guarantee of accuracy.
TRUSTED = set("""
apnews.com reuters.com bbc.com bbc.co.uk npr.org pbs.org theguardian.com
nytimes.com washingtonpost.com wsj.com bloomberg.com ft.com economist.com
aljazeera.com dw.com france24.com cnn.com nbcnews.com cbsnews.com abcnews.go.com
usatoday.com politico.com axios.com time.com cbc.ca abc.net.au forbes.com
skynews.com independent.co.uk thehill.com cnbc.com euronews.com lapresse.ca
thehindu.com indianexpress.com hindustantimes.com ndtv.com
timesofindia.indiatimes.com economictimes.indiatimes.com livemint.com
business-standard.com indiatoday.in deccanherald.com tribuneindia.com
theprint.in scroll.in pib.gov.in aninews.in
cricinfo.com espncricinfo.com espn.com goal.com athletic.com
nature.com sciencedaily.com newscientist.com
""".split())


def _ev(name, status, summary, vote=0, items=None):
    return {"name": name, "status": status, "summary": summary, "vote": vote, "items": items or []}


def _guard(name):
    """A failing service is skipped, never crashes the request."""
    def deco(fn):
        def wrap(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                log.warning("%s check failed: %s", name, exc)
                return _ev(name, "off", "This service didn't respond, so it was skipped.")
        return wrap
    return deco


def _ttl(seconds=600):
    """Thread-safe TTL caching decorator to save free-tier quota."""
    def deco(fn):
        store = {}
        lock = threading.Lock()
        def wrap(arg):
            with lock:
                hit = store.get(arg)
                if hit and time.time() - hit[0] < seconds:
                    return hit[1]
            value = fn(arg)
            with lock:
                store[arg] = (time.time(), value)
            return value
        return wrap
    return deco


def domain_of(url):
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_trusted(host):
    return any(host == d or host.endswith("." + d) for d in TRUSTED)


def keywords(text, n=5):
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 2 and w.lower() not in _SKIP]
    return " ".join(words[:n]) if words else text[:50]


@_guard("Web verification")
def anthropic_check(title, text, url):
    name = "Web verification"
    if not (anthropic_ready() or gemini_ready()):
        return _ev(name, "off", "Not set up. Add GEMINI_API_KEY (free) or ANTHROPIC_API_KEY to .env to enable web verification.")
    if not (title or text):
        return _ev(name, "off", "A headline or article body is needed for web verification.")

    # Anthropic (paid) wins when its key is present; Gemini free tier is the default.
    if anthropic_ready():
        result = anthropic_web_verify(title, text, url)
    else:
        result = gemini_web_verify(title, text, url)
    verdict = result["verdict"]
    items = [
        {
            "title": s["title"],
            "url": s["url"],
            "source": s["publisher"],
            "rating": s["stance"],
        }
        for s in result.get("sources", [])
    ]
    notes = f" {result['notes']}" if result.get("notes") else ""
    if verdict == "verified":
        return _ev(name, "good", (result.get("summary") or "Current reporting supports the claim.") + notes, 2, items)
    if verdict == "contradicted":
        return _ev(name, "bad", (result.get("summary") or "Current reporting contradicts the claim.") + notes, -2, items)
    return _ev(name, "neutral", (result.get("summary") or "Current reporting could not verify the claim.") + notes, 0, items)


# 1) Source ---------------------------------------------------------------
def check_source(host):
    if not host:
        return _ev("Source", "neutral", "No link was given, so the publisher wasn't checked.")
    if is_trusted(host):
        return _ev("Source", "good", f"{host} is a well-known news organisation.", 1)
    return _ev("Source", "neutral", f"{host} isn't on our short list of trusted outlets. That alone is not a warning sign.")


# 2) Fact-checks (Google Fact Check Tools API, free key) ----------------------
def _rate(text):
    t = (text or "").lower()
    if any(w in t for w in ("false", "fake", "misleading", "incorrect", "hoax", "fabricat", "pants on fire",
                            "no evidence", "not true", "unproven", "baseless", "scam", "satire", "wrong")):
        return "false"
    if any(w in t for w in ("partly", "half", "mixed", "mixture", "unclear", "unverified", "missing context", "exaggerat")):
        return "mixed"
    return "true" if any(w in t for w in ("true", "correct", "accurate")) else "other"


@_ttl()
def _search_factchecks(query):
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_FACTCHECK_API_KEY is not set.")
    r = requests.get(FACTCHECK_URL, timeout=TIMEOUT, params={
        "query": query, "key": api_key, "languageCode": "en", "pageSize": 5})
    r.raise_for_status()
    return r.json().get("claims", [])


@_guard("Fact-checks")
def fact_check(query):
    name = "Fact-checks"
    if not os.getenv("GOOGLE_FACTCHECK_API_KEY"):
        return _ev(name, "off", "Not set up. Add a free GOOGLE_FACTCHECK_API_KEY to .env to search fact-checking sites.")
    items = []
    for claim in _search_factchecks(query[:200]):
        for rev in claim.get("claimReview", [])[:1]:
            items.append({"title": rev.get("title") or claim.get("text", ""), "url": rev.get("url", ""),
                          "source": (rev.get("publisher") or {}).get("name", ""), "rating": rev.get("textualRating", "")})
    if not items:
        return _ev(name, "neutral", "No matching fact-checks. Most stories have never been fact-checked, so this says little either way.")
    kinds = {_rate(i["rating"]) for i in items}
    note = " Open the links to confirm they cover the same claim."
    if "false" in kinds:
        return _ev(name, "bad", "A related claim was rated false or misleading." + note, -1, items)
    if kinds == {"true"}:
        return _ev(name, "good", "Related claims were rated true." + note, 1, items)
    return _ev(name, "neutral", "Related fact-checks exist, with mixed or unclear ratings." + note, 0, items)


# 3) Live coverage (GDELT: free, no key, ~15 min updates; GNews: optional backup) --
_gdelt_lock, _last_gdelt = threading.Lock(), 0.0


@_ttl()
def _gdelt(query):
    global _last_gdelt
    with _gdelt_lock:  # GDELT asks for at most 1 request per 5 seconds
        time.sleep(max(0.0, 5.2 - (time.time() - _last_gdelt)))
        _last_gdelt = time.time()
    r = requests.get(GDELT_URL, timeout=TIMEOUT, params={
        "query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
        "maxrecords": 25, "sort": "datedesc", "timespan": "7d"})
    r.raise_for_status()
    return [{"title": a.get("title", ""), "url": a.get("url", ""), "domain": a.get("domain", "")}
            for a in r.json().get("articles", [])]


@_ttl()
def _gnews(query):
    gnews_key = os.getenv("GNEWS_API_KEY")
    if not gnews_key:
        raise ValueError("GNEWS_API_KEY is not set.")
    r = requests.get(GNEWS_URL, timeout=TIMEOUT, params={
        "q": query, "lang": "en", "max": 10, "apikey": gnews_key})
    r.raise_for_status()
    return [{"title": a.get("title", ""), "url": a.get("url", ""), "domain": domain_of(a.get("url", ""))}
            for a in r.json().get("articles", [])]


@_guard("Live coverage")
def coverage(title, own_host):
    name = "Live coverage"
    query = keywords(title)
    if not query:
        return _ev(name, "off", "A headline is needed to search for other coverage.")
    found, failed = [], False
    try:
        found = _gdelt(query)
    except Exception as exc:
        log.warning("GDELT failed: %s", exc)
        failed = True
    if not found and os.getenv("GNEWS_API_KEY"):
        try:
            found, failed = _gnews(query), False
        except Exception as exc:
            log.warning("GNews failed: %s", exc)
    if not found and failed:
        return _ev(name, "off", "The news search didn't respond, so coverage wasn't checked.")
    seen, items = set(), []
    for a in found:
        host = a["domain"].lower().removeprefix("www.")
        if host and host != own_host and host not in seen:
            seen.add(host)
            items.append({"title": a["title"], "url": a["url"], "source": host, "trusted": is_trusted(host)})
    items.sort(key=lambda i: not i["trusted"])
    trusted = sum(i["trusted"] for i in items)
    s = "" if trusted == 1 else "s"
    if trusted:
        return _ev(name, "good", f"Reported by {trusted} trusted outlet{s} ({len(items)} sources in total, last 7 days).", 1, items[:5])
    if items:
        return _ev(name, "neutral", f"Found {len(items)} other source(s) in the last 7 days, but none on our trusted list.", 0, items[:5])
    return _ev(name, "warn", "No other outlet was found reporting this in the last 7 days. It may be too new or local, or made up.", -1)


# 4) Trusted newsroom feeds (RSS, refreshed every 5 minutes; see feeds.py) ---------
def _ago(ts):
    if not ts:
        return "time unknown"
    m = max(0, int((time.time() - ts) // 60))
    return f"{m} min ago" if m < 90 else f"{m // 60} h ago" if m < 2880 else f"{m // 1440} days ago"


@_guard("Trusted feeds")
def newsroom_feeds(title, own_host):
    name = "Trusted feeds"
    words = set(tokens(title, 6))
    if len(words) < 2:
        return _ev(name, "off", "A longer headline is needed to match against newsroom feeds.")
    matches, meta = find_matches(words, own_host)
    if not meta["updated"]:
        return _ev(name, "off", "The newsroom feeds are still loading. Try again in a few seconds.")
    if not matches:
        return _ev(name, "neutral", f"No matching headline in the latest posts of {meta['ok']} live feeds. "
                                    "Feeds only hold recent stories, so an older story won't appear here.")
    items = [{"title": m["title"], "url": m["url"], "source": f"{m['outlet']}, {_ago(m['time'])}", "trusted": True} for m in matches]
    s = "" if len(items) == 1 else "s"
    return _ev(name, "good", f"{len(items)} trusted outlet{s} published a matching headline in a live feed.", 1, items)


# Combine ----------------------------------------------------------------
def assess(style, evidence):
    # Keep the writing-style model as a weak signal only.
    strength = 1 if style["confidence"] >= 80 else 0
    ml_vote = strength if style["label"] == "Real" else -strength
    votes = [{"name": "Writing style", "vote": ml_vote}]
    votes += [{"name": e["name"], "vote": e["vote"]} for e in evidence]

    web_vote = next((e["vote"] for e in evidence if e["name"] == "Web verification"), 0)
    source_vote = next((e["vote"] for e in evidence if e["name"] == "Source"), 0)

    # The style model is dataset-specific (WELFake: 2016 US politics) and confidently
    # wrong on out-of-domain copy (e.g. cricket coverage on a trusted outlet).
    # A trusted outlet's source check is direct evidence about the publisher, so a
    # lone negative style vote must not be able to outvote it.
    if ml_vote < 0 and source_vote > 0:
        votes[0] = {**votes[0], "vote": 0,
                    "note": "Style model says Fake, but the publisher is a verified trusted outlet. "
                            "The style model was trained on 2016 US political news and misreads "
                            "out-of-domain copy, so its negative vote is dropped here."}
    score = sum(v["vote"] for v in votes)

    has_bad_factcheck = any(e["name"] == "Fact-checks" and e["vote"] < 0 for e in evidence)
    has_trusted_source = any(e["name"] in ("Source", "Trusted feeds") and e["vote"] > 0 for e in evidence)

    if web_vote >= 2:
        level, headline = "reliable", "Verified by Current Reporting"
    elif web_vote <= -2:
        level, headline = "unreliable", "Contradicted by Current Reporting"
    # Guardrail: Prevent declaring "Unreliable" on raw text alone if no negative fact-check or trusted source was triggered
    elif score <= -2 and not has_bad_factcheck and not has_trusted_source:
        level, headline = "mixed", "Unverified or Mixed Signals"
    elif score >= 2:
        level, headline = "reliable", "Likely Reliable"
    elif score <= -2:
        level, headline = "unreliable", "Likely Unreliable"
    else:
        level, headline = "mixed", "Mixed Signals"

    return {"level": level, "headline": headline, "score": score, "votes": votes}


def verify(title, text, url, style):
    query = title.strip() or re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0][:150]
    host = domain_of(url) if url else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        web = pool.submit(anthropic_check, title, text, url)
        facts, news = pool.submit(fact_check, query), pool.submit(coverage, query, host)
        live = pool.submit(newsroom_feeds, query, host)
        web, facts, news, live = web.result(), facts.result(), news.result(), live.result()

    if live["vote"] > 0:
        if news["vote"] < 0:  # news search simply lags behind live RSS feeds
            news = _ev(news["name"], "neutral", "The news search hasn't indexed this yet (it lags about 15 minutes), "
                                                "but a trusted feed already lists it.")
        elif news["vote"] > 0:  # avoid double counting same evidence
            news = {**news, "vote": 0, "summary": news["summary"] + " (Not scored again: the feed check already counted it.)"}

    evidence = [web, check_source(host), live, news, facts]
    return {"overall": assess(style, evidence), "evidence": evidence}

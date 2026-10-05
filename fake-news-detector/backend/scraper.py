"""Fetch a live news URL and extract title + body (newspaper3k, BeautifulSoup fallback)."""
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from newspaper import Article
except Exception:  # newspaper3k missing or broken -> BeautifulSoup only
    Article = None

log = logging.getLogger("fakenews.scraper")
MIN_WORDS = 50
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)", "Accept-Language": "en"}


class ScraperError(Exception):
    status_code = 422


class InvalidURLError(ScraperError):
    status_code = 400


class FetchError(ScraperError):
    status_code = 502


class FetchTimeoutError(FetchError):
    status_code = 504


class ExtractionError(ScraperError):
    status_code = 422


def validate_url(url: str) -> str:
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InvalidURLError("Enter a full article link starting with http:// or https://.")
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except socket.gaierror:
        raise FetchError(f"Could not find the site '{parts.hostname}'. Check the link for typos.")
    for info in infos:  # block localhost / private network targets
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise InvalidURLError("Links to private or local addresses are not allowed.")
    return url


def fetch_html(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=(5, 15))
        resp.raise_for_status()
    except requests.Timeout:
        raise FetchTimeoutError("The site took too long to respond. Try again or paste the text.")
    except requests.HTTPError as exc:
        raise FetchError(f"The site refused the request (HTTP {exc.response.status_code}).")
    except requests.RequestException:
        raise FetchError("Could not reach that site. Check the link and your connection.")
    ctype = resp.headers.get("Content-Type", "")
    if ctype and "html" not in ctype.lower():
        raise ExtractionError("That link is not a web page.")
    return resp.text


def _extract_newspaper(url, html):
    if Article is None:
        return None
    try:
        art = Article(url)
        art.download(input_html=html)
        art.parse()
        date = art.publish_date.isoformat() if art.publish_date else None
        return {"title": art.title or "", "text": art.text or "",
                "authors": list(art.authors or []), "publish_date": date}
    except Exception as exc:
        log.info("newspaper3k failed (%s); falling back to BeautifulSoup", exc)
        return None


def _extract_bs4(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe"]):
        tag.decompose()
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"]
    elif soup.h1:
        title = soup.h1.get_text(" ", strip=True)
    else:
        title = soup.title.get_text(strip=True) if soup.title else ""
    box = soup.find("article") or soup
    paras = [p.get_text(" ", strip=True) for p in box.find_all("p")]
    return {"title": title.strip(), "text": "\n".join(p for p in paras if len(p.split()) >= 6),
            "authors": [], "publish_date": None}


def scrape_article(url: str) -> dict:
    url = validate_url(url)
    html = fetch_html(url)
    method = "newspaper3k"
    data = _extract_newspaper(url, html)
    if not data or len(data["text"].split()) < MIN_WORDS:
        method, data = "beautifulsoup", _extract_bs4(html)
    if len(data["text"].split()) < MIN_WORDS:
        raise ExtractionError(
            "Couldn't find enough article text at that link. It may be paywalled or load its "
            "content with JavaScript. Paste the text instead."
        )
    return {**data, "url": url, "domain": urlparse(url).hostname, "method": method}

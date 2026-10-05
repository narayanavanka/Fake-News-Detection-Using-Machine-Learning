import json
import os
from typing import Any, Dict, List, Optional

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"
TIMEOUT = (10, 90)
MAX_BODY_CHARS = 8000


def gemini_ready() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return key


def _model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def _article_payload(title: str, text: str, url: Optional[str]) -> str:
    clipped = (text or "").strip()
    if len(clipped) > MAX_BODY_CHARS:
        clipped = clipped[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + "..."
    return json.dumps(
        {"url": url or "", "title": (title or "").strip(), "article_text": clipped},
        ensure_ascii=False,
    )


def _prompt(title: str, text: str, url: Optional[str]) -> str:
    return (
        "You verify news claims against current web evidence. Search the web before answering. "
        "Prefer primary sources, official statements, and established reporting. "
        "If evidence is mixed, missing, or too weak, return verdict='unverified'. "
        "Do not guess. If the article cannot be confirmed confidently, return unverified. "
        "Your final answer must be ONLY valid JSON with this schema:\n"
        '{"verdict":"verified|contradicted|unverified",'
        '"summary":"string",'
        '"claims":["string"],'
        '"sources":[{"title":"string","url":"string","publisher":"string","stance":"supports|contradicts|context"}],'
        '"notes":"string"}\n'
        "Keep summary and notes concise. Up to 4 searches.\n"
        "Article JSON:\n"
        f"{_article_payload(title, text, url)}"
    )


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Gemini response did not contain JSON.")
        return json.loads(cleaned[start : end + 1])


def _grounding_sources(candidate: Dict[str, Any]) -> List[Dict[str, str]]:
    """Sources Gemini actually used for grounding, from the response metadata."""
    meta = candidate.get("groundingMetadata") or {}
    out = []
    for chunk in meta.get("groundingChunks") or []:
        web = chunk.get("web") or {}
        uri = (web.get("uri") or "").strip()
        if uri.startswith(("http://", "https://")):
            out.append({"title": (web.get("title") or uri).strip(), "url": uri})
    return out


def web_verify(title: str, text: str, url: Optional[str]) -> Dict[str, Any]:
    endpoint = f"{API_BASE}/{_model()}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": _prompt(title, text, url)}]}],
        "tools": [{"google_search": {}}],  # free-tier Google Search grounding
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1200},
    }
    resp = requests.post(
        endpoint,
        params={"key": _api_key()},
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:400]
        raise RuntimeError(detail or f"Gemini API error {resp.status_code}")
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:300]}")
    candidate = candidates[0]

    text = "".join(
        part.get("text", "") for part in (candidate.get("content") or {}).get("parts", [])
    )
    parsed = _extract_json(text)
    verdict = parsed.get("verdict")
    if verdict not in {"verified", "contradicted", "unverified"}:
        raise ValueError(f"Unexpected Gemini verdict: {verdict}")

    sources: List[Dict[str, str]] = []
    seen = set()
    for src in parsed.get("sources", [])[:5]:
        u = (src.get("url") or "").strip()
        if u.startswith(("http://", "https://")) and u not in seen:
            seen.add(u)
            sources.append(
                {
                    "title": (src.get("title") or u).strip(),
                    "url": u,
                    "publisher": (src.get("publisher") or "").strip(),
                    "stance": (src.get("stance") or "context").strip(),
                }
            )
    for src in _grounding_sources(candidate):  # real grounding links as backup
        if src["url"] not in seen and len(sources) < 5:
            seen.add(src["url"])
            sources.append({**src, "publisher": "", "stance": "context"})

    return {
        "verdict": verdict,
        "summary": (parsed.get("summary") or "").strip(),
        "sources": sources,
        "notes": (parsed.get("notes") or "").strip(),
    }

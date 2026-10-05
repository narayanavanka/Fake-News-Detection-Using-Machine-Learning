"""Anthropic web-search verification for live article claims."""
import json
import os
from typing import Any, Dict, List, Optional

import requests

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 4}
TIMEOUT = (10, 90)
MAX_BODY_CHARS = 8000
MAX_CONTINUATIONS = 2


def anthropic_ready() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _headers() -> Dict[str, str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    return {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }


def _article_payload(title: str, text: str, url: Optional[str]) -> str:
    clipped = (text or "").strip()
    if len(clipped) > MAX_BODY_CHARS:
        clipped = clipped[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + "..."
    return json.dumps(
        {
            "url": url or "",
            "title": (title or "").strip(),
            "article_text": clipped,
        },
        ensure_ascii=False,
    )


def _system_prompt() -> str:
    return (
        "You verify news claims against current web evidence. Use web search before answering. "
        "Prefer primary sources, official statements, and established reporting. "
        "If evidence is mixed, missing, or too weak, return verdict='unverified'. "
        "Return only valid JSON with this schema: "
        '{"verdict":"verified|contradicted|unverified",'
        '"summary":"string",'
        '"claims":["string"],'
        '"sources":[{"title":"string","url":"string","publisher":"string","stance":"supports|contradicts|context"}],'
        '"notes":"string"}'
    )


def _user_prompt(title: str, text: str, url: Optional[str]) -> str:
    return (
        "Check whether the article's main factual claims are supported by current reporting.\n"
        "Use up to 4 searches.\n"
        "Do not guess. If the article cannot be confirmed confidently, return unverified.\n"
        "Keep summary and notes concise.\n"
        "Article JSON:\n"
        f"{_article_payload(title, text, url)}"
    )


def _extract_text_blocks(content: List[Dict[str, Any]]) -> str:
    parts = []
    for block in content:
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Anthropic response did not contain JSON.")
        return json.loads(cleaned[start : end + 1])


def _call_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "model": os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
        "max_tokens": 900,
        "temperature": 0,
        "system": _system_prompt(),
        "messages": messages,
        "tools": [DEFAULT_TOOL],
    }
    resp = requests.post(API_URL, headers=_headers(), json=payload, timeout=TIMEOUT)
    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:400]
        raise RuntimeError(detail or f"Anthropic API error {resp.status_code}")
    return resp.json()


def web_verify(title: str, text: str, url: Optional[str]) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [{"role": "user", "content": _user_prompt(title, text, url)}]
    response = _call_messages(messages)
    for _ in range(MAX_CONTINUATIONS):
        if response.get("stop_reason") != "pause_turn":
            break
        messages.append({"role": "assistant", "content": response["content"]})
        response = _call_messages(messages)
    if response.get("stop_reason") == "pause_turn":
        raise RuntimeError("Anthropic web verification paused too many times.")

    parsed = _extract_json(_extract_text_blocks(response.get("content", [])))
    verdict = parsed.get("verdict")
    if verdict not in {"verified", "contradicted", "unverified"}:
        raise ValueError(f"Unexpected Anthropic verdict: {verdict}")

    sources = []
    for src in parsed.get("sources", [])[:5]:
        url_value = (src.get("url") or "").strip()
        if not url_value.startswith(("http://", "https://")):
            continue
        sources.append(
            {
                "title": (src.get("title") or url_value).strip(),
                "url": url_value,
                "publisher": (src.get("publisher") or "").strip(),
                "stance": (src.get("stance") or "context").strip(),
            }
        )

    return {
        "verdict": verdict,
        "summary": (parsed.get("summary") or "").strip(),
        "sources": sources,
        "notes": (parsed.get("notes") or "").strip(),
    }

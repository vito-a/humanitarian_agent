"""
summarize.py — Narrative section generation for reports (top summary, Why Support, Conclusion).

This module hosts the section writer that used to live in report/llm_sections.py.
It now uses the MAIN LM Studio model/key so you can leverage the larger model.

Config it uses from src/config.py:
- LMSTUDIO_BASE_URL
- LMSTUDIO_MAIN_KEY
- LMSTUDIO_MAIN_MODEL
- TEMPERATURE
- TOP_P
- MAX_TOKENS
- SEED
- LLM_SECTION_TEXT_MAX_LEN

Public API:
    generate_section_via_llm(country: str, purpose: str, corpus: List[dict]) -> str

Behavior:
- Builds a strict prompt that references ONLY the provided `corpus` (list of dicts with
  'ref', 'title', 'summary', 'date').
- Prefers the OpenAI client integration (works with LM Studio via base_url) and falls back
  to a direct HTTP call if needed.
- If anything fails, falls back to a deterministic extractive summarizer that preserves
  bracketed [n] references.

Note: We intentionally KEEP the previous HTTP helper for robustness.
"""

from __future__ import annotations
import re, sqlite3, datetime as dt
from typing import List, Dict, Optional

import requests
from openai import OpenAI  # OpenAI client (compatible with LM Studio when base_url is set)

from ..config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MAIN_KEY,
    LMSTUDIO_MAIN_MODEL,
    TEMPERATURE,
    TOP_P,
    MAIN_MODEL_MAX_TOKENS,
    MAIN_MODEL_SEED,
    LLM_SECTION_TEXT_MAX_LEN,
    DB_PATH
)

def client():
    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key = LMSTUDIO_MAIN_KEY)

BRIEF_PROMPT = """You write short humanitarian-impact digests.
Rules:
- Use plain English; short sentences.
- Only include facts explicitly present in the article text.
- Copy all numbers/dates verbatim from the article; NEVER invent or convert.
- Prefer impacts on civilians, access, energy, water, food, health.
- Avoid geopolitics unless it directly explains humanitarian impact.
- If details are unclear, say “Insufficient detail.”

Return 12-20 sentences (max ~3000 chars)."""

def brief_from_text(article_text: str) -> str:
    c = client()
    msg = [{"role":"system","content":"You are a precise summarizer."},
           {"role":"user","content":BRIEF_PROMPT + "\n\nArticle:\n" + article_text[:12000]}]
    resp = c.chat.completions.create(
        model=LMSTUDIO_MAIN_MODEL, messages=msg,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAIN_MODEL_MAX_TOKENS,
        seed=MAIN_MODEL_SEED
    )
    out = resp.choices[0].message.content.strip()
    # scrub any markdown flourishes
    out = re.sub(r"[*_`]+","", out)
    return out

def numbers_and_dates(s: str) -> list[str]:
    pats = [r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", r"\b\d+(?:\.\d+)? ?(?:%|percent)\b", r"\b\d{4}-\d{2}-\d{2}\b"]
    found = []
    for p in pats:
        found += re.findall(p, s)
    return list(dict.fromkeys(found))

def summary_for_page(page_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT url, title, text, summary, published_at FROM pages WHERE id=?", (page_id,))
        url, title, text, desc, published_at = c.fetchone()
    date_line = ""
    if published_at:
        try:
            dtp = dt.datetime.fromisoformat(published_at.replace("Z",""))
            date_line = dtp.date().isoformat()
        except Exception:
            date_line = published_at

    lead = " ".join(re.split(r"(?<=[.!?])\s+", (text or "").strip())[:2])
    brief = brief_from_text(text or "")

    parts = [p for p in [date_line, (desc or "").strip(), lead, brief] if p]
    body = "\n\n".join(parts)

    nums = numbers_and_dates(body)
    return {"url": url, "title": title, "published_at": published_at, "body": body, "numbers": nums}

# --------------------- Prompting helpers ---------------------

def _compose_llm_prompt(country: str, purpose: str, corpus: List[Dict[str, str]]) -> str:
    """
    Build a deterministic prompt using only the provided corpus.
    Each item: {'ref': int, 'title': str, 'summary': str, 'date': str}
    """
    head = (
        f"You are preparing a formal humanitarian brief about {country.title()}.\n"
        f"Draft the section: {purpose}.\n"
        "Constraints:\n"
        "- Use ONLY the facts in the provided excerpts.\n"
        "- Add bracketed reference numbers like [3] right after any factual sentence that relies on a specific item.\n"
        "- Prefer concrete numbers and dates when present. Do not invent figures.\n"
        "- Keep the tone neutral, analytical, and policy-ready.\n"
        "- 3–6 short paragraphs, each focused and readable.\n"
        "- Do not include a heading; just output the body text.\n"
    )
    lines = ["\n=== EVIDENCE EXCERPTS (with reference numbers) ==="]
    for it in corpus:
        ref = it.get("ref")
        title = (it.get("title") or "").strip()
        date = (it.get("date") or "").strip()
        summ = (it.get("summary") or "").strip()
        lines.append(f"[{ref}] {title} — {date}\n{summ}\n")
    return head + "\n".join(lines)


# --------------------- Deterministic fallback ---------------------

def _fallback_extractive_paragraphs(corpus: List[Dict[str, str]], max_chars: int = 1400) -> str:
    """
    Deterministic fallback: weave 3–4 extractive paragraphs using the top-N items.
    We include the most concrete sentences and keep [n] refs attached.
    """
    def score_text(t: str) -> int:
        s = 0
        # numbers
        s += 4 * len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", t))
        # concrete humanitarian terms
        s += 2 * len(re.findall(
            r"\b(civilian|killed|wounded|injured|drone|missile|strike|attack|hospital|clinic|ambulance|power|blackout|water|pipeline|sanitation|food|market|price|grain|displace(?:d|ment)?)\b",
            t, re.I
        ))
        return s

    chunks: List[str] = []
    budget = max_chars

    ranked = sorted(
        corpus,
        key=lambda x: score_text(x.get("summary") or "") + (10 if re.search(r"\[\d+\]", x.get("summary") or "") else 0),
        reverse=True,
    )
    for it in ranked:
        ref = it.get("ref")
        summ = (it.get("summary") or "").strip()
        if not summ.endswith(f"[{ref}]"):
            summ = (summ + f" [{ref}]").strip()
        para = summ
        if budget - len(para) < 0:
            break
        chunks.append(para)
        budget -= len(para)
        if len(chunks) >= 4:
            break
    return "\n\n".join(chunks)


# --------------------- LM Studio call (MAIN model) - OpenAI client preferred ---------------------

def _call_lmstudio_chat_main_openai(prompt: str) -> Optional[str]:
    """
    Prefer using the OpenAI client against an LM Studio-compatible endpoint.
    This respects LMSTUDIO_BASE_URL and LMSTUDIO_MAIN_KEY.
    """
    try:
        cli = client()
        res = cli.chat.completions.create(
            model=LMSTUDIO_MAIN_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful analyst. Follow constraints strictly and never invent data."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        text = (res.choices[0].message.content or "").strip()
        return text if text else None
    except Exception:
        return None


# --------------------- LM Studio call (MAIN model) - HTTP fallback (kept) ---------------------

def _call_lmstudio_chat_main(prompt: str) -> Optional[str]:
    """
    Direct HTTP call to the LM Studio-compatible Chat Completions endpoint.
    Kept as a fallback for robustness.
    """
    base = LMSTUDIO_BASE_URL.rstrip("/")
    url = f"{base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {LMSTUDIO_MAIN_KEY}" if LMSTUDIO_MAIN_KEY else "",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LMSTUDIO_MAIN_MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAIN_MODEL_MAX_TOKENS,
        "seed": MAIN_MODEL_SEED,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a careful analyst. Follow constraints strictly and never invent data."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        text = (text or "").strip()
        return text if text else None
    except Exception:
        return None

# --------------------- Public API ---------------------

def generate_section_via_llm(country: str, purpose: str, corpus: List[Dict[str, str]]) -> str:
    """
    Generate a narrative section using the MAIN LM Studio model if reachable; otherwise fall back
    to a deterministic extractive summary. Temperature and seed come from config.

    Resolution order:
      1) OpenAI client call (preferred)
      2) HTTP fallback
      3) Deterministic extractive fallback
    """
    if not corpus:
        return ""

    prompt = _compose_llm_prompt(country, purpose, corpus)

    # 1) Try OpenAI client (preferred)
    text = _call_lmstudio_chat_main_openai(prompt)
    if text:
        return text[:LLM_SECTION_TEXT_MAX_LEN].strip()

    # 2) Fallback: HTTP
    text = _call_lmstudio_chat_main(prompt)
    if text:
        return text[:LLM_SECTION_TEXT_MAX_LEN].strip()

    # 3) Deterministic fallback
    return _fallback_extractive_paragraphs(corpus)

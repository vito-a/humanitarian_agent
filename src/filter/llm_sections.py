"""
llm_sections.py — LLM helper for narrative report sections (top summary, why support, conclusion)
and the PRIMARY-TOPIC LLM gate (country + allowed sections).

Uses LM Studio-compatible Chat Completions with the model configured for sections:
- LMSTUDIO_BASE_URL
- LMSTUDIO_SECTION_KEY
- LMSTUDIO_SECTION_MODEL
- TEMPERATURE
- TOP_P
- MAX_TOKENS
- SEED

Public API:
    generate_section_via_llm(country: str, purpose: str, corpus: List[dict]) -> str
    section_titles_for_country(target_country: str) -> List[str]
    assess_primary_topic(...)-> Dict[str, Any]   # secondary LLM gate

Behavior:
- Narrative generation uses ONLY the provided corpus (with [n] refs).
- Primary-topic gate is conservative: it must confirm the article is primarily about the
  target country and matches at least one allowed section title (from keyword_bags.yaml).
"""

from __future__ import annotations
import re
import json
from typing import List, Dict, Optional, Any

import requests

from ..config import (
    LMSTUDIO_BASE_URL, LMSTUDIO_SECTION_KEY, LMSTUDIO_SECTION_MODEL, TEMPERATURE,
    TOP_P, MAX_TOKENS, SEED, LLM_SECTION_TEXT_MAX_LEN, SECTIONS_MAX_CHARS
)

# Read YAML via the same loader used elsewhere
from ..filter.topic_filter import load_bags

# --------------------- Prompting helpers (narratives) ---------------------

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
        "- 3–5 short paragraphs, each focused and readable.\n"
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

def _fallback_extractive_paragraphs(corpus: List[Dict[str, str]], max_chars: int = SECTIONS_MAX_CHARS) -> str:
    """
    Deterministic fallback: weave 3–5 extractive paragraphs using the top-N items.
    We include the most concrete sentences and keep [n] refs attached.
    """
    def score_text(t: str) -> int:
        s = 0
        s += 4 * len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", t))
        s += 2 * len(re.findall(
            r"\b(civilian|killed|wounded|injured|drone|missile|strike|hospital|clinic|ambulance|power|blackout|water|pipeline|sanitation|food|market|price|grain)\b",
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


# --------------------- LM Studio call ---------------------

def _call_lmstudio_chat(prompt: str) -> Optional[str]:
    """
    Call the LM Studio-compatible Chat Completions endpoint.
    Returns the text content on success, or None on failure.
    """
    base = LMSTUDIO_BASE_URL.rstrip("/")
    url = f"{base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {LMSTUDIO_SECTION_KEY}" if LMSTUDIO_SECTION_KEY else "",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LMSTUDIO_SECTION_MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
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
        return (text or "").strip()
    except Exception:
        return None


# --------------------- Public API (narratives) ---------------------

def generate_section_via_llm(country: str, purpose: str, corpus: List[Dict[str, str]]) -> str:
    """
    Generate a narrative section using LM Studio if reachable; otherwise fall back
    to a deterministic extractive summary. Temperature and seed come from config.
    """
    if not corpus:
        return ""

    prompt = _compose_llm_prompt(country, purpose, corpus)
    text = _call_lmstudio_chat(prompt)
    if text:
        return text[:LLM_SECTION_TEXT_MAX_LEN].strip()

    return _fallback_extractive_paragraphs(corpus)


# ===================== Helpers for PRIMARY-TOPIC GATE =====================

def section_titles_for_country(target_country: str) -> List[str]:
    """
    Resolve section titles for the target country using keyword_bags.yaml:
      - Start from global sections (bags['sections']).
      - Merge with per-country overrides (bags['countries'][cc]).
      - Title precedence: per-country > global.
      - Order: use global 'section_order' if available.
    """
    bags = load_bags()
    cc = bags.get("countries", {}).get(target_country, {}) or {}

    global_secs = (bags.get("sections") or {})
    global_keys = list(global_secs.keys())
    country_keys = [
        k for k in cc.keys()
        if isinstance(cc.get(k), dict) and ("patterns" in cc[k] or "title" in cc[k])
    ]

    merged_keys: List[str] = []
    for k in global_keys:
        if k not in merged_keys:
            merged_keys.append(k)
    for k in country_keys:
        if k not in merged_keys:
            merged_keys.append(k)

    def _title_for_key(k: str) -> str:
        cnode = cc.get(k) or {}
        gnode = global_secs.get(k) or {}
        if isinstance(cnode, dict) and cnode.get("title"):
            return str(cnode["title"]).strip()
        if isinstance(gnode, dict) and gnode.get("title"):
            return str(gnode["title"]).strip()
        return k.replace("_", " ").title()

    titles = [_title_for_key(k) for k in merged_keys if k]

    order = [str(x).strip() for x in (bags.get("section_order") or []) if str(x).strip()]
    if order:
        seen = set()
        ordered: List[str] = []
        for want in order:
            if want in titles and want not in seen:
                ordered.append(want); seen.add(want)
        for t in titles:
            if t not in seen:
                ordered.append(t); seen.add(t)
        titles = ordered

    # unique + non-empty
    out: List[str] = []
    seen2: set[str] = set()
    for t in titles:
        tt = t.strip()
        if tt and tt not in seen2:
            out.append(tt); seen2.add(tt)
    return out

# --------------------- Primary-topic LLM gate ---------------------

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _balanced_json_slice(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: return s[start:i+1]
    return None

def _try_parse_to_json(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_code_fences(text)
    try: return json.loads(raw)
    except Exception: pass
    block = _balanced_json_slice(raw)
    if block:
        try: return json.loads(block)
        except Exception: pass
    return None

def _repair_prompt(raw: str) -> str:
    raw_snip = (raw or "")[:1200]
    return (
        "Convert the following content into ONE valid JSON object with keys: "
        '["pass","confidence","sections","reason"]. Return ONLY the JSON.\n\n'
        f"Content:\n{raw_snip}"
    )

def assess_primary_topic(
    title: str,
    desc: str,
    lead: str,
    target_country: str,
    allowed_sections: List[str],
    summary_score: float,
    normalized_score: float,
    prior_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Conservative LLM gate (sections model). Excludes core/country-only cues by instruction.
    """
    allowed_list = "\n".join(f"- {s}" for s in allowed_sections) if allowed_sections else "- (none provided)"
    prior_cats = ", ".join(prior_categories or []) if prior_categories else "(none)"

    json_schema = (
        "{\n"
        '  "pass": true or false,\n'
        '  "confidence": 0-100,\n'
        '  "sections": ["one or more of the allowed sections or empty if none match"],\n'
        '  "reason": "short string explaining the decision"\n'
        "}"
    )

    prompt = (
        f"You are a strict gatekeeper. Decide if this article's PRIMARY topic is about **{target_country.title()}** and\n"
        "whether it concerns one or more of the allowed sections.\n\n"
        f"Allowed section titles:\n{allowed_list}\n\n"
        "Context scores from a previous keyword gate:\n"
        f"- summary_score: {summary_score}\n"
        f"- normalized_score: {normalized_score}\n"
        f"- prior_categories: {prior_cats}\n\n"
        "Important exclusions:\n"
        "- Ignore generic 'core' or country-only cues (e.g., country names, demonyms, leaders' names, generic 'war'/'conflict').\n"
        "- Require sector/impact signals related to civilians, healthcare, energy/infrastructure, water/sanitation, or food/markets.\n\n"
        "Requirements:\n"
        f"- The article must primarily be about {target_country.title()} (not just mentions).\n"
        "- It must relate to at least one allowed section (use the given titles).\n"
        "- Pop culture, memes, entertainment listicles, and unrelated topics must be rejected.\n"
        "- It is about the harm done to the target country, its civilians, health and economy.\n"
        "- Any countermeasures and counterattacks done by the target country and its citizens must be rejected.\n"
        "- Focus only on the target country, the articles about the countries far away must be rejected.\n"
        "- And even the articles mentioning the closest neighbouring states, even if the topic is related, must be rejected.\n"
        "- Be conservative: if unsure, reject (pass=false).\n\n"
        "Return ONE JSON object only. Do NOT include code fences or explanations.\n"
        "The JSON MUST have exactly these keys and types:\n\n"
        f"{json_schema}\n\n"
        "Consider ONLY this text:\n"
        f"TITLE: {title}\n"
        f"DESC: {desc}\n"
        f"LEAD: {lead}\n"
    )

    # Call LM Studio
    base = LMSTUDIO_BASE_URL.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LMSTUDIO_SECTION_KEY}" if LMSTUDIO_SECTION_KEY else "",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LMSTUDIO_SECTION_MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
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
        out = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception:
        out = ""

    parsed = _try_parse_to_json(out)
    if not parsed:
        # try a one-shot repair
        try:
            resp2 = requests.post(url, headers=headers, json={**payload, "messages":[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": _repair_prompt(out)}
            ]}, timeout=60)
            resp2.raise_for_status()
            data2 = resp2.json()
            out2 = (
                data2.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = _try_parse_to_json(out2)
        except Exception:
            parsed = None

    if not parsed:
        return {"pass": False, "confidence": 0, "sections": [], "reason": "parse_error"}

    try: ok = bool(parsed.get("pass", False))
    except Exception: ok = False
    try: conf = int(parsed.get("confidence", 0))
    except Exception: conf = 0
    conf = max(0, min(100, conf))
    secs = parsed.get("sections", [])
    if not isinstance(secs, list): secs = []
    secs = [str(x).strip() for x in secs if str(x).strip()]
    reason = str(parsed.get("reason", "") or "")

    return {"pass": ok, "confidence": conf, "sections": secs, "reason": reason}

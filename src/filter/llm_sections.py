"""
llm_sections.py — Helpers for:
- resolving section titles from keyword_bags.yaml (section_titles_for_country)
- PRIMARY-TOPIC LLM gate (assess_primary_topic) using the SECTIONS model/key

Narrative section generation (generate_section_via_llm) has been moved to:
    src/extract/summarize.py
and now uses the MAIN model/key.
"""

from __future__ import annotations
import re, json
from typing import List, Dict, Optional, Any
import requests

from ..config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_SECTION_KEY,
    LMSTUDIO_SECTION_MODEL,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    SECTION_MODEL_SEED
)
from ..filter.topic_filter import load_bags

# --------------------- Section titles (YAML) ---------------------

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


# --------------------- Primary-topic LLM gate (SECTIONS model) ---------------------

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _balanced_json_slice(s: str) -> Optional[Dict[str, Any]]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _try_parse_to_json(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_code_fences(text)
    try:
        return json.loads(raw)
    except Exception:
        pass
    block = _balanced_json_slice(raw)
    if block:
        try:
            return json.loads(block)
        except Exception:
            pass
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
        "- Any articles about army, soldiers and officers, where civilian topics are not mentioned, must be rejected.\n"
        "- Any articles about the military operations where civilian topics are not mentioned, must be rejected.\n"
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

    # Call LM Studio (sections model)
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
        "seed": SECTION_MODEL_SEED,
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
        # one-shot repair
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

    # Coerce + clamp
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

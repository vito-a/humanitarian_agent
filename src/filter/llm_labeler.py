# src/filter/llm_labeler.py
import json, re, textwrap, logging
from typing import Dict, Any, Optional, List
from openai import OpenAI

from ..config import (
    LMSTUDIO_BASE_URL, LMSTUDIO_LABELER_KEY, LMSTUDIO_LABELER_MODEL,
    TEMPERATURE, TOP_P, LABELER_MODEL_MAX_TOKENS, LABELER_MODEL_SEED,
    TITLE_PROMPT_MAX_LEN, DESC_PROMPT_MAX_LEN, LEAD_PROMPT_MAX_LEN, REPAIR_PROMPT_MAX_LEN
)

# Read section titles dynamically from YAML
from .topic_filter import load_bags

log = logging.getLogger("llm_labeler")


def _client():
    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_LABELER_KEY)


# ------------------ Section titles (from YAML) ------------------

def _country_section_titles(target_country: str) -> List[str]:
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
        seen = set(); ordered: List[str] = []
        for want in order:
            if want in titles and want not in seen:
                ordered.append(want); seen.add(want)
        for t in titles:
            if t not in seen:
                ordered.append(t); seen.add(t)
        titles = ordered

    out: List[str] = []
    seen2: set[str] = set()
    for t in titles:
        tt = t.strip()
        if tt and tt not in seen2:
            out.append(tt); seen2.add(tt)
    return out


def _categories_hint_for_country(target_country: str) -> str:
    titles = _country_section_titles(target_country)
    if not titles:
        return "- Security & Civilian Impact\n- Health & Medical Services\n- Energy & Utilities\n- Water & Sanitation\n- Food & Markets"
    return "\n".join(f"- {t}" for t in titles)


# ------------------ Main classifier (unchanged keys) ------------------

def _build_prompt(title: str, desc: str, lead: str, target_country: str) -> str:
    categories_hint = _categories_hint_for_country(target_country)
    json_schema = textwrap.dedent("""\
        {{
          "main_country": "string",
          "related_countries": ["string", ...],
          "is_primary_for_target": true or false,
          "categories": ["one or more of the section titles listed below"],
          "confidence": 0-100,
          "reason": "short string"
        }}
    """)
    body = textwrap.dedent(f"""\
        You classify a news item for a country-focused brief.

        Target country: {target_country.title()}

        Return ONE JSON object only. Do NOT include code fences or explanations.
        The JSON MUST have exactly these keys and types:

        {json_schema}

        Use these section titles (if applicable) for "categories":
        {categories_hint}

        If unsure, set reasonable defaults (e.g., "Unknown", false, [], 0, "unsure").

        Consider ONLY this text:
        TITLE: {(title or '')[:TITLE_PROMPT_MAX_LEN]}
        DESC: {(desc or '')[:DESC_PROMPT_MAX_LEN]}
        LEAD: {(lead or '')[:LEAD_PROMPT_MAX_LEN]}
    """)
    return body


def _repair_prompt(raw: str) -> str:
    raw_snip = (raw or "")[:REPAIR_PROMPT_MAX_LEN]
    return textwrap.dedent(f"""\
        Convert the following content into ONE valid JSON object with keys:
        ["main_country","related_countries","is_primary_for_target","categories","confidence","reason"].
        Return ONLY the JSON (no markdown, no explanation).

        Content:
        {raw_snip}
    """)


REQUIRED_KEYS = {
    "main_country": str,
    "related_countries": list,
    "is_primary_for_target": bool,
    "categories": list,
    "confidence": int,
    "reason": str,
}


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _balanced_json_slice(s: str) -> Optional[str]:
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


def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    if "confidence" in d:
        try: d["confidence"] = int(d["confidence"])
        except Exception: d["confidence"] = 0
    if "is_primary_for_target" in d and isinstance(d["is_primary_for_target"], str):
        d["is_primary_for_target"] = d["is_primary_for_target"].strip().lower() in {"true","1","yes"}
    for k in ("related_countries","categories"):
        v = d.get(k, [])
        if not isinstance(v, list):
            v = []
        d[k] = [str(x) for x in v]
    for k, t in (("main_country", str), ("reason", str)):
        v = d.get(k)
        d[k] = "" if v is None else str(v)
    try:
        d["confidence"] = max(0, min(100, int(d.get("confidence", 0))))
    except Exception:
        d["confidence"] = 0
    return d


def _validate(d: Dict[str, Any]) -> bool:
    for k, t in REQUIRED_KEYS.items():
        if k not in d:
            return False
        if not isinstance(d[k], t):
            return False
    return True


def _try_parse_to_json(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_code_fences(text)
    try:
        d = json.loads(raw)
        return d
    except Exception:
        pass
    block = _balanced_json_slice(raw)
    if block:
        try:
            return json.loads(block)
        except Exception:
            pass
    return None


def _call_chat(prompt: str) -> str:
    cli = _client()
    res = cli.chat.completions.create(
        model=LMSTUDIO_LABELER_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY a JSON object (no markdown, no explanations)."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1 if TEMPERATURE is None else min(TEMPERATURE, 0.2),
        top_p=TOP_P,
        max_tokens=LABELER_MODEL_MAX_TOKENS,
        seed=LABELER_MODEL_SEED,
    )
    return (res.choices[0].message.content or "").strip()


# ------------------ Public: main classifier ------------------

def classify_item(title: str, desc: str, lead: str, target_country: str) -> Dict[str, Any]:
    """
    Main classifier:
      1) prompts model for strict JSON (with country-aware section titles as categories)
      2) repairs+validates output
    """
    prompt = _build_prompt(title or "", desc or "", lead or "", target_country or "")
    out = _call_chat(prompt)
    parsed = _try_parse_to_json(out)

    retry_reason = None
    if not parsed:
        retry_reason = "first_parse_failed"
        repaired = _call_chat(_repair_prompt(out))
        parsed = _try_parse_to_json(repaired)
        if not parsed:
            retry_reason = "repair_failed"

    if not parsed:
        return {
            "main_country": "Unknown",
            "related_countries": [],
            "is_primary_for_target": False,
            "categories": [],
            "confidence": 0,
            "reason": f"parse_error:{retry_reason or 'no_json'}"
        }

    parsed = _coerce_types(parsed)
    if not _validate(parsed):
        parsed.setdefault("main_country","Unknown")
        parsed.setdefault("related_countries",[])
        parsed.setdefault("is_primary_for_target",False)
        parsed.setdefault("categories",[])
        parsed.setdefault("confidence",0)
        parsed.setdefault("reason","schema_error")
        return parsed

    return parsed


# ------------------ NEW: lightweight primary-topic gate (LABELER model) ------------------

def _build_primary_relevance_prompt_light(
    title: str,
    desc: str,
    lead: str,
    target_country: str,
    allowed_sections: List[str],
    summary_score: float,
    normalized_score: float,
    prior_categories: List[str],
) -> str:
    allowed_list = "\n".join(f"- {s}" for s in allowed_sections) if allowed_sections else "- (none provided)"
    prior_cats = ", ".join(prior_categories) if prior_categories else "(none)"

    json_schema = textwrap.dedent("""\
        {{
          "pass": true or false,
          "confidence": 0-100,
          "sections": ["one or more of the allowed sections or empty if none match"],
          "reason": "short string"
        }}
    """)

    return textwrap.dedent(f"""\
        You are a strict gatekeeper. Decide if this article's PRIMARY topic is about **{target_country.title()}** and
        whether it concerns one or more of the allowed sections.

        Allowed section titles:
        {allowed_list}

        Context scores from a previous keyword gate:
        - summary_score: {summary_score}
        - normalized_score: {normalized_score}
        - prior_categories: {prior_cats}

        Important exclusions:
        - Ignore generic 'core' or country-only cues (e.g., country names, demonyms, leaders' names, generic 'war'/'conflict').
        - Require sector/impact signals related to civilians, healthcare, energy/infrastructure, water/sanitation, or food/markets.

        Requirements:
        - The article must primarily be about {target_country.title()} (not just mentions).
        - It must relate to at least one allowed section (use the given titles).
        - Pop culture, memes, entertainment listicles, and unrelated topics must be rejected.
        - It is about the harm done to the target country, its civilians, health and economy.
        - Any articles about army, soldiers and officers, where civilian topics are not mentioned, must be rejected.
        - Any articles about the military operations where civilian topics are not mentioned, must be rejected.
        - Focus only on the target country, the articles about the countries far away must be rejected.
        - And even the articles mentioning the closest neighbouring states, even if the topic is related, must be rejected.
        - Be conservative: if unsure, reject (pass=false).

        Return ONE JSON object only. Do NOT include code fences or explanations.
        The JSON MUST have exactly these keys and types:

        {json_schema}

        Consider ONLY this text:
        TITLE: {(title or '')[:TITLE_PROMPT_MAX_LEN]}
        DESC: {(desc or '')[:DESC_PROMPT_MAX_LEN]}
        LEAD: {(lead or '')[:LEAD_PROMPT_MAX_LEN]}
    """)


def assess_primary_topic_light(
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
    Lightweight LLM gate using the labeler model/key.
    Returns: {"pass": bool, "confidence": int, "sections": [str], "reason": str}
    """
    prompt = _build_primary_relevance_prompt_light(
        title or "", desc or "", lead or "", target_country or "",
        allowed_sections or [], float(summary_score or 0.0), float(normalized_score or 0.0),
        list(prior_categories or []),
    )
    out = _call_chat(prompt)
    parsed = _try_parse_to_json(out)

    if not parsed:
        repaired = _call_chat(_repair_prompt(out))
        parsed = _try_parse_to_json(repaired)

    if not parsed:
        return {"pass": False, "confidence": 0, "sections": [], "reason": "parse_error"}

    # Coerce types
    try:
        ok = bool(parsed.get("pass", False))
    except Exception:
        ok = False
    try:
        conf = int(parsed.get("confidence", 0))
    except Exception:
        conf = 0
    conf = max(0, min(100, conf))

    secs = parsed.get("sections", [])
    if not isinstance(secs, list):
        secs = []
    secs = [str(x).strip() for x in secs if str(x).strip()]

    reason = str(parsed.get("reason", "") or "")

    return {"pass": ok, "confidence": conf, "sections": secs, "reason": reason}

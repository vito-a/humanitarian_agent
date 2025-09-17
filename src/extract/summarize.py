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
import lmstudio as lms

from ..config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MAIN_KEY,
    LMSTUDIO_MAIN_MODEL,
    LMSTUDIO_SECTION_KEY,
    LMSTUDIO_SECTION_MODEL,
    SECTION_MODEL_SEED,
    SECTION_MODEL_MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    MAIN_MODEL_MAX_TOKENS,
    MAIN_MODEL_SEED,
    LLM_SECTION_TEXT_MAX_LEN,
    SUMMARIZER_BRIEF_TEXT_MAX_LEN,
    SUMMARY_PAGE_MAX_CHARS,
    LEAD_PROMPT_MAX_LEN,
    TITLE_PROMPT_MAX_LEN,
    DESC_PROMPT_MAX_LEN,
    DB_PATH
)

lms.configure_default_client(LMSTUDIO_BASE_URL)

def openai_client():
    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key = LMSTUDIO_MAIN_KEY)

def client():
    return lms.Client(LMSTUDIO_BASE_URL)

BRIEF_PROMPT = """You write short humanitarian-impact digests.
Rules:
- Use plain English; short sentences.
- Only include facts explicitly present in the article text.
- Copy all numbers/dates verbatim from the article; NEVER invent or convert.
- Prefer impacts on civilians, access, energy, water, food, health.
- Avoid geopolitics unless it directly explains humanitarian impact.
- If details are unclear, say “Insufficient detail.”

Return 12-20 sentences (max ~3000 chars)./no_think"""

# def _call_openai_chat(msg: List[Dict[str, str]]) -> str:
def _call_openai_chat(prompt: str) -> str:
    c = openai_client()
    msg = [{"role":"system","content":"You are a precise summarizer."},
          {"role":"user","content":prompt+"/no_think"}]
    resp = c.chat.completions.create(
        model=LMSTUDIO_MAIN_MODEL,
        messages=msg,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAIN_MODEL_MAX_TOKENS,
        seed=MAIN_MODEL_SEED
    )
    out = resp.choices[0].message.content.strip()
    out = re.sub(r"[*_`]+","", out)
    return out

def _call_native_chat(prompt: str) -> str:
    try:
        #cli = client()
        #model = cli.llm.model(
        model = lms.llm(LMSTUDIO_MAIN_MODEL,
            config = {
                "contextLength": MAIN_MODEL_MAX_TOKENS, # Set your desired context length here
                "max_context_length": MAIN_MODEL_MAX_TOKENS, # Set your desired context length here
                "context_length": MAIN_MODEL_MAX_TOKENS, # Set your desired context length here
                "context-length": MAIN_MODEL_MAX_TOKENS, # Set your desired context length here
                "gpu": {
                    "ratio": 1,
                },
            },
        )
        print(f"Model '{LMSTUDIO_MAIN_MODEL}' loaded with context length {MAIN_MODEL_MAX_TOKENS}.")
        res = model.respond(prompt,
            config = {"temperature": TEMPERATURE,
               "top_p": TOP_P,
               "max_tokens": MAIN_MODEL_MAX_TOKENS,
               "seed": MAIN_MODEL_SEED,
            })
        text = (res or "").strip()
        return text if text else None
    except Exception as e:
        print(f"Error loading model: {e}")

# def _call_main_chat(msg: List[Dict[str, str]]) -> str:
def _call_main_chat(prompt: str) -> str:
    #return _call_native_chat(prompt)
    return _call_openai_chat(prompt)

def brief_from_text(article_text: str) -> str:
    article_content = BRIEF_PROMPT + "\n\nArticle:\n" + article_text[:SUMMARIZER_BRIEF_TEXT_MAX_LEN]
    # msg = [{"role":"system","content":"You are a precise summarizer."},
    #       {"role":"user","content":article_content}]
    out = _call_main_chat(article_content)
    # scrub any markdown flourishes
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

# --------------------- SECTION model page-level summarizer ---------------------

def _compose_page_summary_prompt(title: str, desc: str, text: str, country: str) -> str:
    """
    Ask the smaller SECTION model to produce 3–5 factual sentences (<= ~1200 chars),
    grounded ONLY in provided content. Emphasize concrete numbers/dates and humanitarian aspects.
    """
    title = (title or "").strip()
    desc = (desc or "").strip()
    text = (text or "").strip()

    # compact the big text for the prompt
    text_compact = re.sub(r"\s+", " ", text)[:SUMMARIZER_BRIEF_TEXT_MAX_LEN]

    return (
        "You are drafting a concise factual summary for a humanitarian/conflict brief.\n"
        f"Target country: {country.title()}\n"
        "Instructions:\n"
        "- Use ONLY the provided content (no external knowledge, no speculation).\n"
        "- Provide a one-paragraph summary of the following text. The output should contain only the summary paragraph and no additional formatting.\n"
        "- Emphasize concrete facts with numbers/dates: casualties, drones/missiles launched, infrastructure damage (power, water), displacement, medical impact.\n"
        "- 3–5 short sentences (max ~1200 characters total). Neutral, precise.\n"
        "- If numbers are not present in the text, do not invent them.\n\n"
        f"TITLE: {title[:TITLE_PROMPT_MAX_LEN]}\n"
        f"DESC: {desc[:DESC_PROMPT_MAX_LEN]}\n"
        f"TEXT: {text_compact[:SUMMARY_PAGE_MAX_CHARS]}/no_think\n"
    )

def _call_section_model_openai(prompt: str) -> Optional[str]:
    try:
        cli = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_SECTION_KEY)
        res = cli.chat.completions.create(
            model=LMSTUDIO_SECTION_MODEL,
            messages=[
                {"role": "system", "content": "Return a concise factual summary grounded only in the provided text."},
                {"role": "user", "content": prompt+"/no_think"},
            ],
            temperature=TEMPERATURE,  # more deterministic
            top_p=TOP_P,
            max_tokens=SECTION_MODEL_MAX_TOKENS,
            seed=SECTION_MODEL_SEED,
        )
        text = (res.choices[0].message.content or "").strip()
        return text if text else None
    except Exception:
        return None

def _fallback_extractive_sentences(text: str, max_sentences: int = 4) -> str:
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    if not sents:
        return ""
    # score sentences: prefer those with numbers and humanitarian terms
    def _s_score(s: str) -> int:
        sc = 0
        sc += 4 * len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", s))
        sc += 2 * len(re.findall(r"\b(civilian|killed|dead|wounded|injured|drone|missile|strike|attack|hospital|clinic|ambulance|power|blackout|water|pipeline|sanitation|displace(?:d|ment)?)\b", s, re.I))
        return sc
    ranked = sorted(((i, s, _s_score(s)) for i, s in enumerate(sents)), key=lambda x: x[2], reverse=True)
    keep_idx = sorted([i for i, _, _ in ranked[:max_sentences]])
    out = " ".join(sents[i].strip() for i in keep_idx).strip()
    return out or " ".join(sents[:max_sentences]).strip()

def summarize_page_with_section_model(title: str, desc: str, text: str, country: str) -> str:
    """
    Public API: produce a brief factual summary for a single page using the SECTION model.
    Falls back to extractive sentence selection.
    """
    prompt = _compose_page_summary_prompt(title or "", desc or "", text or "", country or "")
    out = _call_section_model_openai(prompt)
    if out:
        return out[:LEAD_PROMPT_MAX_LEN].strip()
    # fallback: extractive
    return _fallback_extractive_sentences((text or "")[:SUMMARIZER_BRIEF_TEXT_MAX_LEN], max_sentences=4)

# --------------------- Prompting helpers ---------------------

def _compose_llm_prompt(country: str, purpose: str, corpus: List[Dict[str, str]]) -> str:
    """
    Build a section-specific prompt based on `purpose`.
    """

    if "Humanitarian conditions summary" in purpose:
        head = (
            f"You are preparing a humanitarian situation summary on {country.title()}.\n"
            "Your task is to synthesize the evidence below into a concise, policy-ready overview.\n"
            "Tone:\n"
            "- Neutral, analytic, evidence-led; no rhetoric.\n"
            "- Confident but cautious; avoid speculation beyond the data.\n"
            "Structure (3–6 short paragraphs):\n"
            "1) Framing overview (what’s happening, where, who is affected).\n"
            "   Example: “Over the last month, civilians in eastern districts experienced escalating strikes that disrupted power and health services [2][5].”\n"
            "2) Civilian impact (casualties, displacement, protection risks).\n"
            "   Example: “Reported incidents indicate X killed and Y injured, including Z children, with evacuations from A–B districts [3][6].”\n"
            "3) Critical systems (energy, water/sanitation, health care) with quantified disruptions.\n"
            "   Example: “Power substations and lines suffered repeated outages, causing rolling blackouts across N oblasts [4][7].”\n"
            "4) Operational constraints (access, aid delivery, security, logistics).\n"
            "   Example: “Humanitarian access remained constrained by shelling and checkpoints along key corridors [8].”\n"
            "5) Near-term trajectory (cautious, evidence-tethered).\n"
            "   Example: “Given continued strikes on grid assets, short-term outages and reduced clinic capacity are likely to persist [4][9].”\n"
            "6) Optional synthesis line to close the summary.\n"
            "   Example: “Overall, humanitarian conditions remain fragile, with essential services and civilians at elevated risk [1][5].”\n"
            "Requirements:\n"
            "- Use ONLY the evidence provided.\n"
            "- Add bracketed reference numbers like [3] immediately after factual claims.\n"
            "- Prefer concrete numbers/dates from the excerpts; NEVER invent figures.\n"
            "- Do not include a heading; output body text only.\n"
            "(no extra commentary; proceed to write the summary)\n"
        )

    elif "Why support" in purpose:
        head = (
            f"You are writing the 'Why Support' section of a humanitarian brief on {country.title()}.\n"
            "Explain—based strictly on the evidence—why the documented conditions justify support for civilians, host communities, and refugees.\n"
            "Tone:\n"
            "- Persuasive but sober; policy-ready, non-polemical.\n"
            "- Center civilian protection, essential services, and stabilization outcomes.\n"
            "Structure (3–5 short paragraphs):\n"
            "1) Problem statement anchored in concrete harm.\n"
            "   Example: “Escalating strikes caused civilian casualties and degraded essential services, heightening protection risks [2][5].”\n"
            "2) Humanitarian rationale (life-saving, dignity, protection).\n"
            "   Example: “Support is warranted to reduce preventable deaths, ensure trauma care, and protect displaced families in high-risk zones [3][6].”\n"
            "3) Systems rationale (keep energy/water/health systems from cascading failure).\n"
            "   Example: “Targeted aid to grid repairs, water pumping, and clinic supplies prevents service collapse and disease outbreaks [4][7].”\n"
            "4) Refugee/host-community rationale (regional stability, burden-sharing).\n"
            "   Example: “Assistance for refugees and hosts reduces negative coping, price shocks, and cross-border pressure [8].”\n"
            "5) Implementation guardrails (neutrality, accountability, coordination).\n"
            "   Example: “Channels should be neutral, monitored, and closely coordinated to maximize impact and limit diversion [1][9].”\n"
            "Requirements:\n"
            "- Use ONLY the evidence provided; attribute factual claims with [n].\n"
            "- Reference numbers must follow the sentence that uses the fact.\n"
            "- Avoid moralizing; argue from humanitarian principles and documented impacts.\n"
            "- Do not include a heading; output body text only.\n"
            "(no extra commentary; proceed to write the section)\n"
        )

    elif "Conclusion" in purpose:
        head = (
            f"You are writing the 'Conclusion' section of a humanitarian situation brief for {country.title()}.\n"
            "Provide a concise synthesis and a cautious forward look grounded in the evidence.\n"
            "Tone:\n"
            "- Clear, measured, forward-looking; no sensationalism.\n"
            "- Strategic but tied to documented trends.\n"
            "Structure (2–4 short paragraphs):\n"
            "1) Synthesis of the most salient findings (people, places, systems).\n"
            "   Example: “Evidence shows sustained threats to civilians and recurring damage to power and medical services in key urban areas [2][4][5].”\n"
            "2) Risk outlook (plausible near- to medium-term outcomes anchored in patterns).\n"
            "   Example: “If grid assets remain targeted, expect intermittent blackouts with knock-on effects on clinics and water pumping [4][7].”\n"
            "3) Implications for humanitarian posture (priorities, access, coordination).\n"
            "   Example: “Response should prioritize trauma care, power restoration for health facilities, WASH continuity, and safe delivery corridors [3][6][8].”\n"
            "4) Closing line that orients decision-makers.\n"
            "   Example: “Absent de-escalation, needs will likely rise; timely, well-coordinated support can mitigate the worst impacts [1][5].”\n"
            "Requirements:\n"
            "- Use ONLY the evidence provided; attach [n] after each factual claim.\n"
            "- No invented numbers/dates or external facts.\n"
            "- Do not include a heading; output body text only.\n"
            "(no extra commentary; proceed to write the conclusion)\n"
        )

    else:
        # Fallback to the generic instructions
        head = (
            f"You are preparing a formal humanitarian brief about {country.title()}.\n"
            f"Draft the section: {purpose}.\n"
            "Constraints:\n"
            "- Use ONLY the facts in the provided excerpts.\n"
            "- Add bracketed reference numbers like [3] right after any factual sentence that relies on a specific item.\n"
            "- Prefer concrete numbers and dates when present. Do not invent figures.\n"
            "- Keep the tone neutral, analytical, and policy-ready.\n"
            "- 3–6 short paragraphs.\n"
        )

    # Evidence excerpts
    lines = ["\n=== EVIDENCE EXCERPTS (with reference numbers) ==="]
    for it in corpus:
        ref = it.get("ref")
        title = (it.get("title") or "").strip()
        date = (it.get("date") or "").strip()
        summ = (it.get("summary") or "").strip()
        lines.append(f"[{ref}] {title} — {date}\n{summ}\n")

    return head + "\n".join(lines) + "\n/no_think"

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


# --------------------- LM Studio OpenAI compatible call (MAIN model) - OpenAI client preferred ---------------------

def _call_lmstudio_chat_main_openai(prompt: str) -> Optional[str]:
    """
    Prefer using the OpenAI client against an LM Studio-compatible endpoint.
    This respects LMSTUDIO_BASE_URL and LMSTUDIO_MAIN_KEY.
    """
    try:
        text = _call_main_chat(prompt)
        return text if text else None
    except Exception:
        return None

# --------------------- LM Studio native call (MAIN model) - OpenAI client preferred ---------------------

def _call_lmstudio_chat_main_native(prompt: str) -> Optional[str]:
    """
    Prefer using the OpenAI client against an LM Studio-compatible endpoint.
    This respects LMSTUDIO_BASE_URL and LMSTUDIO_MAIN_KEY.
    """
    try:
#        cli = client()
#        model = cli.llm.model(
        res = _call_main_chat(prompt)
        text = (res or "").strip()
        return text if text else None
    except Exception as e:
        print(f"Error loading model: {e}")

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
            {"role": "user", "content": prompt + "/no_think"},
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

    # 1) Try native client (preferred)
#    text = _call_lmstudio_chat_main_native(prompt)
#    if text:
#       return text[:LLM_SECTION_TEXT_MAX_LEN].strip()

    # 2) Try OpenAI client
    text = _call_lmstudio_chat_main_openai(prompt)
    if text:
       return text[:LLM_SECTION_TEXT_MAX_LEN].strip()

    # 3) Fallback: HTTP
    text = _call_lmstudio_chat_main(prompt)
    if text:
        return text[:LLM_SECTION_TEXT_MAX_LEN].strip()

    # 4) Deterministic fallback
    return _fallback_extractive_paragraphs(corpus)

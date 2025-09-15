import re, json, sqlite3, datetime as dt
from openai import OpenAI
from ..config import (LMSTUDIO_BASE_URL, LMSTUDIO_MAIN_KEY, LMSTUDIO_MAIN_MODEL,
                      TEMPERATURE, TOP_P, MAX_TOKENS, SEED, DB_PATH)

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
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=SEED
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

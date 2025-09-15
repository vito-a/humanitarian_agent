import re, sqlite3, datetime as dt
from ..config import DB_PATH, MAX_DOC_AGE_DAYS

NUM_PATTERNS = [
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
    r"\b\d+(?:\.\d+)? ?(?:%|percent)\b",
]
DATE_PATTERNS = [r"\b\d{4}-\d{2}-\d{2}\b"]

def extract_tokens(s: str, patterns: list[str]) -> set[str]:
    out = set()
    for p in patterns:
        out |= set(re.findall(p, s))
    return out

def every_number_in_source(item_summary: str, source_text: str) -> bool:
    nums = extract_tokens(item_summary, NUM_PATTERNS)
    if not nums: return True
    for n in nums:
        if n not in source_text:
            return False
    return True

def every_date_in_source(item_summary: str, source_text: str) -> bool:
    dates = extract_tokens(item_summary, DATE_PATTERNS)
    if not dates: return True
    for d in dates:
        if d not in source_text and not item_summary.strip().startswith(d):
            return False
    return True

def max_doc_age_ok(published_at: str|None) -> bool:
    if not published_at: return True
    try:
        d = dt.datetime.fromisoformat(published_at.replace("Z",""))
    except Exception:
        return True
    return (dt.datetime.utcnow() - d).days <= MAX_DOC_AGE_DAYS

def validate_report(report: dict) -> list[str]:
    """Return list of human-readable errors (empty if pass)."""
    errors = []
    refs = report.get("references_by_num", {})
    for sec, items in report.get("sections", {}).items():
        for it in items:
            ref = refs.get(it["ref_num"], {})
            url = ref.get("url","")
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT text FROM pages WHERE url=?", (url,))
                row = c.fetchone()
            src_text = row[0] if row and row[0] else ""
            if not every_number_in_source(it["summary"], src_text):
                errors.append(f"Number mismatch in section '{sec}' for {url}")
            if not every_date_in_source(it["summary"], src_text):
                errors.append(f"Date mismatch in section '{sec}' for {url}")
            if not max_doc_age_ok(it.get("published_at")):
                errors.append(f"Stale doc (> {MAX_DOC_AGE_DAYS}d): {url}")
    return errors

# src/ingest/text_clean.py
import re
from typing import List

# Headers that typically introduce junk lists
BLACKLIST_HEADERS = [
    r"\b(top stories|top news|trending|most (read|viewed)|watch live|live updates|live blog|latest videos)\b",
    r"\b(related|you might also like|more on this|read more|from our partners)\b",
    r"\b(also from|newsletter|subscribe|podcast|photo essay|specials?|abc news live presents)\b",
]

# Single-line junk traits
JUNK_LINE_PATTERNS = [
    r"^\s*(watch|video|live|latest|breaking):",                  # “Video:”, “Watch:”, “Live:”
    r"^\s*\d{1,2}:\d{2}\s*$",                                     # “0:41”
    r"^\s*(sep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug)\.?\s+\d{1,2},?\s+\d{4}\s*$",
    r"^\s*(live|live stream|live now)\s*$",
    r"\b(abc news live|nightline|specials?|presents)\b",
    r"\btiktok|facebook|instagram|newsletter|subscribe\b",
    r"\b(click here|read full story|continue reading)\b",
    r"\b(prince harry|martha raddatz|robin roberts|nightline)\b",  # ABC program/person brands
]

# Lines that look like navigation titles or headline stacks (short, title-cased, no punctuation)
TITLEISH = re.compile(r"^[A-Z][A-Za-z0-9'’\-:, ]{3,80}$")
HAS_SENTENCE_PUNCT = re.compile(r"[.!?][\"')\]]?\s*$")
HAS_WORDS = re.compile(r"[A-Za-z]")

BLACKLIST_HEADERS_RE = re.compile("|".join(BLACKLIST_HEADERS), re.I)
JUNK_LINE_RE = re.compile("|".join(JUNK_LINE_PATTERNS), re.I)

def _is_all_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)

def _likely_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if len(s) < 4 and not s.isdigit():  # ultra short
        return True
    if JUNK_LINE_RE.search(s):
        return True
    if s.count(" | ") >= 2:  # nav bars
        return True
    if _is_all_caps(s) and len(s) < 40:
        return True
    if TITLEISH.match(s) and not HAS_SENTENCE_PUNCT.search(s):
        # Looks like a bare headline w/o sentence punctuation
        return True
    return False

def _looks_like_content_paragraph(p: str) -> bool:
    s = " ".join(p.strip().split())
    if not HAS_WORDS.search(s):
        return False
    # Must have a verb-ish pattern or sentence punctuation or be long enough
    if len(s) >= 60:
        return True
    if HAS_SENTENCE_PUNCT.search(s):
        return True
    # Disallow “stack” of headlines: many short title-like lines
    if len(s.split()) < 8:
        return False
    return True

def strip_nav_boiler(text: str) -> str:
    """
    Remove navigation/related/videos/live sections and headline stacks from plain text.
    Input: text with \n newlines (already stripped of HTML tags).
    """
    # Normalize whitespace and split into lines
    raw_lines = [ln.rstrip() for ln in text.replace("\r", "\n").split("\n")]

    # Phase 1: drop obvious junk lines
    lines = [ln for ln in raw_lines if not _likely_junk_line(ln)]

    # Phase 2: remove blocks that follow blacklisted headers (up to the next 'contenty' paragraph)
    cleaned: List[str] = []
    skipping_block = False
    skip_budget = 0  # avoid nuking entire doc if site is weird
    for ln in lines:
        if not skipping_block and BLACKLIST_HEADERS_RE.search(ln.strip().lower()):
            skipping_block = True
            skip_budget = 40  # skip up to 40 subsequent short items
            continue
        if skipping_block:
            s = ln.strip()
            if _looks_like_content_paragraph(s) and len(s) > 80:
                # content resumed
                skipping_block = False
                cleaned.append(ln)
            else:
                skip_budget -= 1
                if skip_budget <= 0:
                    skipping_block = False
                continue
        else:
            cleaned.append(ln)

    # Phase 3: collapse consecutive short “headliney” lines
    collapsed: List[str] = []
    stack: List[str] = []
    def flush_stack():
        if not stack:
            return
        # If it’s a stack of likely headlines (most short/titleish), drop it; else keep as paragraphs
        titleish = sum(1 for s in stack if TITLEISH.match(s) and len(s) < 90)
        if titleish >= max(2, int(0.7 * len(stack))):
            pass  # drop stack
        else:
            collapsed.extend(stack)
        stack.clear()

    for ln in cleaned:
        s = ln.strip()
        if not s:
            flush_stack()
            collapsed.append("")
            continue
        if (len(s) < 90 and TITLEISH.match(s) and not HAS_SENTENCE_PUNCT.search(s)) or _likely_junk_line(s):
            stack.append(ln)
        else:
            flush_stack()
            collapsed.append(ln)
    flush_stack()

    # Phase 4: join, then re-split by paragraphs and keep only contenty ones
    paragraphs: List[str] = []
    buf: List[str] = []
    def push_buf():
        if not buf:
            return
        para = " ".join(" ".join(buf).split())
        if _looks_like_content_paragraph(para):
            paragraphs.append(para)
        buf.clear()

    for ln in collapsed:
        if not ln.strip():
            push_buf()
        else:
            buf.append(ln)
    push_buf()

    # Post-filter: drop paragraphs that still start with common video/promo markers
    paragraphs = [p for p in paragraphs if not JUNK_LINE_RE.search(p[:80])]

    # Keep first long paragraphs primarily; if nothing remains, fall back to original text minimally cleaned
    if paragraphs:
        return "\n\n".join(paragraphs)
    return " ".join(" ".join(raw_lines).split())

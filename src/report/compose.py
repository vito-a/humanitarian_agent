import re
import json
import sqlite3
import datetime as dt
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse

from ..config import (
    DB_PATH,
    REPORTS_DIR,
    MAX_DOC_AGE_DAYS,
    MAX_ARTICLES,
    BODY_SUMMARY_MAX_LENGTH,
    CONCLUSION_CORPUS_MAX_REFS,
    CATEGORY_BOOST_FACTOR,                  # Keep category boost only for section patterns found in categories (no core boost here)
)
from ..extract.summarize import (
    summary_for_page,                      # if you have a helper like this elsewhere, keep using it
    generate_section_via_llm,              # MAIN model narrative writer (unchanged)
    summarize_page_with_section_model,     # NEW: SECTION model backfill for page.summary
)
from ..filter.topic_filter import load_bags
from ..extract.summarize import generate_section_via_llm
from ..utils.coat_of_arms import ensure_coat_of_arms

_NORM_RE = re.compile(r"norm=([0-9]+(?:\.[0-9]+)?)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def format_us_date(date: dt.date | None = None) -> str:
    """
    Return the given date formatted in US style 'Month Day, Year'.
    If no date is passed, use today's date.
    """
    d = date or dt.date.today()
    return d.strftime("%B %d, %Y")

def _parse_norm(reason: str | None) -> float:
    if not reason:
        return 0.0
    m = _NORM_RE.search(reason)
    try:
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0

def _source_site_from_url(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."): domain = domain[4:]
        parts = domain.split(".")
        return parts[0].capitalize() if parts else domain.capitalize()
    except Exception:
        return "Source"

def _ensure_patterns(val) -> List[str]:
    if isinstance(val, list): return val
    if isinstance(val, dict): return val.get("patterns", []) or []
    return []

def _merged_section(bags: dict, country: str, node_key: str) -> Tuple[str, List[Tuple[str, re.Pattern]], float]:
    cc = bags["countries"][country]
    weights = ((cc.get("summary_gate") or {}).get("weights") or {})
    glob_sec = ((bags.get("sections") or {}).get(node_key) or {})
    country_sec = (cc.get(node_key) or {})

    title = (
        (country_sec.get("title") if isinstance(country_sec, dict) else None)
        or (glob_sec.get("title") if isinstance(glob_sec, dict) else None)
        or node_key.replace("_", " ").title()
    )

    gp = _ensure_patterns(glob_sec)
    cp = _ensure_patterns(country_sec)
    merged = list(gp) + [p for p in cp if p not in gp]

    pats = [(p, re.compile(p, re.I)) for p in merged]
    w = float(weights.get(node_key, 1.0))
    return title, pats, w

def _build_section_defs(bags: dict, country: str) -> List[Tuple[str, List[Tuple[str, re.Pattern]], float]]:
    keys = ["human_harm", "health", "energy", "water", "food"]
    defs = []
    for k in keys:
        title, pats, w = _merged_section(bags, country, k)
        defs.append((title, pats, w))
    return defs

def _resolve_global_order(bags: dict, titles: List[str]) -> List[str]:
    want = [str(t).strip() for t in (bags.get("section_order") or []) if str(t).strip()]
    if not want: return titles
    seen, ordered = set(), []
    for t in want:
        if t in titles and t not in seen:
            ordered.append(t); seen.add(t)
    for t in titles:
        if t not in seen:
            ordered.append(t); seen.add(t)
    return ordered

def _score_group_and_matches(pats: List[Tuple[str, re.Pattern]], text: str, weight: float = 1.0, cat_text: Optional[str] = None, cat_factor: float = CATEGORY_BOOST_FACTOR) -> Tuple[int, List[str], int, int]:
    hits_text = 0
    hits_cat = 0
    matched: List[str] = []
    for pat_str, pat in pats:
        ct = sum(1 for _ in pat.finditer(text))
        if ct > 0:
            matched.append(pat_str)
            hits_text += ct
        if cat_text:
            cc = sum(1 for _ in pat.finditer(cat_text))
            if cc > 0 and pat_str not in matched:
                matched.append(pat_str)
            hits_cat += cc
    score = int(hits_text * weight + hits_cat * weight * cat_factor)
    return score, matched, hits_text, hits_cat

def _extract_key_points(text: str, max_sentences: int = 2) -> str:
    t = (text or "").strip()
    if not t: return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    if not sents: return t
    def score_sent(s: str) -> int:
        score = 0
        score += 4 * len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", s))
        score += 3 * len(re.findall(r"\b(killed|dead|deaths?|casualt(y|ies)|wound(?:ed)?|injur(?:ed|ies)?)\b", s, re.I))
        score += 2 * len(re.findall(r"\b(drone|missile|strike|attack|blackout|power|grid|substation|water|pipeline|sanitation|hospital|clinic|ambulance|displace(?:d|ment)?)\b", s, re.I))
        score += 1 * len(re.findall(r"\[\d+\]", s))
        return score
    ranked = sorted(((i, s, score_sent(s)) for i, s in enumerate(sents)), key=lambda x: x[2], reverse=True)
    picked_idx = sorted([i for i, _, _ in ranked[:max_sentences]])
    out = " ".join(sents[i] for i in picked_idx).strip()
    return out or (sents[0] if sents else t)

def build_country_report(country: str, max_items: int = MAX_ARTICLES, per_section_cap: int = 4) -> dict:
    print(f"Building report for country: {country} (max_items={max_items}, per_section_cap={per_section_cap})")
    try: max_items = int(max_items)
    except Exception: max_items = MAX_ARTICLES
    try: per_section_cap = max(1, int(per_section_cap))
    except Exception: per_section_cap = 4

    # Pull candidate pages (may include rows with NULL/empty summaries)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT p.id, p.url, p.title, p.published_at, p.text, p.summary,
                   L.categories_json, L.classifier_reason
            FROM pages p
            JOIN page_labels L
              ON L.url = p.url
            WHERE L.country_key = ?
              AND L.is_primary = 1
              AND (p.published_at IS NULL OR julianday('now') - julianday(p.published_at) <= ?)
            ORDER BY p.published_at DESC NULLS LAST, p.id DESC
            LIMIT ?;
        """,
            (country, MAX_DOC_AGE_DAYS, max_items),
        )
        rows = c.fetchall()

        # ---------- Backfill missing summaries using SECTION model ----------
        updated = 0
        for (pid, url, title, published_at, text, summary, cats_json, classifier_reason) in rows:
            needs = (summary is None) or (isinstance(summary, str) and summary.strip() == "")
            if needs:
                # Prefer a compact "desc" from page.summary (none) -> build quick lead from text
                lead = " ".join((text or "").split()[:120])
                # Generate short factual summary using the smaller SECTION model
                new_sum = summarize_page_with_section_model(title or "", lead or "", text or "", country)
                new_sum = (new_sum or "").strip()
                if new_sum:
                    c.execute("UPDATE pages SET summary = ? WHERE id = ?", (new_sum, pid))
                    updated += 1
        if updated:
            conn.commit()
            print(f"compose: backfilled {updated} missing summaries using SECTION model")

    print(f"compose: found {len(rows)} rows for country: {country}")

    bags = load_bags()
    section_defs = _build_section_defs(bags, country)
    titles = [t for (t, _, _) in section_defs]
    evidence_titles = _resolve_global_order(bags, titles)
    country_title = country.title()

    coa_path = ensure_coat_of_arms(country)
    if coa_path:
        print(f"compose: coat of arms ready at {coa_path}")
    else:
        print("compose: coat of arms not available (skipping image)")

    # Re-read with updated summaries to proceed consistently
    with sqlite3.connect(DB_PATH) as conn2:
        c2 = conn2.cursor()
        c2.execute(
            """
            SELECT p.id, p.url, p.title, p.published_at, p.text, p.summary,
                   L.categories_json, L.classifier_reason
            FROM pages p
            JOIN page_labels L
              ON L.url = p.url
            WHERE L.country_key = ?
              AND L.is_primary = 1
              AND (p.published_at IS NULL OR julianday('now') - julianday(p.published_at) <= ?)
            ORDER BY p.published_at DESC NULLS LAST, p.id DESC
            LIMIT ?;
        """,
            (country, MAX_DOC_AGE_DAYS, max_items),
        )
        rows = c2.fetchall()

    # Prepare items
    items: List[Dict[str, Any]] = []
    for (pid, url, title, published_at, text, summary, cats_json, classifier_reason) in rows:
        cats_list: List[str] = []
        cat_text = ""
        try:
            data = json.loads(cats_json or "[]")
            if isinstance(data, list):
                cats_list = [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
        if cats_list:
            cat_text = " ".join(cats_list).lower()

        items.append(
            {
                "id": pid,
                "url": url,
                "title": title or "",
                "published_at": published_at,
                "text": text or "",
                "summary": (summary or ""),  # now guaranteed to be non-empty for rows we backfilled
                "categories": cats_list,
                "categories_text": cat_text,
                "classifier_reason": classifier_reason or "",
                "norm_score": _parse_norm(classifier_reason or ""),
            }
        )

    top_summary_title = f"Humanitarian conditions summary for {country_title}"
    other_section_title = "Ongoing hostilities & Strikes"
    section_order: List[str] = [
        top_summary_title,
        other_section_title,
        *evidence_titles,
        "Why These Conditions Justify Support",
        "Conclusion",
    ]
    sections: Dict[str, List[Dict[str, Any]]] = {name: [] for name in section_order}

    title_to_rule: Dict[str, Tuple[List[Tuple[str, re.Pattern]], float]] = {
        title: (pats, w) for (title, pats, w) in section_defs
    }

    # --- Precompute each item's section preferences (NO core-based boost) ---
    for it in items:
        title = it.get("title") or ""
        body = it.get("summary") or it.get("text", "")[:BODY_SUMMARY_MAX_LENGTH]
        hay = f"{title}\n{body}".lower()
        cat_text = it.get("categories_text") or ""

        choices: List[Tuple[str, int, List[str], int, int]] = []
        for sec_title in evidence_titles:
            pats, w = title_to_rule[sec_title]
            s, matches, h_text, h_cat = _score_group_and_matches(
                pats, hay, weight=w, cat_text=cat_text, cat_factor=CATEGORY_BOOST_FACTOR
            )
            if s > 0:
                choices.append((sec_title, s, matches, h_text, h_cat))

        choices.sort(key=lambda x: (x[1], x[4], x[3], x[0]), reverse=True)
        it["section_choices"] = choices

    # --- Capacity-aware assignment ---
    def _item_priority_key(it: Dict[str, Any]):
        best_s = it["section_choices"][0][1] if it.get("section_choices") else 0
        return (best_s, float(it.get("norm_score") or 0.0), (it.get("published_at") or ""))

    capacities = {sec: per_section_cap for sec in evidence_titles}
    assigned_section_for_id: Dict[int, str] = {}
    for it in sorted(items, key=_item_priority_key, reverse=True):
        chosen: Optional[str] = None
        for sec_title, s, matches, h_text, h_cat in (it.get("section_choices") or []):
            if capacities.get(sec_title, 0) > 0:
                chosen = sec_title
                capacities[sec_title] -= 1
                print(
                    f"[ROUTE] id={it.get('id')} -> section='{sec_title}' score={s} "
                    f"(cat_hits={h_cat}, txt_hits={h_text}) (cap left: {capacities[sec_title]})"
                )
                if matches:
                    shown = ", ".join(matches[:15])
                    if len(matches) > 15: shown += ", ..."
                    print(f"        matches: {shown}")
                break
        assigned_section_for_id[it["id"]] = chosen or other_section_title

    refs_by_num, url_to_num, next_num = {}, {}, 1
    for it in items:
        pid, url, title, published_at = it["id"], it["url"], it["title"], it["published_at"]
        s = (it["summary"] or None)
        needs = (s is None) or (isinstance(s, str) and s.strip() == "")
        if needs:
            print(f"Summary empty, regenerating\n")
            s = summary_for_page(pid)  # keep any additional per-page summarizer logic you already have
        body = (s.get("body") or s.get("summary") or "") if isinstance(s, dict) else str(s or "")
        if not body.strip():
            body = it.get("summary") or ""  # fallback to DB summary we just backfilled
        if url not in url_to_num:
            url_to_num[url] = next_num
            refs_by_num[next_num] = {"url": url, "title": title, "published_at": published_at}
            next_num += 1
        ref_num = url_to_num[url]
        it["summary"] = (body.strip() + (f" [{ref_num}]" if ref_num else "")).strip()
        it["ref_num"] = ref_num
        it["source_site"] = _source_site_from_url(url)
        it["key_points"] = _extract_key_points(it["summary"])

    for it in items:
        target = assigned_section_for_id.get(it["id"], other_section_title)
        sections.setdefault(target, []).append(it)

    for name, lst in sections.items():
        lst.sort(key=lambda x: (float(x.get("norm_score") or 0.0), (x.get("published_at") or "")), reverse=True)

    top_summary_title = f"Humanitarian conditions summary for {country_title}"

    # Try to reuse existing narratives from today's JSON
    as_of_today = dt.datetime.utcnow().date().isoformat()
    json_path = REPORTS_DIR / f"{country}_{as_of_today}.json"

    cached_top = cached_why = cached_conc = ""

    if json_path.exists():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
            prev_secs = prev.get("sections", {})

            def _pull(name: str) -> str:
                arr = prev_secs.get(name) or []
                if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                    return str(arr[0].get("summary") or "").strip()
                return ""

            cached_top = _pull(top_summary_title)
            cached_why = _pull("Why These Conditions Justify Support")
            cached_conc = _pull("Conclusion")

            if any([cached_top, cached_why, cached_conc]):
                print("compose: reusing non-empty narrative sections from existing JSON")
        except Exception:
            pass

    def collect_corpus(max_refs: int = 12) -> List[Dict[str, Any]]:
        seen = set()
        corpus: List[Dict[str, Any]] = []
        idx = 0
        while len(corpus) < max_refs:
            added_any = False
            for sect in evidence_titles:
                lst = sections.get(sect, [])
                if idx < len(lst):
                    it = lst[idx]
                    r = it.get("ref_num")
                    if r and r not in seen:
                        corpus.append({
                            "ref": r,
                            "title": it.get("title") or "",
                            "summary": it.get("summary") or "",
                            "date": (it.get("published_at") or "")[:10],
                        })
                        seen.add(r)
                        if len(corpus) >= max_refs:
                            break
                    added_any = True
            if not added_any:
                break
            idx += 1

        if len(corpus) < max_refs:
            for sect in evidence_titles:
                for it in sections.get(sect, []):
                    r = it.get("ref_num")
                    if r and r not in seen:
                        corpus.append({
                            "ref": r,
                            "title": it.get("title") or "",
                            "summary": it.get("summary") or "",
                            "date": (it.get("published_at") or "")[:10],
                        })
                        seen.add(r)
                        if len(corpus) >= max_refs:
                            break
                if len(corpus) >= max_refs:
                    break
        return corpus

    # Only build a corpus if at least one narrative is missing
    corpus = []
    if not (cached_top and cached_why and cached_conc):
        corpus = collect_corpus(max_refs=CONCLUSION_CORPUS_MAX_REFS)

    # Generate only missing narratives; reuse cached if present
    print(f"Generating top summary body\n")
    top_summary_body = cached_top or generate_section_via_llm(country, top_summary_title, corpus)
    why_support_body = cached_why or generate_section_via_llm(country, "Why These Conditions Justify Support", corpus)
    print(f"Generating Why Support\n")
    conclusion_body  = cached_conc or generate_section_via_llm(country, "Conclusion", corpus)
    print(f"Generating Conclusion\n")

    # Store narratives back into sections (these will be written to JSON)
    sections[top_summary_title] = [{"title": top_summary_title, "summary": top_summary_body.replace('‑','-').replace('■', '-'), "ref_num": None}]
    sections["Why These Conditions Justify Support"] = [{"title": "Why Support", "summary": why_support_body.replace('‑','-').replace('■', '-'), "ref_num": None}]
    sections["Conclusion"] = [{"title": "Conclusion", "summary": conclusion_body.replace('‑','-').replace('■', '-'), "ref_num": None}]

    if other_section_title in sections and other_section_title not in section_order:
        section_order.append(other_section_title)

    report = {
        "country": country,
        "as_of": dt.datetime.utcnow().date().isoformat(), # ensure we use today's string used for caching
        "sections": sections,
        "references_by_num": refs_by_num,
        "overview_texts": [],
        "section_order": section_order,
        "assets": {
            "coat_of_arms": str(coa_path) if coa_path else ""
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    outp = REPORTS_DIR / f"{country}_{report['as_of']}.json"
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report

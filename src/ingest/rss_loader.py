import re as _re
import datetime as dt
import feedparser, requests, re, sqlite3, json
from typing import Optional, Iterable, List, Dict, Any
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from datetime import timezone
from openai import OpenAI

from ..config import (
    DB_PATH, MAX_DOC_AGE_DAYS, ALLOW_DOMAINS,
    FEED_CACHE_TTL_SECONDS, DESCRIPTION_MAX_LENGTH, HTTP_READ_TIMEOUT,
    LMSTUDIO_BASE_URL, LMSTUDIO_MAIN_KEY, LMSTUDIO_MAIN_MODEL,
    LMSTUDIO_LABELER_MODEL, LMSTUDIO_SECTION_MODEL,
    TOP_P, MAIN_MODEL_MAX_TOKENS, MAIN_MODEL_SEED, CLEAN_WITH_LLM,
    LLM_CLEANED_TITLE_MAX_LEN, LLM_CLEANED_DESC_MAX_LEN, LLM_CLEANED_TEXT_MAX_LEN
)
from ..sources.country_feeds import COUNTRY_FEEDS, GLOBAL_WESTERN_FEEDS, KNOWN_DOMAINS, guess_country
from .rss_loader_cache import get_or_fetch_daily, get_feed_with_ttl
from ..filter.topic_filter import CountryGate, load_bags, _get_domain_from_url
from ..filter.llm_labeler import (
    classify_item,
    assess_primary_topic_light,   # NEW: lightweight secondary gate (labeler model)
)
from ..filter.llm_sections import (
    assess_primary_topic,          # Heavy third gate (sections model)
    section_titles_for_country,    # For allowed sections list
)
from ..db import upsert_page_label
from ..ingest.text_clean import strip_nav_boiler
from ..validate.calibration import compute_calibration

from ..ingest.categories import (
    extract_rss_categories,
    extract_html_categories,
    merge_categories_to_json,
)

UA = "Mozilla/5.0 (MVP Humanitarian Agent)"
DEFAULT_ATTEMPT_FACTOR = 6
# Separate knobs for confidence thresholds:
DEFAULT_MIN_LLM_PRIMARY_CONF_LIGHT = 50  # lighter model can be a bit lower
DEFAULT_MIN_LLM_PRIMARY_CONF_HEAVY = 55  # heavy/sections gate


# --------- Crisis-only pattern compiler (excludes core/country) ---------
def _compile_crisis_patterns(bags: dict, country: str) -> List[re.Pattern]:
    def _ensure_patterns(val):
        if isinstance(val, list): return val
        if isinstance(val, dict): return val.get("patterns", []) or []
        return []
    glob = (bags.get("sections") or {})
    cc   = (bags.get("countries", {}).get(country, {}) or {})

    def merged(node_key: str) -> List[str]:
        gp = _ensure_patterns(glob.get(node_key) or {})
        cp = _ensure_patterns(cc.get(node_key) or {})
        out = list(gp)
        out += [p for p in cp if p not in out]
        return out

    crisis_keys = ["human_harm", "health", "energy", "water", "food"]
    pats: List[re.Pattern] = []
    for key in crisis_keys:
        for s in merged(key):
            try:
                pats.append(re.compile(s, re.I))
            except re.error:
                pass
    return pats


def _crisis_signal_hits(pats: List[re.Pattern], hay: str, cat_text: str = "") -> int:
    hits = 0
    for p in pats:
        hits += len(list(p.finditer(hay)))
        if cat_text:
            hits += len(list(p.finditer(cat_text)))
    return hits


def _source_site_from_url(url: str) -> str:
    try:
        domain = _get_domain_from_url(url)
        known = KNOWN_DOMAINS
        for dom, label in known.items():
            if domain.endswith(dom):
                return label
        parts = domain.split('.')
        return (parts[0].capitalize() if parts else domain.capitalize())
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return None


def llm_clean_text(title: str, desc: str, text: str, country: str) -> str:
    try:
        prompt = (
            "You are a cleaner. Keep ONLY article paragraphs relevant to the target country's humanitarian/conflict situation.\n"
            f"Target country: {country}\n"
            "Remove: navigation, related links, program schedules, video/live prompts, unrelated world headlines.\n"
            "Return the cleaned paragraphs as plain text, preserving sentences; no explanations, no JSON.\n\n"
            f"TITLE: {title[:LLM_CLEANED_TITLE_MAX_LEN]}\n"
            f"DESC: {desc[:LLM_CLEANED_DESC_MAX_LEN]}\n"
            "TEXT:\n" + text[:LLM_CLEANED_TEXT_MAX_LEN]
        )
        cli = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_MAIN_KEY)
        res = cli.chat.completions.create(
            model=LMSTUDIO_MAIN_MODEL,
            messages=[
                {"role": "system", "content": "Return only the cleaned article text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=TOP_P,
            max_tokens=MAIN_MODEL_MAX_TOKENS,
            seed=MAIN_MODEL_SEED
        )
        out = (res.choices[0].message.content or "").strip()
        if out.startswith("{") or out.startswith("```"):
            return text
        if len(out.split()) < 40:
            return text
        return out
    except Exception:
        return text


def clean_html_to_text(html: str) -> tuple[str | None, str]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
        t.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    text = re.sub(r"\s+", " ", soup.get_text(" ").strip())
    return title, text[:250_000]


def _clean_fragment_to_text(s: str | None) -> str:
    if not s:
        return ""
    soup = BeautifulSoup(s, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    txt = soup.get_text(" ").strip()
    return " ".join(txt.split())


def fetch(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_READ_TIMEOUT)
    r.raise_for_status()
    return r.text


def within_age(published: str) -> bool:
    try:
        d = dt.datetime.fromisoformat(published.replace("Z", ""))
    except Exception:
        return True
    return (dt.datetime.utcnow() - d).days <= MAX_DOC_AGE_DAYS


def to_iso8601_utc(published_raw: str | None, entry: dict | None = None) -> str | None:
    if published_raw:
        try:
            dt_obj = parsedate_to_datetime(published_raw)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            dt_obj = dt_obj.astimezone(timezone.utc).replace(microsecond=0)
            return dt_obj.isoformat().replace("+00:00", "Z")
        except Exception:
            pass
    if entry:
        st = entry.get("published_parsed") or entry.get("updated_parsed")
        if st:
            try:
                dt_obj = dt.datetime(*st[:6], tzinfo=timezone.utc).replace(microsecond=0)
                return dt_obj.isoformat().replace("+00:00", "Z")
            except Exception:
                pass
    if published_raw:
        try:
            s = published_raw.strip().replace("Z", "+00:00")
            dt_obj = dt.datetime.fromisoformat(s)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            dt_obj = dt_obj.astimezone(timezone.utc).replace(microsecond=0)
            return dt_obj.isoformat().replace("+00:00", "Z")
        except Exception:
            pass
    return None


def upsert_page(url, title, published_at_raw, published_at_iso, text, summary_desc, country, source_type, categories_json: str | None):
    domain = urlparse(url).netloc
    if not any(domain.endswith(ad) or ad in domain for ad in ALLOW_DOMAINS):
        return None

    source_site = _source_site_from_url(url)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
                """
            INSERT INTO pages(url, domain, path, title, author, published_at_orig, published_at,
                              category, country, source_type, text, summary, content_hash, fetched_at, source_site,
                              categories_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
                title=COALESCE(excluded.title, pages.title),
                author=COALESCE(excluded.author, pages.author),
                published_at_orig=COALESCE(excluded.published_at_orig, pages.published_at_orig),
                published_at=COALESCE(excluded.published_at, pages.published_at),
                country=COALESCE(excluded.country, pages.country),
                source_type=COALESCE(excluded.source_type, pages.source_type),
                text=COALESCE(excluded.text, pages.text),
                summary=COALESCE(excluded.summary, pages.summary),
                content_hash=excluded.content_hash,
                fetched_at=excluded.fetched_at,
                source_site=COALESCE(excluded.source_site, pages.source_site),
                categories_json=COALESCE(excluded.categories_json, pages.categories_json)
            """,
            (
                url,
                domain,
                urlparse(url).path,
                title,
                source_site,
                published_at_raw,
                published_at_iso,
                None,
                country,
                source_type,
                text,
                summary_desc,
                None,
                dt.datetime.utcnow().isoformat() + "Z",
                source_site,
                categories_json,
            ),
        )
        conn.commit()


def load_country(country: str, primary_city: Optional[str] = None, max_items: int = 20, attempt_factor: int = DEFAULT_ATTEMPT_FACTOR):
    max_items = int(max_items)
    attempt_factor = max(1, int(attempt_factor))
    max_attempts = max_items * attempt_factor

    print(f"\n🌍 Loading articles for: {country.title()} (target accepts={max_items}, max attempts={max_attempts})")

    gate = CountryGate(country, primary_city)
    feeds = [f for f in COUNTRY_FEEDS.get(country, []) if f] + GLOBAL_WESTERN_FEEDS
    seen = set()

    accepted = 0
    attempted = 0

    bags = load_bags()
    sg = (bags["countries"].get(country, {}).get("summary_gate") or {})
    # confidence thresholds (configurable)
    min_llm_conf_light = int(sg.get("min_llm_primary_conf_light", DEFAULT_MIN_LLM_PRIMARY_CONF_LIGHT))
    min_llm_conf_heavy = int(sg.get("min_llm_primary_conf",       DEFAULT_MIN_LLM_PRIMARY_CONF_HEAVY))

    allowed_sections = section_titles_for_country(country)
    crisis_pats = _compile_crisis_patterns(bags, country)

    for feed in feeds:
        if accepted >= max_items or attempted >= max_attempts:
            break

        try:
            print(f"  🔗 Fetching feed: {feed}")
            xml, _ = get_feed_with_ttl(feed, fetch_fn=fetch, ttl_seconds=FEED_CACHE_TTL_SECONDS)
            if not xml:
                print("    ⚠️ Skipping feed (no XML):", feed)
                continue
            parsed = feedparser.parse(xml)
        except Exception as e:
            print(f"    ⚠️ Failed to parse feed: {feed} ({e})")
            continue

        print(f"      ⚠️ Got {len(parsed.entries)} from feed: {feed}\n")
        for e in parsed.entries:
            if accepted >= max_items or attempted >= max_attempts:
                break

            url = e.get("link") or e.get("id")
            print(f"        Got article {url}\n")
            if not url or url in seen:
                print(f"        URL {url} is empty or was seen already\n")
                continue
            seen.add(url)

            ctry = country or guess_country(e.get("title", "") + " " + e.get("summary", ""))
            if ctry != country:
                continue

            domain = urlparse(url).netloc
            if not any(domain.endswith(ad) or ad in domain for ad in ALLOW_DOMAINS):
                continue

            attempted += 1

            try:
                html, _ = get_or_fetch_daily(url, fetch_fn=fetch)
                if not html:
                    continue
                title, text = clean_html_to_text(html)

                # === cleaning ===
                text = strip_nav_boiler(text)
                paras = [p for p in text.split("\n\n") if len(p) > 40]
                text = "\n\n".join(paras[:12]) if paras else text
                if CLEAN_WITH_LLM:
                    desc_raw = e.get("summary") or e.get("description") or ""
                    desc_text = _clean_fragment_to_text(desc_raw)[:DESCRIPTION_MAX_LENGTH]
                    text = llm_clean_text(title or e.get("title", ""), desc_text, text, country)

                pub_raw = e.get("published") or e.get("updated") or ""
                pub_iso = to_iso8601_utc(pub_raw, e)

                title_str = (e.get("title") or "").strip()
                desc_raw = e.get("summary") or e.get("description") or ""
                desc_text = _clean_fragment_to_text(desc_raw)[:DESCRIPTION_MAX_LENGTH]

                # ------------- CATEGORIES -------------
                rss_cats  = extract_rss_categories(e)
                html_cats = extract_html_categories(html)
                # --------------------------------------

                # ---------- 1) FAST KEYWORD GATE ----------
                keep, score, explain = gate.summary_score(title_str, desc_text, url, categories=(rss_cats + html_cats))
                if not keep:
                    print(f"  1. ❌ Pre-LLM outer gate reject (low score) : keep={keep}, score={score}, reason={explain}\n")
                    continue

                # must-token in full text
                if not gate.strict_token_in_fulltext(f"{title_str}\n{desc_text}\n{text or ''}"):
                    categories_json = merge_categories_to_json(rss_cats, html_cats)
                    upsert_page_label(
                        url=url,
                        country_key=country,
                        page_id=None,
                        relevance_score=score,
                        main_country=None,
                        is_primary=False,
                        categories_json=categories_json,
                        classifier_reason="missing_must_token_post",
                        classifier_confidence=0,
                    )
                    continue

                print(f"  1. ✅ Pre-LLM outer gate pass (good score) : keep={keep}, score={score}, reason={explain}\n")

                # ---------- 1b) CRISIS-SIGNAL PRECHECK (EXCLUDES core/country) ----------
                hay = f"{title_str}\n{desc_text}\n" + " ".join((text or "").split()[:200])
                cat_text = " ".join((rss_cats + html_cats)).lower()
                crisis_hits = _crisis_signal_hits(crisis_pats, hay.lower(), cat_text)
                if crisis_hits <= 0:
                    reason = f"no_crisis_signals crisis_hits=0 score={score}"
                    upsert_page_label(
                        url=url,
                        country_key=country,
                        page_id=None,
                        relevance_score=score,
                        main_country=None,
                        is_primary=False,
                        categories_json=merge_categories_to_json(rss_cats, html_cats),
                        classifier_reason=reason,
                        classifier_confidence=0,
                    )
                    print(f"  2. ❌ Pre-LLM inner gate keyword + crisis-signal check reject, reason : no crisis signals, crisis_hits={crisis_hits} score={score}\n")
                    continue

                print(f"  2. ✅ Pre-LLM inner gate keyword + crisis-signal check pass : {title_str}, reason : crisis_hits={crisis_hits} score={score}\n")

                # ---------- 2) LLM CLASSIFICATION ----------
                lead = " ".join((text or "").split()[:150])
                lab = classify_item(title_str, desc_text, lead, target_country=country)
                lab_cats = lab.get("categories", []) or []

                # Merge categories → json
                all_cats       = rss_cats + html_cats + lab_cats
                all_cats_json  = merge_categories_to_json(rss_cats, html_cats, lab_cats)

                # Recompute summary/normalized with all categories (ok if core still present; precheck handled it)
                keep_sum, boosted_sum, sum_details = gate.summary_score(title_str, desc_text, url, categories=all_cats)
                norm_total = float(sum_details.get("boosted_score", boosted_sum))

                # Calibration
                zbags = load_bags()
                zsg = (zbags["countries"].get(country, {}).get("summary_gate") or {})
                z_k   = float(zsg.get("z_k", 1.2))
                pctl  = float(zsg.get("pctl", 0.70))
                floor = float(zsg.get("min_floor", 10.0))
                win   = int(zsg.get("window_days", 14))

                cal = compute_calibration(country, days=win)
                mu, sd, p70 = float(cal["mean"]), max(float(cal["std"]), 1e-6), float(cal["p70"])

                z_pass     = (norm_total >= (mu + z_k * sd))
                p_pass     = (norm_total >= p70)
                floor_pass = (norm_total >= floor)
                keep_by_summary = (z_pass or p_pass) and floor_pass

                # ---------- 2b) NEW LIGHTWEIGHT LLM GATE ----------
                prim_light = assess_primary_topic_light(
                    title=title_str,
                    desc=desc_text,
                    lead=lead,
                    target_country=country,
                    allowed_sections=allowed_sections,
                    summary_score=boosted_sum,
                    normalized_score=norm_total,
                    prior_categories=lab_cats,
                )
                light_pass = bool(prim_light.get("pass", False))
                light_conf = int(prim_light.get("confidence", 0))
                if not light_pass or light_conf < min_llm_conf_light:
                    reason = (
                        f"llm_light_reject pass={light_pass} conf={light_conf}<{min_llm_conf_light}; "
                        f"boosted={boosted_sum}; norm={norm_total:.2f}"
                    )
                    upsert_page_label(
                        url=url,
                        country_key=country,
                        page_id=None,
                        relevance_score=score,
                        main_country=lab.get("main_country"),
                        is_primary=False,
                        categories_json=all_cats_json,
                        classifier_reason=reason,
                        classifier_confidence=int(lab.get("confidence") or 0),
                    )
                    print(f"  3. ❌ Light LLM main gate rejected: {title_str}, {LMSTUDIO_LABELER_MODEL} rejects that, reason : {prim_light}\n")
                    continue

                print(f"  3. ✅ Light LLM main gate pass: {title_str}, {LMSTUDIO_LABELER_MODEL} approves that, reason : {prim_light}\n")

                # ---------- 3) HEAVY LLM GATE (sections model) ----------
                prim = assess_primary_topic(
                    title=title_str,
                    desc=desc_text,
                    lead=lead,
                    target_country=country,
                    allowed_sections=allowed_sections,
                    summary_score=boosted_sum,
                    normalized_score=norm_total,
                    prior_categories=lab_cats,
                )
                llm_pass = bool(prim.get("pass", False))
                llm_conf = int(prim.get("confidence", 0))
                llm_secs = prim.get("sections", [])
                llm_reason = prim.get("reason", "")

                if not llm_pass or llm_conf < min_llm_conf_heavy:
                    reason = (
                        f"llm_heavy_reject pass={llm_pass} conf={llm_conf}<{min_llm_conf_heavy}; "
                        f"secs={llm_secs}; why={llm_reason}; boosted={boosted_sum}; norm={norm_total:.2f}; "
                        f"z_ref=({mu:.2f},{sd:.2f}); p70={p70:.2f}"
                    )
                    upsert_page_label(
                        url=url,
                        country_key=country,
                        page_id=None,
                        relevance_score=score,
                        main_country=lab.get("main_country"),
                        is_primary=False,
                        categories_json=all_cats_json,
                        classifier_reason=reason,
                        classifier_confidence=int(lab.get("confidence") or 0),
                    )
                    print(f"  4. ❌ Heavy LLM outer gate rejected: {title_str} (keep={keep}, keep_by_summary={keep_by_summary}, llm_pass={llm_pass})\n"
                          f"simple={score}; summary_raw={boosted_sum}; norm={norm_total:.2f};z_ref=({mu:.2f},{sd:.2f}); p70={p70:.2f}; "
                          f"llm_primary_pass conf={llm_conf}, secs={llm_secs}, {LMSTUDIO_SECTION_MODEL} preliminary rejects that, reason: {reason}")
                    continue

                print(f"  4. ✅ Heavy LLM outer gate pass: {title_str} (keep={keep}, keep_by_summary={keep_by_summary}, llm_pass={llm_pass})\n"
                        f"simple={score}; summary_raw={boosted_sum}; norm={norm_total:.2f};z_ref=({mu:.2f},{sd:.2f}); p70={p70:.2f}; "
                        f"llm_primary_pass conf={llm_conf}, secs={llm_secs}, {LMSTUDIO_SECTION_MODEL} preliminary approves that, reason: {prim}")

                keep_final = (keep or keep_by_summary) and llm_pass
                if not keep_final:
                    reason = (
                        f"keep_final_false boosted={boosted_sum}; norm={norm_total:.2f}; "
                        f"z_ref=({mu:.2f},{sd:.2f}); p70={p70:.2f}; llm_pass={llm_pass}; llm_conf={llm_conf}"
                    )
                    upsert_page_label(
                        url=url,
                        country_key=country,
                        page_id=None,
                        relevance_score=score,
                        main_country=lab.get("main_country"),
                        is_primary=False,
                        categories_json=all_cats_json,
                        classifier_reason=reason,
                        classifier_confidence=int(lab.get("confidence") or 0),
                    )
                    print(f"  5. ❌ Heavy LLM final gate rejected: {title_str} (keep={keep}, keep_by_summary={keep_by_summary}, llm_pass={llm_pass})\n"
                          f"simple={score}; summary_raw={boosted_sum}; norm={norm_total:.2f};z_ref=({mu:.2f},{sd:.2f}); p70={p70:.2f}; "
                          f"llm_primary_pass conf={llm_conf}, secs={llm_secs}, {LMSTUDIO_SECTION_MODEL} finally rejects that, reason: {reason}")
                    continue

                print(f"  5. ✅ Heavy LLM final gate pass: {title_str} (keep={keep}, keep_by_summary={keep_by_summary}, llm_pass={llm_pass})\n"
                      f"simple={score}; summary_raw={boosted_sum}; norm={norm_total:.2f};z_ref=({mu:.2f},{sd:.2f}); p70={p70:.2f}; "
                      f"llm_primary_pass conf={llm_conf}, secs={llm_secs}, {LMSTUDIO_SECTION_MODEL} finally approves that, reason: {prim}")

                # ---------- Persist accepted ----------
                upsert_page(url, title, pub_raw, pub_iso, text, desc_text, country, "rss", all_cats_json)
                upsert_page_label(
                    url=url,
                    country_key=country,
                    page_id=None,
                    relevance_score=score,
                    main_country=lab.get("main_country"),
                    is_primary=True,
                    categories_json=all_cats_json,
                    classifier_reason=(
                        f"simple={score}; summary_raw={boosted_sum}; norm={norm_total:.2f}; "
                        f"z_ref=({mu:.2f},{sd:.2f}); p70={p70:.2f}; "
                        f"llm_light_pass conf={light_conf}; llm_heavy_pass conf={llm_conf} secs={llm_secs}"
                    ),
                    classifier_confidence=int(lab.get("confidence") or 0),
                )
                accepted += 1
                print(f"    ✅ Cached: {title[:80]}... [{domain}]")
            except Exception as e:
                print(f"    ❌ Failed to fetch {url} ({e})")

            if accepted >= max_items or attempted >= max_attempts:
                break

    print(f"  📦 Done: accepted={accepted}, attempted={attempted}, seen={len(seen)}, target={max_items}, cap={max_attempts})")

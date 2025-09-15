# src/ingest/rss_loader_cache.py
import json
import hashlib
import datetime as dt
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict
from urllib.parse import urlparse
import re, shutil

from ..config import CACHE_DIR

# ---------- Page (article) daily cache (BY DOMAIN) ----------
PAGES_DIR = CACHE_DIR / "rss_pages"
PAGES_DIR.mkdir(parents=True, exist_ok=True)
PAGES_INDEX = CACHE_DIR / "rss_pages_index.json"
# index entry shape:
# { url: { "short_id": "...", "date": "YYYY-MM-DD", "domain": "reuters.com", "filename": "reuters.com/<sid>_<date>.html" } }

# ---------- Feed (RSS/Atom) TTL cache (unchanged layout) ----------
FEEDS_DIR = CACHE_DIR / "rss_feeds"
FEEDS_DIR.mkdir(parents=True, exist_ok=True)
FEEDS_INDEX = CACHE_DIR / "rss_feeds_index.json"
# { url: { "short_id": "...", "fetched_at": "...Z", "filename": "<sid>_<YYYY-MM-DD>.xml", "ttl_seconds": 86400 } }

# ---------- Shared helpers ----------
def _today_iso() -> str:
    return dt.datetime.utcnow().date().isoformat()  # 'YYYY-MM-DD'

def _short_id(url: str, n: int = 10) -> str:
    return hashlib.blake2s(url.encode("utf-8"), digest_size=16).hexdigest()[:n]

def _read_text(path: Path) -> Optional[str]:
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    return None

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="ignore")

def _load_index(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def _save_index(path: Path, idx: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2))
    tmp.replace(path)

def _safe_domain(url: str) -> str:
    d = urlparse(url).netloc or "unknown"
    # Be conservative; keep dots and hyphens, replace the rest
    return "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in d).strip(".") or "unknown"

def get_or_fetch_daily(url: str, fetch_fn: Callable[[str], str]) -> Tuple[Optional[str], bool]:
    """
    Daily cache per URL for article HTML, stored under rss_pages/<domain>/<short>_<YYYY-MM-DD>.html.
    Returns (html, is_cached_today).
    Backwards-compatible with old index entries that had a root-level filename.
    """
    idx = _load_index(PAGES_INDEX)
    today = _today_iso()
    entry = idx.get(url)
    sid = entry.get("short_id") if entry else _short_id(url)
    domain = entry.get("domain") if entry and entry.get("domain") else _safe_domain(url)

    # Try to serve today's cached file
    if entry and entry.get("date") == today:
        rel = entry.get("filename") or ""
        # If old index used a root-level filename, resolve against PAGES_DIR
        p = (PAGES_DIR / rel) if "/" in rel else (PAGES_DIR / rel)
        html = _read_text(p)
        if html is not None:
            print(f"      💾 Cache hit (today): {url} ({rel})")
            return html, True

    # Fetch fresh and write to by-domain path
    try:
        html = fetch_fn(url)
        rel_name = f"{domain}/{sid}_{today}.html"
        _write_text(PAGES_DIR / rel_name, html)
        idx[url] = {"short_id": sid, "date": today, "domain": domain, "filename": rel_name}
        _save_index(PAGES_INDEX, idx)
        print(f"      🆕 Fetched & cached: {rel_name}")
        return html, False
    except Exception as e:
        # Fallback: attempt prior cached file (old or new layout)
        if entry:
            rel = entry.get("filename") or ""
            p = (PAGES_DIR / rel) if "/" in rel else (PAGES_DIR / rel)
            html = _read_text(p)
            if html is not None:
                print(f"      ⚠️ Fetch failed, using previous cache: {rel} ({e})")
                return html, True
        print(f"      ❌ Fetch failed, no cache available: {url} ({e})")
        return None, False

def get_feed_with_ttl(url: str, fetch_fn: Callable[[str], str], ttl_seconds: int) -> Tuple[Optional[str], bool]:
    """
    TTL cache for feed XML. Returns (xml, is_cached).
    - If cached and fresh (now - fetched_at <= ttl_seconds), return cached XML.
    - Else fetch, cache, return fresh XML.
    - On fetch failure, fall back to cached XML if available.
    Stored path: rss_feeds/<domain>/<short_id>_<YYYY-MM-DD>.xml
    """
    now = dt.datetime.utcnow()
    idx = _load_index(FEEDS_INDEX)
    entry = idx.get(url)
    sid = entry.get("short_id") if entry else _short_id(url)
    domain = entry.get("domain") if entry and entry.get("domain") else _safe_domain(url)

    # Fresh enough?
    if entry:
        fetched_at = entry.get("fetched_at")
        try:
            fetched_dt = dt.datetime.fromisoformat(str(fetched_at).replace("Z", ""))
        except Exception:
            fetched_dt = None
        if fetched_dt and (now - fetched_dt).total_seconds() <= ttl_seconds:
            rel = entry.get("filename", "")
            # Support both legacy flat ("file.xml") and new ("domain/file.xml")
            p = FEEDS_DIR / rel
            xml = _read_text(p)
            if xml is not None:
                print(f"  💾 Feed cache hit (fresh): {rel}")
                return xml, True

    # Refresh
    try:
        xml = fetch_fn(url)
        stamp = now.date().isoformat()
        rel_name = f"{domain}/{sid}_{stamp}.xml"   # <-- by-domain path
        _write_text(FEEDS_DIR / rel_name, xml)
        idx[url] = {
            "short_id": sid,
            "domain": domain,                       # <-- store domain in index
            "fetched_at": now.isoformat() + "Z",
            "filename": rel_name,
            "ttl_seconds": ttl_seconds,
        }
        _save_index(FEEDS_INDEX, idx)
        print(f"  🆕 Feed fetched & cached: {rel_name}")
        return xml, False
    except Exception as e:
        if entry:
            rel = entry.get("filename", "")
            p = FEEDS_DIR / rel
            xml = _read_text(p)
            if xml is not None:
                print(f"  ⚠️ Feed fetch failed, using cached: {rel} ({e})")
                return xml, True
        print(f"  ❌ Feed fetch failed, no cache: {url} ({e})")
        return None, False

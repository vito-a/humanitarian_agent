from typing import Dict
import re, shutil

from src.ingest.rss_loader_cache import _load_index, _safe_domain, _save_index

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

def migrate_page_cache_to_domains(dry_run: bool = True) -> dict:
    """
    Move page cache files from flat layout into by-domain folders and update the index.
    Returns a summary dict. Set dry_run=False to actually move files.

    - Looks at PAGES_INDEX entries.
    - If an entry has no 'domain' or 'filename' not under '<domain>/', it will be migrated.
    - Keeps original short_id and date if known; else derives from filename; else uses today.
    - On filename collision, appends '-migrated', '-migrated2', ...
    """
    idx = _load_index(PAGES_INDEX)
    if not idx:
        print("Nothing to migrate: page index empty.")
        return {"total": 0, "migrated": 0, "skipped_ok": 0, "missing_files": 0, "collisions": 0}

    # helper: extract YYYY-MM-DD from '<anything>_YYYY-MM-DD.html'
    date_re = re.compile(r"_(\d{4}-\d{2}-\d{2})\.html$")

    total = 0
    migrated = 0
    skipped_ok = 0
    missing_files = 0
    collisions = 0

    for url, meta in list(idx.items()):
        total += 1
        meta = meta or {}
        sid = meta.get("short_id") or _short_id(url)
        domain = meta.get("domain") or _safe_domain(url)
        rel = meta.get("filename") or ""
        # Resolve current path
        src_path = (PAGES_DIR / rel) if rel else None

        # Detect if already correct (has domain prefix and matches our domain)
        already_domainized = isinstance(rel, str) and "/" in rel and rel.split("/", 1)[0] == domain

        # If filename missing, try to infer a flat file name (legacy pattern '<sid>_<date>.html')
        if not src_path or not src_path.exists():
            # Legacy flat: just filename at root
            if rel and "/" not in rel and (PAGES_DIR / rel).exists():
                src_path = PAGES_DIR / rel
            elif rel and "/" in rel and (PAGES_DIR / rel).exists():
                src_path = PAGES_DIR / rel

        if already_domainized and src_path and src_path.exists():
            skipped_ok += 1
            continue

        if not src_path or not src_path.exists():
            # Nothing to move; keep index entry as-is
            missing_files += 1
            continue

        # Determine date
        date_str = meta.get("date")
        if not date_str:
            m = date_re.search(src_path.name)
            date_str = m.group(1) if m else _today_iso()

        # Determine destination path
        base_name = f"{sid}_{date_str}.html"
        dest_rel = f"{domain}/{base_name}"
        dest_path = PAGES_DIR / dest_rel

        # Handle collisions
        if dest_path.exists():
            collisions += 1
            stem = dest_path.stem  # e.g., 'abcd_2025-09-07'
            suffix = dest_path.suffix
            k = 1
            while dest_path.exists():
                dest_rel = f"{domain}/{stem}-migrated{k}{suffix}"
                dest_path = PAGES_DIR / dest_rel
                k += 1

        # Move (or simulate)
        print(("DRY-RUN: " if dry_run else "") + f"Move {src_path.relative_to(PAGES_DIR)} -> {dest_rel}")
        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))
            # Update index entry
            idx[url] = {
                "short_id": sid,
                "date": date_str,
                "domain": domain,
                "filename": dest_rel,
            }

    # Save updated index
    if not dry_run:
        _save_index(PAGES_INDEX, idx)

    summary = {
        "total": total,
        "migrated": migrated,         # note: we count by prints; update counter below
        "skipped_ok": skipped_ok,
        "missing_files": missing_files,
        "collisions": collisions,
    }
    # Fix 'migrated' count by recomputing from prints? Instead, recompute from idx consistency:
    # For simplicity, estimate migrated as totals minus skipped_ok minus missing_files.
    summary["migrated"] = max(0, total - skipped_ok - missing_files)
    print("Migration summary:", summary)
    return summary

def migrate_feeds_cache_to_domains(dry_run: bool = True) -> dict:
    """
    Move feed cache files from flat layout into by-domain folders and update the feed index.
    Returns a summary dict. Set dry_run=False to actually move files.

    Index before (legacy):
      { url: { "short_id": "...", "fetched_at": "...Z", "filename": "<sid>_<YYYY-MM-DD>.xml", "ttl_seconds": 86400 } }

    Index after (domainized):
      { url: { "short_id": "...", "fetched_at": "...Z", "filename": "reuters.com/<sid>_<YYYY-MM-DD>.xml",
               "ttl_seconds": 86400, "domain": "reuters.com" } }
    """
    idx = _load_index(FEEDS_INDEX)
    if not idx:
        print("Nothing to migrate: feed index empty.")
        return {"total": 0, "migrated": 0, "skipped_ok": 0, "missing_files": 0, "collisions": 0}

    date_re = re.compile(r"_(\d{4}-\d{2}-\d{2})\.xml$")

    total = 0
    migrated = 0
    skipped_ok = 0
    missing_files = 0
    collisions = 0

    for url, meta in list(idx.items()):
        total += 1
        meta = meta or {}
        sid = meta.get("short_id") or _short_id(url)
        domain = meta.get("domain") or _safe_domain(url)
        rel = meta.get("filename") or ""

        # Already in domain/<file>?
        already_domainized = isinstance(rel, str) and "/" in rel and rel.split("/", 1)[0] == domain

        # Resolve current (legacy) path possibilities
        src_path = FEEDS_DIR / rel if rel else None
        if (not src_path or not src_path.exists()) and rel and "/" not in rel:
            # Legacy flat file at FEEDS_DIR root
            maybe = FEEDS_DIR / rel
            if maybe.exists():
                src_path = maybe

        if already_domainized and src_path and src_path.exists():
            skipped_ok += 1
            continue

        if not src_path or not src_path.exists():
            missing_files += 1
            continue

        # Determine date: prefer fetched_at, else parse from filename, else today
        date_str = None
        fetched_at = meta.get("fetched_at")
        if fetched_at:
            try:
                # Keep original date component
                date_str = str(fetched_at).split("T", 1)[0]
            except Exception:
                date_str = None
        if not date_str:
            m = date_re.search(src_path.name)
            date_str = m.group(1) if m else _today_iso()

        base_name = f"{sid}_{date_str}.xml"
        dest_rel = f"{domain}/{base_name}"
        dest_path = FEEDS_DIR / dest_rel

        # Collision handling
        if dest_path.exists():
            collisions += 1
            stem, suffix = dest_path.stem, dest_path.extname if hasattr(dest_path, "extname") else dest_path.suffix
            k = 1
            while dest_path.exists():
                dest_rel = f"{domain}/{stem}-migrated{k}{suffix}"
                dest_path = FEEDS_DIR / dest_rel
                k += 1

        print(("DRY-RUN: " if dry_run else "") + f"Move {src_path.relative_to(FEEDS_DIR)} -> {dest_rel}")
        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))
            # Update index entry (preserve existing metadata)
            new_meta = dict(meta)
            new_meta["domain"] = domain
            new_meta["filename"] = dest_rel
            idx[url] = new_meta
            migrated += 1

    if not dry_run:
        _save_index(FEEDS_INDEX, idx)

    summary = {
        "total": total,
        "migrated": migrated,
        "skipped_ok": skipped_ok,
        "missing_files": missing_files,
        "collisions": collisions,
    }
    print("Feed migration summary:", summary)
    return summary

def purge_domain_cache(domain: str) -> int:
    """
    Delete all cached page files for a domain and remove corresponding index entries.
    Returns count of files removed.
    """
    d = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in domain).strip(".") or "unknown"
    folder = PAGES_DIR / d
    idx = _load_index(PAGES_INDEX)

    removed = 0
    if folder.exists():
        # Count files first
        removed = sum(1 for _ in folder.rglob("*") if _.is_file())
        shutil.rmtree(folder, ignore_errors=True)

    # Clean index entries for that domain
    victims = [u for u, meta in idx.items() if (meta or {}).get("domain") == d]
    for u in victims:
        idx.pop(u, None)
    _save_index(PAGES_INDEX, idx)

    print(f"🧹 Purged domain cache: {d} (files removed: {removed}, index entries: {len(victims)})")
    return removed

def list_pages_domain_cache_counts() -> Dict[str, int]:
    """
    Return a dict {domain: file_count} for page cache folders.
    """
    counts: Dict[str, int] = {}
    if not PAGES_DIR.exists():
        return counts
    for sub in PAGES_DIR.iterdir():
        if sub.is_dir():
            counts[sub.name] = sum(1 for p in sub.rglob("*") if p.is_file())
    return counts

def list_feeds_domain_cache_counts() -> Dict[str, int]:
    """
    Return a dict {domain: file_count} for page cache folders.
    """
    counts: Dict[str, int] = {}
    if not FEEDS_DIR.exists():
        return counts
    for sub in FEEDS_DIR.iterdir():
        if sub.is_dir():
            counts[sub.name] = sum(1 for p in sub.rglob("*") if p.is_file())
    return counts

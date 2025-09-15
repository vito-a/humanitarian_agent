import sqlite3
import datetime as dt
from pathlib import Path
from .config import DB_PATH

PAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    domain TEXT,
    source_site TEXT,
    path TEXT,
    title TEXT,
    author TEXT,
    published_at_orig TEXT,   -- raw from feed (NEW)
    published_at TEXT,        -- normalized ISO-8601 UTC (NEW)
    category TEXT,
    categories_json TEXT,
    country TEXT,
    source_type TEXT,         -- 'rss' | 'web'
    text TEXT,
    summary TEXT,
    content_hash TEXT,
    fetched_at TEXT
);
"""

CALIB_SCHEMA = """
CREATE TABLE IF NOT EXISTS summary_calibration (
  country TEXT,
  as_of TEXT,
  mean REAL,
  std REAL,
  p70 REAL,
  p85 REAL,
  n INTEGER,
  PRIMARY KEY(country, as_of)
);
"""

TAGS_SCHEMA = """CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);"""
PAGE_TAGS_SCHEMA = """CREATE TABLE IF NOT EXISTS page_tags (page_id INTEGER, tag_id INTEGER, UNIQUE(page_id, tag_id));"""

EXTRACTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS extractions (
    page_id INTEGER PRIMARY KEY,
    json TEXT,           -- optional in Step 1 (we still export report JSON)
    created_at TEXT
);
"""

# --- Label table (generic, per URL per country) ---

LABELS_SCHEMA = """
CREATE TABLE IF NOT EXISTS page_labels (
    url TEXT NOT NULL,
    page_id INTEGER,                 -- optional: denorm for convenience
    country_key TEXT NOT NULL,       -- e.g. 'ukraine', 'syria', 'yemen'
    relevance_score REAL,            -- numeric gate score
    main_country TEXT,               -- LLM: primary country text
    is_primary INTEGER,              -- LLM: 1 if primary focus is 'country_key'
    categories_json TEXT,            -- LLM: categories array as JSON
    classifier_reason TEXT,
    classifier_confidence INTEGER,
    labeled_at TEXT,                 -- ISO timestamp UTC
    PRIMARY KEY (url, country_key)
);
"""

LABELS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_labels_country ON page_labels(country_key);",
    "CREATE INDEX IF NOT EXISTS idx_labels_primary ON page_labels(is_primary);",
    "CREATE INDEX IF NOT EXISTS idx_labels_url ON page_labels(url);",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pages_domain   ON pages(domain);",
    "CREATE INDEX IF NOT EXISTS idx_pages_published ON pages(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_pages_country  ON pages(country);",
]

def _table_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table});")
    return {row[1] for row in cur.fetchall()}

def _migrate_pages_table(conn):
    cols = _table_columns(conn, "pages")
    if not cols:
        # Fresh create
        conn.executescript(PAGES_SCHEMA)
        for ddl in INDEXES: conn.execute(ddl)
        conn.commit()
        return

    need_rename = ("published_at" in cols) and ("published_at_orig" not in cols)
    need_iso    = ("published_at" not in cols) or need_rename  # we want a clean ISO column named 'published_at'

    if need_rename:
        # Try SQLite native rename first (SQLite >= 3.25)
        try:
            conn.execute("ALTER TABLE pages RENAME COLUMN published_at TO published_at_orig;")
            conn.commit()
        except sqlite3.OperationalError:
            # Rebuild table (portable path)
            conn.execute("ALTER TABLE pages RENAME TO pages_old;")
            conn.executescript(PAGES_SCHEMA)
            # Copy data: old published_at → published_at_orig; new published_at left NULL
            conn.execute("""
                INSERT INTO pages (id,url,domain,path,title,author,published_at_orig,category,country,source_type,text,summary,content_hash,fetched_at)
                SELECT id,url,domain,path,title,author,published_at,category,country,source_type,text,summary,content_hash,fetched_at
                FROM pages_old;
            """)
            conn.execute("DROP TABLE pages_old;")
            conn.commit()
            cols = _table_columns(conn, "pages")

    # Ensure ISO column exists
    cols = _table_columns(conn, "pages")
    if "published_at" not in cols:
        conn.execute("ALTER TABLE pages ADD COLUMN published_at TEXT;")
        conn.commit()

    # Ensure indexes
    for ddl in INDEXES: conn.execute(ddl)
    conn.commit()

def _migrate_page_labels(conn):
    conn.executescript(LABELS_SCHEMA)
    for ddl in LABELS_INDEXES:
        conn.execute(ddl)
    conn.commit()

def upsert_page_label(
    url: str,
    country_key: str,
    page_id: int | None,
    relevance_score: float | None,
    main_country: str | None,
    is_primary: bool | int | None,
    categories_json: str | None,
    classifier_reason: str | None,
    classifier_confidence: int | None,
):
    labeled_at = dt.datetime.utcnow().isoformat() + "Z"
    is_primary_int = 1 if is_primary else 0
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        INSERT INTO page_labels(url, country_key, page_id, relevance_score, main_country, is_primary,
                                categories_json, classifier_reason, classifier_confidence, labeled_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(url, country_key) DO UPDATE SET
            page_id=COALESCE(excluded.page_id, page_labels.page_id),
            relevance_score=COALESCE(excluded.relevance_score, page_labels.relevance_score),
            main_country=COALESCE(excluded.main_country, page_labels.main_country),
            is_primary=excluded.is_primary,
            categories_json=COALESCE(excluded.categories_json, page_labels.categories_json),
            classifier_reason=COALESCE(excluded.classifier_reason, page_labels.classifier_reason),
            classifier_confidence=COALESCE(excluded.classifier_confidence, page_labels.classifier_confidence),
            labeled_at=excluded.labeled_at
        """, (
            url, country_key, page_id, relevance_score, main_country, is_primary_int,
            categories_json, classifier_reason, classifier_confidence, labeled_at
        ))
        conn.commit()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        print(DB_PATH)
#        _migrate_pages_table(conn)   # pages table migration
        _migrate_page_labels(conn)   # labels table
        c = conn.cursor()
        c.execute(PAGES_SCHEMA)
        c.execute(CALIB_SCHEMA)

        # Ensure indexes
        for ddl in INDEXES:
            conn.execute(ddl)
        conn.commit()

        c.execute(TAGS_SCHEMA)
        c.execute(PAGE_TAGS_SCHEMA)

        # structured extractions per page (JSON blob for forward-compat)
        c.execute(EXTRACTIONS_SCHEMA)
        conn.commit()

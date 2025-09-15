# src/ingest/categories.py
from __future__ import annotations
import json, re
from typing import Any, Dict, Iterable, List, Optional

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # graceful fallback

def _normalize_terms(terms: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in terms:
        s = re.sub(r"\s+", " ", (t or "").strip())
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out

def extract_rss_categories(entry: Dict[str, Any]) -> List[str]:
    """
    Handles feedparser entries:
      - entry.tags: [{'term': 'Oil', 'label': 'Oil & Gas', 'scheme': '...'}, ...]
      - entry.category: 'Energy'
      - entry.categories: ['Energy', ...] (some libs)
    """
    terms: List[str] = []
    # tags[]
    tags = entry.get("tags") or []
    for t in tags:
        if isinstance(t, dict):
            for k in ("term", "label", "value", "name"):
                v = t.get(k)
                if isinstance(v, str):
                    terms.append(v)
        elif isinstance(t, str):
            terms.append(t)

    # category / categories
    for k in ("category", "categories"):
        v = entry.get(k)
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    terms.append(x)
                elif isinstance(x, dict):
                    for key in ("term", "label", "value", "name"):
                        if isinstance(x.get(key), str):
                            terms.append(x[key])
        elif isinstance(v, str):
            terms.append(v)

    return _normalize_terms(terms)

def extract_html_categories(html: str, url: str | None = None) -> List[str]:
    """
    Extracts categories/keywords inside the article HTML:
      - <meta property="article:tag" content="...">
      - <meta property="og:article:tag" content="...">
      - <meta name="keywords" content="..., ...">
      - <meta name="news_keywords" content="...">
      - schema.org: itemprop="keywords"
      - <a rel="tag">label</a>, <a class*="tag|category">label</a>
    Falls back to simple regex if BeautifulSoup isn't available.
    """
    terms: List[str] = []
    if not html:
        return terms

    if BeautifulSoup is None:
        # lightweight regex fallbacks
        metas = re.findall(r'<meta[^>]+(?:name|property)=["\']([^"\']+)["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
        for key, content in metas:
            key_l = key.lower()
            if any(x in key_l for x in ["keyword", "article:tag", "og:article:tag"]):
                terms.extend([s.strip() for s in re.split(r"[;,/|]", content) if s.strip()])
        links = re.findall(r'<a[^>]+(?:rel=["\']tag["\']|class=["\'][^"\']*(tag|category)[^"\']*["\'])[^>]*>(.*?)</a>', html, re.I|re.S)
        terms.extend([re.sub(r"<[^>]+>", "", t[1]).strip() for t in links if t[1].strip()])
        return _normalize_terms(terms)

    soup = BeautifulSoup(html, "html.parser")

    def _content_of(selector: str, attr: str = "content") -> List[str]:
        out: List[str] = []
        for el in soup.select(selector):
            v = el.get(attr)
            if isinstance(v, str):
                out.append(v)
        return out

    metas = []
    metas += _content_of('meta[property="article:tag"]')
    metas += _content_of('meta[property="og:article:tag"]')
    metas += _content_of('meta[name="keywords"]')
    metas += _content_of('meta[name="news_keywords"]')
    metas += _content_of('[itemprop="keywords"]', attr="content")

    for m in metas:
        metas_terms = [s.strip() for s in re.split(r"[;,/|]", m) if s.strip()]
        terms.extend(metas_terms)

    # Tag/category links
    for a in soup.find_all("a"):
        rel = " ".join(a.get("rel", [])).lower() if a.get("rel") else ""
        cls = " ".join(a.get("class", [])).lower() if a.get("class") else ""
        if ("tag" in rel) or ("tag" in cls) or ("category" in cls):
            txt = a.get_text(" ", strip=True)
            if txt:
                terms.append(txt)

    return _normalize_terms(terms)

def merge_categories_to_json(*groups: Iterable[str]) -> str:
    """
    Merge any number of category lists, de-duplicate, return JSON array string.
    Backward compatible with old 2-arg calls: merge_categories_to_json(list1, list2).
    """
    merged: List[str] = []
    for g in groups:
        if not g:
            continue
        for s in g:
            if isinstance(s, str):
                merged.append(s)
    merged = _normalize_terms(merged)
    return json.dumps(merged, ensure_ascii=False)

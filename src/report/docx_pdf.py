"""
docx_pdf.py — Rendering helpers for humanitarian MVP reports.

Inputs expected:
- sections: Dict[str, List[{"title": str|None, "summary": str, "key_points": str|None, "ref_num": int|None, "published_at": str|None, "url": str, "source_site": str|None, ...}]]
- references_by_num: Dict[int|str, {"url": str, "title": str|None, "as_of" or "published_at": str|None}]
- exec_overview: Optional[str] — 1–3 paragraphs of overview (can be empty)
- filename: Output file path ('.docx' or '.pdf')
- country: Lowercase key like "ukraine" (used in headings)
- report_title: Optional override for the document title
- section_order: Optional[List[str]] — explicit order of sections

This module uses:
- python-docx for .docx
- reportlab for .pdf
"""

from typing import Dict, List, Any, Optional, Iterable
from pathlib import Path
import re
from math import ceil

# --- DOCX ---
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

# --- PDF (ReportLab) ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from xml.sax.saxutils import escape as _esc


# ------------------------ Common helpers ------------------------

def _pretty_country(country: str) -> str:
    return (country or "").strip().title()

def _default_title(country: str) -> str:
    return f"Humanitarian Situation Digest: {_pretty_country(country)}"

def _iter_sections_in_order(sections: Dict[str, Any], order: Optional[Iterable[str]]):
    if order:
        for k in order:
            if k in sections:
                yield k, sections[k]
    else:
        for k, v in sections.items():
            yield k, v

def _sorted_ref_nums(refs: dict) -> list:
    """Sort reference keys numerically even if JSON loaded them as strings."""
    def _k(n):
        try:
            return int(n)
        except Exception:
            return n
    return sorted(refs.keys(), key=_k)

def _ymd(dt_str: Optional[str]) -> str:
    """Format ISO-like or RFC3339-like dates to YYYY-MM-DD; fallback empty."""
    if not dt_str:
        return ""
    # assume first 10 chars are date in most sources
    return str(dt_str)[:10]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _shorten_key_points_twofold(text: str) -> str:
    """
    Reduce key points roughly two-fold by sentence count:
    keep the first ceil(n/2) sentences.
    """
    t = (text or "").strip()
    if not t:
        return ""
    sents = _SENT_SPLIT.split(t)
    if len(sents) <= 1:
        return t
    keep = max(1, ceil(len(sents) / 2))
    return " ".join(sents[:keep]).strip()


# ------------------------ DOCX utilities ------------------------

def _add_hyperlink(paragraph, url: str, text: str, color="0000FF", underline=True):
    """Create a clickable hyperlink within a paragraph with blue/underline style."""
    if not url:
        r = paragraph.add_run(text)
        return r
    # relationship id
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    # hyperlink tag
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    # run with properties
    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    if color:
        c = OxmlElement('w:color'); c.set(qn('w:val'), color); rPr.append(c)
    if underline:
        u = OxmlElement('w:u'); u.set(qn('w:val'), 'single'); rPr.append(u)
    new_run.append(rPr)
    t = OxmlElement('w:t'); t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink

def _strip_table_borders(tbl):
    """Remove all table borders in a python-docx table."""
    tblPr = tbl._tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl._tbl.insert(0, tblPr)
    # Create/replace tblBorders with nil edges
    borders = tblPr.find(qn('w:tblBorders'))
    if borders is None:
        borders = OxmlElement('w:tblBorders')
        tblPr.append(borders)

    # clear previous children
    for child in list(borders):
        borders.remove(child)

    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        tag = OxmlElement(f'w:{edge}')
        tag.set(qn('w:val'), 'nil')
        borders.append(tag)

def _set_cell_bottom_border(cell, sz=12, color="000000"):
    """Add only a bottom border to a cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.find(qn('w:tcBorders'))
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        tcPr.append(borders)
    # remove previous bottom if exists
    for child in list(borders):
        if child.tag == qn('w:bottom'):
            borders.remove(child)
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), str(int(sz)))
    bottom.set(qn('w:color'), color)
    borders.append(bottom)


# ------------------ DOCX ------------------

def generate_docx_structured(
    sections: Dict[str, List[Dict[str, Any]]],
    references_by_num: Dict[int, Dict[str, Any]],
    exec_overview: str,
    filename: str,
    country: str,
    report_title: Optional[str] = None,
    section_order: Optional[List[str]] = None,
) -> None:
    doc = Document()
    title = report_title or _default_title(country)

    # Title
    h = doc.add_heading(title, 0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Executive summary (optional)
    if exec_overview:
        doc.add_heading("Executive Summary", level=1)
        for block in [b.strip() for b in exec_overview.split("\n\n") if b.strip()]:
            p = doc.add_paragraph()
            run = p.add_run(block)
            run.font.size = Pt(11)

    # Sections in provided order
    first_section = True
    for section_name, items in _iter_sections_in_order(sections, section_order):
        if not items:
            continue
        if first_section:
            first_section = False
        else:
            doc.add_page_break()   # page break before each section
            doc.add_heading(section_name, level=1)

        # Separate narrative (no ref_num) and article items
        article_items = [it for it in items if it.get("ref_num") is not None]
        narrative_items = [it for it in items if it.get("ref_num") is None]

        # Narrative blocks (Summary / Why support / Conclusion)
        for it in narrative_items:
            text = (it.get("summary") or "").strip()
            if text:
                for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
                    p = doc.add_paragraph()
                    run = p.add_run(block)
                    run.font.size = Pt(11)

        # Articles: one Evidence—Key points table per section
        if article_items:
            tbl = doc.add_table(rows=len(article_items) + 1, cols=2)
            # Remove all borders and then add a single line below the header row
            _strip_table_borders(tbl)

            # Header row with bottom border only
            hdr0 = tbl.cell(0, 0).paragraphs[0].add_run("Evidence"); hdr0.bold = True
            hdr1 = tbl.cell(0, 1).paragraphs[0].add_run("Key points"); hdr1.bold = True
            for cell in tbl.rows[0].cells:
                _set_cell_bottom_border(cell, sz=16, color="000000")

            for r, it in enumerate(article_items, start=1):
                url = it.get("url", "")
                title_txt = it.get("short_title") or it.get("title") or "Untitled"
                src = it.get("source_site") or "Source"
                ref = it.get("ref_num")

                # Pick a date: prefer item's own; fallback to references_by_num
                dt_raw = it.get("published_at") or (references_by_num.get(ref, {}) if ref else {}).get("as_of") or \
                         (references_by_num.get(ref, {}) if ref else {}).get("published_at")
                date_ymd = _ymd(dt_raw)

                # Evidence cell: Title (hyperlink), Source (hyperlink), Date
                p_e = tbl.cell(r, 0).paragraphs[0]
                _add_hyperlink(p_e, url, title_txt, color="0000FF", underline=True)
                p_e2 = tbl.cell(r, 0).add_paragraph("— ")
                _add_hyperlink(p_e2, url, src, color="0000FF", underline=True)
                if date_ymd:
                    tbl.cell(r, 0).add_paragraph(date_ymd)

                # Key points cell (shortened by 2x; append [n])
                base_kp = (it.get("key_points") or it.get("summary") or "").strip()
                kp_short = base_kp # _shorten_key_points_twofold(base_kp)
                if ref and not kp_short.rstrip().endswith(f"[{ref}]"):
                    kp_short = f"{kp_short.rstrip()} [{ref}]"
                tbl.cell(r, 1).paragraphs[0].add_run(kp_short)

            for row in tbl.rows:
                for cell in row.cells:
                    _set_cell_bottom_border(cell, sz=8, color="000000")

    # References (with clickable [n], blue + underline, short dates, and full URL hyperlink visible)
    if references_by_num:
        doc.add_page_break()
        doc.add_heading("References", level=1)
        for num in _sorted_ref_nums(references_by_num):
            meta = references_by_num[num] or {}
            url = meta.get("url", "") or ""
            title_txt = meta.get("title") or url
            dt_short = _ymd(meta.get("as_of") or meta.get("published_at") or "")

            p = doc.add_paragraph()
            _add_hyperlink(p, url, f"[{num}]", color="0000FF", underline=True)
            p.add_run(" " + title_txt)
            if dt_short:
                p.add_run(f" — {dt_short}")
            if url:
                p.add_run(" — ")
                _add_hyperlink(p, url, url, color="0000FF", underline=True)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    doc.save(filename)


# ------------------ PDF ------------------

def generate_pdf_structured(
    sections: Dict[str, List[Dict[str, Any]]],
    references_by_num: Dict[int, Dict[str, Any]],
    exec_overview: str,
    filename: str,
    country: str,
    report_title: Optional[str] = None,
    section_order: Optional[List[str]] = None,
) -> None:
    title = report_title or _default_title(country)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER)
    h1 = ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=6, spaceAfter=6)
    body = ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14)

    flow = []
    flow.append(Paragraph(title, title_style))
    flow.append(Spacer(1, 12))

    # Executive Summary
    if exec_overview:
        flow.append(Paragraph("Executive Summary", h1))
        flow.append(Spacer(1, 6))
        for block in [b.strip() for b in exec_overview.split("\n\n") if b.strip()]:
            flow.append(Paragraph(_esc(block), body))
            flow.append(Spacer(1, 6))
        flow.append(Spacer(1, 12))

    # Sections in order
    first_section = True
    for section_name, items in _iter_sections_in_order(sections, section_order):
        if not items:
            continue
        if first_section:
            first_section = False
        else:
            flow.append(PageBreak())   # page break before each section

        flow.append(Paragraph(_esc(section_name), h1))
        flow.append(Spacer(1, 6))

        # Narrative
        narrative_items = [it for it in items if it.get("ref_num") is None]
        for it in narrative_items:
            text = (it.get("summary") or "").strip()
            if text:
                for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
                    flow.append(Paragraph(_esc(block), body))
                    flow.append(Spacer(1, 6))

        # Articles table
        article_items = [it for it in items if it.get("ref_num") is not None]
        if article_items:
            # Header
            data = [[Paragraph("<b>Evidence</b>", body), Paragraph("<b>Key points</b>", body)]]
            for it in article_items:
                url = it.get("url", "")
                title_txt = _esc(it.get("short_title") or it.get("title") or "Untitled")
                src = _esc(it.get("source_site") or "Source")
                ref = it.get("ref_num")

                # Date selection (same as DOCX) and format to YYYY-MM-DD
                meta = references_by_num.get(ref, {}) if ref else {}
                dt_raw = it.get("published_at") or meta.get("as_of") or meta.get("published_at")
                date_ymd = _ymd(dt_raw)

                # Evidence links (title + source, then date)
                ev_html = (
                    f'<link href="{url}" color="blue" underline="1">{title_txt}</link>'
                    f'<br/>— <link href="{url}" color="blue" underline="1">{src}</link>'
                )
                if date_ymd:
                    ev_html += f"<br/>{_esc(date_ymd)}"

                # Key points (shortened 2x; make [n] clickable, blue + underlined)
                base_kp = (it.get("key_points") or it.get("summary") or "").strip()
                kp_short = base_kp # _shorten_key_points_twofold(base_kp)
                if ref and f"[{ref}]" not in kp_short:
                    kp_short = f"{kp_short.rstrip()} [{ref}]"
                if ref:
                    kp_html = _esc(kp_short).replace(
                        f"[{ref}]",
                        f'<link href="{url}" color="blue" underline="1">[{ref}]</link>'
                    )
                else:
                    kp_html = _esc(kp_short)

                data.append([Paragraph(ev_html, body), Paragraph(kp_html, body)])

            tbl = Table(data, colWidths=[180, 300])

            # Clean style: only a line below the header row
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                ("LINEBELOW", (0,0), (-1,-1), 0.5, colors.black),  # line under each row
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ]))
            flow.append(tbl)
            flow.append(Spacer(1, 12))

    # References: [n] link, short date, and full URL as clickable text
    if references_by_num:
        flow.append(PageBreak())
        flow.append(Paragraph("References", h1))
        flow.append(Spacer(1, 6))
        for num in _sorted_ref_nums(references_by_num):
            meta = references_by_num[num] or {}
            url = meta.get("url", "") or ""
            title_txt = _esc(meta.get("title") or url)
            dt_short = _ymd(meta.get("as_of") or meta.get("published_at") or "")
            line = f'<link href="{url}" color="blue" underline="1">[{num}]</link> {title_txt}'
            if dt_short:
                line += f" — {_esc(dt_short)}"
            if url:
                line += f' — <link href="{url}" color="blue" underline="1">{_esc(url)}</link>'
            flow.append(Paragraph(line, body))
            flow.append(Spacer(1, 4))

    doc = SimpleDocTemplate(filename, pagesize=letter, title=title)
    doc.build(flow)

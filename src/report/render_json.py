import re, json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..config import REPORTS_DIR
from .docx_pdf import generate_docx_structured, generate_pdf_structured

def _find_latest_json(country: str) -> Optional[Path]:
    country = country.lower()
    candidates = sorted((REPORTS_DIR).glob(f"{country}_*.json"), reverse=True)
    return candidates[0] if candidates else None

def _default_outputs(report: Dict[str, Any], json_path: Path) -> (Path, Path):
    country = (report.get("country") or "report").lower()
    as_of = report.get("as_of") or datetime.now().date().isoformat()
    base = f"humanitarian_digest_{country}_{as_of}"
    out_docx = REPORTS_DIR / f"{base}.docx"
    out_pdf  = REPORTS_DIR / f"{base}.pdf"
    return out_docx, out_pdf

def render_from_json(json_path: str,
                     report_title: str,
                     out_docx: Optional[str] = None,
                     out_pdf: Optional[str] = None):
    p = Path(json_path)
    report = json.loads(p.read_text(encoding="utf-8"))
    #report = _ensure_min_fields(report)

    sections = report["sections"]
    references = report.get("references_by_num", {}) or {}
    order = report.get("section_order")
    country = report.get("country", "report").lower()
    overview_texts = report.get("overview_texts") or []
    exec_overview = "\n\n".join(t for t in overview_texts if t and t.strip())

    # Output paths
    d_docx, d_pdf = _default_outputs(report, p)
    out_docx = Path(out_docx) if out_docx else d_docx
    out_pdf  = Path(out_pdf) if out_pdf else d_pdf

    # Generate DOCX/PDF
    generate_docx_structured(
        sections=sections,
        references_by_num=references,
        exec_overview=exec_overview,
        filename=str(out_docx),
        country=country,
        report_title=report_title,
        section_order=order,
    )
    generate_pdf_structured(
        sections=sections,
        references_by_num=references,
        exec_overview=exec_overview,
        filename=str(out_pdf),
        country=country,
        report_title=report_title,
        section_order=order,
    )
    return str(out_docx), str(out_pdf)

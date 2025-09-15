# src/validate/calibration.py
import re, sqlite3, datetime as dt
from statistics import mean, pstdev
from typing import Dict, Any
from ..config import DB_PATH

_NORM_RE = re.compile(r"norm=([0-9]+(?:\.[0-9]+)?)")

def _extract_norm(s: str | None) -> float | None:
    if not s: 
        return None
    m = _NORM_RE.search(s)
    return float(m.group(1)) if m else None

def compute_calibration(country: str, days: int = 14) -> Dict[str, Any]:
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat() + "Z"
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # Pull recent labels for the country
        c.execute("""
            SELECT classifier_reason
            FROM page_labels
            WHERE country_key = ? AND (labeled_at IS NULL OR labeled_at >= ?)
        """, (country, since))
        norms = []
        for (reason,) in c.fetchall():
            v = _extract_norm(reason)
            if v is not None:
                norms.append(v)
    if not norms:
        return {"mean": 0.0, "std": 1.0, "p70": 0.0, "n": 0}
    norms_sorted = sorted(norms)
    def pct(p: float) -> float:
        if not norms_sorted: return 0.0
        i = int(round(p * (len(norms_sorted)-1)))
        i = max(0, min(len(norms_sorted)-1, i))
        return norms_sorted[i]
    mu = mean(norms)
    sd = pstdev(norms) if len(norms) > 1 else 1.0
    return {"mean": mu, "std": sd, "p70": pct(0.70), "n": len(norms)}

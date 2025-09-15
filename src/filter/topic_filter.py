import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Dict, Any, List, Optional

_YAML = None

# ---------- New knobs (tweak if needed) ----------
CATEGORY_BOOST_FACTOR: float = 3.0   # when a bag's patterns match categories, boost that bag's subscore
CORE_CATEGORY_BOOST: float   = 2.0   # if any 'core' keyword appears in categories, multiply the core subscore


def load_bags(path: str | Path = None) -> dict:
    import yaml
    global _YAML
    if _YAML is None:
        p = Path(path or Path(__file__).with_name("keyword_bags.yaml"))
        _YAML = yaml.safe_load(p.read_text(encoding="utf-8"))
    return _YAML

def _ensure_patterns(val) -> List[str]:
    """Accept legacy list OR new {title, patterns} dict."""
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return val.get("patterns", []) or []
    return []

def _merged_section(
    bags: dict, country: str, node_key: str
) -> Tuple[str, List[Tuple[str, re.Pattern]], float]:
    """
    Merge global.sections[node_key] with countries[country][node_key].
    Title precedence: per-country > global.
    Patterns: global + country (dedup, keep order).
    Weight: from per-country summary_gate.weights, fallback 1.0.

    Returns:
      (title, [(pattern_str, compiled_regex), ...], weight)
    """
    cc = bags["countries"][country]
    weights = ((cc.get("summary_gate") or {}).get("weights") or {})
    glob_sec = ((bags.get("sections") or {}).get(node_key) or {})
    country_sec = (cc.get(node_key) or {})

    # Title precedence
    title = (
        (country_sec.get("title") if isinstance(country_sec, dict) else None)
        or (glob_sec.get("title") if isinstance(glob_sec, dict) else None)
        or node_key.replace("_", " ").title()
    )

    gp = _ensure_patterns(glob_sec)
    cp = _ensure_patterns(country_sec)
    merged = list(gp) + [p for p in cp if p not in gp]

    pats: List[Tuple[str, re.Pattern]] = [(p, re.compile(p, re.I)) for p in merged]
    w = float(weights.get(node_key, 1.0))
    return title, pats, w

def _build_section_defs(
    bags: dict, country: str
) -> List[Tuple[str, List[Tuple[str, re.Pattern]], float]]:
    """Return list of (title, [(pattern_str, compiled), ...], weight) for the evidence sections."""
    keys = ["human_harm", "health", "energy", "water", "food"]
    defs = []
    for k in keys:
        title, pats, w = _merged_section(bags, country, k)
        defs.append((title, pats, w))
    return defs

def _resolve_global_order(bags: dict, titles: List[str]) -> List[str]:
    """Use global section_order; append any titles missing in that order at the end."""
    want = [t.strip() for t in (bags.get("section_order") or []) if str(t).strip()]
    if not want:
        return titles
    seen = set()
    ordered = []
    for t in want:
        if t in titles and t not in seen:
            ordered.append(t); seen.add(t)
    for t in titles:
        if t not in seen:
            ordered.append(t); seen.add(t)
    return ordered

def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.I) for p in patterns]

def _get_domain_from_url(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        parts = domain.split(".")
        parts.reverse()
        if parts and len(parts) > 1:
            return parts[1] + "." + parts[0]
        return domain
    except Exception:
        return ""


def _ensure_patterns(val) -> List[str]:
    """Accept legacy list OR new {title, patterns} dict."""
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return val.get("patterns", []) or []
    return []


def _merged_section_patterns(bags: dict, country_key: str, node_key: str) -> List[str]:
    """Merge global.sections[node_key].patterns with countries[country_key][node_key].patterns."""
    global_node = ((bags.get("sections") or {}).get(node_key) or {})
    country_node = ((bags.get("countries") or {}).get(country_key, {}).get(node_key) or {})
    gp = _ensure_patterns(global_node)
    cp = _ensure_patterns(country_node)
    return list(gp) + [p for p in cp if p not in gp]  # keep order, avoid dupes


class CountryGate:
    """
    Uses merged global+country patterns for the five evidence groups.
    - score(): fast boolean gate
    - summary_score(): frequency-based score (now category-aware)
    """
    def __init__(self, country_key: str, bags: dict | None = None,
                 primary_city: Optional[str] = None):
        bags = bags or load_bags()
        if country_key not in bags["countries"]:
            raise ValueError(f"Unknown country_key: {country_key}")
        self.country_key = country_key
        c = bags["countries"][country_key]

        # hard token
        self.must_token = re.compile(c.get("must_token", ""), re.I) if c.get("must_token") else None

        # allowlist
        self.allow = set(c.get("allow_domains", [])) or set(bags.get("allow_domains", []))

        # generic positives (legacy keys remain list-like)
        self.core   = _compile(_ensure_patterns(c.get("core", [])))
        self.cities = _compile(_ensure_patterns(c.get("cities", [])))
        self.actors = _compile(_ensure_patterns(c.get("actors", [])))
        self.mil    = _compile(_ensure_patterns(c.get("military", [])))

        # evidence groups: merged global + country
        self.harm   = _compile(_merged_section_patterns(bags, country_key, "human_harm"))
        self.health = _compile(_merged_section_patterns(bags, country_key, "health"))
        self.energy = _compile(_merged_section_patterns(bags, country_key, "energy"))
        self.water  = _compile(_merged_section_patterns(bags, country_key, "water"))
        self.food   = _compile(_merged_section_patterns(bags, country_key, "food"))

        # negatives
        self.neg_geo_only = _compile(bags.get("negative_geo_only", []))
        self.neg_exclude  = _compile(bags.get("negative_exclude", []))
        self.neg_frontline = _compile(bags.get("negative_frontline", []))

        # thresholds
        th = bags.get("thresholds", {})
        self.t_allow   = int(th.get("allowlist_keep", 60))
        self.t_default = int(th.get("default_keep", 65))
        self.t_world   = int(th.get("world_hub_keep", 70))

        # weights (prefer per-country summary_gate.weights)
        default_weights = {
            "core": 8, "cities": 5, "actors": 3, "military": 4,
            "human_harm": 4, "health": 4, "energy": 4, "water": 3, "food": 3,
            "city_geo": 10,
            "neg_geo_only": -5, "neg_exclude": -8, "neg_frontline": -6,
            "allow_bonus": 2, "url_country_bonus": 1,
        }
        sg = (c.get("summary_gate") or {})
        cw = (sg.get("weights") or {})
        self.weights = {**default_weights, **cw}

        # primary city
        pc = (primary_city or c.get("primary_city") or "").strip()
        self.city_geo = [re.compile(rf"\b{re.escape(pc)}\b", re.I)] if pc else []

    def _hit(self, pats, text: str) -> bool:
        return any(p.search(text) for p in pats)

    def _count(self, pats, text: str) -> int:
        cnt = 0
        for p in pats:
            cnt += sum(1 for _ in p.finditer(text))
        return cnt

    def score(self, title: str, summary: str, url: str) -> Tuple[bool, int, Dict[str, Any]]:
        t = f"{title or ''} {summary or ''}".lower()
        host = _get_domain_from_url(url)

        if self.must_token and not self.must_token.search(t):
            return False, 0, {"reason": "missing_must_token_pre", "host": host}

        allow = host in self.allow

        # positives
        core   = self._hit(self.core, t)
        city   = self._hit(self.cities, t)
        actor  = self._hit(self.actors, t)
        mil    = self._hit(self.mil, t)
        harm   = self._hit(self.harm, t)
        health = self._hit(self.health, t)
        energy = self._hit(self.energy, t)
        water  = self._hit(self.water, t)
        food   = self._hit(self.food, t)
        city_geo_hit = self._hit(self.city_geo, t) if self.city_geo else False

        # negatives
        neg_geo = self._hit(self.neg_geo_only, t)
        neg_x   = self._hit(self.neg_exclude, t)
        neg_front  = self._hit(self.neg_frontline, t)

        score = 0
        if core:   score += 30
        if city:   score += 25
        if harm:   score += 20
        if health: score += 20
        if energy: score += 20
        if water:  score += 20
        if food:   score += 20
        if actor:  score += 15
        if mil:    score += 10
        if city_geo_hit:
            score += max(1, int(self.weights.get("city_geo", 10) // 2))

        if allow: score += 10
        if f"/{self.country_key}" in (url or "").lower(): score += 5

        if neg_geo:    score -= 25
        if neg_x:      score -= 40
        if neg_front:  score -= 50

        threshold = self.t_allow if allow else self.t_default
        if "/world" in (url or "").lower() and not allow:
            threshold = max(threshold, self.t_world)

        return score >= threshold, score, {
            "host": host, "allowlisted": allow, "threshold": threshold,
            "core": core, "city": city, "actor": actor, "mil": mil,
            "harm": harm, "health": health, "energy": energy, "water": water, "food": food,
            "city_geo": city_geo_hit,
            "neg_geo": neg_geo, "neg_exclude": neg_x, "neg_frontline": neg_front,
            "method": "score"
        }

    def summary_score(
        self,
        title: str,
        summary: str,
        url: str,
        categories: Optional[List[str]] = None,  # <-- NEW (optional)
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """
        Frequency-based score (bag counts * weights), now category-aware:
        - If categories contain a bag's keywords -> that bag's subscore is multiplied by CATEGORY_BOOST_FACTOR.
        - If categories contain any 'core' keyword -> the 'core' subscore is further multiplied by CORE_CATEGORY_BOOST.
        """
        t = f"{title or ''} {summary or ''}".lower()
        host = _get_domain_from_url(url) or ""
        allow = host in self.allow

        counts = {
            "core":          self._count(self.core, t),
            "cities":        self._count(self.cities, t),
            "actors":        self._count(self.actors, t),
            "military":      self._count(self.mil, t),
            "human_harm":    self._count(self.harm, t),
            "health":        self._count(self.health, t),
            "energy":        self._count(self.energy, t),
            "water":         self._count(self.water, t),
            "food":          self._count(self.food, t),
            "city_geo":      self._count(self.city_geo, t) if self.city_geo else 0,
            "neg_geo_only":  self._count(self.neg_geo_only, t),
            "neg_exclude":   self._count(self.neg_exclude, t),
            "neg_frontline": self._count(self.neg_frontline, t),
        }

        # Base contribs (before boosts)
        contrib_before = {k: counts[k] * self.weights.get(k, 0) for k in counts}

        # ------------- Category-aware boosts -------------
        cat_list = [s for s in (categories or []) if str(s).strip()]
        cat_text = " ".join(cat_list).lower() if cat_list else ""

        # Hits in categories per bag (only for positive/neutral bags)
        cat_hits_by_bag: Dict[str, int] = {}
        if cat_text:
            cat_hits_by_bag = {
                "core":       self._count(self.core, cat_text),
                "cities":     self._count(self.cities, cat_text),
                "actors":     self._count(self.actors, cat_text),
                "military":   self._count(self.mil, cat_text),
                "human_harm": self._count(self.harm, cat_text),
                "health":     self._count(self.health, cat_text),
                "energy":     self._count(self.energy, cat_text),
                "water":      self._count(self.water, cat_text),
                "food":       self._count(self.food, cat_text),
                # generally we don't expect city_geo/negatives in categories; skip those
            }
        else:
            cat_hits_by_bag = {k: 0 for k in ["core","cities","actors","military","human_harm","health","energy","water","food"]}

        # Apply per-bag CATEGORY boost
        contrib_after = dict(contrib_before)
        for bag, hits_in_text in counts.items():
            if bag not in cat_hits_by_bag:
                continue
            if cat_hits_by_bag[bag] > 0 and hits_in_text > 0:
                contrib_after[bag] = contrib_after[bag] * CATEGORY_BOOST_FACTOR

        # Additional CORE boost if any core keyword appears in categories
        core_cat_hit = (cat_hits_by_bag.get("core", 0) > 0)
        if core_cat_hit and counts.get("core", 0) > 0 and contrib_after.get("core", 0) != 0:
            contrib_after["core"] = contrib_after["core"] * CORE_CATEGORY_BOOST

        # -------------------------------------------------

        # Sum base (negatives included), then add bonuses (these are not boosted)
        score = sum(contrib_after.values())

        if allow:
            score += self.weights.get("allow_bonus", 0)
            contrib_after["allow_bonus"] = self.weights.get("allow_bonus", 0)
        if f"/{self.country_key}" in (url or "").lower():
            score += self.weights.get("url_country_bonus", 0)
            contrib_after["url_country_bonus"] = self.weights.get("url_country_bonus", 0)

        threshold = self.t_allow if allow else self.t_default
        if "/world" in (url or "").lower() and not allow:
            threshold = max(threshold, self.t_world)

        keep = score >= threshold
        details = {
            "host": host,
            "counts": counts,
            "contrib_before_boost": contrib_before,
            "contrib_after_boost": contrib_after,
            "boosted_score": score,
            "threshold": threshold,
            "allowlisted": allow,
            "cat_hits_by_bag": cat_hits_by_bag,
            "core_cat_hit": core_cat_hit,
            "method": "summary_score",
        }
        return keep, int(score), details

    def strict_token_in_fulltext(self, text: str) -> bool:
        if not self.must_token:
            return True
        return bool(self.must_token.search(text.lower()))

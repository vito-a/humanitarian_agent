# src/utils/coat_of_arms.py
from __future__ import annotations
import pathlib
import requests
from typing import Optional
import cairosvg
from ..config import COAT_OF_ARMS_DEFAULT_PATH

# Where to store downloaded/converted images
COAT_OF_ARMS_DEFAULT_DIR = pathlib.Path(COAT_OF_ARMS_DEFAULT_PATH)
COAT_OF_ARMS_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

UA = "HumanitarianAgent/1.0 (+https://example.org)"
TIMEOUT = 20

def _country_title(country: str) -> str:
    return (country or "").strip().title()


def _wiki_title_variants(country: str) -> list[str]:
    ct = _country_title(country)
    return [
        f"Coat_of_arms_of_{ct.replace(' ', '_')}",
        f"National_emblem_of_{ct.replace(' ', '_')}",
        f"Emblem_of_{ct.replace(' ', '_')}",
    ]

def _summary_endpoint(title: str) -> str:
    return f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def _get_original_image_url_from_summary(title: str) -> Optional[str]:
    url = _summary_endpoint(title)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        # Prefer originalimage, then thumbnail
        img = (data.get("originalimage") or {}).get("source")
        if img:
            return img
        thumb = (data.get("thumbnail") or {}).get("source")
        return thumb
    except Exception:
        return None


def _download_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def _looks_svg(url: str, data: Optional[bytes]) -> bool:
    if url and url.lower().endswith((".svg", ".svgz")):
        return True
    if data:
        head = data[:200].lower()
        return (b"<svg" in head) or (b"<!doctype svg" in head)
    return False


def _to_png_bytes(url: str, data: bytes) -> Optional[bytes]:
    """
    Convert SVG → PNG if cairosvg is available; if already PNG/JPEG, just return bytes.
    """
    if _looks_svg(url, data):
        if cairosvg is None:
            # No converter installed
            return None
        try:
            # Convert using cairosvg
            return cairosvg.svg2png(bytestring=data)
        except Exception:
            # As a fallback, try converting by URL (some SVG reference external resources)
            try:
                return cairosvg.svg2png(url=url)
            except Exception:
                return None
    # Assume it's already a raster (png/jpg)
    return data


def ensure_coat_of_arms(country: str, out_dir: pathlib.Path | None = None) -> Optional[pathlib.Path]:
    """
    Ensure there's a local PNG for the country's coat of arms.
    Returns the path to the PNG (or None on failure).

    Strategy:
    - Try Wikipedia REST summary on a few title variants (Coat_of_arms_of_*, Emblem_of_*).
    - Fetch the image; if SVG, convert to PNG (if cairosvg installed).
    - Save to assets/coat_of_arms/{country_key}.png
    """
    out_dir = out_dir or COAT_OF_ARMS_DEFAULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # canonical output path
    country_key = (country or "").strip().lower().replace(" ", "_")
    out_path = out_dir / f"{country_key}.png"

    # Already present?
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Coat of arms exists for title={country}\n")
        return out_path

    # Try multiple title variants
    for title in _wiki_title_variants(country):
        img_url = _get_original_image_url_from_summary(title)
        print("Coat of arms loaded for title={title}: img_url={img_url} \n")
        if not img_url:
            continue

        data = _download_bytes(img_url)
        if not data:
            continue

        png_data = _to_png_bytes(img_url, data)
        if not png_data:
            # If SVG and no converter, skip this variant
            continue

        try:
            out_path.write_bytes(png_data)
            return out_path
        except Exception:
            # Try another variant
            continue

    # If we reach here, no success
    return None

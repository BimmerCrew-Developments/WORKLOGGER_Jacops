#!/usr/bin/env python3
"""
SSV ZIP Processor (Images + PDF) — GUI + CLI (Final)

What it does
------------
Given a SafetyAuditor ZIP export that contains:
  - exactly one CSV export (vertical "ID,Type,Label,...,Media" format)
  - multiple JPEG images named <media_id>.jpeg inside the ZIP
  - (optional) one or more PDF attachments (kept unchanged and included in output ZIP)

This tool will:
  1) extract the ZIP to a temp folder
  2) rename + compress all referenced JPEGs (<= 350 KB each) into the chosen output folder
  3) generate a "WERKLOGGER RAPPORT" style PDF in the same output folder, matching the provided
     reference layout (2 photos side-by-side per row; no approver signature).

Key rules
---------
- Image naming comes from CSV Label, normalized to OS-safe filenames
- Media IDs in CSV are semicolon-separated; extension is .jpeg
- Images are compressed iteratively to <= 350 KB (no EXIF/metadata)
- PDF sections "Gebruikte materialen" and "Post Afmeldingen" are populated from quantity rows in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# GUI (optional at import time; required for desktop use)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    filedialog = messagebox = scrolledtext = None
# Images
from PIL import Image, ImageOps

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader



# =========================
# Data models
# =========================
@dataclass
class AuditRow:
    row_id: str
    parent_id: str
    row_type: str
    label: str
    primary: str
    secondary: str
    note: str
    media: str


@dataclass
class MediaRow:
    label: str
    media_ids: List[str]


@dataclass
class Photo:
    label: str
    image_path: Path  # processed output path


@dataclass
class ArticleRow:
    code: str
    description: str
    unit: str
    quantity: float


@dataclass
class ReportRow:
    report_datetime: str
    naam_onderaannemer: str
    project_locatie_naam: str
    building_id: str
    adres: str
    postcode_stad: str
    contactpersoon: str
    quadrant: str
    duct_kleur: str
    units_gelast: str
    gebruikte_materialen_lines: List[str]
    post_afmeldingen_lines: List[str]
    photos: List[Photo]


@dataclass
class ProcessResult:
    report: ReportRow
    pdf_path: Path
    written_images: int
    output_zip_path: Optional[Path] = None
    copied_pdfs: int = 0


# =========================
# Utility helpers
# =========================
MAX_BYTES = 350 * 1024


def normalize_label(label: str) -> str:
    """Normalize label for filenames (OS-safe)."""
    s = (label or "").strip()
    if not s:
        s = "UNLABELED"
    s = s.replace(" ", "_").replace("?", "").replace(":", "")
    # Remove Windows forbidden filename chars and path separators
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._ ")
    return s or "UNLABELED"


def safe_filename(name: str) -> str:
    """More aggressive filename sanitization."""
    s = (name or "").strip()
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._ ")
    return s or "output"


def parse_list_value(value: Optional[str]) -> str:
    """Normalize list-like exported values.

    SafetyAuditor list exports often look like:
      <uuid>|T3(x)
      <uuid>|T3(x);<uuid>|T2(x)

    We strip the UUID part and keep only the human-readable option text.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"none", "null"}:
        return ""

    # Split on ';' first (multi-select lists)
    raw_parts = [p.strip() for p in s.split(";") if p.strip() and p.strip().lower() not in {"none", "null"}]
    parts: list[str] = []

    uuid_prefix = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\|")

    for p in raw_parts:
        # If token contains uuid|value, keep only value
        if "|" in p:
            p = p.split("|")[-1].strip()
        p = uuid_prefix.sub("", p).strip()
        if p:
            parts.append(p)

    if len(parts) >= 2:
        return ", ".join(parts)
    if len(parts) == 1:
        return parts[0]

    # Fallback: strip uuid| if it exists
    if "|" in s:
        return s.split("|")[-1].strip()
    return s



def sanitize_address(value: str) -> str:
    """Remove coordinates/newlines/odd glyphs from the exported address."""
    s = (value or "")
    # Replace control whitespace (ReportLab will render \n as a square glyph)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Remove coordinate suffix like: (50.8255728, 3.265294)
    s = re.sub(r"\(\s*-?\d+(?:[\.,]\d+)?\s*,\s*-?\d+(?:[\.,]\d+)?\s*\)\s*$", "", s).strip()

    # Remove stray black-square / replacement chars if present
    s = s.replace("\u25A0", " ").replace("\uFFFD", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s
def _split_semicolon_ids(value: str) -> List[str]:
    parts = []
    for p in (value or "").split(";"):
        p = p.strip()
        if not p:
            continue
        parts.append(p)
    return parts


def parse_media_ids(media_value: Optional[str], note_value: Optional[str] = None) -> List[str]:
    """Extract media ids from Media column (preferred) and optionally Note."""
    ids: List[str] = []
    if media_value:
        ids.extend(_split_semicolon_ids(str(media_value)))

    # Some exports put IDs in note; keep it conservative to avoid false positives
    if note_value and not ids:
        note = str(note_value)
        candidates = re.findall(r"\b[a-f0-9]{8,}\b", note, flags=re.IGNORECASE)
        # Only accept if it looks like one or more IDs
        for c in candidates:
            ids.append(c)

    # De-dup while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for i in ids:
        k = i.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(i)
    return out


def fmt_epoch(value: str) -> str:
    """Convert milliseconds/seconds epoch into dd/mm/YYYY HH:MM when possible."""
    if not value:
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"none", "null"}:
        return ""
    # Some exports store ms epoch
    try:
        n = int(float(s))
        # Heuristic: if very large, treat as ms
        if n > 10_000_000_000:
            n = n // 1000
        return dt.datetime.fromtimestamp(n).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return ""


def compress_jpeg_to_limit(im: Image.Image, max_bytes: int = MAX_BYTES) -> bytes:
    """Compress a PIL image to JPEG <= max_bytes by lowering quality."""
    # We always store RGB, no metadata
    if im.mode != "RGB":
        im = im.convert("RGB")

    # Start high, go down
    for q in [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data

    # If still too big, attempt small downscale steps
    w, h = im.size
    for scale in [0.9, 0.8, 0.7]:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        im2 = im.resize((nw, nh), Image.LANCZOS)
        for q in [70, 60, 50, 40, 30]:
            buf = io.BytesIO()
            im2.save(buf, format="JPEG", quality=q, optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data

    # Last resort: return smallest we got
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=25, optimize=True)
    return buf.getvalue()




# =========================
# Project list persistence (GUI dropdown)
# =========================

DEFAULT_PROJECTS: list[str] = [
    "MRO_ARDOOIE_01",
    "MRO_ARDOOIE_02",
    "MRO_ARDOOIE_03",
    "MRO_INGELMUNSTER_01",
    "MRO_MEULEBEKE_01",
    "MRO_OOSTROZEBEKE_01",
    "MRO_ROESELARE_02",
    "MRO_ROESELARE_03",
    "MRO_ROESELARE_04",
    "MRO_ROESELARE_05",
    "MRO_ROESELARE_06",
    "MRO_ROESELARE_07",
    "MRO_ROESELARE_08",
    "MRO_ROESELARE_09",
    "MRO_ROESELARE_10",
    "MRO_ROESELARE_11",
    "MRO_ROESELARE_12",
    "MRO_ROESELARE_13",
    "MRO_ROESELARE_14",
    "MRO_ROESELARE_15",
    "MRO_ROESELARE_16",
    "MRO_ROESELARE_17",
    "MRO_ROESELARE_18",
    "MRO_ROESELARE_19",
    "MRO_ROESELARE_20",
    "MRO_ROESELARE_21",
    "MRO_ROESELARE_22",
    "MRO_ROESELARE_23",
    "MRO_ROESELARE_24",
    "MRO_ROESELARE_25",
    "MRO_ROESELARE_26",
    "MRO_ROESELARE_27",
    "MRO_ROESELARE_28",
    "MRO_ROESELARE_29",
    "MRO_ROESELARE_30",
    "MRO_ROESELARE_31",
    "MRO_ROESELARE_32",
    "MRO_ROESELARE_33",
    "MRO_ROESELARE_34",
    "MRO_ROESELARE_35",
    "MRO_ROESELARE_36",
    "MRO_ROESELARE_37",
    "MRO_ROESELARE_38",
    "MRO_ROESELARE_39",
    "MRO_ROESELARE_40",
    "MRO_ROESELARE_41",
    "MRO_ROESELARE_42",
    "MRO_ROESELARE_43",
    "MRO_ROESELARE_44",
    "MRO_ROESELARE_45",
    "MRO_ROESELARE_46",
    "MRO_ROESELARE_47",
    "MRO_ROESELARE_48",
    "MRO_ROESELARE_49",
    "MRO_ROESELARE_50",
    "MRO_TIELT_01",
]


def _project_store_path() -> Path:
    """Prefer a local projects.json next to the script/exe, fallback to user home."""
    try:
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))  # type: ignore[attr-defined]
    except Exception:
        base = Path(__file__).resolve().parent
    # When frozen, _MEIPASS is temp; store next to executable instead
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent  # type: ignore[attr-defined]
    p = base / "projects.json"
    return p


def load_projects() -> list[str]:
    projects = list(DEFAULT_PROJECTS)
    p = _project_store_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        projects.append(item.strip())
    except Exception:
        # Ignore malformed config; keep defaults
        pass
    # Deduplicate while preserving order
    out: list[str] = []
    seen = set()
    for x in projects:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def save_projects(custom_projects: list[str]) -> None:
    """Save only custom additions (not the defaults)."""
    defaults = set(DEFAULT_PROJECTS)
    custom = [p for p in custom_projects if p and p not in defaults]
    p = _project_store_path()
    try:
        p.write_text(json.dumps(custom, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # If we can't write next to exe/script (permissions), fallback to user home
        try:
            home_p = Path.home() / ".ssv_zip_processor_projects.json"
            home_p.write_text(json.dumps(custom, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


# =========================
# CSV parsing
# =========================
REQUIRED_COLUMNS = ["ID", "Type", "Label", "Primary", "Secondary", "Note", "Media"]


def find_header_row(csv_path: Path) -> int:
    """Find the header row index (0-based) that contains required columns."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            row_norm = [c.strip() for c in row]
            if not row_norm:
                continue
            # check if it includes required columns
            cols = {c for c in row_norm}
            if all(c in cols for c in REQUIRED_COLUMNS):
                return idx
    raise ValueError("Could not find CSV header row with required columns.")


def load_audit_csv(csv_path: Path) -> Tuple[Dict[str, str], Dict[str, str], List[MediaRow], List[AuditRow]]:
    """Load SafetyAuditor export CSV and return meta, field-values, media rows, audit rows."""
    header_idx = find_header_row(csv_path)

    audit_rows: List[AuditRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        # skip to header
        for _ in range(header_idx):
            f.readline()

        reader = csv.DictReader(f)
        for raw in reader:
            rid = (raw.get("ID") or "").strip()
            row = AuditRow(
                row_id=rid,
                parent_id=(raw.get("Parent ID") or raw.get("ParentID") or raw.get("Parent") or "").strip(),
                row_type=(raw.get("Type") or "").strip(),
                label=(raw.get("Label") or "").strip(),
                primary=(raw.get("Primary") or "").strip(),
                secondary=(raw.get("Secondary") or "").strip(),
                note=(raw.get("Note") or "").strip(),
                media=(raw.get("Media") or "").strip(),
            )
            audit_rows.append(row)

    # Meta fields (from typical exports)
    meta: Dict[str, str] = {}
    for r in audit_rows:
        if (r.label or "").casefold() == "audit_started":
            meta["audit_started"] = r.primary or r.secondary
        if (r.label or "").casefold() == "audit_completed":
            meta["audit_completed"] = r.primary or r.secondary

    # Flatten fields by label (keep first non-empty)
    fields: Dict[str, str] = {}
    for r in audit_rows:
        t = (r.row_type or "").strip().casefold()
        if t in {"section", "media", "signature"}:
            continue
        label = (r.label or "").strip()
        if not label:
            continue

        val = ""
        if t == "address":
            val_raw = (r.secondary or r.primary or "").strip()
            val = sanitize_address(val_raw)
        elif t == "list":
            val = parse_list_value(r.primary or r.secondary or r.note)
        elif t == "datetime":
            val = fmt_epoch(r.primary) or fmt_epoch(r.secondary) or r.primary or r.secondary
        else:
            val = r.primary or r.secondary or r.note

        if val:
            fields.setdefault(label, val)

    # Media rows (type=media)
    media_rows: List[MediaRow] = []
    for r in audit_rows:
        if (r.row_type or "").strip().casefold() != "media":
            continue
        label = (r.label or "").strip() or "UNLABELED"
        media_ids = parse_media_ids(r.media, r.note)
        if media_ids:
            media_rows.append(MediaRow(label=label, media_ids=media_ids))

    return meta, fields, media_rows, audit_rows


# =========================
# Extract Materials / Work from Excel-export PDF or XLSX (preferred)
# =========================
_MAT_HEADERS = ["material code", "material description", "unit", "quantity"]
_WORK_HEADERS = ["work article code", "work article description", "unit", "quantity"]


def _norm_cell(cell: object) -> str:
    if cell is None:
        return ""
    s = str(cell).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _header_matches(row: List[str], target_headers: List[str]) -> bool:
    if not row:
        return False
    row_norm = [re.sub(r"\s+", " ", (c or "").strip()).casefold() for c in row]
    # allow extra columns; match first 4 somewhere in row order
    # common pdf extraction returns exactly 4 columns; we'll just check each target exists in row
    return all(any(th in c for c in row_norm) for th in target_headers)


def _rows_to_articles(rows: List[List[str]], expect_code: bool = True) -> List[ArticleRow]:
    out: List[ArticleRow] = []
    for r in rows:
        if not r:
            continue
        cells = [(_norm_cell(c)) for c in r]
        # make sure at least 4 cols
        while len(cells) < 4:
            cells.append("")
        code, desc, unit, qty = cells[0], cells[1], cells[2], cells[3]
        if not desc and not code:
            continue
        q = _parse_quantity(qty)
        if not q:
            # only keep filled quantities
            continue
        if expect_code and not code and desc:
            # some extracts shift; accept
            code = ""
        out.append(ArticleRow(code=code, description=desc, unit=unit, quantity=q))
    return out


# =========================
# Article extraction from CSV (materials + work articles)
# =========================

def _parse_quantity(value: str) -> Optional[float]:
    """Parse quantities like '1', '1.0', '1,5'. Returns None if empty/non-numeric."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # ignore dimension-like strings that contain 'x' between numbers
    if re.search(r"\d\s*[x×]\s*\d", s.casefold()):
        return None
    s = s.replace(" ", "")
    # replace comma decimals
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # strip trailing unit letters
    s2 = re.match(r"^[+-]?(\d+(?:\.\d+)?)", s)
    if not s2:
        return None
    try:
        q = float(s2.group(1))
        return q if q > 0 else None
    except ValueError:
        return None


def _fmt_qty(q: float) -> str:
    if q is None:
        return ""
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    s = f"{q:.6f}".rstrip("0").rstrip(".")
    return s


def _norm_label(s: str) -> str:
    # Normalize labels for robust matching across templates (e.g., with/without trailing ':')
    s2 = re.sub(r"\s+", " ", (s or "").strip())
    s2 = re.sub(r"[:：]+$", "", s2).strip()
    return s2.casefold()


def _build_tree(audit_rows: List[AuditRow]) -> Tuple[Dict[str, AuditRow], Dict[str, List[AuditRow]]]:
    by_id: Dict[str, AuditRow] = {}
    children: Dict[str, List[AuditRow]] = {}
    for r in audit_rows:
        by_id[r.row_id] = r
        pid = (r.parent_id or "").strip()
        if pid:
            children.setdefault(pid, []).append(r)
    return by_id, children


def _collect_descendants(root_ids: List[str], children: Dict[str, List[AuditRow]]) -> List[AuditRow]:
    """Depth-first in original CSV order (children lists preserve input order)."""
    out: List[AuditRow] = []
    stack: List[str] = list(reversed([rid for rid in root_ids if rid]))  # process first root first
    while stack:
        rid = stack.pop()
        for child in children.get(rid, []):
            out.append(child)
            stack.append(child.row_id)
    return out


def _find_section_roots(audit_rows: List[AuditRow], primary_phrase: str, fallback_keywords: List[str]) -> List[str]:
    """Find section-like roots by label. Prefer exact phrase; fallback to keyword matches on 'section' rows."""
    phrase = primary_phrase.casefold()
    exact = [r.row_id for r in audit_rows if _norm_label(r.label) == phrase]
    if exact:
        return exact

    roots: List[str] = []
    for r in audit_rows:
        lab = _norm_label(r.label)
        if (r.row_type or "").strip().casefold() in {"section", "category"} and any(k in lab for k in fallback_keywords):
            roots.append(r.row_id)
    return roots


def _row_value(r: AuditRow) -> str:
    return (r.primary or "").strip() or (r.secondary or "").strip()


def _extract_structured_items(
    section_desc: List[AuditRow],
    children: Dict[str, List[AuditRow]],
    code_labels: List[str],
    desc_labels: List[str],
    unit_labels: List[str],
    qty_labels: List[str],
) -> List[ArticleRow]:
    """Extract table-style rows where each item is a parent node with children fields for code/desc/unit/qty."""
    desc_set = {r.row_id for r in section_desc}
    code_set = {_norm_label(x) for x in code_labels}
    desc_set_l = {_norm_label(x) for x in desc_labels}
    unit_set = {_norm_label(x) for x in unit_labels}
    qty_set = {_norm_label(x) for x in qty_labels}

    candidate_parents: List[str] = []
    seen: Set[str] = set()
    for r in section_desc:
        if "signature" in (r.row_type or "").casefold():
            continue
        nl = _norm_label(r.label)
        if nl in code_set or nl in desc_set_l or nl in unit_set or nl in qty_set:
            pid = (r.parent_id or "").strip()
            if pid and pid not in seen:
                seen.add(pid)
                candidate_parents.append(pid)

    out: List[Tuple[int, ArticleRow]] = []
    # Keep stable ordering by first appearance in section_desc
    row_index_by_id = {r.row_id: i for i, r in enumerate(section_desc)}

    for pid in candidate_parents:
        sibs = [c for c in children.get(pid, []) if c.row_id in desc_set and "signature" not in (c.row_type or "").casefold()]
        if not sibs:
            continue

        code = ""
        desc = ""
        unit = ""
        qty: Optional[float] = None

        for s in sibs:
            nl = _norm_label(s.label)
            val = _row_value(s)
            if not val:
                continue
            if nl in code_set and not code:
                code = val
            elif nl in desc_set_l and not desc:
                desc = val
            elif nl in unit_set and not unit:
                unit = val
            elif nl in qty_set and qty is None:
                qty = _parse_quantity(val)

        if qty is None:
            continue
        if not code and not desc:
            continue

        ar = ArticleRow(code=code.strip(), description=desc.strip(), unit=unit.strip(), quantity=float(qty))
        # Order by earliest child row occurrence
        order_key = min(row_index_by_id.get(s.row_id, 10**9) for s in sibs)
        out.append((order_key, ar))

    out.sort(key=lambda t: t[0])
    return [a for _, a in out]


def _extract_simple_quantity_items(
    section_desc: List[AuditRow],
    skip_labels: Set[str],
) -> List[ArticleRow]:
    """Fallback: each row is an item; quantity is in Primary/Secondary; Label is the description."""
    out: List[ArticleRow] = []
    for r in section_desc:
        if "signature" in (r.row_type or "").casefold():
            continue
        nl = _norm_label(r.label)
        if nl in skip_labels:
            continue
        q = _parse_quantity(r.primary) or _parse_quantity(r.secondary)
        if q is None:
            continue
        desc = (r.label or "").strip()
        if not desc:
            continue
        out.append(ArticleRow(code="", description=desc, unit="", quantity=float(q)))
    return out


def extract_articles_from_csv(audit_rows: List[AuditRow]) -> Tuple[List[ArticleRow], List[ArticleRow]]:
    """Extract materials and work articles from the audit CSV tree."""
    by_id, children = _build_tree(audit_rows)

    materials_roots = _find_section_roots(
        audit_rows,
        primary_phrase="Gebruikte materialen",
        fallback_keywords=["gebruikte materialen", "materialen", "materiaal"],
    )
    work_roots = _find_section_roots(
        audit_rows,
        primary_phrase="Post Afmeldingen",
        fallback_keywords=["post afmeld", "afmeld", "werk artikel", "work article"],
    )

    mat_desc = _collect_descendants(materials_roots, children) if materials_roots else []
    work_desc = _collect_descendants(work_roots, children) if work_roots else []

    # Structured labels (adaptable; supports EN/NL)
    mat_code = ["Material code", "Materiaal code", "Materiaalcode"]
    mat_desc_l = ["Material description", "Materiaal omschrijving", "Materiaalbeschrijving", "Omschrijving"]
    mat_unit = ["Unit", "Eenheid"]
    mat_qty = ["Quantity", "Aantal", "Hoeveelheid"]

    work_code = ["Work Article Code", "Werk artikel code", "Werkartikel code"]
    work_desc_l = ["Work Article Description", "Werk artikel omschrijving", "Werkartikel omschrijving", "Omschrijving"]
    work_unit = ["Unit", "Eenheid"]
    work_qty = ["Quantity", "Aantal", "Hoeveelheid"]

    # Skip labels in fallback simple mode
    mat_skip = {_norm_label(x) for x in (mat_code + mat_desc_l + mat_unit + mat_qty)}
    work_skip = {_norm_label(x) for x in (work_code + work_desc_l + work_unit + work_qty)}

    mat_items = _extract_structured_items(mat_desc, children, mat_code, mat_desc_l, mat_unit, mat_qty) if mat_desc else []
    work_items = _extract_structured_items(work_desc, children, work_code, work_desc_l, work_unit, work_qty) if work_desc else []

    if not mat_items and mat_desc:
        mat_items = _extract_simple_quantity_items(mat_desc, mat_skip)

    if not work_items and work_desc:
        work_items = _extract_simple_quantity_items(work_desc, work_skip)

    return mat_items, work_items


def format_material_lines(mat: List[ArticleRow]) -> List[str]:
    out: List[str] = []
    for it in mat:
        q = _fmt_qty(it.quantity)
        if it.code:
            out.append(f"{q}x {it.code} {it.description}".strip())
        else:
            out.append(f"{q}x {it.description}".strip())
    return out



def format_work_lines(work: List[ArticleRow]) -> List[str]:
    out: List[str] = []
    for it in work:
        q = _fmt_qty(it.quantity)
        base = f"{q}x {it.code} {it.description}".strip() if it.code else f"{q}x {it.description}".strip()
        # In the sample, unit is shown in parentheses
        if it.unit and "(" not in base and ")" not in base:
            base = f"{base} ({it.unit})"
        out.append(base)
    return out



# =========================
# Fallback: quantities from CSV sections
# =========================
def _is_under_section(by_id: Dict[str, AuditRow], row: AuditRow, keywords: List[str]) -> bool:
    kw = [k.casefold() for k in keywords]
    cur = row
    seen: Set[str] = set()
    while cur.parent_id:
        pid = cur.parent_id
        if pid in seen:
            break
        seen.add(pid)
        parent = by_id.get(pid)
        if not parent:
            break
        if (parent.row_type or "").strip().casefold() == "section":
            lab = (parent.label or "").casefold()
            if any(k in lab for k in kw):
                return True
        cur = parent
    return False


def extract_quantity_section_items(audit_rows: List[AuditRow]) -> Tuple[List[str], List[str]]:
    """Fallback: extract bullet items for materials and work from CSV quantity rows."""
    by_id: Dict[str, AuditRow] = {r.row_id: r for r in audit_rows if r.row_id}
    materials_keywords = ["gebruikte materialen", "materialen", "materiaal"]
    post_keywords = ["post afmeldingen", "post afmelding", "afmeldingen", "afmelding"]

    materials: List[str] = []
    post: List[str] = []

    for r in audit_rows:
        t = (r.row_type or "").strip().casefold()
        if t in {"section", "media", "signature"}:
            continue

        qty = _parse_quantity(r.primary) or _parse_quantity(r.secondary)
        if not qty:
            continue

        label = (r.label or "").strip()
        if not label:
            continue

        if _is_under_section(by_id, r, materials_keywords):
            materials.append(f"{_fmt_qty(qty)}x {label}")
        elif _is_under_section(by_id, r, post_keywords):
            post.append(f"{_fmt_qty(qty)}x {label}")

    return materials, post


# =========================
# PDF rendering (ReportLab)
# =========================
def _wrap_lines(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    # simple, deterministic wrap
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _draw_section_title(c: Canvas, x: float, y: float, title: str) -> float:
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(0.75, 0.0, 0.0)  # red-ish
    c.drawString(x, y, title)
    c.setFillColorRGB(0, 0, 0)
    return y - 16


def _draw_kv_lines(c: Canvas, x_label: float, x_val: float, y: float, items: List[Tuple[str, str]]) -> float:
    c.setFont("Helvetica", 10)
    for k, v in items:
        c.drawString(x_label, y, f"{k}:")
        c.drawString(x_val, y, v or "")
        y -= 14
    return y


def render_page1(c: Canvas, report: ReportRow) -> None:
    """
    Render the first report page to match the reference PDF layout (A4).

    Notes:
    - Uses fixed coordinates derived from the reference PDF.
    - Designed to match visually; if lists grow beyond the reserved space, they will be clipped.
      (Photos always start on the next page.)
    """
    w, h = A4

    # Reference-derived coordinates (points)
    X_LEFT = 56.6929
    X_RIGHT = 538.5827

    # Colors from reference PDF vectors
    RED = (0.702, 0.0, 0.0)
    GOLD = (0.549, 0.427, 0.0)
    GREEN_BAR = (0.83, 0.93, 0.85)
    YES_GREEN = (0.2, 0.6, 0.2)

    LINE_W = 1.4173

    # Header banner
    banner_y0 = 771.0236
    banner_h = 42.5197
    c.setFillColorRGB(*RED)
    c.rect(0, banner_y0, w, banner_h, stroke=0, fill=1)

    # Title (black) + datetime (white, right-aligned)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(X_LEFT, 784.72, "WERKLOGGER RAPPORT")

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(X_RIGHT, 791.63, report.report_datetime)

    # Helper: section title + underline
    def draw_section(title: str, y_title: float, line_y: float, color: Tuple[float, float, float]) -> None:
        c.setFillColorRGB(*color)
        c.setFont("Helvetica", 14)
        c.drawString(X_LEFT, y_title, title)
        c.setStrokeColorRGB(*color)
        c.setLineWidth(LINE_W)
        c.line(X_LEFT, line_y, X_RIGHT, line_y)
        c.setFillColorRGB(0, 0, 0)

    # ======================
    # Adresgegevens
    # ======================
    draw_section("Adresgegevens:", y_title=739.78, line_y=731.34, color=RED)

    addr_x_val = 198.4252
    y = 715.0956
    dy = 17.0079

    addr_items = [
        ("Naam Onderaannemer:", report.naam_onderaannemer),
        ("Project/Locatie Naam:", report.project_locatie_naam),
        ("Building ID:", report.building_id),
        ("Adres:", report.adres),
        ("Postcode + Stad:", report.postcode_stad),
        ("Contactpersoon:", report.contactpersoon),
        ("Quadrant:", report.quadrant),
    ]

    for k, v in addr_items:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(X_LEFT, y, k)
        c.setFont("Helvetica", 10)
        c.drawString(addr_x_val, y, v or "")
        y -= dy

    # ======================
    # LMRA Checklist
    # ======================
    draw_section("LMRA Checklist:", y_title=581.04, line_y=572.60, color=RED)

    # Status bar
    bar_y0 = 538.5827
    bar_h = 19.8425
    c.setFillColorRGB(*GREEN_BAR)
    c.rect(X_LEFT, bar_y0, X_RIGHT - X_LEFT, bar_h, stroke=0, fill=1)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    c.drawCentredString((X_LEFT + X_RIGHT) / 2.0, bar_y0 + 5.0, "LMRA Status: OK - Werk kan worden uitgevoerd")

    # Checklist rows
    checklist_x_yes = 497.5433
    checklist_items = [
        "Veiligheidsrisico's geïdentificeerd",
        "Juiste PBM aanwezig",
        "Werknemers geïnformeerd",
        "Noodprocedures bekend",
        "Werkgebied afgezet",
        "Vergunningen aanwezig",
    ]
    y = 522.3386
    for item in checklist_items:
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(X_LEFT, y, item)

        c.setFillColorRGB(*YES_GREEN)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(checklist_x_yes, y, "JA")
        y -= dy

    c.setFillColorRGB(0, 0, 0)

    # ======================
    # Uitgevoerde Werken - Details
    # ======================
    draw_section("Uitgevoerde Werken - Details:", y_title=405.29, line_y=396.85, color=RED)

    details_x_val = 240.9449
    y = 380.6142
    details_items = [
        ("Gekoppelde kleur duct:", report.duct_kleur),
        ("Hoeveel units gelast:", report.units_gelast),
    ]
    for k, v in details_items:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(X_LEFT, y, k)
        c.setFont("Helvetica", 10)
        c.drawString(details_x_val, y, v or "")
        y -= dy
    # ======================
    # Gebruikte materialen + Post Afmeldingen (paginated)
    # ======================

    bullet_x = X_LEFT
    text_x = 62.97
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 0, 0)

    TITLE_TO_LINE = 8.45
    TITLE_TO_TEXT = 24.6898
    BOTTOM_Y = 70.0
    GAP_AFTER_SECTION = 14.0

    def draw_header_only() -> None:
        # Header banner
        c.setFillColorRGB(*RED)
        c.rect(0, banner_y0, w, banner_h, stroke=0, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(X_LEFT, 784.72, "WERKLOGGER RAPPORT")
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(X_RIGHT, 791.63, report.report_datetime)
        c.setFillColorRGB(0, 0, 0)

    def draw_bullet_lines(lines: list[str], y_start: float) -> tuple[list[str], float]:
        """Draw bullets until we hit BOTTOM_Y. Keeps each bullet together (no split mid-bullet)."""
        y = y_start
        remaining: list[str] = []
        for i, line in enumerate(lines):
            wrapped = _wrap_lines(line, 110) or ["-"]
            needed = max(1, len(wrapped))
            if y - dy * (needed - 1) < BOTTOM_Y:
                remaining = lines[i:]
                break
            # bullet line
            c.drawString(bullet_x, y, u"\u2022")
            c.drawString(text_x, y, wrapped[0])
            y -= dy
            for cont in wrapped[1:]:
                c.drawString(text_x, y, cont)
                y -= dy
        return remaining, y

    # Render materials across pages if needed
    mat_lines = list(report.gebruikte_materialen_lines or ["-"])
    work_lines = list(report.post_afmeldingen_lines or ["-"])

    on_first_page = True
    mat_title = "Gebruikte materialen:"
    mat_y_title = 334.43
    mat_y_text = 309.7402

    while True:
        draw_section(mat_title, y_title=mat_y_title, line_y=mat_y_title - TITLE_TO_LINE, color=GOLD)
        c.setFont("Helvetica", 10)
        mat_remaining, y_after_mat = draw_bullet_lines(mat_lines, mat_y_text)
        if not mat_remaining:
            break
        # continue on a new page
        c.showPage()
        on_first_page = False
        draw_header_only()
        mat_title = "Gebruikte materialen (vervolg):"
        mat_y_title = 760.0
        mat_y_text = mat_y_title - TITLE_TO_TEXT
        mat_lines = mat_remaining

    # Render work/post sections; start below materials if needed
    post_title = "Post Afmeldingen:"
    fixed_post_y_title = 244.09

    def ensure_space_for_section(y_title: float) -> bool:
        # True if we can draw at least one bullet line on this page
        return (y_title - TITLE_TO_TEXT) >= (BOTTOM_Y + dy)

    # Determine starting title position
    if on_first_page and y_after_mat > (fixed_post_y_title + 30.0):
        post_y_title = fixed_post_y_title
    else:
        post_y_title = y_after_mat - GAP_AFTER_SECTION

    if not ensure_space_for_section(post_y_title):
        c.showPage()
        on_first_page = False
        draw_header_only()
        post_y_title = 725.0

    post_y_text = post_y_title - TITLE_TO_TEXT
    post_lines = work_lines

    while True:
        draw_section(post_title, y_title=post_y_title, line_y=post_y_title - TITLE_TO_LINE, color=GOLD)
        c.setFont("Helvetica", 10)
        post_remaining, y_after_post = draw_bullet_lines(post_lines, post_y_text)
        if not post_remaining:
            break
        c.showPage()
        on_first_page = False
        draw_header_only()
        post_title = "Post Afmeldingen (vervolg):"
        post_y_title = 725.0
        post_y_text = post_y_title - TITLE_TO_TEXT
        post_lines = post_remaining



def _fit_image_in_box(im: Image.Image, box_w: float, box_h: float) -> Image.Image:
    """Return resized image that fits box while preserving aspect ratio."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    iw, ih = im.size
    if iw <= 0 or ih <= 0:
        return im
    scale = min(box_w / iw, box_h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    return im.resize((nw, nh), Image.LANCZOS)


def render_photos_pages(c: Canvas, report: ReportRow) -> None:
    """Render photo pages matching the reference layout (2 columns, 3 rows per page)."""
    w, h = A4

    X_LEFT = 56.6929
    X_RIGHT = 538.5827
    RED = (0.702, 0.0, 0.0)
    LINE_W = 1.4173

    # Heading positions from reference
    heading_y = 782.30
    underline_y = 773.8585

    # Image boxes from reference (points)
    col1_x0, col1_x1 = 56.6929, 283.4646
    col2_x0, col2_x1 = 311.8110, 538.5827
    box_w = col1_x1 - col1_x0
    box_h = 170.0787

    row_top_start = 745.5118
    row_step = 198.4252  # box_h + vertical gap (~28.35)
    label_offset = 12.1102  # label baseline above first row images

    def draw_page_heading(first: bool) -> None:
        c.setFillColorRGB(*RED)
        c.setFont("Helvetica", 14)
        c.drawString(X_LEFT, heading_y, "Foto's:" if first else "Foto's (vervolg):")
        c.setStrokeColorRGB(*RED)
        c.setLineWidth(LINE_W)
        c.line(X_LEFT, underline_y, X_RIGHT, underline_y)
        c.setFillColorRGB(0, 0, 0)

    # Group photos by label (preserve original order)
    groups: List[Tuple[str, List[Photo]]] = []
    seen_order: List[str] = []
    by_label: Dict[str, List[Photo]] = {}
    for p in (report.photos or []):
        lbl = p.label or "UNLABELED"
        if lbl not in by_label:
            by_label[lbl] = []
            seen_order.append(lbl)
        by_label[lbl].append(p)
    for lbl in seen_order:
        groups.append((lbl, by_label[lbl]))

    first_page = True
    draw_page_heading(first_page)
    first_page = False

    row_index = 0  # 0..2 (3 rows)

    def ensure_row_available() -> None:
        nonlocal row_index, first_page
        if row_index >= 3:
            c.showPage()
            draw_page_heading(False)
            row_index = 0

    for label, photos in groups:
        if not photos:
            continue

        # We show the label once per group (repeat if the group continues on a new page).
        label_printed_on_this_page = False
        remaining = list(photos)

        while remaining:
            ensure_row_available()

            row_top = row_top_start - (row_index * row_step)
            row_bottom = row_top - box_h
            label_y = row_top + label_offset

            # Print label once at the first row for the group on this page.
            if not label_printed_on_this_page:
                c.setFont("Helvetica-Bold", 10)
                c.setFillColorRGB(0, 0, 0)
                c.drawString(X_LEFT, label_y, f"{label} ({len(photos)})")
                label_printed_on_this_page = True

            left_photo = remaining.pop(0) if remaining else None
            right_photo = remaining.pop(0) if remaining else None

            def draw_photo(photo: Optional[Photo], x0: float, x1: float) -> None:
                if not photo or not photo.image_path or not photo.image_path.exists():
                    return
                try:
                    with Image.open(photo.image_path) as im:
                        im_fit = _fit_image_in_box(im, box_w, box_h)
                        bio = io.BytesIO()
                        im_fit.save(bio, format="JPEG", quality=90)
                        bio.seek(0)
                        iw, ih = im_fit.size
                        # Convert pixels to points 1:1 approximation; we sized by points anyway
                        dx = x0 + (box_w - iw) / 2.0
                        dy = row_bottom + (box_h - ih) / 2.0
                        c.drawImage(ImageReader(bio), dx, dy, width=iw, height=ih, preserveAspectRatio=True, mask='auto')
                except Exception:
                    return

            draw_photo(left_photo, col1_x0, col1_x1)
            draw_photo(right_photo, col2_x0, col2_x1)

            row_index += 1

            # If the group continues onto a new page, allow label to repeat.
            if row_index >= 3 and remaining:
                label_printed_on_this_page = False



def create_output_zip(out_zip_path: Path, files: List[Tuple[Path, str]]) -> None:
    """Create a zip at out_zip_path containing the given files (src_path, arcname)."""
    if out_zip_path.exists():
        out_zip_path.unlink()
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src, arcname in files:
            z.write(src, arcname)



def build_pdf(out_pdf_path: Path, report: ReportRow) -> None:
    c = Canvas(str(out_pdf_path), pagesize=A4)
    render_page1(c, report)
    c.showPage()
    render_photos_pages(c, report)
    c.save()


# =========================
# Image processing + report building
# =========================
def process_zip_to_folder_and_pdf(zip_path: Path, out_dir: Path, project_override: Optional[str] = None, log: Optional[callable] = None) -> ProcessResult:
    def _log(msg: str) -> None:
        if log:
            log(msg)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    _log("Extracting ZIP...")
    with tempfile.TemporaryDirectory(prefix="ssv_zip_") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        # Find CSV
        csv_files = list(tmp.rglob("*.csv"))
        if len(csv_files) != 1:
            raise ValueError(f"Expected exactly 1 CSV in ZIP, found {len(csv_files)}.")
        csv_path = csv_files[0]
        _log(f"Found CSV: {csv_path.name}")

        meta, fields, media_rows, audit_rows = load_audit_csv(csv_path)

        # Extract materials/work tables (preferred)
        mat_articles, work_articles = extract_articles_from_csv(audit_rows)
        gebruikte_materialen = format_material_lines(mat_articles)
        post_afmeldingen = format_work_lines(work_articles)

        if not gebruikte_materialen and not post_afmeldingen:
            _log("No structured materials/work sections found; falling back to simple CSV quantity rows.")
            gebruikte_materialen, post_afmeldingen = extract_quantity_section_items(audit_rows)

        # Map core fields
        dt_str = fmt_epoch(meta.get("audit_completed", "")) or fmt_epoch(meta.get("audit_started", "")) or fields.get("Date and time of approval", "")
        if not dt_str:
            dt_str = dt.datetime.now().strftime("%d/%m/%Y %H:%M")

        report = ReportRow(
            report_datetime=dt_str,
            # Always force subcontractor name (requested)
            naam_onderaannemer="F.A.S.T. Support BV.",
            project_locatie_naam=(project_override.strip() if project_override and project_override.strip() else (fields.get("Project/Locatie Naam", "").strip() or "Niet ingevuld")),
            building_id=fields.get("Building ID", "").strip() or "Niet ingevuld",
            adres=sanitize_address(fields.get("Adres", "").strip()) or "Niet ingevuld",
            postcode_stad=(fields.get("Postcode + Stad", "") or "").replace("+", " ").strip() or "Niet ingevuld",
            contactpersoon=fields.get("Contactpersoon", "").strip() or "Niet ingevuld",
            quadrant=fields.get("Quadrant", "").strip() or "",
            duct_kleur=fields.get("Gekoppelde kleur subduct", "").strip() or fields.get("Gekoppelde kleur duct", "").strip() or "",
            units_gelast=fields.get("Hoeveel units gelast?", "").strip() or fields.get("Hoeveel units gelast", "").strip() or "",
            gebruikte_materialen_lines=gebruikte_materialen,
            post_afmeldingen_lines=post_afmeldingen,
            photos=[],
        )

        # Build an index for images (case-insensitive)
        all_files = list(tmp.rglob("*"))
        img_by_stem: Dict[str, Path] = {}
        for p in all_files:
            if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}:
                img_by_stem[p.stem.lower()] = p
        nonimg_stems: Set[str] = {p.stem.lower() for p in all_files if p.is_file() and p.suffix.lower() not in {'.jpeg','.jpg'}}

        # Copy any PDF attachments from the input ZIP to output folder (unchanged)
        pdf_attachments: List[Path] = []
        for p in all_files:
            if p.is_file() and p.suffix.lower() == ".pdf":
                dest = out_dir / p.name
                try:
                    shutil.copy2(p, dest)
                    pdf_attachments.append(dest)
                except Exception:
                    pass
        if pdf_attachments:
            _log(f"Copied {len(pdf_attachments)} PDF attachment(s).")


        # Process images
        written = 0
        used_names: Set[str] = set()
        processed_photos: List[Photo] = []

        _log("Processing images...")
        for mrow in media_rows:
            base_label = normalize_label(mrow.label)
            for idx, mid in enumerate(mrow.media_ids, start=1):
                src = img_by_stem.get(mid.lower())
                if not src:
                    # If a non-image file shares this stem (e.g. a PDF attachment), silently skip.
                    if mid.lower() in nonimg_stems:
                        continue
                    _log(f"WARNING: image not found for media id: {mid}")
                    continue

                out_base = base_label if len(mrow.media_ids) == 1 else f"{base_label}_{idx}"
                out_name = f"{out_base}.jpeg"
                if out_name.lower() in used_names:
                    n = 2
                    while f"{out_base}_{n}.jpeg".lower() in used_names:
                        n += 1
                    out_name = f"{out_base}_{n}.jpeg"
                used_names.add(out_name.lower())

                out_path = out_dir / out_name
                with Image.open(src) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    data = compress_jpeg_to_limit(im, MAX_BYTES)
                out_path.write_bytes(data)
                written += 1
                processed_photos.append(Photo(label=mrow.label, image_path=out_path))

        report.photos = processed_photos

        # Output PDF name
        fn = f"{safe_filename(report.building_id)}-{safe_filename(report.project_locatie_naam)}-{safe_filename(report.report_datetime)}-RAPPORT.pdf"
        pdf_path = out_dir / fn

        _log("Generating PDF...")
        build_pdf(pdf_path, report)
        _log(f"Saved PDF: {pdf_path.name}")

        # Create output ZIP containing: generated report PDF, processed JPEGs, and any input PDF attachments.
        out_zip_name = f"{safe_filename(report.building_id)}-{safe_filename(report.project_locatie_naam)}-{safe_filename(report.report_datetime)}-OUTPUT.zip"
        out_zip_path = out_dir / out_zip_name

        zip_files: List[Tuple[Path, str]] = []
        zip_files.append((pdf_path, pdf_path.name))
        for ph in processed_photos:
            zip_files.append((ph.image_path, ph.image_path.name))
        for p in pdf_attachments:
            zip_files.append((p, p.name))

        _log("Creating output ZIP...")
        create_output_zip(out_zip_path, zip_files)
        _log(f"Saved ZIP: {out_zip_path.name}")

        return ProcessResult(report=report, pdf_path=pdf_path, written_images=written, output_zip_path=out_zip_path, copied_pdfs=len(pdf_attachments))


# =========================
# GUI
# =========================
if tk is not None:
    class App(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("SSV ZIP Processor (Images + PDF)")
            self.geometry("980x640")
    
            self.zip_path: Optional[Path] = None
            self.out_dir: Optional[Path] = None
    
            frm = tk.Frame(self)
            frm.pack(fill="x", padx=10, pady=10)
    
            self.lbl_zip = tk.Label(frm, text="Input ZIP: (none)", anchor="w")
            self.lbl_zip.pack(fill="x")
    
            self.lbl_out = tk.Label(frm, text="Output folder: (none)", anchor="w")
            self.lbl_out.pack(fill="x", pady=(6, 0))


            # Project selection (overrides CSV "Project/Locatie Naam" when set)
            proj_row = tk.Frame(frm)
            proj_row.pack(fill="x", pady=(10, 0))
            tk.Label(proj_row, text="Project/Locatie Naam:", width=22, anchor="w").pack(side="left")

            self.projects = load_projects()
            self.project_var = tk.StringVar(value="")
            self.cmb_project = ttk.Combobox(
                proj_row,
                textvariable=self.project_var,
                values=([''] + self.projects),
                state="readonly",
                width=32,
            )
            self.cmb_project.pack(side="left", padx=(0, 10))

            self.new_project_var = tk.StringVar()
            tk.Entry(proj_row, textvariable=self.new_project_var, width=26).pack(side="left")
            tk.Button(proj_row, text="Add project", command=self.add_project).pack(side="left", padx=(8, 0))
    
            btns = tk.Frame(frm)
            btns.pack(fill="x", pady=(10, 0))
    
            tk.Button(btns, text="Select input ZIP...", command=self.pick_zip, width=20).pack(side="left")
            tk.Button(btns, text="Select output folder...", command=self.pick_out, width=22).pack(side="left", padx=(10, 0))
    
            self.btn_run = tk.Button(self, text="Process ZIP (images + PDF)", command=self.run, height=2, font=("Segoe UI", 12, "bold"))
            self.btn_run.pack(fill="x", padx=10, pady=12)
    
            self.logbox = scrolledtext.ScrolledText(self, height=22, font=("Consolas", 10))
            self.logbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.log("Ready.")
    
        def log(self, msg: str) -> None:
            self.logbox.insert("end", msg + "\n")
            self.logbox.see("end")
            self.update_idletasks()
    

        def add_project(self) -> None:
            name = (self.new_project_var.get() or "").strip()
            if not name:
                return
            # Normalize similar to filename rules (keeps your MRO_* format)
            name = name.replace(" ", "_")
            name = re.sub(r'[<>:"/\\|?*]+', "_", name)
            name = re.sub(r"_+", "_", name).strip("._ ")
            if not name:
                return
            if name not in self.projects:
                self.projects.append(name)
                self.cmb_project["values"] = ([""] + self.projects)
                save_projects(self.projects)
            self.project_var.set(name)
            self.new_project_var.set("")
            self.log(f"Project added/selected: {name}")

        def pick_zip(self) -> None:
            path = filedialog.askopenfilename(title="Select input ZIP", filetypes=[("ZIP files", "*.zip")])
            if not path:
                return
            self.zip_path = Path(path)
            self.lbl_zip.config(text=f"Input ZIP: {self.zip_path}")
            self.log(f"Selected ZIP: {self.zip_path}")
    
        def pick_out(self) -> None:
            path = filedialog.askdirectory(title="Select output folder")
            if not path:
                return
            self.out_dir = Path(path)
            self.lbl_out.config(text=f"Output folder: {self.out_dir}")
            self.log(f"Selected output folder: {self.out_dir}")
    
        def run(self) -> None:
            if not self.zip_path:
                messagebox.showerror("Error", "Please select an input ZIP first.")
                return
            if not self.out_dir:
                messagebox.showerror("Error", "Please select an output folder first.")
                return
    
            self.btn_run.config(state="disabled")
            try:
                self.log("Starting processing...")
                res = process_zip_to_folder_and_pdf(self.zip_path, self.out_dir, project_override=self.project_var.get(), log=self.log)
                self.log(f"Done. Images written: {res.written_images}")
                messagebox.showinfo("Success", f"Finished.\nImages: {res.written_images}\nPDF: {res.pdf_path.name}")
            except Exception as e:
                self.log(f"ERROR: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self.btn_run.config(state="normal")
    
    

# =========================
# CLI / Entry point (module-level)
# =========================
def main() -> None:
    ap = argparse.ArgumentParser(description="Process a SafetyAuditor ZIP export to images + PDF.")
    ap.add_argument("--zip", dest="zip_path", help="Path to input ZIP")
    ap.add_argument("--out", dest="out_dir", help="Output folder path")
    ap.add_argument("--project", dest="project", help="Override Project/Locatie Naam")
    args = ap.parse_args()

    if args.zip_path and args.out_dir:
        res = process_zip_to_folder_and_pdf(Path(args.zip_path), Path(args.out_dir), project_override=args.project, log=print)
        print(f"PDF: {res.pdf_path}")
        return

    # default to GUI
    if tk is None:
        raise SystemExit(
            "Tkinter is not available in this Python environment. "
            "Run with --zip and --out for CLI mode, or install Python with Tk support."
        )
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
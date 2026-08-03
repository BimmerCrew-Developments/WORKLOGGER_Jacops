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
  2) rename all referenced JPEGs into the chosen output folder (original quality preserved)
  3) generate a "WERKLOGGER RAPPORT" style PDF in the same output folder, matching the provided
     reference layout (2 photos side-by-side per row; no approver signature).

Key rules
---------
- Image naming comes from CSV Label, normalized to OS-safe filenames
- Media IDs in CSV are semicolon-separated; extension is .jpeg
- Images are kept at original quality (no recompression)
- PDF sections "Gebruikte materialen" and "Post Afmeldingen" are populated from quantity rows in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

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


@dataclass(frozen=True)
class ExportField:
    """A value that templates are allowed to read from :class:`ReportRow`."""

    display_name: str
    getter: Callable[[ReportRow], Any]


# This is the sole interface between untrusted presentation templates and imported
# report data.  Keys are stable; Python attribute paths and expressions are never
# accepted from a template file.
EXPORT_FIELD_REGISTRY: Mapping[str, ExportField] = {
    "report_datetime": ExportField("Report date and time", lambda row: row.report_datetime),
    "subcontractor": ExportField("Subcontractor name", lambda row: row.naam_onderaannemer),
    "project_name": ExportField("Project/location name", lambda row: row.project_locatie_naam),
    "building_id": ExportField("Building ID", lambda row: row.building_id),
    "address": ExportField("Address", lambda row: row.adres),
    "postal_city": ExportField("Postal code and city", lambda row: row.postcode_stad),
    "contact": ExportField("Contact person", lambda row: row.contactpersoon),
    "quadrant": ExportField("Quadrant", lambda row: row.quadrant),
    "duct_color": ExportField("Connected duct color", lambda row: row.duct_kleur),
    "units_welded": ExportField("Units welded", lambda row: row.units_gelast),
    "materials": ExportField("Materials used", lambda row: row.gebruikte_materialen_lines),
    "post_registrations": ExportField("Post registrations", lambda row: row.post_afmeldingen_lines),
    "photos": ExportField("Photos", lambda row: row.photos),
}

SECTION_FIELD_KEYS: Mapping[str, Tuple[str, ...]] = {
    "address": ("subcontractor", "project_name", "building_id", "address", "postal_city", "contact", "quadrant"),
    "lmra": (),
    "work_details": ("duct_color", "units_welded"),
    "materials": ("materials",),
    "post_registrations": ("post_registrations",),
    "photos": ("photos",),
}


@dataclass(frozen=True)
class ExportTemplate:
    """Declarative PDF export layout.

    ``template_id`` is stable and suitable for persisted preferences.  The mapping
    fields deliberately keep the model forwards-compatible: an installation can
    add a layout value without changing CSV ingestion or the report data model.
    """

    template_id: str
    display_name: str
    page_settings: Mapping[str, Any]
    branding: Mapping[str, Any]
    colors: Mapping[str, Tuple[float, float, float]]
    enabled_sections: Tuple[str, ...]
    section_order: Tuple[str, ...]
    section_fields: Mapping[str, Tuple[str, ...]]
    field_labels: Mapping[str, str]
    photo_grid: Mapping[str, Any]
    output_filename_pattern: str
    include_photo_pages: bool = True
    include_loose_images: bool = True
    include_pdf_attachments: bool = True
    create_output_zip: bool = True
    empty_value_fallback: str = "Niet ingevuld"
    layout: Mapping[str, Any] = field(default_factory=dict)
    # The uploaded CSV is used as a schema sample only; no customer rows are
    # persisted.  These mappings let otherwise identical exports rename their
    # structural columns and feed PDF fields directly from a chosen column.
    csv_headers: Tuple[str, ...] = ()
    csv_column_mapping: Mapping[str, str] = field(default_factory=dict)
    pdf_field_columns: Mapping[str, str] = field(default_factory=dict)


# All report copy which is not sourced from the CSV lives here.  Keeping these
# keys together makes old template documents straightforward to migrate.
DEFAULT_TEMPLATE_BRANDING: Mapping[str, str] = {
    "report_title": "WERKLOGGER RAPPORT",
    "section_address": "Adresgegevens:",
    "section_lmra": "LMRA Checklist:",
    "section_work_details": "Uitgevoerde Werken - Details:",
    "section_materials": "Gebruikte materialen:",
    "section_post_registrations": "Post Afmeldingen:",
    "section_continuation_pattern": "{title} (vervolg):",
    "lmra_status": "LMRA Status: OK - Werk kan worden uitgevoerd",
    "lmra_item_1": "Veiligheidsrisico's geïdentificeerd",
    "lmra_item_2": "Juiste PBM aanwezig",
    "lmra_item_3": "Werknemers geïnformeerd",
    "lmra_item_4": "Noodprocedures bekend",
    "lmra_item_5": "Werkgebied afgezet",
    "lmra_item_6": "Vergunningen aanwezig",
    "yes_label": "JA",
    "no_label": "NEE",
    "photo_title": "Foto's:",
    "photo_continuation_title": "Foto's (vervolg):",
}
REQUIRED_BRANDING_KEYS = frozenset(DEFAULT_TEMPLATE_BRANDING)


# This is the compatibility template.  Its values are the measurements and copy
# used by the original WERKLOGGER renderer.
WERKLOGGER_EXPORT_TEMPLATE = ExportTemplate(
    template_id="werklogger-report-v1",
    display_name="WERKLOGGER RAPPORT",
    page_settings={"pagesize": A4, "bottom_y": 70.0},
    branding=DEFAULT_TEMPLATE_BRANDING,
    colors={
        "primary": (0.702, 0.0, 0.0), "secondary": (0.549, 0.427, 0.0),
        "status_bar": (0.83, 0.93, 0.85), "status_yes": (0.2, 0.6, 0.2),
    },
    enabled_sections=("address", "lmra", "work_details", "materials", "post_registrations", "photos"),
    section_order=("address", "lmra", "work_details", "materials", "post_registrations", "photos"),
    section_fields=SECTION_FIELD_KEYS,
    field_labels={
        "subcontractor": "Naam Onderaannemer:",
        "project_name": "Project/Locatie Naam:", "building_id": "Building ID:",
        "address": "Adres:", "postal_city": "Postcode + Stad:",
        "contact": "Contactpersoon:", "quadrant": "Quadrant:",
        "duct_color": "Gekoppelde kleur duct:", "units_welded": "Hoeveel units gelast:",
    },
    photo_grid={
        "columns": 2, "rows": 3, "left": 56.6929, "right": 538.5827,
        "column_gap": 28.3464, "top_margin": 56.6929,
        "heading_area": 39.685, "label_allowance": 18.0, "row_gap": 10.0,
    },
    output_filename_pattern="{building_id}-{project_name}-{report_datetime}-RAPPORT.pdf",
    include_photo_pages=True,
    include_loose_images=True,
    include_pdf_attachments=True,
    create_output_zip=True,
    layout={
        "left": 56.6929, "right": 538.5827, "line_width": 1.4173,
        "banner_y": 771.0236, "banner_height": 42.5197, "title_y": 784.72,
        "datetime_y": 791.63, "address_value_x": 198.4252,
        "details_value_x": 240.9449, "checklist_yes_x": 497.5433,
        "line_spacing": 17.0079,
    },
)

TEMPLATE_SECTIONS = (
    ("address", "Address"), ("lmra", "LMRA checklist"),
    ("work_details", "Work details"), ("materials", "Materials"),
    ("post_registrations", "Post registrations"), ("photos", "Photos"),
)
SECTION_BRANDING_KEYS = {
    "address": "section_address", "lmra": "section_lmra",
    "work_details": "section_work_details", "materials": "section_materials",
    "post_registrations": "section_post_registrations",
}
FILENAME_PLACEHOLDERS = frozenset({"building_id", "project_name", "report_datetime"})
CSV_COLUMN_ROLES = ("ID", "Parent ID", "Type", "Label", "Primary", "Secondary", "Note", "Media")
PDF_COLUMN_FIELDS = (
    "report_datetime", "subcontractor", "project_name", "building_id", "address",
    "postal_city", "contact", "quadrant", "duct_color", "units_welded",
)


EXPORT_TEMPLATE_CONFIG_VERSION = 2


class ExportTemplateConfigurationError(RuntimeError):
    """An export-template configuration could not be safely read or written."""

    def __init__(self, message: str, recovered_templates: Optional[List[ExportTemplate]] = None) -> None:
        super().__init__(message)
        self.recovered_templates = recovered_templates


# Photo cells smaller than one inch are not useful in a printed work report.
# The explicit count limits also keep accidental values from producing enormous
# documents, even on an unusually large custom page size.
MIN_PHOTO_WIDTH = 72.0
MIN_PHOTO_HEIGHT = 72.0
MAX_PHOTO_COLUMNS = 5
MAX_PHOTO_ROWS = 6


@dataclass(frozen=True)
class PhotoGridGeometry:
    """Fully resolved photo geometry, in PDF points."""

    column_boxes: Tuple[Tuple[float, float], ...]
    row_tops: Tuple[float, ...]
    box_width: float
    box_height: float
    row_step: float
    heading_y: float
    underline_y: float


def calculate_photo_grid_geometry(page_settings: Mapping[str, Any],
                                  photo_grid: Mapping[str, Any]) -> PhotoGridGeometry:
    """Derive photo cells from the printable region instead of fixed A4 rows."""
    page_width, page_height = (float(value) for value in page_settings["pagesize"])
    left = float(photo_grid["left"])
    right = float(photo_grid["right"])
    bottom = float(page_settings["bottom_y"])
    top_margin = float(photo_grid.get("top_margin", left))
    heading_area = float(photo_grid.get("heading_area", 39.685))
    label_allowance = float(photo_grid.get("label_allowance", 18.0))
    row_gap = float(photo_grid.get("row_gap", 10.0))
    column_gap = float(photo_grid["column_gap"])
    rows, columns = int(photo_grid["rows"]), int(photo_grid["columns"])

    if not 1 <= rows <= MAX_PHOTO_ROWS or not 1 <= columns <= MAX_PHOTO_COLUMNS:
        raise ValueError(
            f"Photo grid supports 1-{MAX_PHOTO_ROWS} rows and "
            f"1-{MAX_PHOTO_COLUMNS} columns."
        )
    printable_top = page_height - top_margin
    images_top = printable_top - heading_area - label_allowance
    available_image_height = images_top - bottom - row_gap * (rows - 1) - label_allowance * (rows - 1)
    box_height = available_image_height / rows
    box_width = (right - left - column_gap * (columns - 1)) / columns
    if left < 0 or right > page_width or right <= left or bottom < 0 or printable_top > page_height:
        raise ValueError("Photo margins must define a printable region inside the selected page size.")
    if heading_area < 0 or label_allowance < 0 or row_gap < 0 or column_gap < 0:
        raise ValueError("Photo heading, label, and gap allowances cannot be negative.")
    if box_width < MIN_PHOTO_WIDTH or box_height < MIN_PHOTO_HEIGHT:
        raise ValueError(
            "Photo grid does not fit the printable page: each image must be at least "
            f"{MIN_PHOTO_WIDTH / 72:g} x {MIN_PHOTO_HEIGHT / 72:g} inch including label space."
        )
    row_step = box_height + label_allowance + row_gap
    row_tops = tuple(images_top - index * row_step for index in range(rows))
    column_boxes = tuple(
        (left + index * (box_width + column_gap), left + index * (box_width + column_gap) + box_width)
        for index in range(columns)
    )
    return PhotoGridGeometry(
        column_boxes, row_tops, box_width, box_height, row_step,
        printable_top - 14.0, printable_top - heading_area + 8.0,
    )


def _adjacent_export_template_path() -> Path:
    """Return the location used by releases that stored data beside the app."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    return base / "export_templates.json"


def export_template_config_path() -> Path:
    """Use the same platform-aware per-user configuration root as projects."""
    return project_config_path().with_name("export_templates.json")


def export_template_to_dict(template: ExportTemplate) -> Dict[str, Any]:
    """Convert a template to its portable JSON representation."""
    return {
        "template_id": template.template_id, "display_name": template.display_name,
        "page_settings": dict(template.page_settings), "branding": dict(template.branding),
        "colors": {key: list(value) for key, value in template.colors.items()},
        "enabled_sections": list(template.enabled_sections), "section_order": list(template.section_order),
        "section_fields": {key: list(value) for key, value in template.section_fields.items()},
        "field_labels": dict(template.field_labels), "photo_grid": dict(template.photo_grid),
        "output_filename_pattern": template.output_filename_pattern,
        "include_photo_pages": template.include_photo_pages,
        "include_loose_images": template.include_loose_images,
        "include_pdf_attachments": template.include_pdf_attachments,
        "create_output_zip": template.create_output_zip,
        "empty_value_fallback": template.empty_value_fallback, "layout": dict(template.layout),
        "csv_headers": list(template.csv_headers),
        "csv_column_mapping": dict(template.csv_column_mapping),
        "pdf_field_columns": dict(template.pdf_field_columns),
    }


def export_template_from_dict(data: Mapping[str, Any]) -> ExportTemplate:
    """Build and validate a custom template from untrusted JSON data."""
    # Templates written by earlier releases only have report_title/photo_title.
    # Defaults are intentionally applied only to absent keys: explicitly blank
    # required copy remains a validation error rather than being silently fixed.
    branding = {**DEFAULT_TEMPLATE_BRANDING, **dict(data["branding"])}
    if branding.get("photo_title") == "Foto's":
        branding["photo_title"] = DEFAULT_TEMPLATE_BRANDING["photo_title"]
    page_settings = dict(data["page_settings"])
    # JSON has no tuple type, so restore ReportLab's pagesize pair after a
    # template has passed through persistent storage.
    page_settings["pagesize"] = tuple(page_settings["pagesize"])
    template = ExportTemplate(
        template_id=str(data["template_id"]), display_name=str(data["display_name"]),
        page_settings=page_settings, branding=branding,
        colors={key: tuple(value) for key, value in dict(data["colors"]).items()},
        enabled_sections=tuple(data["enabled_sections"]), section_order=tuple(data["section_order"]),
        section_fields={str(k): tuple(v) for k, v in dict(data.get("section_fields", SECTION_FIELD_KEYS)).items()},
        field_labels={str(k): str(v) for k, v in dict(data["field_labels"]).items()},
        photo_grid={**WERKLOGGER_EXPORT_TEMPLATE.photo_grid, **dict(data["photo_grid"])},
        output_filename_pattern=str(data["output_filename_pattern"]),
        include_photo_pages=bool(data.get("include_photo_pages", True)),
        include_loose_images=bool(data.get("include_loose_images", True)),
        include_pdf_attachments=bool(data.get("include_pdf_attachments", True)),
        create_output_zip=bool(data.get("create_output_zip", True)),
        empty_value_fallback=str(data.get("empty_value_fallback", "Niet ingevuld")),
        layout=dict(data["layout"]),
        csv_headers=tuple(str(value) for value in data.get("csv_headers", ())),
        csv_column_mapping={str(k): str(v) for k, v in dict(data.get("csv_column_mapping", {})).items()},
        pdf_field_columns={str(k): str(v) for k, v in dict(data.get("pdf_field_columns", {})).items()},
    )
    if resolve_export_template(template) is not template:
        raise ValueError("Invalid export template")
    return template


def validate_template_values(name: str, title: str, pattern: str, colors: Mapping[str, Any],
                             sections: List[str], rows: Any, columns: Any,
                             page_settings: Optional[Mapping[str, Any]] = None,
                             photo_grid: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Return all user-facing validation errors for editable template values."""
    errors: List[str] = []
    if not name.strip(): errors.append("Template name is required.")
    if not title.strip(): errors.append("Report title is required.")
    if not pattern.strip(): errors.append("Filename pattern is required.")
    try:
        substitute_export_placeholders(pattern, {key: "value" for key in FILENAME_PLACEHOLDERS}, FILENAME_PLACEHOLDERS)
    except ValueError as exc:
        errors.append(f"Invalid filename pattern: {exc}")
    if not sections: errors.append("Enable at least one section.")
    try:
        parsed_rows, parsed_columns = int(rows), int(columns)
        if str(rows).strip() != str(parsed_rows) or str(columns).strip() != str(parsed_columns): raise ValueError
        candidate_grid = dict(photo_grid or WERKLOGGER_EXPORT_TEMPLATE.photo_grid)
        candidate_grid.update(rows=parsed_rows, columns=parsed_columns)
        calculate_photo_grid_geometry(page_settings or WERKLOGGER_EXPORT_TEMPLATE.page_settings, candidate_grid)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc) if str(exc).startswith("Photo grid") else
                      "Photo rows and columns must be positive whole numbers within the printable page.")
    for key, value in colors.items():
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(value).strip()):
            errors.append(f"Color {key} must use #RRGGBB format.")
    return errors


def substitute_export_placeholders(pattern: str, values: Mapping[str, str], allowed: Set[str] | frozenset[str]) -> str:
    """Substitute plain, allow-listed ``{key}`` placeholders only.

    Format specifications, conversions, indexing, and attribute access are
    deliberately rejected rather than delegated to ``str.format``.
    """
    output: List[str] = []
    try:
        parsed = Formatter().parse(pattern)
        for literal, key, format_spec, conversion in parsed:
            output.append(literal)
            if key is None:
                continue
            if not key or key not in allowed:
                raise ValueError(f"unknown placeholder: {key or '<empty>'}")
            if format_spec or conversion:
                raise ValueError(f"formatting is not allowed for placeholder: {key}")
            output.append(str(values.get(key, "")))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return "".join(output)


def export_field_value(report: ReportRow, key: str, empty_fallback: str = "") -> Any:
    """Read a registered report field and apply fallback to missing text values."""
    try:
        value = EXPORT_FIELD_REGISTRY[key].getter(report)
    except KeyError as exc:
        raise ValueError(f"Unknown export field: {key}") from exc
    if value is None or (isinstance(value, str) and not value.strip()):
        return empty_fallback
    return value


def _decode_export_template_config(path: Path, *, legacy: bool = False) -> List[ExportTemplate]:
    """Decode and validate either a current document or the historical bare list."""
    result = [WERKLOGGER_EXPORT_TEMPLATE]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if legacy:
            items = data
        else:
            version = data.get("version") if isinstance(data, dict) else None
            if type(version) is not int or version not in (1, EXPORT_TEMPLATE_CONFIG_VERSION):
                raise ExportTemplateConfigurationError(
                    f"Unsupported export-template configuration version {version!r} in {path}; "
                    f"expected {EXPORT_TEMPLATE_CONFIG_VERSION}."
                )
            items = data.get("templates")
        if not isinstance(items, list):
            raise ExportTemplateConfigurationError(f"Invalid export-template list in {path}.")
        for item in items:
            template = export_template_from_dict(item)
            if template.template_id != WERKLOGGER_EXPORT_TEMPLATE.template_id:
                result.append(template)
    except ExportTemplateConfigurationError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ExportTemplateConfigurationError(
            f"Could not read export-template configuration {path}: {exc}"
        ) from exc
    return result


def load_export_templates(path: Optional[Path] = None) -> List[ExportTemplate]:
    """Load templates, migrating old storage and recovering a damaged file once."""
    destination = path or export_template_config_path()
    if not destination.exists():
        legacy = _adjacent_export_template_path()
        if path is None and legacy.exists() and legacy != destination:
            templates = _decode_export_template_config(legacy, legacy=True)
            save_export_templates(templates, destination)
            return templates
        return [WERKLOGGER_EXPORT_TEMPLATE]
    try:
        return _decode_export_template_config(destination)
    except ExportTemplateConfigurationError as original_error:
        backup = destination.with_suffix(destination.suffix + ".bak")
        if not backup.exists():
            raise
        try:
            recovered = _decode_export_template_config(backup)
            damaged = destination.with_name(f"{destination.name}.damaged-{uuid.uuid4().hex}")
            os.replace(destination, damaged)
            _atomic_write(destination, backup.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ExportTemplateConfigurationError) as recovery_error:
            raise ExportTemplateConfigurationError(
                f"{original_error} Backup recovery also failed: {recovery_error}"
            ) from original_error
        raise ExportTemplateConfigurationError(
            f"The damaged export-template configuration was preserved as {damaged}. "
            f"Templates were recovered from {backup}.",
            recovered_templates=recovered,
        ) from original_error


def save_export_templates(templates: List[ExportTemplate], path: Optional[Path] = None) -> None:
    """Atomically persist templates and back up the last valid document."""
    destination = path or export_template_config_path()
    invalid = [template.template_id for template in templates if resolve_export_template(template) is not template]
    if invalid:
        raise ExportTemplateConfigurationError("Refusing to save invalid template(s): " + ", ".join(invalid))
    document = {"version": EXPORT_TEMPLATE_CONFIG_VERSION, "templates": [
        export_template_to_dict(t) for t in templates
        if t.template_id != WERKLOGGER_EXPORT_TEMPLATE.template_id
    ]}
    try:
        if destination.exists():
            _decode_export_template_config(destination)
            _atomic_write(destination.with_suffix(destination.suffix + ".bak"),
                          destination.read_text(encoding="utf-8"))
        _atomic_write(destination, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    except ExportTemplateConfigurationError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ExportTemplateConfigurationError(
            f"Could not write export-template configuration {destination}: {exc}"
        ) from exc


def resolve_export_template(template: Optional[ExportTemplate]) -> ExportTemplate:
    """Return a usable template, falling back atomically to the built-in one."""
    if not isinstance(template, ExportTemplate):
        return WERKLOGGER_EXPORT_TEMPLATE
    required_sections = {"address", "lmra", "work_details", "materials", "post_registrations", "photos"}
    required_layout = set(WERKLOGGER_EXPORT_TEMPLATE.layout)
    required_labels = set(WERKLOGGER_EXPORT_TEMPLATE.field_labels)
    required_colors = set(WERKLOGGER_EXPORT_TEMPLATE.colors)
    required_grid = set(WERKLOGGER_EXPORT_TEMPLATE.photo_grid)
    try:
        valid = (
            bool(template.template_id and template.display_name and template.output_filename_pattern)
            and set(template.enabled_sections).issubset(required_sections)
            and len(template.section_order) == len(set(template.section_order))
            and set(template.enabled_sections).issubset(set(template.section_order))
            and set(template.section_order).issubset(required_sections)
            and set(template.section_fields) == required_sections
            and all(set(keys).issubset(EXPORT_FIELD_REGISTRY) for keys in template.section_fields.values())
            and all(tuple(template.section_fields[key]) == keys for key, keys in SECTION_FIELD_KEYS.items())
            and bool(template.page_settings.get("pagesize"))
            and "bottom_y" in template.page_settings
            and required_layout.issubset(template.layout)
            and set(template.field_labels) == required_labels
            and set(template.field_labels).issubset(EXPORT_FIELD_REGISTRY)
            and required_colors.issubset(template.colors)
            and required_grid.issubset(template.photo_grid)
            and set(template.csv_column_mapping).issubset(CSV_COLUMN_ROLES)
            and set(template.pdf_field_columns).issubset(PDF_COLUMN_FIELDS)
            and all(value in template.csv_headers for value in template.csv_column_mapping.values())
            and all(value in template.csv_headers for value in template.pdf_field_columns.values())
            and REQUIRED_BRANDING_KEYS.issubset(template.branding)
            and all(isinstance(template.branding[key], str) and template.branding[key].strip()
                    for key in REQUIRED_BRANDING_KEYS)
            and "{title}" in template.branding["section_continuation_pattern"]
            and bool(calculate_photo_grid_geometry(template.page_settings, template.photo_grid))
        )
        if valid:
            substitute_export_placeholders(
                template.output_filename_pattern,
                {key: "value" for key in FILENAME_PLACEHOLDERS},
                FILENAME_PLACEHOLDERS,
            )
    except (AttributeError, TypeError, ValueError):
        valid = False
    return template if valid else WERKLOGGER_EXPORT_TEMPLATE


@dataclass
class ProcessResult:
    report: ReportRow
    pdf_path: Path
    written_images: int
    output_zip_path: Optional[Path] = None
    copied_pdfs: int = 0
    generated_artifacts: List[Path] = field(default_factory=list)


# =========================
# Utility helpers
# =========================


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



# =========================
# Project list persistence (GUI dropdown)
# =========================

CONFIG_VERSION = 2
DEFAULT_EXPORT_TEMPLATE_ID = "werklogger-report-v1"


@dataclass
class ProjectRecord:
    id: str
    display_name: str
    default_template_id: Optional[str] = None


@dataclass
class TemplateRecord:
    id: str
    display_name: str


@dataclass(frozen=True)
class ExportTemplate:
    """Rendering configuration selected by a stable template record ID."""

    id: str
    display_name: str
    page_settings: dict
    section_order: Tuple[str, ...]
    enabled_sections: Set[str]
    output_filename_pattern: str


@dataclass
class AppConfig:
    version: int
    projects: List[ProjectRecord]
    templates: List[TemplateRecord]


WERKLOGGER_EXPORT_TEMPLATE = ExportTemplate(
    id=DEFAULT_EXPORT_TEMPLATE_ID,
    display_name="Werklogger report",
    page_settings={"pagesize": A4},
    section_order=("report", "materials", "post_registrations", "photos"),
    enabled_sections={"report", "materials", "post_registrations", "photos"},
    output_filename_pattern="{building_id}-{project_name}-{report_datetime}-RAPPORT.pdf",
)
EXPORT_TEMPLATES = {WERKLOGGER_EXPORT_TEMPLATE.id: WERKLOGGER_EXPORT_TEMPLATE}
DEFAULT_TEMPLATES = [
    TemplateRecord(template.id, template.display_name)
    for template in EXPORT_TEMPLATES.values()
]


def resolve_export_template(
    template: Optional[ExportTemplate] = None,
    template_id: Optional[str] = None,
) -> ExportTemplate:
    """Resolve a renderer or its stable ID to a supported export template."""
    if template is not None:
        return template
    resolved_id = template_id or DEFAULT_EXPORT_TEMPLATE_ID
    try:
        return EXPORT_TEMPLATES[resolved_id]
    except KeyError as exc:
        raise ValueError(f"Unknown export template ID: {resolved_id}") from exc

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


PROJECT_CONFIG_VERSION = 2


@dataclass(frozen=True)
class ProjectRecord:
    """A persisted project and its preferred export template."""

    project_id: str
    display_name: str
    default_template_id: str = WERKLOGGER_EXPORT_TEMPLATE.template_id


def _project_record(name: str) -> ProjectRecord:
    """Convert a historical project name to a stable version-2 record."""
    name = name.strip()
    return ProjectRecord(name, name, WERKLOGGER_EXPORT_TEMPLATE.template_id)


DEFAULT_PROJECT_RECORDS = tuple(_project_record(name) for name in DEFAULT_PROJECTS)


class ProjectConfigurationError(RuntimeError):
    """A project configuration could not be safely read or written."""


def project_config_path() -> Path:
    """Return the platform-appropriate, per-user project configuration file."""
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return root / "SSV ZIP Processor" / "projects.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SSV ZIP Processor" / "projects.json"
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "ssv-zip-processor" / "projects.json"


def _legacy_project_paths() -> Tuple[Path, Path]:
    adjacent_base = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
                     else Path(__file__).resolve().parent)
    return adjacent_base / "projects.json", Path.home() / ".ssv_zip_processor_projects.json"


def _decode_project_config(path: Path, *, legacy: bool = False) -> List[ProjectRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectConfigurationError(f"Could not read project configuration {path}: {exc}") from exc
    version = 1 if legacy else data.get("version") if isinstance(data, dict) else None
    if legacy:
        projects = data
    else:
        if (not isinstance(data, dict) or type(data.get("version")) is not int
                or data.get("version") not in (1, PROJECT_CONFIG_VERSION)):
            raise ProjectConfigurationError(
                f"Unsupported project configuration version {version!r} in {path}; "
                f"expected {PROJECT_CONFIG_VERSION}."
            )
        projects = data.get("projects")
    if not isinstance(projects, list):
        raise ProjectConfigurationError(f"Invalid project list in {path}.")
    if version == 1:
        if any(not isinstance(item, str) for item in projects):
            raise ProjectConfigurationError(f"Invalid version-1 project list in {path}.")
        return [_project_record(item) for item in projects if item.strip()]
    records: List[ProjectRecord] = []
    for item in projects:
        if (not isinstance(item, dict)
                or any(not isinstance(item.get(key), str)
                       for key in ("project_id", "display_name", "default_template_id"))):
            raise ProjectConfigurationError(f"Invalid project record in {path}.")
        try:
            record = ProjectRecord(
                project_id=str(item["project_id"]).strip(),
                display_name=str(item["display_name"]).strip(),
                default_template_id=str(item["default_template_id"]).strip(),
            )
        except (KeyError, TypeError) as exc:
            raise ProjectConfigurationError(f"Invalid project record in {path}.") from exc
        if not record.project_id or not record.display_name or not record.default_template_id:
            raise ProjectConfigurationError(f"Invalid project record in {path}.")
        records.append(record)
    return records


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_projects(projects: List[ProjectRecord]) -> None:
    """Atomically save project records and back up the last valid configuration."""
    unique = list({project.project_id: project for project in projects}.values())
    path = project_config_path()
    try:
        if path.exists():
            _decode_project_config(path)  # Never replace an invalid file or call it a valid backup.
            _atomic_write(path.with_suffix(path.suffix + ".bak"), path.read_text(encoding="utf-8"))
        document = {"version": PROJECT_CONFIG_VERSION, "projects": [
            {"project_id": project.project_id, "display_name": project.display_name,
             "default_template_id": project.default_template_id}
            for project in unique
        ]}
        _atomic_write(path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    except (OSError, UnicodeError, ProjectConfigurationError) as exc:
        if isinstance(exc, ProjectConfigurationError):
            raise
        raise ProjectConfigurationError(f"Could not write project configuration {path}: {exc}") from exc


def load_projects() -> List[ProjectRecord]:
    """Load the versioned configuration, migrating both historical locations once."""
    path = project_config_path()
    custom: List[ProjectRecord] = []
    if path.exists():
        custom = _decode_project_config(path)
        # Reading a v1 document is also its in-place, lossless migration.
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") == 1:
            save_projects([*DEFAULT_PROJECT_RECORDS, *custom])
    else:
        legacy_found = False
        for legacy_path in dict.fromkeys(_legacy_project_paths()):
            if legacy_path.exists():
                legacy_found = True
                custom.extend(_decode_project_config(legacy_path, legacy=True))
        if legacy_found:
            save_projects(custom)

    return list({project.project_id: project for project in [*DEFAULT_PROJECT_RECORDS, *custom]}.values())


def project_default_template(project: ProjectRecord, templates: List[ExportTemplate]) -> ExportTemplate:
    """Resolve a project's default, safely falling back when it was deleted."""
    return next((template for template in templates
                 if template.template_id == project.default_template_id), WERKLOGGER_EXPORT_TEMPLATE)


# =========================
# CSV parsing
# =========================
REQUIRED_COLUMNS = ["ID", "Type", "Label", "Primary", "Secondary", "Note", "Media"]


def find_header_row(csv_path: Path, column_mapping: Optional[Mapping[str, str]] = None) -> int:
    """Find the header row index (0-based) that contains required columns."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if configured_row is not None and idx != configured_row:
                continue
            row_norm = [c.strip() for c in row[:column_count]]
            if not row_norm:
                continue
            # check if it includes required columns
            cols = {c for c in row_norm}
            required = [column_mapping.get(key, key) if column_mapping else key for key in REQUIRED_COLUMNS]
            if all(c in cols for c in required):
                return idx
    raise ValueError("Could not find CSV header row with required columns.")


def inspect_csv_headers(csv_path: Path) -> Tuple[str, ...]:
    """Read the first widest row, allowing metadata lines before the header."""
    candidate: Tuple[str, ...] = ()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            values = tuple(cell.strip() for cell in row)
            if len(values) > len(candidate):
                candidate = values
    headers = tuple(value for value in candidate if value)
    if not headers:
        raise ValueError("The selected CSV does not contain a header row.")
    if len(headers) != len(candidate) or len(headers) != len(set(headers)):
        raise ValueError("CSV template column names must be non-empty and unique.")
    return headers


def read_first_csv_values(csv_path: Path, header_idx: int) -> Dict[str, str]:
    """Return the first non-empty value per source column for direct PDF mappings."""
    values: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for _ in range(header_idx):
            next(handle)
        for row in csv.DictReader(handle):
            for key, value in row.items():
                cleaned = (value or "").strip()
                if key and cleaned:
                    values.setdefault(key.strip(), cleaned)
    return values


def load_audit_csv(csv_path: Path, column_mapping: Optional[Mapping[str, str]] = None) -> Tuple[Dict[str, str], Dict[str, str], List[MediaRow], List[AuditRow]]:
    """Load SafetyAuditor export CSV and return meta, field-values, media rows, audit rows."""
    mapping = dict(column_mapping or {})
    header_idx = find_header_row(csv_path, mapping)
    source = lambda raw, key: raw.get(mapping.get(key, key))  # noqa: E731

    audit_rows: List[AuditRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        # skip to header
        for _ in range(header_idx):
            f.readline()

        reader = csv.DictReader(f)
        for raw in reader:
            rid = (source(raw, "ID") or "").strip()
            row = AuditRow(
                row_id=rid,
                parent_id=(source(raw, "Parent ID") or raw.get("ParentID") or raw.get("Parent") or "").strip(),
                row_type=(source(raw, "Type") or "").strip(),
                label=(source(raw, "Label") or "").strip(),
                primary=(source(raw, "Primary") or "").strip(),
                secondary=(source(raw, "Secondary") or "").strip(),
                note=(source(raw, "Note") or "").strip(),
                media=(source(raw, "Media") or "").strip(),
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


@dataclass
class _Page1RenderContext:
    """Shared drawing state used by the ordered page-one section renderers."""

    canvas: Canvas
    report: ReportRow
    template: ExportTemplate
    page_number: int = 1

    def value(self, key: str) -> Any:
        return export_field_value(self.report, key, self.template.empty_value_fallback)

    @property
    def top_y(self) -> float:
        return 739.78

    def draw_header(self, continuation: bool = False) -> None:
        c, layout = self.canvas, self.template.layout
        width, _ = self.template.page_settings["pagesize"]
        red = self.template.colors["primary"]
        c.setFillColorRGB(*red)
        c.rect(0, layout["banner_y"], width, layout["banner_height"], stroke=0, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(layout["left"], layout["title_y"], self.template.branding["report_title"])
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(layout["right"], layout["datetime_y"], self.value("report_datetime"))
        if not continuation:
            # The compatibility layout has a black report title on its first page.
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(layout["left"], layout["title_y"], self.template.branding["report_title"])
        c.setFillColorRGB(0, 0, 0)

    def new_page(self) -> float:
        self.canvas.showPage()
        self.page_number += 1
        self.draw_header(continuation=True)
        return self.top_y

    def draw_section_title(self, title: str, y: float, color: Tuple[float, float, float]) -> None:
        c, layout = self.canvas, self.template.layout
        c.setFillColorRGB(*color)
        c.setFont("Helvetica", 14)
        c.drawString(layout["left"], y, title)
        c.setStrokeColorRGB(*color)
        c.setLineWidth(layout["line_width"])
        c.line(layout["left"], y - 8.45, layout["right"], y - 8.45)
        c.setFillColorRGB(0, 0, 0)

    def ensure_space(self, y: float, required_height: float) -> float:
        if y - required_height < self.template.page_settings["bottom_y"]:
            return self.new_page()
        return y


def _render_address_section(ctx: _Page1RenderContext, y: float) -> float:
    y = ctx.ensure_space(y, 145.0)
    ctx.draw_section_title(ctx.template.branding["section_address"], y, ctx.template.colors["primary"])
    row_y = y - 24.6844
    for key in ctx.template.section_fields["address"]:
        ctx.canvas.setFont("Helvetica-Bold", 10)
        ctx.canvas.drawString(ctx.template.layout["left"], row_y, ctx.template.field_labels[key])
        ctx.canvas.setFont("Helvetica", 10)
        ctx.canvas.drawString(ctx.template.layout["address_value_x"], row_y, ctx.value(key) or "")
        row_y -= ctx.template.layout["line_spacing"]
    return y - 158.74


def _render_lmra_section(ctx: _Page1RenderContext, y: float) -> float:
    y = ctx.ensure_space(y, 165.0)
    c, layout = ctx.canvas, ctx.template.layout
    ctx.draw_section_title(ctx.template.branding["section_lmra"], y, ctx.template.colors["primary"])
    bar_y = y - 42.4573
    c.setFillColorRGB(*ctx.template.colors["status_bar"])
    c.rect(layout["left"], bar_y, layout["right"] - layout["left"], 19.8425, stroke=0, fill=1)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    c.drawCentredString((layout["left"] + layout["right"]) / 2.0, bar_y + 5.0,
                        ctx.template.branding["lmra_status"])
    row_y = y - 58.7014
    for item in (ctx.template.branding[f"lmra_item_{index}"] for index in range(1, 7)):
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(layout["left"], row_y, item)
        c.setFillColorRGB(*ctx.template.colors["status_yes"])
        c.drawString(layout["checklist_yes_x"], row_y, ctx.template.branding["yes_label"])
        row_y -= layout["line_spacing"]
    c.setFillColorRGB(0, 0, 0)
    return y - 175.75


def _render_work_details_section(ctx: _Page1RenderContext, y: float) -> float:
    y = ctx.ensure_space(y, 65.0)
    ctx.draw_section_title(ctx.template.branding["section_work_details"], y, ctx.template.colors["primary"])
    row_y = y - 24.6758
    for key in ctx.template.section_fields["work_details"]:
        ctx.canvas.setFont("Helvetica-Bold", 10)
        ctx.canvas.drawString(ctx.template.layout["left"], row_y, ctx.template.field_labels[key])
        ctx.canvas.setFont("Helvetica", 10)
        ctx.canvas.drawString(ctx.template.layout["details_value_x"], row_y, ctx.value(key) or "")
        row_y -= ctx.template.layout["line_spacing"]
    return y - 70.86


def _render_bullet_section(ctx: _Page1RenderContext, y: float, section: str) -> float:
    raw = ctx.value(ctx.template.section_fields[section][0])
    lines = list(raw if isinstance(raw, list) else [raw])
    title = ctx.template.branding[SECTION_BRANDING_KEYS[section]]
    dy = ctx.template.layout["line_spacing"]
    while True:
        y = ctx.ensure_space(y, 24.6898 + dy)
        ctx.draw_section_title(title, y, ctx.template.colors["secondary"])
        text_y = y - 24.6898
        remaining: List[str] = []
        for index, line in enumerate(lines):
            wrapped = _wrap_lines(str(line), 110) or ["-"]
            if text_y - dy * (len(wrapped) - 1) < ctx.template.page_settings["bottom_y"]:
                remaining = lines[index:]
                break
            ctx.canvas.setFont("Helvetica", 10)
            ctx.canvas.drawString(ctx.template.layout["left"], text_y, u"\u2022")
            ctx.canvas.drawString(62.97, text_y, wrapped[0])
            text_y -= dy
            for continuation in wrapped[1:]:
                ctx.canvas.drawString(62.97, text_y, continuation)
                text_y -= dy
        if not remaining:
            return text_y - 14.0
        y = ctx.new_page()
        title = ctx.template.branding["section_continuation_pattern"].replace(
            "{title}", ctx.template.branding[SECTION_BRANDING_KEYS[section]].rstrip(":"))
        lines = remaining


def _render_materials_section(ctx: _Page1RenderContext, y: float) -> float:
    return _render_bullet_section(ctx, y, "materials")


def _render_post_registrations_section(ctx: _Page1RenderContext, y: float) -> float:
    return _render_bullet_section(ctx, y, "post_registrations")


PAGE1_SECTION_RENDERERS: Mapping[str, Callable[[_Page1RenderContext, float], float]] = {
    "address": _render_address_section,
    "lmra": _render_lmra_section,
    "work_details": _render_work_details_section,
    "materials": _render_materials_section,
    "post_registrations": _render_post_registrations_section,
}


def render_page1(c: Canvas, report: ReportRow, template: Optional[ExportTemplate] = None) -> None:
    """
    Render enabled non-photo sections in the order declared by the template.

    The built-in template retains its historic fixed section coordinates.  Custom
    templates use the returned vertical position from each renderer, so reordering
    ``section_order`` also reorders the visible PDF content.
    """
    template = resolve_export_template(template)
    ctx = _Page1RenderContext(c, report, template)
    ctx.draw_header()
    enabled = set(template.enabled_sections)
    ordered_sections = [name for name in template.section_order
                        if name in enabled and name in PAGE1_SECTION_RENDERERS]
    compatibility_layout = template is WERKLOGGER_EXPORT_TEMPLATE
    compatibility_y = {
        "address": 739.78, "lmra": 581.04, "work_details": 405.29,
        "materials": 334.43, "post_registrations": 244.09,
    }
    y = ctx.top_y
    for section in ordered_sections:
        # Fixed anchors are intentionally limited to the built-in template.  A
        # custom template always consumes the preceding renderer's position.
        if compatibility_layout and ctx.page_number == 1:
            y = compatibility_y[section]
        y = PAGE1_SECTION_RENDERERS[section](ctx, y)



def render_photos_pages(c: Canvas, report: ReportRow, template: Optional[ExportTemplate] = None) -> None:
    """Render photo pages using geometry derived from the template and page."""
    template = resolve_export_template(template)
    w, h = template.page_settings["pagesize"]
    grid, layout = template.photo_grid, template.layout
    X_LEFT, X_RIGHT = grid["left"], grid["right"]
    RED, LINE_W = template.colors["primary"], layout["line_width"]
    geometry = calculate_photo_grid_geometry(template.page_settings, grid)
    heading_y, underline_y = geometry.heading_y, geometry.underline_y
    columns = int(grid["columns"])
    box_w, column_boxes = geometry.box_width, geometry.column_boxes
    box_h = geometry.box_height
    label_allowance = float(grid.get("label_allowance", 18.0))
    rows_per_page = int(grid["rows"])

    def draw_page_heading(first: bool) -> None:
        c.setFillColorRGB(*RED)
        c.setFont("Helvetica", 14)
        key = "photo_title" if first else "photo_continuation_title"
        c.drawString(X_LEFT, heading_y, template.branding[key])
        c.setStrokeColorRGB(*RED)
        c.setLineWidth(LINE_W)
        c.line(X_LEFT, underline_y, X_RIGHT, underline_y)
        c.setFillColorRGB(0, 0, 0)

    # Group photos by label (preserve original order)
    groups: List[Tuple[str, List[Photo]]] = []
    seen_order: List[str] = []
    by_label: Dict[str, List[Photo]] = {}
    registered_photos = export_field_value(report, template.section_fields["photos"][0], [])
    for p in registered_photos:
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

    row_index = 0

    def ensure_row_available() -> None:
        nonlocal row_index, first_page
        if row_index >= rows_per_page:
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

            row_top = geometry.row_tops[row_index]
            row_bottom = row_top - box_h
            label_y = row_top + label_allowance * 0.35

            # Print label once at the first row for the group on this page.
            if not label_printed_on_this_page:
                c.setFont("Helvetica-Bold", 10)
                c.setFillColorRGB(0, 0, 0)
                c.drawString(X_LEFT, label_y, f"{label} ({len(photos)})")
                label_printed_on_this_page = True

            row_photos = [remaining.pop(0) if remaining else None for _ in range(columns)]

            def draw_photo(photo: Optional[Photo], x0: float, x1: float) -> None:
                if not photo or not photo.image_path or not photo.image_path.exists():
                    return
                try:
                    with Image.open(photo.image_path) as im:
                        # Keep photo quality maximal in the PDF:
                        # - do not downscale to a target DPI
                        # - do not re-encode to JPEG (no extra lossy compression)
                        im_oriented = ImageOps.exif_transpose(im).convert("RGB")
                        dx = x0
                        dy = row_bottom
                        c.drawImage(
                            ImageReader(im_oriented),
                            dx,
                            dy,
                            width=box_w,
                            height=box_h,
                            preserveAspectRatio=True,
                            mask='auto',
                            anchor='c',
                        )
                except Exception:
                    return

            for photo, (x0, x1) in zip(row_photos, column_boxes):
                draw_photo(photo, x0, x1)

            row_index += 1

            # If the group continues onto a new page, allow label to repeat.
            if row_index >= rows_per_page and remaining:
                label_printed_on_this_page = False



def create_output_zip(out_zip_path: Path, files: List[Tuple[Path, str]]) -> None:
    """Create an output ZIP without silently replacing files or archive members."""
    if out_zip_path.exists():
        raise FileExistsError(f"Output file already exists: {out_zip_path}")
    safe_files: List[Tuple[Path, str]] = []
    seen: Set[str] = set()
    for src, arcname in files:
        sanitized = safe_filename(Path(arcname).name)
        key = sanitized.casefold()
        if key in seen:
            raise FileExistsError(f"Duplicate output ZIP member: {sanitized}")
        if not src.is_file():
            raise FileNotFoundError(f"ZIP input does not exist: {src}")
        seen.add(key)
        safe_files.append((src, sanitized))
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src, arcname in safe_files:
            z.write(src, arcname)



def build_pdf(out_pdf_path: Path, report: ReportRow, template: Optional[ExportTemplate] = None) -> None:
    template = resolve_export_template(template)
    c = Canvas(str(out_pdf_path), pagesize=template.page_settings["pagesize"])
    non_photo_sections = [s for s in template.section_order if s != "photos" and s in template.enabled_sections]
    if non_photo_sections:
        render_page1(c, report, template)
    if template.include_photo_pages and "photos" in template.enabled_sections and report.photos:
        if non_photo_sections:
            c.showPage()
        render_photos_pages(c, report, template)
    c.save()


# =========================
# Image processing + report building
# =========================
def process_zip_to_folder_and_pdf(zip_path: Path, out_dir: Path, project_override: Optional[str] = None, log: Optional[callable] = None, template: Optional[ExportTemplate] = None) -> ProcessResult:
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

        template = resolve_export_template(template)
        header_idx = find_header_row(csv_path, template.csv_column_mapping)
        meta, fields, media_rows, audit_rows = load_audit_csv(csv_path, template.csv_column_mapping)
        direct_values = read_first_csv_values(csv_path, header_idx)

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
            project_locatie_naam=(project_override.strip() if project_override and project_override.strip() else fields.get("Project/Locatie Naam", "").strip()),
            building_id=fields.get("Building ID", "").strip(),
            adres=sanitize_address(fields.get("Adres", "").strip()),
            postcode_stad=(fields.get("Postcode + Stad", "") or "").replace("+", " ").strip(),
            contactpersoon=fields.get("Contactpersoon", "").strip(),
            quadrant=fields.get("Quadrant", "").strip() or "",
            duct_kleur=fields.get("Gekoppelde kleur subduct", "").strip() or fields.get("Gekoppelde kleur duct", "").strip() or "",
            units_gelast=fields.get("Hoeveel units gelast?", "").strip() or fields.get("Hoeveel units gelast", "").strip() or "",
            gebruikte_materialen_lines=gebruikte_materialen,
            post_afmeldingen_lines=post_afmeldingen,
            photos=[],
        )
        report_attributes = {
            "report_datetime": "report_datetime", "subcontractor": "naam_onderaannemer",
            "project_name": "project_locatie_naam", "building_id": "building_id",
            "address": "adres", "postal_city": "postcode_stad", "contact": "contactpersoon",
            "quadrant": "quadrant", "duct_color": "duct_kleur", "units_welded": "units_gelast",
        }
        for field_key, column_name in template.pdf_field_columns.items():
            value = direct_values.get(column_name, "")
            if value:
                if field_key == "address": value = sanitize_address(value)
                setattr(report, report_attributes[field_key], value)

        # Build an index for images (case-insensitive)
        all_files = list(tmp.rglob("*"))
        img_by_stem: Dict[str, Path] = {}
        for p in all_files:
            if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}:
                img_by_stem[p.stem.lower()] = p
        nonimg_stems: Set[str] = {p.stem.lower() for p in all_files if p.is_file() and p.suffix.lower() not in {'.jpeg','.jpg'}}

        # Plan every output before writing anything, so collisions never cause a
        # partially overwritten export. Image discovery itself is unchanged.
        used_names: Set[str] = set()
        planned_images: List[Tuple[Path, str, str]] = []

        for mrow in media_rows:
            base_label = safe_filename(normalize_label(mrow.label))
            for idx, mid in enumerate(mrow.media_ids, start=1):
                src = img_by_stem.get(mid.lower())
                if not src:
                    # If a non-image file shares this stem (e.g. a PDF attachment), silently skip.
                    if mid.lower() in nonimg_stems:
                        continue
                    _log(f"WARNING: image not found for media id: {mid}")
                    continue

                out_base = base_label if len(mrow.media_ids) == 1 else f"{base_label}_{idx}"
                out_name = safe_filename(f"{out_base}.jpeg")
                if out_name.lower() in used_names:
                    n = 2
                    while safe_filename(f"{out_base}_{n}.jpeg").lower() in used_names:
                        n += 1
                    out_name = safe_filename(f"{out_base}_{n}.jpeg")
                used_names.add(out_name.lower())
                planned_images.append((src, out_name, mrow.label))

        planned_attachments: List[Tuple[Path, Path]] = []
        if template.include_pdf_attachments:
            for source in all_files:
                if source.is_file() and source.suffix.lower() == ".pdf":
                    planned_attachments.append((source, out_dir / safe_filename(source.name)))

        # Output naming is presentation configuration; CSV parsing remains template-independent.
        filename_values = {
            key: safe_filename(str(export_field_value(report, key, template.empty_value_fallback)))
            for key in FILENAME_PLACEHOLDERS
        }
        fn = substitute_export_placeholders(
            template.output_filename_pattern, filename_values, FILENAME_PLACEHOLDERS
        )
        fn = safe_filename(Path(fn).stem) + ".pdf"
        pdf_path = out_dir / fn
        out_zip_path: Optional[Path] = None
        if template.create_output_zip:
            out_zip_path = out_dir / (safe_filename(Path(fn).stem) + "-OUTPUT.zip")

        loose_image_paths = [out_dir / name for _source, name, _label in planned_images]
        planned_outputs = [pdf_path]
        if template.include_loose_images:
            planned_outputs.extend(loose_image_paths)
        planned_outputs.extend(destination for _source, destination in planned_attachments)
        if out_zip_path is not None:
            planned_outputs.append(out_zip_path)
        seen_outputs: Dict[str, Path] = {}
        for destination in planned_outputs:
            key = destination.name.casefold()
            if key in seen_outputs:
                raise FileExistsError(
                    f"Output filename collision: {seen_outputs[key].name} and {destination.name}"
                )
            if destination.exists():
                raise FileExistsError(f"Output file already exists: {destination}")
            seen_outputs[key] = destination

        _log("Processing images...")
        processed_photos: List[Photo] = []
        for source, name, label in planned_images:
            destination = out_dir / name if template.include_loose_images else tmp / "processed" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            processed_photos.append(Photo(label=label, image_path=destination))
        report.photos = processed_photos

        pdf_attachments: List[Path] = []
        for source, destination in planned_attachments:
            shutil.copy2(source, destination)
            pdf_attachments.append(destination)
        if pdf_attachments:
            _log(f"Copied {len(pdf_attachments)} PDF attachment(s).")

        _log("Generating PDF...")
        build_pdf(pdf_path, report, template)
        _log(f"Saved PDF: {pdf_path.name}")

        artifacts = [pdf_path]
        if template.include_loose_images:
            artifacts.extend(loose_image_paths)
        artifacts.extend(pdf_attachments)

        if out_zip_path is not None:
            zip_files = [(path, path.name) for path in artifacts]
            _log("Creating output ZIP...")
            create_output_zip(out_zip_path, zip_files)
            artifacts.append(out_zip_path)
            _log(f"Saved ZIP: {out_zip_path.name}")

        return ProcessResult(
            report=report, pdf_path=pdf_path,
            written_images=len(loose_image_paths) if template.include_loose_images else 0,
            output_zip_path=out_zip_path, copied_pdfs=len(pdf_attachments),
            generated_artifacts=artifacts,
        )


# =========================
# GUI
# =========================
if tk is not None:
    class CsvSchemaPreviewDialog(tk.Toplevel):
        """Preview raw CSV rows and collect the table start and width."""

        def __init__(self, master: tk.Misc, csv_path: Path) -> None:
            super().__init__(master)
            self.title("Select CSV table range")
            self.geometry("1000x600")
            self.transient(master)
            self.result: Optional[Tuple[int, int, Tuple[str, ...]]] = None
            self.rows = read_csv_preview(csv_path)
            self.max_columns = max(len(row) for row in self.rows)
            widest_row = next(index for index, row in enumerate(self.rows)
                              if len(row) == self.max_columns)

            instructions = ttk.Label(
                self, padding=10, wraplength=950,
                text="Preview the CSV below. Choose the row containing the column names and "
                     "the number of leading columns that belong to the table. Row numbers are 1-based.")
            instructions.pack(fill="x")
            controls = ttk.Frame(self, padding=(10, 0, 10, 8)); controls.pack(fill="x")
            self.row_var = tk.IntVar(value=widest_row + 1)
            self.count_var = tk.IntVar(value=self.max_columns)
            ttk.Label(controls, text="Table starts at row:").pack(side="left")
            ttk.Spinbox(controls, from_=1, to=len(self.rows), textvariable=self.row_var,
                        width=7, command=self._selection_changed).pack(side="left", padx=(5, 18))
            ttk.Label(controls, text="Columns to read:").pack(side="left")
            ttk.Spinbox(controls, from_=1, to=self.max_columns, textvariable=self.count_var,
                        width=7, command=self._selection_changed).pack(side="left", padx=5)
            self.selection_note = ttk.Label(controls); self.selection_note.pack(side="left", padx=12)
            self.row_var.trace_add("write", lambda *_: self._selection_changed())
            self.count_var.trace_add("write", lambda *_: self._selection_changed())

            table_frame = ttk.Frame(self, padding=(10, 0)); table_frame.pack(fill="both", expand=True)
            columns = [f"column-{index}" for index in range(1, self.max_columns + 1)]
            self.preview = ttk.Treeview(table_frame, columns=columns, show="tree headings")
            self.preview.heading("#0", text="Row"); self.preview.column("#0", width=55, stretch=False)
            for index, column in enumerate(columns, start=1):
                self.preview.heading(column, text=f"Column {index}")
                self.preview.column(column, width=150, stretch=False)
            yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.preview.yview)
            xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.preview.xview)
            self.preview.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
            self.preview.grid(row=0, column=0, sticky="nsew")
            yscroll.grid(row=0, column=1, sticky="ns"); xscroll.grid(row=1, column=0, sticky="ew")
            table_frame.rowconfigure(0, weight=1); table_frame.columnconfigure(0, weight=1)
            self.preview.tag_configure("header", background="#d9edf7")
            for index, row in enumerate(self.rows, start=1):
                self.preview.insert("", "end", iid=str(index), text=str(index),
                                    values=(*row, *("" for _ in range(self.max_columns - len(row)))))

            footer = ttk.Frame(self, padding=10); footer.pack(fill="x")
            ttk.Button(footer, text="Cancel", command=self.destroy).pack(side="right")
            ttk.Button(footer, text="Use selected range", command=self._accept).pack(side="right", padx=8)
            self._selection_changed()
            self.grab_set(); self.wait_window()

        def _selection_changed(self) -> None:
            try:
                row_number, count = self.row_var.get(), self.count_var.get()
            except (tk.TclError, ValueError):
                return
            for item in self.preview.get_children():
                self.preview.item(item, tags=("header",) if item == str(row_number) else ())
            if 1 <= row_number <= len(self.rows):
                self.preview.see(str(row_number))
                available = len(self.rows[row_number - 1])
                self.selection_note.config(text=f"Selected row contains {available} column(s).")

        def _accept(self) -> None:
            try:
                header_row = self.row_var.get() - 1
                column_count = self.count_var.get()
                if not 0 <= header_row < len(self.rows) or not 1 <= column_count <= self.max_columns:
                    raise ValueError("Select a valid table row and column count.")
                headers = tuple(self.rows[header_row][:column_count])
                if len(headers) != column_count:
                    raise ValueError("The selected row has fewer columns than the requested count.")
                if any(not value for value in headers) or len(headers) != len(set(headers)):
                    raise ValueError("Selected column names must be non-empty and unique.")
            except (tk.TclError, ValueError) as exc:
                messagebox.showerror("Invalid table range", str(exc), parent=self); return
            self.result = (header_row, column_count, headers)
            self.destroy()


    class TemplateManagerDialog(tk.Toplevel):
        """Editor for export presentation templates and their uploaded CSV schema."""
        def __init__(self, master: "App") -> None:
            super().__init__(master)
            self.app = master
            self.title("Manage export templates")
            self.geometry("900x650")
            self.transient(master)
            self.templates = list(master.export_templates)

            left = ttk.Frame(self, padding=8); left.pack(side="left", fill="y")
            ttk.Label(left, text="Templates").pack(anchor="w")
            self.template_list = tk.Listbox(left, width=28, exportselection=False)
            self.template_list.pack(fill="y", expand=True, pady=4)
            self.template_list.bind("<<ListboxSelect>>", self._select)
            for text, command in (("New", self._new), ("Duplicate", self._duplicate),
                                  ("Rename", self._rename), ("Delete", self._delete)):
                ttk.Button(left, text=text, command=command).pack(fill="x", pady=2)

            right = ttk.Frame(self, padding=8); right.pack(side="left", fill="both", expand=True)
            self.readonly_note = ttk.Label(right, foreground="#9b0000")
            self.readonly_note.pack(anchor="w")
            book = ttk.Notebook(right); book.pack(fill="both", expand=True, pady=5)
            general = ttk.Frame(book, padding=8); sections = ttk.Frame(book, padding=8)
            labels = ttk.Frame(book, padding=8); text_tab = ttk.Frame(book, padding=8)
            csv_tab = ttk.Frame(book, padding=8)
            book.add(general, text="Report & photos"); book.add(sections, text="Sections")
            book.add(labels, text="CSV-backed fields / labels")
            book.add(csv_tab, text="CSV column mapping")
            book.add(text_tab, text="Report text")

            self.name_var = tk.StringVar(); self.title_var = tk.StringVar(); self.pattern_var = tk.StringVar()
            self.fallback_var = tk.StringVar()
            self.rows_var = tk.StringVar(); self.columns_var = tk.StringVar()
            self.output_flag_vars = {
                "include_photo_pages": tk.BooleanVar(),
                "include_loose_images": tk.BooleanVar(),
                "include_pdf_attachments": tk.BooleanVar(),
                "create_output_zip": tk.BooleanVar(),
            }
            general.columnconfigure(1, weight=1)
            fields = (("Template name", self.name_var), ("Report title", self.title_var),
                      ("Filename pattern", self.pattern_var), ("Photo rows", self.rows_var),
                      ("Photo columns", self.columns_var), ("Empty value fallback", self.fallback_var))
            for row, (caption, variable) in enumerate(fields):
                ttk.Label(general, text=caption).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
                ttk.Entry(general, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
            ttk.Label(general, text="Placeholders: {building_id}, {project_name}, {report_datetime}").grid(
                row=6, column=1, sticky="w")
            for offset, (key, caption) in enumerate((
                ("include_photo_pages", "Include photo pages in report"),
                ("include_loose_images", "Write loose processed images"),
                ("include_pdf_attachments", "Copy imported PDF attachments"),
                ("create_output_zip", "Create output ZIP"),
            ), start=7):
                ttk.Checkbutton(general, text=caption, variable=self.output_flag_vars[key]).grid(
                    row=offset, column=1, sticky="w", pady=2)
            self.color_vars = {key: tk.StringVar() for key in WERKLOGGER_EXPORT_TEMPLATE.colors}
            for offset, (key, variable) in enumerate(self.color_vars.items(), start=11):
                ttk.Label(general, text=f"{key.replace('_', ' ').title()} color").grid(row=offset, column=0, sticky="w", pady=3)
                ttk.Entry(general, textvariable=variable, width=12).grid(row=offset, column=1, sticky="w", pady=3)

            self.section_vars = {key: tk.BooleanVar() for key, _ in TEMPLATE_SECTIONS}
            for row, (key, caption) in enumerate(TEMPLATE_SECTIONS):
                ttk.Checkbutton(sections, text=caption, variable=self.section_vars[key]).grid(row=row, column=0, sticky="w")
            ttk.Label(sections, text="Order").grid(row=0, column=1, sticky="w", padx=(30, 0))
            self.order_list = tk.Listbox(sections, height=9, exportselection=False)
            self.order_list.grid(row=1, column=1, rowspan=6, padx=(30, 4), sticky="nsew")
            moves = ttk.Frame(sections); moves.grid(row=1, column=2, sticky="n")
            ttk.Button(moves, text="Move up", command=lambda: self._move(-1)).pack(fill="x")
            ttk.Button(moves, text="Move down", command=lambda: self._move(1)).pack(fill="x", pady=4)

            ttk.Label(labels, text="Import/report field ID (fixed)").grid(row=0, column=0, sticky="w")
            ttk.Label(labels, text="Display label (editable)").grid(row=0, column=1, sticky="w")
            canvas = tk.Canvas(labels, highlightthickness=0); scroll = ttk.Scrollbar(labels, command=canvas.yview)
            label_frame = ttk.Frame(canvas); canvas.create_window((0, 0), window=label_frame, anchor="nw")
            canvas.configure(yscrollcommand=scroll.set); canvas.grid(row=1, column=0, columnspan=2, sticky="nsew")
            scroll.grid(row=1, column=2, sticky="ns"); labels.rowconfigure(1, weight=1); labels.columnconfigure(1, weight=1)
            label_frame.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
            self.label_vars = {key: tk.StringVar() for key in WERKLOGGER_EXPORT_TEMPLATE.field_labels}
            for row, (key, variable) in enumerate(self.label_vars.items()):
                ttk.Label(label_frame, text=key, width=22).grid(row=row, column=0, sticky="w", pady=2)
                ttk.Entry(label_frame, textvariable=variable, width=42).grid(row=row, column=1, sticky="ew", pady=2)

            ttk.Label(csv_tab, text="Upload a CSV when creating a template, then map its column names below.",
                      wraplength=560).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
            self.csv_schema_note = ttk.Label(csv_tab)
            self.csv_schema_note.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))
            ttk.Label(csv_tab, text="CSV purpose").grid(row=2, column=0, sticky="w")
            ttk.Label(csv_tab, text="Uploaded column").grid(row=2, column=1, sticky="w")
            self.csv_role_vars = {key: tk.StringVar() for key in CSV_COLUMN_ROLES}
            self.csv_role_boxes = {}
            for row, (key, variable) in enumerate(self.csv_role_vars.items(), start=3):
                ttk.Label(csv_tab, text=key).grid(row=row, column=0, sticky="w", pady=2)
                box = ttk.Combobox(csv_tab, textvariable=variable, state="readonly", width=34)
                box.grid(row=row, column=1, sticky="ew", pady=2); self.csv_role_boxes[key] = box
            ttk.Label(csv_tab, text="PDF field (optional direct value)").grid(row=2, column=2, sticky="w", padx=(24, 0))
            ttk.Label(csv_tab, text="Uploaded column").grid(row=2, column=3, sticky="w")
            self.pdf_column_vars = {key: tk.StringVar() for key in PDF_COLUMN_FIELDS}
            self.pdf_column_boxes = {}
            for row, (key, variable) in enumerate(self.pdf_column_vars.items(), start=3):
                ttk.Label(csv_tab, text=EXPORT_FIELD_REGISTRY[key].display_name).grid(row=row, column=2, sticky="w", padx=(24, 6), pady=2)
                box = ttk.Combobox(csv_tab, textvariable=variable, state="readonly", width=34)
                box.grid(row=row, column=3, sticky="ew", pady=2); self.pdf_column_boxes[key] = box
            csv_tab.columnconfigure(3, weight=1)

            text_canvas = tk.Canvas(text_tab, highlightthickness=0)
            text_scroll = ttk.Scrollbar(text_tab, command=text_canvas.yview)
            text_frame = ttk.Frame(text_canvas)
            text_canvas.create_window((0, 0), window=text_frame, anchor="nw")
            text_canvas.configure(yscrollcommand=text_scroll.set)
            text_canvas.pack(side="left", fill="both", expand=True)
            text_scroll.pack(side="right", fill="y")
            text_frame.bind("<Configure>", lambda _e: text_canvas.configure(scrollregion=text_canvas.bbox("all")))
            self.branding_vars = {
                key: tk.StringVar() for key in DEFAULT_TEMPLATE_BRANDING if key != "report_title"
            }
            for row, (key, variable) in enumerate(self.branding_vars.items()):
                ttk.Label(text_frame, text=key.replace("_", " ").title(), width=32).grid(
                    row=row, column=0, sticky="w", pady=2)
                ttk.Entry(text_frame, textvariable=variable, width=55).grid(
                    row=row, column=1, sticky="ew", pady=2)

            footer = ttk.Frame(right); footer.pack(fill="x")
            ttk.Button(footer, text="Save changes", command=self._save).pack(side="right")
            ttk.Button(footer, text="Close", command=self.destroy).pack(side="right", padx=6)
            self._refresh_list(0)

        @staticmethod
        def _hex(color: Tuple[float, float, float]) -> str:
            return "#" + "".join(f"{max(0, min(255, round(component * 255))):02X}" for component in color)

        def _refresh_list(self, index: int) -> None:
            self.template_list.delete(0, "end")
            for template in self.templates:
                suffix = " (built-in)" if template.template_id == WERKLOGGER_EXPORT_TEMPLATE.template_id else ""
                self.template_list.insert("end", template.display_name + suffix)
            self.template_list.selection_set(max(0, min(index, len(self.templates) - 1))); self._load(index)

        def _index(self) -> Optional[int]:
            selection = self.template_list.curselection()
            return selection[0] if selection else None

        def _select(self, _event: Any = None) -> None:
            index = self._index()
            if index is not None: self._load(index)

        def _load(self, index: int) -> None:
            template = self.templates[index]; built_in = template.template_id == WERKLOGGER_EXPORT_TEMPLATE.template_id
            self.readonly_note.config(text="Built-in compatibility template is read-only; duplicate it to customize." if built_in else "")
            self.name_var.set(template.display_name); self.title_var.set(str(template.branding["report_title"]))
            self.pattern_var.set(template.output_filename_pattern)
            self.fallback_var.set(template.empty_value_fallback)
            self.rows_var.set(str(template.photo_grid["rows"])); self.columns_var.set(str(template.photo_grid["columns"]))
            for key, variable in self.output_flag_vars.items(): variable.set(getattr(template, key))
            for key, variable in self.color_vars.items(): variable.set(self._hex(template.colors[key]))
            for key, variable in self.section_vars.items(): variable.set(key in template.enabled_sections)
            self.order_list.delete(0, "end")
            for section in template.section_order: self.order_list.insert("end", section)
            for key, variable in self.label_vars.items(): variable.set(template.field_labels[key])
            for key, variable in self.branding_vars.items(): variable.set(template.branding[key])
            choices = ("", *template.csv_headers)
            self.csv_schema_note.config(text=(f"{len(template.csv_headers)} columns loaded: " + ", ".join(template.csv_headers))
                                        if template.csv_headers else "No uploaded CSV schema; standard column names are used.")
            for key, variable in self.csv_role_vars.items():
                self.csv_role_boxes[key]["values"] = choices
                variable.set(template.csv_column_mapping.get(key, ""))
            for key, variable in self.pdf_column_vars.items():
                self.pdf_column_boxes[key]["values"] = choices
                variable.set(template.pdf_field_columns.get(key, ""))

        def _new(self) -> None:
            path = filedialog.askopenfilename(
                title="Select the CSV export for this template", filetypes=[("CSV files", "*.csv")], parent=self)
            if not path:
                return
            try:
                headers = inspect_csv_headers(Path(path))
            except (OSError, UnicodeError, ValueError) as exc:
                messagebox.showerror("Invalid CSV template", str(exc), parent=self); return
            data = export_template_to_dict(WERKLOGGER_EXPORT_TEMPLATE)
            data.update(template_id="custom-" + uuid.uuid4().hex, display_name="New template",
                        csv_headers=list(headers))
            folded = {header.casefold(): header for header in headers}
            data["csv_column_mapping"] = {
                role: folded[role.casefold()] for role in CSV_COLUMN_ROLES if role.casefold() in folded
            }
            self.templates.append(export_template_from_dict(data))
            self._refresh_list(len(self.templates) - 1)

        def _duplicate(self) -> None:
            index = self._index()
            if index is not None: self._append_copy(self.templates[index], self.templates[index].display_name + " copy")

        def _append_copy(self, source: ExportTemplate, name: str) -> None:
            data = export_template_to_dict(source); data["template_id"] = "custom-" + uuid.uuid4().hex
            data["display_name"] = name; self.templates.append(export_template_from_dict(data))
            self._refresh_list(len(self.templates) - 1)

        def _rename(self) -> None:
            index = self._index()
            if index is None: return
            if index == 0:
                messagebox.showinfo("Read-only", "Duplicate the built-in template before renaming it.", parent=self); return
            name = self.name_var.get().strip()
            if not name:
                messagebox.showerror("Invalid name", "Enter the new name in Template name first.", parent=self); return
            self._save()

        def _delete(self) -> None:
            index = self._index()
            if index is None: return
            if self.templates[index].template_id == WERKLOGGER_EXPORT_TEMPLATE.template_id:
                messagebox.showinfo("Read-only", "The built-in compatibility template cannot be deleted.", parent=self); return
            if messagebox.askyesno("Delete template", f"Delete {self.templates[index].display_name}?", parent=self):
                del self.templates[index]; save_export_templates(self.templates); self.app.set_export_templates(self.templates)
                self._refresh_list(max(0, index - 1))

        def _move(self, delta: int) -> None:
            selection = self.order_list.curselection()
            if not selection: return
            old = selection[0]; new = old + delta
            if not 0 <= new < self.order_list.size(): return
            value = self.order_list.get(old); self.order_list.delete(old); self.order_list.insert(new, value)
            self.order_list.selection_set(new)

        def _save(self) -> None:
            index = self._index()
            if index is None: return
            current = self.templates[index]
            if current.template_id == WERKLOGGER_EXPORT_TEMPLATE.template_id:
                messagebox.showinfo("Read-only", "Duplicate the built-in template to customize it.", parent=self); return
            order = list(self.order_list.get(0, "end"))
            enabled = [key for key in order if self.section_vars[key].get()]
            errors = validate_template_values(self.name_var.get(), self.title_var.get(), self.pattern_var.get(),
                                              {k: v.get() for k, v in self.color_vars.items()}, enabled,
                                              self.rows_var.get(), self.columns_var.get(),
                                              current.page_settings, current.photo_grid)
            if any(not variable.get().strip() for variable in self.label_vars.values()):
                errors.append("All display labels are required.")
            if any(not variable.get().strip() for variable in self.branding_vars.values()):
                errors.append("All report text values are required.")
            if "{title}" not in self.branding_vars["section_continuation_pattern"].get():
                errors.append("Section continuation pattern must contain {title}.")
            if any(t.display_name.casefold() == self.name_var.get().strip().casefold() and i != index
                   for i, t in enumerate(self.templates)):
                errors.append("Template names must be unique.")
            if current.csv_headers:
                available = set(current.csv_headers)
                missing_roles = [role for role in REQUIRED_COLUMNS
                                 if self.csv_role_vars[role].get() not in available and role not in available]
                if missing_roles:
                    errors.append("Map the required CSV purposes: " + ", ".join(missing_roles) + ".")
            if errors:
                messagebox.showerror("Cannot save template", "\n".join(errors), parent=self); return
            data = export_template_to_dict(current)
            data.update(display_name=self.name_var.get().strip(), output_filename_pattern=self.pattern_var.get().strip(),
                        empty_value_fallback=self.fallback_var.get(), enabled_sections=enabled, section_order=order)
            data.update({key: variable.get() for key, variable in self.output_flag_vars.items()})
            data["branding"]["report_title"] = self.title_var.get().strip()
            data["branding"].update({key: variable.get().strip()
                                     for key, variable in self.branding_vars.items()})
            data["colors"] = {key: [int(value.get().strip()[i:i+2], 16) / 255 for i in (1, 3, 5)] for key, value in self.color_vars.items()}
            data["photo_grid"].update(rows=int(self.rows_var.get()), columns=int(self.columns_var.get()))
            data["field_labels"] = {key: variable.get().strip() for key, variable in self.label_vars.items()}
            data["csv_column_mapping"] = {key: variable.get() for key, variable in self.csv_role_vars.items()
                                          if variable.get()}
            data["pdf_field_columns"] = {key: variable.get() for key, variable in self.pdf_column_vars.items()
                                         if variable.get()}
            self.templates[index] = export_template_from_dict(data)
            try: save_export_templates(self.templates)
            except ExportTemplateConfigurationError as exc:
                messagebox.showerror("Cannot save template", str(exc), parent=self); return
            self.app.set_export_templates(self.templates, self.templates[index].template_id)
            self._refresh_list(index)
            messagebox.showinfo("Template saved", "The export template is ready to use.", parent=self)

    class App(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("SSV ZIP Processor (Images + PDF)")
            self.geometry("980x640")
    
            self.zip_path: Optional[Path] = None
            self.out_dir: Optional[Path] = None
            try:
                self.export_templates = load_export_templates()
            except ExportTemplateConfigurationError as exc:
                if exc.recovered_templates is not None:
                    self.export_templates = exc.recovered_templates
                    messagebox.showwarning("Export templates recovered", str(exc), parent=self)
                else:
                    self.export_templates = [WERKLOGGER_EXPORT_TEMPLATE]
                    messagebox.showerror("Export-template configuration error", str(exc), parent=self)
    
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

            try:
                self.projects = load_projects()
            except ProjectConfigurationError as exc:
                self.projects = list(DEFAULT_PROJECT_RECORDS)
                messagebox.showerror("Project configuration error", str(exc), parent=self)
            self.project_var = tk.StringVar(value="")
            self.cmb_project = ttk.Combobox(
                proj_row,
                textvariable=self.project_var,
                values=([''] + [project.display_name for project in self.projects]),
                state="readonly",
                width=32,
            )
            self.cmb_project.pack(side="left", padx=(0, 10))
            self.cmb_project.bind("<<ComboboxSelected>>", self.on_project_selected)

            self.new_project_var = tk.StringVar()
            tk.Entry(proj_row, textvariable=self.new_project_var, width=26).pack(side="left")
            tk.Button(proj_row, text="Add project", command=self.add_project).pack(side="left", padx=(8, 0))

            template_row = tk.Frame(frm); template_row.pack(fill="x", pady=(8, 0))
            tk.Label(template_row, text="Export template:", width=22, anchor="w").pack(side="left")
            self.template_var = tk.StringVar()
            self.cmb_template = ttk.Combobox(template_row, textvariable=self.template_var, state="readonly", width=32)
            self.cmb_template.pack(side="left", padx=(0, 10))
            tk.Button(template_row, text="Save as project default",
                      command=self.save_project_default).pack(side="left", padx=(0, 10))
            tk.Button(template_row, text="Manage templates...", command=self.manage_templates).pack(side="left")
            self.set_export_templates(self.export_templates)
    
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
            existing = next((project for project in self.projects if project.display_name == name), None)
            if existing is None:
                existing = ProjectRecord("project-" + uuid.uuid4().hex, name,
                                         WERKLOGGER_EXPORT_TEMPLATE.template_id)
                self.projects.append(existing)
                self.cmb_project["values"] = ([""] + [project.display_name for project in self.projects])
                try:
                    save_projects(self.projects)
                except ProjectConfigurationError as exc:
                    self.projects.remove(existing)
                    self.cmb_project["values"] = ([""] + [project.display_name for project in self.projects])
                    messagebox.showerror("Project configuration error", str(exc), parent=self)
                    self.log(f"Could not save project: {exc}")
                    return
            self.project_var.set(name)
            self.on_project_selected()
            self.new_project_var.set("")
            self.log(f"Project added/selected: {name}")

        def selected_project(self) -> Optional[ProjectRecord]:
            return next((project for project in self.projects
                         if project.display_name == self.project_var.get()), None)

        def on_project_selected(self, _event: Any = None) -> None:
            """Apply the selected project's default; later template choices remain one-off."""
            project = self.selected_project()
            if project is None:
                return
            template = project_default_template(project, self.export_templates)
            self.template_var.set(template.display_name)
            if template.template_id != project.default_template_id:
                self.log(f"Project {project.display_name}'s default template is missing; "
                         "using the built-in template. Save a new default to repair it.")

        def save_project_default(self) -> None:
            project = self.selected_project()
            if project is None:
                messagebox.showinfo("Select project", "Select a project first.", parent=self)
                return
            updated = ProjectRecord(project.project_id, project.display_name,
                                    self.selected_template().template_id)
            index = self.projects.index(project)
            self.projects[index] = updated
            try:
                save_projects(self.projects)
            except ProjectConfigurationError as exc:
                self.projects[index] = project
                messagebox.showerror("Project configuration error", str(exc), parent=self)
                return
            self.log(f"Saved {self.selected_template().display_name} as the default for {project.display_name}.")

        def set_export_templates(self, templates: List[ExportTemplate], selected_id: Optional[str] = None) -> None:
            """Refresh template choices immediately after a management operation."""
            old_id = selected_id
            if old_id is None and hasattr(self, "cmb_template"):
                current = self.template_var.get()
                old_id = next((t.template_id for t in self.export_templates if t.display_name == current), None)
            self.export_templates = list(templates)
            self.cmb_template["values"] = [t.display_name for t in self.export_templates]
            selected = next((t for t in self.export_templates if t.template_id == old_id), self.export_templates[0])
            self.template_var.set(selected.display_name)

        def manage_templates(self) -> None:
            TemplateManagerDialog(self)

        def selected_template(self) -> ExportTemplate:
            return next((t for t in self.export_templates if t.display_name == self.template_var.get()),
                        WERKLOGGER_EXPORT_TEMPLATE)

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
                res = process_zip_to_folder_and_pdf(self.zip_path, self.out_dir, project_override=self.project_var.get(),
                                                    log=self.log, template=self.selected_template())
                self.log(f"Done. Images written: {res.written_images}")
                files = "\n".join(f"• {path.name}" for path in res.generated_artifacts)
                messagebox.showinfo("Success", f"Finished. Files produced:\n{files}")
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
    ap.add_argument(
        "--export-template", metavar="TEMPLATE_ID",
        help="Use the persisted export template with this stable ID",
    )
    ap.add_argument(
        "--list-export-templates", action="store_true",
        help="List available export-template IDs and display names, then exit",
    )
    args = ap.parse_args()

    if args.list_export_templates:
        try:
            templates = load_export_templates()
        except ExportTemplateConfigurationError as exc:
            ap.error(str(exc))
        for template in templates:
            print(f"{template.template_id}\t{template.display_name}")
        return

    if args.zip_path and args.out_dir:
        template = None
        if args.export_template is not None:
            try:
                templates = load_export_templates()
            except ExportTemplateConfigurationError as exc:
                ap.error(str(exc))
            template = next(
                (item for item in templates if item.template_id == args.export_template),
                None,
            )
            if template is None:
                ap.error(
                    f"export template ID {args.export_template!r} does not exist; "
                    "use --list-export-templates to see available IDs"
                )
        res = process_zip_to_folder_and_pdf(
            Path(args.zip_path), Path(args.out_dir), project_override=args.project,
            log=print, template=template,
        )
        print("Files produced:")
        for artifact in res.generated_artifacts:
            print(artifact)
        return

    if args.export_template is not None:
        ap.error("--export-template requires both --zip and --out")

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

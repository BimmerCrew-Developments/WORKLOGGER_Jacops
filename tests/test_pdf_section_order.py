from dataclasses import replace

from pypdf import PdfReader

from ssv_zip_processor_gui_v22 import (
    ReportRow,
    WERKLOGGER_EXPORT_TEMPLATE,
    build_pdf,
)


def _report() -> ReportRow:
    return ReportRow(
        report_datetime="02-08-2026 12:00",
        naam_onderaannemer="Testbedrijf",
        project_locatie_naam="Testproject",
        building_id="TEST-1",
        adres="Teststraat 1",
        postcode_stad="1234 AB Teststad",
        contactpersoon="Testpersoon",
        quadrant="Noord",
        duct_kleur="Rood",
        units_gelast="2",
        gebruikte_materialen_lines=["Materiaal A"],
        post_afmeldingen_lines=["Afmelding A"],
        photos=[],
    )


def _render_text(tmp_path, order):
    template = replace(
        WERKLOGGER_EXPORT_TEMPLATE,
        template_id="custom-" + "-".join(order),
        section_order=order,
    )
    output = tmp_path / (template.template_id + ".pdf")
    build_pdf(output, _report(), template)
    return "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)


def _assert_titles_in_order(text, titles):
    positions = [text.index(title) for title in titles]
    assert positions == sorted(positions)


def test_custom_section_order_is_reflected_in_extracted_pdf_text(tmp_path):
    first_order = (
        "address", "lmra", "work_details", "materials",
        "post_registrations", "photos",
    )
    second_order = (
        "materials", "work_details", "address", "lmra",
        "post_registrations", "photos",
    )

    first_text = _render_text(tmp_path, first_order)
    second_text = _render_text(tmp_path, second_order)

    _assert_titles_in_order(
        first_text,
        ["Adresgegevens:", "Uitgevoerde Werken - Details:", "Gebruikte materialen:"],
    )
    _assert_titles_in_order(
        second_text,
        ["Gebruikte materialen:", "Uitgevoerde Werken - Details:", "Adresgegevens:"],
    )


def test_custom_headings_and_lmra_copy_are_present_in_extracted_pdf_text(tmp_path):
    branding = {
        **WERKLOGGER_EXPORT_TEMPLATE.branding,
        "section_address": "Projectadres:",
        "section_lmra": "Veiligheidscontrole:",
        "lmra_status": "Controle akkoord - werkzaamheden toegestaan",
        "lmra_item_1": "Aangepast veiligheidsitem",
        "yes_label": "GOED",
    }
    template = replace(
        WERKLOGGER_EXPORT_TEMPLATE,
        template_id="custom-report-copy",
        branding=branding,
    )
    output = tmp_path / "custom-report-copy.pdf"

    build_pdf(output, _report(), template)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)

    assert "Projectadres:" in text
    assert "Veiligheidscontrole:" in text
    assert "Controle akkoord - werkzaamheden toegestaan" in text
    assert "Aangepast veiligheidsitem" in text
    assert "GOED" in text

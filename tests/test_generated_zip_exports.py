import zipfile
from dataclasses import replace

import pytest
from pypdf import PdfReader

import ssv_zip_processor_gui_v22 as app


def pdf_text(path):
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _template(**changes):
    return replace(
        app.WERKLOGGER_EXPORT_TEMPLATE,
        template_id="generated-fixture-template",
        output_filename_pattern="{building_id}-{project_name}.pdf",
        **changes,
    )


def test_generated_zip_produces_verifiable_report_and_artifacts(tmp_path, audit_zip):
    output = tmp_path / "output"

    result = app.process_zip_to_folder_and_pdf(audit_zip, output, template=_template())

    text = pdf_text(result.pdf_path)
    for expected in (
        "WERKLOGGER RAPPORT", "Adresgegevens:", "Building ID:", "BLDG-42",
        "Project/Locatie Naam:", "Fixture Project", "Gebruikte materialen:",
        "2x Kabel", "Post Afmeldingen:", "3x Las geregistreerd", "Foto's:",
        "Inspectiefoto (2)",
    ):
        assert expected in text
    assert result.written_images == 2
    assert result.copied_pdfs == 1
    assert result.output_zip_path is not None
    assert {path.suffix.lower() for path in result.generated_artifacts} >= {".pdf", ".jpeg", ".zip"}
    with zipfile.ZipFile(result.output_zip_path) as bundle:
        assert set(bundle.namelist()) == {path.name for path in result.generated_artifacts[:-1]}


def test_section_enablement_and_order_are_visible_in_generated_pdf(tmp_path, audit_zip):
    template = _template(
        enabled_sections=("materials", "address", "work_details"),
        section_order=("materials", "address", "work_details"),
        include_photo_pages=False,
        create_output_zip=False,
    )

    text = pdf_text(app.process_zip_to_folder_and_pdf(audit_zip, tmp_path / "out", template=template).pdf_path)

    headings = ["Gebruikte materialen:", "Adresgegevens:", "Uitgevoerde Werken - Details:"]
    assert [text.index(heading) for heading in headings] == sorted(text.index(heading) for heading in headings)
    assert "LMRA Checklist:" not in text
    assert "Foto's:" not in text


def test_output_flags_disable_optional_artifacts(tmp_path, audit_zip):
    template = _template(
        include_photo_pages=False,
        include_loose_images=False,
        include_pdf_attachments=False,
        create_output_zip=False,
    )

    result = app.process_zip_to_folder_and_pdf(audit_zip, tmp_path / "out", template=template)

    assert result.generated_artifacts == [result.pdf_path]
    assert result.written_images == result.copied_pdfs == 0
    assert result.output_zip_path is None
    assert "Foto's:" not in pdf_text(result.pdf_path)


def test_preexisting_output_is_rejected_before_any_artifact_is_written(tmp_path, audit_zip):
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "BLDG-42-Fixture_Project.pdf"
    existing.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        app.process_zip_to_folder_and_pdf(audit_zip, output, template=_template())

    assert existing.read_bytes() == b"keep me"
    assert list(output.iterdir()) == [existing]


def test_duplicate_attachment_artifact_names_are_rejected(tmp_path, audit_zip):
    duplicate_zip = tmp_path / "duplicates.zip"
    with zipfile.ZipFile(audit_zip) as source, zipfile.ZipFile(duplicate_zip, "w") as target:
        for item in source.infolist():
            target.writestr(item.filename, source.read(item))
        attachment = source.read("export/inspection-attachment.pdf")
        target.writestr("another/inspection-attachment.pdf", attachment)

    output = tmp_path / "out"
    with pytest.raises(FileExistsError, match="collision"):
        app.process_zip_to_folder_and_pdf(duplicate_zip, output, template=_template())

    assert not any(output.iterdir())

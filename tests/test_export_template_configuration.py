import json
from dataclasses import replace

import pytest

import ssv_zip_processor_gui_v22 as app


def _custom(template_id="custom-template", name="Custom"):
    return replace(app.WERKLOGGER_EXPORT_TEMPLATE, template_id=template_id, display_name=name)


def _legacy_document(template):
    return json.dumps([app.export_template_to_dict(template)])


def test_adjacent_templates_are_migrated_to_per_user_config(tmp_path, monkeypatch):
    legacy = tmp_path / "installation" / "export_templates.json"
    legacy.parent.mkdir()
    custom = _custom()
    legacy.write_text(_legacy_document(custom), encoding="utf-8")
    destination = tmp_path / "user" / "export_templates.json"
    monkeypatch.setattr(app, "_adjacent_export_template_path", lambda: legacy)
    monkeypatch.setattr(app, "export_template_config_path", lambda: destination)

    loaded = app.load_export_templates()

    assert [template.template_id for template in loaded] == [
        app.WERKLOGGER_EXPORT_TEMPLATE.template_id, custom.template_id
    ]
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored["version"] == app.EXPORT_TEMPLATE_CONFIG_VERSION
    assert stored["templates"][0]["template_id"] == custom.template_id
    assert legacy.exists()


def test_migration_never_overwrites_existing_user_file(tmp_path, monkeypatch):
    legacy = tmp_path / "export_templates.json"
    legacy.write_text(_legacy_document(_custom("legacy")), encoding="utf-8")
    destination = tmp_path / "user.json"
    app.save_export_templates([_custom("user")], destination)
    monkeypatch.setattr(app, "_adjacent_export_template_path", lambda: legacy)
    monkeypatch.setattr(app, "export_template_config_path", lambda: destination)

    loaded = app.load_export_templates()

    assert {item.template_id for item in loaded} == {
        app.WERKLOGGER_EXPORT_TEMPLATE.template_id, "user"
    }


def test_unwritable_template_storage_raises_typed_error(tmp_path, monkeypatch):
    destination = tmp_path / "templates.json"

    def deny_write(path, content):
        raise PermissionError("read-only storage")

    monkeypatch.setattr(app, "_atomic_write", deny_write)
    with pytest.raises(app.ExportTemplateConfigurationError, match="read-only storage"):
        app.save_export_templates([_custom()], destination)


def test_malformed_json_raises_typed_error(tmp_path):
    destination = tmp_path / "templates.json"
    destination.write_text("{not json", encoding="utf-8")

    with pytest.raises(app.ExportTemplateConfigurationError, match="Could not read"):
        app.load_export_templates(destination)


def test_unsupported_configuration_version_is_rejected(tmp_path):
    destination = tmp_path / "templates.json"
    destination.write_text(json.dumps({"version": 99, "templates": []}), encoding="utf-8")

    with pytest.raises(app.ExportTemplateConfigurationError, match="Unsupported.*99"):
        app.load_export_templates(destination)


def test_old_template_branding_receives_new_report_text_defaults(tmp_path):
    custom = _custom()
    document = app.export_template_to_dict(custom)
    document["branding"] = {"report_title": "Oud rapport", "photo_title": "Foto's"}
    destination = tmp_path / "templates.json"
    destination.write_text(json.dumps({"version": 1, "templates": [document]}), encoding="utf-8")

    loaded = app.load_export_templates(destination)[1]

    assert loaded.branding["section_lmra"] == "LMRA Checklist:"
    assert loaded.branding["lmra_status"] == "LMRA Status: OK - Werk kan worden uitgevoerd"
    assert loaded.branding["photo_title"] == "Foto's:"


def test_required_report_text_is_validated_on_load_and_save(tmp_path):
    invalid = replace(_custom(), branding={**app.DEFAULT_TEMPLATE_BRANDING, "lmra_status": ""})

    with pytest.raises(app.ExportTemplateConfigurationError, match="invalid template"):
        app.save_export_templates([invalid], tmp_path / "templates.json")

    document = app.export_template_to_dict(_custom())
    document["branding"]["section_address"] = ""
    destination = tmp_path / "invalid-load.json"
    destination.write_text(json.dumps({"version": 1, "templates": [document]}), encoding="utf-8")
    with pytest.raises(app.ExportTemplateConfigurationError, match="Could not read"):
        app.load_export_templates(destination)


def test_template_serialization_round_trips_all_output_and_layout_settings():
    original = replace(
        _custom(), enabled_sections=("photos", "address"),
        section_order=("photos", "address"), include_photo_pages=False,
        include_loose_images=False, include_pdf_attachments=False,
        create_output_zip=False, empty_value_fallback="--",
        photo_grid={**app.WERKLOGGER_EXPORT_TEMPLATE.photo_grid, "rows": 2, "columns": 1},
    )

    restored = app.export_template_from_dict(app.export_template_to_dict(original))

    assert restored == original


def test_template_json_storage_preserves_pagesize_tuple(tmp_path):
    original = _custom()
    destination = tmp_path / "templates.json"

    app.save_export_templates([original], destination)
    restored = app.load_export_templates(destination)[1]

    assert restored == original
    assert isinstance(restored.page_settings["pagesize"], tuple)


def test_uploaded_csv_schema_and_column_mappings_round_trip():
    original = replace(
        _custom(), csv_headers=("Record", "Kind", "Caption", "Answer", "Files"),
        csv_column_mapping={"ID": "Record", "Type": "Kind", "Label": "Caption",
                            "Primary": "Answer", "Media": "Files"},
        pdf_field_columns={"building_id": "Record", "project_name": "Answer"},
        csv_header_row=2, csv_column_count=5,
    )

    restored = app.export_template_from_dict(app.export_template_to_dict(original))

    assert restored.csv_headers == original.csv_headers
    assert restored.csv_column_mapping == original.csv_column_mapping
    assert restored.pdf_field_columns == original.pdf_field_columns
    assert restored.csv_header_row == 2
    assert restored.csv_column_count == 5


def test_configured_csv_range_is_used_for_parsing_and_direct_values(tmp_path):
    source = tmp_path / "custom-range.csv"
    source.write_text(
        "Export metadata\n"
        "Generated by test\n"
        "Record,Kind,Caption,Answer,Extra,Comment,Files,Ignored\n"
        "1,text,Building,BLD-42,,,,do-not-import\n",
        encoding="utf-8",
    )
    mapping = {"ID": "Record", "Type": "Kind", "Label": "Caption",
               "Primary": "Answer", "Secondary": "Extra", "Note": "Comment",
               "Media": "Files"}

    meta, fields, media, rows = app.load_audit_csv(source, mapping, 2, 7)
    header_idx = app.find_header_row(source, mapping, 2, 7)
    direct_values = app.read_first_csv_values(source, header_idx, 7)

    assert meta == {}
    assert fields == {"Building": "BLD-42"}
    assert media == []
    assert rows[0].row_id == "1"
    assert direct_values["Answer"] == "BLD-42"
    assert "Ignored" not in direct_values


def test_custom_csv_column_names_are_used_for_import_and_direct_values(tmp_path):
    source = tmp_path / "custom.csv"
    source.write_text(
        "Record,Kind,Caption,Answer,Extra,Comment,Files\n"
        "1,text,Building,BLD-42,,,\n"
        "2,media,Overview,,,,photo-1\n",
        encoding="utf-8",
    )
    mapping = {"ID": "Record", "Type": "Kind", "Label": "Caption",
               "Primary": "Answer", "Secondary": "Extra", "Note": "Comment",
               "Media": "Files"}

    meta, fields, media, rows = app.load_audit_csv(source, mapping)
    header_idx = app.find_header_row(source, mapping)

    assert meta == {}
    assert fields["Building"] == "BLD-42"
    assert media[0].media_ids == ["photo-1"]
    assert rows[0].row_id == "1"
    assert app.read_first_csv_values(source, header_idx)["Answer"] == "BLD-42"


@pytest.mark.parametrize(
    "pattern",
    ["{building_id.__class__}.pdf", "{project_name[0]}.pdf", "{building_id!r}.pdf",
     "{report_datetime:20}.pdf", "{unknown}.pdf", "{}.pdf"],
)
def test_placeholder_substitution_rejects_expression_and_formatting_syntax(pattern):
    with pytest.raises(ValueError):
        app.substitute_export_placeholders(
            pattern, {key: "safe" for key in app.FILENAME_PLACEHOLDERS},
            app.FILENAME_PLACEHOLDERS,
        )


def test_placeholder_substitution_only_inserts_allowlisted_literal_values():
    rendered = app.substitute_export_placeholders(
        "{{report}}-{building_id}-{project_name}",
        {"building_id": "B-1", "project_name": "Project", "report_datetime": "ignored"},
        app.FILENAME_PLACEHOLDERS,
    )

    assert rendered == "{report}-B-1-Project"


def test_valid_backup_recovers_and_preserves_damaged_document(tmp_path):
    destination = tmp_path / "templates.json"
    first = _custom("first", "First")
    app.save_export_templates([first], destination)
    app.save_export_templates([first, _custom("second", "Second")], destination)
    damaged_content = "{damaged"
    destination.write_text(damaged_content, encoding="utf-8")

    with pytest.raises(app.ExportTemplateConfigurationError, match="recovered") as caught:
        app.load_export_templates(destination)

    assert caught.value.recovered_templates is not None
    assert {item.template_id for item in caught.value.recovered_templates} == {
        app.WERKLOGGER_EXPORT_TEMPLATE.template_id, first.template_id
    }
    assert app.load_export_templates(destination)[1].template_id == first.template_id
    damaged_files = list(tmp_path.glob("templates.json.damaged-*"))
    assert len(damaged_files) == 1
    assert damaged_files[0].read_text(encoding="utf-8") == damaged_content

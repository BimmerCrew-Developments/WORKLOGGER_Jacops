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

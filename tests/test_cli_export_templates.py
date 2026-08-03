import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ssv_zip_processor_gui_v22 as app


def _run_export(monkeypatch, template_id, config_path, captured):
    monkeypatch.setattr(app, "export_template_config_path", lambda: config_path)
    monkeypatch.setattr(
        app,
        "process_zip_to_folder_and_pdf",
        lambda *args, **kwargs: captured.update(template=kwargs["template"])
        or SimpleNamespace(generated_artifacts=[]),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["worklogger", "--zip", "input.zip", "--out", "output", "--export-template", template_id],
    )
    app.main()


def test_cli_uses_built_in_export_template(tmp_path, monkeypatch):
    captured = {}

    _run_export(
        monkeypatch, app.WERKLOGGER_EXPORT_TEMPLATE.template_id,
        tmp_path / "missing.json", captured,
    )

    assert captured["template"] is app.WERKLOGGER_EXPORT_TEMPLATE


def test_cli_loads_custom_export_template_by_id(tmp_path, monkeypatch):
    config_path = tmp_path / "export_templates.json"
    custom = replace(
        app.WERKLOGGER_EXPORT_TEMPLATE,
        template_id="stable-custom-id",
        display_name="Editable display name",
    )
    app.save_export_templates([custom], config_path)
    captured = {}

    _run_export(monkeypatch, custom.template_id, config_path, captured)

    assert captured["template"] == custom


def test_cli_rejects_unknown_template_id_even_when_it_matches_a_name(
        tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "export_templates.json"
    custom = replace(
        app.WERKLOGGER_EXPORT_TEMPLATE,
        template_id="stable-custom-id",
        display_name="Editable display name",
    )
    app.save_export_templates([custom], config_path)
    monkeypatch.setattr(app, "export_template_config_path", lambda: config_path)
    monkeypatch.setattr(
        sys, "argv",
        ["worklogger", "--zip", "input.zip", "--out", "output",
         "--export-template", custom.display_name],
    )

    with pytest.raises(SystemExit) as caught:
        app.main()

    assert caught.value.code == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_lists_available_template_ids_and_names(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "export_templates.json"
    custom = replace(
        app.WERKLOGGER_EXPORT_TEMPLATE,
        template_id="stable-custom-id",
        display_name="Editable display name",
    )
    app.save_export_templates([custom], config_path)
    monkeypatch.setattr(app, "export_template_config_path", lambda: config_path)
    monkeypatch.setattr(sys, "argv", ["worklogger", "--list-export-templates"])

    app.main()

    assert capsys.readouterr().out.splitlines() == [
        "werklogger-report-v1\tWERKLOGGER RAPPORT",
        "stable-custom-id\tEditable display name",
    ]

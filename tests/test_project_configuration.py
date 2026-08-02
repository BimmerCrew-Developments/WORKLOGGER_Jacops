import json
from dataclasses import replace

import ssv_zip_processor_gui_v22 as app


def _custom_template(template_id="custom-template"):
    return replace(app.WERKLOGGER_EXPORT_TEMPLATE, template_id=template_id, display_name="Custom")


def test_version_one_projects_are_migrated_without_loss(tmp_path, monkeypatch):
    path = tmp_path / "projects.json"
    path.write_text(json.dumps({"version": 1, "projects": ["Existing A", "Existing B"]}), encoding="utf-8")
    monkeypatch.setattr(app, "project_config_path", lambda: path)

    projects = app.load_projects()

    assert {project.display_name for project in projects} >= {"Existing A", "Existing B"}
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["version"] == 2
    migrated_names = {project["display_name"] for project in migrated["projects"]}
    assert {"Existing A", "Existing B"}.issubset(migrated_names)
    assert all({"project_id", "display_name", "default_template_id"} <= project.keys()
               for project in migrated["projects"])


def test_missing_project_template_falls_back_to_built_in():
    project = app.ProjectRecord("one", "One", "deleted-template")

    selected = app.project_default_template(project, [app.WERKLOGGER_EXPORT_TEMPLATE, _custom_template()])

    assert selected is app.WERKLOGGER_EXPORT_TEMPLATE
    # The missing ID remains available so the GUI can explicitly repair it.
    assert project.default_template_id == "deleted-template"


def test_project_selection_resolves_each_projects_default():
    custom = _custom_template()
    projects = [
        app.ProjectRecord("one", "One", custom.template_id),
        app.ProjectRecord("two", "Two", app.WERKLOGGER_EXPORT_TEMPLATE.template_id),
    ]

    assert app.project_default_template(projects[0], [app.WERKLOGGER_EXPORT_TEMPLATE, custom]) is custom
    assert app.project_default_template(projects[1], [app.WERKLOGGER_EXPORT_TEMPLATE, custom]) is app.WERKLOGGER_EXPORT_TEMPLATE


def test_version_two_default_template_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "projects.json"
    monkeypatch.setattr(app, "project_config_path", lambda: path)
    custom = _custom_template()
    record = app.ProjectRecord("stable-id", "Project", custom.template_id)

    app.save_projects([record])

    loaded = app.load_projects()
    assert record in loaded

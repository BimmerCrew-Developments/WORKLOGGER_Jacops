from dataclasses import replace

import pytest

import ssv_zip_processor_gui_v22 as app


class RecordingCanvas:
    def __init__(self):
        self.images = []

    def drawImage(self, _image, x, y, width, height, **_options):
        self.images.append((x, y, x + width, y + height))

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


@pytest.mark.parametrize("rows,columns", [(1, 1), (2, 2), (3, 3), (app.MAX_PHOTO_ROWS, app.MAX_PHOTO_COLUMNS)])
def test_rendered_photo_boxes_stay_inside_printable_page(tmp_path, rows, columns):
    """Exercise every resolved render slot at representative and maximum counts."""
    grid = dict(app.WERKLOGGER_EXPORT_TEMPLATE.photo_grid, rows=rows, columns=columns)
    template = replace(app.WERKLOGGER_EXPORT_TEMPLATE, photo_grid=grid)
    image_path = tmp_path / "photo.png"
    app.Image.new("RGB", (20, 20), "white").save(image_path)
    photo_count = rows * columns
    report = app.ReportRow(
        "", "", "", "", "", "", "", "", "", "", [], [],
        [app.Photo("Inspection", image_path) for _ in range(photo_count)],
    )
    canvas = RecordingCanvas()

    app.render_photos_pages(canvas, report, template)

    page_width, page_height = template.page_settings["pagesize"]
    printable_bottom = template.page_settings["bottom_y"]
    printable_top = page_height - grid["top_margin"]
    assert len(canvas.images) == photo_count
    for left, bottom, right, top in canvas.images:
        assert 0 <= left < right <= page_width
        assert printable_bottom <= bottom < top <= printable_top


@pytest.mark.parametrize(
    "rows,columns",
    [(app.MAX_PHOTO_ROWS + 1, 1), (1, app.MAX_PHOTO_COLUMNS + 1)],
)
def test_template_editor_validation_rejects_unsupported_photo_counts(rows, columns):
    errors = app.validate_template_values(
        "Template", "Report", "{building_id}.pdf", {"primary": "#000000"},
        ["photos"], str(rows), str(columns),
        app.WERKLOGGER_EXPORT_TEMPLATE.page_settings,
        app.WERKLOGGER_EXPORT_TEMPLATE.photo_grid,
    )

    assert any("Photo grid supports" in error for error in errors)


def test_geometry_rejects_cells_below_minimum_printed_size():
    narrow_grid = dict(app.WERKLOGGER_EXPORT_TEMPLATE.photo_grid, right=150, columns=2)

    with pytest.raises(ValueError, match="each image must be at least"):
        app.calculate_photo_grid_geometry(app.WERKLOGGER_EXPORT_TEMPLATE.page_settings, narrow_grid)

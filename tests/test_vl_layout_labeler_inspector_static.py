from pathlib import Path


def test_layout_inspector_exposes_crop_and_accessible_separators():
    html = Path("vl_layout_labeler/static/index.html").read_text(encoding="utf-8")
    script = Path("vl_layout_labeler/static/inspector_ui.mjs").read_text(
        encoding="utf-8"
    )
    stylesheet = Path("vl_layout_labeler/static/inspector.css").read_text(
        encoding="utf-8"
    )

    for element_id in (
        "crop-panel",
        "crop-viewport",
        "crop-image",
        "crop-fit",
        "crop-actual",
        "editor-focus-toggle",
        "inspector-separator",
        "crop-editor-separator",
    ):
        assert f'id="{element_id}"' in html

    assert 'role="separator"' in html
    assert 'aria-orientation="vertical"' in html
    assert 'aria-orientation="horizontal"' in html
    assert 'aria-valuemin="340"' in html
    assert 'aria-valuemin="130"' in html
    assert 'src="/static/inspector_ui.mjs"' in html
    assert "computeCropTransform" in script
    assert "setPointerCapture" in script
    assert "releasePointerCapture" in script
    assert 'event.key === "ArrowLeft"' in script
    assert 'event.key === "ArrowDown"' in script
    assert "window.localStorage" in script
    assert "setEditorFocus" in script
    assert 'event.key === "Escape"' in script
    assert "--inspector-width" in stylesheet
    assert "--crop-height" in stylesheet
    assert ".editor-scroll" in stylesheet
    assert ".inspector.editor-focus" in stylesheet


def test_secondary_block_and_export_controls_are_collapsed_by_default():
    html = Path("vl_layout_labeler/static/index.html").read_text(encoding="utf-8")

    assert '<details class="block-options">' in html
    assert '<details class="export-box">' in html
    assert '<summary>Tùy chọn block</summary>' in html
    assert '<summary>Xuất dữ liệu</summary>' in html
    assert '<details class="block-options" open>' not in html
    assert '<details class="export-box" open>' not in html


def test_layout_inspector_keeps_image_endpoint_and_codec_contracts_unchanged():
    html = Path("vl_layout_labeler/static/index.html").read_text(encoding="utf-8")
    app_script = Path("vl_layout_labeler/static/app.mjs").read_text(encoding="utf-8")
    inspector_script = Path("vl_layout_labeler/static/inspector_ui.mjs").read_text(
        encoding="utf-8"
    )

    assert 'from "./target_codec.mjs"' in app_script
    assert "inspectTarget(block.task, block.text)" in app_script
    assert "/api/images/${id}/content" in app_script
    assert 'id="text"' in html
    assert 'id="visual-editor"' in html
    assert 'id="layout-only-note"' in html
    assert "annotation" not in inspector_script
    assert "fetch(" not in inspector_script

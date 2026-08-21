from pathlib import Path


def test_table_editor_uses_one_context_toolbar_instead_of_actions_per_cell():
    script = Path("vl_layout_labeler/static/app.mjs").read_text(encoding="utf-8")
    stylesheet = Path("vl_layout_labeler/static/inspector.css").read_text(
        encoding="utf-8"
    )

    assert 'element("div", "table-toolbar")' in script
    assert 'table-cell-actions' in script
    assert 'element("span", "active-cell-label"' in script
    assert 'holder.classList.add("active-cell")' in script
    assert 'element("div", "cell-actions")' not in script
    assert "table-task-mode" in stylesheet
    assert ".table-workspace-scroll" in stylesheet
    assert ".compact-table-editor td.active-cell" in stylesheet


def test_table_compact_mode_keeps_all_existing_structure_operations():
    script = Path("vl_layout_labeler/static/app.mjs").read_text(encoding="utf-8")

    for action in (
        'targetButton("Gộp →"',
        'targetButton("Gộp ↓"',
        'targetButton("Tách"',
        'targetButton("+ Hàng"',
        'targetButton("+ Cột"',
        'targetButton("− Hàng"',
        'targetButton("− Cột"',
    ):
        assert action in script

    assert 'commitVisualModel("table", model)' in script
    assert 'commitVisualModel("table", next, { rerender: true })' in script

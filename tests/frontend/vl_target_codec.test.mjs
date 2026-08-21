import test from "node:test";
import assert from "node:assert/strict";

import {
  cloneTargetModel,
  createStarterModel,
  inspectTarget,
  parseTarget,
  serializeHtmlTable,
  serializeTarget,
} from "../../vl_layout_labeler/static/target_codec.mjs";

test("OCR preserves line count, blank lines, and surrounding whitespace", () => {
  const raw = "  dòng một  \n\n dòng ba";
  const model = parseTarget("ocr", raw);
  assert.deepEqual(model.lines, ["  dòng một  ", "", " dòng ba"]);
  assert.equal(serializeTarget("ocr", model), raw);
});

test("OTSL round-trips content and every structural cell token", () => {
  const raw = [
    "<fcel>Họ tên<lcel><fcel>Tuổi<nl>",
    "<fcel>Nguyễn Văn A<lcel><fcel>20<nl>",
    "<ucel><xcel><ecel><nl>",
  ].join("");
  const model = parseTarget("table", raw);
  assert.deepEqual(
    model.cells.find((cell) => cell.row === 1 && cell.column === 0),
    { row: 1, column: 0, rowspan: 2, colspan: 2, text: "Nguyễn Văn A" },
  );
  assert.equal(serializeTarget("table", model), raw);
  assert.equal(
    serializeHtmlTable(model),
    "<table><tr><td colspan=\"2\">Họ tên</td><td>Tuổi</td></tr>"
      + "<tr><td rowspan=\"2\" colspan=\"2\">Nguyễn Văn A</td><td>20</td></tr>"
      + "<tr><td></td></tr></table>",
  );
});

test("OTSL accepts formatting newlines but serializes one canonical stream", () => {
  const model = parseTarget("table", "<fcel>A<nl>\n<fcel>B<nl>\n");
  assert.equal(serializeTarget("table", model), "<fcel>A<nl><fcel>B<nl>");
});

test("OTSL rejects non-rectangular rows and invalid span references", () => {
  assert.match(inspectTarget("table", "<fcel>A<nl><fcel>B<ecel><nl>").error, /cùng số cột/);
  assert.match(inspectTarget("table", "<lcel><nl>").error, /nối một ô bên trái/);
  assert.match(inspectTarget("table", "<ucel><nl>").error, /nối một ô phía trên/);
});

test("Markdown chart round-trips alignments and escaped pipes", () => {
  const raw = "| Nhãn | Giá trị |\n| :--- | ---: |\n| A \\| B | 12 |";
  const model = parseTarget("chart", raw);
  assert.deepEqual(model.alignments, ["left", "right"]);
  assert.equal(model.rows[0][0], "A | B");
  assert.equal(serializeTarget("chart", model), raw);
});

test("Markdown chart rejects missing separator and inconsistent columns", () => {
  assert.match(inspectTarget("chart", "| A | B |\n| C | D |\n| E | F |").error, /phân cách/);
  assert.match(inspectTarget("chart", "| A | B |\n| --- | --- |\n| C |").error, /cùng số cột/);
});

test("formula is an exact raw LaTeX projection", () => {
  const raw = String.raw`\frac{a_1}{b^2} + \sqrt{x}`;
  const model = parseTarget("formula", raw);
  assert.equal(serializeTarget("formula", model), raw);
});

test("formula rejects HTML tables consistently with the backend", () => {
  const result = inspectTarget("formula", "<table><tr><td>x</td></tr></table>");
  assert.equal(result.valid, false);
  assert.match(result.error, /LaTeX|HTML/);
});

test("starter models are independent and produce editable structures", () => {
  const first = createStarterModel("table");
  const second = cloneTargetModel(first);
  second.cells[0].text = "A";
  assert.equal(first.cells[0].text, "");
  assert.equal(serializeTarget("table", second), "<fcel>A<ecel><nl><ecel><ecel><nl>");
});

test("HTML rowspan and colspan convert back to canonical PaddleOCR-VL OTSL", () => {
  const htmlModel = {
    rowCount: 2,
    columnCount: 3,
    cells: [
      { row: 0, column: 0, rowspan: 2, colspan: 2, text: "A & B" },
      { row: 0, column: 2, rowspan: 1, colspan: 1, text: "C" },
      { row: 1, column: 2, rowspan: 1, colspan: 1, text: "" },
    ],
  };
  assert.equal(
    serializeHtmlTable(htmlModel),
    "<table><tr><td rowspan=\"2\" colspan=\"2\">A &amp; B</td><td>C</td></tr>"
      + "<tr><td></td></tr></table>",
  );
  assert.equal(
    serializeTarget("table", htmlModel),
    "<fcel>A & B<lcel><fcel>C<nl><ucel><xcel><ecel><nl>",
  );
});

test("invalid raw remains available while the codec reports why it cannot visualize", () => {
  const raw = "<table><tr><td>A</td></tr></table>";
  const result = inspectTarget("table", raw);
  assert.equal(result.parseOk, false);
  assert.equal(result.model, null);
  assert.match(result.error, /không dùng HTML/);
  assert.equal(raw, "<table><tr><td>A</td></tr></table>");
});

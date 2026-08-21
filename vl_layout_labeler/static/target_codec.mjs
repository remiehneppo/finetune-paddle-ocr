export const TASK_FORMATS = Object.freeze({
  ocr: { label: "OCR", format: "Plain text" },
  table: { label: "Bảng", format: "OTSL" },
  formula: { label: "Công thức", format: "LaTeX" },
  chart: { label: "Biểu đồ", format: "Markdown table" },
});

export const OTSL_CELL_TOKENS = Object.freeze([
  "<fcel>",
  "<ecel>",
  "<lcel>",
  "<ucel>",
  "<xcel>",
]);

const OTSL_TOKEN_PATTERN = /<(?:fcel|ecel|lcel|ucel|xcel|nl)>/g;
const ANGLE_TAG_PATTERN = /<[^>\n]+>/g;
const HTML_TABLE_PATTERN = /<\/?(?:table|thead|tbody|tfoot|tr|th|td)\b/i;
const OTSL_TAGS = new Set([...OTSL_CELL_TOKENS, "<nl>"]);

export class TargetCodecError extends Error {
  constructor(message) {
    super(message);
    this.name = "TargetCodecError";
  }
}

function assert(condition, message) {
  if (!condition) throw new TargetCodecError(message);
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function normalizeLineEndings(raw) {
  return String(raw ?? "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function parseOtsl(raw) {
  const text = normalizeLineEndings(raw);
  assert(text.length > 0, "OTSL chưa có nội dung");
  assert(!HTML_TABLE_PATTERN.test(text), "Bảng phải dùng OTSL, không dùng HTML");
  const unknownTags = [...new Set(text.match(ANGLE_TAG_PATTERN) || [])]
    .filter((tag) => !OTSL_TAGS.has(tag));
  assert(
    unknownTags.length === 0,
    `OTSL có token không hỗ trợ: ${unknownTags.join(", ")}`,
  );

  const matches = [...text.matchAll(OTSL_TOKEN_PATTERN)];
  assert(matches.length > 0, "OTSL phải có token ô");
  assert(matches[0].index === 0, "OTSL phải bắt đầu bằng token ô");

  const rows = [];
  let row = [];
  for (let index = 0; index < matches.length; index += 1) {
    const match = matches[index];
    const token = match[0];
    const nextIndex = matches[index + 1]?.index ?? text.length;
    const content = text.slice(match.index + token.length, nextIndex);
    if (token === "<nl>") {
      assert(!content.trim(), "Không được có nội dung sau <nl>");
      assert(row.length > 0, "OTSL không được có hàng rỗng");
      rows.push(row);
      row = [];
      continue;
    }
    assert(
      token === "<fcel>" || !content.trim(),
      `${token} không được chứa text; chỉ <fcel> chứa nội dung`,
    );
    row.push({ token, text: token === "<fcel>" ? content : "" });
  }

  assert(row.length === 0, "Mỗi hàng OTSL phải kết thúc bằng <nl>");
  assert(rows.length > 0, "OTSL phải có ít nhất một hàng");
  const width = rows[0].length;
  assert(width > 0, "OTSL phải có ít nhất một cột");
  assert(rows.every((cells) => cells.length === width), "Các hàng OTSL phải có cùng số cột");
  validateOtslSpans(rows);
  const cells = [];
  rows.forEach((tokens, rowIndex) => {
    tokens.forEach((cell, columnIndex) => {
      if (!["<fcel>", "<ecel>"].includes(cell.token)) return;
      let colspan = 1;
      while (
        columnIndex + colspan < width
        && ["<lcel>", "<xcel>"].includes(tokens[columnIndex + colspan].token)
      ) colspan += 1;
      let rowspan = 1;
      while (
        rowIndex + rowspan < rows.length
        && ["<ucel>", "<xcel>"].includes(rows[rowIndex + rowspan][columnIndex].token)
      ) rowspan += 1;
      cells.push({
        row: rowIndex,
        column: columnIndex,
        rowspan,
        colspan,
        text: cell.token === "<fcel>" ? cell.text : "",
      });
    });
  });
  const model = { rowCount: rows.length, columnCount: width, cells };
  const regenerated = buildOtslRows(model);
  assert(
    regenerated.every((tokens, rowIndex) => tokens.every(
      (token, columnIndex) => token.token === rows[rowIndex][columnIndex].token,
    )),
    "Các token merge OTSL không tạo thành bảng HTML hợp lệ",
  );
  return model;
}

function validateOtslSpans(rows) {
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < rows[rowIndex].length; columnIndex += 1) {
      const token = rows[rowIndex][columnIndex].token;
      const left = rows[rowIndex][columnIndex - 1]?.token;
      const above = rows[rowIndex - 1]?.[columnIndex]?.token;
      if (token === "<lcel>") {
        assert(
          ["<fcel>", "<ecel>", "<lcel>"].includes(left),
          `<lcel> tại hàng ${rowIndex + 1}, cột ${columnIndex + 1} phải nối một ô bên trái`,
        );
      } else if (token === "<ucel>") {
        assert(
          ["<fcel>", "<ecel>", "<ucel>"].includes(above),
          `<ucel> tại hàng ${rowIndex + 1}, cột ${columnIndex + 1} phải nối một ô phía trên`,
        );
      } else if (token === "<xcel>") {
        assert(
          ["<ucel>", "<xcel>"].includes(left)
            && ["<lcel>", "<xcel>"].includes(above),
          `<xcel> tại hàng ${rowIndex + 1}, cột ${columnIndex + 1} phải nằm trong vùng gộp 2 chiều`,
        );
      }
    }
  }
}

function buildOtslRows(model) {
  assert(Number.isInteger(model?.rowCount) && model.rowCount > 0, "Bảng HTML phải có ít nhất một hàng");
  assert(Number.isInteger(model?.columnCount) && model.columnCount > 0, "Bảng HTML phải có ít nhất một cột");
  assert(Array.isArray(model.cells), "Bảng HTML phải có danh sách ô");
  const rows = Array.from({ length: model.rowCount }, () => Array(model.columnCount).fill(null));
  for (const cell of model.cells) {
    const { row, column, rowspan = 1, colspan = 1 } = cell;
    assert(Number.isInteger(row) && row >= 0, "Vị trí hàng của ô HTML không hợp lệ");
    assert(Number.isInteger(column) && column >= 0, "Vị trí cột của ô HTML không hợp lệ");
    assert(Number.isInteger(rowspan) && rowspan > 0, "rowspan phải là số nguyên dương");
    assert(Number.isInteger(colspan) && colspan > 0, "colspan phải là số nguyên dương");
    assert(row + rowspan <= model.rowCount, "rowspan vượt quá bảng HTML");
    assert(column + colspan <= model.columnCount, "colspan vượt quá bảng HTML");
    for (let rowOffset = 0; rowOffset < rowspan; rowOffset += 1) {
      for (let columnOffset = 0; columnOffset < colspan; columnOffset += 1) {
        const targetRow = row + rowOffset;
        const targetColumn = column + columnOffset;
        assert(rows[targetRow][targetColumn] === null, "Các ô HTML không được chồng lấn");
        let token = String(cell.text ?? "").length ? "<fcel>" : "<ecel>";
        if (rowOffset === 0 && columnOffset > 0) token = "<lcel>";
        else if (rowOffset > 0 && columnOffset === 0) token = "<ucel>";
        else if (rowOffset > 0 && columnOffset > 0) token = "<xcel>";
        rows[targetRow][targetColumn] = {
          token,
          text: rowOffset === 0 && columnOffset === 0 ? String(cell.text ?? "") : "",
        };
      }
    }
  }
  assert(rows.every((row) => row.every(Boolean)), "Bảng HTML phải phủ kín mọi hàng và cột");
  return rows;
}

function serializeOtsl(model) {
  const rows = buildOtslRows(model);
  const raw = rows.map((row) => `${row.map(
    (cell) => `${cell.token}${cell.token === "<fcel>" ? cell.text : ""}`,
  ).join("")}<nl>`).join("");
  parseOtsl(raw);
  return raw;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function serializeHtmlTable(model) {
  buildOtslRows(model);
  const cellsByRow = Array.from({ length: model.rowCount }, () => []);
  for (const cell of model.cells) cellsByRow[cell.row].push(cell);
  return `<table>${cellsByRow.map((cells) => `<tr>${cells
    .sort((left, right) => left.column - right.column)
    .map((cell) => {
      const rowspan = cell.rowspan > 1 ? ` rowspan="${cell.rowspan}"` : "";
      const colspan = cell.colspan > 1 ? ` colspan="${cell.colspan}"` : "";
      return `<td${rowspan}${colspan}>${escapeHtml(cell.text)}</td>`;
    }).join("")}</tr>`).join("")}</table>`;
}

function splitMarkdownRow(line) {
  const trimmed = line.trim();
  assert(trimmed.includes("|"), "Mỗi hàng Markdown phải có ký tự |");
  let body = trimmed;
  if (body.startsWith("|")) body = body.slice(1);
  if (body.endsWith("|") && !body.endsWith("\\|")) body = body.slice(0, -1);

  const cells = [];
  let current = "";
  for (let index = 0; index < body.length; index += 1) {
    const character = body[index];
    if (character === "\\" && body[index + 1] === "|") {
      current += "|";
      index += 1;
    } else if (character === "\\" && body[index + 1] === "\\") {
      current += "\\";
      index += 1;
    } else if (character === "|") {
      cells.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }
  cells.push(current.trim());
  return cells;
}

function parseAlignment(cell, columnIndex) {
  const value = cell.trim();
  assert(
    /^:?-{3,}:?$/.test(value),
    `Cột ${columnIndex + 1} của hàng phân cách Markdown phải dùng --- và dấu : tùy chọn`,
  );
  if (value.startsWith(":") && value.endsWith(":")) return "center";
  if (value.startsWith(":")) return "left";
  if (value.endsWith(":")) return "right";
  return "none";
}

function parseMarkdown(raw) {
  const text = normalizeLineEndings(raw);
  assert(text.trim().length > 0, "Markdown table chưa có nội dung");
  const lines = text.split("\n");
  assert(lines.every((line) => line.trim().length > 0), "Markdown table không được có hàng trống");
  assert(lines.length >= 3, "Markdown table cần hàng tiêu đề, hàng phân cách và ít nhất một hàng dữ liệu");
  const parsedRows = lines.map(splitMarkdownRow);
  const width = parsedRows[0].length;
  assert(width > 0, "Markdown table phải có ít nhất một cột");
  assert(parsedRows.every((row) => row.length === width), "Các hàng Markdown phải có cùng số cột");
  return {
    headers: parsedRows[0],
    alignments: parsedRows[1].map(parseAlignment),
    rows: parsedRows.slice(2),
  };
}

function escapeMarkdownCell(value) {
  return String(value ?? "").replace(/\\/g, "\\\\").replace(/\|/g, "\\|");
}

function serializeMarkdown(model) {
  assert(Array.isArray(model?.headers) && model.headers.length > 0, "Markdown table phải có tiêu đề");
  assert(Array.isArray(model.rows) && model.rows.length > 0, "Markdown table phải có ít nhất một hàng dữ liệu");
  const width = model.headers.length;
  assert(model.rows.every((row) => row.length === width), "Các hàng Markdown phải có cùng số cột");
  assert(model.alignments?.length === width, "Số căn lề phải khớp số cột Markdown");
  const alignmentTokens = model.alignments.map((alignment) => ({
    none: "---",
    left: ":---",
    center: ":---:",
    right: "---:",
  }[alignment] || "---"));
  const line = (cells) => `| ${cells.map(escapeMarkdownCell).join(" | ")} |`;
  const raw = [line(model.headers), line(alignmentTokens), ...model.rows.map(line)].join("\n");
  parseMarkdown(raw);
  return raw;
}

export function parseTarget(task, raw) {
  const text = normalizeLineEndings(raw);
  if (task === "ocr") return { lines: text.split("\n") };
  if (task === "formula") {
    assert(!HTML_TABLE_PATTERN.test(text), "Formula phải dùng LaTeX, không dùng HTML table");
    return { text };
  }
  if (task === "table") return parseOtsl(text);
  if (task === "chart") return parseMarkdown(text);
  throw new TargetCodecError(`Task không hỗ trợ: ${task}`);
}

export function serializeTarget(task, model) {
  if (task === "ocr") return (model?.lines || []).map((line) => String(line ?? "")).join("\n");
  if (task === "formula") return String(model?.text ?? "");
  if (task === "table") return serializeOtsl(model);
  if (task === "chart") return serializeMarkdown(model);
  throw new TargetCodecError(`Task không hỗ trợ: ${task}`);
}

export function inspectTarget(task, raw) {
  try {
    const model = parseTarget(task, raw);
    const text = normalizeLineEndings(raw);
    if (!text.trim()) {
      return { parseOk: true, valid: false, model, error: "Output chưa có nội dung" };
    }
    if (task === "table") {
      const hasContent = model.cells.some((cell) => cell.text.trim());
      if (!hasContent) {
        return { parseOk: true, valid: false, model, error: "OTSL cần ít nhất một <fcel> có nội dung" };
      }
    }
    if (task === "chart") {
      const values = [...model.headers, ...model.rows.flat()];
      if (!values.some((value) => value.trim())) {
        return { parseOk: true, valid: false, model, error: "Markdown table chưa có dữ liệu" };
      }
    }
    return { parseOk: true, valid: true, model, error: null };
  } catch (error) {
    return {
      parseOk: false,
      valid: false,
      model: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function createStarterModel(task) {
  if (task === "ocr") return { lines: [""] };
  if (task === "formula") return { text: "" };
  if (task === "table") {
    return {
      rowCount: 2,
      columnCount: 2,
      cells: [
        { row: 0, column: 0, rowspan: 1, colspan: 1, text: "" },
        { row: 0, column: 1, rowspan: 1, colspan: 1, text: "" },
        { row: 1, column: 0, rowspan: 1, colspan: 1, text: "" },
        { row: 1, column: 1, rowspan: 1, colspan: 1, text: "" },
      ],
    };
  }
  if (task === "chart") {
    return {
      headers: ["Cột 1", "Cột 2"],
      alignments: ["none", "none"],
      rows: [["", ""]],
    };
  }
  throw new TargetCodecError(`Task không hỗ trợ: ${task}`);
}

export function cloneTargetModel(model) {
  return clone(model);
}

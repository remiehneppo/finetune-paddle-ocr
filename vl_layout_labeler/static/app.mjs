import {
  TASK_FORMATS,
  cloneTargetModel,
  createStarterModel,
  inspectTarget,
  mergeOcrLine,
  pasteOcrLines,
  serializeTarget,
  splitOcrLine,
} from "./target_codec.mjs";
import {
  appendValidationPreview,
  invalidateBlockValidation,
  loadValidationEnabled,
  saveValidationEnabled,
  validationIssueCount,
  validationSelectionRange,
} from "./validation_ui.mjs";
import {
  fitCanvasView,
  panCanvasView,
  resizeRectangleEdge,
  zoomCanvasView,
} from "./canvas_geometry.mjs";

const $ = (id) => document.getElementById(id);
const svg = (name) => document.createElementNS("http://www.w3.org/2000/svg", name);
const state = {
  images: [], current: null, currentId: null, selected: null,
  view: { scale: 1, x: 0, y: 0 }, viewMode: "fit", add: false, dirty: false,
  timer: null, drag: null, pan: null, spacePressed: false,
  targetMode: "visual", targetError: null, busy: null, batchBusy: null,
  workspaceBusy: false, workspaceStartedAt: null, workspaceTimer: null,
  validationConfigured: false, validationEnabled: false,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || response.statusText);
  return payload;
}

function setStatus(text) { $("save-status").textContent = text; }
function blockById(id) { return state.current?.blocks.find((block) => block.id === id) || null; }
function selectedBlock() { return blockById(state.selected); }
const IMAGE_STATUS_LABELS = {
  draft: "Draft",
  detected: "Đã detect",
  edited: "Đang chỉnh sửa",
  completed: "✓ Complete",
};
const OPERATION_LABELS = {
  detect: "Model đang detect layout…",
  prelabel: "Model đang prelabel nội dung…",
  validate: "LLM đang kiểm tra OCR…",
  complete: "Đang hoàn tất ảnh…",
  draft: "Đang mở lại chế độ chỉnh sửa…",
  "batch-detect": "Model đang detect toàn bộ folder…",
  "batch-prelabel": "Model đang prelabel toàn bộ folder…",
};
function isCompleted() { return state.current?.status === "completed"; }
function editingLocked() { return isCompleted() || Boolean(state.busy); }
function currentOperation() { return state.busy || state.batchBusy; }
function setBusy(action) {
  state.busy = action;
  renderInteractionState();
}
function setBatchBusy(action) {
  state.batchBusy = action;
  renderInteractionState();
}
function setWorkspaceBusy(busy, message = "Đang mở folder…") {
  state.workspaceBusy = busy;
  if (busy) {
    state.workspaceStartedAt = Date.now();
    clearInterval(state.workspaceTimer);
    state.workspaceTimer = setInterval(renderWorkspaceStatus, 1000);
  } else {
    state.workspaceStartedAt = null;
    clearInterval(state.workspaceTimer);
    state.workspaceTimer = null;
  }
  $("open-folder").textContent = busy ? "Đang mở…" : "Mở folder";
  $("workspace-message").textContent = message;
  renderWorkspaceStatus();
  renderInteractionState();
}
function renderWorkspaceStatus() {
  const status = $("workspace-status");
  const visible = state.workspaceBusy;
  status.hidden = !visible;
  if (!visible) return;
  const elapsed = Math.max(0, Math.floor((Date.now() - state.workspaceStartedAt) / 1000));
  if (elapsed >= 10 && $("workspace-message").textContent === "Đang mở folder…") {
    $("workspace-message").textContent = "Vẫn đang mở folder, có thể folder khá lớn…";
  }
  $("workspace-elapsed").textContent = `${elapsed}s`;
}
function renderInteractionState() {
  const completed = isCompleted();
  const operation = currentOperation();
  const processing = Boolean(operation) || state.workspaceBusy;
  const noCurrent = !state.current;
  const block = selectedBlock();
  const operationStatus = $("operation-status");
  operationStatus.hidden = !processing;
  $("operation-message").textContent = OPERATION_LABELS[operation] || "Đang xử lý…";
  $("completed-banner").hidden = !completed;
  $("complete-current").hidden = completed;
  $("reopen-current").hidden = !completed;
  $("complete-current").disabled = noCurrent || processing;
  $("reopen-current").disabled = noCurrent || processing;
  $("detect-current").disabled = noCurrent || completed || processing;
  $("prelabel-page").disabled = noCurrent || completed || processing;
  $("prelabel-current").disabled = (
    noCurrent || completed || processing || !block || block.task === null || block.skipped
  );
  $("llm-validation").disabled = !state.validationConfigured || processing;
  $("validate-current").disabled = (
    noCurrent || completed || processing || !state.validationConfigured
  );
  $("add-mode").disabled = noCurrent || completed || processing;
  $("detect-batch").disabled = processing;
  $("prelabel-batch").disabled = processing;
  $("cancel-batch").disabled = !state.batchBusy;
  $("open-folder").disabled = processing;
  $("folder-path").disabled = processing;
  for (const id of ["page-fit", "page-actual", "page-zoom-out", "page-zoom-in"]) {
    $(id).disabled = noCurrent;
  }
  $("page-stage").classList.toggle("read-only", completed || Boolean(state.busy));
  $("page-stage").setAttribute("aria-busy", String(Boolean(state.busy)));
  document.querySelector(".canvas-panel").setAttribute("aria-busy", String(processing));
  const locked = completed || Boolean(state.busy);
  $("editor").querySelectorAll(
    ".block-fields input, .block-fields textarea, .block-fields select, "
      + "#target-editor input, #target-editor textarea, #target-editor select, #target-editor button, "
      + ".block-options input, .block-options textarea, .block-options select, .block-options button",
  ).forEach((control) => { control.disabled = locked; });
}
function element(name, className, text) {
  const value = document.createElement(name);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}
function imageToScreen([x, y]) { return [x * state.view.scale + state.view.x, y * state.view.scale + state.view.y]; }
function screenToImage([x, y]) { return [(x - state.view.x) / state.view.scale, (y - state.view.y) / state.view.scale]; }
function rectangle(start, end) {
  const left = Math.min(start[0], end[0]);
  const right = Math.max(start[0], end[0]);
  const top = Math.min(start[1], end[1]);
  const bottom = Math.max(start[1], end[1]);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

function updatePageZoomValue() {
  $("page-zoom-value").textContent = state.viewMode === "fit"
    ? "Fit"
    : `${Math.round(state.view.scale * 100)}%`;
}
function renderResizeHandles(overlay, block) {
  const points = block.polygon.map(imageToScreen);
  const edges = [
    ["top", 0, 1, "horizontal"],
    ["right", 1, 2, "vertical"],
    ["bottom", 2, 3, "horizontal"],
    ["left", 3, 0, "vertical"],
  ];
  for (const [edge, startIndex, endIndex, orientation] of edges) {
    const start = points[startIndex];
    const end = points[endIndex];
    const hit = svg("line");
    hit.setAttribute("x1", start[0]);
    hit.setAttribute("y1", start[1]);
    hit.setAttribute("x2", end[0]);
    hit.setAttribute("y2", end[1]);
    hit.classList.add("edge-hit", orientation);
    hit.onpointerdown = (event) => beginEdgeDrag(event, block, edge);
    overlay.append(hit);

    const grip = svg("rect");
    const midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
    const width = orientation === "horizontal" ? 28 : 8;
    const height = orientation === "horizontal" ? 8 : 28;
    grip.setAttribute("x", midpoint[0] - width / 2);
    grip.setAttribute("y", midpoint[1] - height / 2);
    grip.setAttribute("width", width);
    grip.setAttribute("height", height);
    grip.setAttribute("rx", 4);
    grip.classList.add("edge-grip", orientation);
    grip.onpointerdown = (event) => beginEdgeDrag(event, block, edge);
    overlay.append(grip);
  }
  points.forEach(([x, y], index) => {
    const corner = svg("circle");
    corner.setAttribute("cx", x);
    corner.setAttribute("cy", y);
    corner.setAttribute("r", 6);
    corner.classList.add("corner");
    corner.onpointerdown = (event) => beginCornerDrag(event, block, index);
    overlay.append(corner);
  });
}
function renderCanvas() {
  const overlay = $("overlay");
  overlay.replaceChildren();
  for (const block of state.current?.blocks || []) {
    const polygon = svg("polygon");
    polygon.setAttribute("points", block.polygon.map(imageToScreen).map((point) => point.join(",")).join(" "));
    polygon.classList.add("bbox", block.task || "layout-only");
    if (block.id === state.selected) polygon.classList.add("selected");
    if (block.skipped) polygon.classList.add("skipped");
    polygon.onclick = (event) => { event.stopPropagation(); state.selected = block.id; state.targetError = null; render(); };
    polygon.onpointerdown = (event) => beginDrag(event, block);
    overlay.append(polygon);
    if (block.id === state.selected && !editingLocked()) renderResizeHandles(overlay, block);
  }
}
function applyPageView() {
  const image = $("page-image");
  Object.assign(image.style, {
    left: `${state.view.x}px`, top: `${state.view.y}px`,
    width: `${image.naturalWidth * state.view.scale}px`,
    height: `${image.naturalHeight * state.view.scale}px`,
  });
  updatePageZoomValue();
  renderCanvas();
}
function fitImage() {
  const image = $("page-image");
  const stage = $("page-stage");
  if (!image.naturalWidth || !stage.clientWidth) return;
  state.view = fitCanvasView(
    stage.clientWidth, stage.clientHeight, image.naturalWidth, image.naturalHeight,
  );
  state.viewMode = "fit";
  applyPageView();
}

function renderImages() {
  const list = $("image-list");
  list.replaceChildren();
  $("image-count").textContent = state.images.length;
  for (const image of state.images) {
    const item = document.createElement("li");
    if (image.image_id === state.currentId) item.classList.add("selected");
    if (image.status === "completed") item.classList.add("completed");
    const name = document.createElement("span");
    const status = document.createElement("span");
    name.className = "name";
    status.className = `status ${image.error ? "error" : image.status}`;
    name.textContent = image.name;
    status.textContent = image.error || IMAGE_STATUS_LABELS[image.status] || image.status;
    item.append(name, status);
    item.onclick = () => { if (!state.busy) loadImage(image.image_id); };
    list.append(item);
  }
}

function renderBlocks() {
  const list = $("block-list");
  list.replaceChildren();
  $("block-count").textContent = state.current?.blocks.length || 0;
  for (const block of state.current?.blocks || []) {
    const item = document.createElement("li");
    if (block.id === state.selected) item.classList.add("selected");
    const task = block.task || "layout-only";
    const content = block.task ? (block.text || "(chưa có text)") : block.layout_label;
    item.append(document.createTextNode(`${block.order + 1}. ${task} · ${content}`));
    const warnings = validationIssueCount(block);
    if (warnings) item.append(element("span", "validation-badge", String(warnings)));
    item.onclick = () => { state.selected = block.id; state.targetError = null; render(); };
    list.append(item);
  }
}

function renderInspector() {
  const block = selectedBlock();
  $("editor").hidden = !block;
  $("prelabel-current").disabled = !block || block.task === null || block.skipped;
  renderValidationIssues(block);
  if (!block) return;
  $("layout-label").value = block.layout_label;
  $("task").value = block.task || "";
  $("layout-only-note").hidden = block.task !== null;
  $("skipped").checked = block.skipped;
  $("block-meta").textContent = `${block.source}${block.score == null ? "" : ` · ${(block.score * 100).toFixed(1)}%`}`;
  renderTargetEditor();
}

function renderValidationIssues(block) {
  const list = $("validation-issues");
  const empty = $("validation-empty");
  const meta = $("validation-meta");
  list.replaceChildren();
  const validation = block?.validation;
  empty.hidden = Boolean(validation);
  meta.textContent = validation
    ? `${validation.model} · ${new Date(validation.checked_at).toLocaleString()}`
    : "";
  if (!validation) return;
  if (!validation.issues.length) {
    empty.hidden = false;
    empty.textContent = "Không có cảnh báo.";
    return;
  }
  empty.textContent = "Chưa chạy validation cho block này.";
  for (const issue of validation.issues) {
    const item = element("li", "validation-issue");
    const button = element("button", "validation-issue-button");
    button.type = "button";
    button.append(
      element("strong", "", issue.category),
      element("span", "validation-reason", issue.reason),
      element("span", "validation-suggestion", `Gợi ý: ${issue.suggestion}`),
    );
    const preview = element("span", "validation-preview");
    appendValidationPreview(document, preview, block.text, issue);
    button.append(preview);
    button.onclick = () => {
      state.selected = issue.block_id;
      state.targetMode = "raw";
      state.targetError = null;
      render();
      const editor = $("text");
      editor.focus();
      const range = validationSelectionRange(block.text, issue);
      if (range) editor.setSelectionRange(range.start, range.end);
    };
    item.append(button);
    list.append(item);
  }
}

function setTargetMessage(inspection) {
  const message = $("target-message");
  const error = state.targetError || inspection.error;
  message.className = `target-message ${inspection.valid && !state.targetError ? "valid" : "error"}`;
  message.textContent = error || "Output hợp lệ và đồng bộ với raw";
}

function targetButton(label, callback, className = "secondary") {
  const button = element("button", className, label);
  button.type = "button";
  button.onclick = callback;
  return button;
}

function markBlockEdited(block) {
  if (editingLocked()) return;
  invalidateBlockValidation(block);
  markManual(block);
  renderBlocks();
  renderValidationIssues(block);
  markDirty();
}

function commitVisualModel(task, model, { rerender = false } = {}) {
  if (editingLocked()) return false;
  const block = selectedBlock();
  if (!block || block.task !== task) return false;
  try {
    const raw = serializeTarget(task, model);
    block.text = raw;
    $("text").value = raw;
    state.targetError = null;
    markBlockEdited(block);
    if (rerender) renderTargetEditor();
    else setTargetMessage(inspectTarget(task, raw));
    return true;
  } catch (error) {
    state.targetError = `Không áp dụng thay đổi: ${error.message}`;
    renderTargetEditor();
    return false;
  }
}

let focusLineRequest = null;

function autoResizeLineInput(textarea) {
  if (!textarea) return;
  textarea.style.height = "auto";
  const nextHeight = Math.max(textarea.scrollHeight, 38);
  textarea.style.height = `${nextHeight}px`;
}

function renderOcrEditor(container, model) {
  const list = element("div", "line-editor");
  const isVertical = $("layout-label")?.value === "vertical_text";
  if (!model.lines || model.lines.length === 0) {
    model.lines = [""];
  }

  model.lines.forEach((line, index) => {
    const row = element("div", "line-row");
    row.append(element("span", "line-number", String(index + 1)));
    const input = element("textarea", "line-input");
    input.value = line;
    input.rows = 1;
    input.spellcheck = false;
    input.setAttribute("aria-label", `Dòng OCR ${index + 1}`);

    const syncHeight = () => {
      if (!isVertical) autoResizeLineInput(input);
    };

    input.oninput = (event) => {
      syncHeight();
      model.lines[index] = event.target.value;
      commitVisualModel("ocr", model);
    };

    input.onkeydown = (event) => {
      if (event.isComposing || event.keyCode === 229) return;

      if (event.key === "Enter") {
        event.preventDefault();
        const start = input.selectionStart ?? input.value.length;
        const end = input.selectionEnd ?? start;
        const result = splitOcrLine(model.lines, index, start, end);
        model.lines = result.lines;
        focusLineRequest = result.focus;
        commitVisualModel("ocr", model, { rerender: true });
        return;
      }

      if (
        event.key === "Backspace"
        && input.selectionStart === 0
        && input.selectionEnd === 0
        && index > 0
      ) {
        event.preventDefault();
        const result = mergeOcrLine(model.lines, index);
        model.lines = result.lines;
        focusLineRequest = result.focus;
        commitVisualModel("ocr", model, { rerender: true });
        return;
      }
    };

    input.onpaste = (event) => {
      const pasteText = event.clipboardData?.getData("text") || "";
      if (!pasteText.includes("\n")) return;
      event.preventDefault();
      const start = input.selectionStart ?? input.value.length;
      const end = input.selectionEnd ?? start;
      const result = pasteOcrLines(model.lines, index, start, end, pasteText);
      model.lines = result.lines;
      focusLineRequest = result.focus;
      commitVisualModel("ocr", model, { rerender: true });
    };

    row.append(
      input,
      targetButton(
        "Xóa",
        () => {
          if (model.lines.length === 1) return;
          const targetIndex = index > 0 ? index - 1 : 0;
          const targetLine = index > 0 ? (model.lines[index - 1] || "") : (model.lines[1] || "");
          focusLineRequest = {
            index: targetIndex,
            position: targetLine.length,
          };
          model.lines.splice(index, 1);
          commitVisualModel("ocr", model, { rerender: true });
        },
        "icon danger",
      ),
    );
    list.append(row);

    requestAnimationFrame(syncHeight);

    if (focusLineRequest && focusLineRequest.index === index) {
      const { position } = focusLineRequest;
      focusLineRequest = null;
      requestAnimationFrame(() => {
        input.focus();
        if (typeof input.setSelectionRange === "function") {
          input.setSelectionRange(position, position);
        }
        syncHeight();
        if (typeof input.scrollIntoView === "function") {
          input.scrollIntoView({ block: "nearest" });
        }
      });
    }
  });

  if (focusLineRequest) {
    focusLineRequest = null;
  }

  container.append(
    list,
    targetButton("+ Thêm dòng", () => {
      focusLineRequest = { index: model.lines.length, position: 0 };
      model.lines.push("");
      commitVisualModel("ocr", model, { rerender: true });
    }),
  );
}

function renderFormulaEditor(container, model) {
  const input = element("textarea", "formula-input");
  const preview = element("div", "formula-preview");
  input.value = model.text;
  input.spellcheck = false;
  input.setAttribute("aria-label", "LaTeX công thức");
  const refresh = () => {
    preview.textContent = model.text || "Công thức chưa có nội dung";
    preview.classList.toggle("empty", !model.text);
  };
  input.oninput = (event) => {
    model.text = event.target.value;
    refresh();
    commitVisualModel("formula", model);
  };
  refresh();
  container.append(
    element("p", "editor-hint", "LaTeX được giữ nguyên từng ký tự; khung dưới giúp kiểm tra nhanh output."),
    input,
    preview,
  );
}

function renderOtslEditor(container, model) {
  const wrapper = element("div", "table-scroll table-workspace-scroll");
  const table = element("table", "html-table-editor compact-table-editor");
  const body = element("tbody");
  const workspace = element("div", "table-editor-workspace");
  const toolbar = element("div", "table-toolbar");
  const structureActions = element("div", "table-toolbar-group table-structure-actions");
  const cellActions = element("div", "table-toolbar-group table-cell-actions");
  const activeLabel = element("span", "active-cell-label", "Chọn một ô");
  let activeCell = model.cells[0] || null;
  let activeHolder = null;

  const adjacentCell = (cell, direction) => {
    if (!cell) return null;
    return direction === "right"
      ? model.cells.find((candidate) => (
        candidate.row === cell.row
        && candidate.column === cell.column + cell.colspan
        && candidate.rowspan === cell.rowspan
      ))
      : model.cells.find((candidate) => (
        candidate.row === cell.row + cell.rowspan
        && candidate.column === cell.column
        && candidate.colspan === cell.colspan
      ));
  };

  const mergeText = (first, second) => [first, second].filter(Boolean).join("\n");
  const merge = (cell, direction) => {
    const adjacent = adjacentCell(cell, direction);
    if (!adjacent) return;
    const next = cloneTargetModel(model);
    const target = next.cells.find(
      (candidate) => candidate.row === cell.row && candidate.column === cell.column,
    );
    target.text = mergeText(target.text, adjacent.text);
    if (direction === "right") target.colspan += adjacent.colspan;
    else target.rowspan += adjacent.rowspan;
    next.cells = next.cells.filter(
      (candidate) => candidate.row !== adjacent.row || candidate.column !== adjacent.column,
    );
    commitVisualModel("table", next, { rerender: true });
  };

  const split = (cell) => {
    if (!cell || (cell.rowspan === 1 && cell.colspan === 1)) return;
    const next = cloneTargetModel(model);
    next.cells = next.cells.filter(
      (candidate) => candidate.row !== cell.row || candidate.column !== cell.column,
    );
    for (let rowOffset = 0; rowOffset < cell.rowspan; rowOffset += 1) {
      for (let columnOffset = 0; columnOffset < cell.colspan; columnOffset += 1) {
        next.cells.push({
          row: cell.row + rowOffset,
          column: cell.column + columnOffset,
          rowspan: 1,
          colspan: 1,
          text: rowOffset === 0 && columnOffset === 0 ? cell.text : "",
        });
      }
    }
    commitVisualModel("table", next, { rerender: true });
  };

  const mergeRight = targetButton("Gộp →", () => merge(activeCell, "right"), "icon");
  const mergeDown = targetButton("Gộp ↓", () => merge(activeCell, "down"), "icon");
  const splitCell = targetButton("Tách", () => split(activeCell), "icon");
  const updateCellToolbar = () => {
    const spanText = activeCell && (activeCell.rowspan > 1 || activeCell.colspan > 1)
      ? ` · gộp ${activeCell.rowspan}×${activeCell.colspan}`
      : "";
    activeLabel.textContent = activeCell
      ? `Ô ${activeCell.row + 1}, ${activeCell.column + 1}${spanText}`
      : "Chọn một ô";
    mergeRight.disabled = !adjacentCell(activeCell, "right");
    mergeDown.disabled = !adjacentCell(activeCell, "down");
    splitCell.disabled = !activeCell || (activeCell.rowspan === 1 && activeCell.colspan === 1);
  };
  const selectCell = (cell, holder) => {
    activeHolder?.classList.remove("active-cell");
    activeCell = cell;
    activeHolder = holder;
    activeHolder.classList.add("active-cell");
    updateCellToolbar();
  };

  structureActions.append(
    targetButton("+ Hàng", () => {
      const next = cloneTargetModel(model);
      for (let column = 0; column < next.columnCount; column += 1) {
        next.cells.push({ row: next.rowCount, column, rowspan: 1, colspan: 1, text: "" });
      }
      next.rowCount += 1;
      commitVisualModel("table", next, { rerender: true });
    }, "icon"),
    targetButton("+ Cột", () => {
      const next = cloneTargetModel(model);
      for (let row = 0; row < next.rowCount; row += 1) {
        next.cells.push({ row, column: next.columnCount, rowspan: 1, colspan: 1, text: "" });
      }
      next.columnCount += 1;
      commitVisualModel("table", next, { rerender: true });
    }, "icon"),
    targetButton("− Hàng", () => {
      if (model.rowCount === 1) return;
      const lastRow = model.rowCount - 1;
      const next = cloneTargetModel(model);
      next.cells = next.cells.flatMap((cell) => {
        if (cell.row === lastRow) return [];
        if (cell.row + cell.rowspan - 1 === lastRow) return [{ ...cell, rowspan: cell.rowspan - 1 }];
        return [cell];
      });
      next.rowCount -= 1;
      commitVisualModel("table", next, { rerender: true });
    }, "icon danger"),
    targetButton("− Cột", () => {
      if (model.columnCount === 1) return;
      const lastColumn = model.columnCount - 1;
      const next = cloneTargetModel(model);
      next.cells = next.cells.flatMap((cell) => {
        if (cell.column === lastColumn) return [];
        if (cell.column + cell.colspan - 1 === lastColumn) return [{ ...cell, colspan: cell.colspan - 1 }];
        return [cell];
      });
      next.columnCount -= 1;
      commitVisualModel("table", next, { rerender: true });
    }, "icon danger"),
  );
  cellActions.append(activeLabel, mergeRight, mergeDown, splitCell);
  toolbar.append(structureActions, cellActions);

  for (let rowIndex = 0; rowIndex < model.rowCount; rowIndex += 1) {
    const row = element("tr");
    const cells = model.cells
      .filter((cell) => cell.row === rowIndex)
      .sort((left, right) => left.column - right.column);
    cells.forEach((cell) => {
      const holder = element("td");
      holder.rowSpan = cell.rowspan;
      holder.colSpan = cell.colspan;
      const input = element("textarea", "cell-input");
      input.value = cell.text;
      input.rows = Math.max(2, Math.min(6, cell.text.split("\n").length));
      input.placeholder = "Nội dung ô";
      input.setAttribute("aria-label", `Nội dung hàng ${rowIndex + 1}, cột ${cell.column + 1}`);
      input.addEventListener("focus", () => selectCell(cell, holder));
      holder.addEventListener("pointerdown", () => selectCell(cell, holder));
      input.oninput = (event) => {
        cell.text = event.target.value;
        input.rows = Math.max(2, Math.min(6, cell.text.split("\n").length));
        commitVisualModel("table", model);
      };
      const span = element(
        "span",
        "cell-span",
        cell.rowspan > 1 || cell.colspan > 1 ? `${cell.rowspan}×${cell.colspan}` : "",
      );
      holder.append(input, span);
      if (cell === activeCell) {
        activeHolder = holder;
        holder.classList.add("active-cell");
      }
      row.append(holder);
    });
    body.append(row);
  }
  table.append(body);
  wrapper.append(table);
  workspace.append(toolbar, wrapper);
  updateCellToolbar();
  container.append(
    element("p", "editor-hint table-hint", "Chọn một ô để gộp hoặc tách. Các thao tác cấu trúc nằm cố định phía trên bảng."),
    workspace,
  );
}

function renderChartEditor(container, model) {
  const wrapper = element("div", "table-scroll");
  const table = element("table", "target-grid chart-grid");
  const header = element("thead");
  const headerRow = element("tr");
  const alignmentRow = element("tr", "alignment-row");
  model.headers.forEach((value, columnIndex) => {
    const cell = element("th");
    const input = element("input");
    input.value = value;
    input.setAttribute("aria-label", `Tiêu đề cột ${columnIndex + 1}`);
    input.oninput = (event) => {
      model.headers[columnIndex] = event.target.value;
      commitVisualModel("chart", model);
    };
    cell.append(input);
    headerRow.append(cell);
    const alignmentCell = element("th");
    const select = element("select");
    for (const [key, label] of [["none", "Mặc định"], ["left", "Trái"], ["center", "Giữa"], ["right", "Phải"]]) {
      const option = element("option", "", label);
      option.value = key;
      option.selected = model.alignments[columnIndex] === key;
      select.append(option);
    }
    select.onchange = (event) => {
      model.alignments[columnIndex] = event.target.value;
      commitVisualModel("chart", model);
    };
    alignmentCell.append(select);
    alignmentRow.append(alignmentCell);
  });
  header.append(headerRow, alignmentRow);
  const body = element("tbody");
  model.rows.forEach((values, rowIndex) => {
    const row = element("tr");
    values.forEach((value, columnIndex) => {
      const cell = element("td");
      const input = element("input");
      input.value = value;
      input.setAttribute("aria-label", `Dữ liệu hàng ${rowIndex + 1}, cột ${columnIndex + 1}`);
      input.oninput = (event) => {
        model.rows[rowIndex][columnIndex] = event.target.value;
        commitVisualModel("chart", model);
      };
      cell.append(input);
      row.append(cell);
    });
    const action = element("td", "row-actions");
    action.append(targetButton("Xóa", () => {
      if (model.rows.length === 1) return;
      const next = cloneTargetModel(model);
      next.rows.splice(rowIndex, 1);
      commitVisualModel("chart", next, { rerender: true });
    }, "icon danger"));
    row.append(action);
    body.append(row);
  });
  table.append(header, body);
  wrapper.append(table);
  const controls = element("div", "grid-actions");
  controls.append(
    targetButton("+ Hàng", () => {
      const next = cloneTargetModel(model);
      next.rows.push(next.headers.map(() => ""));
      commitVisualModel("chart", next, { rerender: true });
    }),
    targetButton("+ Cột", () => {
      const next = cloneTargetModel(model);
      next.headers.push(`Cột ${next.headers.length + 1}`);
      next.alignments.push("none");
      next.rows.forEach((row) => row.push(""));
      commitVisualModel("chart", next, { rerender: true });
    }),
    targetButton("− Cột cuối", () => {
      if (model.headers.length === 1) return;
      const next = cloneTargetModel(model);
      next.headers.pop();
      next.alignments.pop();
      next.rows.forEach((row) => row.pop());
      commitVisualModel("chart", next, { rerender: true });
    }, "secondary danger"),
  );
  container.append(wrapper, controls);
}

function renderTargetEditor() {
  const block = selectedBlock();
  const targetEditor = $("target-editor");
  targetEditor.hidden = !block || block.task === null;
  if (!block || block.task === null) return;
  const inspection = inspectTarget(block.task, block.text);
  $("target-format").textContent = `${TASK_FORMATS[block.task].label} · ${TASK_FORMATS[block.task].format}`;
  $("text").value = block.text;
  $("visual-tab").classList.toggle("active", state.targetMode === "visual");
  $("raw-tab").classList.toggle("active", state.targetMode === "raw");
  $("visual-tab").setAttribute("aria-selected", String(state.targetMode === "visual"));
  $("raw-tab").setAttribute("aria-selected", String(state.targetMode === "raw"));
  $("visual-tab").tabIndex = state.targetMode === "visual" ? 0 : -1;
  $("raw-tab").tabIndex = state.targetMode === "raw" ? 0 : -1;
  $("visual-panel").hidden = state.targetMode !== "visual";
  $("raw-panel").hidden = state.targetMode !== "raw";
  setTargetMessage(inspection);
  $("starter-target").hidden = inspection.valid;
  if (state.targetMode !== "visual") return;
  const container = $("visual-editor");
  container.replaceChildren();
  if (!inspection.parseOk) {
    container.append(element("div", "visual-unavailable", "Raw output chưa thể trực quan hóa. Sửa ở tab Raw hoặc tạo một mẫu mới."));
    return;
  }
  const model = cloneTargetModel(inspection.model);
  if (block.task === "ocr") renderOcrEditor(container, model);
  else if (block.task === "formula") renderFormulaEditor(container, model);
  else if (block.task === "table") renderOtslEditor(container, model);
  else renderChartEditor(container, model);
}

function render() {
  renderBlocks();
  renderInspector();
  renderCanvas();
  renderInteractionState();
}

async function loadImages() {
  const payload = await api("/api/images");
  state.images = payload.images;
  renderImages();
  if (!state.currentId && state.images[0]) await loadImage(state.images[0].image_id);
}

async function loadImage(id) {
  if (state.dirty) await save();
  state.current = await api(`/api/images/${id}/annotation`);
  state.currentId = id;
  state.selected = state.current.blocks[0]?.id || null;
  state.targetError = null;
  $("current-name").textContent = state.images.find((item) => item.image_id === id)?.name || state.current.image.path;
  $("page-image").src = `/api/images/${id}/content?${Date.now()}`;
  $("page-image").onload = fitImage;
  render();
  renderImages();
  setStatus("Đã tải");
}

function markDirty() {
  if (isCompleted()) return;
  state.dirty = true;
  setStatus("Đang chờ lưu…");
  clearTimeout(state.timer);
  state.timer = setTimeout(save, 500);
}

async function save() {
  if (!state.current || !state.dirty) return true;
  try {
    state.current = await api(`/api/images/${state.currentId}/annotation`, {
      method: "PUT", body: JSON.stringify(state.current),
    });
    state.dirty = false;
    setStatus(`Đã lưu revision ${state.current.revision}`);
    return true;
  } catch (error) {
    setStatus(`Lỗi lưu: ${error.message}`);
    return false;
  }
}

function mutate(callback) {
  if (!state.current || editingLocked()) {
    if (isCompleted()) setStatus("Ảnh đã Complete. Mở lại chỉnh sửa trước khi thay đổi.");
    return;
  }
  callback();
  render();
  markDirty();
}

async function runCurrent(action, body) {
  if (!state.currentId || state.busy || state.batchBusy) return;
  if (isCompleted() && action !== "draft") {
    setStatus("Ảnh đã Complete. Mở lại chỉnh sửa trước khi chạy model.");
    return;
  }
  setBusy(action);
  try {
    if (action !== "draft" && !(await save())) return;
    const options = { method: "POST" };
    if (body !== undefined) options.body = JSON.stringify(body);
    const selectedId = state.selected;
    const payload = await api(`/api/images/${state.currentId}/${action}`, options);
    const validationError = payload.validation_error;
    state.current = payload.annotation || payload;
    state.selected = blockById(selectedId)?.id || state.current.blocks[0]?.id || null;
    render();
    await loadImages();
    setStatus(
      validationError
        ? `Prelabel đã lưu; validation lỗi: ${validationError}`
        : action === "draft" ? "Đã mở lại chế độ chỉnh sửa" : "Hoàn tất"
    );
  } catch (error) {
    setStatus(`Lỗi: ${error.message}`);
  } finally {
    setBusy(null);
    render();
  }
}

function eventPoint(event) {
  const bounds = $("page-stage").getBoundingClientRect();
  return screenToImage([event.clientX - bounds.left, event.clientY - bounds.top]);
}
function stagePoint(event) {
  const bounds = $("page-stage").getBoundingClientRect();
  return [event.clientX - bounds.left, event.clientY - bounds.top];
}
function beginDrag(event, block) {
  if (event.button !== 0 || state.spacePressed || state.add || editingLocked()) return;
  event.preventDefault();
  const activeBlock = blockById(block.id);
  if (!activeBlock) return;
  const selectionChanged = state.selected !== activeBlock.id;
  state.selected = activeBlock.id;
  state.targetError = null;
  if (selectionChanged) render();
  state.drag = {
    block: activeBlock,
    start: eventPoint(event),
    original: activeBlock.polygon.map((point) => [...point]),
    pending: null,
    frame: null,
  };
}
function beginCornerDrag(event, block, corner) {
  if (event.button !== 0 || state.spacePressed || editingLocked()) return;
  event.preventDefault();
  event.stopPropagation();
  const activeBlock = blockById(block.id);
  if (!activeBlock) return;
  state.drag = { block: activeBlock, corner, pending: null, frame: null };
}
function beginEdgeDrag(event, block, edge) {
  if (event.button !== 0 || state.spacePressed || editingLocked()) return;
  event.preventDefault();
  event.stopPropagation();
  const activeBlock = blockById(block.id);
  if (!activeBlock) return;
  state.drag = {
    block: activeBlock,
    edge,
    original: activeBlock.polygon.map((point) => [...point]),
    pending: null,
    frame: null,
  };
}
function renderDraggedBlock(block) {
  const polygon = $("overlay").querySelector(".bbox.selected");
  if (!polygon) return;
  const points = block.polygon.map(imageToScreen);
  polygon.setAttribute("points", points.map((point) => point.join(",")).join(" "));
  $("overlay").querySelectorAll(".corner").forEach((corner, index) => {
    const point = points[index];
    if (!point) return;
    corner.setAttribute("cx", point[0]);
    corner.setAttribute("cy", point[1]);
  });
  const edges = [
    [0, 1, "horizontal"],
    [1, 2, "vertical"],
    [2, 3, "horizontal"],
    [3, 0, "vertical"],
  ];
  const hits = $("overlay").querySelectorAll(".edge-hit");
  const grips = $("overlay").querySelectorAll(".edge-grip");
  edges.forEach(([startIndex, endIndex, orientation], edgeIndex) => {
    const start = points[startIndex];
    const end = points[endIndex];
    if (!start || !end) return;
    const hit = hits[edgeIndex];
    if (hit) {
      hit.setAttribute("x1", start[0]);
      hit.setAttribute("y1", start[1]);
      hit.setAttribute("x2", end[0]);
      hit.setAttribute("y2", end[1]);
    }
    const grip = grips[edgeIndex];
    if (grip) {
      const midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
      const width = orientation === "horizontal" ? 28 : 8;
      const height = orientation === "horizontal" ? 8 : 28;
      grip.setAttribute("x", midpoint[0] - width / 2);
      grip.setAttribute("y", midpoint[1] - height / 2);
    }
  });
}
function applyDragPoint(point) {
  if (!state.drag || !point) return;
  if (state.drag.corner !== undefined) {
    state.drag.block.polygon[state.drag.corner] = [
      Math.max(0, Math.min(state.current.image.width - 1, point[0])),
      Math.max(0, Math.min(state.current.image.height - 1, point[1])),
    ];
  } else if (state.drag.edge !== undefined) {
    state.drag.block.polygon = resizeRectangleEdge(
      state.drag.original,
      state.drag.edge,
      point,
      state.current.image.width,
      state.current.image.height,
    );
  } else {
    const dx = point[0] - state.drag.start[0];
    const dy = point[1] - state.drag.start[1];
    state.drag.block.polygon = state.drag.original.map(([x, y]) => [
      Math.max(0, Math.min(state.current.image.width - 1, x + dx)),
      Math.max(0, Math.min(state.current.image.height - 1, y + dy)),
    ]);
  }
  renderDraggedBlock(state.drag.block);
}
function flushDragFrame() {
  if (!state.drag) return;
  state.drag.frame = null;
  const point = state.drag.pending;
  state.drag.pending = null;
  applyDragPoint(point);
}
function scheduleDragFrame(point) {
  if (!state.drag) return;
  state.drag.pending = point;
  if (state.drag.frame === null) {
    state.drag.frame = requestAnimationFrame(flushDragFrame);
  }
}
let panFrame = null;
function schedulePanFrame() {
  if (panFrame !== null) return;
  panFrame = requestAnimationFrame(() => {
    panFrame = null;
    applyPageView();
  });
}
window.addEventListener("pointermove", (event) => {
  if (state.pan) {
    const point = stagePoint(event);
    state.view = panCanvasView(
      state.pan.view,
      point[0] - state.pan.start[0],
      point[1] - state.pan.start[1],
    );
    state.viewMode = "custom";
    schedulePanFrame();
    return;
  }
  if (state.drag) scheduleDragFrame(eventPoint(event));
});
window.addEventListener("pointerup", (event) => {
  if (state.pan) {
    if (panFrame !== null) {
      cancelAnimationFrame(panFrame);
      panFrame = null;
      applyPageView();
    }
    state.pan = null;
    $("page-stage").classList.remove("panning");
    return;
  }
  if (state.drag) {
    if (state.drag.frame !== null) cancelAnimationFrame(state.drag.frame);
    applyDragPoint(eventPoint(event));
    state.drag.block.source = "manual";
    state.drag.block.score = null;
    state.drag = null;
    const metaEl = $("block-meta");
    if (metaEl) metaEl.textContent = "manual";
    markDirty();
  }
});
window.addEventListener("pointercancel", () => {
  if (panFrame !== null) {
    cancelAnimationFrame(panFrame);
    panFrame = null;
  }
  state.pan = null;
  $("page-stage").classList.remove("panning");
});
$("page-stage").onpointerdown = (event) => {
  const shouldPan = event.button === 1 || (event.button === 0 && state.spacePressed);
  if (shouldPan) {
    event.preventDefault();
    state.pan = { start: stagePoint(event), view: { ...state.view } };
    $("page-stage").classList.add("panning");
    if ($("page-stage").setPointerCapture) $("page-stage").setPointerCapture(event.pointerId);
    return;
  }
  if (editingLocked() || !state.add || event.target !== $("overlay")) return;
  event.preventDefault();
  const stage = $("page-stage");
  const start = eventPoint(event);
  const preview = svg("polygon");
  preview.classList.add("bbox", "draw-preview");
  $("overlay").append(preview);
  let pending = start;
  let frame = null;
  const flush = () => {
    frame = null;
    preview.setAttribute(
      "points",
      rectangle(start, pending).map(imageToScreen).map((point) => point.join(",")).join(" "),
    );
  };
  const move = (moveEvent) => {
    pending = eventPoint(moveEvent);
    if (frame === null) frame = requestAnimationFrame(flush);
  };
  const cleanup = () => {
    if (frame !== null) cancelAnimationFrame(frame);
    preview.remove();
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", finish);
    window.removeEventListener("pointercancel", cancel);
  };
  const leaveAddMode = () => {
    state.add = false;
    $("add-mode").textContent = "Thêm bbox";
  };
  const finish = (endEvent) => {
    const end = eventPoint(endEvent);
    cleanup();
    if (Math.abs(end[0] - start[0]) > 3 && Math.abs(end[1] - start[1]) > 3) {
      mutate(() => {
        const id = crypto.randomUUID();
        state.current.blocks.push({
          id, order: state.current.blocks.length, polygon: rectangle(start, end),
          layout_label: "text", task: "ocr", text: "", score: null,
          source: "manual", skipped: false,
        });
        state.selected = id;
        state.targetError = null;
      });
    }
    leaveAddMode();
  };
  const cancel = () => {
    cleanup();
    leaveAddMode();
  };
  flush();
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", finish);
  window.addEventListener("pointercancel", cancel);
  if (stage.setPointerCapture) stage.setPointerCapture(event.pointerId);
};

async function startBatch(operation) {
  if (currentOperation()) return;
  setBatchBusy(`batch-${operation}`);
  try {
    await pollBatch(await api(`/api/batch/${operation}`, {
      method: "POST",
      body: JSON.stringify({
        post_validate: operation === "prelabel" && state.validationEnabled,
      }),
    }));
  } catch (error) {
    setBatchBusy(null);
    setStatus(`Lỗi batch: ${error.message}`);
  }
}
async function pollBatch(snapshot) {
  $("batch-progress").max = snapshot.total || 1;
  $("batch-progress").value = snapshot.processed + snapshot.skipped + snapshot.failed;
  const validationFailures = snapshot.validation_failed
    ? ` · validation lỗi: ${snapshot.validation_failed}`
    : "";
  $("batch-status").textContent = `${snapshot.state}: ${snapshot.processed}/${snapshot.total}${validationFailures}`;
  if (["queued", "running", "cancelling"].includes(snapshot.state)) {
    setTimeout(async () => {
      try {
        await pollBatch(await api("/api/batch"));
      } catch (error) {
        setBatchBusy(null);
        setStatus(`Lỗi batch: ${error.message}`);
      }
    }, 750);
  } else {
    await loadImages();
    setBatchBusy(null);
    setStatus(`Batch ${snapshot.state}`);
  }
}
function setExportStatus(message, type = "loading") {
  const el = $("export-status");
  if (!el) return;
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    el.className = "export-status";
    return;
  }
  el.hidden = false;
  el.className = `export-status ${type}`;
  el.textContent = message;
}

async function exportDataset(endpoint, label) {
  if (!(await save())) return;
  const pathInput = $("export-path");
  const rawPath = pathInput.value.trim() || pathInput.placeholder.trim();
  if (!rawPath) {
    setExportStatus("Vui lòng nhập đường dẫn thư mục xuất", "danger");
    setStatus("Lỗi: thiếu đường dẫn xuất");
    return;
  }
  const exportButtons = [
    $("export-hf"),
    $("export-layout"),
    $("export-all"),
  ].filter(Boolean);
  exportButtons.forEach((btn) => (btn.disabled = true));
  setExportStatus(`Đang xuất ${label}…`, "loading");
  setStatus(`Đang xuất ${label}…`);
  try {
    const result = await api(endpoint, {
      method: "POST", body: JSON.stringify({ output_dir: rawPath }),
    });
    const count = result.samples ?? result.annotations ?? 0;
    const pages = result.pages ?? 0;
    const msg = `✓ ${label} thành công: ${count} mẫu (${pages} trang) tại ${result.path}`;
    setExportStatus(msg, "success");
    setStatus(`${label}: ${result.path}`);
  } catch (error) {
    setExportStatus(`⚠️ Lỗi export: ${error.message}`, "danger");
    setStatus(`Lỗi export: ${error.message}`);
  } finally {
    exportButtons.forEach((btn) => (btn.disabled = false));
  }
}

$("open-folder").onclick = async () => {
  const path = $("folder-path").value.trim();
  if (!path) {
    setStatus("Lỗi: hãy nhập đường dẫn folder");
    $("folder-path").focus();
    return;
  }
  setWorkspaceBusy(true, "Đang mở folder…");
  setStatus("Đang kết nối workspace…");
  try {
    await api("/api/workspace/open", { method: "POST", body: JSON.stringify({ path }) });
    $("workspace-message").textContent = "Đã mở folder, đang đọc danh sách ảnh…";
    setStatus("Đang đọc danh sách ảnh…");
    state.current = null;
    state.currentId = null;
    await loadImages();
    if (!state.images.length) {
      setStatus("Folder đã mở nhưng không có ảnh hợp lệ");
      return;
    }
    setStatus("Workspace đã mở");
  } catch (error) {
    setStatus(`Lỗi mở folder: ${error.message}`);
  } finally {
    setWorkspaceBusy(false);
  }
};
$("detect-current").onclick = () => runCurrent("detect", { replace_existing: true });
$("prelabel-page").onclick = () => runCurrent("prelabel", {
  block_ids: null, replace_existing: true, post_validate: state.validationEnabled,
});
$("prelabel-current").onclick = () => runCurrent("prelabel", {
  block_ids: state.selected ? [state.selected] : null,
  replace_existing: true,
  post_validate: state.validationEnabled,
});
$("validate-current").onclick = () => runCurrent("validate", { block_ids: null });
$("llm-validation").onchange = (event) => {
  state.validationEnabled = event.target.checked;
  saveValidationEnabled(window.localStorage, state.validationEnabled);
};
$("complete-current").onclick = async () => {
  const invalid = state.current?.blocks.find((block) => (
    !block.skipped && block.task !== null && !inspectTarget(block.task, block.text).valid
  ));
  if (invalid) {
    state.selected = invalid.id;
    state.targetError = null;
    render();
    setStatus(`Chưa thể Complete: ${inspectTarget(invalid.task, invalid.text).error}`);
    return;
  }
  await runCurrent("complete");
};
$("reopen-current").onclick = () => runCurrent("draft");
$("add-mode").onclick = () => { if (editingLocked()) return; state.add = !state.add; $("add-mode").textContent = state.add ? "Hủy thêm bbox" : "Thêm bbox"; };
$("delete-block").onclick = () => mutate(() => {
  state.current.blocks = state.current.blocks.filter((block) => block.id !== state.selected).map((block, index) => ({ ...block, order: index }));
  state.selected = state.current.blocks[0]?.id || null;
  state.targetError = null;
});
function markManual(block) { block.source = "manual"; block.score = null; }
$("layout-label").onchange = (event) => mutate(() => { const block = selectedBlock(); block.layout_label = event.target.value; markManual(block); });
$("task").onchange = (event) => mutate(() => {
  const block = selectedBlock();
  block.task = event.target.value || null;
  state.targetError = null;
  markManual(block);
});
$("text").oninput = (event) => {
  const block = selectedBlock();
  if (!block || block.task === null || editingLocked()) return;
  block.text = event.target.value;
  state.targetError = null;
  setTargetMessage(inspectTarget(block.task, block.text));
  markBlockEdited(block);
};
$("visual-tab").onclick = () => {
  state.targetMode = "visual";
  state.targetError = null;
  renderTargetEditor();
};
$("raw-tab").onclick = () => {
  state.targetMode = "raw";
  state.targetError = null;
  renderTargetEditor();
  $("text").focus();
};
for (const tab of [$("visual-tab"), $("raw-tab")]) {
  tab.onkeydown = (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const target = tab === $("visual-tab") ? $("raw-tab") : $("visual-tab");
    target.click();
    target.focus();
  };
}
$("starter-target").onclick = () => {
  const block = selectedBlock();
  if (!block?.task) return;
  commitVisualModel(block.task, createStarterModel(block.task), { rerender: true });
};
$("skipped").onchange = (event) => mutate(() => { const block = selectedBlock(); block.skipped = event.target.checked; markManual(block); });
$("detect-batch").onclick = () => startBatch("detect");
$("prelabel-batch").onclick = () => startBatch("prelabel");
$("cancel-batch").onclick = async () => pollBatch(await api("/api/batch", { method: "DELETE" }));
$("export-hf").onclick = () => exportDataset("/api/export/hf", "HF export");
$("export-layout").onclick = () => exportDataset("/api/export/layout", "Layout export");
$("export-all").onclick = () => exportDataset("/api/export/all", "Export All");
function setPageScale(scale, anchor = null) {
  const stage = $("page-stage");
  if (!state.current || !stage.clientWidth) return;
  state.view = zoomCanvasView(
    state.view,
    anchor || [stage.clientWidth / 2, stage.clientHeight / 2],
    scale,
  );
  state.viewMode = "custom";
  applyPageView();
}
$("page-fit").onclick = fitImage;
$("page-actual").onclick = () => setPageScale(1);
$("page-zoom-out").onclick = () => setPageScale(state.view.scale / 1.25);
$("page-zoom-in").onclick = () => setPageScale(state.view.scale * 1.25);
let wheelFrame = null;
let pendingScale = null;
let pendingAnchor = null;
$("page-stage").addEventListener("wheel", (event) => {
  if (!state.current) return;
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.0015);
  pendingScale = (pendingScale ?? state.view.scale) * factor;
  pendingAnchor = stagePoint(event);
  if (wheelFrame === null) {
    wheelFrame = requestAnimationFrame(() => {
      wheelFrame = null;
      if (pendingScale !== null) {
        setPageScale(pendingScale, pendingAnchor);
        pendingScale = null;
        pendingAnchor = null;
      }
    });
  }
}, { passive: false });
window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat) return;
  const target = event.target;
  if (target instanceof HTMLInputElement
      || target instanceof HTMLTextAreaElement
      || target instanceof HTMLSelectElement
      || target instanceof HTMLButtonElement) return;
  event.preventDefault();
  state.spacePressed = true;
  $("page-stage").classList.add("pan-ready");
});
window.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  state.spacePressed = false;
  $("page-stage").classList.remove("pan-ready");
});
window.addEventListener("blur", () => {
  state.spacePressed = false;
  state.pan = null;
  $("page-stage").classList.remove("pan-ready", "panning");
});
window.addEventListener("resize", () => {
  if (state.viewMode === "fit") fitImage();
  else applyPageView();
});

const [taxonomy, health] = await Promise.all([api("/api/taxonomy"), api("/api/health")]);
state.validationConfigured = Boolean(health.post_validation?.configured);
state.validationEnabled = loadValidationEnabled(
  window.localStorage, state.validationConfigured,
);
$("llm-validation").checked = state.validationEnabled;
$("llm-validation").title = state.validationConfigured
  ? `Model: ${health.post_validation.model}`
  : "Backend chưa cấu hình LLM validation";
for (const label of taxonomy.layout_labels) {
  const option = document.createElement("option");
  option.value = label;
  option.textContent = label;
  $("layout-label").append(option);
}
renderInteractionState();

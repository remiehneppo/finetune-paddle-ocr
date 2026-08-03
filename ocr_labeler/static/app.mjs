import {
  acknowledgeSave,
  addBlock,
  createEditorState,
  deleteBlock,
  moveBlock,
  moveCorner,
  redo,
  rebaseSaveAcknowledgement,
  reorderBlock,
  selectBlock,
  setStatus,
  undo,
  updateText,
} from "./state.mjs";
import { centerViewOnPolygon, rectanglePolygon, screenToImage } from "./geometry.mjs";
import {
  APIError,
  canNavigateAfterSave,
  canProcessInteraction,
  isCurrentResponse,
  needsDeleteConfirmation,
  nextBatchPollDelay,
  polygonClassNames,
  shouldApplyResponseError,
  shouldContinueBatchPoll,
  shouldPanPointer,
} from "./controller_helpers.mjs";

const $ = (id) => document.getElementById(id);
const svg = (name) => document.createElementNS("http://www.w3.org/2000/svg", name);
const controller = {
  task: "ocr",
  images: [],
  currentImageId: null,
  editor: null,
  view: { scale: 1, offsetX: 0, offsetY: 0 },
  mode: "select",
  autosaveTimer: null,
  savePromise: null,
  saveConflict: false,
  generation: 0,
  mutationVersion: 0,
  drag: null,
  batchTimer: null,
  batchPolling: false,
  batchPollPending: false,
  batchPollPendingDelay: 750,
  batchSnapshot: { state: "idle" },
  workspaceOpening: false,
  filter: "all",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(response.status, payload.detail ?? response.statusText);
  }
  return response.status === 204 ? null : response.json();
}

function make(tag, text, attributes = {}) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
  return element;
}

function selectedBlock() {
  return controller.editor?.annotation.blocks.find((block) => block.id === controller.editor.selectedId) ?? null;
}

function statusLabel(status) {
  const labels = controller.task === "detection"
    ? { all: "Tất cả", not_ocr: "Chưa detect", ocr: "Đã detect", edited: "Đã sửa", completed: "Hoàn tất", error: "Lỗi" }
    : { all: "Tất cả", not_ocr: "Chưa OCR", ocr: "Đã OCR", edited: "Đã sửa", completed: "Hoàn tất", error: "Lỗi" };
  return labels[status] ?? status;
}

function configureTask(task) {
  controller.task = task === "detection" ? "detection" : "ocr";
  const detection = controller.task === "detection";
  document.title = detection ? "PaddleOCR Detection Labeler" : "PaddleOCR Labeler";
  $("app-title").textContent = document.title;
  $("task-name").textContent = detection ? "Gán nhãn vùng chữ" : "Vẽ vùng văn bản";
  $("task-help").textContent = detection
    ? "Vẽ, kéo hoặc chỉnh 4 góc của bbox"
    : "Kéo trên trang để tạo vùng";
  $("ocr-current").textContent = detection ? "Detect ảnh này" : "OCR ảnh này";
  $("ocr-batch").textContent = detection ? "Detect toàn folder" : "OCR toàn folder";
  $("export-jsonl").textContent = detection ? "Xuất nhãn detection" : "Xuất JSONL";
  $("text-editor").hidden = detection;
  $("editor-label").hidden = detection;
  $("detection-controls").hidden = !detection;
  renderFilters();
}

function deleteSelectedBlock() {
  const block = selectedBlock();
  if (!block) return;
  const message = controller.task === "detection"
    ? "Xóa bbox đã chọn?"
    : "Xóa block có nội dung này?";
  if ((controller.task === "detection" || needsDeleteConfirmation(block)) && !window.confirm(message)) return;
  applyState(deleteBlock(controller.editor, block.id));
}

function setSaveStatus(text, conflict = false) {
  $("save-status").textContent = text;
  $("reload-current").hidden = !conflict;
}

function stagePoint(event) {
  const rect = $("page-stage").getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function imagePoint(event) {
  return screenToImage(stagePoint(event), controller.view);
}

function setMode(mode) {
  controller.mode = mode;
  $("page-stage").classList.toggle("is-adding", mode === "add");
  $("add-mode").textContent = mode === "add" ? "Hủy vẽ vùng" : "Vẽ vùng";
}

function fitImage() {
  const image = $("page-image");
  const stage = $("page-stage");
  if (!image.naturalWidth || !stage.clientWidth || !stage.clientHeight) return;
  const scale = Math.min(stage.clientWidth / image.naturalWidth, stage.clientHeight / image.naturalHeight, 1);
  controller.view = {
    scale,
    offsetX: (stage.clientWidth - image.naturalWidth * scale) / 2,
    offsetY: (stage.clientHeight - image.naturalHeight * scale) / 2,
  };
  renderViewport();
}

function renderViewport() {
  const image = $("page-image");
  const overlay = $("overlay");
  const { scale, offsetX, offsetY } = controller.view;
  if (!image.naturalWidth) return;
  Object.assign(image.style, {
    left: `${offsetX}px`, top: `${offsetY}px`, width: `${image.naturalWidth * scale}px`, height: `${image.naturalHeight * scale}px`,
  });
  Object.assign(overlay.style, {
    left: `${offsetX}px`, top: `${offsetY}px`, width: `${image.naturalWidth * scale}px`, height: `${image.naturalHeight * scale}px`,
  });
  overlay.setAttribute("viewBox", `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  $("zoom-level").textContent = `${Math.round(scale * 100)}%`;
  renderOverlay();
}

function polygonFor(block) {
  if (!controller.drag || controller.drag.id !== block.id) return block.polygon;
  return controller.drag.polygon;
}

function points(polygon) {
  return polygon.map(([x, y]) => `${x},${y}`).join(" ");
}

function renderOverlay() {
  const overlay = $("overlay");
  overlay.replaceChildren();
  if (!controller.editor) return;
  controller.editor.annotation.blocks.forEach((block, index) => {
    const group = svg("g");
    group.dataset.blockId = block.id;
    group.classList.add("annotation-block");
    const polygon = svg("polygon");
    polygon.setAttribute("points", points(polygonFor(block)));
    polygon.setAttribute(
      "class",
      polygonClassNames(block, block.id === controller.editor.selectedId),
    );
    polygon.dataset.blockId = block.id;
    group.append(polygon);
    if (block.id === controller.editor.selectedId) {
      polygonFor(block).forEach(([x, y], corner) => {
        const handle = svg("circle");
        handle.setAttribute("cx", x);
        handle.setAttribute("cy", y);
        handle.setAttribute("r", 5 / controller.view.scale);
        handle.setAttribute("class", "polygon-handle");
        handle.dataset.blockId = block.id;
        handle.dataset.corner = corner;
        group.append(handle);
      });
    }
    const label = svg("text");
    const [x, y] = polygonFor(block)[0];
    label.setAttribute("x", x + 4 / controller.view.scale);
    label.setAttribute("y", y - 5 / controller.view.scale);
    label.setAttribute("class", "polygon-label");
    label.textContent = String(index + 1);
    group.append(label);
    overlay.append(group);
  });
  if (controller.drag?.kind === "add") {
    const preview = svg("polygon");
    preview.setAttribute("points", points(controller.drag.polygon));
    preview.setAttribute("class", "polygon--manual polygon--preview");
    overlay.append(preview);
  }
}

function renderImageList() {
  const list = $("image-list");
  const term = $("image-search").value.trim().toLocaleLowerCase();
  list.replaceChildren();
  const visible = controller.images.filter((image) =>
    (controller.filter === "all" || image.status === controller.filter)
    && image.name.toLocaleLowerCase().includes(term),
  );
  visible.forEach((image) => {
    const item = make("li");
    const button = make("button", image.name, { type: "button" });
    button.className = `image-item image-item--${image.status}`;
    button.disabled = !canProcessInteraction(controller);
    button.classList.toggle("is-selected", image.image_id === controller.currentImageId);
    button.title = image.error || image.status;
    button.addEventListener("click", () => selectImage(image.image_id));
    item.append(button);
    list.append(item);
  });
  $("image-count").textContent = `${controller.images.length} ảnh`;
}

function renderFilters() {
  const filters = $("image-filters");
  filters.replaceChildren();
  ["all", "not_ocr", "ocr", "edited", "completed", "error"].forEach((status) => {
    const button = make("button", statusLabel(status), { type: "button" });
    button.className = "filter-button";
    button.classList.toggle("is-active", controller.filter === status);
    button.addEventListener("click", () => { controller.filter = status; renderFilters(); renderImageList(); });
    filters.append(button);
  });
}

function renderBlockList() {
  const list = $("block-list");
  list.replaceChildren();
  const blocks = controller.editor?.annotation.blocks ?? [];
  $("block-count").textContent = `${blocks.length} khối`;
  blocks.forEach((block, index) => {
    const item = make("li");
    item.className = "block-item";
    item.draggable = canProcessInteraction(controller);
    item.dataset.blockId = block.id;
    item.classList.toggle("is-selected", block.id === controller.editor.selectedId);
    const blockTitle = controller.task === "detection"
      ? (block.text === "###" ? "Bỏ qua" : "Vùng văn bản")
      : (block.text || "(trống)");
    const title = make("button", String(index + 1) + ". " + blockTitle, { type: "button" });
    title.className = "block-select";
    title.disabled = !canProcessInteraction(controller);
    title.addEventListener("click", () => {
      applyState(selectBlock(controller.editor, block.id), { save: false });
      const stage = $("page-stage");
      controller.view = centerViewOnPolygon(block.polygon, controller.view, {
        width: stage.clientWidth,
        height: stage.clientHeight,
      });
      renderViewport();
    });
    item.append(title);
    if (block.score !== null && block.score !== undefined) {
      const confidence = make("span", ` ${(block.score * 100).toFixed(0)}%`);
      confidence.className = block.score < 0.6 ? "confidence confidence--low" : "confidence";
      item.append(confidence);
    }
    item.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/plain", block.id));
    item.addEventListener("dragover", (event) => event.preventDefault());
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      const source = event.dataTransfer.getData("text/plain");
      if (source && source !== block.id) applyState(reorderBlock(controller.editor, source, index));
    });
    list.append(item);
  });
}

function renderInspector() {
  renderBlockList();
  const block = selectedBlock();
  const editor = $("text-editor");
  const pointEditor = $("point-editor");
  editor.disabled = !block || !canProcessInteraction(controller);
  editor.value = block?.text ?? "";
  const detection = controller.task === "detection";
  editor.hidden = detection;
  $("editor-label").hidden = detection;
  $("detection-controls").hidden = !detection;
  $("ignore-region").disabled = !block || !canProcessInteraction(controller);
  $("ignore-region").checked = block?.text === "###";
  $("delete-block").disabled = !block || !canProcessInteraction(controller);
  pointEditor.replaceChildren();
  if (block) {
    block.polygon.forEach(([x, y], corner) => {
      const row = make("label", `Điểm ${corner + 1}`);
      row.className = "point-row";
      [["x", x], ["y", y]].forEach(([axis, value]) => {
        const input = make("input", undefined, { type: "number", step: "0.1", value: String(value), "aria-label": `Điểm ${corner + 1} ${axis}` });
        input.disabled = !canProcessInteraction(controller);
        input.addEventListener("change", () => {
          const next = Number(input.value);
          if (!Number.isFinite(next)) return;
          const current = selectedBlock().polygon[corner];
          applyState(moveCorner(controller.editor, block.id, corner, axis === "x" ? next : current[0], axis === "y" ? next : current[1]));
        });
        row.append(input);
      });
      pointEditor.append(row);
    });
  }
  drawCrop(block);
}

function drawCrop(block) {
  const canvas = $("crop-preview");
  const image = $("page-image");
  const context = canvas.getContext("2d");
  canvas.width = 1; canvas.height = 1;
  if (!block || !image.naturalWidth) return;
  const xs = block.polygon.map(([x]) => x);
  const ys = block.polygon.map(([, y]) => y);
  const left = Math.max(0, Math.floor(Math.min(...xs)));
  const top = Math.max(0, Math.floor(Math.min(...ys)));
  const width = Math.max(1, Math.ceil(Math.max(...xs)) - left);
  const height = Math.max(1, Math.ceil(Math.max(...ys)) - top);
  const ratio = Math.min(340 / width, 180 / height, 1);
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  context.drawImage(image, left, top, width, height, 0, 0, canvas.width, canvas.height);
}

function render() {
  renderImageList();
  renderViewport();
  renderInspector();
  $("toggle-completed").textContent =
    controller.editor?.annotation.status === "completed"
      ? "Mở lại để sửa"
      : "Đánh dấu hoàn tất";
}

function applyState(next, { save = true } = {}) {
  if (!canProcessInteraction(controller)) return controller.editor;
  controller.editor = next;
  if (save) controller.mutationVersion += 1;
  if (next.annotation) updateCurrentImageStatus(next.annotation.status);
  render();
  if (save && next.dirty) scheduleAutosave();
}

function currentResponseToken() {
  return {
    generation: controller.generation,
    imageId: controller.currentImageId,
    mutationVersion: controller.mutationVersion,
  };
}

function updateCurrentImageStatus(status) {
  if (!controller.currentImageId) return;
  controller.images = controller.images.map((image) => image.image_id === controller.currentImageId ? { ...image, status } : image);
}

function setWorkspaceOpening(opening) {
  controller.workspaceOpening = opening;
  ["open-folder", "folder-path", "ocr-current", "ocr-batch", "toggle-completed", "reload-current", "add-mode", "export-jsonl", "delete-block", "ignore-region"].forEach((id) => {
    const element = $(id);
    if (element) element.disabled = opening;
  });
  $("cancel-batch").disabled = opening || !["queued", "running"].includes(controller.batchSnapshot.state);
  $("page-stage").setAttribute("aria-busy", String(opening));
  render();
}

function scheduleAutosave() {
  if (controller.saveConflict) return;
  clearTimeout(controller.autosaveTimer);
  setSaveStatus("Đang lưu");
  controller.autosaveTimer = setTimeout(() => { controller.autosaveTimer = null; saveCurrent(); }, 500);
}

async function saveCurrent() {
  if (controller.savePromise) return controller.savePromise;
  if (controller.saveConflict) return false;
  if (!controller.editor?.dirty || !controller.currentImageId) return true;
  const editor = controller.editor;
  const request = currentResponseToken();
  controller.savePromise = (async () => {
    try {
      const saved = await api(`/api/images/${request.imageId}/annotation`, { method: "PUT", body: JSON.stringify(editor.annotation) });
      if (controller.currentImageId === request.imageId && controller.generation === request.generation) {
        if (controller.mutationVersion === request.mutationVersion) {
          controller.editor = acknowledgeSave(controller.editor, saved);
        } else {
          controller.editor = rebaseSaveAcknowledgement(controller.editor, saved);
        }
        controller.mutationVersion += 1;
        controller.saveConflict = false;
        updateCurrentImageStatus(saved.status);
        setSaveStatus("Đã lưu");
        render();
      }
      return true;
    } catch (error) {
      const conflict = error instanceof APIError && error.status === 409;
      if (conflict) controller.saveConflict = true;
      setSaveStatus(conflict ? "Xung đột" : "Lỗi lưu", conflict);
      return false;
    } finally {
      controller.savePromise = null;
      if (controller.currentImageId === request.imageId && controller.generation === request.generation && controller.editor?.dirty && controller.mutationVersion !== request.mutationVersion && !controller.saveConflict) scheduleAutosave();
    }
  })();
  return controller.savePromise;
}

async function flushPendingSave() {
  let succeeded = true;
  if (controller.autosaveTimer) {
    clearTimeout(controller.autosaveTimer);
    controller.autosaveTimer = null;
    succeeded = await saveCurrent();
  }
  if (succeeded && controller.savePromise) succeeded = await controller.savePromise;
  if (succeeded && controller.editor?.dirty) {
    clearTimeout(controller.autosaveTimer);
    controller.autosaveTimer = null;
    succeeded = await saveCurrent();
  }
  return canNavigateAfterSave({ succeeded, dirty: Boolean(controller.editor?.dirty) });
}

async function refreshImages() {
  const payload = await api("/api/images");
  controller.images = payload.images;
  renderImageList();
}

async function selectImage(imageId) {
  if (!canProcessInteraction(controller)) return;
  if (imageId === controller.currentImageId) return;
  try {
    if (!canNavigateAfterSave(await flushPendingSave())) return;
    if (!canProcessInteraction(controller)) return;
    controller.generation += 1;
    controller.currentImageId = imageId;
    controller.editor = null;
    controller.saveConflict = false;
    controller.drag = null;
    const request = currentResponseToken();
    setMode("select");
    setSaveStatus("Đang tải");
    const image = $("page-image");
    image.onload = () => {
      if (isCurrentResponse(request, currentResponseToken())) fitImage();
    };
    image.src = `/api/images/${imageId}/content`;
    try {
      const annotation = await api(`/api/images/${imageId}/annotation`);
      if (!isCurrentResponse(request, currentResponseToken())) return;
      controller.editor = createEditorState(annotation);
      setSaveStatus("Đã lưu");
    } catch (error) {
      if (!isCurrentResponse(request, currentResponseToken())) return;
      if (error instanceof APIError && error.status === 409) {
        controller.saveConflict = true;
        setSaveStatus("Xung đột", true);
      } else if (!/unknown image|not found/i.test(error.message)) setSaveStatus(error.message);
      else setSaveStatus(controller.task === "detection" ? "Chưa detect" : "Chưa OCR");
    }
    render();
  } catch (error) {
    setSaveStatus(`Lỗi: ${error.message}`);
  }
}

async function openFolder() {
  if (!canProcessInteraction(controller)) return;
  try {
    if (!canNavigateAfterSave(await flushPendingSave())) return;
    if (!canProcessInteraction(controller)) return;
    controller.generation += 1;
    setWorkspaceOpening(true);
    setSaveStatus("Đang mở folder");
    const result = await api("/api/workspace/open", { method: "POST", body: JSON.stringify({ path: $("folder-path").value.trim() }) });
    $("folder-path").value = result.root;
    controller.currentImageId = null;
    controller.editor = null;
    controller.saveConflict = false;
    controller.mutationVersion += 1;
    await refreshImages();
    renderInspector();
    setSaveStatus("Chọn một ảnh");
  } catch (error) { setSaveStatus(`Lỗi: ${error.message}`); }
  finally { setWorkspaceOpening(false); }
}

function startDrag(event, kind, block = null, corner = null) {
  const start = imagePoint(event);
  $("overlay").setPointerCapture(event.pointerId);
  controller.drag = {
    kind, id: block?.id, corner, start, startPolygon: block?.polygon, polygon: block?.polygon,
    startScreen: stagePoint(event), startOffset: { ...controller.view },
  };
}

function onPointerDown(event) {
  if (!canProcessInteraction(controller)) return;
  if (![0, 1].includes(event.button) || !$("page-image").naturalWidth) return;
  const handle = event.target.closest?.(".polygon-handle");
  const blockElement = event.target.closest?.("[data-block-id]");
  if (shouldPanPointer(event.button, controller.spacePan)) {
    event.preventDefault();
    startDrag(event, "pan");
    return;
  }
  if (event.button !== 0) return;
  if (handle && controller.editor) {
    const block = controller.editor.annotation.blocks.find((item) => item.id === handle.dataset.blockId);
    startDrag(event, "corner", block, Number(handle.dataset.corner));
    event.preventDefault();
    return;
  }
  if (blockElement && controller.editor) {
    const block = controller.editor.annotation.blocks.find((item) => item.id === blockElement.dataset.blockId);
    if (block) { applyState(selectBlock(controller.editor, block.id), { save: false }); startDrag(event, "move", block); }
    return;
  }
  if (controller.mode === "add" && controller.editor) {
    startDrag(event, "add");
    controller.drag.polygon = rectanglePolygon(controller.drag.start, controller.drag.start);
  } else if (controller.editor) applyState(selectBlock(controller.editor, null), { save: false });
}

function onPointerMove(event) {
  if (!canProcessInteraction(controller)) { cancelDrag(); return; }
  const drag = controller.drag;
  if (!drag) return;
  if (drag.kind === "pan") {
    const [x, y] = stagePoint(event);
    controller.view.offsetX = drag.startOffset.offsetX + x - drag.startScreen[0];
    controller.view.offsetY = drag.startOffset.offsetY + y - drag.startScreen[1];
    renderViewport();
    return;
  }
  const point = imagePoint(event);
  if (drag.kind === "add") drag.polygon = rectanglePolygon(drag.start, point);
  if (drag.kind === "corner") drag.polygon = drag.startPolygon.map((item, index) => index === drag.corner ? point : item);
  if (drag.kind === "move") {
    const dx = point[0] - drag.start[0]; const dy = point[1] - drag.start[1];
    drag.polygon = drag.startPolygon.map(([x, y]) => [x + dx, y + dy]);
  }
  renderOverlay();
}

function onPointerUp(event) {
  if (!canProcessInteraction(controller)) { cancelDrag(); return; }
  const drag = controller.drag;
  if (!drag) return;
  controller.drag = null;
  if ($("overlay").hasPointerCapture(event.pointerId)) $("overlay").releasePointerCapture(event.pointerId);
  if (drag.kind === "corner") {
    const [x, y] = imagePoint(event);
    applyState(moveCorner(controller.editor, drag.id, drag.corner, x, y));
  } else if (drag.kind === "move") {
    const end = imagePoint(event);
    applyState(moveBlock(controller.editor, drag.id, end[0] - drag.start[0], end[1] - drag.start[1]));
  } else if (drag.kind === "add") {
    const end = imagePoint(event);
    if (Math.abs(end[0] - drag.start[0]) > 2 && Math.abs(end[1] - drag.start[1]) > 2) applyState(addBlock(controller.editor, rectanglePolygon(drag.start, end), controller.task === "detection" ? "text" : ""));
    setMode("select");
  } else renderOverlay();
}

function cancelDrag() { controller.drag = null; setMode("select"); renderOverlay(); }

function zoomAt(factor, screen = null) {
  const stage = $("page-stage");
  const pivot = screen ?? [stage.clientWidth / 2, stage.clientHeight / 2];
  const image = screenToImage(pivot, controller.view);
  controller.view.scale = Math.min(8, Math.max(0.1, controller.view.scale * factor));
  controller.view.offsetX = pivot[0] - image[0] * controller.view.scale;
  controller.view.offsetY = pivot[1] - image[1] * controller.view.scale;
  renderViewport();
}

function onKeyDown(event) {
  if (!canProcessInteraction(controller)) return;
  const typing = /INPUT|TEXTAREA|SELECT/.test(event.target.tagName);
  if (event.code === "Space" && !typing) {
    controller.spacePan = true;
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    if (!controller.editor) return;
    event.preventDefault();
    applyState(event.shiftKey ? redo(controller.editor) : undo(controller.editor));
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    if (!controller.editor) return;
    event.preventDefault(); applyState(redo(controller.editor));
  } else if (event.key === "Delete" && !typing && controller.editor?.selectedId) {
    event.preventDefault();
    deleteSelectedBlock();
  } else if (event.key.toLowerCase() === "a" && !typing) {
    event.preventDefault(); setMode(controller.mode === "add" ? "select" : "add");
  } else if (event.key === "Escape") { cancelDrag(); }
}

function onKeyUp(event) {
  if (event.code === "Space") controller.spacePan = false;
}

async function runCurrentOcr() {
  if (!canProcessInteraction(controller)) return;
  if (!controller.currentImageId) return;
  const operation = controller.task === "detection" ? "Detect" : "OCR";
  if (!window.confirm(operation + " sẽ thay thế toàn bộ annotation hiện có. Tiếp tục?")) return;
  let request = null;
  try {
    if (!canNavigateAfterSave(await flushPendingSave())) return;
    if (!canProcessInteraction(controller)) return;
    controller.generation += 1;
    request = currentResponseToken();
    const endpoint = controller.task === "detection" ? "detect" : "ocr";
    const annotation = await api("/api/images/" + request.imageId + "/" + endpoint, { method: "POST", body: JSON.stringify({ replace_existing: true }) });
    if (!isCurrentResponse(request, currentResponseToken())) return;
    controller.editor = { ...createEditorState(annotation), dirty: true };
    controller.mutationVersion += 1;
    updateCurrentImageStatus(annotation.status);
    render();
    scheduleAutosave();
  } catch (error) {
    if (request && !shouldApplyResponseError(request, currentResponseToken())) return;
    if (error instanceof APIError && error.status === 409) {
      controller.saveConflict = true;
      setSaveStatus("Xung đột", true);
    } else setSaveStatus("Lỗi " + operation + ": " + error.message);
  }
}

async function toggleCompleted() {
  if (!canProcessInteraction(controller)) return;
  if (!controller.editor) return;
  try {
    const status = controller.editor.annotation.status === "completed" ? "edited" : "completed";
    applyState(setStatus(controller.editor, status));
  } catch (error) { setSaveStatus(error.message); }
}

function updateBatch(snapshot) {
  controller.batchSnapshot = snapshot;
  const total = snapshot.total || 1;
  $("batch-progress").max = total;
  $("batch-progress").value = Math.min(total, snapshot.processed + snapshot.skipped + snapshot.failed);
  $("cancel-batch").disabled = !["queued", "running", "cancelling"].includes(snapshot.state) || snapshot.state === "cancelling";
  $("model-status").textContent = snapshot.current_image ? `${snapshot.state}: ${snapshot.current_image}` : `Batch: ${snapshot.state}`;
}

function scheduleBatchPoll(delay = 750) {
  if (controller.batchPolling) {
    controller.batchPollPending = true;
    controller.batchPollPendingDelay = Math.min(controller.batchPollPendingDelay, delay);
    return;
  }
  clearTimeout(controller.batchTimer);
  controller.batchTimer = setTimeout(() => {
    controller.batchTimer = null;
    pollBatch();
  }, delay);
}

async function pollBatch() {
  if (controller.batchPolling) return;
  controller.batchPolling = true;
  let active = false;
  let terminal = false;
  try {
    const snapshot = await api("/api/batch");
    updateBatch(snapshot);
    active = shouldContinueBatchPoll(snapshot);
    terminal = !active;
    if (terminal && snapshot.state !== "idle") {
      clearTimeout(controller.batchTimer); controller.batchTimer = null;
      await refreshImages();
    }
  } catch (error) {
    $("model-status").textContent = `Lỗi batch: ${error.message}`;
    active = shouldContinueBatchPoll(controller.batchSnapshot);
    terminal = !active;
  } finally {
    const pending = controller.batchPollPending;
    controller.batchPollPending = false;
    const pendingDelay = controller.batchPollPendingDelay;
    controller.batchPollPendingDelay = 750;
    controller.batchPolling = false;
    const delay = nextBatchPollDelay({ active, pending, terminal });
    if (delay !== null) scheduleBatchPoll(pending ? Math.min(delay, pendingDelay) : delay);
  }
}

async function startBatch() {
  if (!canProcessInteraction(controller)) return;
  try {
    updateBatch(await api("/api/batch", { method: "POST" }));
    scheduleBatchPoll(0);
  } catch (error) { $("model-status").textContent = `Lỗi batch: ${error.message}`; }
}

async function cancelBatch() {
  if (!canProcessInteraction(controller)) return;
  $("cancel-batch").disabled = true;
  try {
    updateBatch(await api("/api/batch/cancel", { method: "POST" }));
    if (shouldContinueBatchPoll(controller.batchSnapshot)) scheduleBatchPoll(0);
  } catch (error) {
    updateBatch(controller.batchSnapshot);
    if (shouldContinueBatchPoll(controller.batchSnapshot)) scheduleBatchPoll(0);
    $("model-status").textContent = `Lỗi dừng: ${error.message}`;
  }
}

async function exportJsonl() {
  if (!canProcessInteraction(controller)) return;
  try {
    if (!canNavigateAfterSave(await flushPendingSave())) return;
    if (!canProcessInteraction(controller)) return;
    const result = await api("/api/export", { method: "POST" });
    $("model-status").textContent = `Xuất ${result.records} bản ghi: ${result.path}`;
  } catch (error) { $("model-status").textContent = `Lỗi xuất: ${error.message}`; }
}

async function reloadCurrent() {
  if (!canProcessInteraction(controller)) return;
  if (!controller.currentImageId) return;
  let request = null;
  try {
    controller.generation += 1;
    request = currentResponseToken();
    const annotation = await api(`/api/images/${request.imageId}/annotation`);
    if (!isCurrentResponse(request, currentResponseToken())) return;
    controller.editor = createEditorState(annotation);
    controller.saveConflict = false;
    setSaveStatus("Đã tải lại"); render();
  } catch (error) {
    if (request && !shouldApplyResponseError(request, currentResponseToken())) return;
    if (error instanceof APIError && error.status === 409) {
      controller.saveConflict = true;
      setSaveStatus("Xung đột", true);
    } else setSaveStatus(`Lỗi tải lại: ${error.message}`);
  }
}

function bindEvents() {
  $("reload-current").addEventListener("click", reloadCurrent);
  $("open-folder").addEventListener("click", openFolder);
  $("folder-path").addEventListener("keydown", (event) => { if (event.key === "Enter") openFolder(); });
  $("image-search").addEventListener("input", renderImageList);
  $("ocr-current").addEventListener("click", runCurrentOcr);
  $("ocr-batch").addEventListener("click", startBatch);
  $("cancel-batch").addEventListener("click", cancelBatch);
  $("export-jsonl").addEventListener("click", exportJsonl);
  $("toggle-completed").addEventListener("click", toggleCompleted);
  $("add-mode").addEventListener("click", () => setMode(controller.mode === "add" ? "select" : "add"));
  $("zoom-in").addEventListener("click", () => zoomAt(1.2));
  $("zoom-out").addEventListener("click", () => zoomAt(1 / 1.2));
  $("overlay").addEventListener("pointerdown", onPointerDown);
  $("overlay").addEventListener("pointermove", onPointerMove);
  $("overlay").addEventListener("pointerup", onPointerUp);
  $("overlay").addEventListener("pointercancel", cancelDrag);
  $("page-stage").addEventListener("wheel", (event) => { event.preventDefault(); zoomAt(event.deltaY < 0 ? 1.12 : 1 / 1.12, stagePoint(event)); }, { passive: false });
  $("delete-block").addEventListener("click", deleteSelectedBlock);
  $("ignore-region").addEventListener("change", (event) => {
    if (!canProcessInteraction(controller)) return;
    const block = selectedBlock();
    if (block) applyState(updateText(controller.editor, block.id, event.target.checked ? "###" : "text"));
  });
  $("text-editor").addEventListener("input", (event) => {
    if (!canProcessInteraction(controller)) return;
    const block = selectedBlock(); if (block) applyState(updateText(controller.editor, block.id, event.target.value));
  });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  window.addEventListener("resize", () => renderViewport());
}

async function start() {
  const reload = make("button", "Tải lại", { id: "reload-current", type: "button" });
  reload.hidden = true;
  $("save-status").after(reload);
  const add = make("button", "Vẽ vùng", { id: "add-mode", type: "button" });
  $("zoom-out").before(add);
  $("cancel-batch").disabled = true;
  bindEvents(); renderFilters(); renderInspector();
  try {
    const health = await api("/api/health");
    configureTask(health.task);
    $("model-status").textContent = "Sẵn sàng (" + health.device + ")";
    if (health.workspace) {
      $("folder-path").value = health.workspace;
      await refreshImages();
    }
    await pollBatch();
  } catch (error) { $("model-status").textContent = `Lỗi kết nối: ${error.message}`; }
}

start();

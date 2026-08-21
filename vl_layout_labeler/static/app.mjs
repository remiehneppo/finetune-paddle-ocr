const $ = (id) => document.getElementById(id);
const svg = (name) => document.createElementNS("http://www.w3.org/2000/svg", name);
const state = {
  images: [], current: null, currentId: null, selected: null,
  view: { scale: 1, x: 0, y: 0 }, add: false, dirty: false, timer: null, drag: null,
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
function selectedBlock() { return state.current?.blocks.find((block) => block.id === state.selected) || null; }
function imageToScreen([x, y]) { return [x * state.view.scale + state.view.x, y * state.view.scale + state.view.y]; }
function screenToImage([x, y]) { return [(x - state.view.x) / state.view.scale, (y - state.view.y) / state.view.scale]; }
function rectangle(start, end) {
  const left = Math.min(start[0], end[0]);
  const right = Math.max(start[0], end[0]);
  const top = Math.min(start[1], end[1]);
  const bottom = Math.max(start[1], end[1]);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

function fitImage() {
  const image = $("page-image");
  const stage = $("page-stage");
  if (!image.naturalWidth || !stage.clientWidth) return;
  state.view.scale = Math.min(
    stage.clientWidth / image.naturalWidth,
    stage.clientHeight / image.naturalHeight,
    1,
  );
  state.view.x = (stage.clientWidth - image.naturalWidth * state.view.scale) / 2;
  state.view.y = (stage.clientHeight - image.naturalHeight * state.view.scale) / 2;
  Object.assign(image.style, {
    left: `${state.view.x}px`, top: `${state.view.y}px`,
    width: `${image.naturalWidth * state.view.scale}px`,
    height: `${image.naturalHeight * state.view.scale}px`,
  });
  render();
}

function renderImages() {
  const list = $("image-list");
  list.replaceChildren();
  $("image-count").textContent = state.images.length;
  for (const image of state.images) {
    const item = document.createElement("li");
    if (image.image_id === state.currentId) item.classList.add("selected");
    const name = document.createElement("span");
    const status = document.createElement("span");
    name.className = "name";
    status.className = `status${image.error ? " error" : ""}`;
    name.textContent = image.name;
    status.textContent = image.error || image.status;
    item.append(name, status);
    item.onclick = () => loadImage(image.image_id);
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
    item.textContent = `${block.order + 1}. ${task} · ${content}`;
    item.onclick = () => { state.selected = block.id; render(); };
    list.append(item);
  }
}

function renderInspector() {
  const block = selectedBlock();
  $("editor").hidden = !block;
  $("prelabel-current").disabled = !block || block.task === null || block.skipped;
  if (!block) return;
  $("layout-label").value = block.layout_label;
  $("task").value = block.task || "";
  $("text").value = block.text;
  $("text").disabled = block.task === null;
  $("layout-only-note").hidden = block.task !== null;
  $("skipped").checked = block.skipped;
  $("block-meta").textContent = `${block.source}${block.score == null ? "" : ` · ${(block.score * 100).toFixed(1)}%`}`;
}

function render() {
  renderBlocks();
  renderInspector();
  const overlay = $("overlay");
  overlay.replaceChildren();
  for (const block of state.current?.blocks || []) {
    const element = svg("polygon");
    element.setAttribute("points", block.polygon.map(imageToScreen).map((point) => point.join(",")).join(" "));
    element.classList.add("bbox", block.task || "layout-only");
    if (block.id === state.selected) element.classList.add("selected");
    if (block.skipped) element.classList.add("skipped");
    element.onclick = (event) => { event.stopPropagation(); state.selected = block.id; render(); };
    element.onpointerdown = (event) => beginDrag(event, block);
    overlay.append(element);
    if (block.id === state.selected) {
      block.polygon.map(imageToScreen).forEach(([x, y], index) => {
        const corner = svg("circle");
        corner.setAttribute("cx", x);
        corner.setAttribute("cy", y);
        corner.setAttribute("r", 6);
        corner.classList.add("corner");
        corner.onpointerdown = (event) => beginCornerDrag(event, block, index);
        overlay.append(corner);
      });
    }
  }
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
  $("current-name").textContent = state.images.find((item) => item.image_id === id)?.name || state.current.image.path;
  $("page-image").src = `/api/images/${id}/content?${Date.now()}`;
  $("page-image").onload = fitImage;
  render();
  renderImages();
  setStatus("Đã tải");
}

function markDirty() {
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
  if (!state.current) return;
  callback();
  if (state.current.status === "completed") state.current.status = "edited";
  render();
  markDirty();
}

async function runCurrent(action, body) {
  if (!state.currentId || !(await save())) return;
  try {
    const options = { method: "POST" };
    if (body !== undefined) options.body = JSON.stringify(body);
    state.current = await api(`/api/images/${state.currentId}/${action}`, options);
    state.selected = state.current.blocks[0]?.id || null;
    render();
    await loadImages();
    setStatus("Hoàn tất");
  } catch (error) {
    setStatus(`Lỗi: ${error.message}`);
  }
}

function eventPoint(event) {
  const bounds = $("page-stage").getBoundingClientRect();
  return screenToImage([event.clientX - bounds.left, event.clientY - bounds.top]);
}
function beginDrag(event, block) {
  if (state.add) return;
  event.preventDefault();
  state.selected = block.id;
  state.drag = { block, start: eventPoint(event), original: block.polygon.map((point) => [...point]) };
}
function beginCornerDrag(event, block, corner) {
  event.preventDefault();
  event.stopPropagation();
  state.drag = { block, corner };
}
window.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const point = eventPoint(event);
  if (state.drag.corner !== undefined) {
    state.drag.block.polygon[state.drag.corner] = [
      Math.max(0, Math.min(state.current.image.width - 1, point[0])),
      Math.max(0, Math.min(state.current.image.height - 1, point[1])),
    ];
  } else {
    const dx = point[0] - state.drag.start[0];
    const dy = point[1] - state.drag.start[1];
    state.drag.block.polygon = state.drag.original.map(([x, y]) => [
      Math.max(0, Math.min(state.current.image.width - 1, x + dx)),
      Math.max(0, Math.min(state.current.image.height - 1, y + dy)),
    ]);
  }
  render();
});
window.addEventListener("pointerup", () => {
  if (state.drag) {
    state.drag.block.source = "manual";
    state.drag.block.score = null;
    state.drag = null;
    if (state.current.status === "completed") state.current.status = "edited";
    markDirty();
  }
});
$("page-stage").onpointerdown = (event) => {
  if (!state.add || event.target !== $("overlay")) return;
  const start = eventPoint(event);
  const finish = (endEvent) => {
    const end = eventPoint(endEvent);
    if (Math.abs(end[0] - start[0]) > 3 && Math.abs(end[1] - start[1]) > 3) {
      mutate(() => {
        const id = crypto.randomUUID();
        state.current.blocks.push({
          id, order: state.current.blocks.length, polygon: rectangle(start, end),
          layout_label: "text", task: "ocr", text: "", score: null,
          source: "manual", skipped: false,
        });
        state.selected = id;
      });
    }
    state.add = false;
    $("add-mode").textContent = "Thêm bbox";
    window.removeEventListener("pointerup", finish);
  };
  window.addEventListener("pointerup", finish);
};

async function startBatch(operation) {
  try { pollBatch(await api(`/api/batch/${operation}`, { method: "POST" })); }
  catch (error) { setStatus(`Lỗi batch: ${error.message}`); }
}
async function pollBatch(snapshot) {
  $("batch-progress").max = snapshot.total || 1;
  $("batch-progress").value = snapshot.processed + snapshot.skipped + snapshot.failed;
  $("batch-status").textContent = `${snapshot.state}: ${snapshot.processed}/${snapshot.total}`;
  if (["queued", "running", "cancelling"].includes(snapshot.state)) {
    setTimeout(async () => pollBatch(await api("/api/batch")), 750);
  } else {
    await loadImages();
    setStatus(`Batch ${snapshot.state}`);
  }
}
async function exportDataset(endpoint, label) {
  if (!(await save())) return;
  try {
    const result = await api(endpoint, {
      method: "POST", body: JSON.stringify({ output_dir: $("export-path").value }),
    });
    setStatus(`${label}: ${result.path}`);
  } catch (error) {
    setStatus(`Lỗi export: ${error.message}`);
  }
}

$("open-folder").onclick = async () => {
  try {
    await api("/api/workspace/open", { method: "POST", body: JSON.stringify({ path: $("folder-path").value }) });
    state.current = null;
    state.currentId = null;
    await loadImages();
    setStatus("Workspace đã mở");
  } catch (error) { setStatus(`Lỗi: ${error.message}`); }
};
$("detect-current").onclick = () => runCurrent("detect", { replace_existing: true });
$("prelabel-page").onclick = () => runCurrent("prelabel", { block_ids: null, replace_existing: true });
$("prelabel-current").onclick = () => runCurrent("prelabel", { block_ids: state.selected ? [state.selected] : null, replace_existing: true });
$("complete-current").onclick = () => runCurrent("complete");
$("add-mode").onclick = () => { state.add = !state.add; $("add-mode").textContent = state.add ? "Hủy thêm bbox" : "Thêm bbox"; };
$("delete-block").onclick = () => mutate(() => {
  state.current.blocks = state.current.blocks.filter((block) => block.id !== state.selected).map((block, index) => ({ ...block, order: index }));
  state.selected = state.current.blocks[0]?.id || null;
});
function markManual(block) { block.source = "manual"; block.score = null; }
$("layout-label").onchange = (event) => mutate(() => { const block = selectedBlock(); block.layout_label = event.target.value; markManual(block); });
$("task").onchange = (event) => mutate(() => { const block = selectedBlock(); block.task = event.target.value || null; markManual(block); });
$("text").oninput = (event) => mutate(() => { const block = selectedBlock(); block.text = event.target.value; markManual(block); });
$("skipped").onchange = (event) => mutate(() => { const block = selectedBlock(); block.skipped = event.target.checked; markManual(block); });
$("detect-batch").onclick = () => startBatch("detect");
$("prelabel-batch").onclick = () => startBatch("prelabel");
$("cancel-batch").onclick = async () => pollBatch(await api("/api/batch", { method: "DELETE" }));
$("export-hf").onclick = () => exportDataset("/api/export/hf", "HF export");
$("export-layout").onclick = () => exportDataset("/api/export/layout", "Layout export");
$("export-all").onclick = () => exportDataset("/api/export/all", "Export All");
window.addEventListener("resize", fitImage);

const taxonomy = await api("/api/taxonomy");
for (const label of taxonomy.layout_labels) {
  const option = document.createElement("option");
  option.value = label;
  option.textContent = label;
  $("layout-label").append(option);
}

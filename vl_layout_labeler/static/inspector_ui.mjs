import {
  LAYOUT_DEFAULTS,
  LAYOUT_LIMITS,
  clamp,
  clampInspectorWidth,
  computeCropTransform,
  cropHeightBounds,
  cropHeightFromSplit,
  cropSplitFromHeight,
  inspectorWidthBounds,
  loadLayoutPreferences,
  normalizeCropZoom,
  resetLayoutPreferences,
  saveLayoutPreferences,
} from "./inspector_layout.mjs";

const $ = (id) => document.getElementById(id);
let preferences = loadLayoutPreferences(window.localStorage);
let scheduled = false;

function setEditorFocus(enabled) {
  const inspector = document.querySelector(".inspector");
  const button = $("editor-focus-toggle");
  const active = Boolean(enabled && !$("editor").hidden);
  inspector.classList.toggle("editor-focus", active);
  document.body.classList.toggle("editor-focus-open", active);
  button.setAttribute("aria-pressed", String(active));
  button.textContent = active ? "Thu nhỏ" : "Tập trung";
  scheduleRefresh();
}

function persist() {
  preferences = saveLayoutPreferences(window.localStorage, preferences);
}

function refitCanvas() {
  if ($("page-image").naturalWidth && typeof $("page-image").onload === "function") {
    $("page-image").onload();
  }
}

function desktopInspectorEnabled() {
  return !window.matchMedia("(max-width: 1024px)").matches;
}

function applyInspectorWidth() {
  const separator = $("inspector-separator");
  const sidebarWidth = document.querySelector(".sidebar")?.getBoundingClientRect().width || 245;
  const width = clampInspectorWidth(preferences.inspectorWidth, window.innerWidth, sidebarWidth);
  const bounds = inspectorWidthBounds(window.innerWidth, sidebarWidth);
  if (desktopInspectorEnabled()) document.documentElement.style.setProperty("--inspector-width", `${width}px`);
  separator.setAttribute("aria-valuemin", String(Math.round(bounds.minimum)));
  separator.setAttribute("aria-valuemax", String(Math.round(bounds.maximum)));
  separator.setAttribute("aria-valuenow", String(Math.round(width)));
  return width;
}

function applyCropSplit() {
  const split = $("editor-split");
  const separator = $("crop-editor-separator");
  if (!split || split.clientHeight <= 0) return;
  const bounds = cropHeightBounds(split.clientHeight, separator.offsetHeight || 9);
  const height = cropHeightFromSplit(split.clientHeight, preferences.cropSplit, separator.offsetHeight || 9);
  document.documentElement.style.setProperty("--crop-height", `${height}px`);
  separator.setAttribute("aria-valuemin", String(Math.round(bounds.minimum)));
  separator.setAttribute("aria-valuemax", String(Math.round(bounds.maximum)));
  separator.setAttribute("aria-valuenow", String(Math.round(height)));
}

function screenPolygon() {
  const polygon = document.querySelector("#overlay .bbox.selected");
  if (!polygon) return null;
  return (polygon.getAttribute("points") || "").trim().split(/\s+/).map((pair) => (
    pair.split(",").map(Number)
  )).filter((point) => point.length === 2 && point.every(Number.isFinite));
}

function imagePolygon() {
  const points = screenPolygon();
  const pageImage = $("page-image");
  if (!points?.length || !pageImage.naturalWidth || !pageImage.clientWidth) return null;
  const scale = pageImage.clientWidth / pageImage.naturalWidth;
  const left = Number.parseFloat(pageImage.style.left || "0");
  const top = Number.parseFloat(pageImage.style.top || "0");
  return points.map(([x, y]) => [(x - left) / scale, (y - top) / scale]);
}

function renderCrop() {
  const blockPolygon = imagePolygon();
  const pageImage = $("page-image");
  const cropImage = $("crop-image");
  const viewport = $("crop-viewport");
  const space = $("crop-space");
  const clip = $("crop-clip");
  if (!blockPolygon || !pageImage.naturalWidth || !viewport.clientWidth || !viewport.clientHeight) {
    $("crop-meta").textContent = "Chọn một block để đối chiếu";
    clip.hidden = true;
    return;
  }
  clip.hidden = false;
  if (cropImage.src !== pageImage.src) cropImage.src = pageImage.src;
  const transform = computeCropTransform({
    polygon: blockPolygon,
    imageWidth: pageImage.naturalWidth,
    imageHeight: pageImage.naturalHeight,
    viewportWidth: viewport.clientWidth,
    viewportHeight: viewport.clientHeight,
    zoom: preferences.cropZoom,
  });
  Object.assign(space.style, {
    width: `${transform.canvasWidth}px`,
    height: `${transform.canvasHeight}px`,
  });
  Object.assign(clip.style, {
    left: `${transform.clipLeft}px`,
    top: `${transform.clipTop}px`,
    width: `${transform.contentWidth}px`,
    height: `${transform.contentHeight}px`,
  });
  Object.assign(cropImage.style, {
    left: `${transform.imageLeft}px`,
    top: `${transform.imageTop}px`,
    width: `${transform.imageWidth}px`,
    height: `${transform.imageHeight}px`,
  });
  const layoutLabel = $("layout-label").value || "layout";
  const task = $("task").value || "layout-only";
  $("crop-meta").textContent = `${layoutLabel} · ${task} · ${transform.box.width}×${transform.box.height}px`;
  $("crop-zoom-value").textContent = preferences.cropZoom === "fit"
    ? `Fit ${Math.round(transform.scale * 100)}%`
    : `${Math.round(transform.scale * 100)}%`;
  $("crop-fit").classList.toggle("active", preferences.cropZoom === "fit");
  $("crop-actual").classList.toggle("active", preferences.cropZoom === 1);
}

function enhanceEditorPresentation() {
  const visualEditor = $("visual-editor");
  const tableVisualMode = $("task").value === "table" && !$("visual-panel").hidden;
  document.querySelector(".editor-scroll").classList.toggle("table-task-mode", tableVisualMode);
  visualEditor.classList.toggle(
    "vertical-text-mode",
    $("task").value === "ocr" && $("layout-label").value === "vertical_text",
  );
  for (const textarea of visualEditor.querySelectorAll(".cell-input, .target-grid textarea")) {
    const lines = textarea.value.split("\n");
    textarea.rows = clamp(lines.length, 2, 8);
    textarea.cols = clamp(Math.max(...lines.map((line) => line.length), 8), 12, 36);
  }
}

function refresh() {
  scheduled = false;
  const focusButton = $("editor-focus-toggle");
  focusButton.disabled = $("editor").hidden;
  if (focusButton.disabled && document.querySelector(".inspector").classList.contains("editor-focus")) {
    setEditorFocus(false);
    return;
  }
  applyInspectorWidth();
  applyCropSplit();
  renderCrop();
  enhanceEditorPresentation();
}

function scheduleRefresh() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(refresh);
}

function setCropZoom(value) {
  preferences.cropZoom = normalizeCropZoom(value);
  persist();
  renderCrop();
}

function bindPaneSeparator() {
  const separator = $("inspector-separator");
  separator.addEventListener("pointerdown", (event) => {
    if (!desktopInspectorEnabled()) return;
    event.preventDefault();
    separator.setPointerCapture(event.pointerId);
    separator.classList.add("dragging");
  });
  separator.addEventListener("pointermove", (event) => {
    if (!separator.hasPointerCapture(event.pointerId)) return;
    const sidebarWidth = document.querySelector(".sidebar")?.getBoundingClientRect().width || 245;
    preferences.inspectorWidth = clampInspectorWidth(
      window.innerWidth - event.clientX,
      window.innerWidth,
      sidebarWidth,
    );
    applyInspectorWidth();
    refitCanvas();
    scheduleRefresh();
  });
  const finish = (event) => {
    if (separator.hasPointerCapture(event.pointerId)) separator.releasePointerCapture(event.pointerId);
    separator.classList.remove("dragging");
    persist();
  };
  separator.addEventListener("pointerup", finish);
  separator.addEventListener("pointercancel", finish);
  separator.addEventListener("dblclick", () => {
    preferences.inspectorWidth = LAYOUT_DEFAULTS.inspectorWidth;
    persist();
    applyInspectorWidth();
    refitCanvas();
    scheduleRefresh();
  });
  separator.addEventListener("keydown", (event) => {
    if (!desktopInspectorEnabled() || !["ArrowLeft", "ArrowRight", "Home"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Home") preferences.inspectorWidth = LAYOUT_DEFAULTS.inspectorWidth;
    else preferences.inspectorWidth += event.key === "ArrowLeft" ? 24 : -24;
    persist();
    applyInspectorWidth();
    refitCanvas();
    scheduleRefresh();
  });
}

function bindCropSeparator() {
  const separator = $("crop-editor-separator");
  const split = $("editor-split");
  separator.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    separator.setPointerCapture(event.pointerId);
    separator.classList.add("dragging");
  });
  separator.addEventListener("pointermove", (event) => {
    if (!separator.hasPointerCapture(event.pointerId)) return;
    const bounds = split.getBoundingClientRect();
    preferences.cropSplit = cropSplitFromHeight(
      split.clientHeight,
      event.clientY - bounds.top,
      separator.offsetHeight || 9,
    );
    applyCropSplit();
    scheduleRefresh();
  });
  const finish = (event) => {
    if (separator.hasPointerCapture(event.pointerId)) separator.releasePointerCapture(event.pointerId);
    separator.classList.remove("dragging");
    persist();
  };
  separator.addEventListener("pointerup", finish);
  separator.addEventListener("pointercancel", finish);
  separator.addEventListener("dblclick", () => {
    preferences.cropSplit = LAYOUT_DEFAULTS.cropSplit;
    persist();
    scheduleRefresh();
  });
  separator.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown", "Home"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Home") preferences.cropSplit = LAYOUT_DEFAULTS.cropSplit;
    else {
      const current = cropHeightFromSplit(split.clientHeight, preferences.cropSplit, separator.offsetHeight || 9);
      preferences.cropSplit = cropSplitFromHeight(
        split.clientHeight,
        current + (event.key === "ArrowDown" ? 20 : -20),
        separator.offsetHeight || 9,
      );
    }
    persist();
    scheduleRefresh();
  });
}

function bindCropControls() {
  $("crop-fit").addEventListener("click", () => setCropZoom("fit"));
  $("crop-actual").addEventListener("click", () => setCropZoom(1));
  $("crop-zoom-out").addEventListener("click", () => {
    const current = preferences.cropZoom === "fit"
      ? computeCurrentScale()
      : preferences.cropZoom;
    setCropZoom(current / 1.25);
  });
  $("crop-zoom-in").addEventListener("click", () => {
    const current = preferences.cropZoom === "fit"
      ? computeCurrentScale()
      : preferences.cropZoom;
    setCropZoom(current * 1.25);
  });
}

function bindFocusMode() {
  $("editor-focus-toggle").addEventListener("click", () => {
    setEditorFocus(!document.querySelector(".inspector").classList.contains("editor-focus"));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.querySelector(".inspector").classList.contains("editor-focus")) {
      setEditorFocus(false);
    }
  });
}

function computeCurrentScale() {
  const blockPolygon = imagePolygon();
  const pageImage = $("page-image");
  const viewport = $("crop-viewport");
  if (!blockPolygon || !pageImage.naturalWidth) return 1;
  return computeCropTransform({
    polygon: blockPolygon,
    imageWidth: pageImage.naturalWidth,
    imageHeight: pageImage.naturalHeight,
    viewportWidth: viewport.clientWidth,
    viewportHeight: viewport.clientHeight,
    zoom: "fit",
  }).scale;
}

bindPaneSeparator();
bindCropSeparator();
bindCropControls();
bindFocusMode();

const observer = new MutationObserver(scheduleRefresh);
observer.observe($("overlay"), { childList: true, subtree: true, attributes: true, attributeFilter: ["points", "class"] });
observer.observe($("editor"), { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden"] });
observer.observe($("page-image"), { attributes: true, attributeFilter: ["src", "style"] });
$("crop-image").addEventListener("load", scheduleRefresh);
$("layout-label").addEventListener("change", scheduleRefresh);
$("task").addEventListener("change", scheduleRefresh);
window.addEventListener("resize", scheduleRefresh);
if (window.ResizeObserver) {
  const resizeObserver = new ResizeObserver(scheduleRefresh);
  resizeObserver.observe($("editor-split"));
  resizeObserver.observe($("crop-viewport"));
}

window.vlLayoutInspector = {
  resetPreferences() {
    preferences = resetLayoutPreferences(window.localStorage);
    scheduleRefresh();
  },
};

scheduleRefresh();

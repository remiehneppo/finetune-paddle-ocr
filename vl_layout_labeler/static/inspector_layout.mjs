export const LAYOUT_PREFERENCE_KEY = "vl-layout-labeler.inspector.v1";

export const LAYOUT_DEFAULTS = Object.freeze({
  inspectorWidth: 440,
  cropSplit: 0.42,
  cropZoom: "fit",
});

export const LAYOUT_LIMITS = Object.freeze({
  inspectorMin: 340,
  inspectorMax: 760,
  canvasMin: 420,
  cropMin: 130,
  editorMin: 180,
  splitMin: 0.2,
  splitMax: 0.72,
  zoomMin: 0.25,
  zoomMax: 4,
});

export function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function finiteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function cropBoxFromPolygon(polygon, imageWidth, imageHeight) {
  const width = Math.max(1, finiteNumber(imageWidth, 1));
  const height = Math.max(1, finiteNumber(imageHeight, 1));
  const points = Array.isArray(polygon)
    ? polygon.filter((point) => (
      Array.isArray(point)
      && point.length >= 2
      && Number.isFinite(Number(point[0]))
      && Number.isFinite(Number(point[1]))
    ))
    : [];
  if (!points.length) return { left: 0, top: 0, right: 1, bottom: 1, width: 1, height: 1 };
  const xs = points.map((point) => Number(point[0]));
  const ys = points.map((point) => Number(point[1]));
  const left = clamp(Math.floor(Math.min(...xs)), 0, Math.max(width - 1, 0));
  const top = clamp(Math.floor(Math.min(...ys)), 0, Math.max(height - 1, 0));
  const right = clamp(Math.ceil(Math.max(...xs)), left + 1, width);
  const bottom = clamp(Math.ceil(Math.max(...ys)), top + 1, height);
  return { left, top, right, bottom, width: right - left, height: bottom - top };
}

export function normalizeCropZoom(value) {
  if (value === "fit") return "fit";
  const zoom = Number(value);
  return Number.isFinite(zoom)
    ? clamp(zoom, LAYOUT_LIMITS.zoomMin, LAYOUT_LIMITS.zoomMax)
    : LAYOUT_DEFAULTS.cropZoom;
}

export function computeCropTransform({
  polygon,
  imageWidth,
  imageHeight,
  viewportWidth,
  viewportHeight,
  zoom,
}) {
  const box = cropBoxFromPolygon(polygon, imageWidth, imageHeight);
  const availableWidth = Math.max(1, finiteNumber(viewportWidth, 1));
  const availableHeight = Math.max(1, finiteNumber(viewportHeight, 1));
  const normalizedZoom = normalizeCropZoom(zoom);
  const scale = normalizedZoom === "fit"
    ? Math.min(availableWidth / box.width, availableHeight / box.height)
    : normalizedZoom;
  const contentWidth = Math.max(1, box.width * scale);
  const contentHeight = Math.max(1, box.height * scale);
  const canvasWidth = Math.max(availableWidth, contentWidth);
  const canvasHeight = Math.max(availableHeight, contentHeight);
  return {
    box,
    scale,
    contentWidth,
    contentHeight,
    canvasWidth,
    canvasHeight,
    clipLeft: (canvasWidth - contentWidth) / 2,
    clipTop: (canvasHeight - contentHeight) / 2,
    imageLeft: -box.left * scale,
    imageTop: -box.top * scale,
    imageWidth: Math.max(1, finiteNumber(imageWidth, 1) * scale),
    imageHeight: Math.max(1, finiteNumber(imageHeight, 1) * scale),
  };
}

export function inspectorWidthBounds(viewportWidth, sidebarWidth = 245, separatorWidth = 9) {
  const available = finiteNumber(viewportWidth, 0) - sidebarWidth - separatorWidth - LAYOUT_LIMITS.canvasMin;
  return {
    minimum: LAYOUT_LIMITS.inspectorMin,
    maximum: Math.max(LAYOUT_LIMITS.inspectorMin, Math.min(LAYOUT_LIMITS.inspectorMax, available)),
  };
}

export function clampInspectorWidth(value, viewportWidth, sidebarWidth = 245, separatorWidth = 9) {
  const bounds = inspectorWidthBounds(viewportWidth, sidebarWidth, separatorWidth);
  return clamp(finiteNumber(value, LAYOUT_DEFAULTS.inspectorWidth), bounds.minimum, bounds.maximum);
}

export function clampCropSplit(value) {
  return clamp(
    finiteNumber(value, LAYOUT_DEFAULTS.cropSplit),
    LAYOUT_LIMITS.splitMin,
    LAYOUT_LIMITS.splitMax,
  );
}

export function cropHeightBounds(totalHeight, separatorHeight = 9) {
  const available = Math.max(0, finiteNumber(totalHeight, 0) - separatorHeight);
  const maximum = Math.max(LAYOUT_LIMITS.cropMin, available - LAYOUT_LIMITS.editorMin);
  return { minimum: LAYOUT_LIMITS.cropMin, maximum };
}

export function cropHeightFromSplit(totalHeight, split, separatorHeight = 9) {
  const bounds = cropHeightBounds(totalHeight, separatorHeight);
  return clamp(totalHeight * clampCropSplit(split), bounds.minimum, bounds.maximum);
}

export function cropSplitFromHeight(totalHeight, height, separatorHeight = 9) {
  const bounds = cropHeightBounds(totalHeight, separatorHeight);
  const clampedHeight = clamp(finiteNumber(height, bounds.minimum), bounds.minimum, bounds.maximum);
  return clampCropSplit(clampedHeight / Math.max(1, totalHeight));
}

export function loadLayoutPreferences(storage) {
  let stored = {};
  try {
    stored = JSON.parse(storage?.getItem(LAYOUT_PREFERENCE_KEY) || "{}") || {};
  } catch {
    stored = {};
  }
  return {
    inspectorWidth: finiteNumber(stored.inspectorWidth, LAYOUT_DEFAULTS.inspectorWidth),
    cropSplit: clampCropSplit(stored.cropSplit),
    cropZoom: normalizeCropZoom(stored.cropZoom),
  };
}

export function saveLayoutPreferences(storage, preferences) {
  const normalized = {
    inspectorWidth: finiteNumber(preferences.inspectorWidth, LAYOUT_DEFAULTS.inspectorWidth),
    cropSplit: clampCropSplit(preferences.cropSplit),
    cropZoom: normalizeCropZoom(preferences.cropZoom),
  };
  try {
    storage?.setItem(LAYOUT_PREFERENCE_KEY, JSON.stringify(normalized));
  } catch {
    return normalized;
  }
  return normalized;
}

export function resetLayoutPreferences(storage) {
  try {
    storage?.removeItem(LAYOUT_PREFERENCE_KEY);
  } catch {
    // Preferences are optional; an unavailable storage backend must not block editing.
  }
  return { ...LAYOUT_DEFAULTS };
}

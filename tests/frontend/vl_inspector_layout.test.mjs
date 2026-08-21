import test from "node:test";
import assert from "node:assert/strict";

import {
  LAYOUT_DEFAULTS,
  LAYOUT_PREFERENCE_KEY,
  clampInspectorWidth,
  computeCropTransform,
  cropBoxFromPolygon,
  cropHeightFromSplit,
  cropSplitFromHeight,
  loadLayoutPreferences,
  resetLayoutPreferences,
  saveLayoutPreferences,
} from "../../vl_layout_labeler/static/inspector_layout.mjs";

class MemoryStorage {
  constructor(value = null) {
    this.value = value;
  }

  getItem(key) {
    return key === LAYOUT_PREFERENCE_KEY ? this.value : null;
  }

  setItem(key, value) {
    if (key === LAYOUT_PREFERENCE_KEY) this.value = value;
  }

  removeItem(key) {
    if (key === LAYOUT_PREFERENCE_KEY) this.value = null;
  }
}

test("crop geometry matches horizontal, vertical, edge, and irregular polygons", () => {
  assert.deepEqual(
    cropBoxFromPolygon([[10, 20], [90, 20], [90, 50], [10, 50]], 200, 100),
    { left: 10, top: 20, right: 90, bottom: 50, width: 80, height: 30 },
  );
  assert.deepEqual(
    cropBoxFromPolygon([[20, 5], [42, 5], [42, 95], [20, 95]], 100, 100),
    { left: 20, top: 5, right: 42, bottom: 95, width: 22, height: 90 },
  );
  assert.deepEqual(
    cropBoxFromPolygon([[-4, -2], [102, 0], [101, 51], [0, 50]], 100, 50),
    { left: 0, top: 0, right: 100, bottom: 50, width: 100, height: 50 },
  );
  assert.deepEqual(
    cropBoxFromPolygon([[9.8, 4.2], [81.1, 7.9], [77.4, 48.6], [12.3, 45.1]], 100, 60),
    { left: 9, top: 4, right: 82, bottom: 49, width: 73, height: 45 },
  );
});

test("crop transform translates the full image into a clipped crop", () => {
  const transform = computeCropTransform({
    polygon: [[50, 20], [150, 20], [150, 70], [50, 70]],
    imageWidth: 400,
    imageHeight: 200,
    viewportWidth: 300,
    viewportHeight: 180,
    zoom: "fit",
  });
  assert.equal(transform.scale, 3);
  assert.equal(transform.contentWidth, 300);
  assert.equal(transform.contentHeight, 150);
  assert.equal(transform.imageLeft, -150);
  assert.equal(transform.imageTop, -60);
  assert.equal(transform.imageWidth, 1200);
  assert.equal(transform.imageHeight, 600);
  assert.equal(transform.clipTop, 15);
});

test("pane and crop split clamping preserve usable canvas and editor sizes", () => {
  assert.equal(clampInspectorWidth(900, 1400), 726);
  assert.equal(clampInspectorWidth(100, 1400), 340);
  assert.equal(cropHeightFromSplit(700, 0.9), 504);
  assert.equal(cropHeightFromSplit(500, 0.05), 130);
  assert.equal(cropSplitFromHeight(700, 350), 0.5);
});

test("layout preferences recover from invalid storage and persist normalized values", () => {
  const invalid = new MemoryStorage("not-json");
  assert.deepEqual(loadLayoutPreferences(invalid), LAYOUT_DEFAULTS);

  const storage = new MemoryStorage();
  const saved = saveLayoutPreferences(storage, {
    inspectorWidth: 510,
    cropSplit: 4,
    cropZoom: 99,
  });
  assert.equal(saved.inspectorWidth, 510);
  assert.equal(saved.cropSplit, 0.72);
  assert.equal(saved.cropZoom, 4);
  assert.deepEqual(loadLayoutPreferences(storage), saved);

  assert.deepEqual(resetLayoutPreferences(storage), LAYOUT_DEFAULTS);
  assert.equal(storage.value, null);
});

import test from "node:test";
import assert from "node:assert/strict";

import {
  fitCanvasView,
  panCanvasView,
  resizeRectangleEdge,
  zoomCanvasView,
} from "../../vl_layout_labeler/static/canvas_geometry.mjs";

test("fit view centers the original image without upscaling", () => {
  assert.deepEqual(
    fitCanvasView(500, 400, 1000, 200),
    { scale: 0.5, x: 0, y: 150 },
  );
  assert.deepEqual(
    fitCanvasView(800, 600, 200, 100),
    { scale: 1, x: 300, y: 250 },
  );
});

test("zoom preserves the image point under the cursor and clamps scale", () => {
  const view = { scale: 0.5, x: 20, y: 30 };
  const zoomed = zoomCanvasView(view, [120, 80], 2);
  assert.deepEqual(zoomed, { scale: 2, x: -280, y: -120 });
  assert.deepEqual(zoomCanvasView(view, [20, 30], 99), { scale: 8, x: 20, y: 30 });
});

test("pan translates the viewport without changing scale", () => {
  assert.deepEqual(
    panCanvasView({ scale: 2, x: 10, y: 20 }, -5, 12),
    { scale: 2, x: 5, y: 32 },
  );
});

test("dragging a bbox edge moves both vertices on that edge", () => {
  const polygon = [[10, 20], [90, 20], [90, 60], [10, 60]];
  assert.deepEqual(
    resizeRectangleEdge(polygon, "top", [50, 5], 120, 100),
    [[10, 5], [90, 5], [90, 60], [10, 60]],
  );
  assert.deepEqual(
    resizeRectangleEdge(polygon, "right", [110, 40], 120, 100),
    [[10, 20], [110, 20], [110, 60], [10, 60]],
  );
  assert.deepEqual(
    resizeRectangleEdge(polygon, "bottom", [50, 90], 120, 100),
    [[10, 20], [90, 20], [90, 90], [10, 90]],
  );
  assert.deepEqual(
    resizeRectangleEdge(polygon, "left", [2, 40], 120, 100),
    [[2, 20], [90, 20], [90, 60], [2, 60]],
  );
});

test("edge resizing stays inside the image and preserves positive area", () => {
  const polygon = [[10, 20], [90, 20], [90, 60], [10, 60]];
  assert.deepEqual(
    resizeRectangleEdge(polygon, "top", [50, 99], 120, 100),
    [[10, 59], [90, 59], [90, 60], [10, 60]],
  );
  assert.deepEqual(
    resizeRectangleEdge(polygon, "left", [200, 40], 120, 100),
    [[89, 20], [90, 20], [90, 60], [89, 60]],
  );
});

test("edge resizing clamps within image boundaries [0, max] and avoids negative coordinates", () => {
  const polygon = [[10, 20], [90, 20], [90, 60], [10, 60]];
  // Drag top to negative coordinates -> clamps to 0
  assert.deepEqual(
    resizeRectangleEdge(polygon, "top", [50, -50], 120, 100),
    [[10, 0], [90, 0], [90, 60], [10, 60]],
  );
  // Drag left to negative coordinates -> clamps to 0
  assert.deepEqual(
    resizeRectangleEdge(polygon, "left", [-30, 40], 120, 100),
    [[0, 20], [90, 20], [90, 60], [0, 60]],
  );
  // Drag bottom past height (100 -> max 99)
  assert.deepEqual(
    resizeRectangleEdge(polygon, "bottom", [50, 250], 120, 100),
    [[10, 20], [90, 20], [90, 99], [10, 99]],
  );
  // Drag right past width (120 -> max 119)
  assert.deepEqual(
    resizeRectangleEdge(polygon, "right", [300, 40], 120, 100),
    [[10, 20], [119, 20], [119, 60], [10, 60]],
  );

  // When bottom is at y=0, top cannot become negative
  const zeroBottom = [[10, 0], [90, 0], [90, 0], [10, 0]];
  assert.deepEqual(
    resizeRectangleEdge(zeroBottom, "top", [50, 50], 120, 100),
    [[10, 0], [90, 0], [90, 0], [10, 0]],
  );

  // When top is at maxY (99), bottom cannot exceed 99
  const maxTop = [[10, 99], [90, 99], [90, 99], [10, 99]];
  assert.deepEqual(
    resizeRectangleEdge(maxTop, "bottom", [50, 50], 120, 100),
    [[10, 99], [90, 99], [90, 99], [10, 99]],
  );
});

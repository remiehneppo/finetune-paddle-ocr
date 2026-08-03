import test from "node:test";
import assert from "node:assert/strict";
import {
  centerViewOnPolygon,
  imageToScreen,
  rectanglePolygon,
  screenToImage,
  translatePolygon,
} from "../../ocr_labeler/static/geometry.mjs";

test("screen and image transforms are inverse", () => {
  const viewport = { scale: 0.5, offsetX: 20, offsetY: 30 };
  const screen = imageToScreen([100, 80], viewport);
  assert.deepEqual(screen, [70, 70]);
  assert.deepEqual(screenToImage(screen, viewport), [100, 80]);
});

test("translation clamps all four points", () => {
  const polygon = [[0, 0], [90, 0], [90, 20], [0, 20]];
  assert.deepEqual(
    translatePolygon(polygon, 20, 70, 100, 80),
    [[9, 59], [99, 59], [99, 79], [9, 79]],
  );
});

test("rectangle normalizes reverse drag direction", () => {
  assert.deepEqual(
    rectanglePolygon([20, 30], [5, 10]),
    [[5, 10], [20, 10], [20, 30], [5, 30]],
  );
});

test("centering a polygon preserves scale and puts its bounds at viewport center", () => {
  const view = centerViewOnPolygon(
    [[10, 20], [30, 20], [30, 40], [10, 40]],
    { scale: 2, offsetX: 9, offsetY: 12 },
    { width: 300, height: 200 },
  );

  assert.deepEqual(view, { scale: 2, offsetX: 110, offsetY: 40 });
  assert.deepEqual(imageToScreen([20, 30], view), [150, 100]);
});

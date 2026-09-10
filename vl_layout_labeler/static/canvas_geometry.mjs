export const MIN_CANVAS_SCALE = 0.1;
export const MAX_CANVAS_SCALE = 8;

export function clampCanvasScale(scale) {
  return Math.max(MIN_CANVAS_SCALE, Math.min(MAX_CANVAS_SCALE, scale));
}

export function fitCanvasView(stageWidth, stageHeight, imageWidth, imageHeight) {
  if (!stageWidth || !stageHeight || !imageWidth || !imageHeight) {
    return { scale: 1, x: 0, y: 0 };
  }
  const scale = Math.min(stageWidth / imageWidth, stageHeight / imageHeight, 1);
  return {
    scale,
    x: (stageWidth - imageWidth * scale) / 2,
    y: (stageHeight - imageHeight * scale) / 2,
  };
}

export function zoomCanvasView(view, anchor, requestedScale) {
  const scale = clampCanvasScale(requestedScale);
  const imageX = (anchor[0] - view.x) / view.scale;
  const imageY = (anchor[1] - view.y) / view.scale;
  return {
    scale,
    x: anchor[0] - imageX * scale,
    y: anchor[1] - imageY * scale,
  };
}

export function panCanvasView(view, dx, dy) {
  return { ...view, x: view.x + dx, y: view.y + dy };
}

export function resizeRectangleEdge(polygon, edge, point, imageWidth, imageHeight) {
  const next = polygon.map(([x, y]) => [x, y]);
  const maxX = Math.max(0, imageWidth - 1);
  const maxY = Math.max(0, imageHeight - 1);
  const clampX = (value) => Math.max(0, Math.min(maxX, value));
  const clampY = (value) => Math.max(0, Math.min(maxY, value));

  if (edge === "top") {
    const maxAllowedY = Math.max(0, Math.min(next[2][1], next[3][1]) - 1);
    const y = Math.max(0, Math.min(clampY(point[1]), maxAllowedY));
    next[0][1] = y;
    next[1][1] = y;
  } else if (edge === "right") {
    const minAllowedX = Math.min(maxX, Math.max(next[0][0], next[3][0]) + 1);
    const x = Math.min(maxX, Math.max(clampX(point[0]), minAllowedX));
    next[1][0] = x;
    next[2][0] = x;
  } else if (edge === "bottom") {
    const minAllowedY = Math.min(maxY, Math.max(next[0][1], next[1][1]) + 1);
    const y = Math.min(maxY, Math.max(clampY(point[1]), minAllowedY));
    next[2][1] = y;
    next[3][1] = y;
  } else if (edge === "left") {
    const maxAllowedX = Math.max(0, Math.min(next[1][0], next[2][0]) - 1);
    const x = Math.max(0, Math.min(clampX(point[0]), maxAllowedX));
    next[0][0] = x;
    next[3][0] = x;
  }
  return next;
}

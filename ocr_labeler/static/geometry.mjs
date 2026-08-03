export const imageToScreen = ([x, y], view) => [
  x * view.scale + view.offsetX,
  y * view.scale + view.offsetY,
];

export const screenToImage = ([x, y], view) => [
  (x - view.offsetX) / view.scale,
  (y - view.offsetY) / view.scale,
];

export function rectanglePolygon(start, end) {
  const left = Math.min(start[0], end[0]);
  const right = Math.max(start[0], end[0]);
  const top = Math.min(start[1], end[1]);
  const bottom = Math.max(start[1], end[1]);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

export function translatePolygon(polygon, dx, dy, width, height) {
  const xs = polygon.map(([x]) => x);
  const ys = polygon.map(([, y]) => y);
  const safeDx = Math.min(
    Math.max(dx, -Math.min(...xs)),
    width - 1 - Math.max(...xs),
  );
  const safeDy = Math.min(
    Math.max(dy, -Math.min(...ys)),
    height - 1 - Math.max(...ys),
  );
  return polygon.map(([x, y]) => [x + safeDx, y + safeDy]);
}

export function centerViewOnPolygon(polygon, view, viewport) {
  const xs = polygon.map(([x]) => x);
  const ys = polygon.map(([, y]) => y);
  const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
  return {
    scale: view.scale,
    offsetX: viewport.width / 2 - centerX * view.scale,
    offsetY: viewport.height / 2 - centerY * view.scale,
  };
}

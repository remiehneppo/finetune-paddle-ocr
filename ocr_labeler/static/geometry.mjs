export const imageToScreen = ([x, y], view) => [
  x * (view?.scale || 1) + (view?.offsetX ?? 0),
  y * (view?.scale || 1) + (view?.offsetY ?? 0),
];

export const screenToImage = ([x, y], view) => {
  const scale = view?.scale || 1;
  const offsetX = view?.offsetX ?? 0;
  const offsetY = view?.offsetY ?? 0;
  return [
    (x - offsetX) / scale,
    (y - offsetY) / scale,
  ];
};

export function rectanglePolygon(start, end) {
  const left = Math.min(start[0], end[0]);
  const right = Math.max(start[0], end[0]);
  const top = Math.min(start[1], end[1]);
  const bottom = Math.max(start[1], end[1]);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

export function translatePolygon(polygon, dx, dy, width, height) {
  if (!polygon?.length) return [];
  const xs = polygon.map(([x]) => x);
  const ys = polygon.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const minSafeDx = -minX;
  const maxSafeDx = Math.max(minSafeDx, width - 1 - maxX);
  const safeDx = Math.min(Math.max(dx, minSafeDx), maxSafeDx);

  const minSafeDy = -minY;
  const maxSafeDy = Math.max(minSafeDy, height - 1 - maxY);
  const safeDy = Math.min(Math.max(dy, minSafeDy), maxSafeDy);

  return polygon.map(([x, y]) => [
    Math.max(0, Math.min(width - 1, x + safeDx)),
    Math.max(0, Math.min(height - 1, y + safeDy)),
  ]);
}

export function centerViewOnPolygon(polygon, view, viewport) {
  if (!polygon?.length) {
    return { ...view };
  }
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

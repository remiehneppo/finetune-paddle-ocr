const clone = (value) => structuredClone(value);

const blockById = (annotation, id) => {
  const block = annotation.blocks.find((item) => item.id === id);
  if (!block) throw new Error(`Unknown block: ${id}`);
  return block;
};

const bounds = (annotation) => ({
  maxX: Math.max(0, annotation.image.width - 1),
  maxY: Math.max(0, annotation.image.height - 1),
});

const clampPoint = (annotation, [x, y]) => {
  const { maxX, maxY } = bounds(annotation);
  return [Math.min(Math.max(x, 0), maxX), Math.min(Math.max(y, 0), maxY)];
};

const clampPolygon = (annotation, polygon) =>
  polygon.map((point) => clampPoint(annotation, point));

function normalized(annotation) {
  const next = clone(annotation);
  next.blocks.sort((left, right) => left.order - right.order);
  next.blocks.forEach((block, index) => {
    block.order = index;
  });
  next.text = next.blocks.map((block) => block.text).join("\n");
  return next;
}

function mutate(state, change, { preserveStatus = false } = {}) {
  const before = clone(state.annotation);
  const annotation = normalized(change(clone(state.annotation)));
  if (!preserveStatus) annotation.status = "edited";
  return {
    ...state,
    annotation,
    undoStack: [...state.undoStack.slice(-49), before],
    redoStack: [],
    dirty: true,
  };
}

export function createEditorState(annotation) {
  const normalizedAnnotation = normalized(annotation);
  return {
    annotation: normalizedAnnotation,
    selectedId: normalizedAnnotation.blocks[0]?.id ?? null,
    undoStack: [],
    redoStack: [],
    dirty: false,
  };
}

export function selectBlock(state, id) {
  if (id !== null) blockById(state.annotation, id);
  return { ...state, selectedId: id };
}

export function updateText(state, id, text) {
  return mutate(state, (annotation) => {
    blockById(annotation, id).text = text;
    return annotation;
  });
}

export function moveCorner(state, id, corner, x, y) {
  return mutate(state, (annotation) => {
    const block = blockById(annotation, id);
    if (!Number.isInteger(corner) || corner < 0 || corner >= block.polygon.length) {
      throw new Error(`Unknown polygon corner: ${corner}`);
    }
    block.polygon[corner] = clampPoint(annotation, [x, y]);
    return annotation;
  });
}

export function resizeBlock(state, id, corner, x, y) {
  return moveCorner(state, id, corner, x, y);
}

export function moveBlock(state, id, dx, dy) {
  return mutate(state, (annotation) => {
    const block = blockById(annotation, id);
    const xs = block.polygon.map(([x]) => x);
    const ys = block.polygon.map(([, y]) => y);
    const { maxX, maxY } = bounds(annotation);
    const safeDx = Math.min(Math.max(dx, -Math.min(...xs)), maxX - Math.max(...xs));
    const safeDy = Math.min(Math.max(dy, -Math.min(...ys)), maxY - Math.max(...ys));
    block.polygon = block.polygon.map(([x, y]) => [x + safeDx, y + safeDy]);
    return annotation;
  });
}

export function addBlock(state, polygon, initialText = "") {
  const id = crypto.randomUUID();
  const next = mutate(state, (annotation) => {
    annotation.blocks.push({
      id,
      order: annotation.blocks.length,
      text: initialText,
      polygon: clampPolygon(annotation, polygon),
      score: null,
      source: "manual",
    });
    return annotation;
  });
  return { ...next, selectedId: id };
}

export function deleteBlock(state, id) {
  const next = mutate(state, (annotation) => {
    const index = annotation.blocks.findIndex((block) => block.id === id);
    if (index === -1) throw new Error(`Unknown block: ${id}`);
    annotation.blocks.splice(index, 1);
    return annotation;
  });
  return state.selectedId === id ? { ...next, selectedId: null } : next;
}

export function reorderBlock(state, id, targetIndex) {
  return mutate(state, (annotation) => {
    const sourceIndex = annotation.blocks.findIndex((block) => block.id === id);
    if (sourceIndex === -1) throw new Error(`Unknown block: ${id}`);
    const [block] = annotation.blocks.splice(sourceIndex, 1);
    const index = Math.min(Math.max(targetIndex, 0), annotation.blocks.length);
    annotation.blocks.splice(index, 0, block);
    annotation.blocks.forEach((item, order) => {
      item.order = order;
    });
    return annotation;
  });
}

export function setStatus(state, status) {
  if (!["ocr", "edited", "completed"].includes(status)) {
    throw new Error(`Unknown annotation status: ${status}`);
  }
  if (status === "completed" && state.annotation.blocks.some((block) => !block.text.trim())) {
    throw new Error("Không thể hoàn tất khi còn block rỗng");
  }
  return mutate(
    state,
    (annotation) => {
      annotation.status = status;
      return annotation;
    },
    { preserveStatus: true },
  );
}

export function undo(state) {
  if (state.undoStack.length === 0) return state;
  const annotation = clone(state.undoStack.at(-1));
  annotation.revision = state.annotation.revision;
  return {
    ...state,
    annotation,
    undoStack: state.undoStack.slice(0, -1),
    redoStack: [...state.redoStack, clone(state.annotation)],
    dirty: true,
  };
}

export function redo(state) {
  if (state.redoStack.length === 0) return state;
  const annotation = clone(state.redoStack.at(-1));
  annotation.revision = state.annotation.revision;
  return {
    ...state,
    annotation,
    undoStack: [...state.undoStack, clone(state.annotation)],
    redoStack: state.redoStack.slice(0, -1),
    dirty: true,
  };
}

export function acknowledgeSave(state, savedAnnotation = state.annotation) {
  return {
    ...state,
    annotation: normalized(savedAnnotation),
    dirty: false,
  };
}

export function rebaseSaveAcknowledgement(state, savedAnnotation) {
  const annotation = clone(state.annotation);
  annotation.revision = savedAnnotation.revision;
  annotation.updated_at = savedAnnotation.updated_at ?? annotation.updated_at;
  return {
    ...state,
    annotation: normalized(annotation),
    dirty: true,
  };
}

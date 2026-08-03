export class APIError extends Error {
  constructor(status, detail) {
    super(detail);
    this.name = "APIError";
    this.status = status;
  }
}

export function isCurrentResponse(request, current) {
  return request.generation === current.generation
    && request.imageId === current.imageId
    && request.mutationVersion === current.mutationVersion;
}

export function canNavigateAfterSave(succeeded) {
  if (typeof succeeded === "boolean") return succeeded;
  return succeeded.succeeded === true && succeeded.dirty === false;
}

export function canProcessInteraction({ workspaceOpening }) {
  return workspaceOpening !== true;
}

export function shouldContinueBatchPoll(snapshot) {
  return ["queued", "running", "cancelling"].includes(snapshot.state);
}

export function nextBatchPollDelay({ active, pending, terminal }) {
  if (pending) return 0;
  if (terminal) return null;
  return active ? 750 : null;
}

export function shouldApplyResponseError(request, current) {
  return isCurrentResponse(request, current);
}

export function needsDeleteConfirmation(block) {
  return Boolean(block?.text?.trim());
}

export function shouldPanPointer(button, spacePan) {
  return button === 1 || (button === 0 && spacePan === true);
}

export function polygonClassNames(block, selected) {
  const tone = selected
    ? "polygon--selected"
    : block.source === "ocr" && block.score !== null && block.score < 0.6
      ? "polygon--low-confidence"
      : "polygon--text";
  return `polygon--${block.source} ${tone}`;
}

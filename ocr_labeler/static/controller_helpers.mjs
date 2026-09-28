export class APIError extends Error {
  constructor(status, detail) {
    super(detail);
    this.name = "APIError";
    this.status = status;
  }
}

export const BATCH_POLL_ACTIVE_STATES = Object.freeze(["queued", "running", "cancelling"]);
export const BATCH_POLL_DELAY_MS = 750;
export const OCR_LOW_CONFIDENCE_THRESHOLD = 0.6;

export function isCurrentResponse(request, current) {
  if (!request || !current) return false;
  return request.generation === current.generation
    && request.imageId === current.imageId
    && request.mutationVersion === current.mutationVersion;
}

export function canNavigateAfterSave(succeeded) {
  if (typeof succeeded === "boolean") return succeeded;
  return succeeded?.succeeded === true && succeeded?.dirty === false;
}

export function canProcessInteraction(state = {}) {
  return state?.workspaceOpening !== true;
}

export function shouldContinueBatchPoll(snapshot) {
  return BATCH_POLL_ACTIVE_STATES.includes(snapshot?.state);
}

export function nextBatchPollDelay(options = {}) {
  const { active = false, pending = false, terminal = false } = options || {};
  if (pending) return 0;
  if (terminal) return null;
  return active ? BATCH_POLL_DELAY_MS : null;
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
  if (!block) return "";
  let tone = "polygon--text";
  if (selected) {
    tone = "polygon--selected";
  } else if (
    block.source === "ocr" &&
    typeof block.score === "number" &&
    block.score < OCR_LOW_CONFIDENCE_THRESHOLD
  ) {
    tone = "polygon--low-confidence";
  }
  return `polygon--${block.source ?? "unknown"} ${tone}`;
}

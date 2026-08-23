export const VALIDATION_STORAGE_KEY = "vl-layout-labeler.llm-validation";

export function loadValidationEnabled(storage, configured) {
  if (!configured) return false;
  try {
    return storage.getItem(VALIDATION_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

export function saveValidationEnabled(storage, enabled) {
  try {
    storage.setItem(VALIDATION_STORAGE_KEY, String(Boolean(enabled)));
  } catch {
    // The toggle still works for this session when storage is unavailable.
  }
}

export function validationIssueCount(block) {
  return block?.validation?.issues?.length || 0;
}

export function invalidateBlockValidation(block) {
  if (block) block.validation = null;
}

export function validationPreviewParts(text, issue) {
  const start = Number.isInteger(issue?.start) ? issue.start : -1;
  const end = Number.isInteger(issue?.end) ? issue.end : -1;
  const codePoints = Array.from(text);
  if (start < 0 || end <= start || end > codePoints.length) return null;
  return {
    before: codePoints.slice(0, start).join(""),
    marked: codePoints.slice(start, end).join(""),
    after: codePoints.slice(end).join(""),
  };
}

export function validationSelectionRange(text, issue) {
  const parts = validationPreviewParts(text, issue);
  if (!parts) return null;
  return {
    start: parts.before.length,
    end: parts.before.length + parts.marked.length,
  };
}

export function appendValidationPreview(document, container, text, issue) {
  const parts = validationPreviewParts(text, issue);
  if (!parts) return false;
  container.append(document.createTextNode(parts.before));
  const mark = document.createElement("mark");
  mark.textContent = parts.marked;
  container.append(mark, document.createTextNode(parts.after));
  return true;
}

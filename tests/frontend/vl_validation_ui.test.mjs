import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  VALIDATION_STORAGE_KEY,
  appendValidationPreview,
  invalidateBlockValidation,
  loadValidationEnabled,
  saveValidationEnabled,
  validationIssueCount,
  validationPreviewParts,
  validationSelectionRange,
} from "../../vl_layout_labeler/static/validation_ui.mjs";


function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    values,
  };
}

test("validation toggle persists but remains off when capability is unavailable", () => {
  const storage = memoryStorage({ [VALIDATION_STORAGE_KEY]: "true" });
  assert.equal(loadValidationEnabled(storage, true), true);
  assert.equal(loadValidationEnabled(storage, false), false);
  saveValidationEnabled(storage, false);
  assert.equal(storage.values.get(VALIDATION_STORAGE_KEY), "false");
});

test("warning count and text edits invalidate old metadata", () => {
  const block = { validation: { issues: [{}, {}] } };
  assert.equal(validationIssueCount(block), 2);
  invalidateBlockValidation(block);
  assert.equal(validationIssueCount(block), 0);
  assert.equal(block.validation, null);
});

test("preview uses exact half-open offsets", () => {
  assert.deepEqual(validationPreviewParts("Vỉet Nam", { start: 0, end: 3 }), {
    before: "",
    marked: "Vỉe",
    after: "t Nam",
  });
  assert.equal(validationPreviewParts("text", { start: 0, end: 9 }), null);
  assert.deepEqual(validationPreviewParts("😀Vỉet", { start: 1, end: 4 }), {
    before: "😀",
    marked: "Vỉe",
    after: "t",
  });
  assert.deepEqual(validationSelectionRange("😀Vỉet", { start: 1, end: 4 }), {
    start: 2,
    end: 5,
  });
});

test("preview creates text nodes and one mark without innerHTML", () => {
  const container = { children: [], append(...items) { this.children.push(...items); } };
  const document = {
    createTextNode: (text) => ({ nodeType: 3, textContent: text }),
    createElement: (tagName) => ({ tagName, textContent: "" }),
  };
  assert.equal(
    appendValidationPreview(document, container, "Vỉet Nam", { start: 0, end: 3 }),
    true,
  );
  assert.deepEqual(
    container.children.map((item) => [item.nodeType || item.tagName, item.textContent]),
    [[3, ""], ["mark", "Vỉe"], [3, "t Nam"]],
  );
});

test("application contract locks duplicate requests and focuses exact raw span", () => {
  const script = readFileSync("vl_layout_labeler/static/app.mjs", "utf8");
  const html = readFileSync("vl_layout_labeler/static/index.html", "utf8");
  assert.match(script, /if \(!state\.currentId \|\| state\.busy \|\| state\.batchBusy\) return/);
  assert.match(script, /setSelectionRange\(range\.start, range\.end\)/);
  assert.match(script, /invalidateBlockValidation\(block\)/);
  assert.doesNotMatch(script, /innerHTML/);
  for (const id of [
    "llm-validation",
    "validate-current",
    "validation-panel",
    "validation-issues",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
});

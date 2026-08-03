import test from "node:test";
import assert from "node:assert/strict";

import {
  APIError,
  canNavigateAfterSave,
  canProcessInteraction,
  isCurrentResponse,
  needsDeleteConfirmation,
  nextBatchPollDelay,
  polygonClassNames,
  shouldPanPointer,
  shouldApplyResponseError,
  shouldContinueBatchPoll,
} from "../../ocr_labeler/static/controller_helpers.mjs";

test("API errors preserve HTTP status for conflict policy", () => {
  const error = new APIError(409, "expected revision 2, got 1");
  assert.equal(error.status, 409);
  assert.equal(error.message, "expected revision 2, got 1");
});

test("response gating rejects stale navigation and local-mutation responses", () => {
  const current = { generation: 4, imageId: "new", mutationVersion: 7 };
  assert.equal(isCurrentResponse({ generation: 3, imageId: "old", mutationVersion: 0 }, current), false);
  assert.equal(isCurrentResponse({ generation: 4, imageId: "new", mutationVersion: 6 }, current), false);
  assert.equal(isCurrentResponse({ generation: 4, imageId: "new", mutationVersion: 7 }, current), true);
});

test("navigation guard allows only a successful pending save", () => {
  assert.equal(canNavigateAfterSave(true), true);
  assert.equal(canNavigateAfterSave(false), false);
  assert.equal(canNavigateAfterSave({ succeeded: true, dirty: true }), false);
  assert.equal(canNavigateAfterSave({ succeeded: true, dirty: false }), true);
});

test("workspace interaction lock gates editing and navigation", () => {
  assert.equal(canProcessInteraction({ workspaceOpening: false }), true);
  assert.equal(canProcessInteraction({ workspaceOpening: true }), false);
});

test("batch polling continues only for active batch states", () => {
  assert.equal(shouldContinueBatchPoll({ state: "running" }), true);
  assert.equal(shouldContinueBatchPoll({ state: "cancelling" }), true);
  assert.equal(shouldContinueBatchPoll({ state: "completed" }), false);
  assert.equal(shouldContinueBatchPoll({ state: "failed" }), false);
});

test("an active poll preserves one explicit queued follow-up", () => {
  assert.equal(nextBatchPollDelay({ active: true, pending: true, terminal: false }), 0);
  assert.equal(nextBatchPollDelay({ active: true, pending: false, terminal: false }), 750);
  assert.equal(nextBatchPollDelay({ active: false, pending: false, terminal: true }), null);
});

test("a pending explicit wakeup wins over an older terminal poll response", () => {
  assert.equal(nextBatchPollDelay({ active: false, pending: true, terminal: true }), 0);
  assert.equal(nextBatchPollDelay({ active: false, pending: false, terminal: true }), null);
});

test("stale response errors are ignored by the same token policy", () => {
  const current = { generation: 5, imageId: "new", mutationVersion: 3 };
  assert.equal(shouldApplyResponseError({ generation: 4, imageId: "old", mutationVersion: 1 }, current), false);
  assert.equal(shouldApplyResponseError({ generation: 5, imageId: "new", mutationVersion: 3 }, current), true);
});

test("only non-empty blocks require deletion confirmation", () => {
  assert.equal(needsDeleteConfirmation({ text: "Có nội dung" }), true);
  assert.equal(needsDeleteConfirmation({ text: " \n " }), false);
});

test("middle pointer pans without Space and left pointer pans with Space", () => {
  assert.equal(shouldPanPointer(1, false), true);
  assert.equal(shouldPanPointer(0, true), true);
  assert.equal(shouldPanPointer(0, false), false);
  assert.equal(shouldPanPointer(2, true), false);
});

test("low-confidence OCR polygons are orange unless selected", () => {
  const low = { source: "ocr", score: 0.42 };
  assert.match(polygonClassNames(low, false), /polygon--low-confidence/);
  assert.doesNotMatch(polygonClassNames(low, true), /polygon--low-confidence/);
  assert.match(polygonClassNames(low, true), /polygon--selected/);
});

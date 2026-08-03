import test from "node:test";
import assert from "node:assert/strict";

import {
  acknowledgeSave,
  addBlock,
  createEditorState,
  deleteBlock,
  moveBlock,
  moveCorner,
  redo,
  rebaseSaveAcknowledgement,
  reorderBlock,
  resizeBlock,
  selectBlock,
  setStatus,
  undo,
  updateText,
} from "../../ocr_labeler/static/state.mjs";

const annotation = {
  revision: 1,
  status: "ocr",
  text: "A\nB",
  image: { width: 100, height: 80 },
  blocks: [
    {
      id: "a",
      order: 0,
      text: "A",
      polygon: [[0, 0], [20, 0], [20, 10], [0, 10]],
      source: "ocr",
    },
    {
      id: "b",
      order: 1,
      text: "B",
      polygon: [[0, 20], [20, 20], [20, 30], [0, 30]],
      source: "ocr",
    },
  ],
};

const stateFor = () => createEditorState(annotation);

test("selection accepts blocks or null without changing history and rejects unknown ids", () => {
  const initial = stateFor();
  const selected = selectBlock(initial, "b");
  const cleared = selectBlock(selected, null);

  assert.notStrictEqual(selected, initial);
  assert.equal(initial.selectedId, "a");
  assert.equal(selected.selectedId, "b");
  assert.equal(cleared.selectedId, null);
  assert.strictEqual(selected.annotation, initial.annotation);
  assert.strictEqual(selected.undoStack, initial.undoStack);
  assert.strictEqual(cleared.redoStack, initial.redoStack);
  assert.throws(() => selectBlock(initial, "missing"), /Unknown block: missing/);
});

test("text mutation rebuilds aggregate text and undo restores it", () => {
  const initial = stateFor();
  const changed = updateText(initial, "a", "Xin");

  assert.notStrictEqual(changed, initial);
  assert.equal(changed.annotation.text, "Xin\nB");
  assert.equal(changed.annotation.status, "edited");
  assert.equal(changed.undoStack.length, 1);
  assert.equal(undo(changed).annotation.text, "A\nB");
});

test("corner movement clamps to image bounds", () => {
  const changed = moveCorner(stateFor(), "a", 0, -5, 100);

  assert.deepEqual(changed.annotation.blocks[0].polygon[0], [0, 79]);
});

test("moving and resizing blocks keep every polygon point within image bounds", () => {
  const moved = moveBlock(stateFor(), "a", 200, -50);
  const resized = resizeBlock(moved, "a", 2, 200, -20);

  assert.deepEqual(moved.annotation.blocks[0].polygon, [
    [79, 0], [99, 0], [99, 10], [79, 10],
  ]);
  assert.deepEqual(resized.annotation.blocks[0].polygon[2], [99, 0]);
  assert.equal(resized.undoStack.length, 2);
});

test("reorder writes contiguous order values and rebuilds text in reading order", () => {
  const state = reorderBlock(stateFor(), "b", 0);

  assert.deepEqual(
    state.annotation.blocks.map((block) => [block.id, block.order]),
    [["b", 0], ["a", 1]],
  );
  assert.equal(state.annotation.text, "B\nA");
});

test("manual block creation and deletion participate in history", () => {
  const initial = stateFor();
  const added = addBlock(initial, [[5, 5], [15, 5], [15, 15], [5, 15]]);
  const id = added.selectedId;
  const removed = deleteBlock(added, id);

  assert.equal(added.annotation.blocks.at(-1).source, "manual");
  assert.equal(added.annotation.blocks.at(-1).text, "");
  assert.equal(removed.annotation.blocks.length, 2);
  assert.equal(undo(removed).annotation.blocks.length, 3);
});

test("a new mutation clears redo and undo redo preserve the current selection", () => {
  const changed = updateText(selectBlock(stateFor(), "b"), "a", "Xin");
  const undone = undo(changed);
  const branched = updateText(undone, "a", "Chao");
  const redone = redo(undone);

  assert.equal(undone.selectedId, "b");
  assert.equal(redone.selectedId, "b");
  assert.equal(undone.redoStack.length, 1);
  assert.equal(branched.redoStack.length, 0);
  assert.equal(branched.undoStack.length, 1);
});

test("completion rejects empty blocks", () => {
  const empty = updateText(stateFor(), "a", "  ");
  assert.throws(
    () => setStatus(empty, "completed"),
    /Không thể hoàn tất khi còn block rỗng/,
  );
});

test("every ordinary mutation reopens a completed annotation for editing", () => {
  const completed = setStatus(stateFor(), "completed");
  assert.equal(completed.annotation.status, "completed");

  const mutations = [
    updateText(completed, "a", "Đã sửa"),
    moveCorner(completed, "a", 0, 3, 4),
    moveBlock(completed, "a", 2, 3),
    addBlock(completed, [[5, 5], [15, 5], [15, 15], [5, 15]]),
    deleteBlock(completed, "a"),
    reorderBlock(completed, "b", 0),
  ];
  assert.deepEqual(
    mutations.map((state) => state.annotation.status),
    Array(mutations.length).fill("edited"),
  );
});

test("undo restores completed and redo restores the edited mutation", () => {
  const completed = setStatus(stateFor(), "completed");
  const edited = updateText(completed, "a", "Đã sửa");
  const undone = undo(edited);
  const redone = redo(undone);

  assert.equal(edited.annotation.status, "edited");
  assert.equal(undone.annotation.status, "completed");
  assert.equal(redone.annotation.status, "edited");
});

test("save acknowledgement accepts the server revision without creating history", () => {
  const changed = updateText(stateFor(), "a", "Xin");
  const acknowledged = acknowledgeSave(changed, {
    ...changed.annotation,
    revision: 2,
  });

  assert.notStrictEqual(acknowledged, changed);
  assert.equal(acknowledged.annotation.revision, 2);
  assert.equal(acknowledged.dirty, false);
  assert.equal(acknowledged.selectedId, changed.selectedId);
  assert.strictEqual(acknowledged.undoStack, changed.undoStack);
  assert.strictEqual(acknowledged.redoStack, changed.redoStack);
});

test("undo and redo retain the acknowledged authoritative revision", () => {
  const changed = updateText(stateFor(), "a", "Xin");
  const acknowledged = acknowledgeSave(changed, { ...changed.annotation, revision: 2 });

  const undone = undo(acknowledged);
  const redone = redo(undone);

  assert.equal(undone.annotation.revision, 2);
  assert.equal(redone.annotation.revision, 2);
});

test("in-flight save acknowledgement rebases a newer local document and keeps it dirty", () => {
  const sent = updateText(stateFor(), "a", "Bản đã gửi");
  const newer = updateText(sent, "a", "Bản mới hơn");
  const rebased = rebaseSaveAcknowledgement(newer, { ...sent.annotation, revision: 2 });

  assert.equal(rebased.annotation.revision, 2);
  assert.equal(rebased.annotation.blocks[0].text, "Bản mới hơn");
  assert.equal(rebased.dirty, true);
  assert.equal(rebased.undoStack.length, newer.undoStack.length);
});

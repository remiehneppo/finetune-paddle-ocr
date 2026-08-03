import test from "node:test";
import assert from "node:assert/strict";

import {
  addBlock,
  createEditorState,
  setStatus,
  updateText,
} from "../../ocr_labeler/static/state.mjs";

const annotation = {
  revision: 0,
  status: "edited",
  text: "",
  image: { width: 100, height: 80 },
  blocks: [],
};

test("detection boxes can start with a non-empty PaddleOCR transcription", () => {
  const added = addBlock(
    createEditorState(annotation),
    [[1, 2], [40, 2], [40, 12], [1, 12]],
    "text",
  );

  assert.equal(added.annotation.blocks[0].text, "text");
  assert.equal(added.annotation.blocks[0].source, "manual");
  assert.equal(added.annotation.blocks[0].score, null);
  assert.doesNotThrow(() => setStatus(added, "completed"));
});

test("detection boxes can be marked ignored with PaddleOCR marker", () => {
  const added = addBlock(
    createEditorState(annotation),
    [[1, 2], [40, 2], [40, 12], [1, 12]],
    "text",
  );
  const ignored = updateText(
    added,
    added.annotation.blocks[0].id,
    "###",
  );

  assert.equal(ignored.annotation.blocks[0].text, "###");
  assert.doesNotThrow(() => setStatus(ignored, "completed"));
});

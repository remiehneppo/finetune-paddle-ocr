# Native PaddleOCR-VL Evaluation Design

## Goal

Replace the custom PaddleOCR-VL decoding path with a native Hugging Face evaluation path that uses the same processor, chat template, image tokens, and deterministic generation contract as vLLM. Evaluate the original base model and the merged Vietnamese LoRA model on identical validation fixtures, then publish reproducible metrics in the adapter model card.

## Scope

- Modify `evaluate_paddleocr_vl.py` and its focused tests.
- Evaluate the base snapshot and merged export; adapter-to-merge integrity remains covered by the existing weight/logit verification instead of loading PaddleFormers LoRA in the Hugging Face evaluator.
- Preserve the exact user text `OCR:`.
- Report overall and per-dataset CER, exact match, and normalized edit distance.
- Update and push the Hugging Face adapter model card only after a successful evaluation.

Checkpoint ranking from the old evaluator is out of scope because its multimodal input contract was invalid and only the final three late checkpoints remain.

## Architecture

`evaluate_paddleocr_vl.py` will load `AutoProcessor` and the model's Hugging Face conditional-generation class from each candidate directory with remote model code enabled. A prompt helper will construct one user message containing one image and the exact text `OCR:`, then call the processor's native chat template with a generation prompt. The processor will expand the image placeholder and produce all tensors passed to the model.

Generation will be deterministic: sampling disabled, one beam, no temperature/top-p/top-k sampling, and an explicit maximum number of new tokens. Only newly generated token IDs will be decoded. The same selected rows and generation settings will be used for base and merged candidates.

## Data Flow

1. Read the prepared validation JSONL files and validate their existing ERNIEKit mask contract.
2. Select the first fixed number of rows from each source, preserving the current deterministic fixture behavior.
3. Load the native processor and base model, generate predictions, and calculate metrics.
4. Release the base model, load the merged model, repeat on the exact same fixtures, and calculate metrics.
5. Write `ocr_predictions.jsonl` and `ocr_metrics.json` with runtime metadata and candidate results.
6. Render the verified base-versus-merged results into the Hugging Face model card and upload a new Hub commit.

## Error Handling

- Fail if a validation row, image path, or expected `OCR:` mask contract is invalid.
- Fail if the processor lacks a chat template or fails to produce image tensors.
- Fail if generated outputs are shorter than the prompt length or candidate fixture coverage differs.
- Never publish metrics when evaluation exits non-zero or a candidate has incomplete coverage.

## Testing

- A regression test must fail against the old implementation by asserting that the native chat-template path is used rather than manually concatenating image-token IDs and `OCR:`.
- Unit tests cover deterministic generation arguments, prompt construction, decoding only newly generated tokens, and identical fixture coverage for candidate comparison.
- Existing loader and metric tests remain green.
- Integration verification runs at least one fixture through base and merged before the full evaluation.
- Final verification checks the generated report, prediction count, model-card metric values, and the Hugging Face commit contents.

## Success Criteria

- Base inference no longer produces the systematic repeated location-token garbage caused by the malformed prompt path.
- Base and merged candidates evaluate the same number of fixtures with deterministic decoding.
- The report contains overall and per-source metrics for both candidates.
- The public adapter model card identifies the evaluation settings, fixture count, and measured base-versus-merged results without reusing any metric from the invalid evaluator.

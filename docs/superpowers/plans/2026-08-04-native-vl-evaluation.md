# Native PaddleOCR-VL Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid manual Paddle evaluator with deterministic native Hugging Face multimodal inference, evaluate the base and merged Vietnamese model on identical fixtures, and publish the verified metrics in the adapter model card.

**Architecture:** The evaluator will use the model snapshot's `AutoProcessor` chat template and `AutoModelForCausalLM` remote-code mapping, so input construction and generation match the native contract used by vLLM. Pure helpers will isolate message construction, generation options, new-token decoding, and candidate coverage for unit testing; model loading remains lazy so unit tests do not require Torch or Transformers.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu128 from system site packages, Transformers 4.55.4, Pillow, unittest/pytest, Hugging Face Hub CLI.

## Global Constraints

- Preserve the exact user text `OCR:`.
- Use native BOS, role, image-boundary, and assistant-generation tokens from `chat_template.jinja`.
- Use deterministic decoding: `do_sample=False`, `num_beams=1`, and no temperature, top-p, or top-k sampling.
- Compare base and merged models on the same fixed validation rows.
- Do not reuse metrics produced by the invalid manual evaluator.
- Do not publish the model card unless full candidate coverage and evaluation exit code are successful.

---

### Task 1: Native multimodal contract regression tests

**Files:**
- Modify: `tests/test_finetune_vl.py`
- Test: `tests/test_finetune_vl.py`

**Interfaces:**
- Consumes: existing `evaluate_paddleocr_vl.PROMPT` and validation-row dictionaries.
- Produces: required interfaces `ocr_messages(image)`, `deterministic_generation_kwargs(max_new_tokens)`, `decode_new_tokens(processor, generated_ids, prompt_token_count)`, and `validate_candidate_coverage(predictions, candidates, fixture_count)`.

- [ ] **Step 1: Replace obsolete manual-decoder tests with failing native-contract tests**

```python
def test_evaluator_builds_native_ocr_message():
    assert evaluate_paddleocr_vl.ocr_messages("line.png") == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "line.png"},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]

def test_evaluator_uses_deterministic_native_generation():
    assert evaluate_paddleocr_vl.deterministic_generation_kwargs(64) == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 64,
        "use_cache": True,
    }
```

Add a fake processor test proving `decode_new_tokens` slices off exactly `prompt_token_count` IDs before calling `batch_decode`, plus coverage tests that reject a missing base or merged prediction.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_finetune_vl.py -k 'native_ocr_message or deterministic_native_generation or decode_new_tokens or candidate_coverage'
```

Expected: failures because the four native helper interfaces do not exist.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_finetune_vl.py
git commit -m "test: define native VL evaluation contract"
```

### Task 2: Native Hugging Face evaluator

**Files:**
- Modify: `evaluate_paddleocr_vl.py`
- Modify: `finetune_vl.py`
- Create: `requirements-vl-eval.txt`
- Test: `tests/test_finetune_vl.py`

**Interfaces:**
- Consumes: Task 1 helper contracts, `load_validation_rows`, and `finetune_vl.compute_ocr_metrics`.
- Produces: CLI `--base-model`, `--merged-model`, `--validation-jsonl`, `--output-dir`, `--samples-per-dataset`, `--max-new-tokens`, and JSON reports containing candidates `base` and `merged`.

- [ ] **Step 1: Implement the pure helper functions minimally**

```python
def ocr_messages(image: str) -> list[dict[str, object]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]

def deterministic_generation_kwargs(max_new_tokens: int) -> dict[str, object]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
```

`decode_new_tokens` must slice `generated_ids[:, prompt_token_count:]`, and `validate_candidate_coverage` must require every candidate to cover every fixture exactly once.

- [ ] **Step 2: Run focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all selected tests pass.

- [ ] **Step 3: Add failing orchestration tests**

Use fake `AutoProcessor`, model, Torch tensors, and image loader to assert:

```python
rendered = processor.apply_chat_template(
    ocr_messages(image_path), tokenize=False, add_generation_prompt=True
)
inputs = processor(text=[rendered], images=[image], return_tensors="pt")
generated = model.generate(**inputs, **deterministic_generation_kwargs(64))
```

Assert that the report has only `base` and `merged`, both have equal fixture counts, and no Paddle/LoRA/checkpoint-loading API is called.

- [ ] **Step 4: Run orchestration test and verify RED**

Run:

```bash
pytest -q tests/test_finetune_vl.py -k 'native_candidate or native_processor'
```

Expected: failure because `evaluate` still imports Paddle/ERNIEKit and manually creates image tokens and 3D positions.

- [ ] **Step 5: Replace the manual runtime with native Transformers**

Load dependencies lazily inside `evaluate`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained(
    str(model_path), trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda").eval()
```

For every image, render the native chat template, call the processor with both text and PIL RGB image, move tensor inputs to CUDA, call `generate`, decode only new tokens, and record prediction/runtime. Load and release the base candidate before loading merged to stay within 16 GB VRAM.

Remove obsolete manual-position, augmentation patch, checkpoint selection, and Paddle LoRA paths. Make the CLI require `--merged-model` and stop accepting `--adapter-dir`, `--min-pixels`, `--max-pixels`, and `--max-checkpoints`.

- [ ] **Step 6: Update the finetune pipeline command contract**

Change `finetune_vl.evaluation_command` so it invokes the evaluator only after merge and passes the base model, merged export, validation paths, output directory, fixture count, and max-new-token limit. Do not use the invalid evaluator to select an adapter checkpoint; the final adapter remains the export source.

- [ ] **Step 7: Add the isolated evaluation dependency file**

Create `requirements-vl-eval.txt`:

```text
--extra-index-url https://download.pytorch.org/whl/cu128
transformers==4.55.4
safetensors>=0.4.5
Pillow>=10.0.0
```

The venv will be created with `python3 -m venv --system-site-packages .venv-vl-eval` to reuse the verified system PyTorch 2.11.0+cu128.

- [ ] **Step 8: Run evaluator and pipeline unit tests**

```bash
pytest -q tests/test_finetune_vl.py
```

Expected: all tests pass, including the revised evaluator-command contract.

- [ ] **Step 9: Commit implementation**

```bash
git add evaluate_paddleocr_vl.py finetune_vl.py requirements-vl-eval.txt tests/test_finetune_vl.py
git commit -m "fix: evaluate VL models with native processor"
```

### Task 3: GPU integration and full evaluation

**Files:**
- Create runtime artifact: `.venv-vl-eval/` (ignored, not committed)
- Create report: `runs/vl16_vi_full_v2/metrics/native_evaluation/ocr_metrics.json`
- Create predictions: `runs/vl16_vi_full_v2/metrics/native_evaluation/ocr_predictions.jsonl`

**Interfaces:**
- Consumes: native evaluator from Task 2, local base model, merged export, and five prepared validation JSONLs.
- Produces: verified base-versus-merged metrics used by Task 4.

- [ ] **Step 1: Create the isolated runtime and install pinned dependencies**

```bash
python3 -m venv --system-site-packages .venv-vl-eval
.venv-vl-eval/bin/pip install -r requirements-vl-eval.txt
```

- [ ] **Step 2: Run one-fixture GPU smoke evaluation**

```bash
.venv-vl-eval/bin/python evaluate_paddleocr_vl.py \
  --base-model /home/tieubaoca/AI/models/paddleocr-cache/official_models/PaddleOCR-VL-1.6 \
  --merged-model runs/vl16_vi_full_v2/adapter/export \
  --validation-jsonl runs/vl16_vi_all_datasets_prepare/prepared/validation-source-000.jsonl \
  --output-dir runs/vl16_vi_full_v2/metrics/native_smoke \
  --samples-per-dataset 1 \
  --max-new-tokens 256
```

Expected: exit 0, two predictions, one for each candidate, and neither prediction shows the systematic malformed-prompt location-token loop.

- [ ] **Step 3: Run full fixed-fixture evaluation**

Run with all five validation JSONLs, `--samples-per-dataset 32`, and `--output-dir runs/vl16_vi_full_v2/metrics/native_evaluation`. Expected: 160 predictions per candidate and a report containing overall plus five per-source groups for both base and merged.

- [ ] **Step 4: Validate report integrity**

Check that candidates are exactly `base` and `merged`, each has 160 samples, all metric values are finite, prompt is exactly `OCR:`, decoding is marked deterministic, and the prediction JSONL has exactly 320 records.

### Task 4: Publish verified metrics to Hugging Face

**Files:**
- Modify: `runs/vl16_vi_full_v2/hf_adapter_release/README.md`

**Interfaces:**
- Consumes: verified `native_evaluation/ocr_metrics.json` from Task 3.
- Produces: public model-card metric table and a new Hub commit.

- [ ] **Step 1: Update the model card**

Replace the pending-evaluation section with:

```markdown
## Evaluation

Deterministic greedy decoding with the native PaddleOCR-VL processor and chat template, prompt `OCR:`, 32 fixed validation samples from each of five sources, and 256 maximum new tokens.

| Model | Samples | CER | Exact match | Normalized edit distance |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.6 base | ... | ... | ... | ... |
| Vietnamese LoRA merged | ... | ... | ... | ... |
```

Populate the table only from the verified report and add the five per-source merged metrics below it.

- [ ] **Step 2: Verify local card values against JSON**

Run a read-only comparison script that parses `ocr_metrics.json`, extracts every numeric value rendered in the table, and fails on any mismatch.

- [ ] **Step 3: Upload the updated card**

```bash
hf upload tieubaoca/PaddleOCR-VL-1.6-Vietnamese-LoRA \
  runs/vl16_vi_full_v2/hf_adapter_release/README.md README.md \
  --repo-type model \
  --commit-message "Add native OCR evaluation results"
```

- [ ] **Step 4: Verify the Hub commit**

Fetch the public README and confirm the commit contains the exact local metric table, prompt, fixture count, and deterministic decoding description.

- [ ] **Step 5: Run final verification**

```bash
pytest -q tests/test_finetune_vl.py
git diff --check
hf download tieubaoca/PaddleOCR-VL-1.6-Vietnamese-LoRA README.md --dry-run
```

Expected: tests exit 0, no whitespace errors, and the Hub README resolves successfully.

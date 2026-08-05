# Reuse Prepared Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--prepared-from RUN_DIR` so a new PaddleOCR-VL-1.6 LoRA run safely references an existing prepared dataset without copying or reprocessing it.

**Architecture:** Add a mutually exclusive fresh-data input at the CLI boundary and a focused `load_prepared_run()` integrity loader. The main fresh-run path either prepares source datasets or loads the prepared summary, then uses the existing `create_resolved_config()` seam unchanged.

**Tech Stack:** Python 3, argparse, JSON/JSONL, pathlib, unittest, PyYAML, Ruff.

## Global Constraints

- The referenced prepared run is read-only and no image or JSONL is copied.
- Every JSONL record and referenced image path is checked, but images are not decoded.
- Every OCR sample must keep the exact prompt `OCR:` and ERNIEKit mask contract.
- `--prepared-from` cannot be combined with `--dataset-dir`, `--prepare-only`, or `--resume-from`.
- Existing non-empty work directories must never be overwritten.

---

### Task 1: CLI and prepared-run validation

**Files:**
- Modify: `finetune_vl.py`
- Test: `tests/test_finetune_vl.py`

**Interfaces:**
- Consumes: existing `parse_args(argv)`, `validate_args(args)`, and ERNIEKit JSONL mask contract.
- Produces: `args.prepared_from: Path | None` and `load_prepared_run(prepared_from: Path, work_dir: Path) -> dict[str, Any]`.

- [ ] **Step 1: Write failing CLI validation tests**

Add tests that parse `--prepared-from`, allow it without `--dataset-dir`, and
reject no data input, both data inputs, `--prepare-only`, and `--resume-from`.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m unittest tests.test_finetune_vl.FinetuneVLTests.test_prepared_from_cli_contract -v`

Expected: FAIL because `--dataset-dir` is still required or the new flag does not exist.

- [ ] **Step 3: Write failing integrity-loader tests**

Create a two-record fixture with absolute JSONL paths and real image files.
Assert the returned summary has canonical JSONL paths, exact counts, and
`prepared_from`. Add corrupt fixtures for a missing image and a sample-count
mismatch.

- [ ] **Step 4: Run the loader tests and confirm failure**

Run: `python -m unittest tests.test_finetune_vl.FinetuneVLTests.test_load_prepared_run_validates_jsonl_and_images -v`

Expected: FAIL because `load_prepared_run` does not exist.

- [ ] **Step 5: Implement the minimal CLI and integrity loader**

Make `--dataset-dir` optional, add `--prepared-from`, enforce the combinations
in `validate_args`, and implement a streaming JSONL validator that checks the
two `text_info` entries, one matched image, file existence, source totals, and
probability vector lengths.

- [ ] **Step 6: Run the focused tests and confirm they pass**

Run: `python -m unittest tests.test_finetune_vl.FinetuneVLTests.test_prepared_from_cli_contract tests.test_finetune_vl.FinetuneVLTests.test_load_prepared_run_validates_jsonl_and_images -v`

Expected: PASS.

### Task 2: Main flow and documentation

**Files:**
- Modify: `finetune_vl.py`
- Modify: `docs/finetune-paddleocr-vl-1.6.md`
- Modify: `README.md`
- Test: `tests/test_finetune_vl.py`

**Interfaces:**
- Consumes: `load_prepared_run()` from Task 1 and existing `create_resolved_config()`.
- Produces: a fresh run containing `summary.json` and `resolved.yaml` whose dataset paths reference the prepared run.

- [ ] **Step 1: Write a failing main-flow test**

Patch `load_tokenizer` and `prepare_datasets` to raise if called. Invoke `main`
with `--prepared-from`, `--prepare-only` disabled, and model inspection patched
to return. Assert `resolved.yaml` references the original JSONL and the new run
has no `prepared/` directory.

- [ ] **Step 2: Run the main-flow test and confirm failure**

Run: `python -m unittest tests.test_finetune_vl.FinetuneVLTests.test_main_reuses_prepared_run_without_preparing_or_copying -v`

Expected: FAIL because `main()` always loads a tokenizer and prepares datasets.

- [ ] **Step 3: Route the main flow through the prepared-run loader**

When `args.prepared_from` is present, call `load_prepared_run`, write the new
run's provenance summary, and create the resolved config. Keep prepare and
resume branches unchanged.

- [ ] **Step 4: Document the exact reuse command**

Document the existing prepared path, local model path, ERNIEKit checkout, new
work directory, and that the integrity scan checks paths without decoding or
copying images.

- [ ] **Step 5: Run the main-flow test and confirm it passes**

Run: `python -m unittest tests.test_finetune_vl.FinetuneVLTests.test_main_reuses_prepared_run_without_preparing_or_copying -v`

Expected: PASS.

### Task 3: Full verification and real prepared-run check

**Files:**
- Verify: `finetune_vl.py`
- Verify: `tests/test_finetune_vl.py`
- Verify: `docs/finetune-paddleocr-vl-1.6.md`

**Interfaces:**
- Consumes: completed CLI and loader.
- Produces: test, lint, compile, and real-data evidence.

- [ ] **Step 1: Run the full unit suite**

Run: `/tmp/paddleocr-vl-prepare-venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 2: Run static verification**

Run: `/tmp/paddleocr-vl-prepare-venv/bin/python -m ruff check finetune_vl.py tests/test_finetune_vl.py`

Run: `/tmp/paddleocr-vl-prepare-venv/bin/python -m ruff format --check finetune_vl.py tests/test_finetune_vl.py`

Run: `/tmp/paddleocr-vl-prepare-venv/bin/python -m py_compile finetune_vl.py`

Expected: all commands exit 0.

- [ ] **Step 3: Validate the real prepared run without training**

Run `load_prepared_run()` against
`runs/vl16_vi_all_datasets_prepare` using a temporary empty work directory,
then remove only that temporary directory.

Expected: 220,691 train and 4,504 validation records validate; no `prepared/`
directory is created in the temporary run and the source run is unchanged.

- [ ] **Step 4: Check the patch for whitespace errors**

Run: `git diff --check`

Expected: exit 0.

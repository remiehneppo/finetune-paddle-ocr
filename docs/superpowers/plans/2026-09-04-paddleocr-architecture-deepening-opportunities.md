# PaddleOCR Architecture Deepening Opportunities

**Date:** 2026-09-04
**Status:** Candidates 2, 4, and 5 implemented; Candidates 1, 3, and 6 partially implemented on 2026-09-04
**Scope:** Custom PaddleOCR-VL fine-tuning, prepared runs, OCR/VL labelers, exports, and post-validation

## Context

This review found no project `CONTEXT.md` or `docs/adr/` directory. Existing
design decisions are recorded in the `docs/superpowers/` plans/specs and in the
workspace memory. In particular:

- Prepared runs remain reusable, use explicit normalized relative weights, and
  must not stage or copy source data.
- The VL labeler owns PP-DocLayoutV3 sidecars, nullable layout-only tasks, and
  canonical OTSL/training output.
- Post-validation is a separate review path and must not alter HF or layout
  exports.
- Existing uncommitted canvas/UI changes are intentionally out of scope.

The terms below use the architecture vocabulary **Module**, **Interface**,
**Implementation**, **Depth**, **Seam**, **Adapter**, **Leverage**, and
**Locality**.

## Candidates

### 1. Unify the duplicated labeler shell

**Files:**

- `ocr_labeler/app.py:41`, `ocr_labeler/app.py:150`
- `vl_layout_labeler/app.py:96`, `vl_layout_labeler/app.py:118`
- `ocr_labeler/catalog.py:48`
- `vl_layout_labeler/catalog.py:48`

**Problem:** Workspace state, catalog scanning, source-image integrity,
streaming, lifecycle, and route error translation are duplicated. The current
**Interface** is a large collection of route-local assumptions.

**Solution:** Deepen a shared workspace/runtime **Module** that owns catalog
access, source-image integrity, annotation lookup, lifecycle, and common route
behavior. Keep OCR and VL behavior behind task-specific **Adapter**s at a
deliberate **Seam**.

**Benefits:** One **Implementation** provides more **Depth** and **Leverage**;
source-confinement and revision fixes gain **Locality**. Tests cover common
behavior once and focus package tests on task-specific differences.

### 2. Separate batch lifecycle from operation policy

**Files:**

- `ocr_labeler/batch.py:27`
- `vl_layout_labeler/batch.py:34`
- `vl_layout_labeler/app.py:370`

**Problem:** `BatchManager` combines threading, cancellation, progress,
persistence, skip rules, inference, error classification, and optional
validation. String operations such as `"detect"` and `"prelabel"` broaden the
**Interface** and make the **Implementation** timing-sensitive to test.

**Solution:** Deepen a batch-lifecycle **Module** that owns concurrency,
cancellation, progress, and terminal-state guarantees. Move detect, prelabel,
and validation policy behind operation **Adapter**s at the **Seam**.

**Benefits:** Lifecycle tests become deterministic and independent of model
behavior. **Locality** improves because concurrency bugs and operation-policy
bugs no longer share one `_run` implementation.

### 3. Make annotation export a separate deep Module

**Files:**

- `vl_layout_labeler/storage.py:153`
- `vl_layout_labeler/storage.py:388`
- `vl_layout_labeler/storage.py:526`
- `ocr_labeler/storage.py:174`

**Problem:** `AnnotationStore` owns durable persistence, source validation,
revision checks, HF/ERNIEKit output, COCO layout output, split selection,
temporary directories, and atomic export promotion. Its **Interface** leaks
format rules and output transaction behavior.

**Solution:** Keep `AnnotationStore` focused on durable annotation state and
source invariants. Add an export **Module** consuming a validated annotation
snapshot, with format-specific **Adapter**s for training and layout artifacts.
Keep atomic multi-export behavior behind that **Seam**.

**Benefits:** Multiple output **Adapter**s justify a real **Seam**. Export gains
**Depth** and **Leverage** while persistence tests and format tests become
independent, improving **Locality** when a training contract changes.

### 4. Centralize the PaddleOCR-VL target contract

**Files:**

- `paddleocr_vl_tasks.py:25`
- `vl_layout_labeler/task_map.py:36`
- `vl_layout_labeler/models.py:118`
- `vl_layout_labeler/storage.py:271`
- `finetune_vl.py:537`
- `finetune_vl.py:827`

**Problem:** Task mapping, prompt resolution, target validation, completed
annotation validation, and prepared JSONL validation all participate in one
compatibility contract but are distributed across several **Module**s.

**Solution:** Deepen a task-target contract **Module** owning task recognition,
prompt migration, target normalization, and validation. Let the labeler,
export, and preparation paths consume that contract instead of repeating
decisions.

**Benefits:** Strong **Locality** around OCR/table/formula/chart compatibility.
One **Interface** becomes the test surface for annotation completion, export,
and prepared-run admission.

### 5. Turn prepared-run loading and training planning into a deep Module

**Files:**

- `finetune_vl.py:827` (`_validate_prepared_jsonl`)
- `finetune_vl.py:934` (`read_prepared_run`)
- `finetune_vl.py:1003` (`aggregate_prepared_runs`)
- `finetune_vl.py:1082` (`load_prepared_runs`)
- `finetune_vl.py:1113` (`create_resolved_config`)
- `finetune_vl.py:2035` (`main`)

**Problem:** The script combines filesystem resolution, JSONL validation,
legacy migration, task union, weight normalization, provenance, subprocess
execution, checkpoint selection, evaluation, and export promotion. Helper
tests cover details, but the composition remains concentrated in `main()`.

**Solution:** Deepen a prepared-run planning **Module** that produces an
immutable training plan containing model identity, tasks, weights, paths,
provenance, and validation artifacts. Keep ERNIEKit execution, evaluation, and
export as separate **Adapter**s behind the plan’s **Seam**.

**Benefits:** Planning becomes testable without GPU, subprocesses, or model
downloads. Reuse, smoke, and resume workflows gain **Leverage**; failures gain
clearer classification and **Locality**.

### 6. Consolidate dataset admission across training workflows

**Files:**

- `finetune.py:234`, `finetune.py:537`
- `finetune_det.py:231`
- `finetune_vl.py:537`

**Problem:** Recognition, detection, and VL preparation duplicate image
opening, pixel checks, split handling, rejection accounting, and sample
construction. Their separate rejection and sample **Module**s can drift.

**Solution:** Deepen a dataset-admission **Module** for common image/path
safety and rejection accounting, with task-specific **Adapter**s for
recognition text, detection polygons, and VL targets. Preserve each output
format behind the **Seam**.

**Benefits:** Shared safety behavior gains **Depth** while task semantics stay
explicit. Common tests cover admission once; focused tests preserve each
training artifact contract and improve **Locality**.

## Suggested Order

1. Prepared-run planning (**Candidate 5**) — highest leverage for current VL
   training and the clearest GPU-independent test surface.
2. Target contract (**Candidate 4**) — protects compatibility before moving
   more policy behind new modules.
3. Annotation export (**Candidate 3**) — isolates persistence from training
   artifacts.
4. Batch lifecycle (**Candidate 2**) — reduces threaded workflow coupling.
5. Shared labeler shell (**Candidate 1**) — larger consolidation after the
   differing task contracts are explicit.
6. Dataset admission (**Candidate 6**) — broadest cross-workflow refactor;
   consider after the VL contracts stabilize.

## Follow-up

Candidate 5 was selected and implemented first. `prepared_run_planning.py`
now owns the GPU-independent orchestration and immutable
`PreparedRunPlan`; `finetune_vl.py` supplies the existing validation and prompt
callbacks and retains its public compatibility wrappers. The implementation
preserves in-place source references, normalized weights, task union, model
compatibility checks, legacy single-run metadata, and atomic summary writing.

The remaining candidates still require a separate design pass before moving
interfaces or files.

Candidate 4 is implemented in `paddleocr_vl_contract.py`, with
`paddleocr_vl_tasks.py` and `vl_layout_labeler/task_map.py` retained as
compatibility facades. Candidate 3 remains partial: the public orchestration
and atomic `export_all` seam live in `vl_layout_labeler/export.py`, but the
format-specific implementations still depend on private `AnnotationStore`
helpers. It is intentionally not described as a fully independent export
module until those helpers consume a validated snapshot interface.

Candidate 2 is implemented with the shared lifecycle mixin in
`batch_lifecycle.py` used by both labeler batch managers. Operation-specific
detect, prelabel, validation, and OCR policy remain local adapters by design.
Candidate 1 centralizes catalog and safe source streaming in
`labeler_catalog.py` and `labeler_streaming.py`, while workspace state and
task-specific routes remain separate. Candidate 6 centralizes image admission
and rejection accounting in `dataset_admission.py`; task-specific sample
construction and summary field names remain separate. Candidates 1, 3, and 6
are deliberately partial implementations, not claims that all labeler or
training duplication is gone.

## Verification boundary

- Focused runtime regression after the 2026-09-04 changes: `49 passed`; the
  environment emits one existing Starlette/httpx deprecation warning.
- The local Llama endpoint was smoke-tested through `/v1/models` and the VL
  client path. A generated answer is endpoint evidence only, not an OCR quality
  gate, GPU-training result, browser QA result, or live-resume guarantee.
- The `.venv-vl-eval` environment still lacks `datasets`, while `.venv` lacks
  `pytest`; use the documented combined `PYTHONPATH` test command when needed.

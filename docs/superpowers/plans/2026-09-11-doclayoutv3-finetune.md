# PP-DocLayoutV3 Fine-tuning Implementation Plan

**Date:** 2026-09-11
**Status:** In Progress (Entrypoint, configs, tests and docs implemented; GPU pilot pending)
**Scope:** Train PP-DocLayoutV3 from the COCO layout branch exported by
`vl_layout_labeler` on the local RTX 5060 Ti 16 GB workstation

## Goal

Add a repository-owned, repeatable PP-DocLayoutV3 fine-tuning path that:

- accepts `<export-root>/layout` without converting or copying source images;
- fails before training when the PaddleX/PaddleDetection backend, dataset, GPU,
  label taxonomy, or output path is invalid;
- preserves all 25 PP-DocLayoutV3 classes and per-page continuous
  `read_order` values;
- uses one RTX 5060 Ti through `gpu:0` and only options supported by the
  installed PP-DocLayoutV3 backend;
- records the resolved configuration, environment, dataset identity, metrics,
  and produced artifacts;
- selects batch size and loader workers from measured runs rather than an
  undocumented machine-wide default.

## Current State

The labeler output already matches the PaddleX `COCOInstSegDataset` contract:

```text
layout/
├── images/
├── annotations/
│   ├── instance_train.json
│   └── instance_val.json
└── export_manifest.json
```

Each annotation has `bbox`, polygon `segmentation`, `area`, `iscrowd`, and a
zero-based continuous `read_order` within its page. Both splits contain the
same ordered 25-category taxonomy.

Fine-tuning is currently documented as a direct PaddleX `Engine` invocation.
There is no repository-owned DocLayoutV3 entrypoint or config. The installed
environment contains PaddleX 3.7.2, PaddlePaddle 3.2.1, PaddleDetection, and
`pycocotools`, but PaddleX does not register the PaddleDetection backend because
its managed checkout lacks the `.installed` marker. The documented command
therefore fails with:

```text
UnsupportedParamError: 'PP-DocLayoutV3' is not a registered model name.
```

The module config also defaults to four GPUs, while the target machine has one
RTX 5060 Ti 16 GB. Its backend defaults include batch size 4, 24 loader
workers, EMA, variable resize up to 928 pixels, and AMP disabled.

## Decisions

1. Add `finetune_doclayout_v3.py` as a thin orchestration layer over PaddleX.
   Do not reuse `finetune_vl.py`: layout detection and PaddleOCR-VL SFT have
   different datasets, backends, checkpoints, and output contracts.
2. Keep a repository-owned backend YAML under `configs/doclayoutv3/`. Record
   the upstream PaddleX version and source config path in its header. Do not
   patch files inside `.venv`.
3. Default to `gpu:0`, batch size 1, and 4 loader workers for the first smoke
   run. These are safe starting values, not the final tuned values.
4. Keep `amp=OFF` and `dy2st=False`. PaddleX 3.7.2 declares no other supported
   values for PP-DocLayoutV3.
5. Keep all 25 classes. Compare the ordered names in both COCO files against
   `PP_DOCLAYOUTV3_LABELS`; checking only the class count is insufficient.
6. Never overwrite an existing work directory. Resume requires an explicit
   checkpoint and must preserve the original run configuration and dataset
   identity.
7. Store training output under the user-provided work directory. Do not mutate
   the base/pretrained weight artifact.
8. Run gates in this order: static validation, PaddleX dataset check, backend
   construction, GPU smoke, pilot, then full training.

## Proposed Interface

The exact option names may be adjusted during implementation. This is the
initial smoke/pilot invocation; epochs and learning rate must be selected from
pilot evidence before the full run:

```bash
.venv/bin/python finetune_doclayout_v3.py \
  --dataset-dir /path/to/export/layout \
  --work-dir runs/doclayoutv3 \
  --device gpu:0 \
  --batch-size 1 \
  --num-workers 4 \
  --epochs 100 \
  --learning-rate 0.0001
```

Additional modes:

```bash
# Validate contracts and dump the resolved config without training.
.venv/bin/python finetune_doclayout_v3.py ... --check-only

# Run the bounded GPU smoke profile defined by the repository config.
.venv/bin/python finetune_doclayout_v3.py ... --smoke

# Resume only from an explicit PaddleDetection checkpoint.
.venv/bin/python finetune_doclayout_v3.py ... \
  --resume-from /path/to/checkpoint.pdparams
```

The script must reject incompatible combinations such as `--check-only` with
`--resume-from`, non-positive numeric values, non-single-GPU device strings for
this workstation profile, and output reuse without a valid resume contract.

## Implementation Steps

### 1. Repair and verify the training backend

- Install/register PaddleDetection through the PaddleX plugin command using
  the existing local checkout when possible.
- Confirm that normal PaddleX initialization registers `PP-DocLayoutV3`; do
  not rely on a manual import in the final entrypoint.
- Confirm the runner root points to the managed PaddleDetection checkout.
- Record PaddleX, PaddlePaddle, PaddleDetection, CUDA, cuDNN, Python, GPU, and
  driver versions in a machine-readable preflight artifact.
- Verify that the pretrained weight is available before the GPU run. If a
  download is required, make it an explicit preparation step and record its
  path and checksum.

### 2. Add the repository-owned configuration

- Copy the PaddleX 3.7.2 PP-DocLayoutV3 backend config into
  `configs/doclayoutv3/PP-DocLayoutV3-rtx5060ti.yaml`.
- Preserve the architecture, 25-class head, loss coefficients, variable-size
  augmentation, EMA, AdamW, gradient clipping, and evaluation behavior unless
  a measured pilot justifies a change.
- Set the initial profile to one GPU, batch size 1, 4 workers, and a unique
  output directory supplied by the entrypoint.
- Keep evaluation batch size 1.
- Make `drop_last` explicit and decide it from dataset size. Reject a run that
  would produce zero train batches; prefer `drop_last: false` for small custom
  datasets if the backend handles the partial batch correctly.
- Replace the ineffective `gamma: 1.0` schedule only after a pilot compares it
  with a real decay schedule. Record the chosen schedule in the resolved YAML.

### 3. Implement `finetune_doclayout_v3.py`

- Parse and validate all CLI options before importing the heavy training
  backend.
- Resolve dataset, work, config, pretrained-weight, and resume paths.
- Require `images/`, both COCO annotation files, and
  `export_manifest.json`.
- Validate JSON syntax, unique image/annotation IDs, referenced image files,
  positive image dimensions, valid XYWH boxes, valid polygons and areas,
  category IDs, and continuous per-image `read_order`.
- Require the exact ordered 25-label taxonomy in both splits.
- Reject train/validation page leakage by image ID and file name.
- Invoke the official PaddleX dataset checker and persist its report.
- Build the trainer before creating a long-running job so registry and runner
  failures are reported during preflight.
- Apply `Global.device=gpu:0`, the dataset directory, work directory, class
  count, batch size, learning rate, epoch count, warmup, checkpoint, and the
  repository backend config through supported PaddleX configuration fields.
- Dump the final PaddleDetection config and a run summary before launching the
  subprocess.
- Propagate non-zero subprocess exit codes and preserve stdout/stderr logs.
- Update the summary atomically to `passed`, `failed`, or `blocked` with the
  relevant artifact paths and error reason.

### 4. Add deterministic tests

- Unit-test CLI defaults, invalid numeric boundaries, incompatible options,
  device parsing, new-work-dir enforcement, and resume admission.
- Generate a two-page COCO fixture covering all 25 classes and `read_order`.
- Test rejection of missing files, unknown/reordered categories, bad category
  IDs, degenerate polygons, invalid boxes, duplicate IDs, missing image files,
  discontinuous `read_order`, and split leakage.
- Test that check-only mode builds and writes the resolved config without
  starting PaddleDetection training.
- Mock the subprocess boundary to verify the exact runner arguments, exit-code
  propagation, and terminal summary status.
- Add an integration test using the installed PaddleX checker when the optional
  backend dependencies are present; report an explicit skip otherwise.

### 5. Run a GPU smoke test

- Use a deliberately small, versioned fixture or bounded dataset view that
  references source images without copying them.
- Run the smallest backend-supported training unit with batch size 1 and 4
  workers.
- Require at least one completed optimizer step, finite loss, a written
  checkpoint/config/log, and clean process exit.
- Capture peak GPU memory, median steady-state step time, CPU utilization, data
  wait time when available, and the exact configuration.
- Treat a successful smoke run as execution evidence only, not model-quality
  evidence.

### 6. Tune batch size and loader workers

Hold the dataset view, seed, augmentation, learning rate, and measurement
window fixed while changing one variable at a time.

1. Test batch sizes 1, 2, and 4. Stop increasing after OOM or when peak VRAM
   leaves insufficient safety margin for evaluation/checkpointing.
2. With the selected batch size, test 2, 4, 6, and 8 workers.
3. Select the fastest stable configuration with no OOM, worker crash, data
   starvation, or excessive host memory use.
4. If changing effective batch size, explicitly decide whether learning rate
   should scale; do not scale it automatically.

Write results to `benchmark.json` with samples/second, median step time, peak
VRAM, worker count, batch size, and pass/fail reason for every candidate.

### 7. Run a pilot and select training policy

- Run a short pilot on the real train/validation split with the selected
  resource profile.
- Inspect training loss, validation metric, qualitative predictions, rare-class
  behavior, and read-order output.
- Choose epochs, evaluation interval, warmup, learning-rate schedule, and early
  stopping policy from pilot evidence.
- Record per-class sample counts. Flag classes with no training samples or no
  validation samples before claiming a meaningful 25-class evaluation.
- Freeze the selected full-run config and its dataset fingerprint.

### 8. Document and hand off

- Replace the raw Engine commands in README and the VL fine-tuning guide with
  the repository entrypoint.
- Document backend setup, check-only, smoke, full train, resume, evaluation,
  export, artifact layout, and recovery from an interrupted run.
- Include the verified PaddleX/PaddleDetection versions and the supported AMP
  limitation.
- Keep the VL SFT instructions separate from DocLayoutV3 layout training.

## Expected Artifacts

```text
<work-dir>/
├── preflight.json
├── dataset-check/
├── resolved-config.yaml
├── summary.json
├── train.log
├── benchmark.json              # benchmark runs only
└── output/
    ├── best_model/
    └── ... PaddleDetection checkpoints
```

`summary.json` should include the command, timestamps, status, environment,
dataset path and fingerprint, split counts, per-class counts, config checksum,
pretrained-weight checksum, resume source, selected device, batch size, worker
count, key metrics, and artifact paths.

## Acceptance Criteria

- A clean environment can follow the documented setup and register
  `PP-DocLayoutV3` through normal PaddleX initialization.
- `--check-only` accepts a valid labeler layout export and rejects every tested
  contract violation before GPU work begins.
- The resolved config uses one GPU, exactly 25 classes, the expected dataset
  paths, and repository-owned backend settings.
- The GPU smoke run completes at least one optimizer step with finite loss and
  produces the required artifacts.
- The selected batch/worker values have benchmark evidence on the RTX 5060 Ti
  16 GB machine.
- The pilot produces validation metrics and qualitative predictions; class
  coverage limitations are reported explicitly.
- Resume continues the same dataset/config lineage and does not overwrite an
  unrelated run.
- Unit and integration tests pass, and README commands match the implemented
  CLI help.
- No base model, exported labeler dataset, or source image is modified.

## Implementation Checklist

### Backend readiness

- [ ] Register/install the PaddleDetection plugin through PaddleX.
- [ ] Verify the managed checkout has a valid installation marker.
- [ ] Verify normal PaddleX initialization registers `PP-DocLayoutV3`.
- [ ] Verify the runner root resolves to the expected PaddleDetection checkout.
- [ ] Record and pin the verified runtime versions.
- [ ] Resolve and checksum the pretrained weight.

### Configuration

- [x] Add `configs/doclayoutv3/PP-DocLayoutV3-rtx5060ti.yaml`.
- [x] Record its PaddleX 3.7.2 upstream source and intentional differences.
- [x] Set the initial one-GPU profile.
- [x] Set initial batch size 1 and worker count 4.
- [x] Keep AMP off and dynamic-to-static off.
- [x] Preserve the 25-class architecture and read-order loss.
- [x] Decide and test `drop_last` behavior for small datasets.
- [x] Keep evaluation batch size 1.

### Entrypoint and validation

- [x] Add `finetune_doclayout_v3.py`.
- [x] Implement strict CLI and path validation.
- [x] Validate both COCO files and exact ordered taxonomy.
- [x] Validate geometry, IDs, image references, splits, and `read_order`.
- [x] Run and capture the official PaddleX dataset checker.
- [x] Validate registry, runner, GPU, and free output path before training.
- [x] Write preflight and resolved config artifacts.
- [x] Implement atomic terminal run summaries.
- [x] Implement explicit, lineage-safe resume.

### Verification

- [x] Add unit tests for CLI, dataset admission, and run planning.
- [x] Add malformed-dataset regression fixtures.
- [x] Add mocked subprocess and failure-propagation tests.
- [ ] Add the optional PaddleX checker integration test.
- [x] Run Python syntax checks.
- [x] Run focused DocLayoutV3 tests.
- [ ] Run the broader relevant regression suite.
- [ ] Run the bounded GPU smoke test.
- [ ] Confirm finite loss, optimizer progress, and output artifacts.

### Optimization and pilot

- [ ] Benchmark batch sizes 1, 2, and 4.
- [ ] Benchmark worker counts 2, 4, 6, and 8.
- [ ] Record throughput, latency, peak VRAM, CPU load, and failures.
- [ ] Select and freeze the resource profile from measurements.
- [ ] Run a pilot on the real dataset.
- [ ] Review class coverage and qualitative predictions.
- [ ] Select epochs, warmup, evaluation interval, and LR schedule.
- [ ] Freeze the full-run config and dataset fingerprint.

### Documentation and delivery

- [x] Update README to use the new entrypoint.
- [x] Update the full VL/layout fine-tuning guide.
- [x] Document setup, check, smoke, train, resume, evaluate, and export.
- [x] Document artifact locations and recovery steps.
- [x] Confirm documented commands match `--help` output.
- [ ] Review the final diff for accidental `.venv`, dataset, model, or run
      artifacts.
- [ ] Commit and push only after implementation and all applicable gates pass.

## Out of Scope

- Changing the VL labeler sidecar or export schema.
- Fine-tuning PaddleOCR-VL content tasks.
- Enabling unsupported AMP or dynamic-to-static modes.
- Multi-GPU or multi-node tuning.
- Claiming production model quality from checker or smoke-test success alone.

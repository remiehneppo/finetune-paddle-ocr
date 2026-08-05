# Reuse Prepared Dataset Design

## Goal

Add `--prepared-from RUN_DIR` to `finetune_vl.py` so a new LoRA run can reuse
an existing prepared dataset without loading the Hugging Face sources,
re-encoding images, or copying the prepared 5.6 GB payload.

## CLI contract

- A fresh run accepts exactly one of `--dataset-dir ...` and
  `--prepared-from RUN_DIR`.
- `--prepared-from` is incompatible with `--prepare-only`: the referenced run
  is already prepared.
- `--prepared-from` is incompatible with `--resume-from`: resume uses the
  summary and config belonging to its existing work directory.
- A reused training run still requires a new, empty `--work-dir`, a local
  `--model`, and a compatible `--erniekit-dir`.

## Loading and validation

The referenced run remains read-only. The loader reads `summary.json`, checks
that the prompt is exactly `OCR:`, checks the source/count/probability shape,
then scans every train and validation JSONL row. Each row must be valid JSON,
must follow the ERNIEKit OCR mask contract, and its `image_url` must resolve to
an existing regular file. Images are not decoded again.

Paths already stored as absolute paths remain unchanged. Relative JSONL paths
in `summary.json` are resolved against the referenced run. Relative image paths
are resolved against the JSONL's directory, matching ERNIEKit dataset loading.

## New-run artifacts

The new run writes its own `summary.json` and `resolved.yaml`, but no
`prepared/` directory. Its summary keeps the validated source metadata and
adds `prepared_from` with the canonical source-run path. The resolved config
points directly to the old JSONLs. The old run and its files are never
modified.

## Failure behavior

Validation fails before model inspection or training when the source summary,
JSONL, sample counts, mask contract, or referenced images are invalid. The new
work directory may already have been created, but it contains no copied
dataset. Existing non-empty run directories remain protected from overwrite.

#!/usr/bin/env python3
"""Deterministically compare PaddleOCR-VL base and merged models."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from finetune_vl import compute_ocr_metrics

PROMPT = "OCR:"
CANDIDATES = ("base", "merged")


def ocr_messages(image: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def deterministic_generation_kwargs(max_new_tokens: int) -> dict[str, object]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }


def decode_new_tokens(processor, generated_ids, prompt_token_count: int) -> str:
    if prompt_token_count < 0:
        raise ValueError("prompt_token_count must be non-negative")
    new_token_ids = generated_ids[:, prompt_token_count:]
    decoded = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def validate_candidate_coverage(
    predictions: Sequence[Mapping[str, str]],
    candidates: Sequence[str],
    fixture_count: int,
) -> None:
    expected_candidates = tuple(candidates)
    if fixture_count <= 0 or not expected_candidates:
        raise ValueError("Candidate coverage requires positive fixtures and candidates")
    if len(set(expected_candidates)) != len(expected_candidates):
        raise ValueError("Candidate coverage contains duplicate candidate names")

    fixtures_by_candidate: dict[str, set[tuple[str, str]]] = {
        candidate: set() for candidate in expected_candidates
    }
    counts = {candidate: 0 for candidate in expected_candidates}
    for prediction in predictions:
        candidate = prediction.get("candidate")
        if candidate not in fixtures_by_candidate:
            raise ValueError(f"Candidate coverage contains unexpected candidate: {candidate}")
        fixture = (prediction["dataset"], prediction["image"])
        counts[candidate] += 1
        if fixture in fixtures_by_candidate[candidate]:
            raise ValueError(f"Candidate coverage contains duplicate fixture: {candidate}")
        fixtures_by_candidate[candidate].add(fixture)

    expected_fixtures: set[tuple[str, str]] | None = None
    for candidate in expected_candidates:
        fixtures = fixtures_by_candidate[candidate]
        if counts[candidate] != fixture_count or len(fixtures) != fixture_count:
            raise ValueError(f"Candidate coverage is incomplete for {candidate}")
        if expected_fixtures is None:
            expected_fixtures = fixtures
        elif fixtures != expected_fixtures:
            raise ValueError("Candidate coverage does not match across candidates")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def load_validation_rows(
    paths: Sequence[Path], samples_per_dataset: int
) -> list[dict[str, str]]:
    if samples_per_dataset <= 0:
        raise ValueError("samples_per_dataset must be positive")
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Validation JSONL not found: {path}")
        dataset = path.stem.removeprefix("validation-")
        selected = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                try:
                    prompt, target = payload["text_info"]
                    image = payload["image_info"][0]
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError(f"Invalid OCR row {path}:{line_number}") from exc
                if prompt != {"text": PROMPT, "tag": "mask"}:
                    raise ValueError(f"Invalid OCR prompt {path}:{line_number}")
                if (
                    not isinstance(target, dict)
                    or target.get("tag") != "no_mask"
                    or not isinstance(target.get("text"), str)
                ):
                    raise ValueError(f"Invalid OCR target {path}:{line_number}")
                if image.get("matched_text_index") != 0:
                    raise ValueError(f"Invalid OCR image match {path}:{line_number}")
                image_path = Path(image["image_url"])
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Validation image not found {path}:{line_number}: {image_path}"
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "image": str(image_path.resolve()),
                        "target": target["text"],
                    }
                )
                selected += 1
                if selected >= samples_per_dataset:
                    break
    if not rows:
        raise ValueError("No validation rows were loaded")
    return rows


def _load_rgb_image(image_path: str):
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def seed_runtime(torch_module: Any) -> None:
    manual_seed = getattr(torch_module, "manual_seed", None)
    if manual_seed is not None:
        manual_seed(2026)
    cuda_manual_seed_all = getattr(torch_module.cuda, "manual_seed_all", None)
    if cuda_manual_seed_all is not None:
        cuda_manual_seed_all(2026)


def configure_deterministic_runtime(torch_module: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_runtime(torch_module)
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.enable_flash_sdp(False)
    torch_module.backends.cuda.enable_mem_efficient_sdp(False)
    torch_module.backends.cuda.enable_math_sdp(True)


def install_masking_utils_compatibility(masking_utils_module: Any) -> None:
    create_causal_mask = masking_utils_module.create_causal_mask
    if getattr(create_causal_mask, "_paddleocr_vl_inputs_embeds_alias", False):
        return
    parameters = inspect.signature(create_causal_mask).parameters
    if "inputs_embeds" in parameters or "input_embeds" not in parameters:
        return

    def compatible_create_causal_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            if "input_embeds" in kwargs:
                raise TypeError("Pass only one of input_embeds or inputs_embeds")
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return create_causal_mask(*args, **kwargs)

    compatible_create_causal_mask._paddleocr_vl_inputs_embeds_alias = True
    masking_utils_module.create_causal_mask = compatible_create_causal_mask


def _runtime_dependencies():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, masking_utils

        configure_deterministic_runtime(torch)
        install_masking_utils_compatibility(masking_utils)
    except ImportError as exc:
        raise RuntimeError(
            "Missing native evaluation dependencies. Install requirements-vl-eval.txt."
        ) from exc
    return torch, AutoProcessor, AutoModelForCausalLM


def _move_inputs_to_cuda(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.to("cuda") if hasattr(value, "to") else value
        for name, value in inputs.items()
    }


def _synchronize_cuda(torch_module: Any) -> None:
    synchronize = getattr(torch_module.cuda, "synchronize", None)
    if synchronize is not None:
        synchronize()


def evaluate(
    args: argparse.Namespace,
    *,
    torch_module: Any | None = None,
    auto_processor_class: Any | None = None,
    auto_model_class: Any | None = None,
    image_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    rows = load_validation_rows(args.validation_jsonl, args.samples_per_dataset)
    generation_kwargs = deterministic_generation_kwargs(args.max_new_tokens)
    if (
        torch_module is None
        or auto_processor_class is None
        or auto_model_class is None
    ):
        torch_module, auto_processor_class, auto_model_class = _runtime_dependencies()
    if not torch_module.cuda.is_available():
        raise RuntimeError("Native PaddleOCR-VL evaluation requires a CUDA GPU")
    image_loader = image_loader or _load_rgb_image

    model_paths = {
        "base": args.base_model.expanduser().resolve(),
        "merged": args.merged_model.expanduser().resolve(),
    }
    for candidate, model_path in model_paths.items():
        if not model_path.is_dir():
            raise FileNotFoundError(f"{candidate} model directory not found: {model_path}")

    predictions: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}

    for candidate in CANDIDATES:
        model_path = model_paths[candidate]
        processor = auto_processor_class.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=False
        )
        seed_runtime(torch_module)
        model = auto_model_class.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch_module.bfloat16,
            attn_implementation="eager",
        ).to("cuda").eval()
        try:
            candidate_rows: list[dict[str, Any]] = []
            for row in rows:
                rendered = processor.apply_chat_template(
                    ocr_messages(row["image"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[rendered],
                    images=[image_loader(row["image"])],
                    return_tensors="pt",
                )
                model_inputs = _move_inputs_to_cuda(inputs)
                prompt_token_count = int(model_inputs["input_ids"].shape[-1])
                _synchronize_cuda(torch_module)
                started = time.perf_counter()
                with torch_module.inference_mode():
                    generated_ids = model.generate(
                        **model_inputs,
                        **generation_kwargs,
                    )
                _synchronize_cuda(torch_module)
                prediction = decode_new_tokens(
                    processor, generated_ids, prompt_token_count
                )
                record = {
                    "candidate": candidate,
                    "dataset": row["dataset"],
                    "image": row["image"],
                    "target": row["target"],
                    "prediction": prediction,
                    "runtime_seconds": time.perf_counter() - started,
                }
                predictions.append(record)
                candidate_rows.append(record)
            reports[candidate] = compute_ocr_metrics(candidate_rows)
        finally:
            del model, processor
            gc.collect()
            torch_module.cuda.empty_cache()

    validate_candidate_coverage(predictions, CANDIDATES, len(rows))
    report = {
        "status": "passed",
        "prompt": PROMPT,
        "fixture_count": len(rows),
        "samples_per_dataset": args.samples_per_dataset,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {
            "deterministic": True,
            **generation_kwargs,
        },
        "candidates": reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ocr_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "ocr_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = evaluate(parse_args(argv))
    print(
        "PADDLEOCR_VL_EVALUATION="
        + json.dumps(
            {
                "status": report["status"],
                "fixture_count": report["fixture_count"],
                "candidates": list(report["candidates"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

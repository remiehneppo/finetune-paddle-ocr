"""Merge an ERNIEKit PaddleOCR-VL LoRA adapter into Hugging Face weights."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np

MAX_LOGIT_ABS_ERROR = 1.0
MEAN_LOGIT_ABS_ERROR = 0.2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-jsonl", type=Path, required=True)
    parser.add_argument("--min-pixels", type=int, required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    return parser.parse_args(argv)


def validate_inputs(base_model: Path, adapter_dir: Path, output_dir: Path) -> None:
    config_path = base_model / "config.json"
    lora_config_path = adapter_dir / "lora_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Base model config not found: {config_path}")
    if not lora_config_path.is_file():
        raise FileNotFoundError(f"LoRA config not found: {lora_config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "paddleocr_vl":
        raise ValueError("Compatibility merge only supports model_type='paddleocr_vl'")
    if output_dir.exists() and any(output_dir.glob("*.safetensors")):
        raise FileExistsError(f"Merged model output already exists: {output_dir}")


def expected_merged_weight(
    base: np.ndarray, lora_a: np.ndarray, lora_b: np.ndarray, scaling: float
) -> np.ndarray:
    """Apply Paddle-layout LoRA matrices to a Hugging Face-layout weight."""
    return base + (lora_a @ lora_b * scaling).T


def expected_serialized_merged_weight(
    base: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    scaling: float,
    output_dtype: np.dtype,
) -> np.ndarray:
    """Reproduce Paddle low-precision LoRA merge and serialization rounding."""
    delta = (lora_a.astype(np.float32) @ lora_b.astype(np.float32) * scaling).astype(
        output_dtype
    )
    return (base.astype(np.float32) + delta.astype(np.float32).T).astype(output_dtype)


def compare_logits(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    delta = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return {
        "max_abs_error": float(np.max(delta)),
        "mean_abs_error": float(np.mean(delta)),
        "argmax_equal": bool(
            np.array_equal(left.argmax(axis=-1), right.argmax(axis=-1))
        ),
    }


def validate_logits_comparison(
    comparison: dict[str, object],
    *,
    max_abs_error: float,
    mean_abs_error: float,
) -> None:
    if (
        float(comparison["max_abs_error"]) > max_abs_error
        or float(comparison["mean_abs_error"]) > mean_abs_error
    ):
        raise RuntimeError(
            "Merged model logits exceed tolerance: "
            f"max={comparison['max_abs_error']} (limit {max_abs_error}), "
            f"mean={comparison['mean_abs_error']} (limit {mean_abs_error})"
        )


def _weight_map(
    directory: Path, index_name: str, single_patterns: Sequence[str]
) -> dict[str, str]:
    index_path = directory / index_name
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return dict(payload["weight_map"])
    matches = [
        path for pattern in single_patterns for path in sorted(directory.glob(pattern))
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot resolve safetensors in: {directory}")
    from safetensors import safe_open

    with safe_open(matches[0], framework="np") as handle:
        return {
            name: matches[0].name
            for name in handle.keys()  # noqa: SIM118
        }


def verify_merged_weights(
    base_model: Path, adapter_dir: Path, output_dir: Path
) -> dict[str, object]:
    """Prove merged tensors equal base plus LoRA, with unchanged tensors exact."""
    import ml_dtypes  # noqa: F401  # registers NumPy's bfloat16 dtype
    from safetensors import safe_open

    base_map = _weight_map(
        base_model, "model.safetensors.index.json", ("model.safetensors",)
    )
    adapter_map = _weight_map(
        adapter_dir,
        "peft_model.safetensors.index.json",
        ("peft_model*.safetensors",),
    )
    merged_map = _weight_map(
        output_dir,
        "model.safetensors.index.json",
        ("model-*.safetensors", "model.safetensors"),
    )
    if set(base_map) != set(merged_map):
        raise RuntimeError("Merged model tensor names do not match the base model")

    lora_a_names = sorted(name for name in adapter_map if name.endswith(".lora_A"))
    adapted_keys = {name.removesuffix(".lora_A") + ".weight" for name in lora_a_names}
    lora_config = json.loads(
        (adapter_dir / "lora_config.json").read_text(encoding="utf-8")
    )
    scaling = float(
        lora_config.get("scaling", lora_config["lora_alpha"] / lora_config["r"])
    )
    max_abs_error = 0.0
    with ExitStack() as stack:
        handles: dict[Path, object] = {}

        def tensor(directory: Path, mapping: dict[str, str], name: str) -> np.ndarray:
            path = directory / mapping[name]
            if path not in handles:
                handles[path] = stack.enter_context(safe_open(path, framework="np"))
            return handles[path].get_tensor(name)  # type: ignore[union-attr]

        for key in sorted(base_map):
            base = tensor(base_model, base_map, key)
            actual = tensor(output_dir, merged_map, key)
            if key in adapted_keys:
                prefix = key.removesuffix(".weight")
                lora_a = tensor(adapter_dir, adapter_map, prefix + ".lora_A")
                lora_b = tensor(adapter_dir, adapter_map, prefix + ".lora_B")
                expected = expected_serialized_merged_weight(
                    base, lora_a, lora_b, scaling, actual.dtype
                )
            else:
                expected = base
            error = float(
                np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
            )
            matches = (
                np.allclose(
                    actual.astype(np.float32),
                    expected.astype(np.float32),
                    rtol=5e-4,
                    atol=5e-4,
                )
                if key in adapted_keys
                else np.array_equal(actual, expected)
            )
            if not matches:
                raise RuntimeError(
                    f"Merged weight verification failed for {key}; max error={error}"
                )
            if key in adapted_keys:
                max_abs_error = max(max_abs_error, error)

    report: dict[str, object] = {
        "status": "passed",
        "base_tensor_count": len(base_map),
        "adapted_tensor_count": len(adapted_keys),
        "unchanged_tensor_count": len(base_map) - len(adapted_keys),
        "scaling": scaling,
        "max_abs_error_after_output_cast": max_abs_error,
        "adapted_tensor_rtol": 5e-4,
        "adapted_tensor_atol": 5e-4,
    }
    (output_dir / "merge_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def merge(
    base_model: Path,
    adapter_dir: Path,
    output_dir: Path,
    fixture_jsonl: Path,
    min_pixels: int,
    max_pixels: int,
) -> None:
    import gc

    import paddle
    from data_processor.image_preprocessor.image_preprocessor_siglip import (
        SiglipImageProcessor,
    )
    from ernie.configuration_paddleocr_vl import PaddleOCRVLConfig
    from ernie.modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration
    from ernie.tokenizer import Ernie4_5_Tokenizer
    from paddleformers.peft import LoRAModel
    from paddleformers.transformers.image_utils import ChannelDimension
    from PIL import Image

    validate_inputs(base_model, adapter_dir, output_dir)
    if not fixture_jsonl.is_file():
        raise FileNotFoundError(f"OCR fixture JSONL not found: {fixture_jsonl}")
    paddle.set_device("gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu")
    paddle.set_default_dtype("bfloat16")
    config = PaddleOCRVLConfig.from_pretrained(str(base_model.resolve()))
    config.dtype = "bfloat16"
    config.tensor_parallel_degree = 1
    config.tensor_parallel_rank = 0
    config.sequence_parallel = False
    config.use_cache = False
    config.use_flash_attention = False
    config.vision_config.tensor_parallel_degree = 1
    config.vision_config.tensor_parallel_rank = 0
    config.vision_config.use_flash_attention = False
    model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        str(base_model.resolve()),
        config=config,
        convert_from_hf=True,
    )
    wrapped = LoRAModel.from_pretrained(
        model=model, lora_path=str(adapter_dir.resolve())
    )
    tokenizer = Ernie4_5_Tokenizer.from_pretrained(
        str(base_model.resolve()), model_max_length=2048
    )
    fixture_row = json.loads(
        next(
            line
            for line in fixture_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    if fixture_row.get("text_info", [{}])[0] != {"text": "OCR:", "tag": "mask"}:
        raise ValueError("Merge fixture must use exactly the masked OCR: prompt")
    fixture_image = fixture_jsonl.parent / fixture_row["image_info"][0]["image_url"]
    image_processor = SiglipImageProcessor.from_pretrained(str(base_model.resolve()))
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    with Image.open(fixture_image) as image:
        image_inputs = image_processor.preprocess(
            image.convert("RGB"),
            return_tensors="pd",
            input_data_format=ChannelDimension.LAST,
        )
    image_grid_thw = image_inputs["image_grid_thw"]
    image_token_count = (
        int(np.prod(image_grid_thw.numpy()[0])) // image_processor.merge_size**2
    )
    prompt_ids = tokenizer.encode(
        "OCR:", add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    fixture_inputs = {
        "input_ids": paddle.to_tensor(
            [[config.image_token_id] * image_token_count + prompt_ids], dtype="int64"
        ),
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_grid_thw,
    }
    fixture_inputs["attention_mask"] = paddle.ones_like(fixture_inputs["input_ids"])

    def fixture_logits(fixture_model) -> np.ndarray:
        fixture_model.eval()
        with paddle.no_grad():
            return (
                fixture_model(return_dict=True, **fixture_inputs)
                .logits[:, -1, :]
                .astype("float32")
                .cpu()
                .numpy()
            )

    adapter_logits = fixture_logits(wrapped)
    wrapped.merge()
    in_memory_merged_logits = fixture_logits(wrapped)
    merged_state = {
        name: tensor
        for name, tensor in wrapped.model.state_dict().items()
        if ".lora_" not in name
    }
    wrapped.model.save_pretrained(
        str(output_dir.resolve()),
        state_dict=merged_state,
        safe_serialization=True,
        save_to_hf=True,
        max_shard_size="5GB",
    )
    report = verify_merged_weights(base_model, adapter_dir, output_dir)
    del merged_state, wrapped, model
    gc.collect()
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()

    exported_config = PaddleOCRVLConfig.from_pretrained(str(output_dir.resolve()))
    exported_config.dtype = "bfloat16"
    exported_config.tensor_parallel_degree = 1
    exported_config.tensor_parallel_rank = 0
    exported_config.sequence_parallel = False
    exported_config.use_cache = False
    exported_config.use_flash_attention = False
    exported_config.vision_config.tensor_parallel_degree = 1
    exported_config.vision_config.tensor_parallel_rank = 0
    exported_config.vision_config.use_flash_attention = False
    exported_model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        str(output_dir.resolve()), config=exported_config, convert_from_hf=True
    )
    exported_logits = fixture_logits(exported_model)
    logits_report: dict[str, object] = {
        "status": "passed",
        "fixture": str(fixture_image.resolve()),
        "prompt": "OCR:",
        "max_abs_error_tolerance": MAX_LOGIT_ABS_ERROR,
        "mean_abs_error_tolerance": MEAN_LOGIT_ABS_ERROR,
        "adapter_vs_in_memory_merge": compare_logits(
            adapter_logits, in_memory_merged_logits
        ),
        "in_memory_merge_vs_export": compare_logits(
            in_memory_merged_logits, exported_logits
        ),
        "adapter_vs_export": compare_logits(adapter_logits, exported_logits),
    }
    in_memory_comparison = logits_report["in_memory_merge_vs_export"]
    adapter_comparison = logits_report["adapter_vs_export"]
    try:
        validate_logits_comparison(
            in_memory_comparison, max_abs_error=0.0, mean_abs_error=0.0
        )
        validate_logits_comparison(
            adapter_comparison,
            max_abs_error=MAX_LOGIT_ABS_ERROR,
            mean_abs_error=MEAN_LOGIT_ABS_ERROR,
        )
    except RuntimeError as exc:
        logits_report["status"] = "failed"
        logits_report["error"] = str(exc)
    (output_dir / "logits_verification.json").write_text(
        json.dumps(logits_report, indent=2) + "\n", encoding="utf-8"
    )
    if logits_report["status"] != "passed":
        raise RuntimeError("Merged model logits verification failed")
    print(f"PADDLEOCR_VL_MERGE_VERIFICATION={report['status']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    merge(
        args.base_model,
        args.adapter_dir,
        args.output_dir,
        args.fixture_jsonl,
        args.min_pixels,
        args.max_pixels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

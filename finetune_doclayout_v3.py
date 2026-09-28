#!/usr/bin/env python3
"""Validate and fine-tune PP-DocLayoutV3 through the PaddleX backend.

The entrypoint intentionally owns orchestration and run artifacts only.  Model
construction and long-running training remain in the installed PaddleX and
PaddleDetection packages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from vl_layout_labeler.task_map import PP_DOCLAYOUTV3_LABELS


MODEL_NAME = "PP-DocLayoutV3"
CLASS_COUNT = len(PP_DOCLAYOUTV3_LABELS)
DEFAULT_CONFIG = Path("configs/doclayoutv3/PP-DocLayoutV3-rtx5060ti.yaml")
DEFAULT_BACKEND_CONFIG = Path("configs/doclayoutv3/PP-DocLayoutV3-backend.yaml")
DEFAULT_PRETRAINED = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/PP-DocLayoutV3_pretrained.pdparams"
)
COCO_FILES = {
    "train": Path("annotations/instance_train.json"),
    "validation": Path("annotations/instance_val.json"),
}
IMAGE_DIR = Path("images")
MANIFEST = Path("export_manifest.json")


class ContractError(ValueError):
    """Raised when a dataset or run contract is not admissible."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and fine-tune PP-DocLayoutV3 from a COCO layout export."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend-config", type=Path, default=DEFAULT_BACKEND_CONFIG)
    parser.add_argument("--pretrained-weight", default=DEFAULT_PRETRAINED)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=("full", "check-only", "smoke", "benchmark", "pilot"),
        default="full",
    )
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument(
        "--install-backend",
        action="store_true",
        help="Run PaddleX's explicit local PaddleDetection installation command first.",
    )
    parser.add_argument(
        "--skip-backend-check",
        action="store_true",
        help="Skip PaddleX registry/trainer preflight (for offline unit tests only).",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ContractError("batch size must be positive")
    if args.num_workers < 0:
        raise ContractError("num workers must be non-negative")
    if args.epochs <= 0:
        raise ContractError("epochs must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ContractError("learning rate must be finite and positive")
    if args.warmup_steps < 0:
        raise ContractError("warmup steps must be non-negative")
    if args.eval_interval <= 0:
        raise ContractError("eval interval must be positive")
    if args.seed < 0:
        raise ContractError("seed must be non-negative")
    if not re.fullmatch(r"gpu:0", args.device):
        raise ContractError("this RTX 5060 Ti profile supports only device gpu:0")
    if args.mode == "check-only" and args.resume_from is not None:
        raise ContractError("--check-only cannot be combined with --resume-from")
    if args.mode == "benchmark" and args.resume_from is not None:
        raise ContractError("--benchmark cannot be combined with --resume-from")
    if args.benchmark_json is not None and args.mode != "benchmark":
        raise ContractError("--benchmark-json requires --mode benchmark")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = [
        dataset_dir / IMAGE_DIR,
        *(dataset_dir / path for path in COCO_FILES.values()),
        dataset_dir / MANIFEST,
    ]
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    for path in sorted(files):
        digest.update(str(path.relative_to(dataset_dir)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"COCO payload must be an object: {path}")
    return payload


def _finite_numbers(values: Any, label: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ContractError(f"{label} must be a non-empty list")
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in numbers):
        raise ContractError(f"{label} contains a non-finite value")
    return numbers


def _polygon_area(flat_polygon: list[float]) -> float:
    if len(flat_polygon) < 6 or len(flat_polygon) % 2:
        raise ContractError("segmentation polygon must contain at least three points")
    points = list(zip(flat_polygon[::2], flat_polygon[1::2]))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        / 2.0
    )


def _validate_categories(payload: dict[str, Any], path: Path) -> None:
    categories = payload.get("categories")
    if not isinstance(categories, list):
        raise ContractError(f"categories must be a list: {path}")
    actual = [(item.get("id"), item.get("name")) for item in categories]
    expected = list(enumerate(PP_DOCLAYOUTV3_LABELS, start=1))
    if actual != expected:
        raise ContractError(
            f"categories in {path} must exactly match ordered PP-DocLayoutV3 taxonomy"
        )


def _validate_split(
    dataset_dir: Path, split: str, payload: dict[str, Any]
) -> tuple[set[int], set[str], dict[int, int], Counter[str]]:
    path = dataset_dir / COCO_FILES[split]
    _validate_categories(payload, path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ContractError(f"{path} requires images and annotations lists")
    image_ids: set[int] = set()
    image_names: set[str] = set()
    image_sizes: dict[int, tuple[int, int]] = {}
    images_root = (dataset_dir / IMAGE_DIR).resolve()
    for image in images:
        if not isinstance(image, dict):
            raise ContractError(f"image entries must be objects: {path}")
        image_id = image.get("id")
        name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(image_id, int) or image_id in image_ids:
            raise ContractError(f"duplicate or invalid image id in {path}: {image_id}")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ContractError(f"invalid image file name in {path}: {name!r}")
        resolved = (dataset_dir / IMAGE_DIR / name).resolve()
        if images_root not in resolved.parents and resolved != images_root:
            raise ContractError(f"image escapes images directory: {name}")
        if not resolved.is_file():
            raise ContractError(f"missing image referenced by {path}: {name}")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ContractError(f"image dimensions must be positive: {name}")
        image_ids.add(image_id)
        image_names.add(name)
        image_sizes[image_id] = (width, height)

    annotation_ids: set[int] = set()
    orders: defaultdict[int, list[int]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ContractError(f"annotation entries must be objects: {path}")
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if not isinstance(annotation_id, int) or annotation_id in annotation_ids:
            raise ContractError(f"duplicate or invalid annotation id in {path}: {annotation_id}")
        if image_id not in image_ids:
            raise ContractError(f"annotation references unknown image id: {image_id}")
        if not isinstance(category_id, int) or not 1 <= category_id <= CLASS_COUNT:
            raise ContractError(f"annotation has invalid category id: {category_id}")
        width, height = image_sizes[image_id]
        bbox = _finite_numbers(annotation.get("bbox"), "bbox")
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < 0 or bbox[1] < 0:
            raise ContractError(f"invalid bbox in {path}: {bbox}")
        if bbox[0] + bbox[2] > width + 1e-4 or bbox[1] + bbox[3] > height + 1e-4:
            raise ContractError(f"bbox exceeds image bounds in {path}: {bbox}")
        segmentations = annotation.get("segmentation")
        if not isinstance(segmentations, list) or not segmentations:
            raise ContractError(f"segmentation is required in {path}")
        polygon_area = 0.0
        for polygon in segmentations:
            polygon_area += _polygon_area(_finite_numbers(polygon, "segmentation"))
        area = annotation.get("area")
        if not isinstance(area, (int, float)) or not math.isfinite(float(area)) or area <= 0:
            raise ContractError(f"annotation area must be positive in {path}")
        if polygon_area <= 0:
            raise ContractError(f"degenerate polygon in {path}")
        read_order = annotation.get("read_order")
        if not isinstance(read_order, int) or read_order < 0:
            raise ContractError(f"read_order must be a non-negative integer in {path}")
        annotation_ids.add(annotation_id)
        orders[image_id].append(read_order)
        class_counts[PP_DOCLAYOUTV3_LABELS[category_id - 1]] += 1
    for image_id, values in orders.items():
        if sorted(values) != list(range(len(values))):
            raise ContractError(
                f"read_order must be continuous from zero for image {image_id} in {path}"
            )
    return image_ids, image_names, {key: len(value) for key, value in orders.items()}, class_counts


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise ContractError(f"dataset directory not found: {dataset_dir}")
    if not (dataset_dir / IMAGE_DIR).is_dir():
        raise ContractError(f"missing images directory: {dataset_dir / IMAGE_DIR}")
    manifest_path = dataset_dir / MANIFEST
    if not manifest_path.is_file():
        raise ContractError(f"missing export manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("format") != "COCOInstSegDataset":
        raise ContractError("export_manifest.json must declare COCOInstSegDataset")
    split_payloads = {split: _load_json(dataset_dir / rel) for split, rel in COCO_FILES.items()}
    split_info: dict[str, Any] = {}
    all_ids: set[int] = set()
    all_names: set[str] = set()
    class_counts: dict[str, dict[str, int]] = {}
    for split, payload in split_payloads.items():
        ids, names, order_counts, counts = _validate_split(dataset_dir, split, payload)
        split_info[split] = {
            "images": len(ids),
            "annotations": sum(order_counts.values()),
            "image_ids": sorted(ids),
            "image_names": sorted(names),
            "per_class": dict(counts),
        }
        if all_ids & ids or all_names & names:
            raise ContractError("train/validation split leakage detected")
        all_ids.update(ids)
        all_names.update(names)
        class_counts[split] = {label: counts.get(label, 0) for label in PP_DOCLAYOUTV3_LABELS}
    manifest_categories = manifest.get("categories")
    if manifest_categories is not None and manifest_categories != CLASS_COUNT:
        raise ContractError("export manifest category count must be 25")
    return {
        "path": str(dataset_dir),
        "fingerprint": dataset_fingerprint(dataset_dir),
        "format": "COCOInstSegDataset",
        "classes": CLASS_COUNT,
        "labels": list(PP_DOCLAYOUTV3_LABELS),
        "splits": split_info,
        "per_class": class_counts,
    }


def _module_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    return Path(spec.origin).resolve().parent if spec and spec.origin else None


def collect_environment() -> dict[str, Any]:
    versions = {}
    for package in ("paddlex", "paddlepaddle", "paddlepaddle-gpu", "pycocotools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    return {
        "versions": versions,
        "paddlex_path": str(_module_path("paddlex")) if _module_path("paddlex") else None,
        "paddledetection_path": os.environ.get("PADDLE_PDX_PADDLEDETECTION_PATH"),
    }


def install_backend_command() -> list[str]:
    paddlex = Path(sys.executable).with_name("paddlex")
    return [str(paddlex), "--install", "PaddleDetection", "--use_local_repos", "-y"]


def preflight_backend(config_path: Path, backend_config: Path) -> dict[str, Any]:
    try:
        import paddlex  # noqa: F401
        from paddlex.modules import build_trainer
        from paddlex.repo_apis.base.config import _create_config
        from paddlex.repo_apis.base.register import get_registered_model_info
    except Exception as exc:
        raise ContractError(f"PaddleX backend import failed: {exc}") from exc
    try:
        info = get_registered_model_info(MODEL_NAME)
    except Exception as exc:
        raise ContractError(
            f"{MODEL_NAME} is not registered; run: {' '.join(install_backend_command())}"
        ) from exc
    config = _create_config(MODEL_NAME, config_path=str(backend_config))
    trainer_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trainer_config["Train"]["basic_config_path"] = str(backend_config.resolve())
    from paddlex.utils.config import AttrDict, create_attr_dict

    create_attr_dict(trainer_config)
    build_trainer(AttrDict(trainer_config))
    runner_root = info.get("runner_root_path")
    marker = None
    if runner_root:
        marker = Path(runner_root) / ".installed"
    if marker is None or not marker.is_file():
        paddlex_root = _module_path("paddlex")
        candidate = paddlex_root / "repo_manager/repos/PaddleDetection/.installed" if paddlex_root else None
        marker = candidate if candidate else marker
    if marker is None or not marker.is_file():
        raise ContractError(
            "PaddleDetection runner is not installed (.installed marker missing); "
            f"run: {' '.join(install_backend_command())}"
        )
    return {
        "registered_model": MODEL_NAME,
        "model_info_config": info.get("config_path"),
        "runner_root": runner_root,
        "installation_marker": str(marker),
        "installation_marker_sha256": sha256_file(marker),
        "config_path": str(config_path),
        "backend_config": str(backend_config),
        "config_object": type(config).__name__,
    }


def resolve_config(args: argparse.Namespace, dataset: dict[str, Any], work_dir: Path) -> Path:
    template = args.config.expanduser().resolve()
    backend = args.backend_config.expanduser().resolve()
    if not template.is_file():
        raise ContractError(f"config not found: {template}")
    if not backend.is_file():
        raise ContractError(f"backend config not found: {backend}")
    try:
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"invalid repository config: {template}: {exc}") from exc
    if not isinstance(config, dict):
        raise ContractError("repository config must be a YAML object")
    config.setdefault("Global", {})
    config.setdefault("Train", {})
    config.setdefault("Evaluate", {})
    config["Global"].update(
        {
            "model": MODEL_NAME,
            "mode": "train" if args.mode != "check-only" else "check_dataset",
            "dataset_dir": str(dataset["path"]),
            "device": args.device,
            "output": str((work_dir / "output").resolve()),
        }
    )
    config["Train"].update(
        {
            "num_classes": CLASS_COUNT,
            "epochs_iters": 1 if args.mode == "smoke" else (args.epochs if args.mode != "pilot" else min(args.epochs, 3)),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "resume_path": str(args.resume_from.resolve()) if args.resume_from else None,
            "pretrain_weight_path": args.pretrained_weight,
            "num_workers": args.num_workers,
            "eval_interval": args.eval_interval,
            "amp": "OFF",
            "dy2st": False,
            "basic_config_path": str(backend),
        }
    )
    config["Evaluate"]["weight_path"] = str(
        (work_dir / "output/best_model/best_model.pdparams").resolve()
    )
    resolved = work_dir / "resolved-config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return resolved


def build_engine_command(config_path: Path, mode: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        "from paddlex.engine import Engine; Engine().run()",
        "-c",
        str(config_path),
        "-o",
        f"Global.mode={mode}",
    ]


def run_engine(config_path: Path, mode: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    command = build_engine_command(config_path, mode)
    completed = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise ContractError(f"PaddleX {mode} failed with exit code {completed.returncode}")
    return completed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _prepare_work_dir(args: argparse.Namespace) -> Path:
    if args.work_dir is None:
        work_dir = (Path("runs") / f"doclayoutv3_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    work_dir = args.work_dir.expanduser().resolve()
    if work_dir.exists():
        if args.resume_from is None:
            raise ContractError(f"work directory already exists; choose a new path: {work_dir}")
        if not work_dir.is_dir() or not (work_dir / "summary.json").is_file():
            raise ContractError("resume requires an existing run directory with summary.json")
    else:
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir()
    return work_dir


def _validate_resume(args: argparse.Namespace, work_dir: Path, dataset: dict[str, Any]) -> None:
    if args.resume_from is None:
        return
    checkpoint = args.resume_from.expanduser().resolve()
    if checkpoint.suffix != ".pdparams" or not checkpoint.is_file():
        raise ContractError("--resume-from must point to an existing .pdparams checkpoint")
    try:
        checkpoint.relative_to(work_dir)
    except ValueError as exc:
        raise ContractError("resume checkpoint must be inside --work-dir") from exc
    summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("dataset", {}).get("fingerprint") != dataset["fingerprint"]:
        raise ContractError("resume dataset fingerprint does not match the original run")
    if summary.get("config", {}).get("backend_config") != str(args.backend_config.expanduser().resolve()):
        raise ContractError("resume backend config does not match the original run")


def _write_preflight(
    work_dir: Path, args: argparse.Namespace, dataset: dict[str, Any], resolved: Path, backend: dict[str, Any] | None
) -> dict[str, Any]:
    payload = {
        "created_at": utc_now(),
        "command": sys.argv,
        "environment": collect_environment(),
        "dataset": dataset,
        "config": {
            "resolved": str(resolved),
            "checksum": sha256_file(resolved),
            "backend_config": str(args.backend_config.expanduser().resolve()),
            "backend_checksum": sha256_file(args.backend_config.expanduser().resolve()),
            "pretrained_weight": args.pretrained_weight,
            "pretrained_weight_checksum": sha256_file(Path(args.pretrained_weight).expanduser().resolve())
            if not str(args.pretrained_weight).startswith("http") and Path(args.pretrained_weight).expanduser().is_file()
            else None,
        },
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "backend": backend,
    }
    _atomic_json(work_dir / "preflight.json", payload)
    return payload


def _benchmark(args: argparse.Namespace, work_dir: Path, resolved: Path) -> None:
    candidates = [(batch, workers) for batch in (1, 2, 4) for workers in (2, 4, 6, 8)]
    records = [
        {
            "batch_size": batch,
            "num_workers": workers,
            "status": "not_run",
            "reason": "benchmark candidates require an explicit GPU benchmark invocation",
        }
        for batch, workers in candidates
    ]
    _atomic_json(
        args.benchmark_json.expanduser().resolve() if args.benchmark_json else work_dir / "benchmark.json",
        {"created_at": utc_now(), "resolved_config": str(resolved), "candidates": records},
    )


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    work_dir = _prepare_work_dir(args)
    summary: dict[str, Any] = {
        "started_at": utc_now(),
        "status": "running",
        "command": sys.argv,
        "mode": args.mode,
        "artifacts": {"work_dir": str(work_dir)},
    }
    try:
        dataset = validate_dataset(args.dataset_dir)
        _validate_resume(args, work_dir, dataset)
        resolved = resolve_config(args, dataset, work_dir)
        if args.install_backend:
            install_log = work_dir / "backend-install.log"
            completed = subprocess.run(install_backend_command(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            install_log.write_text(completed.stdout or "", encoding="utf-8")
            if completed.returncode != 0:
                raise ContractError(f"backend installation failed with exit code {completed.returncode}")
        backend = None if args.skip_backend_check else preflight_backend(resolved, args.backend_config.expanduser().resolve())
        _write_preflight(work_dir, args, dataset, resolved, backend)
        summary.update(
            {
                "dataset": dataset,
                "config": {
                    "resolved": str(resolved),
                    "checksum": sha256_file(resolved),
                    "backend_config": str(args.backend_config.expanduser().resolve()),
                },
                "device": args.device,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
            }
        )
        _atomic_json(work_dir / "summary.json", summary)
        if args.mode == "benchmark":
            _benchmark(args, work_dir, resolved)
        elif args.mode == "check-only":
            run_engine(resolved, "check_dataset", work_dir / "dataset-check.log")
            summary["artifacts"]["dataset_check_log"] = str(work_dir / "dataset-check.log")
        else:
            run_engine(resolved, "train", work_dir / "train.log")
            summary["artifacts"]["train_log"] = str(work_dir / "train.log")
        summary["status"] = "passed"
        summary["finished_at"] = utc_now()
        _atomic_json(work_dir / "summary.json", summary)
        return 0
    except Exception as exc:
        summary.update({"status": "failed", "finished_at": utc_now(), "error": str(exc)})
        _atomic_json(work_dir / "summary.json", summary)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

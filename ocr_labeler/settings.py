from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
from typing import Literal


@dataclass(frozen=True)
class LabelerSettings:
    task: Literal["ocr", "detection"] = "ocr"
    rec_model_dir: Path = Path("runs/vi_rec_3datasets_v1/inference/best_accuracy")
    det_model_dir: Path | None = None
    det_model_name: str = "PP-OCRv6_medium_det"
    device: str = "gpu:0"
    text_det_limit_side_len: int = 1600
    text_det_limit_type: Literal["min", "max"] = "max"
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.6
    text_det_unclip_ratio: float = 1.5
    text_rec_input_shape: tuple[int, int, int] = (3, 48, 1600)
    text_rec_score_thresh: float = 0.0
    confidence_warning_threshold: float = 0.60
    autosave_delay_ms: int = 500
    host: str = "127.0.0.1"
    port: int = 8010

    @property
    def data_dir_name(self) -> str:
        return (
            ".paddleocr-det-labeler"
            if self.task == "detection"
            else ".paddleocr-labeler"
        )

    def validate(self) -> "LabelerSettings":
        if self.task not in {"ocr", "detection"}:
            raise ValueError("task must be ocr or detection")
        if self.task == "ocr":
            rec_dir = self.rec_model_dir.expanduser().resolve()
            required = {
                "inference.json",
                "inference.pdiparams",
                "inference.yml",
                "ppocr_keys.txt",
            }
            missing = sorted(
                name for name in required if not (rec_dir / name).is_file()
            )
            if missing:
                raise ValueError(
                    "recognition model is missing: " + ", ".join(missing)
                )
        if self.det_model_dir is not None:
            det_dir = self.det_model_dir.expanduser().resolve()
            det_required = {"inference.json", "inference.pdiparams", "inference.yml"}
            det_missing = sorted(
                name for name in det_required if not (det_dir / name).is_file()
            )
            if det_missing:
                raise ValueError(
                    "detection model is missing: " + ", ".join(det_missing)
                )
        if self.text_det_limit_side_len <= 0:
            raise ValueError("text_det_limit_side_len must be positive")
        if self.text_det_limit_type not in {"min", "max"}:
            raise ValueError("text_det_limit_type must be min or max")
        for name, value in (
            ("text_det_thresh", self.text_det_thresh),
            ("text_det_box_thresh", self.text_det_box_thresh),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.text_det_unclip_ratio <= 0:
            raise ValueError("text_det_unclip_ratio must be positive")
        if self.device != "cpu" and re.fullmatch(r"gpu:[0-9]+", self.device) is None:
            raise ValueError("device must be cpu or gpu:<index>")
        if self.host != "localhost":
            try:
                host_address = ipaddress.ip_address(self.host)
            except ValueError as exc:
                raise ValueError(
                    "host must be localhost or a loopback address"
                ) from exc
            if not host_address.is_loopback:
                raise ValueError("host must be localhost or a loopback address")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return self

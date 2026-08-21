from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path


@dataclass(frozen=True)
class LabelerSettings:
    layout_model_dir: Path = Path("/home/tieubaoca/.paddlex/official_models/PP-DocLayoutV3")
    device: str = "gpu:0"
    vl_base_url: str = "http://127.0.0.1:8000/v1"
    vl_model: str = "paddleocr-vl"
    vl_api_key: str | None = None
    vl_timeout: float = 120.0
    vl_max_tokens: int = 4096
    host: str = "127.0.0.1"
    port: int = 8012
    data_dir_name: str = ".paddleocr-vl-labeler"

    def validate(self, *, require_runtime_models: bool = True) -> LabelerSettings:
        if self.host != "localhost":
            try:
                if not ipaddress.ip_address(self.host).is_loopback:
                    raise ValueError("host must be loopback-only")
            except ValueError as exc:
                if str(exc) == "host must be loopback-only":
                    raise
                raise ValueError("host must be localhost or a loopback IP") from exc
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if not self.vl_base_url.strip():
            raise ValueError("vl_base_url must not be empty")
        if not self.vl_model.strip():
            raise ValueError("vl_model must not be empty")
        if self.vl_timeout <= 0 or self.vl_max_tokens <= 0:
            raise ValueError("VL timeout and max tokens must be positive")
        if require_runtime_models:
            model_dir = self.layout_model_dir.expanduser().resolve()
            required = {"inference.json", "inference.pdiparams", "inference.yml"}
            missing = sorted(name for name in required if not (model_dir / name).is_file())
            if missing:
                raise ValueError("layout model is missing: " + ", ".join(missing))
        return self

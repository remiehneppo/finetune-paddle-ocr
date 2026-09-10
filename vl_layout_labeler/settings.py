from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re


def _is_valid_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if len(host) > 253:
        return False
    allowed = re.compile(r"^(?!-)[A-Z0-9-]{1,63}(?<!-)$", re.IGNORECASE)
    return all(allowed.match(part) for part in host.split("."))


@dataclass(frozen=True)
class LabelerSettings:
    layout_model_dir: Path = Path("/home/tieubaoca/.paddlex/official_models/PP-DocLayoutV3")
    device: str = "gpu:0"
    vl_base_url: str = "http://127.0.0.1:8000/v1"
    vl_model: str = "paddleocr-vl"
    vl_api_key: str | None = None
    vl_timeout: float = 120.0
    vl_max_tokens: int = 4096
    validation_base_url: str | None = None
    validation_model: str | None = None
    validation_api_key: str | None = None
    validation_timeout: float = 30.0
    validation_max_tokens: int = 2048
    threads: int = 10
    host: str = "127.0.0.1"
    port: int = 8012
    data_dir_name: str = ".paddleocr-vl-labeler"
    allowed_root: Path | None = None

    @property
    def validation_configured(self) -> bool:
        return bool(
            (self.validation_base_url or "").strip()
            and (self.validation_model or "").strip()
        )

    def validate(self, *, require_runtime_models: bool = True) -> LabelerSettings:
        if not self.host or not self.host.strip():
            raise ValueError("host must not be empty")
        if not _is_valid_host(self.host.strip()):
            raise ValueError("host must be a valid IP address or hostname")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if not self.vl_base_url.strip():
            raise ValueError("vl_base_url must not be empty")
        if not self.vl_model.strip():
            raise ValueError("vl_model must not be empty")
        if self.vl_timeout <= 0 or self.vl_max_tokens <= 0:
            raise ValueError("VL timeout and max tokens must be positive")
        has_validation_url = bool((self.validation_base_url or "").strip())
        has_validation_model = bool((self.validation_model or "").strip())
        if has_validation_url != has_validation_model:
            raise ValueError(
                "validation_base_url and validation_model must be configured together"
            )
        if self.validation_timeout <= 0 or self.validation_max_tokens <= 0:
            raise ValueError("validation timeout and max tokens must be positive")
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if require_runtime_models:
            model_dir = self.layout_model_dir.expanduser().resolve()
            required = {"inference.json", "inference.pdiparams", "inference.yml"}
            missing = sorted(name for name in required if not (model_dir / name).is_file())
            if missing:
                raise ValueError("layout model is missing: " + ", ".join(missing))
        return self

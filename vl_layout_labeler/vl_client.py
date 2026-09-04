from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
from PIL import Image

from .geometry import crop_box_from_polygon
from paddleocr_vl_contract import prompt_for_task

prompt_for_block_task = prompt_for_task


class VLClientError(RuntimeError):
    pass


def image_data_url(path: Path, bbox: tuple[int, int, int, int] | None = None) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if bbox is not None:
            image = image.crop(bbox)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_chat_payload(model: str, image_url: str, task: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    {"type": "text", "text": prompt_for_block_task(task)},
                ],
            }
        ],
    }


def extract_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VLClientError("VL response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
    raise VLClientError("VL response content is not text")


class VLClient:
    def __init__(self, settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(timeout=settings.vl_timeout)
        self._owns_client = client is None

    def check_ready(self) -> None:
        endpoint = self.settings.vl_base_url.rstrip("/") + "/models"
        headers = {}
        if self.settings.vl_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vl_api_key}"
        try:
            response = self.client.get(endpoint, headers=headers)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VLClientError(f"VL endpoint is unavailable: {exc}") from exc
        models = payload.get("data", []) if isinstance(payload, dict) else []
        identifiers = {
            item.get("id") for item in models if isinstance(item, dict) and item.get("id")
        }
        if identifiers and self.settings.vl_model not in identifiers:
            raise VLClientError(
                f"VL model {self.settings.vl_model!r} is not served; available: "
                + ", ".join(sorted(identifiers))
            )

    def prelabel(self, path: Path, polygon, task: str, width: int, height: int) -> str:
        bbox = crop_box_from_polygon(polygon, width, height)
        payload = build_chat_payload(
            self.settings.vl_model,
            image_data_url(path, bbox),
            task,
            self.settings.vl_max_tokens,
        )
        headers = {"Content-Type": "application/json"}
        if self.settings.vl_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vl_api_key}"
        endpoint = self.settings.vl_base_url.rstrip("/") + "/chat/completions"
        try:
            response = self.client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            text = extract_content(response.json())
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            raise VLClientError(f"VL request failed: {exc}") from exc
        if not text:
            raise VLClientError("VL returned empty text")
        return text

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

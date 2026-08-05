"""Keep PaddleOCR-VL LoRA on the text decoder when ERNIEKit starts Python."""

from __future__ import annotations

import json
import os
from pathlib import Path

SCOPE_MARKER = "PADDLEOCR_VL_LORA_SCOPE=text_decoder_only"
TRAINABLE_NAMES_MARKER = "PADDLEOCR_VL_TRAINABLE_PARAMETER_NAMES="
CHECKPOINT_MARKER = "PADDLEOCR_VL_CHECKPOINT_SAVE_COMPAT=enabled"


def decoder_target_modules() -> list[str]:
    """Return exact decoder projection patterns, excluding every vision path."""
    return [
        rf"model\.layers\.\d+\.self_attn\.{projection}.*"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    ] + [
        rf"model\.layers\.\d+\.mlp\.{projection}.*"
        for projection in ("up_proj", "gate_proj", "down_proj")
    ]


def trainable_parameter_names(model) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if not getattr(parameter, "stop_gradient", True)
    ]


def patch_pretraining_trainer_class(trainer_class) -> None:
    if getattr(trainer_class.save_model, "_paddleocr_vl_compatible", False):
        return

    def save_model(
        self,
        output_dir=None,
        merge_tensor_parallel=False,
        last_fc_to_hf=False,
    ):
        super(trainer_class, self).save_model(
            output_dir, merge_tensor_parallel, last_fc_to_hf
        )
        if self.args.should_save:
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "static_name_to_dyg_name.json").write_text(
                json.dumps(self.static_name_to_dyg_name), encoding="utf-8"
            )

    save_model._paddleocr_vl_compatible = True
    trainer_class.save_model = save_model


def _patch_trainable_parameter_report(peft_utils) -> None:
    original = peft_utils.initialize_lora_model
    if getattr(original, "_paddleocr_vl_trainable_names", False):
        return

    def initialize_lora_model(*args, **kwargs):
        model = original(*args, **kwargs)
        names = trainable_parameter_names(model)
        print(
            TRAINABLE_NAMES_MARKER + json.dumps(names, separators=(",", ":")),
            flush=True,
        )
        return model

    initialize_lora_model._paddleocr_vl_trainable_names = True
    peft_utils.initialize_lora_model = initialize_lora_model


def _patch_erniekit_lora_scope() -> None:
    try:
        from ernie.utils import peft_utils
    except ModuleNotFoundError:
        return

    original = peft_utils.LoRAConfig
    if getattr(original, "_paddleocr_vl_text_only", False):
        print(SCOPE_MARKER, flush=True)
        return

    class TextDecoderLoRAConfig(original):
        _paddleocr_vl_text_only = True

        def __init__(self, *args, **kwargs):
            kwargs["target_modules"] = decoder_target_modules()
            super().__init__(*args, **kwargs)

    peft_utils.LoRAConfig = TextDecoderLoRAConfig
    _patch_trainable_parameter_report(peft_utils)
    print(SCOPE_MARKER, flush=True)

    from erniekit.train.ocr_vl_sft.pretraining_trainer import PretrainingTrainer

    patch_pretraining_trainer_class(PretrainingTrainer)
    print(CHECKPOINT_MARKER, flush=True)


if os.environ.get("PADDLEOCR_VL_TEXT_ONLY_LORA") == "1":
    _patch_erniekit_lora_scope()

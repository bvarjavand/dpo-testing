import os
from typing import Callable
import determined as det
from transformers.trainer_utils import get_last_checkpoint

def download_ckpt(ckpt_uuid: str, core_context: det.core.Context) -> str:
    download_dir = os.path.join(os.environ.get("HF_CACHE", "."), ckpt_uuid)

    def selector(path: str) -> bool:
        if any(
            [
                path.endswith(ext)
                for ext in [
                    "config.json",
                    "generation-config.json",
                    ".safetensors",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "tokenizer.model",
                    "model.safetensors.index.json",
                ]
            ]
        ):
            return True

        return False

    core_context.checkpoint.download(ckpt_uuid, download_dir, selector=selector)
    model_dir = get_last_checkpoint(download_dir)
    return model_dir
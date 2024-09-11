from typing import Optional
from transformers import AutoTokenizer

from ..utils.chat_format import CHAT_ML_TEMPLATE

def get_tokenizer(
    model_name: str,
    padding_side: str = "right",
    truncation_side: str = "right",
    model_max_length: Optional[int] = None,
    add_eos_token: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side=truncation_side,
        padding_side=padding_side,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = model_max_length if model_max_length else 2048

    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_ML_TEMPLATE

    tokenizer.add_eos_token = add_eos_token
    return tokenizer

def get_response_template_ids(tokenizer):
    assistant_prompt = "<|im_start|>assistant\n"
    return tokenizer.encode(assistant_prompt, add_special_tokens=False)
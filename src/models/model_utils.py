import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

def get_model(model_name: str, inference: bool = False, device_map: str = "auto") -> PreTrainedModel:
    if inference:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    return model

def setup_special_tokens(
    model: PreTrainedModel,
    tokenizer,
    special_tokens: list[str],
):
    # We won't be changing bos, eos and pad token, though.
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    return model, tokenizer
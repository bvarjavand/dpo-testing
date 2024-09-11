import logging
from typing import Any, Dict

from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from ..data.dataset_loader import load_sft_dataset
from ..models.model_utils import get_model, setup_special_tokens
from ..models.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

def setup_sft_trainer(training_args: TrainingArguments, hparams: Dict[str, Any]):
    dataset = load_sft_dataset(hparams)

    model_name = hparams["model"]
    model = get_model(model_name)
    tokenizer = get_tokenizer(
        model_name,
        padding_side="right",
        truncation_side="right",
        model_max_length=hparams["max_seq_length"],
        add_eos_token=True,
    )

    if hparams["chat_tokens"]["add_chat_tokens"]:
        model, tokenizer = setup_special_tokens(
            model, tokenizer, hparams["chat_tokens"]["special_tokens"]
        )

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["prompt"])):
            prompt = [
                {"role": "user", "content": example["prompt"][i]},
                {"role": "assistant", "content": example["text"][i]},
            ]
            text = tokenizer.apply_chat_template(prompt, tokenize=False)
            output_texts.append(text)
        return output_texts

    if hparams["data_collator"]["on_completions_only"]:
        assistant_prompt = hparams["data_collator"]["response_template"]
        response_template_ids = tokenizer.encode(
            assistant_prompt, add_special_tokens=False
        )
        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
        logger.info("Using DataCollatorForCompletionOnlyLM.")
    else:
        collator = None
        logger.info("Using default data collator")

    logger.info(f"dataset_sample={dataset['train'][0]}")

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_prompts_func,
        max_seq_length=hparams["max_seq_length"],
    )

    return trainer
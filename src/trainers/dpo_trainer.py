import logging
from typing import Any, Dict

from transformers import TrainingArguments
from trl import DPOTrainer

from ..data.dataset_loader import load_dpo_datasets
from ..models.model_utils import get_model
from ..models.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

def setup_dpo_trainer(core_context, training_args: TrainingArguments, hparams: Dict[str, Any]):
    model_ckpt = hparams.get("model_ckpt", None)
    if model_ckpt:
        model_name_or_path = download_ckpt(model_ckpt, core_context)
    else:
        model_name_or_path = hparams["model_name"]

    model = get_model(model_name_or_path)
    if not hparams["precompute_ref_log_probs"]:
        model_ref = get_model(model_name_or_path)
        model_ref.eval()
    else:
        model_ref = None

    tokenizer = get_tokenizer(
        model_name_or_path,
        truncation_side="left",
        padding_side="left",
        model_max_length=hparams["max_length"],
        add_eos_token=False,
    )

    dataset = load_dpo_datasets(hparams["datasets"], tokenizer)

    if core_context.distributed.rank == 0:
        for index in [0, 1, 2]:
            logger.info(
                f"Prompt sample {index} of the raw training set:\n\n{dataset['train'][index]['prompt']}"
            )
            logger.info(
                f"Chosen sample {index} of the raw training set:\n\n{dataset['train'][index]['chosen']}"
            )
            logger.info(
                f"Rejected sample {index} of the raw training set:\n\n{dataset['train'][index]['rejected']}"
            )

    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=hparams["dpo_beta"],
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss_type=hparams["dpo_loss"],
        tokenizer=tokenizer,
        precompute_ref_log_probs=hparams["precompute_ref_log_probs"],
        max_length=hparams["max_length"],
        max_prompt_length=hparams["max_prompt_length"],
        max_target_length=hparams["max_target_length"],
    )

    return trainer
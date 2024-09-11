import determined as det
from determined.transformers import DetCallback
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from models import RewardModel
from data import load_and_preprocess_dataset
from trainer import DPOTrainer
import torch
from typing import Dict, Any
from datasets import Dataset
import evaluate

def build_model(trial_context: det.core.Context) -> Dict[str, Any]:
    hparams = trial_context.get_hparams()
    
    model = AutoModelForCausalLM.from_pretrained(hparams["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(hparams["model_name"])
    
    reward_model = RewardModel(hidden_size=model.config.hidden_size)
    
    dataset = load_and_preprocess_dataset(
        hparams["dataset_name"],
        tokenizer,
        max_length=hparams["max_length"],
    )
    
    training_args = TrainingArguments(
        **hparams["training_args"],
        report_to_determined=trial_context
    )

    dpo_trainer = DPOTrainer(
        model=model,
        reward_model=reward_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        beta=hparams["beta"],
    )
        
    return {
        "model": model,
        "tokenizer": tokenizer,
        "reward_model": reward_model,
        "dataset": dataset,
        "dpo_trainer": dpo_trainer,
    }

def main(core_context):
    model_dict = build_model(core_context)
    dpo_trainer = model_dict["dpo_trainer"]
    
    dpo_trainer.train()

    # Evaluate the model
    eval_results = dpo_trainer.evaluate()


if __name__ == "__main__":
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
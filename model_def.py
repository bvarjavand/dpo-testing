import determined as det
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import RewardModel
from data import load_and_preprocess_dataset
from trainer import DPOTrainer
import torch
from typing import Dict, Any
from datasets import Dataset


class DPOTrial(PyTorchTrial):
    """
    A PyTorch trial for Direct Preference Optimization (DPO) fine-tuning of language models.

    This class implements the necessary methods for training and evaluating a language model
    using the DPO approach within the Determined AI framework.

    Attributes:
        context (PyTorchTrialContext): The context provided by Determined AI.
        hparams (Dict[str, Any]): Hyperparameters for the trial.
        model (AutoModelForCausalLM): The language model being fine-tuned.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        reward_model (RewardModel): The reward model used in DPO.
        dataset (Dataset): The dataset used for training and validation.
        dpo_trainer (DPOTrainer): The DPO trainer instance.
        optimizer (torch.optim.Optimizer): The optimizer for model training.
    """

    def __init__(self, context: PyTorchTrialContext) -> None:
        """
        Initialize the DPOTrial.

        Args:
            context (PyTorchTrialContext): The context provided by Determined AI.
        """
        self.context: PyTorchTrialContext = context
        self.hparams: Dict[str, Any] = context.get_hparams()

        # Load model and tokenizer
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.hparams["model_name"]
        )
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.hparams["model_name"]
        )

        # Define reward model
        self.reward_model: RewardModel = RewardModel(
            hidden_size=self.model.config.hidden_size
        )

        # Load and preprocess dataset
        self.dataset: Dataset = load_and_preprocess_dataset(
            self.hparams["dataset_name"],
            self.tokenizer,
            max_length=self.hparams["max_length"],
        )

        # Create DPO trainer
        self.dpo_trainer: DPOTrainer = DPOTrainer(
            reward_model=self.reward_model, beta=self.hparams["beta"]
        )

        # Wrap the model
        self.model = self.context.wrap_model(self.model)

        # Wrap the optimizer
        self.optimizer: torch.optim.Optimizer = self.context.wrap_optimizer(
            torch.optim.AdamW(self.model.parameters(), lr=self.hparams["learning_rate"])
        )

    def train_batch(
        self, batch: Dict[str, torch.Tensor], epoch_idx: int, batch_idx: int
    ) -> Dict[str, float]:
        """
        Train the model on a single batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): The input batch of data.
            epoch_idx (int): The current epoch index.
            batch_idx (int): The current batch index.

        Returns:
            Dict[str, float]: A dictionary containing the training loss.
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss: torch.Tensor = self.dpo_trainer.compute_loss(self.model, batch)
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        return {"loss": loss.item()}

    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate the model on a single batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): The input batch of data.

        Returns:
            Dict[str, float]: A dictionary containing the validation loss.
        """
        self.model.eval()
        with torch.no_grad():
            loss: torch.Tensor = self.dpo_trainer.compute_loss(self.model, batch)
        return {"validation_loss": loss.item()}

    def build_training_data_loader(self) -> DataLoader:
        """
        Build the DataLoader for training data.

        Returns:
            DataLoader: The DataLoader for training data.
        """
        return DataLoader(
            self.dataset["train"],
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.hparams.get("num_workers", 2),
            pin_memory=True,
        )

    def build_validation_data_loader(self) -> DataLoader:
        """
        Build the DataLoader for validation data.

        Returns:
            DataLoader: The DataLoader for validation data.
        """
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            num_workers=self.hparams.get("num_workers", 2),
            pin_memory=True,
        )

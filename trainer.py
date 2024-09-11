import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import Trainer, PreTrainedModel
from typing import Dict, Tuple, Union


class DPOTrainer(Trainer):
    """
    Direct Preference Optimization (DPO) Trainer for language models.

    This trainer extends the Hugging Face Trainer class to implement the DPO algorithm,
    which fine-tunes language models based on human preferences without using a reward model.

    Attributes:
        reward_model (torch.nn.Module): The reward model used to compute rewards for generated sequences.
        beta (float): The temperature parameter for the DPO loss calculation.

    """

    def __init__(self, reward_model: torch.nn.Module, beta: float = 0.1, **kwargs):
        """
        Initialize the DPOTrainer.

        Args:
            reward_model (torch.nn.Module): The reward model to use.
            beta (float, optional): The temperature parameter for the DPO loss. Defaults to 0.1.
            **kwargs: Additional keyword arguments passed to the parent Trainer class.
        """
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.beta = beta

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Compute the DPO loss for a batch of inputs.

        This method generates two sequences (chosen and rejected) from the model,
        computes their rewards and log-probabilities, and calculates the DPO loss.

        Args:
            model (PreTrainedModel): The pre-trained language model being fine-tuned.
            inputs (Dict[str, Tensor]): The input tensors, including 'input_ids' and 'attention_mask'.
            return_outputs (bool, optional): Whether to return the model outputs along with the loss. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]: The computed loss, or a tuple of the loss and model outputs if return_outputs is True.
        """
        # Generate two sequences: y_w (chosen) and y_l (rejected)
        input_ids: Tensor = inputs["input_ids"]
        attention_mask: Tensor = inputs["attention_mask"]

        # Generate y_w (chosen)
        outputs_w: Dict[str, Tensor] = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits_w: Tensor = outputs_w.logits
        y_w: Tensor = torch.argmax(logits_w, dim=-1)

        # Generate y_l (rejected) by sampling
        with torch.no_grad():
            outputs_l: Dict[str, Tensor] = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits_l: Tensor = outputs_l.logits
            y_l: Tensor = torch.multinomial(
                F.softmax(logits_l, dim=-1), num_samples=1
            ).squeeze(-1)

        # Compute rewards
        r_w: Tensor = self.reward_model(model.get_input_embeddings()(y_w)).squeeze(-1)
        r_l: Tensor = self.reward_model(model.get_input_embeddings()(y_l)).squeeze(-1)

        # Compute log-probabilities
        log_p_w: Tensor = -F.cross_entropy(
            logits_w.transpose(1, 2), y_w, reduction="none"
        )
        log_p_l: Tensor = -F.cross_entropy(
            logits_w.transpose(1, 2), y_l, reduction="none"
        )

        # Compute DPO loss
        loss: Tensor = -torch.mean(log_p_w - log_p_l - self.beta * (r_w - r_l))

        return (loss, outputs_w) if return_outputs else loss

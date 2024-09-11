import torch
import torch.nn as nn
from torch import Tensor


class RewardModel(nn.Module):
    """
    A simple reward model that maps hidden states to scalar rewards.

    This model consists of a single linear layer that takes a hidden state
    vector as input and outputs a scalar reward value.

    Attributes:
        fc (nn.Linear): The linear layer used to compute the reward.

    Args:
        hidden_size (int): The size of the input hidden state vector.
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initialize the RewardModel.

        Args:
            hidden_size (int): The size of the input hidden state vector.
        """
        super(RewardModel, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the reward for a given hidden state.

        Args:
            x (Tensor): The input hidden state tensor of shape (batch_size, hidden_size).

        Returns:
            Tensor: The computed reward tensor of shape (batch_size, 1).
        """
        return self.fc(x)

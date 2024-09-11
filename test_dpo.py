import pytest
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import RewardModel
from data import load_and_preprocess_dataset
from trainer import DPOTrainer
from datasets import Dataset

"""
This module contains unit tests for the Direct Preference Optimization (DPO) implementation.

It includes tests for model loading, tokenizer loading, reward model functionality,
dataset loading and preprocessing, DPO trainer creation, loss computation,
and model generation capabilities.

The tests use pytest fixtures to set up common objects used across multiple tests,
such as the pre-trained language model, tokenizer, reward model, dataset, and DPO trainer.

Each test function focuses on a specific aspect of the DPO implementation,
ensuring that all components work as expected individually and in combination.
"""

@pytest.fixture
def model() -> AutoModelForCausalLM:
    """
    Fixture to load the pre-trained language model.

    Returns:
        AutoModelForCausalLM: The loaded pre-trained language model.
    """
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen-0.5B")


@pytest.fixture
def tokenizer() -> AutoTokenizer:
    """
    Fixture to load the tokenizer associated with the model.

    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained("Qwen/Qwen-0.5B")


@pytest.fixture
def reward_model(model: AutoModelForCausalLM) -> RewardModel:
    """
    Fixture to create a reward model based on the language model's hidden size.

    Args:
        model (AutoModelForCausalLM): The pre-trained language model.

    Returns:
        RewardModel: The created reward model.
    """
    return RewardModel(hidden_size=model.config.hidden_size)


@pytest.fixture
def dataset(tokenizer: AutoTokenizer) -> Dataset:
    """
    Fixture to load and preprocess the dataset.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for preprocessing.

    Returns:
        Dataset: The loaded and preprocessed dataset.
    """
    return load_and_preprocess_dataset("tiny_shakespeare", tokenizer, max_length=32)


@pytest.fixture
def dpo_trainer(reward_model: RewardModel) -> DPOTrainer:
    """
    Fixture to create a DPO trainer with the reward model.

    Args:
        reward_model (RewardModel): The reward model to use in the DPO trainer.

    Returns:
        DPOTrainer: The created DPO trainer.
    """
    return DPOTrainer(reward_model=reward_model, beta=0.1)


def test_model_loading(model: AutoModelForCausalLM) -> None:
    """
    Test to ensure the model is loaded correctly.

    Args:
        model (AutoModelForCausalLM): The loaded pre-trained language model.
    """
    assert isinstance(model, AutoModelForCausalLM)


def test_tokenizer_loading(tokenizer: AutoTokenizer) -> None:
    """
    Test to ensure the tokenizer is loaded correctly.

    Args:
        tokenizer (AutoTokenizer): The loaded tokenizer.
    """
    assert isinstance(tokenizer, AutoTokenizer)


def test_reward_model(reward_model: RewardModel) -> None:
    """
    Test the reward model's forward pass.

    Args:
        reward_model (RewardModel): The reward model to test.
    """
    input_tensor: Tensor = torch.rand(1, reward_model.fc.in_features)
    output: Tensor = reward_model(input_tensor)
    assert output.shape == (1, 1)


def test_dataset_loading(dataset: Dataset) -> None:
    """
    Test to ensure the dataset is loaded and contains data.

    Args:
        dataset (Dataset): The loaded dataset.
    """
    assert "train" in dataset
    assert len(dataset["train"]) > 0


def test_dpo_trainer(dpo_trainer: DPOTrainer) -> None:
    """
    Test to ensure the DPO trainer is created correctly.

    Args:
        dpo_trainer (DPOTrainer): The created DPO trainer.
    """
    assert isinstance(dpo_trainer, DPOTrainer)


def test_dpo_loss_computation(
    model: AutoModelForCausalLM, dpo_trainer: DPOTrainer, dataset: Dataset
) -> None:
    """
    Test the DPO loss computation.

    Args:
        model (AutoModelForCausalLM): The pre-trained language model.
        dpo_trainer (DPOTrainer): The DPO trainer.
        dataset (Dataset): The loaded dataset.
    """
    batch = next(iter(dataset["train"]))
    loss: Tensor = dpo_trainer.compute_loss(model, batch)
    assert isinstance(loss, torch.Tensor)


def test_model_generation(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> None:
    """
    Test the model's text generation capability.

    Args:
        model (AutoModelForCausalLM): The pre-trained language model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """
    input_text: str = "To be"
    input_ids: Tensor = tokenizer(input_text, return_tensors="pt").input_ids
    output: Tensor = model.generate(input_ids, max_length=10)
    output_text: str = tokenizer.decode(output[0], skip_special_tokens=True)
    assert len(output_text) > len(input_text)


def test_dataset_preprocessing(dataset: Dataset) -> None:
    """
    Test the dataset preprocessing.

    Args:
        dataset (Dataset): The preprocessed dataset.
    """
    assert "input_ids" in dataset["train"].features
    assert (
        dataset["train"][0]["input_ids"].shape[0] <= 32
    )  # Ensure max_length is respected

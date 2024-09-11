from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Any


def load_and_preprocess_dataset(
    dataset_name: str, tokenizer: PreTrainedTokenizer, max_length: int
) -> Dataset:
    """
    Load and preprocess a dataset using the specified tokenizer.

    Args:
        dataset_name (str): The name of the dataset to load.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for preprocessing.
        max_length (int): The maximum length for tokenization.

    Returns:
        Dataset: The preprocessed dataset.
    """
    dataset: Dataset = load_dataset(dataset_name)

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize the text examples in the dataset.

        Args:
            examples (Dict[str, Any]): A dictionary containing the examples to tokenize.

        Returns:
            Dict[str, Any]: A dictionary containing the tokenized examples.
        """
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset: Dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return tokenized_dataset

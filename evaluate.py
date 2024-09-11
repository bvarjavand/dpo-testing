import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm


def evaluate_model(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: Dataset
) -> None:
    """
    Evaluate a pre-trained language model on a given dataset.

    This function performs the following evaluations:
    1. Calculates the perplexity of the model on the dataset.
    2. Computes the BLEU score for generated text against reference text.
    3. Generates an example output for a given prompt.

    Args:
        model (PreTrainedModel): The pre-trained language model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        dataset (Dataset): The dataset to use for evaluation.

    Returns:
        None: This function prints the evaluation results and logs them to wandb.
    """
    model.eval()
    total_loss: float = 0.0
    total_tokens: int = 0
    references: list[list[str]] = []
    hypotheses: list[str] = []

    # Evaluate on a subset of the dataset
    eval_subset: Dataset = dataset["validation"].select(
        range(min(1000, len(dataset["validation"])))
    )

    for batch in tqdm(eval_subset, desc="Evaluating"):
        inputs: dict[str, Tensor] = tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss: Tensor = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(0)
            total_tokens += inputs["input_ids"].ne(tokenizer.pad_token_id).sum().item()

            # Generate text for BLEU score calculation
            generated_ids: Tensor = model.generate(**inputs, max_length=100)
            generated_text: list[str] = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            references.extend([[text.split()] for text in batch["text"]])
            hypotheses.extend(generated_text)

    # Calculate perplexity
    perplexity: float = torch.exp(torch.tensor(total_loss / total_tokens)).item()

    # Calculate BLEU score
    bleu_score: float = corpus_bleu(references, [hyp.split() for hyp in hypotheses])

    # Generate example output
    prompt: str = "To be, or not to be:"
    inputs: dict[str, Tensor] = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs: Tensor = model.generate(**inputs, max_length=100)
    output_text: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model output: {output_text}")
    wandb.log({"example_output": output_text})

    print(f"Perplexity: {perplexity}")
    print(f"BLEU Score: {bleu_score}")

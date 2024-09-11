from datasets import Dataset
from transformers import PreTrainedTokenizer

TRIPLET_DATASET = "prompt,chosen,rejected"
CONVERSATION_DATASET = "chosen,rejected"

def is_feature_chat_conversation_format(dataset: Dataset, feature: str) -> bool:
    example = dataset[0][feature]
    if isinstance(example, list) and all(isinstance(x, dict) for x in example):
        for sample in example:
            if "content" not in sample or "role" not in sample:
                raise RuntimeError(
                    f"Column {feature} has data in unsupported format : {sample}"
                )
        return True
    else:
        raise RuntimeError(
            f"Column {feature} has data in unsupported format : {example}"
        )

def get_dataset_format(dataset: Dataset) -> str:
    if "chosen" not in dataset.features or "rejected" not in dataset.features:
        raise RuntimeError(
            f"DPO-compatible dataset requires 'chosen' and 'rejected' features."
        )

    if all(feature in dataset.features for feature in ["prompt", "chosen", "rejected"]):
        return TRIPLET_DATASET

    if is_feature_chat_conversation_format(
        dataset, "chosen"
    ) and is_feature_chat_conversation_format(dataset, "rejected"):
        return CONVERSATION_DATASET

def process_conversation_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_data = {"prompt": [], "chosen": [], "rejected": []}

    for example in dataset:
        assert ". ".join([x["content"] for x in example["chosen"][:-1]]) == ". ".join(
            [x["content"] for x in example["rejected"][:-1]]
        )
        assert all(x["role"] != "system" for x in example["chosen"])

        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

        processed_data["prompt"].append(
            tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        )
        processed_data["chosen"].append(
            tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        )
        processed_data["rejected"].append(
            tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        )

    dataset = Dataset.from_dict(processed_data)
    return dataset

def process_triplet_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer
) -> Dataset:
    def apply_chat_template(example):
        if "system" in example:
            prompt = example["system"] + "\n"
        else:
            prompt = ""

        example["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + example["prompt"]}],
            tokenize=False,
        )
        example["chosen"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["chosen"]}], tokenize=False
        )
        example["rejected"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["rejected"]}], tokenize=False
        )
        return example

    columns = set(dataset.features) - {"prompt", "rejected", "chosen"}
    dataset = dataset.map(apply_chat_template, remove_columns=list(columns))
    return dataset
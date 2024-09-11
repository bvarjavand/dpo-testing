from typing import Any, Dict, List
from datasets import DatasetDict, concatenate_datasets, load_dataset
from data_processors import get_dataset_format, process_conversation_dataset, process_triplet_dataset, TRIPLET_DATASET, CONVERSATION_DATASET

def load_sft_dataset(hparams: Dict[str, Any]) -> DatasetDict:
    dataset_name = hparams["dataset"]
    dataset_subsets = hparams["dataset_subsets"]
    dataset_list = []
    for subset_info in dataset_subsets:
        if "ratio" in subset_info:
            subset_str = f"{int(subset_info['ratio']*100)}%"
        elif "number_of_samples" in subset_info:
            subset_str = str(subset_info["number_of_samples"])
        else:
            raise RuntimeError(f"Unknown subset definition {subset_info}")
        dataset_subset = load_dataset(
            dataset_name, subset_info["subset"], split=f"train[:{subset_str}]"
        )
        dataset_list.append(dataset_subset)

    dataset = concatenate_datasets(dataset_list)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

def load_dpo_datasets(datasets: List[str], tokenizer) -> DatasetDict:
    dataset_list_validated = []
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        if isinstance(dataset, DatasetDict):
            dataset_list = [dataset[k] for k in dataset]
        else:
            dataset_list = [dataset]

        for ds in dataset_list:
            dataset_format = get_dataset_format(ds)
            if dataset_format == CONVERSATION_DATASET:
                ds = process_conversation_dataset(ds, tokenizer)
            elif dataset_format == TRIPLET_DATASET:
                ds = process_triplet_dataset(ds, tokenizer)

            dataset_list_validated.append(ds)

    dataset = concatenate_datasets(dataset_list_validated)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=False)
    return dataset
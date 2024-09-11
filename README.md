# Direct Preference Optimization (DPO) for Language Models

## Overview

This project implements Direct Preference Optimization (DPO) for fine-tuning language models, specifically designed to work with the Qwen-0.5B model. It uses the Determined AI framework for distributed training and experiment management.

DPO is a novel approach to fine-tuning language models that directly optimizes the model to align with human preferences, without the need for a complex reward model or reinforcement learning.

## Key Features

- Implements DPO for language model fine-tuning
- Uses the Qwen-0.5B model as a base
- Distributed training with Determined AI
- Flexible dataset loading and preprocessing
- Customizable reward model

## Project Structure

- `model_def.py`: The entry point for the Determined AI experiment
- `models.py`: Contains the reward model definition
- `data.py`: Handles dataset loading and preprocessing
- `trainer.py`: Implements the DPO training logic
- `config.yaml`: Configuration file for Determined AI experiments
- `evaluate.py`: Script for evaluating the trained model

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/dpo-language-model.git
   cd dpo-language-model
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv dpo_env
   source dpo_env/bin/activate  # On Windows, use `dpo_env\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dependencies

This project requires the following main dependencies:

- Python 3.7+
- PyTorch 1.12+
- Transformers library
- Determined AI

For a complete list of dependencies, refer to the `requirements.txt` file.

## Usage

1. Ensure you have Determined AI set up in your environment.

2. Modify the `config.yaml` file to adjust hyperparameters, dataset, and training settings as needed.

3. Run the experiment:
   ```
   det experiment create config.yaml .
   ```

### Usage Examples

1. Train with default settings:
   ```
   det experiment create config.yaml .
   ```

2. Train with a custom dataset:
   ```
   # Modify config.yaml to set dataset_name: "your_custom_dataset"
   det experiment create config.yaml .
   ```

3. Train with multiple GPUs:
   ```
   # Modify config.yaml to set slots_per_trial: 4
   det experiment create config.yaml .
   ```

4. Fine-tune a different base model:
   ```
   # Modify config.yaml to set model_name: "your_preferred_model"
   det experiment create config.yaml .
   ```

5. Evaluate the trained model:
   ```
   python evaluate.py --model_path /path/to/trained/model --dataset_name your_dataset
   ```

## How It Works

1. **Data Loading**: The script loads the specified dataset (default: tiny_shakespeare) and preprocesses it for the model.

2. **Model Initialization**: It initializes the Qwen-0.5B model and a simple reward model.

3. **DPO Training**: The core of the project is the DPO training loop, which:
   - Generates two sequences from the model: a "chosen" sequence and a "rejected" sequence
   - Computes rewards for both sequences using the reward model
   - Calculates the DPO loss, which encourages the model to assign higher probability to the chosen sequence
   - Updates the model parameters to minimize this loss

4. **Distributed Training**: Leverages Determined AI's distributed training capabilities to train across multiple GPUs efficiently.

5. **Evaluation**: Periodically evaluates the model's performance on a validation set.

## Running Tests

To run the tests for this project:

1. Ensure you're in the project root directory.

2. Run the following command:
   ```
   pytest test_dpo.py
   ```

This will execute all the tests defined in the `test_dpo.py` file, which include tests for model loading, tokenizer loading, reward model functionality, dataset loading, DPO trainer creation, and loss computation.

## Configuration

Key configuration options in `config.yaml`:

- `model_name`: The base model to use (default: "Qwen/Qwen-0.5B")
- `dataset_name`: The dataset to use for training (default: "tiny_shakespeare")
- `batch_size`: Batch size per GPU
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for the optimizer
- `beta`: The beta parameter for DPO loss calculation
- `max_length`: Maximum sequence length for tokenization
- `slots_per_trial`: Number of GPUs to use for distributed training

## Customization

- To use a different base model, modify the `model_name` in `config.yaml`.
- To use a custom dataset, update the `load_and_preprocess_dataset` function in `data.py`.
- To modify the reward model, edit the `RewardModel` class in `models.py`.

## DPOTrial Class

The `DPOTrial` class in `model_def.py` is the main entry point for the Determined AI experiment. It inherits from `PyTorchTrial` and implements the following key methods:

- `__init__`: Initializes the model, tokenizer, reward model, dataset, and optimizer
- `train_batch`: Performs a single training step on a batch of data
- `evaluate_batch`: Evaluates the model on a single batch of validation data
- `build_training_data_loader`: Creates the DataLoader for training data
- `build_validation_data_loader`: Creates the DataLoader for validation data

To use the `DPOTrial` class, ensure that your `config.yaml` file specifies it as the entrypoint:

## References
- Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv preprint 
arXiv:2305.18290.

# Persian QA Fine-tuning with LoRA for Gemma-3 270M

This project provides a modular and efficient pipeline to fine-tune the `google/gemma-3-270m` model for Persian Question Answering using the `SajjadAyoubi/persian_qa` dataset. It uses LoRA for parameter-efficient fine-tuning (PEFT) and is designed to produce a model suitable for a Retrieval-Augmented Generation (RAG) system.

The project includes scripts for training, evaluation, and exporting the model for deployment with NVIDIA Triton Inference Server.

## Project Structure

- `configs/`: Contains configuration files (e.g., hyperparameters).
- `src/`: Main source code, organized into modular components.
  - `config.py`: Loads and parses configuration.
  - `data_utils.py`: Handles dataset loading and preprocessing.
  - `model_utils.py`: Manages model and tokenizer loading, and LoRA setup.
  - `train.py`: The main script to run the training process.
  - `eval.py`: Standalone script for evaluating a trained model checkpoint.
  - `export.py`: Merges LoRA adapters with the base model for deployment.
  - `utils.py`: Utility functions for seeding, device management, etc.
- `scripts/`: Convenience shell scripts to run training and evaluation.
- `requirements.txt`: Python dependencies.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Login to Hugging Face Hub (for Gemma model access):**
    You need to accept the Gemma license on Hugging Face and create an access token with 'read' permissions.
    ```bash
    huggingface-cli login
    ```

## How to Run

### 1. Training

The training process is configured via `configs/default_config.yaml`. You can adjust hyperparameters like learning rate, batch size, and epochs in this file.

To start training, run the provided shell script:

```bash
bash scripts/run_train.sh
```
This will:
- Load the base model and tokenizer.
- Download and preprocess the dataset.
- Apply LoRA adapters to the model.
- Start fine-tuning using the `SFTTrainer`.
- Save model checkpoints (LoRA adapters) to the output directory specified in the config (`./outputs` by default).
### 2. Evaluation

To evaluate a saved checkpoint, use the `run_eval.sh` script. You'll need to update the script to point to the correct checkpoint directory (e.g., `./outputs/final_checkpoint`).

```bash
# Example: Edit run_eval.sh to set CHECKPOINT_PATH
bash scripts/run_eval.sh
```

The script will load the fine-tuned adapters, generate answers for the validation set, and compute Exact Match (EM) and F1 scores.

### 3. Exporting for Triton Deployment

For production deployment with Triton, you typically need a merged, standalone model, not just the LoRA adapters. The `export.py` script handles this.

**To merge the LoRA adapters and save the full model:**

```bash
python src/export.py \
    --config configs/default_config.yaml \
    --adapter_path ./outputs/final_checkpoint \
    --output_dir ./outputs/merged_model_for_triton
```

## Testing

The project includes comprehensive tests to validate the dataset loading, preprocessing, and code functionality before training.

To run the tests:

```bash
bash scripts/run_test.sh
```

The test suite includes:
- Configuration loading validation
- Dataset structure validation
- Prompt creation function testing
- Data quality checks
- Edge case handling

**Note:** The tests use mock data to validate code logic without requiring actual dataset downloads, making them fast and reliable for CI/CD pipelines.
```

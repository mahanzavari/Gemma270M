import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
import logging
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root) 

from src.config import load_config
from src.utils import set_seed, get_device
from src.data_utils import load_and_prepare_dataset
from src.model_utils import load_model_and_tokenizer, apply_lora
from src.callbacks import TrainingLoggerCallback, EarlyStoppingCallback
from src.metrics import compute_metrics, log_system_info

def main():
    """Main function to orchestrate the fine-tuning process."""
    parser = argparse.ArgumentParser(description="Run fine-tuning.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config.OUTPUT_DIR}/training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log system information
    log_system_info()

    # Setup
    set_seed(config.SEED)
    device = get_device()
    if device is None:
        raise ValueError("No valid device detected. Ensure CUDA is available or set to CPU.")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration loaded from: {args.config}")
    
    # Load model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer(config, device)
    
    # Apply LoRA
    lora_model = apply_lora(base_model, config.LORA)

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(config)

    # Training Arguments
    training_args = SFTConfig(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        optim=config.OPTIM,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        learning_rate=config.LEARNING_RATE,
        fp16=config.FP16,
        bf16=config.BF16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=config.WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        report_to="tensorboard",
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        max_length=config.MAX_INPUT_LENGTH,
        packing=False,  # Packing can be beneficial but requires careful data prep
        dataset_text_field="text",
    )

    # Initialize callbacks
    training_logger = TrainingLoggerCallback()
    early_stopping = EarlyStoppingCallback(
        patience=config.EARLY_STOPPING_PATIENCE if hasattr(config, 'EARLY_STOPPING_PATIENCE') else 3,
        min_delta=config.EARLY_STOPPING_MIN_DELTA if hasattr(config, 'EARLY_STOPPING_MIN_DELTA') else 0.001
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=None, 
        processing_class=tokenizer,
        callbacks=[training_logger, early_stopping],
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # Save final model adapters
    final_save_path = f"{config.OUTPUT_DIR}/final_checkpoint"
    trainer.save_model(final_save_path)
    logger.info(f"Final LoRA adapters saved to {final_save_path}")

if __name__ == "__main__":
    import argparse
    main()
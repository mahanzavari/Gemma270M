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

def main():
    """Main function to orchestrate the fine-tuning process."""
    parser = argparse.ArgumentParser(description="Run fine-tuning.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Setup
    set_seed(config.SEED)
    device = get_device()
    if device is None:
        raise ValueError("No valid device detected. Ensure CUDA is available or set to CPU.")
    
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

    # Initialize Trainer
    trainer = SFTTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_model.peft_config['default'],
        processing_class=tokenizer,
    )
    
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")

    # Save final model adapters
    final_save_path = f"{config.OUTPUT_DIR}/final_checkpoint"
    trainer.save_model(final_save_path)
    logging.info(f"Final LoRA adapters saved to {final_save_path}")

if __name__ == "__main__":
    import argparse
    main()
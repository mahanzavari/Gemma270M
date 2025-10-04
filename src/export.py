import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import logging
import os

from src.config import load_config

def merge_and_save_model(config, adapter_path: str, output_dir: str):
    """
    Merges LoRA adapters with the base model and saves the merged model
    in a format suitable for deployment.
    """
    logging.info(f"Loading base model: {config.MODEL_NAME}")

    # Load base model in a compatible format for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        return_dict=True,
        torch_dtype=torch.bfloat16 if config.BF16 else torch.float16,
        device_map="auto",
    )

    logging.info(f"Loading LoRA adapters from: {adapter_path}")
    # Load the PEFT model
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge the weights
    logging.info("Merging LoRA adapters into the base model...")
    merged_model = merged_model.merge_and_unload()
    logging.info("Merging complete.")

    # Load the tokenizer associated with the base model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save the merged model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving merged model and tokenizer to: {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logging.info("Export successful. The directory can now be used for deployment.")
    logging.info("For Triton, you can use this directory with the PyTorch backend.")
    logging.info("For ONNX, you would now load this model and convert it using transformers.onnx or torch.onnx.export.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and save the full model for deployment.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the config file.')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to the trained LoRA adapters checkpoint.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the merged model.')
    args = parser.parse_args()

    config = load_config(args.config)
    merge_and_save_model(config, args.adapter_path, args.output_dir)

if __name__ == "__main__":
    main()
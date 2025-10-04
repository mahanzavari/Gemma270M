import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
import string
import argparse
import logging
from sklearn.metrics import f1_score

from src.config import load_config
from src.utils import get_device

def normalize_text(s: str) -> str:
    """Lowercasing, remove punctuation, and normalize whitespace."""
    s = s.lower()
    s = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_metrics(predictions, references):
    """Computes Exact Match and F1 score."""
    normalized_predictions = [normalize_text(p) for p in predictions]
    normalized_references = [normalize_text(r) for r in references]
    
    # Exact Match
    em_score = sum(p == r for p, r in zip(normalized_predictions, normalized_references)) / len(predictions)
    
    # Token-level F1
    f1_scores = []
    for pred, ref in zip(normalized_predictions, normalized_references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        common_tokens = set(pred_tokens) & set(ref_tokens)
        
        if not common_tokens:
            f1_scores.append(0.0)
            continue
            
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    return {"exact_match": em_score, "f1_score": avg_f1}


def generate_answers(model, tokenizer, dataset, config, device):
    """Generate answers for the evaluation dataset."""
    predictions, references = [], []
    
    prompt_template = "پرسش: {}\nمتن: {}\nجواب کوتاه:"
    
    for sample in tqdm(dataset, desc="Generating answers"):
        question = sample["question"]
        context = sample["context"]
        reference_answer = sample["answers"]["text"][0]

        prompt = prompt_template.format(question, context)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_TARGET_LENGTH,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the generated text, excluding the input prompt
        prediction = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        
        predictions.append(prediction)
        references.append(reference_answer)
        
    return predictions, references


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the config file.')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to the trained LoRA adapters.')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Load base model and tokenizer
    logging.info(f"Loading base model '{config.MODEL_NAME}'")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16 if config.BF16 else torch.float16,
        device_map={"": device.index} if device.type == "cuda" else "auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapters
    logging.info(f"Loading LoRA adapters from '{args.adapter_path}'")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.eval()

    # Load validation dataset
    logging.info("Loading validation data...")
    dataset = load_dataset(config.DATASET_NAME, split='train')
    # Use the same split logic as in training for consistency
    validation_dataset = dataset.train_test_split(test_size=config.VAL_SET_SIZE, seed=config.SEED)['test']

    # Generate and evaluate
    predictions, references = generate_answers(model, tokenizer, validation_dataset, config, device)
    metrics = compute_metrics(predictions, references)

    logging.info(f"Evaluation Metrics for checkpoint '{args.adapter_path}':")
    logging.info(f"  - Exact Match: {metrics['exact_match']:.4f}")
    logging.info(f"  - F1 Score: {metrics['f1_score']:.4f}")

    # Print a few examples
    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(predictions))):
        print(f"Question: {validation_dataset[i]['question']}")
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 20)


if __name__ == "__main__":
    main()
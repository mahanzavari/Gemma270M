import torch
import numpy as np
import logging
from typing import Dict, Any

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation during training.

    Args:
        eval_pred: EvalPrediction object containing predictions and label_ids

    Returns:
        Dict containing computed metrics
    """
    predictions, labels = eval_pred

    # For causal language modeling, predictions are logits
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Shift predictions and labels for causal LM
    shift_logits = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Calculate perplexity
    perplexity = torch.exp(loss).item()

    # Calculate accuracy (optional, for classification-like tasks)
    preds = torch.argmax(shift_logits, dim=-1)
    correct = (preds == shift_labels).float()
    # Mask out padding tokens (assuming pad_token_id = 0)
    mask = (shift_labels != 0).float()
    accuracy = (correct * mask).sum() / mask.sum() if mask.sum() > 0 else 0.0

    return {
        'eval_loss': loss.item(),
        'eval_perplexity': perplexity,
        'eval_accuracy': accuracy.item() if torch.is_tensor(accuracy) else accuracy,
    }

def log_training_metrics(trainer, step_metrics: Dict[str, Any]):
    """
    Log additional training metrics.

    Args:
        trainer: The trainer object
        step_metrics: Dictionary of metrics from the current step
    """
    if step_metrics:
        logging.info("Step Metrics:")
        for key, value in step_metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {key}: {value:.4f}")
            else:
                logging.info(f"  {key}: {value}")

def get_model_memory_usage():
    """
    Get current GPU memory usage if available.

    Returns:
        Dict with memory usage information
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return {
            'gpu_memory_allocated_gb': round(memory_allocated, 2),
            'gpu_memory_reserved_gb': round(memory_reserved, 2),
        }
    return {}

def log_system_info():
    """Log system and training environment information."""
    logging.info("System Information:")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logging.info(f"CPU count: {torch.get_num_threads()}")
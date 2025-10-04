import random
import numpy as np
import torch 
import logging

logging.basicConfig(level= logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def get_device() -> torch.device:
    """Returns the appropriate device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


from datasets import load_dataset, Dataset, DatasetDict
import logging

def create_prompt(sample: dict) -> str:
    """
    Creates a formatted string for a given sample, suitable for SFTTrainer.
    The format includes the question, context, and the answer.
    """
    # For the new dataset format
    question = sample.get("instruction", "").strip() if isinstance(sample.get("instruction"), str) else ""
    context = sample.get("input", "").strip() if isinstance(sample.get("input"), str) else ""
    answer = sample.get("output", "").strip() if isinstance(sample.get("output"), str) else ""
    
    if not question or not answer:
        logging.warning(f"Sample has missing question or answer. Skipping. Question: {question[:50]}...")
        return None
    
    # This full string will be used by SFTTrainer for training
    if context:
        return f"پرسش: {question}\nمتن: {context}\nجواب کوتاه: {answer}"
    else:
        return f"پرسش: {question}\nجواب کوتاه: {answer}"

def load_and_prepare_dataset(config: object) -> DatasetDict:
    """
    Loads the dataset from Hugging Face, applies the prompt format,
    and splits it into training and validation sets.
    """
    logging.info(f"Loading dataset '{config.DATASET_NAME}' from Hugging Face Hub...")
    dataset = load_dataset(config.DATASET_NAME, split='train')

    # Filter out samples with no answers or questions
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: isinstance(x.get("instruction"), str) and x.get("instruction") and isinstance(x.get("output"), str) and x.get("output"))
    if len(dataset) < original_size:
        logging.info(f"Filtered out {original_size - len(dataset)} samples with missing or invalid instruction or output.")
    
    # Create the formatted text column required by SFTTrainer
    dataset = dataset.map(lambda sample: {"text": create_prompt(sample)}, remove_columns=list(dataset.features))

    # Split the dataset into training and validation sets
    logging.info(f"Splitting dataset into train and validation sets (validation size: {config.VAL_SET_SIZE}).")
    dataset_dict = dataset.train_test_split(test_size=config.VAL_SET_SIZE, seed=config.SEED)
    
    logging.info(f"Dataset prepared: {len(dataset_dict['train'])} train samples, {len(dataset_dict['test'])} validation samples.")
    
    # Rename 'test' split to 'validation' for clarity
    dataset_dict['validation'] = dataset_dict.pop('test')
    
    return dataset_dict
from datasets import load_dataset, Dataset, DatasetDict
import logging

def create_prompt(sample: dict) -> str:
    """
    Creates a formatted string for a given sample, suitable for SFTTrainer.
    The format includes the question, context, and the answer.
    """
    question = sample.get("question", "").strip()
    context = sample.get("context", "").strip()
    
    # Use the first answer if available, otherwise handle missing answers
    answers = sample.get("answers", {}).get("text", [])
    if not answers:
        logging.warning(f"Sample has no answers. Skipping. Question: {question[:50]}...")
        return None # This will be filtered out later
    
    answer = answers[0].strip()
    
    # This full string will be used by SFTTrainer for training
    return f"پرسش: {question}\nمتن: {context}\nجواب کوتاه: {answer}"

def load_and_prepare_dataset(config: object) -> DatasetDict:
    """
    Loads the dataset from Hugging Face, applies the prompt format,
    and splits it into training and validation sets.
    """
    logging.info(f"Loading dataset '{config.DATASET_NAME}' from Hugging Face Hub...")
    dataset = load_dataset(config.DATASET_NAME, split='train')

    # Filter out samples with no answers
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: x.get("answers") and x["answers"].get("text"))
    if len(dataset) < original_size:
        logging.info(f"Filtered out {original_size - len(dataset)} samples with no answers.")
    
    # Create the formatted text column required by SFTTrainer
    dataset = dataset.map(lambda sample: {"text": create_prompt(sample)}, remove_columns=list(dataset.features))

    # Split the dataset into training and validation sets
    logging.info(f"Splitting dataset into train and validation sets (validation size: {config.VAL_SET_SIZE}).")
    dataset_dict = dataset.train_test_split(test_size=config.VAL_SET_SIZE, seed=config.SEED)
    
    logging.info(f"Dataset prepared: {len(dataset_dict['train'])} train samples, {len(dataset_dict['test'])} validation samples.")
    
    # Rename 'test' split to 'validation' for clarity
    dataset_dict['validation'] = dataset_dict.pop('test')
    
    return dataset_dict
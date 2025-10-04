import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

def load_model_and_tokenizer(config: object, device: torch.device):
    """
    Loads the base model and tokenizer from Hugging Face Hub.
    Applies quantization if specified.
    """
    logging.info(f"Loading base model: {config.MODEL_NAME}")

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.BF16 else torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Determine torch dtype
    torch_dtype = torch.bfloat16 if config.BF16 else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map={"": device.index} if device.type == "cuda" else "auto",
        trust_remote_code=True,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    logging.info(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer `pad_token` set to `eos_token`.")

    return model, tokenizer

def apply_lora(model, lora_config: object):
    """
    Applies LoRA configuration to the model.
    """
    logging.info("Applying LoRA configuration...")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )
    
    peft_model = get_peft_model(model, peft_config)
    
    logging.info("LoRA applied. Trainable parameters:")
    peft_model.print_trainable_parameters()
    
    return peft_model
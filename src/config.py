import yaml
from types import SimpleNamespace
import argparse

def load_config(config_path: str) -> SimpleNamespace:
    """Loads a YAML configuration file into a SimpleNamespace object."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    def dict_to_namespace(d):
        """"""
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    
    return dict_to_namespace(config_dict)

def get_config() -> SimpleNamespace:
    """Parses command-line arguments to get config path and loads it."""
    parser = argparse.ArgumentParser(description="Fine-tuning script for Persian QA.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    return load_config(args.config)

if __name__ == '__main__':
    # Example usage
    config = get_config()
    print("Model Name:", config.MODEL_NAME)
    print("LoRA R:", config.LORA.r)
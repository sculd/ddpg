import os
import yaml
import json
from typing import Dict, Any


def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_name: Name of the config file (without extension) or full path

    Returns:
        Dictionary containing configuration parameters
    """
    # Check if config_name is a full path
    if os.path.exists(config_name):
        config_path = config_name
    else:
        # Try to find the config file in configs directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        configs_dir = os.path.join(base_dir, "configs")

        # Try YAML first, then JSON
        yaml_path = os.path.join(configs_dir, f"{config_name}.yaml")
        json_path = os.path.join(configs_dir, f"{config_name}.json")

        if os.path.exists(yaml_path):
            config_path = yaml_path
        elif os.path.exists(json_path):
            config_path = json_path
        else:
            raise FileNotFoundError(
                f"Config file '{config_name}' not found. "
                f"Looked for {yaml_path} and {json_path}"
            )

    # Load the config file
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")

    return config


def merge_with_args(config: Dict[str, Any], args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration with command-line arguments.
    Command-line arguments take precedence over config file values.

    Args:
        config: Configuration dictionary from file
        args_dict: Dictionary of command-line arguments

    Returns:
        Merged configuration dictionary
    """
    merged = config.copy()

    # Update with non-None command-line arguments
    for key, value in args_dict.items():
        if value is not None:
            merged[key] = value

    return merged
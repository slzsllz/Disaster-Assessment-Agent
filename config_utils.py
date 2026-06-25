"""
Utility module for loading environment variables and handling API credentials safely.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


def load_env():
    """
    Load environment variables from .env file if it exists.
    Looks for .env file in the project root directory.
    """
    # Try to find .env file in current directory and parent directories
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        env_file = parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            print(f"Loaded environment variables from: {env_file}")
            return True

    # If no .env file found, just load from environment
    print("No .env file found. Using system environment variables.")
    return False


def get_api_credentials(model_name: str) -> Dict[str, str]:
    """
    Get API credentials for a specific model from environment variables.

    Args:
        model_name: Name of the model (e.g., 'openai', 'deepseek', 'kimi', 'gemini', 'glm')

    Returns:
        Dictionary with 'api_key' and 'base_url'

    Raises:
        ValueError: If required environment variables are not set
    """
    # Map model names to environment variable prefixes
    model_prefix_map = {
        'openai': 'OPENAI',
        'gpt': 'OPENAI',
        'gpt-5': 'OPENAI',
        'gpt-4': 'OPENAI',
        'deepseek': 'DEEPSEEK',
        'kimi': 'KIMI',
        'kimi_k2': 'KIMI',
        'gemini': 'GEMINI',
        'glm': 'GLM',
        'glm-4.5': 'GLM',
    }

    prefix = model_prefix_map.get(model_name.lower(), model_name.upper())

    api_key_var = f'{prefix}_API_KEY'
    base_url_var = f'{prefix}_BASE_URL'

    api_key = os.getenv(api_key_var)
    base_url = os.getenv(base_url_var)

    if not api_key:
        raise ValueError(
            f"API key not found! Please set {api_key_var} environment variable.\n"
            f"You can add it to a .env file in the project root directory."
        )

    if not base_url:
        raise ValueError(
            f"Base URL not found! Please set {base_url_var} environment variable.\n"
            f"You can add it to a .env file in the project root directory."
        )

    return {
        'api_key': api_key,
        'base_url': base_url
    }


def load_config_with_env(config_path: str) -> Dict[str, Any]:
    """
    Load a JSON config file and substitute environment variable placeholders.

    Supports placeholders in the format: ${ENV_VAR_NAME}

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Configuration dictionary with environment variables substituted
    """
    with open(config_path, 'r') as f:
        config_str = f.read()

    # Replace environment variable placeholders
    import re
    def replace_env_var(match):
        env_var = match.group(1)
        value = os.getenv(env_var)
        if value is None:
            raise ValueError(
                f"Environment variable {env_var} not found!\n"
                f"Please set it in your .env file or system environment."
            )
        return value

    config_str = re.sub(r'\$\{([^}]+)\}', replace_env_var, config_str)

    return json.loads(config_str)


def create_config_with_credentials(template_path: str, output_path: str, model_name: str):
    """
    Create a config file from a template, filling in API credentials from environment.

    Args:
        template_path: Path to the template config file
        output_path: Path where the final config should be written
        model_name: Name of the model to get credentials for
    """
    # Load environment variables
    load_env()

    # Get credentials
    creds = get_api_credentials(model_name)

    # Load template
    with open(template_path, 'r') as f:
        config = json.load(f)

    # Update credentials in config
    for model in config.get('models', []):
        if 'api_key' in model:
            model['api_key'] = creds['api_key']
        if 'client_args' in model and 'base_url' in model['client_args']:
            model['client_args']['base_url'] = creds['base_url']

    # Write output
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Created config file: {output_path}")


if __name__ == "__main__":
    # Example usage
    load_env()

    try:
        creds = get_api_credentials('openai')
        print(f"Successfully loaded OpenAI credentials")
        print(f"API Key: {creds['api_key'][:20]}...")
        print(f"Base URL: {creds['base_url']}")
    except ValueError as e:
        print(f"Error: {e}")

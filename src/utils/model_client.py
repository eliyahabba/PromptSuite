"""
Client for interacting with language models.
"""
import os
from typing import List, Dict
from together import Together

import together
from dotenv import load_dotenv

from src.utils.constants import DEFAULT_MODEL

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize the Together client only if API key is available
client = None
if API_KEY:
    together.api_key = API_KEY
    client = Together()


def get_model_response(messages: List[Dict[str, str]], model_name: str = DEFAULT_MODEL) -> str:
    """
    Get a response from the language model.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model_name: Name of the model to use (defaults to the value in constants)

    Returns:
        The model's response text
    """
    if not client:
        raise ValueError("Together client not initialized. Please set TOGETHER_API_KEY environment variable.")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    return response.choices[0].message.content


def get_completion(prompt: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Get a completion from the language model using a simple prompt.
    
    Args:
        prompt: The prompt text
        model_name: Name of the model to use
        
    Returns:
        The model's response text
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    return get_model_response(messages, model_name)


def get_completion_with_key(prompt: str, api_key: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Get a completion from the language model using a simple prompt with provided API key.
    
    Args:
        prompt: The prompt text
        api_key: API key for the Together service
        model_name: Name of the model to use
        
    Returns:
        The model's response text
    """
    # Create a temporary client with the provided API key
    temp_client = Together(api_key=api_key)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = temp_client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Test the client
    test_prompt = "What is the capital of France?"
    print(f"Prompt: {test_prompt}")

    if client:
        response = get_completion(test_prompt)
        print(f"Response: {response}")
    else:
        print("No API key available for testing")

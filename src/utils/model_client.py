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

# Initialize the Together client
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


if __name__ == "__main__":
    # Test the client
    test_prompt = "What is the capital of France?"
    print(f"Prompt: {test_prompt}")

    response = get_completion(test_prompt)
    print(f"Response: {response}")

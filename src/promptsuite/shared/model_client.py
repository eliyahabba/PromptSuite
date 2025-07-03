"""
Client for interacting with language models.
"""
import os
from typing import List, Dict, Optional
from together import Together
from openai import OpenAI
from dotenv import load_dotenv

from promptsuite.shared.constants import GenerationDefaults
from promptsuite.core.exceptions import APIKeyMissingError

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the Together client only if API key is available
# together_client = None
# if TOGETHER_API_KEY:
#     together_client = Together(api_key=TOGETHER_API_KEY)

# Initialize OpenAI client if API key is available
# openai_client = None
# if OPENAI_API_KEY:
#     openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_model_response(messages: List[Dict[str, str]],
                       model_name: str = GenerationDefaults.MODEL_NAME,
                       max_tokens: Optional[int] = None,
                       platform: str = "TogetherAI",
                       temperature: float = 0.0,
                       api_key: Optional[str] = None) -> str:
    """
    Get a response from the language model.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model_name: Name of the model to use (defaults to the value in constants)
        max_tokens: Maximum number of tokens for the response
        platform: Platform to use ("TogetherAI" or "OpenAI")
        temperature: Temperature for response generation (0.0 = deterministic, 1.0 = creative)
        api_key: Optional API key to use for the platform

    Returns:
        The model's response text
    """
    if platform == "TogetherAI":
        return _get_together_response(messages, model_name, max_tokens, temperature, api_key)
    elif platform == "OpenAI":
        return _get_openai_response(messages, model_name, max_tokens, temperature, api_key)
    else:
        raise ValueError(f"Unsupported platform: {platform}. Supported platforms: TogetherAI, OpenAI")


def _get_together_response(messages: List[Dict[str, str]], model_name: str, max_tokens: Optional[int] = None,
                           temperature: float = 0.0, api_key: Optional[str] = None) -> str:
    """Get response from TogetherAI."""
    current_api_key = api_key if api_key is not None else TOGETHER_API_KEY
    if not current_api_key:
        raise APIKeyMissingError("TogetherAI")

    client = Together(api_key=current_api_key)

    # Prepare parameters
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    # Add max_tokens if provided
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


def _get_openai_response(messages: List[Dict[str, str]], model_name: str, max_tokens: Optional[int] = None,
                         temperature: float = 0.0, api_key: Optional[str] = None) -> str:
    """Get response from OpenAI."""
    current_api_key = api_key if api_key is not None else OPENAI_API_KEY
    if not current_api_key:
        raise APIKeyMissingError("OpenAI")

    client = OpenAI(api_key=current_api_key)

    # Prepare parameters
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    # Add max_tokens if provided
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


def get_completion(prompt: str,
                   model_name: str = GenerationDefaults.MODEL_NAME,
                   max_tokens: Optional[int] = None,
                   platform: str = "TogetherAI",
                   api_key: Optional[str] = None) -> str:
    """
    Get a completion from the language model using a simple prompt.
    
    Args:
        prompt: The prompt text
        model_name: Name of the model to use
        max_tokens: Maximum number of tokens for the response
        platform: Platform to use ("TogetherAI" or "OpenAI")
        api_key: Optional API key to use for the platform
        
    Returns:
        The model's response text
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    return get_model_response(messages, model_name, max_tokens, platform, api_key=api_key)


# Removed get_completion_with_key as get_completion now handles api_key
# def get_completion_with_key(prompt: str,
#                            api_key: str,
#                            model_name: str = GenerationDefaults.MODEL_NAME,
#                            max_tokens: Optional[int] = None,
#                            platform: str = "TogetherAI") -> str:
#     """
#     Get a completion from the language model using a simple prompt with provided API key.

#     Args:
#         prompt: The prompt text
#         api_key: API key for the service
#         model_name: Name of the model to use
#         max_tokens: Maximum number of tokens for the response
#         platform: Platform to use ("TogetherAI" or "OpenAI")

#     Returns:
#         The model's response text
#     """
#     if platform == "TogetherAI":
#         # Create a temporary Together client with the provided API key
#         temp_client = Together(api_key=api_key)

#         messages = [
#             {"role": "user", "content": prompt}
#         ]

#         # Prepare parameters
#         params = {
#             "model": model_name,
#             "messages": messages,
#         "temperature": 0,
#         }

#         # Add max_tokens if provided
#         if max_tokens is not None:
#             params["max_tokens"] = max_tokens

#         response = temp_client.chat.completions.create(**params)
#         return response.choices[0].message.content

#     elif platform == "OpenAI":
#         # Create a temporary OpenAI client with the provided API key
#         client = OpenAI(api_key=api_key)

#         # Prepare the messages
#         messages = [
#             {"role": "user", "content": prompt}
#         ]

#         # Prepare parameters
#         params = {
#             "model": model_name,
#             "messages": messages,
#         "temperature": 0,
#         }

#         # Add max_tokens if provided
#         if max_tokens is not None:
#             params["max_tokens"] = max_tokens

#         # Create chat completion
#         response = client.chat.completions.create(**params)

#         # Return the response content
#         return response.choices[0].message.content

#     else:
#         raise ValueError(f"Unsupported platform: {platform}")


if __name__ == "__main__":
    # Test the client
    test_prompt = "What is the capital of France?"
    print(f"Prompt: {test_prompt}")

    # Test with TogetherAI using environment variable
    try:
        response = get_completion(test_prompt, platform="TogetherAI")
        print(f"TogetherAI Response (from env): {response}")
    except APIKeyMissingError:
        print("No TogetherAI API key available in environment for testing")

    # Test with OpenAI using environment variable
    try:
        response = get_completion(test_prompt, platform="OpenAI", model_name="gpt-4o-mini")
        print(f"OpenAI Response (from env): {response}")
    except APIKeyMissingError:
        print("No OpenAI API key available in environment for testing")
    
    # Example of testing with a provided API key (replace 'YOUR_TEMP_KEY' with an actual key for a real test)
    try:
        temp_api_key = "YOUR_TEMP_TOGETHER_API_KEY" # Placeholder - replace with actual key if testing
        if temp_api_key != "YOUR_TEMP_TOGETHER_API_KEY":
            response = get_completion(test_prompt, platform="TogetherAI", api_key=temp_api_key)
            print(f"TogetherAI Response (with provided key): {response}")
        else:
            print("Skipping TogetherAI test with provided key (placeholder used)")
    except APIKeyMissingError:
        print("Could not test TogetherAI with provided key")

    try:
        temp_api_key = "YOUR_TEMP_OPENAI_API_KEY" # Placeholder - replace with actual key if testing
        if temp_api_key != "YOUR_TEMP_OPENAI_API_KEY":
            response = get_completion(test_prompt, platform="OpenAI", model_name="gpt-4o-mini", api_key=temp_api_key)
            print(f"OpenAI Response (with provided key): {response}")
        else:
            print("Skipping OpenAI test with provided key (placeholder used)")
    except APIKeyMissingError:
        print("Could not test OpenAI with provided key")

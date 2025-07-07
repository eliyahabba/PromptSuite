"""
Client for interacting with language models with extensible platform support.
"""
import os
from abc import abstractmethod
from typing import List, Dict, Optional, Protocol

from dotenv import load_dotenv

from promptsuite.core.exceptions import APIKeyMissingError
from promptsuite.shared.constants import GenerationDefaults

# Load environment variables from .env file
load_dotenv()


class ModelProvider(Protocol):
    """Protocol for model providers."""

    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        """Get response from the model provider."""
        pass


class TogetherAIProvider:
    """Provider for TogetherAI platform."""

    def __init__(self, api_key: str):
        try:
            from together import Together
            self.client = Together(api_key=api_key)
        except ImportError:
            raise ImportError(
                "together package is required for TogetherAI provider. Install with: pip install together")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class OpenAIProvider:
    """Provider for OpenAI platform."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider. Install with: pip install openai")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class AnthropicProvider:
    """Provider for Anthropic platform."""

    def __init__(self, api_key: str):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. Install with: pip install anthropic")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        # Convert OpenAI format to Anthropic format
        system_message = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        kwargs = {
            "model": model_name,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
        }

        if system_message:
            kwargs["system"] = system_message

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class GoogleProvider:
    """Provider for Google (Gemini) platform."""

    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Google provider. Install with: pip install google-generativeai")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        model = self.genai.GenerativeModel(model_name)

        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")

        prompt = "\n".join(prompt_parts)

        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text


class CohereProvider:
    """Provider for Cohere platform."""

    def __init__(self, api_key: str):
        try:
            import cohere
            self.client = cohere.Client(api_key)
        except ImportError:
            raise ImportError("cohere package is required for Cohere provider. Install with: pip install cohere")

    def get_response(self, messages: List[Dict[str, str]], model_name: str,
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        # Convert to Cohere chat format
        chat_history = []
        message = ""

        for msg in messages:
            if msg["role"] == "system":
                # Add system message as preamble
                message = f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                if chat_history:
                    chat_history.append({"role": "USER", "message": msg["content"]})
                else:
                    message += msg["content"]
            elif msg["role"] == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg["content"]})

        kwargs = {
            "model": model_name,
            "message": message,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if chat_history:
            kwargs["chat_history"] = chat_history

        response = self.client.chat(**kwargs)
        return response.text


# Platform registry
PLATFORM_PROVIDERS = {
    "TogetherAI": TogetherAIProvider,
    "OpenAI": OpenAIProvider,
    "Anthropic": AnthropicProvider,
    "Google": GoogleProvider,
    "Cohere": CohereProvider,
}

# Environment variable mapping
PLATFORM_ENV_VARS = {
    "TogetherAI": "TOGETHER_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "Cohere": "COHERE_API_KEY",
}


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
        platform: Platform to use (supported: TogetherAI, OpenAI, Anthropic, Google, Cohere)
        temperature: Temperature for response generation (0.0 = deterministic, 1.0 = creative)
        api_key: Optional API key to use for the platform

    Returns:
        The model's response text
    """
    if platform not in PLATFORM_PROVIDERS:
        supported_platforms = list(PLATFORM_PROVIDERS.keys())
        raise ValueError(f"Unsupported platform: {platform}. Supported platforms: {supported_platforms}")

    # Get API key
    current_api_key = api_key if api_key is not None else os.getenv(PLATFORM_ENV_VARS[platform])
    if not current_api_key:
        raise APIKeyMissingError(platform)

    # Create provider and get response
    provider_class = PLATFORM_PROVIDERS[platform]
    try:
        provider = provider_class(current_api_key)
        return provider.get_response(messages, model_name, max_tokens, temperature)
    except ImportError as e:
        raise ImportError(f"Failed to initialize {platform} provider: {e}")
    except Exception as e:
        raise RuntimeError(f"Error getting response from {platform}: {e}")


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
        platform: Platform to use (supported: TogetherAI, OpenAI, Anthropic, Google, Cohere)
        api_key: Optional API key to use for the platform
        
    Returns:
        The model's response text
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    return get_model_response(messages, model_name, max_tokens, platform, api_key=api_key)


def get_supported_platforms() -> List[str]:
    """Get list of supported platforms."""
    return list(PLATFORM_PROVIDERS.keys())


def is_platform_available(platform: str) -> bool:
    """Check if a platform is available (has required dependencies)."""
    if platform not in PLATFORM_PROVIDERS:
        return False

    try:
        # Try to import required dependencies for each platform
        if platform == "TogetherAI":
            import together
        elif platform == "OpenAI":
            import openai
        elif platform == "Anthropic":
            import anthropic
        elif platform == "Google":
            import google.generativeai
        elif platform == "Cohere":
            import cohere
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Test the client
    test_prompt = "What is the capital of France?"
    print(f"Prompt: {test_prompt}")
    print(f"Supported platforms: {get_supported_platforms()}")

    for platform in get_supported_platforms():
        if is_platform_available(platform):
            print(f"✅ {platform} is available")
            try:
                # Only test if we have an API key
                env_var = PLATFORM_ENV_VARS[platform]
                if os.getenv(env_var):
                    response = get_completion(test_prompt, platform=platform)
                    print(f"{platform} Response: {response[:100]}...")
                else:
                    print(f"   (No API key found for {env_var})")
            except Exception as e:
                print(f"   Error testing {platform}: {e}")
        else:
            print(f"❌ {platform} is not available (missing dependencies)")

"""
Provider configuration and models for LLM providers.

This module defines the supported providers and their configurations,
allowing for flexible, extensible provider management.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Provider(str, Enum):
    """
    Supported LLM providers.

    Each provider corresponds to a LangChain integration:
    - OPENAI: OpenAI API models (gpt-4, gpt-3.5-turbo, etc.)
    - ANTHROPIC: Anthropic API models (claude-3, claude-opus, etc.)
    - OLLAMA: Local LLM models via Ollama
    - GOOGLE: Google Generative AI models
    """
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    OLLAMA = 'ollama'
    GOOGLE = 'google'


@dataclass
class ProviderConfig:
    """
    Base configuration for all LLM providers.

    Attributes:
        api_key: API key for the provider. If None, will attempt to load from environment variables.
        temperature: Model temperature (0-2). Controls randomness of responses.
                    Lower values (closer to 0) are more deterministic.
                    Defaults to 0 for consistent behavior.
        max_tokens: Maximum number of tokens in the response. Defaults to 8000.
        streaming: Whether to stream the response. Defaults to False.
    """
    api_key: Optional[str] = None
    temperature: int = 0
    max_tokens: int = 8000
    streaming: bool = False


@dataclass
class OllamaConfig(ProviderConfig):
    """
    Ollama-specific configuration for local LLM deployment.

    Attributes:
        base_url: URL of the Ollama server. Defaults to localhost:11434.
        use_proxy: Whether to use a proxy for Ollama connections. Defaults to False.
        num_ctx: Context window size for the model. Defaults to 10000.
        format: Output format for responses. Defaults to 'json'.
    """
    base_url: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_URL", "http://localhost:11434"))
    use_proxy: bool = field(default_factory=lambda: bool(os.getenv("HTTP_PROXY", False)))
    num_ctx: int = 10000
    format: str = 'json'


@dataclass
class GoogleConfig(ProviderConfig):
    """
    Google Generative AI specific configuration.

    Attributes:
        project: Optional Google Cloud project ID.
    """
    project: Optional[str] = None


# Default configurations for each provider
DEFAULT_CONFIGS = {
    Provider.OPENAI: ProviderConfig(),
    Provider.ANTHROPIC: ProviderConfig(),
    Provider.OLLAMA: OllamaConfig(),
    Provider.GOOGLE: GoogleConfig(),
}

"""
Provider handlers for LLM instantiation.

This module implements a factory pattern for creating LLM instances from different
providers. Each handler encapsulates provider-specific logic, making it easy to add
new providers without modifying the core agent code.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI

from ..connectors.ollama.proxy_ollama import ProxyOllama
from .provider_config import Provider, ProviderConfig, OllamaConfig, GoogleConfig, MistralConfig


class ProviderHandler(ABC):
    """
    Abstract base class for LLM provider handlers.

    Each concrete handler implements provider-specific logic for instantiating
    and configuring LLM models.
    """

    @abstractmethod
    def get_model(self, model: str, config: ProviderConfig) -> Any:
        """
        Get a configured LLM model instance.

        Args:
            model: Model identifier/name specific to the provider.
            config: Provider configuration containing API keys, temperature, etc.

        Returns:
            Configured LLM instance (ChatOpenAI, ChatAnthropic, etc.).
        """
        pass


class AnthropicHandler(ProviderHandler):
    """
    Handler for Anthropic Claude models.

    Supports Claude models through the Anthropic API.
    """

    def get_model(self, model: str, config: ProviderConfig) -> ChatAnthropic:
        """
        Get a configured ChatAnthropic instance.

        Args:
            model: Model name (e.g., 'claude-3-opus', 'claude-3-sonnet').
            config: Anthropic configuration with API key and parameters.

        Returns:
            Configured ChatAnthropic instance.
        """
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set"
            )

        return ChatAnthropic(
            model=model,
            temperature=config.temperature,
            streaming=config.streaming,
            api_key=api_key,
            max_tokens=config.max_tokens,
        )


class OpenAIHandler(ProviderHandler):
    """
    Handler for OpenAI models.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """

    def get_model(self, model: str, config: ProviderConfig) -> ChatOpenAI:
        """
        Get a configured ChatOpenAI instance.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo').
            config: OpenAI configuration with API key and parameters.

        Returns:
            Configured ChatOpenAI instance.
        """
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
            )

        return ChatOpenAI(
            model=model,
            temperature=config.temperature,
            streaming=config.streaming,
            api_key=api_key,
            max_tokens=config.max_tokens,
        )


class OllamaHandler(ProviderHandler):
    """
    Handler for local Ollama models.

    Supports local LLM deployment via Ollama with optional proxy support.
    """

    def get_model(self, model: str, config: OllamaConfig) -> ChatOllama | ProxyOllama:
        """
        Get a configured ChatOllama or ProxyOllama instance.

        Args:
            model: Model name (e.g., 'llama2', 'mistral', 'neural-chat').
            config: Ollama configuration with base URL, proxy settings, etc.

        Returns:
            Configured ChatOllama or ProxyOllama instance depending on proxy setting.

        Raises:
            ValueError: If config is not an OllamaConfig instance.
        """
        if not isinstance(config, OllamaConfig):
            raise ValueError("Ollama requires OllamaConfig configuration")

        # Use ProxyOllama if proxy is enabled
        if config.use_proxy:
            return ProxyOllama(
                model=model,
                temperature=config.temperature,
                num_ctx=config.num_ctx,
                base_url=config.base_url,
                streaming=config.streaming,
                format=config.format,
                disable_streaming=True,
            )

        # Use standard ChatOllama for direct connections
        return ChatOllama(
            model=model,
            temperature=config.temperature,
            num_ctx=config.num_ctx,
            base_url=config.base_url,
            streaming=config.streaming,
            format=config.format,
        )


class GoogleHandler(ProviderHandler):
    """
    Handler for Google Generative AI models.

    Supports Google's Gemini and other generative models.
    """

    def get_model(self, model: str, config: GoogleConfig) -> ChatGoogleGenerativeAI:
        """
        Get a configured ChatGoogleGenerativeAI instance.

        Args:
            model: Model name (e.g., 'gemini-pro', 'gemini-pro-vision').
            config: Google configuration with API key and optional project ID.

        Returns:
            Configured ChatGoogleGenerativeAI instance.
        """
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not provided and GOOGLE_API_KEY environment variable not set"
            )

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=config.temperature,
            streaming=config.streaming,
            api_key=api_key,
        )


class MistralHandler(ProviderHandler):
    """
    Handler for Mistral AI models.

    Supports Mistral models through the Mistral AI API.
    """

    def get_model(self, model: str, config: MistralConfig) -> ChatMistralAI:
        """
        Get a configured ChatMistralAI instance.

        Args:
            model: Model name (e.g., 'mistral-7b', 'mistral-large').
            config: Mistral configuration with API key and optional endpoint.

        Returns:
            Configured ChatMistralAI instance.
        """
        api_key = config.api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "Mistral API key not provided and MISTRAL_API_KEY environment variable not set"
            )

        kwargs = {
            "model": model,
            "temperature": config.temperature,
            "streaming": config.streaming,
            "api_key": api_key,
            "max_tokens": config.max_tokens,
        }

        # Add optional endpoint if provided
        if config.endpoint:
            kwargs["endpoint"] = config.endpoint

        return ChatMistralAI(**kwargs)


class ProviderFactory:
    """
    Factory for creating provider handlers.

    This factory manages the creation and caching of provider handlers,
    enabling easy switching between providers and adding new ones.

    Example:
        >>> handler = ProviderFactory.get_handler(Provider.ANTHROPIC)
        >>> config = ProviderConfig(api_key="sk-...", temperature=0.5)
        >>> model = handler.get_model("claude-3-sonnet", config)
    """

    _handlers = {
        Provider.ANTHROPIC: AnthropicHandler(),
        Provider.OPENAI: OpenAIHandler(),
        Provider.OLLAMA: OllamaHandler(),
        Provider.GOOGLE: GoogleHandler(),
        Provider.MISTRAL: MistralHandler(),
    }

    @classmethod
    def get_handler(cls, provider: Provider) -> ProviderHandler:
        """
        Get the handler for a specific provider.

        Args:
            provider: The provider type (from Provider enum).

        Returns:
            The corresponding ProviderHandler instance.

        Raises:
            ValueError: If the provider is not supported.
        """
        handler = cls._handlers.get(provider)
        if not handler:
            supported = ", ".join([p.value for p in Provider])
            raise ValueError(
                f"Unsupported provider: {provider.value}. "
                f"Supported providers are: {supported}"
            )
        return handler

    @classmethod
    def register_handler(cls, provider: Provider, handler: ProviderHandler) -> None:
        """
        Register a custom provider handler.

        This allows extending the system with new providers without modifying
        the existing code.

        Args:
            provider: The provider identifier.
            handler: The handler instance to register.

        Example:
            >>> class CustomHandler(ProviderHandler):
            ...     def get_model(self, model: str, config):
            ...         # Custom implementation
            ...         pass
            >>> ProviderFactory.register_handler(Provider.CUSTOM, CustomHandler())
        """
        cls._handlers[provider] = handler

"""
Base agent class providing model and prompt utilities.

This module provides the BaseAgent class with static methods for creating
and configuring LLM models from different providers, handling prompts,
and managing structured outputs.
"""

import re
import json
import logging
from typing import Any, Callable, List, Optional, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.tools import tool as create_tool
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from langchain_core.tools import BaseTool

from .provider_config import Provider, ProviderConfig, OllamaConfig, GoogleConfig
from .provider_handlers import ProviderFactory
from ..connectors.database.database_connector import DatabaseConnector

load_dotenv(override=True)

# Configure logging for parse errors
logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class providing LLM model creation and management utilities.

    This class offers static methods for instantiating LLM models with different
    configurations, handling prompts, and processing structured outputs.
    All model instantiation is delegated to provider handlers for clean separation
    of concerns.

    Supported providers:
    - anthropic: Claude models via Anthropic API
    - openai: GPT models via OpenAI API
    - ollama: Local LLM models via Ollama
    - google: Gemini models via Google Generative AI
    """

    def current_agent(self) -> str:
        """
        Get the name of the current agent class.

        Returns:
            The class name of the agent.
        """
        return self.__class__.__name__

    @staticmethod
    def _check_model(model: str) -> None:
        """
        Validate that a model name is provided.

        Currently a placeholder for future model validation logic.
        Can be extended to validate model names against supported models per provider.

        Args:
            model: The model identifier to validate.
        """
        pass

    @staticmethod
    def _get_model(
        model: str,
        temperature: int = 0,
        api_key: Optional[str] = None,
        streaming: bool = False,
        provider: str = 'anthropic',
        max_tokens: int = 8000
    ) -> Any:
        """
        Get a configured LLM model instance using the provider factory.

        This method delegates to the appropriate provider handler based on the
        provider parameter. It constructs a ProviderConfig with the given parameters
        and uses the ProviderFactory to instantiate the correct LLM.

        Args:
            model: The model identifier/name (e.g., 'gpt-4', 'claude-3-opus').
            temperature: Model temperature controlling randomness (0-2). Defaults to 0.
            api_key: API key for the provider. If None, will try environment variables.
            streaming: Whether to stream the response. Defaults to False.
            provider: The provider name ('openai', 'anthropic', 'ollama', 'google').
                     Defaults to 'anthropic'.
            max_tokens: Maximum tokens in response. Defaults to 8000.

        Returns:
            Configured LLM instance (ChatOpenAI, ChatAnthropic, ChatOllama, etc.).

        Raises:
            ValueError: If the provider is not supported.
            ValueError: If API key is missing for the provider.
        """
        # Convert string provider to Provider enum
        try:
            provider_enum = Provider(provider)
        except ValueError:
            supported = ", ".join([p.value for p in Provider])
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers are: {supported}"
            )

        # Create appropriate config based on provider type
        if provider_enum == Provider.OLLAMA:
            config = OllamaConfig(
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming
            )
        elif provider_enum == Provider.GOOGLE:
            config = GoogleConfig(
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming
            )
        else:
            config = ProviderConfig(
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming
            )

        # Get the handler and instantiate the model
        handler = ProviderFactory.get_handler(provider_enum)
        return handler.get_model(model, config)

    @staticmethod
    def _check_prompt(prompt: Union[str, list]) -> ChatPromptTemplate:
        """
        Convert a prompt to a ChatPromptTemplate.

        Handles both string and list-based prompt definitions:
        - Strings are treated as templates with variables
        - Lists are treated as message sequences with roles

        Args:
            prompt: Either a template string or list of message definitions.

        Returns:
            A ChatPromptTemplate instance.
        """
        if isinstance(prompt, str):
            return ChatPromptTemplate.from_template(prompt)

        if isinstance(prompt, list):
            return ChatPromptTemplate.from_messages(prompt)

        return prompt

    @staticmethod
    def structured_model(
        prompt: Union[str, list],
        model: str,
        return_class: type[BaseModel],
        api_key: Optional[str] = None,
        include_raw: bool = True,
        temperature: int = 0,
        streaming: bool = False,
        provider: str = 'anthropic',
        max_tokens: int = 8000,
        middleware: Optional[List[Callable]] = None
    ) -> Runnable:
        """
        Create a model that returns structured output based on a Pydantic class.

        This creates a LangChain Runnable that prompts the LLM and ensures the
        output conforms to the provided Pydantic schema. The model will be guided
        to return data matching the structure defined by return_class.

        Args:
            prompt: Prompt template string or list of messages.
            model: The model identifier (e.g., 'gpt-4', 'claude-3-opus').
            return_class: Pydantic BaseModel class defining the output schema.
            api_key: API key for the provider. If None, uses environment variables.
            include_raw: Whether to include raw unstructured output. Defaults to True.
            temperature: Model temperature (0-2). Defaults to 0.
            streaming: Whether to stream output. Defaults to False.
            provider: Provider name ('openai', 'anthropic', 'ollama', 'google').
                     Defaults to 'anthropic'.
            max_tokens: Maximum tokens in response. Defaults to 8000.
            middleware: Optional list of middleware callables to wrap the chain.
                       Each middleware receives the chain and returns a wrapped version.
                       Middleware is applied in reverse order (last middleware wraps first).

        Returns:
            A Runnable that returns structured output conforming to return_class.

        Example:
            >>> from pydantic import BaseModel
            >>> class Answer(BaseModel):
            ...     response: str
            ...     confidence: float
            >>> chain = BaseAgent.structured_model(
            ...     prompt="Answer: {question}",
            ...     model="claude-3-opus",
            ...     return_class=Answer
            ... )
            >>> result = chain.invoke({"question": "What is 2+2?"})
        """
        BaseAgent._check_model(model)
        prompt_obj = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(
            model,
            temperature,
            api_key,
            streaming=streaming,
            provider=provider,
            max_tokens=max_tokens
        )

        # Add structured output parsing to the LLM
        chain = prompt_obj | llm.with_structured_output(
            schema=return_class,
            include_raw=include_raw
        )

        # Apply middleware if provided
        if middleware:
            for mw in reversed(middleware):
                chain = mw(chain)

        return chain

    @staticmethod
    def standard_model(
        prompt: Union[str, list],
        model: str,
        api_key: Optional[str] = None,
        temperature: int = 0,
        streaming: bool = False,
        provider: str = 'anthropic',
        max_tokens: int = 8000,
        middleware: Optional[List[Callable]] = None
    ) -> Runnable:
        """
        Create a basic LLM model with prompt but no additional parsing.

        This creates a simple LangChain Runnable that chains a prompt with the LLM.
        The output is returned as-is from the model (typically a string or AIMessage).

        Args:
            prompt: Prompt template string or list of messages.
            model: The model identifier (e.g., 'gpt-4', 'claude-3-opus').
            api_key: API key for the provider. If None, uses environment variables.
            temperature: Model temperature (0-2). Defaults to 0.
            streaming: Whether to stream output. Defaults to False.
            provider: Provider name ('openai', 'anthropic', 'ollama', 'google').
                     Defaults to 'anthropic'.
            max_tokens: Maximum tokens in response. Defaults to 8000.
            middleware: Optional list of middleware callables to wrap the chain.
                       Each middleware receives the chain and returns a wrapped version.
                       Middleware is applied in reverse order (last middleware wraps first).

        Returns:
            A Runnable chain of prompt | llm.

        Example:
            >>> chain = BaseAgent.standard_model(
            ...     prompt="Answer: {question}",
            ...     model="claude-3-opus"
            ... )
            >>> result = chain.invoke({"question": "What is 2+2?"})
        """
        BaseAgent._check_model(model)
        prompt_obj = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(
            model,
            temperature,
            api_key,
            streaming=streaming,
            provider=provider,
            max_tokens=max_tokens
        )
        chain = prompt_obj | llm

        # Apply middleware if provided
        if middleware:
            for mw in reversed(middleware):
                chain = mw(chain)

        return chain

    @staticmethod
    def _transform_tools(tools: Optional[List] = None) -> List[BaseTool]:
        """
        Transform a list of tools to BaseTool instances.

        Converts callable functions to LangChain BaseTool instances if needed.
        Already-BaseTool instances are left unchanged.

        Args:
            tools: List of tools (functions or BaseTool instances).
                  If None, returns empty list.

        Returns:
            List of BaseTool instances.
        """
        if tools is None:
            tools = []

        t_tools = []
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            t_tools.append(tool_)
        return t_tools

    @staticmethod
    def tool_model(
        prompt: Union[str, list],
        model: str,
        temperature: int = 0,
        api_key: Optional[str] = None,
        tools: Optional[List] = None,
        streaming: bool = False,
        add_parser: bool = False,
        tool_choice: Optional[str] = None,
        provider: str = 'anthropic',
        max_tokens: int = 8000,
        middleware: Optional[List[Callable]] = None
    ) -> Runnable:
        """
        Create an LLM model with access to tools.

        This creates a LangChain Runnable that allows the LLM to call specified tools.
        The model can decide which tools to use and when based on the prompt and context.
        Optionally includes a parser to extract tool calls from the output.

        Args:
            prompt: Prompt template string or list of messages.
            model: The model identifier (e.g., 'gpt-4', 'claude-3-opus').
            temperature: Model temperature (0-2). Defaults to 0.
            api_key: API key for the provider. If None, uses environment variables.
            tools: List of tools for the model to use. Defaults to None.
            streaming: Whether to stream output. Defaults to False.
            add_parser: Whether to add a PydanticToolsParser to extract tool calls.
                       Defaults to False.
            tool_choice: Force the model to use a specific tool. Defaults to None (auto).
            provider: Provider name ('openai', 'anthropic', 'ollama', 'google').
                     Defaults to 'anthropic'.
            max_tokens: Maximum tokens in response. Defaults to 8000.
            middleware: Optional list of middleware callables to wrap the chain.
                       Each middleware receives the chain and returns a wrapped version.
                       Middleware is applied in reverse order (last middleware wraps first).

        Returns:
            A Runnable that can invoke tools. If add_parser=True, also parses tool calls.

        Example:
            >>> def calculator(a: int, b: int) -> int:
            ...     '''Add two numbers'''
            ...     return a + b
            >>> chain = BaseAgent.tool_model(
            ...     prompt="Use tools: {request}",
            ...     model="claude-3-opus",
            ...     tools=[calculator]
            ... )
        """
        BaseAgent._check_model(model)
        prompt_obj = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(
            model,
            temperature,
            api_key,
            streaming,
            provider=provider,
            max_tokens=max_tokens
        )

        unique_tools = BaseAgent._transform_tools(tools)
        chain = prompt_obj | llm.bind_tools(unique_tools, tool_choice=tool_choice)

        if add_parser:
            chain = chain | PydanticToolsParser(tools=unique_tools)

        # Apply middleware if provided
        if middleware:
            for mw in reversed(middleware):
                chain = mw(chain)

        return chain

    @staticmethod
    def string_model(
        prompt: Union[str, list],
        model: str,
        api_key: Optional[str] = None,
        temperature: int = 0,
        provider: str = 'anthropic',
        streaming: bool = False,
        max_tokens: int = 8000,
        middleware: Optional[List[Callable]] = None
    ) -> Runnable:
        """
        Create an LLM model that returns plain string output.

        This creates a LangChain Runnable that chains a prompt with the LLM
        and a StrOutputParser to extract just the text content from the response.

        Args:
            prompt: Prompt template string or list of messages.
            model: The model identifier (e.g., 'gpt-4', 'claude-3-opus').
            api_key: API key for the provider. If None, uses environment variables.
            temperature: Model temperature (0-2). Defaults to 0.
            provider: Provider name ('openai', 'anthropic', 'ollama', 'google').
                     Defaults to 'anthropic'.
            streaming: Whether to stream output. Defaults to False.
            max_tokens: Maximum tokens in response. Defaults to 8000.
            middleware: Optional list of middleware callables to wrap the chain.
                       Each middleware receives the chain and returns a wrapped version.
                       Middleware is applied in reverse order (last middleware wraps first).

        Returns:
            A Runnable chain of prompt | llm | string_parser.

        Example:
            >>> chain = BaseAgent.string_model(
            ...     prompt="Summarize: {text}",
            ...     model="claude-3-opus"
            ... )
            >>> result = chain.invoke({"text": "Long text..."})
            >>> # result is a plain string, not an AIMessage
        """
        BaseAgent._check_model(model)
        prompt_obj = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(
            model,
            temperature,
            api_key,
            provider=provider,
            streaming=streaming,
            max_tokens=max_tokens
        )
        chain = prompt_obj | llm | StrOutputParser()

        # Apply middleware if provided
        if middleware:
            for mw in reversed(middleware):
                chain = mw(chain)

        return chain

    @staticmethod
    def parse_llm_response_generic(
        response_text: str,
        possible_classes: List[type[BaseModel]]
    ) -> Optional[BaseModel]:
        """
        Parse LLM response into one of several possible Pydantic classes.

        Attempts to match the response against multiple Pydantic model classes.
        Expects XML-tagged JSON format: <ClassName>{...json...}</ClassName>

        Tries each class in order and returns the first successfully parsed instance.
        If no class matches, returns None with error logging.

        Args:
            response_text: The LLM response text containing XML-tagged JSON.
            possible_classes: List of Pydantic BaseModel classes to try.

        Returns:
            An instance of one of possible_classes if parsing succeeds, None otherwise.

        Example:
            >>> class Decision(BaseModel):
            ...     choice: str
            ...     confidence: float
            >>> class Error(BaseModel):
            ...     error_message: str
            >>> response = '<Decision>{"choice": "yes", "confidence": 0.95}</Decision>'
            >>> result = BaseAgent.parse_llm_response_generic(response, [Decision, Error])
            >>> print(result.choice)  # "yes"
        """
        # Remove answer_block wrapper if present
        content = re.sub(
            r'</?answer_block[^>]*>',
            '',
            response_text,
            flags=re.IGNORECASE
        ).strip()

        # Try each possible class
        for model_class in possible_classes:
            class_name = model_class.__name__

            # Create pattern to match the class tag with JSON content
            pattern = rf'<{class_name}>\s*({{.*?}})\s*</{class_name}>'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

            if match:
                try:
                    json_str = match.group(1).strip()
                    json_data = json.loads(json_str)
                    return model_class(**json_data)

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON decode error for {class_name}: {e}"
                    )
                    continue
                except ValidationError as e:
                    logger.warning(
                        f"Validation error for {class_name}: {e}"
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        f"Unexpected error parsing {class_name}: {e}"
                    )
                    continue

        return None

    @staticmethod
    def _get_default_prompt_part() -> str:
        """
        Get a default prompt segment with context information.

        This is a template snippet that can be included in prompts to provide
        standard context like current date and user information.

        Returns:
            A prompt template string with placeholders for {time} and {user}.
        """
        return """
                    Context:
                    - The current date and time is: {time}.
                    - You work on behalf of the user with the username: {user}.

                    Remember:
                    DONT make up any information. If your tools did not work, then provide the error.
        """

    @staticmethod
    def _get_prompt_from_db(
        provider: str,
        model: str,
        agent: str,
        type_: str = "system"
    ) -> str:
        """
        Retrieve a prompt from the database for a specific agent/model/provider.

        Queries the agent_definition.prompts table to retrieve pre-configured
        prompts based on provider, model, agent name, and prompt type.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
            model: Model identifier (e.g., 'gpt-4', 'claude-3-opus').
            agent: Agent shortname.
            type_: Prompt type (default 'system'). Other values could be 'user', 'instruction'.

        Returns:
            The prompt text retrieved from the database.

        Raises:
            IndexError: If no matching prompt is found in the database.
            DatabaseConnector exceptions: If database query fails.
        """
        sql_query = """
        SELECT p.prompt
        FROM agent_definition.prompts p
        JOIN agent_definition.agents a ON p.agent_id = a.id
        WHERE p.provider = :provider
        AND p.model = :model
        AND a.shortname = :agent_shortname
        AND p.type = :type;
        """

        params = {
            "provider": provider,
            "model": model,
            "agent_shortname": agent,
            "type": type_,
        }

        db = DatabaseConnector()
        result = db.execute_query(sql_query, params)
        # result is a list of Row objects from SQLAlchemy's fetchall()
        if result:
            return result[0][0]
        raise ValueError(
            f"No prompt found for provider={provider}, model={model}, "
            f"agent={agent}, type={type_}"
        )

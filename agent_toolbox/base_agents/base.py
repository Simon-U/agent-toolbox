import re
import json
import inspect
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.tools import tool as create_tool
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from typing import Any
from langchain_core.tools import BaseTool
from typing import List, Optional

from langchain_ollama import ChatOllama
from ..connectors.ollama.proxy_ollama import ProxyOllama

load_dotenv(override=True)

from ..connectors.database.database_connector import DatabaseConnector


class BaseAgent:

    def current_agent(self):
        class_name = self.__class__.__name__
        return class_name

    @staticmethod
    def _check_model(model):
        pass

    @staticmethod
    def _get_model(model, temperature, api_key, streaming=False, provider='anthropic', max_tokens=8000):
        if provider == 'openai':
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                streaming=streaming,
                api_key=api_key,
                max_tokens=max_tokens
            )
        elif provider == 'anthropic':
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            return ChatAnthropic(
                model=model, temperature=temperature, streaming=streaming, api_key=api_key, max_tokens=max_tokens
            )
        elif provider == 'ollama':
            base_url = os.getenv("LOCAL_LLM_URL", "http://103.133.98.204:11434")
            if os.getenv("HTTP_PROXY", False):
                return ProxyOllama(
                    model=model,
                    temperature=temperature,
                    num_ctx=10000,
                    base_url=base_url,
                    streaming=streaming,
                    format="json",
                    disable_streaming=True,
                )
            return ChatOllama(
                model=model,
                temperature=temperature,
                num_ctx=10000,
                base_url=base_url,
                streaming=streaming,
                format="json",
                #disable_streaming=True
                )
        raise ValueError(f'Your provider {provider} is not supported')

    @staticmethod
    def _check_prompt(prompt):
        if isinstance(prompt, str):
            prompt = ChatPromptTemplate.from_template(prompt)
            
        if isinstance(prompt, list):
            prompt = ChatPromptTemplate.from_messages(prompt)

        return prompt

    @staticmethod
    def structured_model(
        prompt: str,
        model: str,
        return_class: type[BaseModel],
        api_key: Any = None,
        include_raw: bool = True,
        temperature: int = 0,
        streaming: bool = False,
        provider='anthropic',
        max_tokens=8000
    ) -> Runnable:
        """A model return structured output based on the return_class

        Args:
            prompt (str): prompt for the model
            api_key (str): Api key for the provider. If false will try to get env variable keys
            model (str): The model to use
            return_class (type[BaseModel]): A pydantic class showing the desired output
            include_raw (bool, optional): Should the raw, not paresd output be inclued in the respone. Defaults to True.
            temperature (int, optional): The tempreture for the model. Defaults to 0.
            streaming (bool, optional): Streamign the output. Defaults to False.

        Returns:
            Runnable: A invokable model to return structured output
        """

        BaseAgent._check_model(model)
        prompt = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(model, temperature, api_key, provider=provider, max_tokens=max_tokens)
        
        

        llm = prompt | llm.with_structured_output(
            schema=return_class, include_raw=include_raw
        )

        return llm

    @staticmethod
    def standard_model(
        prompt: str,
        model: str,
        api_key: Any = None,
        temperature: int = 0,
        streaming: bool = False,
        provider='anthropic',
        max_tokens=8000
    ) -> Runnable:
        """Basic standard llm model with no additions

        Args:
            prompt (str): prompt for the model
            api_key (str): Api key for the provider. If false will try to get env variable keys
            model (str): The model to use
            temperature (int, optional): The tempreture for the model. Defaults to 0.
            streaming (bool, optional): Streamign the output. Defaults to False.

        Returns:
            Runnable: Invokable Model
        """
        # Check if we support the selected model

        BaseAgent._check_model(model)
        prompt = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(model, temperature, api_key, streaming=streaming, provider=provider, max_tokens=max_tokens)
        return prompt | llm

    @staticmethod
    def _transform_tools(tools):
        t_tools = []
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            t_tools.append(tool_)
        return t_tools
    @staticmethod
    def tool_model(
        prompt: str,
        model: str,
        temperature=0,
        api_key: Any = None,
        tools: list = [],
        streaming=False,
        add_parser=False,
        tool_choice=None,
        provider='anthropic',
        max_tokens=8000
    ) -> Runnable:
        """Creating a model that is able to use tools. These tools can be executed by the model

        Args:
            prompt (str): prompt for the model
            api_key (str): Api key for the provider. If false will try to get env variable keys
            model (str): The model to use
            temperature (int, optional): The tempreture for the model. Defaults to 0.
            tools (list, optional): List of tools for the model to use. Defaults to [].
            streaming (bool, optional): Streamign the output. Defaults to False.

        Returns:
            Runnable: Invokable tool model
        """
        BaseAgent._check_model(model)
        prompt = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(model, temperature, api_key, streaming, provider=provider, max_tokens=max_tokens)

        unique_tools = []
        unique_tools = BaseAgent._transform_tools(tools)
        llm = prompt | llm.bind_tools(unique_tools, tool_choice=tool_choice)
        if add_parser:
            llm = llm | PydanticToolsParser(tools=unique_tools)
        return llm
    

    @staticmethod
    def string_model(prompt, api_key: Any = None, temperature=0, model=None, provider='anthropic', streaming=False, max_tokens=8000):
        # Check if we support the selected model

        BaseAgent._check_model(model)
        prompt = BaseAgent._check_prompt(prompt)
        llm = BaseAgent._get_model(model, temperature, api_key, provider=provider, streaming=streaming, max_tokens=max_tokens)
        return prompt | llm | StrOutputParser()

    @staticmethod 
    def parse_llm_response_generic(
        response_text: str, 
        possible_classes: List[BaseModel]
    ) -> Optional[BaseModel]:
        """
        Generic parser that tries to match any of the provided Pydantic classes
        Expects XML tags with JSON content format
        """
        # Remove answer_block wrapper if present
        content = re.sub(r'</?answer_block[^>]*>', '', response_text, flags=re.IGNORECASE).strip()
        
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
                    print(f"JSON decode error for {class_name}: {e}")
                    continue
                except ValidationError as e:
                    print(f"Validation error for {class_name}: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error parsing {class_name}: {e}")
                    continue
        
        return None

    @staticmethod
    def _get_default_prompt_part():

        return """
                    Context:
                    - The current date and time is: {time}.
                    - You work on behalf of the user with the username: {user}.

                    Remember:
                    DONT make aup any information. If your tools did not work, then provide the error.
        """

    @staticmethod
    def _get_prompt_from_db(provider, model, agent, type="system"):
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
            "type": type,
        }
        db = DatabaseConnector()
        return db.execute_query(sql_query, params)[0][0]

import re
import json
import inspect
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any
from langchain_core.output_parsers import PydanticOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.output_parsers import RetryOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from ..connectors.ollama.ollama import Ollama
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
            return Ollama(
                model=model,
                temperature=temperature,
                num_ctx=10000,
                base_url=base_url,
                streaming=streaming,
                format="json",
                disable_streaming=True)
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
        if not model.startswith("gpt"):
            seen = set()
            for tool in tools:
                tool_name = tool.name if hasattr(tool, "name") else str(tool)
                if tool_name not in seen:
                    unique_tools.append(convert_to_anthropic_tool(tool))
                    seen.add(tool_name)
        else:
            unique_tools = tools
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
    def _get_output(response):
        if response["raw"].additional_kwargs != {}:
            json_string = response["raw"].additional_kwargs["tool_calls"][0][
                "function"
            ]["arguments"]
        else:
            json_string = response["raw"].tool_calls[0]["args"]["properties"]
        return json_string

    @staticmethod
    def _has_list_attribute(pydantic_class):
        # Retrieve the annotations of the class and check if any attribute is a list
        for attr_name, attr_type in pydantic_class.__annotations__.items():
            if (
                inspect.isclass(attr_type)
                and getattr(attr_type, "__origin__", None) is list
            ):
                return True
        return False

    def _process_list(self, json_string):
        if not json_string.endswith('"}'):
            json_string += '"}'

            list_pattern = re.compile(r"(.*?)\s*\[", re.DOTALL)

            # Find the annotations part
            match = list_pattern.search(json_string)
            if match:
                list_text = match.group(0)

            json_string = re.sub(list_pattern, "", json_string)

            # Get all dictionaries from the list
            pattern = re.compile(r"\{.*?\}", re.DOTALL)
            matches = pattern.findall(json_string)
            key_value_pairs_list = []
            num_keys = 0
            for match in matches:
                match_dict, num_keys = self._extract_key_value_pairs(match, num_keys)
                if match_dict:
                    key_value_pairs_list.append(match_dict)

            dicts_str = ", ".join(key_value_pairs_list)
            json_string = list_text + f"{dicts_str}" + "]}"
        return json_string

    def _process_json_string(self, json_string, return_class):
        """
        Handles the JSON processing logic depending on the return class and skip flag.
        """
        # ToDO Here we likely have to check what is a list and what not There might be class with list, and one none list
        if isinstance(json_string, dict):
            return json_string

        elif self._has_list_attribute(return_class):
            final_json_str = self._process_list(json_string)
            return json.loads(final_json_str)

        else:
            final_json_str, num_keys = self._extract_key_value_pairs(json_string)
            return json.loads(final_json_str)

    def format_raw_response(self, response, return_class=None):
        """
        This function tries to format the response if it fails by the model.

        response: The response from the mpdel
        state: The state of the graph. Only needed for causal explanation
        skip:

        """
        step = self.current_agent()
        json_string = self._get_output(response)
        try:
            json_obj = self._process_json_string(json_string, return_class)

            if return_class:
                return return_class(**json_obj)
            return json_obj

        except Exception as e:
            raise ValueError(f"Failed to parse output class in BaseAgent: {e}")

    def _extract_key_value_pairs(self, json_string, num_keys=0):

        # if not json_string.endswith("}"):
        #     json_string += '"}'

        json_string = json_string.replace('"', "")

        json_entities = json_string.split(",")

        matches = []

        for i in json_entities:
            if ":" in i.split(" ")[0]:
                matches.append(i)
            else:
                try:
                    matches[-1] = matches[-1] + i
                except:
                    pass

        # pattern = re.compile(r'(\w+):([^,{}]+)(?=,|}|$)', re.DOTALL)
        # matches = pattern.findall(json_string)

        key_value_pairs = []

        current_num_keys = 0
        for match in matches:
            key = match.split(":")[0].strip("{")
            value = "".join(match.split(":")[1:]).strip("}")
            current_num_keys += 1
            key_value_pairs.append(f'"{key}": "{value}"')

        if current_num_keys < num_keys:
            return None, num_keys

        dict_str = "{" + ", ".join(key_value_pairs) + "}"
        return dict_str, current_num_keys

    def _add_quotes(self, match):
        return f'"{match.group(0)}"'

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

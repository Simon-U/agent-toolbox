import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
import requests
from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon

__all__ = []


@deprecated("0.0.3", alternative="_chat_stream_response_to_chat_generation_chunk")
def _stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(content=parsed_response.get("response", "")),
        generation_info=generation_info,
    )


def _chat_stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=parsed_response.get("message", {}).get("content", "")
        ),
        generation_info=generation_info,
    )


class ChatOllama(BaseChatModel, _OllamaCommon):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatOllama
            ollama = ChatOllama(model="llama2")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "ollama-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="ollama",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("num_predict", self.num_predict):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @deprecated("0.0.3", alternative="_convert_messages_to_ollama_messages")
    def _format_message_as_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            if isinstance(message.content, List):
                first_content = cast(List[Dict], message.content)[0]
                content_type = first_content.get("type")
                if content_type == "text":
                    message_text = f"[INST] {first_content['text']} [/INST]"
                elif content_type == "image_url":
                    message_text = first_content["image_url"]["url"]
            else:
                message_text = f"[INST] {message.content} [/INST]"
        elif isinstance(message, AIMessage):
            message_text = f"{message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"<<SYS>> {message.content} <</SYS>>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            [self._format_message_as_text(message) for message in messages]
        )

    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        ollama_messages: List = []
        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "image_url":
                        image_url = None
                        temp_image_url = content_part.get("image_url")
                        if isinstance(temp_image_url, str):
                            image_url = content_part["image_url"]
                        elif (
                            isinstance(temp_image_url, dict) and "url" in temp_image_url
                        ):
                            image_url = temp_image_url
                        else:
                            raise ValueError(
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )

                        image_url_components = image_url.split(",")
                        # Support data:image/jpeg;base64,<image> format
                        # and base64 strings
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])

                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )

            ollama_messages.append(
                {
                    "role": role,
                    "content": content,
                    "images": images,
                }
            )

        return ollama_messages

    def _create_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                "stop": stop,
                **{k: v for k, v in kwargs.items() if k not in self._default_params},
            }

        if "functions" in params["options"]:
            del params["options"]["functions"]

        # system_msg = {
        #     "role": "system",
        #     "content": 'You have access to the following tools:\n\n[\n  {\n    "name": "EntityExtractor",\n    "parameters": {\n      "title": "EntityExtractor",\n      "type": "object",\n      "properties": {\n        "company_name": {\n          "title": "Company Name",\n          "description": "company name mentioned in the question. If more than one company mentioned, separate them using comma.",\n          "type": "string"\n        },\n        "asx_code": {\n          "title": "Asx Code",\n          "description": "asx code mentioned in the question. If more than one asx code mentioned, separate them using comma.",\n          "type": "string"\n        },\n        "publish_date": {\n          "title": "Publish Date",\n          "description": "date mentioned in the question in YYYY or YYYY-MM or YYYY-MM-DD format.",\n          "type": "string"\n        },\n        "keyword": {\n          "title": "Keyword",\n          "description": "keyword from the user question. this is usually 1-2 words",\n          "type": "string"\n        }\n      },\n      "required": [\n        "company_name",\n        "asx_code",\n        "publish_date",\n        "keyword"\n      ]\n    }\n  },\n  {\n    "name": "__conversational_response",\n    "description": "Respond conversationally if no other tools should be called for a given query.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "response": {\n          "type": "string",\n          "description": "Conversational response to the user."\n        }\n      },\n      "required": [\n        "response"\n      ]\n    }\n  }\n]\n\nYou must always select one of the above tools and You must respond with only a JSON object, without any additional text or explanation outside the JSON schema. Here is the schema:\n\n{\n  "tool": <name of the selected tool>,\n  "tool_input": <parameters for the selected tool, matching the tool\'s JSON schema>\n}\n',
        #     "images": [],
        # }

        # payload["messages"].insert(0, system_msg)

        if payload.get("messages"):
            request_payload = {"messages": payload.get("messages", []), **params}
        else:
            request_payload = {
                "prompt": payload.get("prompt"),
                "images": payload.get("images", []),
                **params,
            }
        response = requests.post(
            url=api_url,
            headers={
                "Content-Type": "application/json",
                **(self.headers if isinstance(self.headers, dict) else {}),
            },
            json=request_payload,
            stream=True,
            timeout=self.timeout,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            if response.status_code == 404:
                raise OllamaEndpointNotFoundError(
                    "Ollama call failed with status code 404. "
                    "Maybe your model is not found "
                    f"and you should pull the model with `ollama pull {self.model}`."
                )
            else:
                optional_detail = response.text
                raise ValueError(
                    f"Ollama call failed with status code {response.status_code}."
                    f" Details: {optional_detail}"
                )
        return response.iter_lines(decode_unicode=True)

    def _create_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_ollama_messages(messages),
        }
        yield from self._create_stream(
            payload=payload, stop=stop, api_url=f"{self.base_url}/api/chat", **kwargs
        )

    async def _acreate_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_ollama_messages(messages),
        }
        async for stream_resp in self._acreate_stream(
            payload=payload, stop=stop, api_url=f"{self.base_url}/api/chat", **kwargs
        ):
            yield stream_resp

    def _chat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[ChatGenerationChunk] = None
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[ChatGenerationChunk] = None
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Ollama's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = ollama([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """
        final_chunk = self._chat_stream_with_aggregation(
            messages,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **kwargs,
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Ollama's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = ollama([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """

        final_chunk = await self._achat_stream_with_aggregation(
            messages,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **kwargs,
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
                if stream_resp:
                    chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            chunk.text,
                            chunk=chunk,
                            verbose=self.verbose,
                        )

                    yield chunk
        except OllamaEndpointNotFoundError:
            yield from self._legacy_stream(messages, stop, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=self.verbose,
                    )
                yield chunk

    @deprecated("0.0.3", alternative="_stream")
    def _legacy_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._format_messages_as_text(messages)
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_chat_generation_chunk(stream_resp)
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=self.verbose,
                    )
                yield chunk

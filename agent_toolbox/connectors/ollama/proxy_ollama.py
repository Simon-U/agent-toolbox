import aiohttp
import requests

from typing import (
    Any,
    List,
    Optional,
    Iterator,
    AsyncIterator,
)

from pydantic import Field
from aiohttp_socks import ProxyConnector

from langchain_ollama import ChatOllama

__all__ = ["ProxyOllama"]


class ProxyOllama(ChatOllama):
    """Function chat model that uses Ollama API."""

    proxies: dict = Field(
        default_factory=lambda: {
            "http": "socks5://localhost:8888",
            "https": "socks5://localhost:8888",
        }
    )

    def __init__(self, *args, **kwargs):
        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)

    """
    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:

        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        llm = self.bind_tools(tools=[schema], format="json")
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticOutputParser(  # type: ignore[type-var]
                pydantic_object=schema  # type: ignore[arg-type]
            )
        else:
            output_parser = JsonOutputParser()

        parser_chain = RunnableLambda(parse_response) | output_parser
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser_chain, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | parser_chain
    """

    async def _acreate_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
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

        if payload.get("messages"):
            request_payload = {"messages": payload.get("messages", []), **params}
        else:
            request_payload = {
                "prompt": payload.get("prompt"),
                "images": payload.get("images", []),
                **params,
            }

        proxy_url = self.proxies["http"]
        connector = ProxyConnector.from_url(proxy_url)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                url=api_url,
                headers={
                    "Content-Type": "application/json",
                    **(self.headers if isinstance(self.headers, dict) else {}),
                },
                auth=self.auth,  # type: ignore[arg-type]
                json=request_payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    if response.status == 404:
                        raise ValueError(
                            "Ollama call failed with status code 404."
                        )
                    else:
                        optional_detail = response.text
                        raise ValueError(
                            f"Ollama call failed with status code {response.status}."
                            f" Details: {optional_detail}"
                        )
                async for line in response.content:
                    yield line.decode("utf-8")

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
            auth=self.auth,
            json=request_payload,
            stream=True,
            timeout=self.timeout,
            proxies=self.proxies,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            if response.status_code == 404:
                raise ValueError(
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

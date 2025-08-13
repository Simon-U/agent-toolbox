from typing import Annotated, Optional
from typing_extensions import TypedDict

import functools
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.graph.message import AnyMessage, add_messages

__all__ = ["ToolAgent"]


class AgentUtils:
    def handle_tool_error(state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    @staticmethod
    def create_tool_node_with_fallback(tools: list) -> dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(AgentUtils.handle_tool_error)], exception_key="error"
        )


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ToolAgent(AgentUtils):
    """ToolAgent is an agent factory. Pass an agent_class which has tools and creates an subgraph to run agent with tools in a loop.

    In the future we might support other agents as well

    assistant_class (class): The agent module
    model (str): The model name
    api_key (str): The api key. If not providded, try to get the one from env
    streaming (bool, optional): Should the agent module stream the output at the end. Defaults to False.

    return; An executable graph
    """

    def __init__(self, assistant_class, provider, model, api_key, streaming=False, state_class = GraphState):
        """_summary_

        Args:

        """
        self.assistant_class = assistant_class
        self.streaming = streaming
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.graph_state = state_class

    @staticmethod
    def _check(message):
        if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
            return "tools"
        return "finished"

    @staticmethod
    def run_agent(state, agent):
        output_dict = {}
        input = {**state}
        response = agent.invoke(input)
        output_dict["messages"] = response
        return output_dict

    @staticmethod
    def router(state, tools=None):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls

        safe_toolnames = [t.name for t in tools]
        if all(tc["name"] in safe_toolnames for tc in tool_calls):
            return "assistant_tools"
        raise ValueError

    def create_agent(self):

        workflow = StateGraph(self.graph_state)
        workflow.set_entry_point("Assistant")

        assistant_node = functools.partial(
            self.run_agent,
            agent=self.assistant_class(
                additional_tools=[],
                streaming=self.streaming,
                model=self.model,
                api_key=self.api_key,
                provider=self.provider,
            ),
        )
        workflow.add_node("Assistant", assistant_node)
        workflow.add_node(
            "assistant_tools",
            self.create_tool_node_with_fallback(self.assistant_class._get_tools()),
        )

        router = functools.partial(
            self.router,
            tools=self.assistant_class._get_tools(),
        )
        workflow.add_conditional_edges("Assistant", router)
        workflow.add_edge("assistant_tools", "Assistant")

        sub_agent = workflow.compile()
        return sub_agent

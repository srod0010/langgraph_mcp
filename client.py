from typing import List
from typing_extensions import TypedDict
from typing import Annotated, cast

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_litellm import ChatLiteLLM
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools

import asyncio
import os

from dotenv import load_dotenv

# Load variables from a local .env file into the environment.
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

"""
Note: uncommented setup becasue both stdio if one of the clients was streamable http or sse you would do something like
client = MultiServerMCPClient(
    {
        "first_server": {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        },
        "second_server": {
            "url": "http://localhost:8001/mcp",
            "transport": "streamable_http",
        }
    }
)
"""
client = MultiServerMCPClient(cast(dict[str, Connection], {
    "first_server": {
        "command": "python",
        "args": ["server_one.py"],
        "transport": "stdio",
    },
    "second_server": {
        "command": "python",
        "args": ["server_two.py"],
        "transport": "stdio",
    }
}))


async def create_graph(first_session, second_session):

    # LiteLLM reads OPENAI_API_KEY from the environment automatically.
    llm = ChatLiteLLM(model="gpt-4o")

    # tool discovery
    first_tools = await load_mcp_tools(first_session)
    second_tools = await load_mcp_tools(second_session)
    tools = first_tools + second_tools
    llm_with_tool = llm.bind_tools(tools)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that uses tools to solve problems."),
        MessagesPlaceholder("messages")
    ])
    chat_llm = prompt_template | llm_with_tool

    # State Management
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    # Nodes
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Building the graph
    # Initialize a stateful flow using the State schema defined above.
    graph_builder = StateGraph(State)
    # chat_node: where reasoning happens.
    graph_builder.add_node("chat_node", chat_node)
    # tool_node: a built-in node that automatically executes tool calls from messages.
    graph_builder.add_node("tool_node", ToolNode(tools=tools))
    # The graph starts at chat_node.
    graph_builder.add_edge(START, "chat_node")
    # After chat_node, check whether the next step involves tools:
    # if yes -> go to tool_node; if no -> end the graph.
    graph_builder.add_conditional_edges("chat_node", tools_condition, {
                                        "tools": "tool_node", "__end__": END})
    # After tool execution, go back to chat_node for further reasoning.
    graph_builder.add_edge("tool_node", "chat_node")
    # Compile into an executable graph. MemorySaver keeps intermediate states
    # in memory (ephemeral).
    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph


async def main():

    # Runtime config passed on every invoke. "thread_id" names the conversation:
    # the MemorySaver checkpointer stores/loads state per thread_id, so reusing
    # the same id keeps memory across turns. A different id = a fresh conversation.
    config = {"configurable": {"thread_id": 1234}}

    async with client.session("first_server") as first_session, \
            client.session("second_server") as second_session:
        agent = await create_graph(first_session, second_session)

        while True:
            message = input("User: ")
            response = await agent.ainvoke({"messages": message},
                                           config=config)
            print("AI: "+response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

from typing import TypedDict, Annotated, Dict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
load_dotenv()

# -------------------- Subgraph Definition --------------------

class ChildState(TypedDict):
    messages: Annotated[list, add_messages]

# Set up tools and model
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools=tools)

# Agent node function
def agent(state: ChildState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }

# Routing logic for tool invocation
def tools_router(state: ChildState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

# Build subgraph
tool_node = ToolNode(tools=tools)
subgraph = StateGraph(ChildState)
subgraph.add_node("agent", agent)
subgraph.add_node("tool_node", tool_node)
subgraph.set_entry_point("agent")
subgraph.add_conditional_edges("agent", tools_router)
subgraph.add_edge("tool_node", "agent")
search_app = subgraph.compile()

# -------------------- Parent Graph 1: Matching Schema --------------------

class ParentState(TypedDict):
    messages: Annotated[list, add_messages]

parent_graph = StateGraph(ParentState)
parent_graph.add_node("search_agent", search_app)
parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)
parent_app_1 = parent_graph.compile()

# Example usage of parent graph 1
# def run_parent_graph_1():
#     print("Running parent graph with same schema...")
#     result = parent_app_1.invoke({"messages": [HumanMessage(content="How is the weather in Chennai?")]})
#     print(result)

# -------------------- Parent Graph 2: Different Schema --------------------

class QueryState(TypedDict):
    query: str
    response: str

def search_agent_transform(state: QueryState) -> Dict:
    subgraph_input = {
        "messages": [HumanMessage(content=state["query"])]
    }
    subgraph_result = search_app.invoke(subgraph_input)
    assistant_message = subgraph_result["messages"][-1]
    return {"response": assistant_message.content}

parent_graph_2 = StateGraph(QueryState)
parent_graph_2.add_node("search_agent", search_agent_transform)
parent_graph_2.add_edge(START, "search_agent")
parent_graph_2.add_edge("search_agent", END)
parent_app_2 = parent_graph_2.compile()

# Example usage of parent graph 2
def run_parent_graph_2():
    print("Running parent graph with different schema...")
    result = parent_app_2.invoke({"query": "How is the weather in Chennai?", "response": ""})
    print(result)

# -------------------- Main --------------------

if __name__ == "__main__":
    # run_parent_graph_1()
    run_parent_graph_2()

from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
# Checkpoint memory
memory = MemorySaver()

# Tool setup
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# LLM setup (with tools)
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools=tools)

# Define state
class ConversationState(TypedDict):
    messages: Annotated[List, add_messages]

# LLM node
def llm_node(state: ConversationState) -> ConversationState:
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

# Router to check if tools should be used
def route_tools(state: ConversationState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return END

# Build graph
graph = StateGraph(ConversationState)
graph.add_node("model", llm_node)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("model")
graph.add_conditional_edges("model", route_tools)
graph.add_edge("tools", "model")

# Compile app
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

# Optional: Visualize
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

# Run the app
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": 1}}

# Step 1: Ask the question
events = app.stream(
    {"messages": [HumanMessage(content="What is the current weather in Chennai?")]},
    config=config,
    stream_mode="values"
)

# Step 2: Show intermediate results
for event in events:
    event["messages"][-1].pretty_print()

# Step 3: Resume from the tool output
snapshot = app.get_state(config=config)
print("Next step in graph:", snapshot.next)

events = app.stream(None, config=config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

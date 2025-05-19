from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing import TypedDict


# Define the shape of the state
class State(TypedDict):
    text: str


# Node A
def node_a(state: State):
    print("Node A")
    return Command(
        goto="node_b",
        update={
            "text": state["text"] + "a"
        }
    )


# Node B
def node_b(state: State):
    print("Node B")
    return Command(
        goto="node_c",
        update={
            "text": state["text"] + "b"
        }
    )


# Node C
def node_c(state: State):
    print("Node C")
    return Command(
        goto=END,
        update={
            "text": state["text"] + "c"
        }
    )


# Define and build the graph
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.set_entry_point("node_a")

# Compile and run the graph
app = graph.compile()

response = app.invoke({
    "text": ""
})

print("\n✅ Final Response:", response)

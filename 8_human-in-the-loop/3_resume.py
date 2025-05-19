from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

# Create in-memory checkpointing
memory = MemorySaver()

# Define the state type
class State(TypedDict):
    value: str

# Node A logic
def node_a(state: State):
    print("Node A")
    return Command(
        goto="node_b",
        update={"value": state["value"] + "a"}
    )

# Node B logic with human interrupt
def node_b(state: State):
    print("Node B")

    # Ask for human input
    human_response = interrupt("Do you want to go to C or D? Type C/D")

    # The code below won't run during the interrupt, but will run after resume
    print("Human Review Value:", human_response)

    if human_response == "C":
        return Command(
            goto="node_c",
            update={"value": state["value"] + "b"}
        )
    elif human_response == "D":
        return Command(
            goto="node_d",
            update={"value": state["value"] + "b"}
        )

# Node C logic
def node_c(state: State):
    print("Node C")
    return Command(
        goto=END,
        update={"value": state["value"] + "c"}
    )

# Node D logic
def node_d(state: State):
    print("Node D")
    return Command(
        goto=END,
        update={"value": state["value"] + "d"}
    )

# Create and configure graph
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)
graph.set_entry_point("node_a")

# Compile the graph
app = graph.compile(checkpointer=memory)

# Set up initial input and config
config = {"configurable": {"thread_id": "1"}}
initial_state = {"value": ""}

# Step 1: Invoke graph to run until interrupt
first_result = app.invoke(initial_state, config=config, stream_mode="updates")
print("\n🔁 First result (before human resumes):")
print(first_result)

# Step 2: Simulate human input to resume
# Uncomment one of the two lines below depending on user choice
# resume_command = Command(resume="C")  # simulate choosing node_c
resume_command = Command(resume="D")    # simulate choosing node_d

second_result = app.invoke(resume_command, config=config, stream_mode="updates")
print("\n✅ Final result (after human resumes):")
print(second_result)

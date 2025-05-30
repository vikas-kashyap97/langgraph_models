
# LangGraph Models Repository

Welcome to the **LangGraph Models** repository by [vikas-kashyap97](https://github.com/vikas-kashyap97).  
This repository presents modular implementations for building complex LLM workflows using **[LangGraph](https://github.com/vikas-kashyap97/langgraph_models.git)**—an extension of LangChain for stateful, agent-based, and memory-aware reasoning.

---

## 📂 Repository Structure

This project contains **step-by-step LangGraph-based LLM applications**, categorized by feature complexity and real-world use cases.

---

## 📦 Module Overview

### `1_Introduction/`
Basic example of an LLM-powered ReAct agent using LangGraph.

- `react_agent_basic.py`: Minimal ReAct implementation for understanding core LangGraph concepts.

---

### `2_basic_reflection_system/`
A simple agent loop with self-reflection capabilities.

- `basic.py`: Defines a reflection loop logic.
- `chains.py`: Defines LangChain chains used within the loop.

---

### `3_structured_outputs/`
Demonstrates how to work with structured outputs from LLMs using Pydantic.

- `structured_outputs.py`: LLM output parsing and validation using Pydantic.
- `pydantic_outputs.json`: Sample output for schema reference.

---

### `4_reflexion_agent_system/`
More advanced reflexion loop with schema-enforced tools and agent planning.

- `chains.py`, `schema.py`: Tool and output definitions.
- `reflexion_graph.py`: Full graph implementation.
- `execute_tools.py`: Tool invocations and execution layer.

---

### `5_state_deepdive/`
Hands-on with LangGraph state management.

- `1_basic_state.py`: Introduction to LangGraph state handling.
- `2_complex_state.py`: Managing multiple variables and transitions across nodes.

---

### `6_react_agent/`
Advanced ReAct agent architecture with execution planning and modular reasoning.

- `agent_reason_runnable.py`: Defines reasoning logic as reusable runnables.
- `nodes.py`, `react_graph.py`, `react_state.py`: Complete node-graph pipeline.

---

### `7_chatbot/`
Multiple chatbot implementations with varying checkpoint and memory strategies.

- `1_basic_chatbot.py`: Stateless chatbot using LangGraph.
- `2_chatbot_with_tools.py`: Tool-augmented chatbot.
- `3_chat_with_in_memory_checkpointer.py`: Memory persistence using in-memory checkpoints.
- `4_chat_with_sqlite_checkpointer.py`: Chat persistence with SQLite.
- `checkpoint.sqlite`: Database for chat state saving.

---

### `8_human-in-the-loop/`
Integrating human feedback in LangGraph loops.

- `1_using_input()`: Capturing manual input mid-process.
- `2_command.py`, `3_resume.py`: Command flow controls.
- `5_multiturn_conversation.py`: Rich multi-turn human-AI dialogue.

---

### `9_RAG_agent/`
Retrieval-Augmented Generation (RAG) agents with progressive complexity.

- `1_basic.py`: Basic RAG agent flow.
- `2_classification_driven_agent.py`: RAG agent using classification to guide actions.
- `3_rag_powered_tool_calling.py`: Tool-augmented RAG agent using knowledge retrieval.
- `4_advanced_multi_step_reasoning.py`: Complex reasoning over retrieved knowledge.

---

### `10_multi_agent_architecture/`
Collaborative multi-agent systems for task delegation and supervision.

- `1_subgraphs.py`: Modular subgraph-based agent interactions.
- `2_supervisor_multiagent_workflow.py`: Supervisor-worker agent architecture.

---

### `11_streaming/`
Streaming outputs from LLM agents in real-time.

- `1_stream_events.py`: Event-driven stream interface for agent outputs.


## 🔧 Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📚 Official Resources

- [LangGraph Documentation](https://docs.langchain.com/langgraph/)
- [LangChain GitHub](https://github.com/hwchase17/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Official LangChain Examples](https://github.com/langchain-ai/langchain/tree/master/cookbook)

---

## 💡 Author

Developed with ❤️ by [vikas-kashyap97](https://github.com/vikas-kashyap97)  
Feel free to contribute or raise issues to enhance this LangGraph resource hub.

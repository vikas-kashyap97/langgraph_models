# multi_agent_workflow.py

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_groq import ChatGroq
import pprint

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Tools
tavily_search = TavilySearchResults(max_results=2)
python_repl_tool = PythonREPLTool()
python_repl_tool.invoke("x = 5; print(x)")  # Tool test

# Supervisor model
class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
                    "'enhancer' when user input requires clarification, expansion, or refinement, "
                    "'researcher' when additional facts, context, or data collection is necessary, "
                    "'coder' when implementation, computation, or technical problem-solving is required."
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )

# Supervisor node
def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:
    system_prompt = '''
        You are a workflow supervisor managing a team of three specialized agents: Prompt Enhancer, Researcher, and Coder.
        Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state
        and needs of the task. Provide a clear, concise rationale for each decision to ensure transparency in your
        decision-making process.

        **Team Members**:
        1. **Prompt Enhancer**: Always consider this agent first. They clarify ambiguous requests, improve poorly defined queries,
        and ensure the task is well-structured before deeper processing begins.
        2. **Researcher**: Specializes in information gathering, fact-finding, and collecting relevant data needed to address the user's request.
        3. **Coder**: Focuses on technical implementation, calculations, data analysis, algorithm development, and coding solutions.

        **Your Responsibilities**:
        1. Analyze each user request and agent response for completeness, accuracy, and relevance.
        2. Route the task to the most appropriate agent at each decision point.
        3. Maintain workflow momentum by avoiding redundant agent assignments.
        4. Continue the process until the user's request is fully and satisfactorily resolved.

        Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps,
        ultimately delivering complete and accurate solutions to user requests.
    '''
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Supervisor).invoke(messages)
    print(f"--- Workflow Transition: Supervisor → {response.next.upper()} ---")
    return Command(
        update={"messages": [HumanMessage(content=response.reason, name="supervisor")]},
        goto=response.next
    )

# Enhancer node
def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    system_prompt = '''
        You are a Query Refinement Specialist with expertise in transforming vague requests into precise instructions.
        Your responsibilities include:

        1. Analyzing the original query to identify key intent and requirements
        2. Resolving any ambiguities without requesting additional user input
        3. Expanding underdeveloped aspects of the query with reasonable assumptions
        4. Restructuring the query for clarity and actionability
        5. Ensuring all technical terminology is properly defined in context

        Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible.
    '''
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    enhanced_query = llm.invoke(messages)
    print(f"--- Workflow Transition: Prompt Enhancer → Supervisor ---")
    return Command(
        update={"messages": [HumanMessage(content=enhanced_query.content, name="enhancer")]},
        goto="supervisor"
    )

# Researcher node
def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    research_agent = create_react_agent(
        llm,
        tools=[tavily_search],
        state_modifier='''
            You are an Information Specialist with expertise in comprehensive research. Your responsibilities include:

            1. Identifying key information needs based on the query context
            2. Gathering relevant, accurate, and up-to-date information from reliable sources
            3. Organizing findings in a structured, easily digestible format
            4. Citing sources when possible to establish credibility
            5. Focusing exclusively on information gathering - avoid analysis or implementation

            Provide thorough, factual responses without speculation where information is unavailable.
        '''
    )
    result = research_agent.invoke(state)
    print(f"--- Workflow Transition: Researcher → Validator ---")
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="researcher")]},
        goto="validator"
    )

# Coder node
def code_node(state: MessagesState) -> Command[Literal["validator"]]:
    code_agent = create_react_agent(
        llm,
        tools=[python_repl_tool],
        state_modifier='''
            You are a coder and analyst. Focus on mathematical calculations, analyzing, solving math questions,
            and executing code. Handle technical problem-solving and data tasks.
        '''
    )
    result = code_agent.invoke(state)
    print(f"--- Workflow Transition: Coder → Validator ---")
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]},
        goto="validator"
    )

# Validator model
class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

# Validator node
def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    system_prompt = '''
        Your task is to ensure reasonable quality. 
        Specifically, you must:
        - Review the user's question (the first message in the workflow).
        - Review the answer (the last message in the workflow).
        - If the answer addresses the core intent of the question, even if not perfectly, signal to end the workflow with 'FINISH'.
        - Only route back to the supervisor if the answer is completely off-topic, harmful, or fundamentally misunderstands the question.

        - Accept answers that are "good enough" rather than perfect
        - Prioritize workflow completion over perfect responses
        - Give benefit of doubt to borderline answers

        Routing Guidelines:
        1. 'supervisor' Agent: ONLY for responses that are completely incorrect or off-topic.
        2. Respond with 'FINISH' in all other cases to end the workflow.
    '''
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]
    response = llm.with_structured_output(Validator).invoke(messages)
    goto = END if response.next == "FINISH" else "supervisor"
    if goto == END:
        print(" --- Transitioning to END ---")
    else:
        print(f"--- Workflow Transition: Validator → Supervisor ---")
    return Command(
        update={"messages": [HumanMessage(content=response.reason, name="validator")]},
        goto=goto
    )

# Graph assembly
graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("enhancer", enhancer_node)
graph.add_node("researcher", research_node)
graph.add_node("coder", code_node)
graph.add_node("validator", validator_node)
graph.add_edge(START, "supervisor")
app = graph.compile()

# Execution
if __name__ == "__main__":
    scenarios = [
        {"messages": [("user", "Weather in Chennai")]},
        {"messages": [("user", "Give me the 20th fibonacci number")]},
    ]

    for input_set in scenarios:
        print("\n\n--- NEW SCENARIO ---\n")
        for event in app.stream(input_set):
            for key, value in event.items():
                if not value:
                    continue
                last_msg = value.get("messages", [])[-1] if "messages" in value else None
                if last_msg:
                    pprint.pprint(f"Output from node '{key}':")
                    pprint.pprint(last_msg, indent=2, width=80)
                    print()

# rag_agent_groq.py

from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from typing import Annotated, Sequence, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.runnables.graph import MermaidDrawMethod

import os
from dotenv import load_dotenv
from IPython.display import Image, display

# Load environment variables (GROQ_API_KEY, GOOGLE_API_KEY)
load_dotenv()

# Setup embedding function
embedding_function = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# Define documents
docs = [
    Document(page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.", metadata={"source": "about.txt"}),
    Document(page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.", metadata={"source": "hours.txt"}),
    Document(page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.", metadata={"source": "membership.txt"}),
    Document(page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.", metadata={"source": "classes.txt"}),
    Document(page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).", metadata={"source": "trainers.txt"}),
    Document(page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.", metadata={"source": "facilities.txt"}),
]

# Load docs to vector DB
db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# Create tools
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_tool",
    "Information related to Gym History & Founder, Operating Hours, Membership Plans, Fitness Classes, Personal Trainers, and Facilities & Equipment of Peak Performance Gym",
)

@tool
def off_topic():
    """Catch all Questions NOT related to Peak Performance Gym's history, hours, membership plans, fitness classes, trainers, or facilities"""
    return "Forbidden - do not respond to the user"

tools = [retriever_tool, off_topic]

# State structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define agent function
def agent(state):
    messages = state["messages"]
    model = ChatGroq(model="llama3-8b-8192")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

# Control flow condition
def should_continue(state) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# Only run if executed directly
if __name__ == "__main__":
    # Optional: visualize
    try:
        display(
            Image(
                graph.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )
    except:
        print("Graph visualization not supported in this environment.")

    # Run test prompts
    print("Running off-topic example:")
    print(
        graph.invoke(input={"messages": [HumanMessage(content="How will the weather be tomorrow?")]})
    )

    print("\nRunning relevant question:")
    print(
        graph.invoke(input={"messages": [HumanMessage(content="Who is the owner and what are the timings?")]})
    )

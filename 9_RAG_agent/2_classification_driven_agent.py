from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END

from typing import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --------------------- Setup Embeddings and Docs ---------------------

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

docs = [
    Document(
        page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.",
        metadata={"source": "about.txt"}
    ),
    Document(
        page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.",
        metadata={"source": "hours.txt"}
    ),
    Document(
        page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.",
        metadata={"source": "membership.txt"}
    ),
    Document(
        page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.",
        metadata={"source": "classes.txt"}
    ),
    Document(
        page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).",
        metadata={"source": "trainers.txt"}
    ),
    Document(
        page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.",
        metadata={"source": "facilities.txt"}
    )
]

db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# --------------------- Prompt & RAG Chain ---------------------

template = """ 
Answer the question based only on the following context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatGroq(model="llama-3.1-8b-instant")
rag_chain = prompt | llm

# --------------------- Agent State ---------------------

class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    on_topic: str

class GradeQuestion(BaseModel):
    """Boolean value to check whether a question is related to the Peak Performance Gym"""
    score: str = Field(description="Question is about gym? If yes -> 'Yes' if not -> 'No' ")

# --------------------- Classifier ---------------------

def question_classifier(state: AgentState): 
    question = state["messages"][-1].content
    system = """You are a classifier that determines whether a user's question is about one of these topics:
    1. Gym History & Founder
    2. Operating Hours
    3. Membership Plans 
    4. Fitness Classes
    5. Personal Trainers
    6. Facilities & Equipment

    If YES, respond with 'Yes'. Else, 'No'.
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "User question: {question}")]
    )

    structured_llm = ChatGroq(model="llama-3.1-8b-instant").with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    
    state["on_topic"] = result.score
    return state

# --------------------- Routing ---------------------

def on_topic_router(state: AgentState): 
    return "on_topic" if state["on_topic"].lower() == "yes" else "off_topic"

# --------------------- RAG Functions ---------------------

def retrieve(state: AgentState):
    question = state["messages"][-1].content
    state["documents"] = retriever.invoke(question)
    return state

def generate_answer(state: AgentState): 
    question = state["messages"][-1].content
    documents = state["documents"]
    answer = rag_chain.invoke({"context": documents, "question": question})
    state["messages"].append(answer)
    return state

def off_topic_response(state: AgentState): 
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question!"))
    return state

# --------------------- LangGraph Workflow ---------------------

workflow = StateGraph(AgentState)

workflow.add_node("topic_decision", question_classifier)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("off_topic_response", off_topic_response)

workflow.add_conditional_edges("topic_decision", on_topic_router, {
    "on_topic": "retrieve",
    "off_topic": "off_topic_response"
})

workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)

workflow.set_entry_point("topic_decision")
graph = workflow.compile()

# --------------------- Run the Agent ---------------------

if __name__ == "__main__":
    print("\n=== Query 1: On-topic ===")
    result = graph.invoke({
        "messages": [HumanMessage(content="Who is the owner and what are the timings?")]
    })
    for m in result["messages"]:
        print(m.content)

    print("\n=== Query 2: Off-topic ===")
    result = graph.invoke({
        "messages": [HumanMessage(content="What does the company Apple do?")]
    })
    for m in result["messages"]:
        print(m.content)

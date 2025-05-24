import os
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, TypedDict
from dotenv import load_dotenv
load_dotenv()


# Load embedding model
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load documents
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
    ),
]

# Create vectorstore
db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

# Prompt template
template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

Chathistory: {history}

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Use ChatGroq model
llm = ChatGroq(model="llama3-8b-8192")
rag_chain = prompt | llm

# Agent state type
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage

# Question relevance grading
class GradeQuestion(BaseModel):
    score: str = Field(description="Is the question on-topic? Respond with 'Yes' or 'No'.")

# Question rewriter node
def question_rewriter(state: AgentState):
    print("Entering question_rewriter")

    # Ensure all required keys are initialized
    state.setdefault("messages", [])
    state.update({
        "documents": [],
        "on_topic": "",
        "rephrased_question": "",
        "proceed_to_generate": False,
        "rephrase_count": 0,
    })

    # Append the new question to the message history if not already present
    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    # Rephrase if there is prior history
    if len(state["messages"]) > 1:
        messages = [
            SystemMessage(content="You are a helpful assistant that rephrases the user's question to be standalone."),
            *state["messages"][:-1],
            HumanMessage(content=state["question"].content)
        ]
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        response = ChatGroq(model="llama3-8b-8192").invoke(rephrase_prompt.format())
        state["rephrased_question"] = response.content.strip()
    else:
        state["rephrased_question"] = state["question"].content

    return state

# Question classifier
def question_classifier(state: AgentState):
    classifier_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Classify whether the question is about:
1. Gym History & Founder
2. Operating Hours
3. Membership Plans
4. Fitness Classes
5. Personal Trainers
6. Facilities & Equipment
7. Anything else about Peak Performance Gym

If YES, reply 'Yes'. Else, 'No'."""),
        HumanMessage(content=f"User question: {state['rephrased_question']}")
    ])
    structured_llm = ChatGroq(model="llama3-8b-8192").with_structured_output(GradeQuestion)
    grader = classifier_prompt | structured_llm
    result = grader.invoke({})
    state["on_topic"] = result.score.strip()
    return state

# Routing based on topic
def on_topic_router(state: AgentState):
    return "retrieve" if state["on_topic"].lower() == "yes" else "off_topic_response"

# Retriever node
def retrieve(state: AgentState):
    state["documents"] = retriever.invoke(state["rephrased_question"])
    return state

# Document grader
class GradeDocument(BaseModel):
    score: str = Field(description="Is this document relevant to the question? Reply 'Yes' or 'No'.")

def retrieval_grader(state: AgentState):
    llm = ChatGroq(model="llama3-8b-8192")
    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        grading_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are checking if the document is relevant to the user's question."),
            HumanMessage(content=f"User question: {state['rephrased_question']}\nDocument:\n{doc.page_content}")
        ])
        result = (grading_prompt | structured_llm).invoke({})
        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)

    state["documents"] = relevant_docs
    state["proceed_to_generate"] = bool(relevant_docs)
    return state

# Route to next step
def proceed_router(state: AgentState):
    if state["proceed_to_generate"]:
        return "generate_answer"
    elif state["rephrase_count"] >= 2:
        return "cannot_answer"
    return "refine_question"

# Refine the question
def refine_question(state: AgentState):
    question = state["rephrased_question"]
    system_msg = SystemMessage(content="Refine the question to improve document retrieval.")
    human_msg = HumanMessage(content=f"Original: {question}\nPlease refine slightly.")
    refine_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
    refined_response = ChatGroq(model="llama3-8b-8192").invoke(refine_prompt.format())
    state["rephrased_question"] = refined_response.content.strip()
    state["rephrase_count"] += 1
    return state

# Generate final answer
def generate_answer(state: AgentState):
    response = rag_chain.invoke({
        "history": state["messages"],
        "context": state["documents"],
        "question": state["rephrased_question"]
    })
    state["messages"].append(AIMessage(content=response.content.strip()))
    return state

# Handle off-topic questions
def off_topic_response(state: AgentState):
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question!"))
    return state

# Handle no-answer fallback
def cannot_answer(state: AgentState):
    state["messages"].append(AIMessage(content="I'm sorry, but I cannot find the information you're looking for."))
    return state

# Graph construction
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
workflow = StateGraph(AgentState)
workflow.set_entry_point("question_rewriter")

workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("refine_question", refine_question)
workflow.add_node("cannot_answer", cannot_answer)
workflow.add_node("off_topic_response", off_topic_response)

workflow.add_edge("question_rewriter", "question_classifier")
workflow.add_conditional_edges("question_classifier", on_topic_router, {
    "retrieve": "retrieve",
    "off_topic_response": "off_topic_response"
})
workflow.add_edge("retrieve", "retrieval_grader")
workflow.add_conditional_edges("retrieval_grader", proceed_router, {
    "generate_answer": "generate_answer",
    "refine_question": "refine_question",
    "cannot_answer": "cannot_answer"
})
workflow.add_edge("refine_question", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.add_edge("off_topic_response", END)

graph = workflow.compile(checkpointer=checkpointer)

# --- TESTING (Example Calls) ---
if __name__ == "__main__":
    test_input = {"question": HumanMessage(content="Who founded Peak Performance Gym?")}
    output = graph.invoke(input=test_input, config={"configurable": {"thread_id": 1}})
    print(output["messages"][-1].content)

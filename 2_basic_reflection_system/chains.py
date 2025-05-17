from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly intelligent and helpful AI assistant. "
            "Generate the best possible responses based on user input. "
            "Always aim for clarity, relevance, and helpfulness. "
            "If the user provides critique or correction, acknowledge it respectfully and adapt accordingly."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert reviewer and critic of AI-generated content. "
            "Your job is to reflect on the assistant’s last response and evaluate its accuracy, clarity, completeness, and tone. "
            "Identify any flaws, oversights, or improvements that could be made. "
            "Be objective, concise, and constructive in your reflection."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

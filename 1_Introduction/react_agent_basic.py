from langchain_groq import ChatGroq
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(
    model="llama-3.1-8b-instant",
)


search_tool = TavilySearchResults()
tools = [search_tool]


agent = initialize_agent(
    tools=tools,
    llm=model,
    agent="zero-shot-react-description",
    verbose=True
)


result = agent.invoke("What is the capital of India?")
print(result)


# result = model.invoke("Give me todays temp of hardoi, uttar pradesh.")
# print(result)


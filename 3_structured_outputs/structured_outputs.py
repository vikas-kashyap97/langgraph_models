from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from typing import Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# --- 1. Pydantic Model Example ---

class Country(BaseModel):
    """Information about a country"""
    name: str = Field(description="Name of the country")
    language: str = Field(description="Language of the country")
    capital: str = Field(description="Capital of the country")

# Structured output with Pydantic model
structured_country_llm = llm.with_structured_output(Country)

# Test
response_country = structured_country_llm.invoke("Tell me about France")
print("Response - Country Info:\n", response_country)


# --- 2. TypedDict Example ---

class Joke(TypedDict):
    """Joke to tell user."""
    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_joke_llm = llm.with_structured_output(Joke)

# Test
response_joke = structured_joke_llm.invoke("Tell me a joke about cats")
print("\nResponse - Joke Info:\n", response_joke)


# --- 3. JSON Schema Example ---

json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}

structured_json_llm = llm.with_structured_output(json_schema)

# Test
response_json_joke = structured_json_llm.invoke("Tell me a joke about cats")
print("\nResponse - Joke from JSON Schema:\n", response_json_joke)

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_groq import ChatGroq  # Changed here
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
# Output parsers
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])
parser = JsonOutputToolsParser(return_id=True)

# Actor Agent Prompt 
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# Instruction to first responder
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

# ⬇️ REPLACED LLM: ChatOpenAI -> ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant")

# First responder chain
first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice='AnswerQuestion'
)

# Validation parser
validator = PydanticToolsParser(tools=[AnswerQuestion])

# Revisor section
revise_instructions = """Revise your previous answer using the new information.
- You MUST include citations in the form of a list of URLs in a field called `references` (e.g., ["https://example1.com", "https://example2.com"])
- The answer itself should still include inline numerical citations like [1], [2] to match the references.
- Do NOT include the full references section inside the answer field. Use the `references` field (list of URLs) for that.
- Keep your final answer under 250 words."""


# Revisor chain
revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(
    tools=[ReviseAnswer],
    tool_choice="ReviseAnswer"
)

# response = first_responder_chain.invoke({
#     "messages": [HumanMessage(content="AI Agents taking over content creation")]
# })
# print(response)

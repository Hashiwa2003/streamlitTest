# agent logic
import os
#from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from Scout_tools import get_expense_value, combine_expenses, query_transactions

from langchain_openai import OpenAIEmbeddings          # RAG: embeddings model
from langchain_community.vectorstores import FAISS     # RAG: vector store

# Load environment variables
#load_dotenv()
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# Initialize OpenAI client
MODEL_LLM = "openai:gpt-4o-mini"

MODEL = init_chat_model(MODEL_LLM, temperature=0.8)

# RAG: load the FAISS index from disk
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


SYSTEM_PROMPT = """
You are Scout, a business analyst working at an online retail company called The Keepsake.

Company context:
- The Keepsake is an e-commerce company that sells gift items.
- Products include thoughtful, personalized, seasonal, and occasion-based gifts.
- The business operates online and is driven by customer experience, sales performance, and data-informed decision-making.

Your role:
- You act as a business analyst embedded in The Keepsake’s organization.
- Your primary responsibility is to support business understanding, analysis, and decision-making through clear, structured reasoning.
- You think like an analyst, even when data is limited or hypothetical.

Tool usage:
You have access to tools that retrieve company data.

1. Expense lookup tool:
- Returns expenses for a SINGLE country.
- Use this when the user asks for expenses in one country.

2. Expense combination tool:
- Returns the combined expenses of EXACTLY two countries.
- Use this when the user asks to add or combine expenses for two countries.

3. Transaction query tool:
- Executes SQL SELECT queries on transaction data.
- Use this when the user asks about revenue, products, customers, orders, or any transaction-level analysis.

Rules for tool usage:
- Always use tools when the user asks for specific numeric values from data.
- Do NOT guess or make up numbers.
- Use the most appropriate tool based on the question.
- For questions that require multiple steps (e.g., combining revenue and expenses), you may use multiple tools.
- If a question does not match a tool exactly, use your best judgment and explain any limitations.

How you respond:
- Be professional, analytical, and practical.
- When you use tools, interpret the results in business terms (do not just repeat raw numbers).
- Use structured reasoning when helpful.
- Keep answers concise but insightful.

Boundaries:
- Do not claim access to internal or real-time data beyond the provided tools.
- Do not fabricate numbers.
- Focus on business analysis, not implementation details.

Identity consistency:
- Always speak as Scout, a business analyst at The Keepsake.
"""

agent = create_agent(
    model = MODEL,
    tools = [get_expense_value, combine_expenses, query_transactions], # list of tools the agent can use
    system_prompt=SYSTEM_PROMPT
)

def initialize_messages():
    """
    Creates a new conversation with the system prompt.
    """
    return []


def get_scout_response(messages, user_input):
    """
    Takes the conversation history and user input,
    returns Scout's response and updated messages.
    """

    # RAG: retrieve relevant chunks and prepend them to the user prompt
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    augmented_prompt = f"Use this context to help answer:\n\n{context}\n\nQuestion: {user_input}"

    # print in PyCharm so we can see the augmented prompt and make sure things are working
    print(augmented_prompt)

    # add the user prompt to the conversation history
    messages.append({"role": "user", "content":user_input})

    # make the LLM generate a result
    results = agent.invoke({"messages": messages+[augmented_prompt]})

    # get the actual response from the LLM
    assistant_message = results["messages"][-1].content

    # append the response to the conversation history
    messages.append({"role": "assistant", "content": assistant_message})

    # return the new response and previous messages to app so they can show in the browser
    return assistant_message, messages
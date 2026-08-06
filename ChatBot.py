import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class ChatState(TypedDict):
    message: str
    response: str

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.7)

def chatbot(state: ChatState) -> ChatState:
    reply = llm.invoke(state["message"])
    state["response"] = reply.content
    return state

graph = StateGraph(ChatState)

graph.add_node("chatbot",chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Chat ended.")
        break

    state: ChatState = {
        "message": user,
        "response": ""
    }

    result = app.invoke(state)
    print("Bot:", result["response"])
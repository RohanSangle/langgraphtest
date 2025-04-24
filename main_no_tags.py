from typing import Annotated, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define simplified state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Predefined careers (attributes removed)
initial_careers = [
    "Software Engineer",
    "Designer",
    "Manager",
    "Construction Worker",
    "Teacher"
]

# Initialize Groq LLM (replace with your API key)
llm = ChatGroq(model="llama3-8b-8192", api_key="GROQ_API_KEY", temperature=0.7)

# Human assistance tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    return "Waiting for human response"

tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

# Chatbot node
def chatbot(state: State):
    messages = state["messages"]
    # Count user responses (ToolMessages)
    tool_message_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))
    
    if tool_message_count < 5:
        # Ask the next question
        if not messages:
            prompt = "You are a career counselor. Start by asking the user about their interests and preferences to help them explore career options."
        else:
            conversation_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
            prompt = (
                "You are a career counselor. Based on the conversation so far: "
                f"{conversation_str}, ask the next question to help the user explore "
                "their career interests and preferences."
            )
        response = llm.invoke(prompt)
        tool_call_id = str(uuid.uuid4())
        message = AIMessage(
            content=response.content,
            tool_calls=[{"id": tool_call_id, "name": "human_assistance", "args": {"query": response.content}}]
        )
    else:
        # After 5 questions, suggest 3 careers
        conversation_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        careers_list = ", ".join(initial_careers)
        prompt = (
            "Based on the following conversation, suggest 3 careers from this list "
            f"that might be a good fit for the user: {careers_list}. "
            "Please list the careers clearly in your response. "
            f"Conversation: {conversation_str}"
        )
        response = llm.invoke(prompt)
        message = AIMessage(content=response.content)
    return {"messages": [message]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_after=["tools"])

# Define request model for /chat endpoint
class ChatRequest(BaseModel):
    thread_id: str
    message: str

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the Career Guidance Chatbot API. Use /start to begin."}

@app.get("/start")
async def start_conversation():
    try:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Starting conversation with thread_id: {thread_id}")
        for _ in graph.stream({"messages": []}, config):
            pass
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            logger.error("No messages in state after stream")
            raise HTTPException(status_code=500, detail="No messages generated")
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.error("Last message is not an AIMessage with tool_calls")
            raise HTTPException(status_code=500, detail="Unexpected state after start")
        return {
            "thread_id": thread_id,
            "message": last_message.content,
            "is_ongoing": True
        }
    except Exception as e:
        logger.error(f"Error in start_conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        logger.info(f"Processing chat with thread_id: {request.thread_id}")
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        last_tool_call = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                last_tool_call = msg.tool_calls[0]
                break
        if not last_tool_call:
            logger.error("No pending tool call found")
            raise HTTPException(status_code=400, detail="No pending tool call")
        tool_message = ToolMessage(
            content=request.message,
            tool_call_id=last_tool_call["id"]
        )
        state_dict = state.values.copy()
        state_dict["messages"].append(tool_message)
        graph.update_state(config, state_dict)
        for _ in graph.stream(None, config):
            pass
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                logger.info(f"Returning AIMessage: {msg}")
                return {"message": msg.content, "is_ongoing": bool(msg.tool_calls)}
        logger.error("No AIMessage found in state")
        raise HTTPException(status_code=500, detail="No AIMessage found in state")
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
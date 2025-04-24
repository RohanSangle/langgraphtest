from typing import Annotated, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
# from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.tools import tool
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
    allow_origins=["http://127.0.0.1:8080"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    careers: List[Dict[str, List[str]]]
    remaining_careers: List[str]
    disliked_attributes: List[str]

# Predefined careers
initial_careers = [
    {"name": "Software Engineer", "required_attributes": ["tech"]},
    {"name": "Designer", "required_attributes": ["creativity"]},
    {"name": "Manager", "required_attributes": ["leadership"]},
    {"name": "Construction Worker", "required_attributes": ["physical_work"]},
    {"name": "Teacher", "required_attributes": ["communication"]}
]

# Initialize Groq LLM (replace with your API key)
llm = ChatGroq(model="llama3-8b-8192", api_key="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", temperature=0.7)

# Human assistance tool
@tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]

def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    return "Waiting for human response"

tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

# Chatbot node
def chatbot(state: State):
    logger.info("Entering chatbot node")
    updates = {}  # Collect all state changes here
    
    # Initialize state if careers are not present
    if not state.get("careers"):
        updates["careers"] = initial_careers
        updates["remaining_careers"] = [career["name"] for career in initial_careers]
        updates["disliked_attributes"] = []
    
    local_state = {**state, **updates}

    remaining_careers = local_state["remaining_careers"]
    # remaining_careers = state.get("remaining_careers", [])
    if len(remaining_careers) <= 3:
        prompt = (
            f"The remaining careers are {remaining_careers}. "
            f"Based on the conversation so far: {state['messages']}, "
            "provide a concluding message summarizing why these might be good fits."
        )
        message = llm.invoke(prompt)
    else:
        conversation = state["messages"]
        prompt = (
            "You are a career counselor. Based on the conversation so far: "
            f"{conversation}, ask the next question to help the user narrow down "
            "their career options from the following list: "
            f"{remaining_careers}. If the user's previous responses "
            "are neutral or unclear (e.g., 'I don't know'), ask a follow-up question "
            "on the same topic from a different perspective to get specific feedback."
        )
        response = llm.invoke(prompt)
        # message.tool_calls = [{"name": "human_assistance", "arguments": {"query": message.content}}]
        tool_call_id = str(uuid.uuid4())
        message = AIMessage(
            content=response.content,
            tool_calls=[{"id": tool_call_id, "name": "human_assistance", "args": {"query": response.content}}]
        )
    updates["messages"] = [message]
    logger.info(f"Chatbot returning updates: {updates}")
    # return {"messages": [message]}
    return updates

# Processor node
def processor(state: State):
    logger.info("Entering processor node")
    updates = {}  # Collect all state changes here

    # Initialize disliked_attributes if not present
    if "disliked_attributes" not in state:
        updates["disliked_attributes"] = []
    else:
        updates["disliked_attributes"] = state["disliked_attributes"]

    conversation = state["messages"]
    prompt = (
        "Based on the following conversation, list any attributes the user has "
        f"expressed dislike for (e.g., tech, creativity, leadership): {conversation}"
    )
    response = llm.invoke(prompt)
    new_disliked = response.content.split(",") if response.content.strip() else []
    updates["disliked_attributes"] = list(set(updates["disliked_attributes"] + new_disliked))
    
    # Get current remaining careers or initialize from all careers
    remaining_careers = state.get("remaining_careers", [career["name"] for career in state.get("careers", initial_careers)])
    
    # Update remaining careers based on disliked attributes
    updates["remaining_careers"] = [
        career_name for career_name in remaining_careers
        if not any(
            attr in updates["disliked_attributes"]
            for attr in next(c["required_attributes"] for c in state.get("careers", initial_careers) if c["name"] == career_name)
        )
    ]
    
    logger.info(f"Processor updated state: {updates}")
    return updates

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("processor", processor)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "processor")
graph_builder.add_edge("processor", "chatbot")

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
        # Generate a unique thread ID
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Starting conversation with thread_id: {thread_id}")
        # Run the graph until it hits the first interrupt
        for _ in graph.stream({"messages": []}, config):
            pass  # Advances the graph to the interrupt
        
        # Get the current state
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            logger.error("No messages in state after stream")
            raise HTTPException(status_code=500, detail="No messages generated")
        last_message = messages[-1]
        # logger.info(f"Last message: {last_message}")
        logger.info(f"Messages in state: {[type(m).__name__ for m in messages]}")

        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                last_ai_message = msg
                break
            # else:
            #     logger.error("Last message is not an AIMessage with tool_calls")
            #     raise HTTPException(status_code=500, detail="Unexpected state after start")
        if not last_ai_message:
            logger.error("No AIMessage with tool calls found")
            raise HTTPException(status_code=500, detail="No AIMessage with tool calls found")
        # Return the chatbot's message and thread_id
        return {
            "thread_id": thread_id,
            "message": last_ai_message.content,
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
        
        # Get the current state
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        
        # Find the last tool call for human_assistance
        last_tool_call = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                last_tool_call = msg.tool_calls[0]
                break
        if not last_tool_call:
            logger.error("No pending tool call found")
            raise HTTPException(status_code=400, detail="No pending tool call")
        
        # Create a ToolMessage with the user's input
        tool_message = ToolMessage(
            content=request.message,
            tool_call_id=last_tool_call["id"]
        )
        
        # Update the state with the user's input
        state_dict = state.values.copy()
        state_dict["messages"].append(tool_message)
        graph.update_state(config, state_dict)
        
        # Resume the graph until the next interrupt or end
        for _ in graph.stream(None, config):
            pass  # Advances the graph
        
        # Get the updated state
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        
        # Find the last AI message
        for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    logger.info(f"Returning AIMessage: {msg}")
                    return {"message": msg.content, "is_ongoing": bool(msg.tool_calls)}
        logger.error("No AIMessage found in state")
        raise HTTPException(status_code=500, detail="No AIMessage found in state")
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
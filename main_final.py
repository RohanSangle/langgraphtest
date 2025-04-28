import os
import uuid
from typing import TypedDict, List, Dict, Optional, Any
from typing_extensions import Annotated
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


# Define HumanTool
@tool
def HumanTool(query: str) -> str:
    """Fetch the latest prompt from the user."""
    return "Waiting for user response"

tools = [HumanTool]

# --- 1) Define shared state schema for the graph ---
class CareerState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    careers: List[str]
    skewed_state: Dict[str, List[str]]
    conversation_number: int
    last_decision: Dict[str, str]
    generate_new_response: Optional[str]
    answer_question_response: Optional[str]

# --- 2) Initialize LLM client ---
os.environ.setdefault("GROQ_API_KEY", "YOUR_GROQ_API_KEY_FROM_.ENV")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=None,
).bind_tools(tools)


# llm_with_tools = llm.bind_tools(tools)

# --- 3) Build the LangGraph StateGraph ---
graph = StateGraph(CareerState)

def start_node(state: CareerState) -> Dict[str, Any]:
    print("[NODE:START] Entering start_node with state:", state)
    prompt = (
        f"You are a career counselor in India. The student is interested in these careers: {', '.join(state['careers'])}. "
        "Act as a personalized counselor and ask relevant questions to guide them."
    )
    response = llm.invoke([{"role": "system", "content": prompt}])
    # response = llm.invoke(prompt)
    # return {"messages": [{"role": "assistant", "content": response.content}]}
    # result = {"messages": [AIMessage(content=response.content)]}
    # print("[NODE:START] Exiting start_node with result:", result)
    # return result
    tool_call_id = str(uuid.uuid4())
    message = AIMessage(
        content=response.content,
        tool_calls=[{"id": tool_call_id, "name": "HumanTool", "args": {"query": response.content}}]
    )
    print("[NODE:START] Exiting start_node with result:", {"messages": [message]})
    return {"messages": [message]}

def human_loop_node(state: CareerState) -> Dict[str, Any]:
    # Pause for user input; frontend drives the next transition
    print("[NODE:HUMAN_LOOP] Pausing at human_loop_node with state:", state)
    return state

def analyser_node(state: CareerState) -> Dict[str, str]:
    print("[NODE:ANALYSER] Entering analyser_node with state:", state)
    user_msg = state['messages'][-1].content
    print("[NODE:ANALYSER] User message:", user_msg)
    classification_prompt = (
        "Analyze the user's message to determine if it answers a previous question or asks a new one. "
        "Return a JSON object with 'answer' and 'question' fields. "
        "The 'answer' field should capture the part of the message responding to previous questions (e.g., interests, goals, skills). "
        "The 'question' field should capture any new query the user asks. "
        "If the message contains no question, set 'question' to an empty string. "
        "If the message contains no answer, set 'answer' to an empty string. "
        "Examples: "
        "- Message: 'I like engineering and design. What’s the best option?' → {'answer': 'I like engineering and design', 'question': 'What’s the best option?'} "
        "- Message: 'I’m interested in tech' → {'answer': 'I’m interested in tech', 'question': ''} "
        "- Message: 'What are entrance exams for Medicine?' → {'answer': '', 'question': 'What are entrance exams for Medicine?'} "
        f"User message: {user_msg}"
    )
    res = llm.invoke([
        {"role": "system", "content": classification_prompt},
        {"role": "user", "content": user_msg}
    ])
   
    try:
        import json
        parsed = json.loads(res.content)
        #Validate parsed output
        if not isinstance(parsed, dict) or 'answer' not in parsed or 'question' not in parsed:
            raise ValueError("Invalid JSON format")
    except (json.JSONDecodeError, ValueError):
        parsed = {"answer": user_msg, "question": ""}
    print("[NODE:ANALYSER] Classification result:", parsed)
    return {"last_decision": {"answer": parsed.get('answer', ''), "question": parsed.get('question', '')}}


def process_input_node(state: CareerState) -> Dict[str, Any]:
    # """Handles generating a new message and answering a question in parallel if both are present."""

    print("[NODE:PROCESS_INPUT] Entering process_input_node with state:", state)
    
    # Extract the user's answer from the last decision
    answer = state['last_decision'].get('answer', '')
    question = state['last_decision'].get('question', '')
    print("[NODE:PROCESS_INPUT] Last decision - answer:", answer, "question:", question)
    
    updates = {}
    
    if answer:
        # Create a prompt for the LLM to generate a follow-up response
        prompt = (
            "You are a career counselor in India helping a student explore career options. "
            "The student has just provided some information about their background. "
            "Acknowledge their input and ask a specific follow-up question to guide them further. "
            f"Student's response: '{answer}'. "
            "Keep the tone friendly and encouraging."
        )
        
        # Call the LLM to generate a response
        response = llm.invoke([{"role": "system", "content": prompt}])
        generated_response = response.content.strip()
        
        # Fallback if the LLM returns an empty response
        if not generated_response:
            generated_response = "Thanks for letting me know! Could you tell me more about your interests or favorite subjects?"
        
        print("[NODE:PROCESS_INPUT] Generate new response:", generated_response)
        updates["generate_new_response"] = generated_response
    
    if question:
        # Handle user questions if present (not applicable here, but included for completeness)
        response = llm.invoke([
            {"role": "system", "content": "Answer the user's question concisely and accurately."},
            {"role": "user", "content": question}
        ])
        updates["answer_question_response"] = response.content.strip()
    
    print("[NODE:PROCESS_INPUT] Exiting process_input_node with updates:", updates)
    return updates

def evaluate_node(state: CareerState) -> Dict[str, Any]:
    print("[NODE:EVALUATE] Entering evaluate_node with state:", state)
    # Prune careers with negative feedback
    negatives = state['skewed_state'].get('negative', [])
    
    filtered = [c for c in state['careers'] if not any(neg.lower() in c.lower() for neg in negatives)]
    result = {
        "careers": filtered,
        "conversation_number": state['conversation_number'] + 1
    }
    print("[NODE:EVALUATE] Exiting evaluate_node with result:", result)
    return result

def new_msg_node(state: CareerState) -> Dict[str, Any]:
    print("[NODE:NEW_MSG] Entering new_msg_node with state:", state)
    generate_resp = state.get('generate_new_response', '')
    answer_resp = state.get('answer_question_response', '')
    combined_msg = f"{generate_resp}\n{answer_resp}".strip()

    tool_call_id = str(uuid.uuid4())
    message = AIMessage(
        content=combined_msg,
        tool_calls=[{"id": tool_call_id, "name": "HumanTool", "args": {"query": combined_msg}}]
    )

    print("[NODE:NEW_MSG] Combined message:", combined_msg)
    # return {
    #     "messages": [message],
    #     "generate_new_response": None,
    #     "answer_question_response": None
    # }
    result = {
        "messages": [message],
        "generate_new_response": None,
        "answer_question_response": None
    }
    print("[NODE:NEW_MSG] Exiting new_msg_node with result:", result)
    return result

def conclude_node(state: CareerState) -> Dict[str, Any]:
    print("[NODE:CONCLUDE] Entering conclude_node with state:", state)
    prompt = f"Based on our discussion, these careers suit you: {', '.join(state['careers'])}."
    result = {"messages": [AIMessage(content=prompt)]}
    print("[NODE:CONCLUDE] Exiting conclude_node with result:", result)
    return result

# Register nodes
graph.add_node("start", start_node)
graph.add_node("human_loop", human_loop_node)
graph.add_node("tools", ToolNode(tools=tools))  # Add tools node
graph.add_node("analyser", analyser_node)
graph.add_node("process_input", process_input_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("new_msg", new_msg_node)
graph.add_node("conclude", conclude_node)

# Define edges and conditionals
graph.add_edge(START, "start")
graph.add_edge("start", "tools")
graph.add_edge("tools", "human_loop")
graph.add_edge("human_loop", "analyser")
graph.add_edge("analyser", "process_input")
graph.add_edge("process_input", "evaluate")
graph.add_conditional_edges(
    "evaluate",
    lambda s: "new_msg" if (len(s['careers']) > 3 or s['conversation_number'] < 5) else "conclude",
    {"new_msg": "new_msg", "conclude": "conclude"}
)

graph.add_edge("new_msg", "tools")
graph.set_finish_point("conclude")

# Compile graph to an app callable
checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer, interrupt_after=["tools"])
print("[GRAPH] Graph compiled with interrupt_after:", ["tools"])

# --- 4) FastAPI wrapper ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all
    allow_headers=["*"],  # Allow all headers
)

# Initial career options
INITIAL_CAREERS = [
    "Engineering (e.g., Mechanical, Electrical, Civil)",
    "Medicine (MBBS, BDS)",
    "Bachelor of Computer Applications (BCA)",
    "Bachelor of Business Administration (BBA)",
    "Chartered Accountancy (CA)",
    "Hotel Management",
    "Law (LLB)",
    "Design (Fashion, Graphic)",
    "Bachelor of Science (B.Sc.)",
    "Journalism and Mass Communication"
]

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    message: str
    is_ongoing: bool

class StartResponse(BaseModel):
    thread_id: str
    message: str
    is_ongoing: bool

@app.get("/start", response_model=StartResponse)
def start_conversation():
    # Create a new state
    tid = str(uuid.uuid4())
    init_state: CareerState = {
        "messages": [],
        "careers": INITIAL_CAREERS.copy(),
        "skewed_state": {"positive": [], "negative": []},
        "conversation_number": 0,
        "last_decision": {"answer": "", "question": ""},
        "generate_new_response": None,
        "answer_question_response": None
    }

    print("[START] Initial state:", init_state)
    print("[START] Invoking graph with thread_id:", tid)

    for _ in compiled_graph.stream(init_state, config={"configurable": {"thread_id": tid}}):
        pass  # Advance to tools interrupt
    state = compiled_graph.get_state({"configurable": {"thread_id": tid}})
    messages = state.values.get("messages", [])
    # last_message = messages[-1] if messages else AIMessage(content="")

    if not messages:
        print("[START] No messages in state after stream")
        raise HTTPException(status_code=500, detail="No messages generated")
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai_message = msg
            break
    if not last_ai_message:
        print("[START] No AIMessage with tool calls found")
        raise HTTPException(status_code=500, detail="No AIMessage with tool calls found")

    print("[START] State after stream:", state.values)
    return StartResponse(thread_id=tid, message=last_ai_message.content, is_ongoing=True)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    config = {"configurable": {"thread_id": req.thread_id}}
    print("[CHAT] Received request with thread_id:", req.thread_id, "message:", req.message)
    # print("[CHAT] Raw checkpointer state:", checkpointer.get(config))

    state = compiled_graph.get_state(config)
    if not state:
        print("[CHAT] Thread not found for thread_id:", req.thread_id)
        raise HTTPException(status_code=404, detail="Thread not found")
    
    messages = state.values.get("messages", [])
    last_tool_call = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_tool_call = msg.tool_calls[0]
            break
    if not last_tool_call:
        print("[CHAT] No pending tool call found")
        raise HTTPException(status_code=400, detail="No pending tool call")
    
    # Resolve the tool call with user input
    tool_message = ToolMessage(
        content=req.message,
        tool_call_id=last_tool_call["id"]
    )
    state_dict = state.values.copy()
    state_dict["messages"].append(tool_message)
    compiled_graph.update_state(config, state_dict)
    print("[CHAT] State after appending tool message:", state_dict)
    
    # Resume the graph until next interrupt or end
    for _ in compiled_graph.stream(None, config):
        pass
    state = compiled_graph.get_state(config)
    print("[CHAT] State after resume:", state.values)
    
    # Return the latest AI message
    messages = state.values.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            ongoing = len(state.values['careers']) > 3 or state.values['conversation_number'] < 5
            print("[CHAT] Returning response:", {"message": msg.content, "is_ongoing": ongoing})
            return ChatResponse(message=msg.content, is_ongoing=ongoing)
    print("[CHAT] No AIMessage found")
    raise HTTPException(status_code=500, detail="No AIMessage found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
from langchain_core.messages import AIMessage, HumanMessage


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
os.environ.setdefault("GROQ_API_KEY", "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")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=None,
)

# --- 3) Build the LangGraph StateGraph ---
graph = StateGraph(CareerState)

def start_node(state: CareerState) -> Dict[str, Any]:
    print("[NODE:START] Entering start_node with state:", state)
    prompt = (
        f"You are a career counselor in India. The student is interested in these careers: {', '.join(state['careers'])}. "
        "Act as a personalized counselor and ask relevant questions to guide them."
    )
    response = llm.invoke([{"role": "system", "content": prompt}])
    # return {"messages": [{"role": "assistant", "content": response.content}]}
    result = {"messages": [AIMessage(content=response.content)]}
    print("[NODE:START] Exiting start_node with result:", result)
    return result

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
    # import json
    # parsed = json.loads(res.content)
    # return {"last_decision": {"answer": parsed.get('answer', ''), "question": parsed.get('question', '')}}
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

# def generate_new_node(state: CareerState) -> Dict[str, Any]:
#     answer = state['last_decision']['answer']
#     history = state['messages']
#     if answer.lower() in ['i don\'t know', 'not sure', 'maybe', '']:
#         follow_up_prompt = "The user is unsure. Discuss the topic from a different perspective to elicit feedback."
#     else:
#         sentiment = 'positive' if any(w in answer.lower() for w in ['like', 'love', 'interested']) else 'negative'
#         state['skewed_state'].setdefault(sentiment, []).append(answer)
#         follow_up_prompt = "Acknowledge the user's feedback and ask the next relevant question."
#     response = llm.invoke(history + [{"role": "system", "content": follow_up_prompt}])
#     return {"generate_new_response": response.content}    

# def answer_question_node(state: CareerState) -> Dict[str, Any]:
#     question = state['last_decision']['question']
#     history = state['messages']
#     # response = llm.invoke([
#     #     {"role": "system", "content": f"You are a helpful counselor. Context: {context}"},
#     #     {"role": "user", "content": question}
#     # ])
#     # return {"messages": [{"role": "assistant", "content": resp.content}]}

#     response = llm.invoke(history + [{"role": "system", "content": "You are a helpful counselor. Answer the user's question."}, {"role": "user", "content": question}])
#     return {"answer_question_response": response.content}

def process_input_node(state: CareerState) -> Dict[str, Any]:
    """Handles generating a new message and answering a question in parallel if both are present."""
    print("[NODE:PROCESS_INPUT] Entering process_input_node with state:", state)
    answer = state['last_decision'].get('answer', '')
    question = state['last_decision'].get('question', '')
    history = state['messages']

    # Task to generate a new message based on the answer
    def generate_new_task():
        if answer.lower() in ['i don\'t know', 'not sure', 'maybe', '']:
            follow_up_prompt = "The user is unsure. Discuss the topic from a different perspective to elicit feedback."
            sentiment = None
        else:
            sentiment = 'positive' if any(w in answer.lower() for w in ['like', 'love', 'interested']) else 'negative'
            follow_up_prompt = "Acknowledge the user's feedback and ask the next relevant question."
        response = llm.invoke(history + [{"role": "system", "content": follow_up_prompt}])
        return response.content, sentiment, answer

    # Task to answer the user's question
    def answer_question_task():
        response = llm.invoke(history + [
            {"role": "system", "content": "You are a helpful counselor. Answer the user's question."},
            {"role": "user", "content": question}
        ])
        return response.content

    updates = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        generate_future = None
        answer_future = None
        # Submit tasks if applicable
        if answer:
            generate_future = executor.submit(generate_new_task)
        if question:
            answer_future = executor.submit(answer_question_task)
        # Collect results
        if generate_future:
            generate_response, sentiment, answer_text = generate_future.result()
            updates["generate_new_response"] = generate_response
            if sentiment:
                state['skewed_state'].setdefault(sentiment, []).append(answer_text)
        if answer_future:
            updates["answer_question_response"] = answer_future.result()

    print("[NODE:PROCESS_INPUT] Exiting process_input_node with updates:", updates)
    return updates

def evaluate_node(state: CareerState) -> Dict[str, Any]:
    print("[NODE:EVALUATE] Entering evaluate_node with state:", state)
    # Prune careers with negative feedback
    negatives = state['skewed_state'].get('negative', [])
    # filtered = [c for c in state['careers'] if c not in negatives]
    # return {
    #     "careers": filtered,
    #     "conversation_number": state['conversation_number'] + 1
    # }
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
    print("[NODE:NEW_MSG] Combined message:", combined_msg)
    return {
        "messages": [AIMessage(content=combined_msg)],
        "generate_new_response": None,
        "answer_question_response": None
    }

def conclude_node(state: CareerState) -> Dict[str, Any]:
    print("[NODE:CONCLUDE] Entering conclude_node with state:", state)
    prompt = f"Based on our discussion, these careers suit you: {', '.join(state['careers'])}."
    result = {"messages": [AIMessage(content=prompt)]}
    print("[NODE:CONCLUDE] Exiting conclude_node with result:", result)
    return result

# Register nodes
graph.add_node("start", start_node)
graph.add_node("human_loop", human_loop_node)
graph.add_node("analyser", analyser_node)
# graph.add_node("generate_new", generate_new_node)
# graph.add_node("answer_question", answer_question_node)
graph.add_node("process_input", process_input_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("new_msg", new_msg_node)
graph.add_node("conclude", conclude_node)

# Define edges and conditionals
graph.add_edge(START, "start")
graph.add_edge("start", "human_loop")
graph.add_edge("human_loop", "analyser")
graph.add_edge("analyser", "process_input")
graph.add_edge("process_input", "evaluate")
graph.add_conditional_edges(
    "evaluate",
    lambda s: "new_msg" if (len(s['careers']) > 3 or s['conversation_number'] < 5) else "conclude",
    {"new_msg": "new_msg", "conclude": "conclude"}
)

# def analyser_condition(state: CareerState) -> str:
#     answer = state['last_decision']['answer']
#     question = state['last_decision']['question']
#     if answer and question:
#         return "both"
#     elif answer:
#         return "generate_new"
#     elif question:
#         return "answer_question"
#     return "generate_new"  # Default to generate_new if no clear decision

# graph.add_conditional_edges(
#     # "analyser",
#     # lambda s: [n for n in ["generate_new" if s['last_decision']['answer'] else None,
#     #                       "answer_question" if s['last_decision']['question'] else None] if n],
#     # path_map={"generate_new": "generate_new", "answer_question": "answer_question"}

#     "analyser",
#     analyser_condition,
#     {
#         "generate_new": "generate_new",
#         "answer_question": "answer_question",
#         "both": ["generate_new", "answer_question"]
#     }
# )
# graph.add_edge("generate_new", "evaluate")
# graph.add_edge("answer_question", "evaluate")
# graph.add_conditional_edges(
#     "evaluate",
#     lambda s: "new_msg" if (len(s['careers']) > 3 or s['conversation_number'] < 5) else END,
#     path_map={"new_msg": "new_msg", "conclude": "conclude"}
# )
graph.add_edge("new_msg", "human_loop")
graph.set_finish_point("conclude")

# Compile graph to an app callable
checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer, interrupt_after=["start", "new_msg"])
print("[GRAPH] Graph compiled with interrupt_after:", ["start", "new_msg"])

# --- 4) FastAPI wrapper ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow specific methods
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

    # Invoke graph
    # new_state = compiled_graph.invoke(init_state)
    # threads[tid] = new_state
    # first_msg = new_state['messages'][-1]['content'] if new_state['messages'] else ""
    print("[START] Initial state:", init_state)
    print("[START] Invoking graph with thread_id:", tid)

    result = compiled_graph.invoke(init_state, config={"configurable": {"thread_id": tid}})
    print("[START] Graph result:", result)
    print("[START] Checkpointer state:", checkpointer.get({"configurable": {"thread_id": tid}}))
    first_msg = result['messages'][-1].content if result['messages'] else ""
    return StartResponse(thread_id=tid, message=first_msg, is_ongoing=True)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    config = {"configurable": {"thread_id": req.thread_id}}
    print("[CHAT] Received request with thread_id:", req.thread_id, "message:", req.message)
    print("[CHAT] Raw checkpointer state:", checkpointer.get(config))

    current_state = checkpointer.get(config)
    if not current_state:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Ensure current_state has all required keys
    # if not isinstance(current_state, dict):
    #     current_state = {}
    # if 'messages' not in current_state:
    #     current_state['messages'] = []
    # if 'careers' not in current_state:
    #     current_state['careers'] = INITIAL_CAREERS.copy()
    # if 'skewed_state' not in current_state:
    #     current_state['skewed_state'] = {"positive": [], "negative": []}
    # if 'conversation_number' not in current_state:
    #     current_state['conversation_number'] = 0
    # if 'last_decision' not in current_state:
    #     current_state['last_decision'] = {"answer": "", "question": ""}
    # if 'generate_new_response' not in current_state:
    #     current_state['generate_new_response'] = None
    # if 'answer_question_response' not in current_state:
    #     current_state['answer_question_response'] = None

    # Append user message
    # current_state['messages'] = current_state['messages'] + [HumanMessage(content=req.message)]
    # print("[CHAT] State after appending user message:", current_state)
    
    # # Debug: Log the state before invoking the graph
    # # print(f"Current state before invoke: {current_state}")
    # print("[CHAT] Resuming graph with config:", config)
    # result = compiled_graph.invoke(current_state, config=config)
    # print("[CHAT] Graph result:", result)
    # print("[CHAT] Checkpointer state after invoke:", checkpointer.get(config))

    # Provide the user's message as input to resume the graph
    input_data = {"messages": [HumanMessage(content=req.message)]}
    print("[CHAT] Resuming graph with input:", input_data)
    
    # Resume the graph with the input
    result = compiled_graph.invoke(input_data, config=config)
    print("[CHAT] Graph result:", result)
    print("[CHAT] Checkpointer state after invoke:", checkpointer.get(config))
    
    
    reply = result['messages'][-1].content if result['messages'] else ""
    ongoing = len(result['careers']) > 3 or result['conversation_number'] < 5
    print("[CHAT] Returning response:", {"message": reply, "is_ongoing": ongoing})
    return ChatResponse(message=reply, is_ongoing=ongoing)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

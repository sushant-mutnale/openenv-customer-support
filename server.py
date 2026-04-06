from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from env import CustomerSupportEnv
from tasks import load_tasks
from models import ActionType, ActionClassify, ActionAskUser, ActionUseTool, ActionResolve, ActionEscalate

app = FastAPI(title="OpenEnv Customer Support", description="Hugging Face Space strict container validation endpoint.")

# Load tasks into memory
try:
    TASKS = load_tasks("tasks_refined.json")
except Exception as e:
    TASKS = []
    print(f"Failed to load tasks: {e}")

# Global state to maintain simplicity in the stateless container evaluation limit
CURRENT_ENV = None

class StepRequest(BaseModel):
    action_type: str
    payload: Dict[str, Any]

@app.post("/reset")
async def reset():
    global CURRENT_ENV
    if not TASKS:
        raise HTTPException(status_code=500, detail="No valid tasks available.")
    
    # Validation ping hits /reset without args, start at task 0
    CURRENT_ENV = CustomerSupportEnv(TASKS[0])
    result = await CURRENT_ENV.reset()
    
    # Strict stable JSON dictionary matching pydantic components identically
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done
    }

@app.post("/step")
async def step(request: StepRequest):
    global CURRENT_ENV
    if CURRENT_ENV is None:
        raise HTTPException(status_code=400, detail="Environment not reset.")
    
    action_type = request.action_type
    payload = request.payload
    
    try:
        if action_type == "classify": action = ActionClassify(action_type=action_type, payload=payload)
        elif action_type == "ask_user": action = ActionAskUser(action_type=action_type, payload=payload)
        elif action_type == "use_tool": action = ActionUseTool(action_type=action_type, payload=payload)
        elif action_type == "resolve": action = ActionResolve(action_type=action_type, payload=payload)
        elif action_type == "escalate": action = ActionEscalate(action_type=action_type, payload=payload)
        else: raise ValueError("Unknown action_type")
    except Exception as e:
        # Invalid schema safety fallback
        action = ActionAskUser(action_type="ask_user", payload={"question": f"SYSTEM_PARSE_ERROR: {str(e)}"})
        
    result = await CURRENT_ENV.step(action)
    
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done
    }

@app.get("/state")
async def state():
    if CURRENT_ENV is None:
        raise HTTPException(status_code=400, detail="Environment not reset.")
        
    # Strict stable schema
    return CURRENT_ENV.state().model_dump()

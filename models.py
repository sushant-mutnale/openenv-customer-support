from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, Union

class GroundTruth(BaseModel):
    category: str
    priority: str
    resolution: str
    requires_escalation: bool

class TaskInput(BaseModel):
    ticket: str
    user_type: str
    channel: str

class Constraints(BaseModel):
    max_steps: int

class Task(BaseModel):
    task_id: str
    input: TaskInput
    ground_truth: GroundTruth
    ideal_steps: List[str]
    constraints: Constraints
    difficulty: Literal["easy", "medium", "hard"]

class ActionClassify(BaseModel):
    action_type: Literal["classify"]
    payload: Dict[str, str]

class ActionAskUser(BaseModel):
    action_type: Literal["ask_user"]
    payload: Dict[str, str]

class ActionUseTool(BaseModel):
    action_type: Literal["use_tool"]
    payload: Dict[str, str]

class ActionResolve(BaseModel):
    action_type: Literal["resolve"]
    payload: Dict[str, str]

class ActionEscalate(BaseModel):
    action_type: Literal["escalate"]
    payload: Dict[str, str] = {}

ActionType = Union[ActionClassify, ActionAskUser, ActionUseTool, ActionResolve, ActionEscalate]

class EnvironmentObservation(BaseModel):
    ticket: str
    conversation_history: List[str]
    last_tool_output: Optional[str]
    remaining_steps: int

class EnvironmentState(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    done: bool
    score: float
    history: List[str]

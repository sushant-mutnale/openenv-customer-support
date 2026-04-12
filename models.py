from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, Union

class GroundTruth(BaseModel):
    category: str
    priority: str
    resolution: str
    requires_escalation: bool

class TaskInput(BaseModel):
    ticket: str
    customer_profile: str
    policy_snippets: str

class Constraints(BaseModel):
    max_steps: int

class Task(BaseModel):
    task_id: str
    input: TaskInput
    ground_truth: GroundTruth
    ideal_steps: List[str]
    constraints: Constraints
    difficulty: Literal["easy", "medium", "hard"]

class ActionReply(BaseModel):
    action_type: Literal["reply"]
    payload: Dict[str, str]

class ActionAskUser(BaseModel):
    action_type: Literal["ask_user"]
    payload: Dict[str, str]

class ActionUseTool(BaseModel):
    action_type: Literal["use_tool"]
    payload: Dict[str, str]

class ActionCloseTicket(BaseModel):
    action_type: Literal["close_ticket"]
    payload: Dict[str, str] = {}

class ActionEscalate(BaseModel):
    action_type: Literal["escalate"]
    payload: Dict[str, str] = {}

ActionType = Union[ActionReply, ActionAskUser, ActionUseTool, ActionCloseTicket, ActionEscalate]

class EnvironmentObservation(BaseModel):
    ticket: str
    customer_profile: str
    policy_snippets: str
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

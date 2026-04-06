from typing import Any
from models import Task, EnvironmentObservation, EnvironmentState
from grader import TaskGrader

class EnvResult:
    def __init__(self, observation: EnvironmentObservation, reward: float, done: bool):
        self.observation = observation
        self.reward = float(reward)
        self.done = done

class CustomerSupportEnv:
    def __init__(self, task: Task):
        self.task = task
        self.max_steps = task.constraints.max_steps
        self.current_step = 0
        self.conversation_history = []
        self.last_tool_output = None
        self.grader = TaskGrader(task)
        self.done = False

    async def reset(self) -> EnvResult:
        self.current_step = 0
        self.conversation_history = []
        self.last_tool_output = None
        self.grader = TaskGrader(self.task)
        self.done = False
        return self._get_observation(reward=0.0)

    def _get_observation(self, reward: float) -> EnvResult:
        obs = EnvironmentObservation(
            ticket=self.task.input.ticket,
            conversation_history=self.conversation_history,
            last_tool_output=self.last_tool_output,
            remaining_steps=self.max_steps - self.current_step
        )
        return EnvResult(observation=obs, reward=reward, done=self.done)

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self.task.task_id,
            step_count=self.current_step,
            max_steps=self.max_steps,
            done=self.done,
            score=self.grader.final_score(),
            history=self.conversation_history
        )

    async def step(self, action: Any) -> EnvResult:
        if self.done:
            return self._get_observation(reward=0.0)

        self.current_step += 1
        
        # Action Simulator Logic
        action_type = action.action_type
        if action_type == "ask_user":
            question = action.payload.get('question', '').lower()
            self.conversation_history.append(f"AGENT: {action.payload.get('question')}")
            
            # Simple contextual match mock
            if "order" in question:
                self.conversation_history.append(f"USER: My order ID is ORD-9921.")
            elif "payment" in question or "transaction" in question:
                self.conversation_history.append(f"USER: The transaction ID is TRX-551.")
            else:
                self.conversation_history.append(f"USER: This issue is severely impacting my workflow, please fix immediately.")

        elif action_type == "use_tool":
            tool = action.payload.get('tool', 'unknown')
            input_val = action.payload.get('input', '')
            
            # Robust, realistic deterministic tool implementations
            if tool == "check_order_status":
                if "ORD-9921" in input_val:
                    self.last_tool_output = '{"order_id": "ORD-9921", "state": "dispatched", "tracking": "active"}'
                else:
                    self.last_tool_output = '{"error": "Invalid Order ID or missing context."}'
                    
            elif tool == "check_payment":
                self.last_tool_output = '{"status": "paid", "balance": 0.0, "last_transaction": "TRX-551"}'
                
            elif tool == "issue_refund":
                if "TRX-551" in input_val:
                    self.last_tool_output = '{"status": "success", "refunded": true}'
                else:
                    self.last_tool_output = '{"error": "Refund failed. Valid Transaction ID required."}'
                    
            elif tool == "verify_user_account":
                self.last_tool_output = '{"account_active": true, "plan": "Enterprise"}'
            else:
                self.last_tool_output = f'{{"error": "Tool {tool} not recognized."}}'
            
            self.conversation_history.append(f"SYSTEM: Invoked tool '{tool}'.")
            
        elif action_type in ["resolve", "escalate"]:
            if action_type == "resolve":
                self.conversation_history.append(f"AGENT RESOLVED: {action.payload.get('resolution')}")
            else:
                self.conversation_history.append(f"AGENT ESCALATED: Sent to Tier 2 support.")
            self.done = True

        if self.current_step >= self.max_steps:
            self.done = True

        reward = self.grader.evaluate_action(action)
        return self._get_observation(reward=reward)

    async def close(self):
        pass

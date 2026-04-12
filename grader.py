from models import Task, ActionType

class TaskGrader:
    def __init__(self, task: Task):
        self.task = task
        self.history = []
        self.score = 0.0
        
        # Trajectory tracking for sequence completeness
        self.has_replied = False
        self.has_asked_context = False
        self.has_used_valid_tool = False
        self.has_escalated = False

    def evaluate_action(self, action: ActionType) -> float:
        """
        Dense reward shaping targeting [0.0, 1.0] scale per step.
        """
        reward = 0.0
        diff = self.task.difficulty
        act_type = action.action_type
        
        # Penalize repeating exactly the same action type immediately
        if len(self.history) > 0 and self.history[-1]['action_type'] == act_type:
            # We allow repeating replies, but repeating tools or questions is bad
            if act_type in ["ask_user", "use_tool"]:
                reward -= 0.1
                
        if act_type == "reply":
            msg = action.payload.get("message", "").lower()
            if diff == "easy":
                if "https://portal.company.com/reset" in msg:
                    reward += 0.5
                    self.has_replied = True
                else:
                    reward -= 0.2
            elif diff == "medium":
                if "dispatched" in msg:
                    reward += 0.3
                    self.has_replied = True
            elif diff == "hard":
                if "refund" in msg and ("cannot" in msg or "deny" in msg or "non-refundable" in msg or "strict" in msg):
                    reward += 0.3
                    self.has_replied = True

        elif act_type == "ask_user":
            if diff == "medium":
                q = action.payload.get("question", "").lower()
                if "order" in q or "id" in q:
                    reward += 0.2
                    self.has_asked_context = True
                else:
                    reward -= 0.1
            else:
                # Asking user in easy/hard is a waste of steps based on the context given
                reward -= 0.1

        elif act_type == "use_tool":
            tool = action.payload.get("tool")
            input_val = action.payload.get("input", "")
            
            if diff == "medium":
                if tool == "check_order_status" and "ORD-9921" in input_val:
                    if not self.has_asked_context:
                        reward -= 0.3 # Cheating hallucination
                    else:
                        reward += 0.3
                        self.has_used_valid_tool = True
                else:
                    reward -= 0.2

            elif diff == "hard":
                if tool == "check_payment" and "TRX-551" in input_val:
                    reward += 0.3
                    self.has_used_valid_tool = True
                elif tool == "issue_refund":
                    # Instant fail scenario basically
                    reward -= 0.5
                else:
                    reward -= 0.1

        elif act_type == "close_ticket":
            if diff == "easy":
                if self.has_replied:
                    reward += 0.5
                else:
                    reward -= 0.4
            elif diff == "medium":
                if self.has_replied and self.has_used_valid_tool:
                    reward += 0.2
                else:
                    reward -= 0.3
            elif diff == "hard":
                # Hard task strictly requires escalation, not resolution
                reward -= 0.5
                
        elif act_type == "escalate":
            if diff == "hard":
                if self.has_replied and self.has_used_valid_tool:
                    reward += 0.4
                    self.has_escalated = True
                else:
                    # Escalated prematurely without checking policy/tool
                    reward += 0.1
            else:
                # Escalating an easy/medium task is a failure
                reward -= 0.5

        # Record action in history
        self.history.append({"action_type": act_type, **action.payload})
        
        # Mathematically ensure the accumulative score stays bounded. 
        # Rewards can be negative intermediately (punishments), but we add them to the total.
        self.score += reward
        
        return reward
        
    def final_score(self) -> float:
        # Guarantee mathematical compliance by bounding to strictly (0.0, 1.0) exactly as spec requires
        return min(max(self.score, 0.01), 0.99)

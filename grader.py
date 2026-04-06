from models import Task, ActionType

class TaskGrader:
    def __init__(self, task: Task):
        self.task = task
        self.history = []
        self.score = 0.0
        
        # Trajectory tracking for hard-mode sequences
        self.has_asked_context = False
        self.has_used_valid_tool = False

    def evaluate_action(self, action: ActionType) -> float:
        reward = 0.0
        
        if action.action_type == "classify":
            cat = action.payload.get("category", "").lower()
            if cat == self.task.ground_truth.category.lower():
                reward += 0.2
            else:
                reward -= 0.1
                
        elif action.action_type == "ask_user":
            if action.payload.get("question") in [h.get("question") for h in self.history if h.get("action_type")=="ask_user"]:
                reward -= 0.1 # Penalty for repetition
            else:
                reward += 0.1
                self.has_asked_context = True # Agent appropriately asked questions

        elif action.action_type == "use_tool":
            tool = action.payload.get("tool")
            input_val = action.payload.get("input", "")
            
            if tool in [h.get("tool") for h in self.history if h.get("action_type")=="use_tool"]:
                reward -= 0.1 # Repetition penalty
            else:
                # HARD MODE Sequence evaluation check!
                if "ORD-9921" in input_val or "TRX-551" in input_val:
                    if not self.has_asked_context:
                        # Cheating / hallucination penalty (agent just guessed the ID without asking user!)
                        reward -= 0.4
                    else:
                        # Valid contextual sequence!
                        reward += 0.3
                        self.has_used_valid_tool = True
                else:
                    # Valid attempt but maybe wrong context input
                    reward += 0.1
                    
        elif action.action_type == "resolve":
            if self.task.ground_truth.requires_escalation:
                reward -= 0.3 # Should have escalated!
            else:
                if self.task.difficulty == "hard":
                    if self.has_used_valid_tool and self.has_asked_context:
                        reward += 0.5 # Perfect hard-mode sequence completeness
                    else:
                        reward += 0.1 # Premature resolution
                else:
                    reward += 0.4 # Default easy/medium resolution success
                    
        elif action.action_type == "escalate":
            if self.task.ground_truth.requires_escalation:
                reward += 0.4
            else:
                reward -= 0.2
                
        self.history.append({"action_type": action.action_type, **action.payload})
        self.score += reward
        return reward
        
    def final_score(self) -> float:
        # Guarantee mathematical compliance by bounding to strictly [0.0, 1.0] exactly as spec requires
        return min(max(self.score, 0.0), 1.0)

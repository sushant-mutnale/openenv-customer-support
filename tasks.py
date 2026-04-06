import json
from typing import List
from models import Task

def load_tasks(filepath: str) -> List[Task]:
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    parsed_tasks = []
    
    for idx, t in enumerate(data):
        if not t.get("refined"):
            continue
            
        task_id = t.get("task_id")
        if not task_id:
            raise ValueError(f"Task at index {idx} completely missing task_id")
            
        difficulty = t.get("difficulty")
        if difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Task {task_id} has invalid difficulty '{difficulty}'. Must be easy, medium, or hard.")
            
        ideal_steps = t.get("ideal_steps")
        if not ideal_steps or not isinstance(ideal_steps, list):
            raise ValueError(f"Task {task_id} must have hidden ideal_steps list.")
            
        gt = t.get("ground_truth", {})
        if "requires_escalation" not in gt:
            raise ValueError(f"Task {task_id} missing requires_escalation attribute.")

        input_data = t.get("input", {})
        
        task_data = {
            "task_id": task_id,
            "input": {
                "ticket": input_data.get("ticket", ""),
                "user_type": input_data.get("user_type", "Standard"), 
                "channel": input_data.get("channel", "Email")
            },
            "ground_truth": {
                "category": input_data.get("category", "General Request"),
                "priority": input_data.get("priority", "medium"),
                "resolution": "Resolution evaluated via trajectory steps and logic.",
                "requires_escalation": gt.get("requires_escalation", False)
            },
            "ideal_steps": ideal_steps,
            "constraints": {
                "max_steps": t.get("constraints", {}).get("max_steps", 8)
            },
            "difficulty": difficulty
        }
        
        parsed_tasks.append(Task(**task_data))
            
    if len(parsed_tasks) < 3:
        raise ValueError("Dataset parsing failed: strictly requires at least 3 valid Refined Tasks.")
        
    return parsed_tasks

import json
from typing import List
from models import Task

def load_tasks(filepath: str) -> List[Task]:
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    parsed_tasks = []
    
    for t in data:
        difficulty = t.get("difficulty")
        task_id = t.get("task_id")
        ideal_steps = t.get("ideal_steps")
        gt = t.get("ground_truth", {})
        input_data = t.get("input", {})
        
        task_data = {
            "task_id": task_id,
            "input": {
                "ticket": input_data.get("ticket", ""),
                "customer_profile": input_data.get("customer_profile", ""), 
                "policy_snippets": input_data.get("policy_snippets", "")
            },
            "ground_truth": {
                "category": gt.get("category", ""),
                "priority": gt.get("priority", "medium"),
                "resolution": gt.get("resolution", ""),
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

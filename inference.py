import os
import json
import asyncio
import textwrap
from typing import List, Optional
from pydantic import ValidationError
from openai import OpenAI

from tasks import load_tasks
from env import CustomerSupportEnv
from models import ActionType, ActionClassify, ActionAskUser, ActionUseTool, ActionResolve, ActionEscalate

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "customer_support"
MAX_TOKENS = 500

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer support agent. You will receive an observation with ticket details, history, and remaining steps.
    You must output exactly ONE JSON action per turn.
    
    Valid action formats:
    {"action_type": "classify", "payload": {"category": "<category>"}}
    {"action_type": "ask_user", "payload": {"question": "<question string>"}}
    {"action_type": "use_tool", "payload": {"tool": "<tool_name>", "input": "<inputs>"}}
    {"action_type": "resolve", "payload": {"resolution": "<resolution description>"}}
    {"action_type": "escalate", "payload": {}}

    Respond with ONLY the raw JSON string.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def parse_action(json_str: str) -> ActionType:
    try:
        data = json.loads(json_str.strip('```json').strip('```').strip())
        a_type = data.get("action_type")
        if a_type == "classify": return ActionClassify(**data)
        if a_type == "ask_user": return ActionAskUser(**data)
        if a_type == "use_tool": return ActionUseTool(**data)
        if a_type == "resolve": return ActionResolve(**data)
        if a_type == "escalate": return ActionEscalate(**data)
        raise ValueError(f"Unknown action type: {a_type}")
    except Exception as e:
        # Default safety fallback
        return ActionAskUser(action_type="ask_user", payload={"question": f"Parse error: {e}"})

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = load_tasks("tasks_refined.json")
    
    if not tasks:
        print("No tasks found.")
        return

    # For the simulation baseline, we will process the first 3 tasks
    for task in tasks[:3]:
        env = CustomerSupportEnv(task)
        history_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        rewards = []
        steps_taken = 0
        success = False
        
        log_start(task=task.task_id, env=BENCHMARK, model=MODEL_NAME)
        
        try:
            result = await env.reset()
            
            for step in range(1, task.constraints.max_steps + 1):
                if result.done:
                    break
                    
                obs_json = result.observation.model_dump_json()
                history_msgs.append({"role": "user", "content": obs_json})
                
                error_msg = None
                action_text = ""
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=history_msgs,
                        max_tokens=MAX_TOKENS
                    )
                    action_text = completion.choices[0].message.content or ""
                    
                    history_msgs.append({"role": "assistant", "content": action_text})
                    
                    action = parse_action(action_text)
                    result = await env.step(action)
                    
                except Exception as e:
                    error_msg = str(e)
                    result = await env.step(ActionAskUser(action_type="ask_user", payload={"question": "Fallback"}))
                    
                reward = result.reward
                done = result.done
                
                rewards.append(reward)
                steps_taken = step
                
                # Truncate action for clean log
                log_action = action_text.replace("\n", " ")[:100]
                log_step(step=step, action=f"'{log_action}'", reward=reward, done=done, error=error_msg)
                
                if done:
                    break
                    
            final_score = env.grader.final_score()
            success = final_score > 0.5
            
        finally:
            await env.close()
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())

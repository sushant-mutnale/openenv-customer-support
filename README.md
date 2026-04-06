---
title: OpenEnv Customer Support
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# OpenEnv Customer Support Simulation

A production-grade RL environment for an OpenEnv submission simulating a real-world helpdesk/customer support workflow.

## Architecture

This structured RL environment evaluates agents dynamically over custom customer support tasks:
- **`env.py`**: A strict async tracking environment `CustomerSupportEnv`. Tracks steps, conversation histories, and resolves tools.
- **`models.py`**: Pydantic definitions enforcing the strict nested Action/Observation schemas.
- **`grader.py`**: Deterministic evaluators scoring the progression trajectory against ground truth resolutions seamlessly normalizing the score to `[0,1]`.
- **`inference.py`**: Benchmark driver enforcing strict regex `[START]`, `[STEP]`, `[END]` evaluation logs out.

## Action & Observation Schema

### Observation
```json
{
  "ticket": "User ticket content...",
  "conversation_history": ["[System]: Tool ran", "[User]: Ok"],
  "last_tool_output": "...",
  "remaining_steps": 7
}
```

### Action (5 variants)
Return a single JSON matching one of these:
1. `{"action_type": "classify", "payload": {"category": "billing"}}`
2. `{"action_type": "ask_user", "payload": {"question": "Are you online?"}}`
3. `{"action_type": "use_tool", "payload": {"tool": "check_payment", "input": "order_123"}}`
4. `{"action_type": "resolve", "payload": {"resolution": "Refund issued."}}`
5. `{"action_type": "escalate", "payload": {}}`

## Setup & Running

1. Build Image: `docker build -t customer_support_env .`
2. Run Validate: `openenv validate`
3. Run Local Inference Baseline:
   ```bash
   export HF_TOKEN="your_hugging_face_token"
   export API_BASE_URL="https://router.huggingface.co/v1"
   export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
   python inference.py
   ```

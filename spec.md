# 🧠 Time-Travel Evals (TTE) — Detailed End-to-End Implementation Spec

## 1. 🎯 Objective
Build a system that evaluates AI agents by running them across **multiple simulated alternate realities (branches)**. This approach measures **robustness** (how an agent reacts to varying perturbations) rather than single-path accuracy. The MVP focuses on a "coding/DevOps agent" domain.

---

## 2. 🏛️ System Architecture

The project will follow a decoupled backend/frontend structure or a single Streamlit app. For rapid prototyping within 48-72 hours, we will use an API-driven Python backend with a lightweight web UI.

### Flow Diagram
```text
User Base Task → [Scenario Generator] 
                      ↳ Branch 1 (Baseline)
                      ↳ Branch 2 (Event: Latency)
                      ↳ Branch 3 (Event: Server Crash)
                             ↓
                 [World Simulator loop per Branch] ← Events applied here
                             ↓
                 [Agent Replay Engine (Codex/GPT4)]
                             ↓
                     [Trace Capture]
                             ↓
                      [Evaluator] → Calculates Success & Robustness Score
                             ↓
                     [UI Visualization]
```

### Folder Structure
```text
tte/
├── core/
│   ├── models.py                # Pydantic data models
│   ├── scenario_generator.py    # LLM branch generation
│   ├── simulator.py             # Event engine & state tracking
│   ├── agent_runner.py          # LLM agent wrapper (OpenAI API call)
│   ├── evaluator.py             # Robustness & rules metrics
│   └── logger.py                # Trace logging
├── data/
│   └── test_cases.json          # Mock domains and base tasks
├── ui/
│   └── app.py                   # Streamlit frontend or React App
├── main.py                      # Orchestrator & CLI entry
├── requirements.txt
└── spec.md                      # This file
```

---

## 3. 🧩 Core Data Models (`core/models.py`)

Using `pydantic` to strongly type the data flowing between modules.

```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Event(BaseModel):
    step: int
    name: str
    description: str
    effect_deltas: Dict[str, Any]

class ScenarioBranch(BaseModel):
    id: str
    description: str
    events: List[Event]
    initial_state_overrides: Dict[str, Any]

class AgentAction(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    reasoning: Optional[str]

class BranchTrace(BaseModel):
    branch_id: str
    steps: List[Dict[str, Any]]  # state, event, agent_action
    success: bool
    score: float  # 0.0 to 1.0

class EvalResult(BaseModel):
    base_task: str
    robustness_score: float
    success_rate: float
    traces: List[BranchTrace]
```

---

## 4. ⚙️ Component Implementation Details

### 4.1 Scenario Generator (`scenario_generator.py`)
**Goal:** Take a base query (e.g., "Deploy v2 of the frontend") and use GPT-4o to generate perturbations (branches).
- **Implementation Strategy:**
  1. Define a Pydantic schema for the output (List of Branches).
  2. Use OpenAI structured outputs (`response_format`) to strictly return JSON.
  3. Include few-shot examples in the system prompt.
  - *Branch Types:* 1 Baseline, 2 "Happy Path Variations", 2 "Crisis/Failure Events".

**Prompt Logic:**
```text
Task: {base_task}
Environment schema: {env_schema_keys}

Generate 3 alternate scenarios. One should be a delay event, one should be a resource failure.
Output JSON format: {"branches": [{"id": "...", "description": "...", "events": [{"step": 1, "name": "cpu_spike"...}]}]}
```

### 4.2 World Simulator (`simulator.py`)
**Goal:** A discrete tick mechanism holding the "mock environment".
- **State tracking:** `self.state = {...}` (e.g., `server_up: True, cpu_utilization: 40`).
- **Tick function:**
  - `def tick(self, current_step: int, branch: ScenarioBranch) -> Observation:`
  - Checks if `current_step` matches any pre-seeded `Event` in the `ScenarioBranch`.
  - Applies event effects to `self.state`.
  - Returns a textual observation for the agent (e.g., "SYSTEM ALERT: CPU Utilization is at 99%").

### 4.3 Agent Replay Engine (`agent_runner.py`)
**Goal:** Sandboxed execution of the agent against the simulator.
- **Agent Setup:** We use OpenAI API (GPT-4o or `gpt-4-turbo-preview`) combined with tools/functions.
- **Tools provided to agent:**
  - `execute_shell_command(cmd)` (Mocked! Just logs the command and returns a mock success).
  - `read_logs()`
  - `rollback()`
  - `finish_task()`
- **Execution Loop:**
  ```python
  def run_agent_on_branch(branch, max_steps=5):
      sim = Simulator(initial_state=branch.initial_state)
      agent = Agent(tools=[...])
      trace = []

      for step in range(max_steps):
          observation = sim.tick(step, branch)
          action = agent.step(observation)
          sim.apply_action(action)
          trace.append({"step": step, "obs": observation, "action": action})
          
          if action.name == "finish_task":
              break
              
      return trace, sim.get_final_state()
  ```

### 4.4 Evaluator (`evaluator.py`)
**Goal:** Convert agent traces into metrics.
- **Success Criteria:** A rule-based check on the Simulator's final state (e.g., `state['deployed'] == True and state['server_up'] == True`).
- **Robustness Score Formula:**
  - Baseline score $s_0$: Score on the non-perturbed branch (1.0 or 0.0)
  - Branch scores $s_i$: Score on branch $i$.
  - `Robustness Score = (Sum of s_i) / (N * (s_0 + epsilon))`
- **Outputs:** An aggregated `EvalResult` object.

### 4.5 Visualization Layer (`ui/app.py`)
**Goal:** Make it visually immediately clear what happened (Red/Green branches).
- **Tool:** `Streamlit`.
- **UI Structure:**
  1. Input: Textbox for "Base Task".
  2. Action: "Run TTE".
  3. Output: 
     - **Hero Metrics:** Big text for `Robustness Score` and `Success Rate`.
     - **Timeline View:** (Using `streamlit-echarts` or plotting HTML) showing the base node and diverging lines to N end-nodes. Nodes colored green or red.
     - **Trace Accordions:** Click on a branch to expand and see:
       - `Step 1: Event injected: Network Timeout`
       - `Agent: Called tool 'retry_connection'`
       - `Step 2: Success observed.`

---

## 5. 🗺️ End-to-End Execution Plan (Hackathon Timeline)

### Step 1: Base Scaffolding & API Wiring (Hours 1 - 6)
- Setup Python venv, install `openai`, `pydantic`, `streamlit`.
- Secure OpenAI API Keys.
- Write the Data Models (`core/models.py`).
- Implement the basic OpenAI API wrapper (`execute_prompt` and structured output logic).

### Step 2: The Mock World & Agent (Hours 6 - 16)
- Implement `Simulator` logic. Make it generic enough to handle a dictionary of state and apply `effect_deltas`.
- Implement `AgentReplayEngine`. Hook up OpenAI tool-calling. Ensure the agent can interact with the Simulator object (mock state changes).
- **Checkpoint Check:** Ensure you can run a simple "Deploy Code" loop where the agent calls `mock_deploy()` and the loop exits.

### Step 3: Branch Generation (Hours 16 - 24)
- Implement `scenario_generator.py`.
- Feed the base prompt and map the OpenAI JSON output to `ScenarioBranch` pydantic models.
- Hook this to the Orchestrator loop so that 1 task -> 3 concurrent Simulator loops.

### Step 4: Trace Matrix & Core Math (Hours 24 - 32)
- Implement `evaluator.py`.
- Verify the Robustness formula yields the correct decimals.
- Add "early deviation tracking" (e.g., which step did the agent diverge from the baseline run).

### Step 5: Web UI & Demo Polish (Hours 32 - 40)
- Write the Streamlit app.
- Provide a canned/cached demo to avoid API rate limits during presenting.
- Include a visual "timeline tree" indicating forks.
- Ensure the Red/Green failure states are highly contrasted for judges.

### Step 6: Testing & Edge Cases (Hours 40 - 48)
- Test with out-of-bounds agent actions (agent hallucinating tools). Keep the simulator robust.
- Polish demo script. Generate 3 good pre-made example tasks in `test_cases.json`.

---

## 6. 🛡️ Safety & API Limits Handling
- **Async Execution:** To ensure the system evaluates branches quickly, we will use Python's `asyncio` to run the 3-5 branches concurrently against the OpenAI API.
- **Fail-safes:** Hard-limit the simulation loops to 10 steps to prevent API billing runaway loops.
- **Mock Execution:** Under NO circumstances does the evaluator run actual system commands. All agent calls string-match to state-mutation rules.

---

## 7. 🎯 Success Checklist
- [ ] LLM generates coherent and distinct sub-scenarios based on a task.
- [ ] The simulation successfully mutates its state asynchronously across branches.
- [ ] Agent recognizes simulation events natively via standard prompt observations.
- [ ] Metrics mathematically verify how well the agent adapts to injected anomalies.
- [ ] The UI renders a branching visualization clearly distinguishing agent victory from failure.

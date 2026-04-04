# Time-Travel Evals (TTE) 🧠

Time-Travel Evals (TTE) is a prototype Python framework that evaluates AI agents not simply on what happened, but on **what could have happened**. It tests agent resilience by forking a single task into a multiverse of alternate branching timelines (e.g. latency spikes, dependency failures, resource constraints), evaluating the agent across all realities simultaneously.

## Features

- **Discrete-Event Simulation**: A rule-based internal engine where various events and scenarios can be introduced safely and deterministically without affecting your actual production systems.
- **Scenario Generation via LLMs**: Uses `gpt-4o` structured output models to automatically generate diverse, plausible failure mode scenarios based solely on a high-level task and environment metadata.
- **Robustness Score (RS)**: A new class of metric that quantifies how much an agent's performance drops when perturbed from the "happy path."
- **Trace Visualization**: A fully featured Streamlit UI displaying visual trees and step-by-step traces of what the agent decided to do in each scenario.

## Setup Requirements

1. **Python 3.10+** (Python 3.12 recommended)
2. Obtain a **NVIDIA API Key** (or OpenAI key).

Ensure `requirements.txt` dependencies are installed:

```bash
pip install -r requirements.txt
```

## Quickstart Configuration

1. Copy `.env.example` to `.env` and fill in your NVIDIA API Key:

```bash
cp .env.example .env
```

2. Set your key:

```env
NVIDIA_API_KEY=nvapi-...
```

## Usage

### Web UI (Streamlit)

Launch the interactive dashboard to generate scenarios, run evaluations, and analyze multiversal traces.

```bash
streamlit run ui/app.py
```

### CLI Orchestrator

You can also run evaluations entirely from the terminal:

```bash
# Run a single task branching into 4 timelines with a max limit of 8 steps each
python main.py --task "Deploy v2 of the frontend application to production" --branches 4 --steps 8

# Run offline using mock rules (No LLM calls required)
python main.py --rules-only --output data/cached_demo.json
```

## Running the Tests

To ensure the engine metrics and simulation mechanics behave accurately against expected assertions:

```bash
pytest tests/
```

## Architecture

- `core/`: Config, environment models, step simulations, evaluator rules, and agent framework wrappers
- `ui/`: ECharts visualizer, Streamlit dashboard and metric presentation pages
- `data/`: Mock environment scenarios and cached records for quick-starts
- `tests/`: Integration definitions

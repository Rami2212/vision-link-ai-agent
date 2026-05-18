# Vision-Link AI Agent 🏥⚡

> **Multi-Agent Autonomous Healthcare Assistant**
> Built with CrewAI + LangGraph for the [AI Agent Olympics — Lablab.ai](https://lablab.ai)
> **Deadline: May 19, 2026**

-----

## Table of Contents

1. [Architecture](#architecture)
1. [Data Flow](#data-flow)
1. [Core Modules](#core-modules)
1. [Self-Healing Loop](#self-healing-loop)
1. [Self-Evolving Backend](#self-evolving-backend)
1. [Models](#models)
1. [Environment Setup](#environment-setup)
1. [Running the Smoke Test](#running-the-smoke-test)
1. [Troubleshooting](#troubleshooting)
1. [CI / CD](#ci--cd)
1. [Dependencies](#dependencies)
1. [File Structure](#file-structure)
1. [Roadmap](#roadmap)
1. [Team](#team)
1. [License](#license)

-----

## Architecture

7-node LangGraph StateGraph with CrewAI agents at each processing stage:

```
Patient Input
     │
     ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Clinical Data  │────▶│  Localization    │────▶│   Validation     │
│  Agent (CrewAI) │     │  Agent (CrewAI)  │     │   Agent (CrewAI) │
│  MedLlama-3     │     │  Llama-3-8B      │     │   Llama-3-8B     │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                              ┌────────────▼────────────┐
                                              │   Conditional Routing    │
                                              │   (LangGraph edges)      │
                                              └────┬──────────────┬──────┘
                                                   │ FAIL         │ PASS
                                                   ▼              ▼
                                            ┌──────────┐   ┌───────────────┐
                                            │  Self-   │   │  Self-Evolve  │
                                            │  Healing │   │  Engine       │
                                            │  Loop    │   │  (Genetic AI) │
                                            └────┬─────┘   └──────┬────────┘
                                                 │                │
                                                 └────────────────▼
                                                          Final Response
```

-----

## Data Flow

```
1. PatientContext (Pydantic v2) → Clinical Agent
2. ClinicalOutput → Localization Agent
3. LocalizationOutput → Validation Agent
4. ValidationOutput → Conditional Router
   ├── FAIL → Self-Healing Node → re-run failed agent (max 3 retries)
   └── PASS → Self-Evolving Engine → fitness benchmark → atomic hot-swap
5. PipelineState.final_response → JSON output
```

Every stage reads from and writes back to a single shared `PipelineState` object — the single source of truth across all nodes.

-----

## Core Modules

|File                       |Owner     |Description                                                                        |
|---------------------------|----------|-----------------------------------------------------------------------------------|
|`workspace/state_schema.py`|Zentomo   |Pydantic v2 shared state — single source of truth across all agents                |
|`workspace/orchestrator.py`|Zentomo   |LangGraph StateGraph, conditional routing, self-healing loop, self-evolving backend|
|`workspace/model_loader.py`|Zentomo   |HuggingFace model loader — Llama-3-8B-Instruct & MedLlama, node-aware routing      |
|`workspace/crew_agents.py` |Subhalaxmi|CrewAI agent definitions, task schemas, Pydantic I/O contracts                     |

-----

## Self-Healing Loop

When the Validation Agent flags **errors or missing fields**, the orchestrator:

1. Identifies which agent produced the failure (clinical or localization)
1. Re-routes to **only that agent** — not the full pipeline
1. Retries up to `MAX_RETRIES` (default: 3)
1. Falls back to LLM-assisted patching for soft errors (e.g. low confidence score)
1. Marks run `failed` only after all retries are exhausted

-----

## Self-Evolving Backend

Every successful run triggers the evolution engine:

1. **3 variants** of the pipeline are generated concurrently via Llama-3-8B
1. Each variant compiles and executes in an **isolated sandbox**
1. Fitness is scored using a single consistent formula (see below)
1. **Winner atomically replaces** the live module via Python hot-swap
1. Zero human intervention. Zero process restarts. Zero downtime.

### Fitness Formula

```
fitness_score = latency_ms + error_penalty - (throughput_rps × 10)
```

Where:

- `latency_ms` — time to completion in milliseconds (lower is better)
- `error_penalty` — 1000.0 if the variant threw an exception, else 0.0
- `throughput_rps` — requests per second (1000 / latency_ms)

**Lower fitness score = better variant.** The winner is the variant with the lowest score that also produced a valid output.

### Configuration

```bash
# Disable evolution engine for faster smoke testing
EVOLUTION_ENABLED=false python orchestrator.py

# Enable full self-evolving run (default)
EVOLUTION_ENABLED=true python orchestrator.py
```

-----

## Models

Models are loaded via the **HuggingFace Inference API** — remote endpoint calls authenticated with `HF_TOKEN`. Models do not run locally; no GPU is required on the developer machine.

|Node                 |Model                                   |Rationale                                                                        |
|---------------------|----------------------------------------|---------------------------------------------------------------------------------|
|Clinical Agent       |`ProbeMedicalYonseiMAILab/medllama3-v20`|Specialized medical knowledge; highest diagnostic accuracy via domain fine-tuning|
|Localization Agent   |`meta-llama/Meta-Llama-3-8B-Instruct`   |Strong multilingual support; efficient context understanding                     |
|Validation Agent     |`meta-llama/Meta-Llama-3-8B-Instruct`   |Reliable structured output generation; excellent for schema validation           |
|Self-Healing / Evolve|`meta-llama/Meta-Llama-3-8B-Instruct`   |Balanced reasoning capability and inference speed                                |
|Fallback             |`mistralai/Mistral-7B-Instruct-v0.3`    |Lightweight alternative for lighter state transitions                            |

Model loading is handled by `workspace/model_loader.py` via `get_llm_for_node(node_name)`, which routes each node to its appropriate model and caches instances with `@lru_cache`.

-----

## Environment Setup

### Prerequisites

- Python 3.11+
- HuggingFace account with API token (free at huggingface.co)
- Git

### Installation

```bash
# 1. Clone
git clone https://github.com/BuhariSalisuAI/vision-link-ai-agent.git
cd vision-link-ai-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Open .env and set your HF_TOKEN
```

### Environment Variables

|Variable           |Required|Default|Purpose                                                            |
|-------------------|--------|-------|-------------------------------------------------------------------|
|`HF_TOKEN`         |✅       |—      |HuggingFace API token — get yours at huggingface.co/settings/tokens|
|`EVOLUTION_ENABLED`|Optional|`false`|Enable self-evolving backend (`true` or `false`)                   |
|`MAX_RETRIES`      |Optional|`3`    |Self-healing retry limit                                           |
|`LOG_LEVEL`        |Optional|`INFO` |Logging verbosity                                                  |


> ⚠️ **Never commit `.env` to GitHub. Never share tokens in chat.**
> For CI/CD, add `HF_TOKEN` as a repository secret:
> Settings → Secrets and variables → Actions → New repository secret

-----

## Running the Smoke Test

```bash
cd workspace

# Recommended first run — evolution disabled, confirms core pipeline health
python orchestrator.py
```

### Expected Output (passing run)

```json
{
  "run_id": "...",
  "patient_id": "TEST-001",
  "diagnosis": ["Hypertensive urgency", "Acute coronary syndrome"],
  "urgency": "high",
  "actions": ["Immediate ECG", "Blood pressure monitoring", "Cardiology referral"],
  "confidence": 0.87,
  "validation_passed": true,
  "warnings": [],
  "evolution": {},
  "error_log": [],
  "stage": "finalized"
}
```

### What a Passing Run Looks Like

- `validation_passed` is `true`
- `stage` is `"finalized"`
- `error_log` is empty `[]`
- `diagnosis` contains at least one candidate
- `urgency` is one of: `low`, `medium`, `high`, `critical`

### Full Evolution Run (optional)

```bash
EVOLUTION_ENABLED=true python orchestrator.py
```

-----

## Troubleshooting

|Error                                  |Cause                                     |Solution                                                                                       |
|---------------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------|
|`EnvironmentError: HF_TOKEN is not set`|Missing `.env` file or unset variable     |Ensure `.env` exists with `HF_TOKEN=your_token`. Verify token at huggingface.co/settings/tokens|
|`403 Forbidden` from HuggingFace       |Token lacks read permissions or is revoked|Regenerate token with Read role at huggingface.co/settings/tokens                              |
|`Model download timeout`               |Network issue or gated model access       |Check internet connection; ensure token has access to gated models                             |
|`LangGraph compilation error`          |Invalid state schema                      |Run `python -c "import workspace.state_schema; print('Schema OK')"` to verify                  |
|`ModuleNotFoundError`                  |Dependencies not installed                |Run `pip install -r requirements.txt`                                                          |
|`ValidationError` on PipelineState     |Agent output doesn’t match schema         |Compare agent output keys against `state_schema.py` field definitions                          |
|`validation_passed: false` in output   |Agent produced incomplete output          |Check `error_log` in the JSON response for the specific failed field                           |

-----

## CI / CD

GitHub Actions runs automatically on every push to `main` and `dev`:

- ✅ Ruff linting
- ✅ Pydantic schema validation
- ✅ LangGraph graph compilation check
- ✅ HuggingFace model loader health check

See `.github/workflows/ci.yml`

-----

## Dependencies

|Package                |Version|Purpose                         |
|-----------------------|-------|--------------------------------|
|`langgraph`            |≥0.2.28|Stateful graph orchestration    |
|`langchain-core`       |≥0.3.0 |LangChain base abstractions     |
|`langchain-huggingface`|≥0.1.0 |HuggingFace LLM integration     |
|`crewai`               |≥0.80.0|Multi-agent framework           |
|`crewai-tools`         |≥0.14.0|CrewAI tool abstractions        |
|`huggingface-hub`      |≥0.23.0|HuggingFace model hub client    |
|`transformers`         |≥4.44.0|Model tokenization and utilities|
|`pydantic`             |≥2.7.0 |Data validation and schemas     |
|`python-dotenv`        |≥1.0.0 |Environment variable loading    |

-----

## File Structure

```
vision-link-ai-agent/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── workspace/
│   ├── state_schema.py         # Pydantic v2 shared state
│   ├── orchestrator.py         # LangGraph StateGraph + self-evolving engine
│   ├── model_loader.py         # HuggingFace model initializer
│   └── crew_agents.py          # CrewAI agents and tasks
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

-----

## Roadmap

- [ ] REST API wrapper (FastAPI) for enterprise integration
- [ ] Async batch processing for multi-patient workloads
- [ ] Persistent evolution metrics store (SQLite / Redis)
- [ ] HL7 FHIR output format support
- [ ] Extended language support (Hausa, Yoruba, Swahili)
- [ ] Dashboard for real-time pipeline health monitoring

-----

## Team

|Member       |GitHub                                              |Role                                                      |
|-------------|----------------------------------------------------|----------------------------------------------------------|
|Buhari Salisu|[@BuhariSalisuAI](https://github.com/BuhariSalisuAI)|Founder & CEO                                             |
|Subhalaxmi   |[@Subhalaxmi](https://github.com/Subhalaxmi)        |Data Engineer — CrewAI Agents & Tasks                     |
|Zentomo      |[@Zentomo](https://github.com/Zentomo)              |Systems Engineer — LangGraph Orchestrator & Infrastructure|
|Ramitha      |[@Ramitha](https://github.com/Ramitha)              |Documentation                                             |
|Meghana      |[@Meghana](https://github.com/Meghana)              |Frontend Engineer                                                      |

-----

## License

MIT License — see <LICENSE> file for details.

-----

*Vision-Link AI HUB · AI Agent Olympics · Lablab.ai · May 2026*

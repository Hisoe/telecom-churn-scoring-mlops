# Telecom Customer Churn & Retention Scoring Engine

An enterprise-grade, real-time machine learning system designed to predict telecommunication subscriber churn and trigger next-best-action (NBA) retention offers.

This platform implements a **Hybrid Local-to-AWS architecture** leveraging tabular gradient boosting (**LightGBM**) compiled to **ONNX Runtime** for sub-10 ms inference latencies, packaged into scale-to-zero containerized services on **AWS ECS Fargate**.

---

## Architectural Blueprint

```text
+---------------------------------------------------------------------------------------+
|                                    LOCAL RUNTIME                                      |
|                                                                                       |
|  +---------------------+      +---------------------+      +-----------------------+  |
|  | Python 3.11 Virtual | ---> |  Strict Type Gates  | ---> |  12-Factor Settings   |  |
|  |  Environment (`uv`) |      | (Ruff, Mypy Strict) |      | (Pydantic BaseSchema) |  |
|  +---------------------+      +---------------------+      +-----------------------+  |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v  (Upcoming: Phase 0.2+)
+---------------------------------------------------------------------------------------+
|                                  CLOUD TARGET (AWS)                                   |
|                                                                                       |
|  +--------------------+      +--------------------+      +-------------------------+  |
|  |  Data & Artifacts  | ---> | Model Registry &   | ---> | Real-Time Serving       |  |
|  |   (Amazon S3)      |      | Tracking (MLflow)  |      | (FastAPI on ECS Fargate)|  |
|  +--------------------+      +--------------------+      +-------------------------+  |
+---------------------------------------------------------------------------------------+
```

---

## Repository Structure

```text
telecom-churn-mlops/
├── .github/
│   └── workflows/              # CI/CD pipelines (test, build, lint, deploy)
├── configs/
│   ├── base_config.yaml        # Pipeline execution baselines
│   └── model_config.yaml       # Hyperparameters, feature groupings, and metrics
├── data/                       # Local volume mounted for mock datasets
├── infra/
│   └── terraform/              # Terraform AWS modules (IaC)
├── src/
│   └── churn_engine/
│       ├── __init__.py
│       ├── config/             # Typed environment & application configuration
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── data/               # Ingestion, schema validation, feature engineering
│       │   └── __init__.py
│       ├── models/             # LightGBM training, ONNX serialization, evaluation
│       │   └── __init__.py
│       ├── pipelines/          # Automated training and batch scoring pipelines
│       │   └── __init__.py
│       └── serving/            # FastAPI real-time scoring application
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures and mock definitions
│   ├── unit/                   # Unit test suite
│   └── integration/            # Integration and pipeline tests
├── .env.example                # Baseline environment configuration template
├── .gitignore                  # Cloud-native ignore rules for ML assets
├── .pre-commit-config.yaml     # Git hook enforcement configuration
├── pyproject.toml              # PEP 517/621 dependency & tool configuration
└── README.md
```

---

## Technical Standards & Engineering Decisions

* **Dependency Management via `uv`**: Sub-second deterministic dependency resolution using strict semantic constraints (`pyproject.toml`). Prevents dependency drift between local development workstations, CI runners, and production ECS containers.
* **12-Factor Configuration (`pydantic-settings`)**: All runtime parameters, cloud bucket targets, tracking endpoints, and model thresholds are strictly parsed and validated from environment variables prior to execution.
* **Shift-Left Quality Gates**:
  * **Ruff**: Unified linting and formatting enforcing import cleanliness, bug prevention (`B`), and modern Python 3.11 idioms (`UP`).
  * **Mypy (Strict Mode)**: Static type verification with `disallow_untyped_defs = true` to prevent unhandled `NoneType` issues and schema mismatches.
  * **Pre-commit**: Automated verification running formatters, linters, large binary asset checkers, and YAML/TOML validators prior to commit acceptance.

---

## Prerequisites

* **Python**: Version `3.11.x`
* **Package Manager**: `uv` (`>=0.3.0`)
* **Version Control**: `Git`

---

## Local Setup

### 1. Clone Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/Hisoe/telecom-churn-scoring-mlops.git
cd telecom-churn-scoring-mlops

# Create isolated Python 3.11 virtual environment via uv
uv venv .venv --python 3.11

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
# Install core dependencies and development tooling in editable mode
uv pip install -e ".[dev]"
```

### 3. Initialize Environment Variables & Quality Hooks

```bash
# Create local environment configuration file from template
cp .env.example .env

# Install pre-commit hooks into git lifecycle
pre-commit install
```

---

## Verification & Validation

Ensure the local environment passes all quality gates and type configurations:

```bash
# 1. Execute all static analysis, linting, and formatting checks
pre-commit run --all-files

# 2. Validate typed settings schema instantiation
python -c "from churn_engine.config.settings import get_settings; print(get_settings().model_dump_json(indent=2))"
```

---

## Roadmap & Implementation Phases

- [x] **Phase 0.1**: Local Architecture, Taxonomy, Strict Configuration, and Git Quality Baseline
- [x] **Phase 0.2**: Local Mock Architecture (MinIO S3 Mock, PostgreSQL Store, Local MLflow Server)
- [ ] **Phase 0.3**: Base AWS Infrastructure via Terraform (VPC, S3, IAM Roles, OIDC)
- [ ] **Phase 1.0**: Synthetic Data Generation & Point-in-Time Feature Engineering
- [ ] **Phase 2.0**: LightGBM Training Pipeline, ONNX Quantization, and MLflow Tracking
- [ ] **Phase 3.0**: CI/CD Automation & Model Governance Promotion Gates
- [ ] **Phase 4.0**: Low-Latency FastAPI Model Serving & ECS Fargate Task Definitions
- [ ] **Phase 5.0**: Production Observability, Payload Logging, and Drift Detection (Evidently AI)
- [ ] **Phase 6.0**: FinOps Optimization, Cost Tagging, and Auto-scaling Policies

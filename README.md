# Autonomous RecBole-Native Recommendation Pipeline

This repository provides an **Autonomous RecBole-native recommendation pipeline** for training/evaluating recommendations and for runtime serving with policy controls.

## What is built

- A multi-stage recommender orchestration (`main.py`) with execution modes for preparation, full staged pipeline, evaluation, serving, and autonomous RecBole mode.
- An autonomous RecBole pipeline (`src/training_pipeline/autonomous_recbole_pipeline.py`) that integrates:
  - mapping output (`outputs/resolved_mappings.json`),
  - business output (`outputs/rec_config.json`),
  - domain manifests in `config/ecommerce/`.
- Serving components in `src/serving/`:
  - `recommendation_engine.py` for runtime recommendation generation.
  - `policy_engine.py` for safe normalization/clamping of user policy inputs.

## Repository structure (high level)

- `main.py` – pipeline entrypoint.
- `config/pipeline_config.yaml` – global pipeline configuration.
- `config/ecommerce/` – domain manifests used by autonomous mapping/business meaning integration.
- `src/training_pipeline/autonomous_recbole_pipeline.py` – autonomous RecBole-native execution.
- `src/serving/policy_engine.py` – policy schema/defaults and value clamping.
- `src/serving/recommendation_engine.py` – real-time recommendation serving API.
- `outputs/` – generated recommendation/evaluation artifacts.

## Integration contract with Autonomous Mapping Engine & Business Meaning Layer

### Expected upstream outputs

- `outputs/resolved_mappings.json`
  - should include canonical column mapping (at minimum `user_id` and `item_id`; optionally `timestamp`, `label`).
- `outputs/rec_config.json`
  - business-role/weighting/classification output from your Business Meaning layer.

### Expected manifests location

Store manifests in:

- `config/ecommerce/schema.yaml`
- `config/ecommerce/manifest.yaml`
- `config/ecommerce/rules.yaml`
- `config/ecommerce/keywords.yaml`
- `config/ecommerce/metrics.yaml`
- `config/ecommerce/validations.yaml`
- `config/ecommerce/questions.yaml`

The autonomous RecBole mode loads these files and attaches them to runtime `business_context` for downstream logic/auditing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### Autonomous RecBole mode (recommended)

```bash
python main.py --config config/pipeline_config.yaml --mode recbole
```

### Other modes

```bash
python main.py --config config/pipeline_config.yaml --mode prepare
python main.py --config config/pipeline_config.yaml --mode full
python main.py --config config/pipeline_config.yaml --mode evaluate
python main.py --config config/pipeline_config.yaml --mode serve --user_ids "U001,U002"
```

## PolicyEngine user inputs (how to pass runtime policy controls)

`PolicyEngine` supports three input layers (applied in this order):

1. `config.policy_control` from `config/pipeline_config.yaml`
2. `context.policy_control` passed at request time
3. direct `policy_inputs` argument (highest priority)

Values are automatically normalized and clamped into safe bounds by `PolicyEngine.resolve(...)`.

### Supported input keys

- `recommendation_yield_limit` (int, clamped to `1..50`)
- `vector_personalization_focus` (float or percent, normalized to `0..1`)
- `curation_discovery_factor` (float or percent, normalized to `0..1`)
- `corporate_sponsored_promotions` (bool/float/percent)
- `candidate_output_diversity_index` (float, `0..1`)
- `retrieval_batch_size` (int, `64..512`)
- `candidate_search_limit` (int, `100..500`)
- `recency_decay_coefficient` (float, `0.1..0.96`)
- `category_capping_threshold` (int, `1..10`)
- `promotions_injection_percentile_threshold` (float, `0..1`)

### Example: set defaults in config

```yaml
# config/pipeline_config.yaml
policy_control:
  recommendation_yield_limit: 20
  vector_personalization_focus: 0.7
  curation_discovery_factor: 0.2
  corporate_sponsored_promotions: false
```

### Example: pass per-request policy inputs in Python

```python
from src.utils.config_loader import ConfigLoader
from src.serving.recommendation_engine import RecommendationEngine

config = ConfigLoader("config/pipeline_config.yaml")
config.load()
engine = RecommendationEngine(config)

context = {
    "policy_control": {
        "curation_discovery_factor": 0.35
    }
}

policy_inputs = {
    "vector_personalization_focus": 65,  # percent-style accepted -> 0.65
    "recommendation_yield_limit": 15,
    "candidate_search_limit": 250
}

recs = engine.get_recommendations(
    user_id="U001",
    context=context,
    top_k=20,
    policy_inputs=policy_inputs,
)

policy_debug = engine.explain_policy_resolution(context=context, policy_inputs=policy_inputs)
print(policy_debug["effective_policy"])
```

## Outputs

- `outputs/top_k_recommendations.parquet`
- `outputs/evaluation_metrics.json`
- `outputs/pipeline_summary.json`
- RecBole-generated intermediate artifacts under `data/inter/`

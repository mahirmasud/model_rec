# Industrial Multi-Stage Recommendation System Pipeline

## Overview

This is a production-ready, multi-stage recommendation system pipeline built with RecBole v1.2.0, designed for large-scale eCommerce personalization. The system implements a four-stage architecture similar to those used by Amazon, Alibaba, TikTok, and YouTube.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Engineering Pipeline                  │
│              (Inputs: cleaned_dataset, features, etc.)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: LightGCN Retrieval Layer                   │
│         - Candidate generation from millions of items            │
│         - Graph-based collaborative filtering                    │
│         - Top-K candidate retrieval                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Stage 2: SASRec Sequential Personalization             │
│         - Session-aware recommendation                           │
│         - Transformer-based sequence modeling                    │
│         - Short-term intent capture                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Stage 3: DeepFM Ranking Layer                     │
│         - Feature interaction learning                           │
│         - CTR-style ranking optimization                         │
│         - Purchase probability scoring                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│        Stage 4: Diversity & Bias Mitigation Re-ranking           │
│         - Maximum Marginal Relevance (MMR)                       │
│         - Popularity bias reduction                              │
│         - Catalog exploration                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final Recommendations                         │
│         - top_k_recommendations.parquet                          │
│         - recommendation_explanations.json                       │
│         - evaluation_metrics.json                                │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- pip or conda

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Place your feature-engineered datasets in the `data/raw/` directory:

```
data/raw/
├── cleaned_dataset.parquet
├── feature_matrix.parquet
├── feature_definitions.json
├── metadata.json
├── session_features.parquet
├── interaction_features.parquet
├── temporal_features.parquet
├── user_features.parquet
└── item_features.parquet
```

### 2. Configure the Pipeline

Edit `config/pipeline_config.yaml` to customize:

- Model hyperparameters
- Train/validation/test split ratios
- Retrieval candidate count
- Ranking features
- Diversity settings

### 3. Run the Pipeline

```bash
# Full pipeline execution
python main.py --config config/pipeline_config.yaml --mode full

# Individual stages
python main.py --config config/pipeline_config.yaml --mode retrieve    # Stage 1
python main.py --config config/pipeline_config.yaml --mode personalize # Stage 2
python main.py --config config/pipeline_config.yaml --mode rank        # Stage 3
python main.py --config config/pipeline_config.yaml --mode rerank      # Stage 4

# Evaluation only
python main.py --config config/pipeline_config.yaml --mode evaluate

# Generate recommendations for specific users
python main.py --config config/pipeline_config.yaml --mode serve --user_ids "U001,U002,U003"
```

## Configuration

### Main Configuration (`config/pipeline_config.yaml`)

```yaml
# Data paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  inter_dir: "data/inter"
  output_dir: "outputs"

# LightGCN retrieval settings
retrieval:
  model: "LightGCN"
  embedding_dim: 64
  n_layers: 3
  reg_weight: 1e-4
  top_k: 500
  learning_rate: 0.001
  epochs: 50

# SASRec sequential settings
sequential:
  model: "SASRec"
  hidden_size: 128
  num_heads: 4
  num_blocks: 2
  dropout_rate: 0.1
  max_seq_length: 50
  top_k: 100

# DeepFM ranking settings
ranking:
  model: "DeepFM"
  embedding_dim: 32
  mlp_hidden_sizes: [128, 64, 32]
  dropout_rates: [0.3, 0.3, 0.3]
  learning_rate: 0.001

# Diversity re-ranking settings
reranking:
  diversity_weight: 0.3
  novelty_weight: 0.2
  freshness_weight: 0.1
  final_top_k: 20

# Training settings
training:
  batch_size: 256
  eval_batch_size: 4096
  early_stopping_patience: 10
  gpu_id: 0
  use_gpu: true

# Evaluation metrics
evaluation:
  metrics: ["Recall", "Precision", "NDCG", "MAP", "HitRate"]
  top_k_list: [10, 20, 50]
```

## Output Files

After running the pipeline, you'll find these outputs in `outputs/`:

| File | Description |
|------|-------------|
| `top_k_recommendations.parquet` | Final ranked recommendations per user |
| `retrieval_candidates.parquet` | LightGCN retrieval candidates |
| `personalized_candidates.parquet` | SASRec personalized candidates |
| `ranking_scores.parquet` | DeepFM ranking scores |
| `diversity_scores.parquet` | Diversity metrics per recommendation |
| `feature_importance.json` | Feature importance from DeepFM |
| `recommendation_explanations.json` | Explainable recommendation metadata |
| `evaluation_metrics.json` | All evaluation metrics |
| `training_logs/` | Training logs and checkpoints |
| `model_checkpoints/` | Saved model weights |

## Features

### Automatic Dataset Conversion

The system automatically converts feature-engineered datasets into RecBole-compatible formats:

- `.inter` files for interactions
- `.user` files for user features
- `.item` files for item features
- Sequence datasets for SASRec

### Intelligent Feature Mapping

Engineered features are automatically mapped to appropriate model layers:

| Feature Type | LightGCN | SASRec | DeepFM |
|-------------|----------|--------|--------|
| Graph interactions | ✓ | | |
| Session sequences | | ✓ | |
| Temporal features | | ✓ | ✓ |
| User demographics | | | ✓ |
| Item attributes | | | ✓ |
| Behavioral clusters | ✓ | ✓ | ✓ |

### Diversity-Aware Re-ranking

Implements Maximum Marginal Relevance (MMR) with:

- Category balancing
- Seller diversity
- Price range distribution
- Novelty optimization
- Freshness consideration

### Explainability

Each recommendation includes:

- Retrieval source explanation
- Sequential influence factors
- Feature contribution scores
- Diversity adjustment reasons

## Advanced Usage

### Custom Feature Engineering Integration

```python
from src.data_loader.feature_mapper import FeatureMapper

mapper = FeatureMapper(config)
mapped_features = mapper.map_features(
    feature_matrix="data/raw/feature_matrix.parquet",
    feature_definitions="data/raw/feature_definitions.json"
)
```

### Real-time Serving

```python
from src.serving.recommendation_engine import RecommendationEngine

engine = RecommendationEngine(config)
recommendations = engine.get_recommendations(
    user_id="U001",
    context={"session_id": "S123", "timestamp": "2024-01-15T10:30:00"}
)
```

### Incremental Updates

```bash
# Update models with new interactions
python main.py --config config/pipeline_config.yaml --mode incremental --new_data data/raw/new_interactions.parquet
```

## Performance Optimization

### For Large-Scale Deployment

1. **Enable GPU Acceleration**
   ```yaml
   training:
     use_gpu: true
     gpu_id: 0
   ```

2. **Use ANN Retrieval**
   ```yaml
   retrieval:
     use_ann: true
     ann_index_type: "IVF_PQ"
   ```

3. **Batch Inference**
   ```python
   recommendations = engine.batch_recommend(user_ids, batch_size=1024)
   ```

4. **Embedding Caching**
   ```yaml
   serving:
     cache_embeddings: true
     cache_ttl: 3600
   ```

## Evaluation Metrics

### Ranking Metrics
- Recall@K
- Precision@K
- NDCG@K
- MAP@K
- HitRate@K
- MRR

### Diversity Metrics
- Intra-list Diversity (ILD)
- Catalog Coverage
- Novelty Score
- Serendipity

### Bias Metrics
- Popularity Bias
- Exposure Bias
- Long-tail Coverage

## Project Structure

```
recsys_pipeline/
├── config/
│   ├── pipeline_config.yaml
│   ├── lightgcn_config.yaml
│   ├── sasrec_config.yaml
│   └── deepfm_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── inter/
├── src/
│   ├── data_loader/
│   │   ├── __init__.py
│   │   ├── dataset_builder.py
│   │   ├── feature_mapper.py
│   │   └── recbole_converter.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── lightgnn_retriever.py
│   │   └── ann_search.py
│   ├── sequential/
│   │   ├── __init__.py
│   │   ├── sasrec_personalizer.py
│   │   └── session_handler.py
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── deepfm_ranker.py
│   │   └── feature_encoder.py
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── diversity_reranker.py
│   │   └── bias_mitigator.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics_calculator.py
│   │   └── bias_evaluator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── config_loader.py
│   │   └── helpers.py
│   └── serving/
│       ├── __init__.py
│       └── recommendation_engine.py
├── models/
├── logs/
├── checkpoints/
├── outputs/
├── main.py
├── requirements.txt
└── README.md
```

## License

MIT License

## Citation

If you use this system in your research, please cite:


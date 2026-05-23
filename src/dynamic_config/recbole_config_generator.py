from pathlib import Path
from typing import Any, Dict

import yaml


class RecBoleConfigGenerator:
    def __init__(self, dataset_name: str = "recsys_dataset"):
        self.dataset_name = dataset_name

    def build(self, data_path: str, mapping: Dict[str, Any], model: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        normalized_data_path = Path(data_path)
        # RecBole expects `data_path` as a directory containing `<dataset>.inter`.
        # If callers pass a dataset-stem path (e.g., data/inter/recsys_dataset),
        # normalize it back to the parent directory.
        if normalized_data_path.name == self.dataset_name:
            normalized_data_path = normalized_data_path.parent

        cfg = {
            "model": model,
            "dataset": self.dataset_name,
            "data_path": str(normalized_data_path.resolve()),
            "USER_ID_FIELD": mapping["user_id"],
            "ITEM_ID_FIELD": mapping["item_id"],
            "TIME_FIELD": mapping.get("timestamp"),
            "LABEL_FIELD": mapping.get("label") or "label",
            "field_separator": "\t",
            "load_col": {
                "inter": "*",
                "user": "*",
                "item": "*",
            },
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1, "alpha": 1.0, "dynamic": False, "candidate_num": 0},
            "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO", "group_by": "user", "mode": "full"},
            "topk": [10, 20, 50],
            "metrics": ["Recall", "Precision", "NDCG", "MAP", "Hit", "MRR"],
            "valid_metric": "NDCG@10",
        }
        # Point-wise CE losses (e.g., DeepFM) should not use training negative sampling.
        if model.lower() in {"deepfm"}:
            cfg["train_neg_sample_args"] = {"distribution": "none", "sample_num": "none", "alpha": "none", "dynamic": False, "candidate_num": 0}
        if overrides:
            cfg.update(overrides)
        return cfg

    def save(self, cfg: Dict[str, Any], path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return str(p)

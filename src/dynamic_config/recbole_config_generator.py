from pathlib import Path
from typing import Any, Dict

import yaml


class RecBoleConfigGenerator:
    def __init__(self, dataset_name: str = "recsys_dataset"):
        self.dataset_name = dataset_name

    def build(self, data_path: str, mapping: Dict[str, Any], model: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cfg = {
            "model": model,
            "dataset": self.dataset_name,
            "data_path": data_path,
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
            "neg_sampling": {"uniform": 1},
            "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO", "group_by": "user", "mode": "full"},
            "topk": [10, 20, 50],
            "metrics": ["Recall", "Precision", "NDCG", "MAP", "Hit", "MRR"],
            "valid_metric": "NDCG@10",
        }
        if overrides:
            cfg.update(overrides)
        return cfg

    def save(self, cfg: Dict[str, Any], path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return str(p)

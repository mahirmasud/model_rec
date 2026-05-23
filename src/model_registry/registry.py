from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelSpec:
    name: str
    recbole_model: str
    task: str


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelSpec] = {
            "retrieval": ModelSpec("LightGCN Retrieval", "LightGCN", "retrieval"),
            "sequential": ModelSpec("SASRec Sequential", "SASRec", "sequential"),
            "ranking": ModelSpec("DeepFM Ranking", "DeepFM", "ranking"),
        }

    def get(self, stage: str) -> ModelSpec:
        return self._models[stage]

    def all(self):
        return self._models

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..schema_inference.schema_inferer import SchemaInferer
from ..data_loader.recbole_converter import RecBoleConverter
from ..dynamic_config.recbole_config_generator import RecBoleConfigGenerator
from ..model_registry.registry import ModelRegistry
from ..recbole_adapter.runner import RecBoleRunner


class AutonomousRecBolePipeline:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.inferer = SchemaInferer()
        self.registry = ModelRegistry()
        self.runner = RecBoleRunner(logger=logger)
        self.converter = RecBoleConverter(config, logger)

    def _standardize(self, df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
        out = pd.DataFrame()
        out["user_id"] = df[mapping["user_id"]]
        out["item_id"] = df[mapping["item_id"]]
        out["rating"] = df[mapping["label"]] if mapping.get("label") and mapping["label"] in df.columns else 1
        if mapping.get("timestamp") and mapping["timestamp"] in df.columns:
            out["timestamp"] = pd.to_datetime(df[mapping["timestamp"]], errors="coerce")
        return out.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)

    def run(self, raw_df: pd.DataFrame, inter_dir: str) -> Dict[str, Any]:
        mapping = self.inferer.infer(raw_df)
        interactions = self._standardize(raw_df, mapping)

        ds_cfg = self.converter.convert_interactions(interactions, inter_dir)
        gen = RecBoleConfigGenerator(dataset_name=ds_cfg["dataset_name"])
        base = Path(inter_dir)
        inter_file = base / f'{ds_cfg["dataset_name"]}.inter'
        if not inter_file.exists():
            raise FileNotFoundError(f"Expected RecBole interaction file not found: {inter_file}")

        outputs = {"mapping": mapping, "dataset": ds_cfg, "stages": {}}
        for stage in ["retrieval", "sequential", "ranking"]:
            spec = self.registry.get(stage)
            cfg = gen.build(str(base), {"user_id": "user_id", "item_id": "item_id", "timestamp": "timestamp", "label": "rating"}, spec.recbole_model)
            cfg_path = gen.save(cfg, str(base / f"{spec.recbole_model.lower()}_autogen.yaml"))
            outputs["stages"][stage] = self.runner.train_and_eval(cfg_path)

        outputs["fallback_predictions"] = self.runner.predict_topk_placeholder(interactions, "user_id", "item_id", k=20)
        return outputs

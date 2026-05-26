from pathlib import Path
from typing import Any, Dict, Optional

import json

import pandas as pd
import yaml

from ..schema_inference.schema_inferer import SchemaInferer
from ..data_loader.recbole_converter import RecBoleConverter
from ..dynamic_config.recbole_config_generator import RecBoleConfigGenerator
from ..model_registry.registry import ModelRegistry
from ..recbole_adapter.runner import RecBoleRunner


class AutonomousRecBolePipeline:
    """RecBole-native pipeline that can consume external mapping + business manifests."""

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.inferer = SchemaInferer()
        self.registry = ModelRegistry()
        self.runner = RecBoleRunner(logger=logger)
        self.converter = RecBoleConverter(config, logger)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_mapping(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        mapping_path = Path(self.config.get("autonomous.mapping_output_path", "outputs/resolved_mappings.json"))
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if self.logger:
                self.logger.info(f"Loaded mapping output from {mapping_path}")
            return payload.get("mapping", payload)

        if self.logger:
            self.logger.warning(
                f"Mapping output not found at {mapping_path}; falling back to built-in schema inference."
            )
        return self.inferer.infer(raw_df)

    def _load_business_context(self) -> Dict[str, Any]:
        domain_dir = Path(self.config.get("autonomous.domain_dir", "config/ecommerce"))
        context = {
            "schema": self._load_yaml(domain_dir / "schema.yaml"),
            "manifest": self._load_yaml(domain_dir / "manifest.yaml"),
            "rules": self._load_yaml(domain_dir / "rules.yaml"),
            "keywords": self._load_yaml(domain_dir / "keywords.yaml"),
            "metrics": self._load_yaml(domain_dir / "metrics.yaml"),
            "validations": self._load_yaml(domain_dir / "validations.yaml"),
            "questions": self._load_yaml(domain_dir / "questions.yaml"),
        }

        rec_config_path = Path(self.config.get("autonomous.business_output_path", "outputs/rec_config.json"))
        if rec_config_path.exists():
            with rec_config_path.open("r", encoding="utf-8") as f:
                context["business_output"] = json.load(f)
        else:
            context["business_output"] = {}
        if self.logger:
            self.logger.info(f"Loaded business manifests from {domain_dir}")
        return context

    def _resolve_column(self, raw_df: pd.DataFrame, mapping: Dict[str, Any], key: str, fallback: Optional[str] = None) -> Optional[str]:
        value = mapping.get(key)
        if value and value in raw_df.columns:
            return value
        return fallback if fallback in raw_df.columns else None

    def _standardize(self, df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
        user_col = self._resolve_column(df, mapping, "user_id")
        item_col = self._resolve_column(df, mapping, "item_id")
        label_col = self._resolve_column(df, mapping, "label")
        ts_col = self._resolve_column(df, mapping, "timestamp")

        if not user_col or not item_col:
            raise ValueError("Autonomous mapping must provide resolvable user_id and item_id columns.")

        out = pd.DataFrame()
        out["user_id"] = df[user_col]
        out["item_id"] = df[item_col]
        out["rating"] = df[label_col] if label_col else 1
        if ts_col:
            out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        return out.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)

    def run(self, raw_df: pd.DataFrame, inter_dir: str) -> Dict[str, Any]:
        mapping = self._load_mapping(raw_df)
        business_context = self._load_business_context()
        interactions = self._standardize(raw_df, mapping)

        base_dir = Path(inter_dir)
        dataset_dir = base_dir / self.converter.dataset_name
        ds_cfg = self.converter.convert_interactions(interactions, str(dataset_dir))
        gen = RecBoleConfigGenerator(dataset_name=ds_cfg["dataset_name"])
        inter_file = dataset_dir / f'{ds_cfg["dataset_name"]}.inter'
        if not inter_file.exists():
            raise FileNotFoundError(f"Expected RecBole interaction file not found: {inter_file}")

        outputs = {
            "mapping": mapping,
            "business_context": business_context,
            "dataset": ds_cfg,
            "stages": {},
        }

        for stage in ["retrieval", "sequential", "ranking"]:
            spec = self.registry.get(stage)
            cfg = gen.build(
                str(base_dir),
                {"user_id": "user_id", "item_id": "item_id", "timestamp": "timestamp", "label": "rating"},
                spec.recbole_model,
            )
            cfg_path = gen.save(cfg, str(dataset_dir / f"{spec.recbole_model.lower()}_autogen.yaml"))
            outputs["stages"][stage] = self.runner.train_and_eval(cfg_path)

        outputs["fallback_predictions"] = self.runner.predict_topk_placeholder(interactions, "user_id", "item_id", k=20)
        return outputs

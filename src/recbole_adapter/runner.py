from typing import Any, Dict
import importlib.util

import pandas as pd


class RecBoleRunner:
    """Thin adapter around native RecBole quick_start APIs with safe fallback."""

    def __init__(self, logger=None):
        self.logger = logger

    def train_and_eval(self, config_file: str) -> Dict[str, Any]:
        """Run native RecBole training/eval; provide actionable fallback diagnostics."""
        missing = []
        if importlib.util.find_spec("recbole") is None:
            missing.append("recbole")
        # RecBole v1.2 frequently imports ray in quick_start paths
        if importlib.util.find_spec("ray") is None:
            missing.append("ray")

        if missing:
            msg = f"Missing runtime dependency(ies): {', '.join(missing)}"
            if self.logger:
                self.logger.warning(f"RecBole training fallback triggered: {msg}")
            return {"status": "fallback", "error": msg, "missing_dependencies": missing}
        try:
            from recbole.quick_start import run_recbole
            result = run_recbole(config_file_list=[config_file])
            return {"status": "trained", "result": result}
        except Exception as e:  # graceful in environments without GPU/RecBole runtime
            if self.logger:
                self.logger.warning(
                    "RecBole training fallback triggered. "
                    "If this is a fresh environment, ensure recbole and ray are installed. "
                    f"Error: {e}"
                )
            return {"status": "fallback", "error": str(e)}

    def predict_topk_placeholder(self, interactions: pd.DataFrame, user_col: str, item_col: str, k: int = 50) -> pd.DataFrame:
        pop = interactions[item_col].value_counts().index.tolist()
        rows = []
        for u in interactions[user_col].dropna().unique():
            for r, i in enumerate(pop[:k], 1):
                rows.append({"user_id": u, "item_id": i, "score": 1.0 / r, "rank": r})
        return pd.DataFrame(rows)

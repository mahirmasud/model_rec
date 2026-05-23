from pathlib import Path
from typing import Any, Dict

import pandas as pd


class RecBoleRunner:
    """Thin adapter around native RecBole quick_start APIs with safe fallback."""

    def __init__(self, logger=None):
        self.logger = logger

    def train_and_eval(self, config_file: str) -> Dict[str, Any]:
        try:
            from recbole.quick_start import run_recbole
            result = run_recbole(config_file_list=[config_file])
            return {"status": "trained", "result": result}
        except ModuleNotFoundError as e:
            if self.logger:
                if getattr(e, "name", "") == "ray":
                    self.logger.warning(
                        "RecBole training fallback triggered: missing optional dependency 'ray'. "
                        "Install it with `pip install ray` (or install -r requirements.txt)."
                    )
                else:
                    self.logger.warning(f"RecBole training fallback triggered: missing module '{e.name}'")
            return {"status": "fallback", "error": str(e)}
        except Exception as e:  # graceful in environments without GPU/RecBole runtime
            if self.logger:
                self.logger.warning(f"RecBole training fallback triggered: {e}")
            return {"status": "fallback", "error": str(e)}

    def predict_topk_placeholder(self, interactions: pd.DataFrame, user_col: str, item_col: str, k: int = 50) -> pd.DataFrame:
        pop = interactions[item_col].value_counts().index.tolist()
        rows = []
        for u in interactions[user_col].dropna().unique():
            for r, i in enumerate(pop[:k], 1):
                rows.append({"user_id": u, "item_id": i, "score": 1.0 / r, "rank": r})
        return pd.DataFrame(rows)

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch


class RecBoleRunner:
    """Thin adapter around native RecBole quick_start APIs with safe fallback."""

    def __init__(self, logger=None):
        self.logger = logger

    def _apply_numpy_compat(self) -> None:
        """Patch NumPy 2.x alias removals that older RecBole paths may still use."""
        patched = []
        if not hasattr(np, "float_"):
            np.float_ = np.float64
            patched.append("np.float_ -> np.float64")
        if not hasattr(np, "int_"):
            np.int_ = np.int64
            patched.append("np.int_ -> np.int64")

        if patched and self.logger:
            self.logger.info(
                "Applied NumPy compatibility aliases for RecBole: %s",
                ", ".join(patched),
            )

    def train_and_eval(self, config_file: str) -> Dict[str, Any]:
        try:
            self._apply_numpy_compat()
            from recbole.quick_start import run_recbole
            result = run_recbole(config_file_list=[config_file])
            return {"status": "trained", "result": result}
        except Exception as e:  # graceful in environments without GPU/RecBole runtime
            # PyTorch 2.6 defaults torch.load(..., weights_only=True), which can fail
            # for older RecBole checkpoint serialization formats. Retry once by
            # explicitly defaulting weights_only to False for trusted local checkpoints.
            if "Weights only load failed" in str(e):
                if self.logger:
                    self.logger.warning(
                        "RecBole hit PyTorch weights_only checkpoint load issue; retrying with "
                        "torch.load(weights_only=False) compatibility shim."
                    )
                try:
                    original_torch_load = torch.load

                    def _compat_torch_load(*args, **kwargs):
                        kwargs.setdefault("weights_only", False)
                        return original_torch_load(*args, **kwargs)

                    torch.load = _compat_torch_load
                    from recbole.quick_start import run_recbole
                    result = run_recbole(config_file_list=[config_file])
                    return {"status": "trained", "result": result, "compat_retry": "weights_only_false"}
                except Exception as retry_error:
                    if self.logger:
                        self.logger.warning(f"RecBole compatibility retry failed: {retry_error}")
                finally:
                    torch.load = original_torch_load
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

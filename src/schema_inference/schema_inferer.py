import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_ratio: float
    unique_ratio: float
    n_unique: int
    semantic_scores: Dict[str, float] = field(default_factory=dict)


class SchemaInferer:
    """Autonomous schema inference for arbitrary recommendation datasets."""

    KEYWORDS = {
        "user_id": ["user", "uid", "customer", "member", "account"],
        "item_id": ["item", "iid", "product", "sku", "asin", "content", "video", "article"],
        "timestamp": ["time", "timestamp", "date", "event_time", "created", "ts"],
        "label": ["rating", "label", "target", "click", "purchase", "watch", "like", "interact"],
        "session_id": ["session", "sid", "visit"],
        "category": ["category", "genre", "taxonomy", "department"],
        "brand": ["brand", "seller", "vendor", "shop"],
        "price": ["price", "cost", "amount"],
        "device": ["device", "os", "browser", "app"],
        "location": ["country", "city", "state", "region", "location", "geo"],
        "platform": ["platform", "channel", "source", "traffic", "campaign"],
    }

    def profile(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        profiles: Dict[str, ColumnProfile] = {}
        n = max(1, len(df))
        for c in df.columns:
            s = df[c]
            profiles[c] = ColumnProfile(
                name=c,
                dtype=str(s.dtype),
                null_ratio=float(s.isna().mean()),
                unique_ratio=float(s.nunique(dropna=True) / n),
                n_unique=int(s.nunique(dropna=True)),
            )
        return profiles

    def infer(self, df: pd.DataFrame) -> Dict[str, Any]:
        profiles = self.profile(df)
        for col, p in profiles.items():
            low = col.lower()
            for role, kws in self.KEYWORDS.items():
                score = 0.0
                for kw in kws:
                    if kw in low:
                        score += 0.6
                if role.endswith("_id") and p.unique_ratio > 0.1:
                    score += 0.2
                if role == "timestamp" and (is_datetime64_any_dtype(df[col]) or "time" in low or "date" in low):
                    score += 0.5
                if role == "price" and is_numeric_dtype(df[col]):
                    score += 0.2
                p.semantic_scores[role] = score

        def best(role: str, fallback: Optional[str] = None) -> Optional[str]:
            ranked = sorted(profiles.values(), key=lambda x: x.semantic_scores.get(role, 0), reverse=True)
            if ranked and ranked[0].semantic_scores.get(role, 0) > 0:
                return ranked[0].name
            return fallback

        user_col = best("user_id")
        item_col = best("item_id")
        time_col = best("timestamp")
        label_col = best("label")

        if user_col == item_col:
            # cardinality-based disambiguation
            cands = sorted(df.columns, key=lambda c: df[c].nunique(dropna=True), reverse=True)
            if len(cands) > 1:
                item_col = cands[1]

        if label_col in {user_col, item_col, time_col}:
            label_col = None

        return {
            "user_id": user_col,
            "item_id": item_col,
            "timestamp": time_col,
            "label": label_col,
            "session_id": best("session_id"),
            "category": best("category"),
            "brand": best("brand"),
            "price": best("price"),
            "device": best("device"),
            "location": best("location"),
            "platform": best("platform"),
            "profiles": {k: vars(v) for k, v in profiles.items()},
        }

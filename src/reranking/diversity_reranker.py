"""
Diversity Reranker - Stage 4: LambdaMART-based learned re-ranking.
Replaces heuristic MMR with a LightGBM ranker while preserving output schema.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from ..utils.config_loader import ConfigLoader


class DiversityReranker:
    """Learned reranker that optimizes final ordering with LambdaMART."""

    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.final_top_k = int(config.get("reranking.final_top_k", 20))
        self.max_same_category = int(config.get("reranking.category_balance.max_same_category", 5))
        self.max_same_seller = int(config.get("reranking.seller_balance.max_same_seller", 3))

        self.model = LGBMRanker(
            objective=config.get("reranking.objective", "lambdarank"),
            metric=config.get("reranking.metric", "ndcg"),
            boosting_type=config.get("reranking.boosting_type", "gbdt"),
            num_leaves=int(config.get("reranking.num_leaves", 31)),
            learning_rate=float(config.get("reranking.learning_rate", 0.05)),
            n_estimators=int(config.get("reranking.n_estimators", 200)),
            max_depth=int(config.get("reranking.max_depth", -1)),
            feature_fraction=float(config.get("reranking.feature_fraction", 0.8)),
            min_data_in_leaf=int(config.get("reranking.min_data_in_leaf", 20)),
            verbosity=int(config.get("reranking.verbosity", -1)),
            random_state=int(config.get("reranking.random_state", 42)),
        )
        self.feature_columns: List[str] = []
        self.item_embeddings: Dict[str, np.ndarray] = {}

    @staticmethod
    def _safe_norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    def _build_item_embeddings(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if "item_id" not in df.columns:
            return {}
        cols = [c for c in ["similarity_score", "ranking_score", "retrieval_score", "sequential_score"] if c in df.columns]
        if not cols:
            return {}
        emb = df.groupby("item_id")[cols].mean().astype(float)
        arr = self._safe_norm(emb.values)
        return {item_id: arr[i] for i, item_id in enumerate(emb.index.astype(str).tolist())}

    def _compute_user_behavior_features(self, g: pd.DataFrame) -> Tuple[float, float, float, float]:
        s = g["ranking_score"].astype(float).values
        if len(s) == 0:
            return 0.0, 0.0, 0.0, 0.0
        p = np.clip(s / (s.sum() + 1e-12), 1e-12, 1.0)
        entropy = float(-(p * np.log(p)).sum())
        exploration = float(np.std(s))
        repeat = float((g["item_popularity"] > g["item_popularity"].median()).mean()) if "item_popularity" in g else 0.0
        session_div = float(g["category"].nunique() / max(len(g), 1)) if "category" in g else 0.0
        return session_div, exploration, repeat, entropy

    def build_feature_matrix(self, candidates: pd.DataFrame) -> pd.DataFrame:
        df = candidates.copy()
        if df.empty:
            return df

        for col, default in [("ranking_score", 0.0), ("similarity_score", 0.0), ("retrieval_score", 0.0), ("sequential_score", 0.0)]:
            if col not in df.columns:
                df[col] = default

        for meta in ["category", "seller", "brand"]:
            if meta not in df.columns:
                df[meta] = "unknown"
        if "seller" not in df.columns and "brand" in df.columns:
            df["seller"] = df["brand"].astype(str)

        pop = df["item_id"].value_counts().to_dict()
        df["item_popularity"] = df["item_id"].map(pop).fillna(1).astype(float)
        max_pop = max(pop.values()) if pop else 1
        df["inverse_popularity"] = 1.0 / (df["item_popularity"] + 1.0)
        df["long_tail_indicator"] = (df["item_popularity"] <= np.median(list(pop.values()) if pop else [1])).astype(int)

        now = datetime.now(timezone.utc)
        if "created_date" in df.columns:
            created = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
            age = (now - created).dt.days.fillna(30).clip(lower=0)
        else:
            age = pd.Series(np.full(len(df), 30), index=df.index)
        df["item_age"] = age.astype(float)
        decay_days = float(self.config.get("reranking.freshness.decay_days", 30))
        df["recency_decay"] = np.exp(-df["item_age"] / max(decay_days, 1.0))
        df["trending_score"] = df["inverse_popularity"] * df["recency_decay"]

        self.item_embeddings = self._build_item_embeddings(df)
        vectors = np.vstack([self.item_embeddings.get(str(i), np.zeros(4)) for i in df["item_id"].astype(str)])
        vectors = self._safe_norm(vectors)

        cosine = vectors @ vectors.T
        np.fill_diagonal(cosine, 0.0)
        df["mean_similarity"] = cosine.mean(axis=1)
        df["max_similarity"] = cosine.max(axis=1)
        df["cosine_similarity_to_selected"] = df["max_similarity"]
        df["embedding_distance"] = 1.0 - df["mean_similarity"]

        df["category_repetition_count"] = df.groupby(["user_id", "category"]).cumcount()
        df["seller_repetition_count"] = df.groupby(["user_id", "seller"]).cumcount()
        df["category_saturation"] = df.groupby("user_id")["category"].transform(lambda s: s.map(s.value_counts()) / len(s))
        df["seller_saturation"] = df.groupby("user_id")["seller"].transform(lambda s: s.map(s.value_counts()) / len(s))
        df["duplicate_penalty"] = df.duplicated(["user_id", "item_id"]).astype(int)

        behavior = df.groupby("user_id").apply(self._compute_user_behavior_features)
        behavior_df = pd.DataFrame(behavior.tolist(), index=behavior.index, columns=[
            "session_diversity_preference", "exploration_tendency", "repeat_interaction_tendency", "sequential_entropy"
        ])
        df = df.merge(behavior_df, left_on="user_id", right_index=True, how="left")

        df["novelty_score"] = df["inverse_popularity"]
        df["freshness_score"] = df["recency_decay"]
        df["diversity_score"] = 1.0 - df["max_similarity"].clip(0, 1)

        return df

    def _build_training_labels(self, df: pd.DataFrame) -> pd.Series:
        if "interaction" in df.columns:
            return (df["interaction"].astype(float) > 0).astype(int)
        blend = 0.6 * df["ranking_score"] + 0.2 * df["novelty_score"] + 0.2 * df["freshness_score"]
        thr = blend.groupby(df["user_id"]).transform("median")
        return (blend >= thr).astype(int)

    def fit(self, candidates: pd.DataFrame) -> None:
        df = self.build_feature_matrix(candidates)
        if df.empty:
            raise ValueError("No candidates for reranker training")

        labels = self._build_training_labels(df)
        self.feature_columns = [
            "ranking_score", "retrieval_score", "sequential_score", "similarity_score",
            "cosine_similarity_to_selected", "mean_similarity", "max_similarity", "category_repetition_count",
            "seller_repetition_count", "embedding_distance", "item_popularity", "inverse_popularity",
            "long_tail_indicator", "item_age", "recency_decay", "trending_score",
            "session_diversity_preference", "exploration_tendency", "repeat_interaction_tendency", "sequential_entropy",
            "category_saturation", "seller_saturation", "duplicate_penalty", "novelty_score", "freshness_score", "diversity_score",
        ]
        x = df[self.feature_columns].fillna(0.0)
        nunique = x.nunique(dropna=False)
        non_constant = nunique[nunique > 1].index.tolist()
        removed = sorted(set(self.feature_columns) - set(non_constant))
        if removed:
            self.logger.info(f"Dropping constant reranker features: {removed}")
        self.feature_columns = non_constant
        x = x[self.feature_columns]
        mu = x.mean(axis=0)
        sigma = x.std(axis=0).replace(0.0, 1.0)
        x = (x - mu) / sigma
        group = df.groupby("user_id").size().tolist()
        self.model.fit(x, labels, group=group)

    def apply_constraints(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        filtered: List[Dict[str, Any]] = []
        for user_id, group in recommendations.groupby("user_id"):
            cat_counts: Dict[str, int] = {}
            seller_counts: Dict[str, int] = {}
            user_rows = []
            for row in group.sort_values("rerank_score", ascending=False).to_dict("records"):
                cat = str(row.get("category", "unknown"))
                seller = str(row.get("seller", row.get("brand", "unknown")))
                if cat_counts.get(cat, 0) >= self.max_same_category:
                    continue
                if seller_counts.get(seller, 0) >= self.max_same_seller:
                    continue
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                seller_counts[seller] = seller_counts.get(seller, 0) + 1
                user_rows.append(row)
                if len(user_rows) >= self.final_top_k:
                    break
            filtered.extend(user_rows)
        out = pd.DataFrame(filtered)
        if not out.empty:
            out["final_rank"] = out.groupby("user_id")["rerank_score"].rank(ascending=False, method="first").astype(int)
        return out

    def run(self, ranking_results: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Running LambdaMART reranking...")
        ranked_candidates = ranking_results.get("ranked_candidates", pd.DataFrame())
        if ranked_candidates.empty:
            return {"n_final": 0, "final_recommendations": pd.DataFrame(), "diversity_metrics": {}}

        self.fit(ranked_candidates)
        features_df = self.build_feature_matrix(ranked_candidates)
        x_pred = features_df[self.feature_columns].fillna(0.0)
        mu = x_pred.mean(axis=0)
        sigma = x_pred.std(axis=0).replace(0.0, 1.0)
        x_pred = (x_pred - mu) / sigma
        features_df["rerank_score"] = self.model.predict(x_pred)
        final_df = self.apply_constraints(features_df)

        if "deepfm_score" not in final_df.columns:
            final_df["deepfm_score"] = final_df.get("ranking_score", 0.0)
        if "sasrec_score" not in final_df.columns:
            final_df["sasrec_score"] = final_df.get("sequential_score", 0.0)
        if "lightgcn_score" not in final_df.columns:
            final_df["lightgcn_score"] = final_df.get("retrieval_score", 0.0)

        required_cols = [
            "user_id", "item_id", "rerank_score", "final_rank", "ranking_score", "retrieval_score", "sequential_score",
            "deepfm_score", "sasrec_score", "lightgcn_score", "similarity_score",
            "diversity_score", "novelty_score", "freshness_score",
        ]
        for c in required_cols:
            if c not in final_df.columns:
                final_df[c] = 0.0
        final_df = final_df[required_cols]

        diversity_metrics = {
            "avg_diversity_score": float(final_df["diversity_score"].mean()) if not final_df.empty else 0.0,
            "n_unique_items": int(final_df["item_id"].nunique()) if not final_df.empty else 0,
            "n_users_covered": int(final_df["user_id"].nunique()) if not final_df.empty else 0,
        }
        return {
            "n_final": len(final_df),
            "n_users": int(final_df["user_id"].nunique()) if not final_df.empty else 0,
            "final_top_k": self.final_top_k,
            "final_recommendations": final_df,
            "diversity_metrics": diversity_metrics,
        }

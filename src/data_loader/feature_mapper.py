"""
Feature Mapper - Maps engineered features to model-specific formats.
Handles automatic feature type detection and assignment to model layers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json

from ..utils.helpers import load_parquet, load_json, save_json, detect_field
from ..utils.config_loader import ConfigLoader


class FeatureMapper:
    """
    Maps feature-engineered datasets to model-specific feature sets.
    
    Automatically assigns features to appropriate model layers:
    - LightGCN: Graph interactions, collaborative signals
    - SASRec: Session sequences, temporal order
    - DeepFM: Dense numerical, sparse categorical, contextual
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.feature_definitions: Optional[Dict] = None
        self.mapped_features: Dict[str, Any] = {}
        
    def load_feature_definitions(self, path: str) -> 'FeatureMapper':
        """Load feature definitions from JSON file."""
        self.feature_definitions = load_json(path)
        self.logger.info(f"Loaded feature definitions with {len(self.feature_definitions.get('features', []))} features")
        return self
    
    def analyze_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze DataFrame and categorize features by type."""
        feature_types = {
            'numerical': [],
            'categorical': [],
            'temporal': [],
            'text': [],
            'id': [],
            'target': []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            n_total = len(df)
            
            # Skip ID columns
            if col.lower().endswith('_id') or n_unique == n_total:
                feature_types['id'].append(col)
            # Temporal
            elif np.issubdtype(dtype, np.datetime64) or 'time' in col.lower() or 'date' in col.lower():
                feature_types['temporal'].append(col)
            # Categorical (low cardinality)
            elif dtype == 'object' or (np.issubdtype(dtype, np.integer) and n_unique < 50):
                feature_types['categorical'].append(col)
            # Numerical
            elif np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer):
                feature_types['numerical'].append(col)
            else:
                feature_types['categorical'].append(col)
        
        return feature_types
    
    def map_for_lightgcn(self, interactions: pd.DataFrame, 
                         user_features: Optional[pd.DataFrame] = None,
                         item_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Map features for LightGCN retrieval layer."""
        self.logger.info("Mapping features for LightGCN...")
        
        mapped = {
            'interactions': interactions[['user_id', 'item_id', 'rating']].copy(),
            'user_graph_features': None,
            'item_graph_features': None,
            'edge_weights': None
        }
        
        # Add graph features if available
        if user_features is not None:
            # Extract affinity scores, frequency features
            affinity_cols = [c for c in user_features.columns if 'affinity' in c.lower() or 'frequency' in c.lower()]
            if affinity_cols:
                mapped['user_graph_features'] = user_features[['user_id'] + affinity_cols]
        
        if item_features is not None:
            popularity_cols = [c for c in item_features.columns if 'popularity' in c.lower() or 'frequency' in c.lower()]
            if popularity_cols:
                mapped['item_graph_features'] = item_features[['item_id'] + popularity_cols]
        
        # Compute edge weights based on interaction features
        if 'timestamp' in interactions.columns:
            # Recency weighting
            max_time = interactions['timestamp'].max()
            min_time = interactions['timestamp'].min()
            time_range = (max_time - min_time).total_seconds()
            if time_range > 0:
                interactions_copy = interactions.copy()
                interactions_copy['recency_weight'] = 1 - (
                    (max_time - interactions_copy['timestamp']).dt.total_seconds() / time_range
                )
                mapped['edge_weights'] = interactions_copy[['user_id', 'item_id', 'recency_weight']]
        
        self.mapped_features['lightgcn'] = mapped
        self.logger.info(f"Mapped {len(mapped['interactions'])} interactions for LightGCN")
        return mapped
    
    def map_for_sasrec(self, interactions: pd.DataFrame,
                       session_features: Optional[pd.DataFrame] = None,
                       temporal_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Map features for SASRec sequential layer."""
        self.logger.info("Mapping features for SASRec...")
        
        max_seq_length = self.config.get('sequential.max_seq_length', 50)
        
        # Build user sequences
        if 'timestamp' not in interactions.columns:
            raise ValueError("SASRec requires timestamp column for sequence ordering")
        
        # Sort by user and timestamp
        sorted_df = interactions.sort_values(['user_id', 'timestamp'])
        
        # Create sequences
        sequences = {}
        for user_id, group in sorted_df.groupby('user_id'):
            items = group['item_id'].tolist()
            timestamps = group['timestamp'].tolist()
            # Truncate to max length
            if len(items) > max_seq_length:
                items = items[-max_seq_length:]
                timestamps = timestamps[-max_seq_length:]
            sequences[user_id] = {
                'items': items,
                'timestamps': timestamps,
                'length': len(items)
            }
        
        mapped = {
            'sequences': sequences,
            'max_seq_length': max_seq_length,
            'session_features': session_features,
            'temporal_features': temporal_features,
            'n_sequences': len(sequences),
            'avg_seq_length': np.mean([s['length'] for s in sequences.values()])
        }
        
        self.mapped_features['sasrec'] = mapped
        self.logger.info(f"Mapped {len(sequences)} sequences, avg length: {mapped['avg_seq_length']:.2f}")
        return mapped
    
    def map_for_deepfm(self, interactions: pd.DataFrame,
                       user_features: Optional[pd.DataFrame] = None,
                       item_features: Optional[pd.DataFrame] = None,
                       feature_matrix: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Map features for DeepFM ranking layer."""
        self.logger.info("Mapping features for DeepFM...")
        
        # Start with interactions
        base_df = interactions.copy()
        
        dense_features = []
        sparse_features = []
        
        # Merge user features
        if user_features is not None:
            user_id_col = detect_field(user_features, 'user_id')
            if user_id_col:
                merge_col = user_id_col if user_id_col != 'user_id' else 'user_id'
                base_df = base_df.merge(user_features, left_on='user_id', right_on=merge_col, how='left')
        
        # Merge item features  
        if item_features is not None:
            item_id_col = detect_field(item_features, 'item_id')
            if item_id_col:
                merge_col = item_id_col if item_id_col != 'item_id' else 'item_id'
                base_df = base_df.merge(item_features, left_on='item_id', right_on=merge_col, how='left')
        
        # Merge additional feature matrix
        if feature_matrix is not None:
            # Try to merge on user_id, item_id combination
            base_df = base_df.merge(feature_matrix, on=['user_id', 'item_id'], how='left')
        
        # Analyze and categorize features
        feature_types = self.analyze_features(base_df)
        
        # Exclude ID and target columns from features
        exclude_cols = ['user_id', 'item_id', 'rating', 'timestamp', 'session_id']
        exclude_cols.extend(feature_types['id'])
        
        for col in feature_types['numerical']:
            if col not in exclude_cols and col in base_df.columns:
                dense_features.append(col)
        
        for col in feature_types['categorical']:
            if col not in exclude_cols and col in base_df.columns:
                sparse_features.append(col)
        
        # Handle temporal features
        for col in feature_types['temporal']:
            if col in base_df.columns:
                # Extract cyclical features
                base_df[f'{col}_hour'] = base_df[col].dt.hour
                base_df[f'{col}_dayofweek'] = base_df[col].dt.dayofweek
                base_df[f'{col}_month'] = base_df[col].dt.month
                sparse_features.extend([f'{col}_hour', f'{col}_dayofweek', f'{col}_month'])
        
        # Fill missing values
        for col in dense_features:
            if col in base_df.columns:
                base_df[col] = base_df[col].fillna(base_df[col].median())
        
        for col in sparse_features:
            if col in base_df.columns:
                base_df[col] = base_df[col].fillna('unknown')
        
        mapped = {
            'dataframe': base_df,
            'dense_features': dense_features,
            'sparse_features': sparse_features,
            'label_column': 'rating',
            'user_id_column': 'user_id',
            'item_id_column': 'item_id'
        }
        
        self.mapped_features['deepfm'] = mapped
        self.logger.info(f"Mapped {len(dense_features)} dense and {len(sparse_features)} sparse features")
        return mapped
    
    def get_feature_importance_template(self) -> Dict[str, Any]:
        """Generate template for feature importance tracking."""
        return {
            'lightgcn': {
                'graph_features': [],
                'interaction_weights': []
            },
            'sasrec': {
                'position_weights': [],
                'attention_scores': []
            },
            'deepfm': {
                'dense_feature_importance': {},
                'sparse_feature_importance': {},
                'fm_interactions': []
            }
        }
    
    def save_mapped_features(self, output_dir: str) -> str:
        """Save mapped features to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save mapping metadata
        mapping_info = {
            'lightgcn': {
                'n_interactions': len(self.mapped_features.get('lightgcn', {}).get('interactions', [])),
                'has_user_features': self.mapped_features.get('lightgcn', {}).get('user_graph_features') is not None,
                'has_item_features': self.mapped_features.get('lightgcn', {}).get('item_graph_features') is not None
            },
            'sasrec': {
                'n_sequences': self.mapped_features.get('sasrec', {}).get('n_sequences', 0),
                'avg_seq_length': self.mapped_features.get('sasrec', {}).get('avg_seq_length', 0),
                'max_seq_length': self.mapped_features.get('sasrec', {}).get('max_seq_length', 0)
            },
            'deepfm': {
                'n_dense_features': len(self.mapped_features.get('deepfm', {}).get('dense_features', [])),
                'n_sparse_features': len(self.mapped_features.get('deepfm', {}).get('sparse_features', []))
            }
        }
        
        save_json(mapping_info, output_path / 'feature_mapping.json')
        self.logger.info(f"Saved feature mapping to {output_path / 'feature_mapping.json'}")
        
        return str(output_path)

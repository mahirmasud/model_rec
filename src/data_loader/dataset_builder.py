"""
Dataset Builder - Constructs interaction datasets from feature-engineered inputs.
Handles multiple input formats and builds unified interaction matrices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from ..utils.helpers import (
    load_parquet, load_json, detect_field, normalize_timestamps,
    compute_statistics, save_parquet, ensure_dir
)
from ..utils.config_loader import ConfigLoader


class DatasetBuilder:
    """
    Builds recommendation datasets from feature-engineered inputs.
    
    Supports multiple input formats and automatically detects
    relevant fields for user-item interactions.
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        """
        Initialize dataset builder.
        
        Args:
            config: Pipeline configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Field mappings from config
        self.field_mapping = config.get('field_mapping', {})
        
        # Loaded data
        self.interactions_df: Optional[pd.DataFrame] = None
        self.user_features_df: Optional[pd.DataFrame] = None
        self.item_features_df: Optional[pd.DataFrame] = None
        self.session_features_df: Optional[pd.DataFrame] = None
        self.temporal_features_df: Optional[pd.DataFrame] = None
        
        # Detected field names
        self.user_id_col: Optional[str] = None
        self.item_id_col: Optional[str] = None
        self.timestamp_col: Optional[str] = None
        self.interaction_col: Optional[str] = None
        self.session_id_col: Optional[str] = None
    
    def load_data(self, raw_dir: Optional[str] = None) -> 'DatasetBuilder':
        """
        Load all input datasets from raw directory.
        
        Args:
            raw_dir: Override for raw data directory
        
        Returns:
            Self for method chaining
        """
        raw_dir = Path(raw_dir or self.config.get('data.raw_dir', 'data/raw'))
        input_files = self.config.get('data.input_files', {})
        
        self.logger.info(f"Loading data from {raw_dir}")
        
        # Load cleaned dataset (main interactions)
        cleaned_file = raw_dir / input_files.get('cleaned_data', 'cleaned_data.parquet')
        if cleaned_file.exists():
            self.interactions_df = load_parquet(cleaned_file)
            self.logger.info(f"Loaded cleaned dataset: {len(self.interactions_df)} rows")
        else:
            self.logger.warning(f"Cleaned dataset not found: {cleaned_file}")
        
        # Load feature files
        feature_matrix_file = raw_dir / input_files.get('feature_matrix', 'feature_matrix.parquet')
        if feature_matrix_file.exists():
            self.feature_matrix_df = load_parquet(feature_matrix_file)
            self.logger.info(f"Loaded feature matrix: {self.feature_matrix_df.shape}")
        
        # Load user features
        user_features_file = raw_dir / input_files.get('user_features', 'user_features.parquet')
        if user_features_file.exists():
            self.user_features_df = load_parquet(user_features_file)
            self.logger.info(f"Loaded user features: {self.user_features_df.shape}")
        
        # Load item features
        item_features_file = raw_dir / input_files.get('item_features', 'item_features.parquet')
        if item_features_file.exists():
            self.item_features_df = load_parquet(item_features_file)
            self.logger.info(f"Loaded item features: {self.item_features_df.shape}")
        
        # Load session features
        session_features_file = raw_dir / input_files.get('session_features', 'session_features.parquet')
        if session_features_file.exists():
            self.session_features_df = load_parquet(session_features_file)
            self.logger.info(f"Loaded session features: {self.session_features_df.shape}")
        
        # Load temporal features
        temporal_features_file = raw_dir / input_files.get('temporal_features', 'temporal_features.parquet')
        if temporal_features_file.exists():
            self.temporal_features_df = load_parquet(temporal_features_file)
            self.logger.info(f"Loaded temporal features: {self.temporal_features_df.shape}")
        
        return self
    
    def detect_fields(self) -> 'DatasetBuilder':
        """
        Detect standard field names in loaded datasets.
        
        Returns:
            Self for method chaining
        """
        if self.interactions_df is None:
            raise ValueError("No interactions loaded. Call load_data() first.")
        
        df = self.interactions_df
        
        # Detect user ID
        self.user_id_col = detect_field(df, 'user_id', self.field_mapping.get('user_id'))
        self.logger.info(f"Detected user_id column: {self.user_id_col}")
        
        # Detect item ID
        self.item_id_col = detect_field(df, 'item_id', self.field_mapping.get('item_id'))
        self.logger.info(f"Detected item_id column: {self.item_id_col}")
        
        # Detect timestamp
        self.timestamp_col = detect_field(df, 'timestamp', self.field_mapping.get('timestamp'))
        self.logger.info(f"Detected timestamp column: {self.timestamp_col}")
        
        # Detect interaction/rating
        self.interaction_col = detect_field(df, 'interaction', self.field_mapping.get('interaction'))
        self.logger.info(f"Detected interaction column: {self.interaction_col}")
        
        # Detect session ID
        self.session_id_col = detect_field(df, 'session_id', self.field_mapping.get('session_id'))
        
        # Validate required fields
        if not all([self.user_id_col, self.item_id_col]):
            raise ValueError("Could not detect required user_id and item_id columns")
        
        return self
    
    def preprocess_interactions(self) -> 'DatasetBuilder':
        """
        Preprocess interaction data for recommendation models.
        
        - Normalize timestamps
        - Handle missing values
        - Convert implicit feedback
        - Filter sparse users/items
        
        Returns:
            Self for method chaining
        """
        if self.interactions_df is None:
            raise ValueError("No interactions loaded")
        
        df = self.interactions_df.copy()
        
        # Normalize timestamps if present
        if self.timestamp_col:
            df = normalize_timestamps(df, self.timestamp_col)
        
        # Handle interaction column
        if self.interaction_col:
            # Convert to binary implicit feedback if needed
            if df[self.interaction_col].dtype == 'object':
                df[self.interaction_col] = 1
            else:
                # Threshold for implicit feedback
                threshold = self.config.get('retrieval.implicit_threshold', 0)
                df[self.interaction_col] = (df[self.interaction_col] > threshold).astype(int)
        else:
            # Create default interaction column
            df['interaction'] = 1
            self.interaction_col = 'interaction'
        
        # Remove duplicates (keep latest)
        if self.timestamp_col:
            df = df.sort_values(self.timestamp_col)
            df = df.drop_duplicates(subset=[self.user_id_col, self.item_id_col], keep='last')
        else:
            df = df.drop_duplicates(subset=[self.user_id_col, self.item_id_col])
        
        # Filter sparse users/items
        min_user_interactions = self.config.get('retrieval.min_user_interactions', 5)
        min_item_interactions = self.config.get('retrieval.min_item_interactions', 5)
        
        # Count interactions per user/item
        user_counts = df[self.user_id_col].value_counts()
        item_counts = df[self.item_id_col].value_counts()
        
        # Filter
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index
        
        df = df[df[self.user_id_col].isin(valid_users)]
        df = df[df[self.item_id_col].isin(valid_items)]
        
        self.logger.info(f"After filtering: {len(df)} interactions, "
                        f"{df[self.user_id_col].nunique()} users, "
                        f"{df[self.item_id_col].nunique()} items")
        
        self.interactions_df = df
        return self
    
    def build_interaction_matrix(self) -> pd.DataFrame:
        """
        Build standardized interaction DataFrame for RecBole.
        
        Returns:
            Standardized interaction DataFrame
        """
        if self.interactions_df is None:
            raise ValueError("No interactions loaded")
        
        df = self.interactions_df.copy()
        
        # Select and rename columns
        result = pd.DataFrame()
        result['user_id'] = df[self.user_id_col]
        result['item_id'] = df[self.item_id_col]
        
        if self.interaction_col:
            result['rating'] = df[self.interaction_col]
        else:
            result['rating'] = 1
        
        if self.timestamp_col:
            result['timestamp'] = df[self.timestamp_col]
        
        # Add session ID if available
        if self.session_id_col and self.session_id_col in df.columns:
            result['session_id'] = df[self.session_id_col]
        
        # Sort by timestamp if available
        if 'timestamp' in result.columns:
            result = result.sort_values('timestamp')
        
        # Reset index
        result = result.reset_index(drop=True)
        
        self.logger.info(f"Built interaction matrix: {result.shape}")
        return result
    
    def merge_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Merge interaction data with user and item features.
        
        Returns:
            Tuple of (interactions, user_features, item_features) DataFrames
        """
        interactions = self.build_interaction_matrix()
        
        user_features = None
        item_features = None
        
        # Process user features
        if self.user_features_df is not None:
            user_features = self.user_features_df.copy()
            # Ensure user_id column exists
            user_id_col = detect_field(user_features, 'user_id', self.field_mapping.get('user_id'))
            if user_id_col and user_id_col != 'user_id':
                user_features = user_features.rename(columns={user_id_col: 'user_id'})
        
        # Process item features
        if self.item_features_df is not None:
            item_features = self.item_features_df.copy()
            # Ensure item_id column exists
            item_id_col = detect_field(item_features, 'item_id', self.field_mapping.get('item_id'))
            if item_id_col and item_id_col != 'item_id':
                item_features = item_features.rename(columns={item_id_col: 'item_id'})
        
        return interactions, user_features, item_features
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary of statistics
        """
        if self.interactions_df is None:
            return {}
        
        stats = {
            'interactions': compute_statistics(self.interactions_df),
            'sparsity': 1 - len(self.interactions_df) / (
                self.interactions_df[self.user_id_col].nunique() * 
                self.interactions_df[self.item_id_col].nunique()
            ),
            'density': len(self.interactions_df) / (
                self.interactions_df[self.user_id_col].nunique() * 
                self.interactions_df[self.item_id_col].nunique()
            ),
            'avg_interactions_per_user': self.interactions_df.groupby(self.user_id_col).size().mean(),
            'avg_interactions_per_item': self.interactions_df.groupby(self.item_id_col).size().mean(),
            'user_stats': {
                'n_users': self.interactions_df[self.user_id_col].nunique(),
                'min_interactions': self.interactions_df.groupby(self.user_id_col).size().min(),
                'max_interactions': self.interactions_df.groupby(self.user_id_col).size().max(),
            },
            'item_stats': {
                'n_items': self.interactions_df[self.item_id_col].nunique(),
                'min_interactions': self.interactions_df.groupby(self.item_id_col).size().min(),
                'max_interactions': self.interactions_df.groupby(self.item_id_col).size().max(),
            }
        }
        
        if self.timestamp_col:
            stats['time_range'] = {
                'start': str(self.interactions_df[self.timestamp_col].min()),
                'end': str(self.interactions_df[self.timestamp_col].max())
            }
        
        return stats
    
    def save_processed_data(self, output_dir: str) -> str:
        """
        Save processed datasets to output directory.
        
        Args:
            output_dir: Output directory path
        
        Returns:
            Path to saved directory
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        interactions, user_features, item_features = self.merge_features()
        
        # Save interactions
        interactions_path = output_dir / 'interactions.parquet'
        save_parquet(interactions, interactions_path)
        self.logger.info(f"Saved interactions to {interactions_path}")
        
        # Save user features
        if user_features is not None:
            user_features_path = output_dir / 'user_features.parquet'
            save_parquet(user_features, user_features_path)
            self.logger.info(f"Saved user features to {user_features_path}")
        
        # Save item features
        if item_features is not None:
            item_features_path = output_dir / 'item_features.parquet'
            save_parquet(item_features, item_features_path)
            self.logger.info(f"Saved item features to {item_features_path}")
        
        # Save statistics
        stats_path = output_dir / 'dataset_statistics.json'
        from ..utils.helpers import save_json
        save_json(self.get_statistics(), stats_path)
        self.logger.info(f"Saved statistics to {stats_path}")
        
        return str(output_dir)

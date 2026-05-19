"""
RecBole Converter - Converts processed data to RecBole-compatible format.
Generates .inter, .user, .item files for RecBole v1.2.0.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from ..utils.helpers import save_parquet, save_json, ensure_dir, detect_field
from ..utils.config_loader import ConfigLoader


class RecBoleConverter:
    """
    Converts processed datasets to RecBole-compatible formats.
    
    RecBole requires specific file formats:
    - .inter: User-item interactions (required)
    - .user: User features (optional)
    - .item: Item features (optional)
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.dataset_name = "recsys_dataset"
    
    def set_dataset_name(self, name: str) -> 'RecBoleConverter':
        """Set the dataset name for output files."""
        self.dataset_name = name
        return self
    
    def _recbole_typed_name(self, column: str, series: pd.Series) -> str:
        """Create RecBole typed field names preserving arbitrary columns."""
        if column.endswith(":token") or column.endswith(":float") or column.endswith(":token_seq"):
            return column

        if is_datetime64_any_dtype(series):
            return f"{column}:float"
        if is_numeric_dtype(series):
            return f"{column}:float"
        return f"{column}:token"

    def _coerce_series_for_export(self, series: pd.Series) -> pd.Series:
        if is_datetime64_any_dtype(series):
            return (series.astype("int64") // 10**9).astype("float64")
        return series

    def convert_interactions(self, interactions: pd.DataFrame, 
                             output_dir: str) -> Dict[str, Any]:
        """
        Convert interactions to RecBole .inter format.
        
        Args:
            interactions: DataFrame with user_id, item_id, rating, timestamp
            output_dir: Output directory
        
        Returns:
            Dataset configuration dictionary
        """
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        df = interactions.copy()
        
        # Ensure required columns
        required_cols = ['user_id', 'item_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'rating' not in df.columns:
            df['rating'] = 1

        # Convert to RecBole format with typed fields, preserving arbitrary columns
        recbole_df = pd.DataFrame()
        for col in df.columns:
            typed_col = self._recbole_typed_name(col, df[col])
            recbole_df[typed_col] = self._coerce_series_for_export(df[col])
        
        # Save as .inter file (tab-separated)
        inter_file = output_path / f"{self.dataset_name}.inter"
        recbole_df.to_csv(inter_file, sep='\t', index=False)
        self.logger.info(f"Saved RecBole interaction file: {inter_file}")
        
        # Generate dataset config
        dataset_config = {
            'dataset_name': self.dataset_name,
            'data_path': str(output_path),
            'inter_file': f"{self.dataset_name}.inter",
            'fields': {
                'user_id': {'type': 'token', 'sep': '\t'},
                'item_id': {'type': 'token', 'sep': '\t'},
                'rating': {'type': 'float', 'sep': '\t'}
            },
            'n_users': int(df['user_id'].nunique()),
            'n_items': int(df['item_id'].nunique()),
            'n_interactions': len(df),
            'density': float(len(df) / (df['user_id'].nunique() * df['item_id'].nunique()))
        }
        
        if 'timestamp' in df.columns:
            dataset_config['fields']['timestamp'] = {'type': 'float', 'sep': '\t'}
            dataset_config['time_field'] = 'timestamp'

        dataset_config['exported_columns'] = list(recbole_df.columns)
        
        # Save dataset config
        config_file = output_path / f"{self.dataset_name}.yaml"
        self._save_recbole_config(config_file, dataset_config)
        
        return dataset_config
    
    def convert_user_features(self, user_features: pd.DataFrame,
                              output_dir: str) -> Dict[str, Any]:
        """Convert user features to RecBole .user format."""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        df = user_features.copy()
        
        user_id_col = detect_field(df, 'user_id')
        if not user_id_col:
            raise ValueError("user_id column required")
        if user_id_col != 'user_id':
            df = df.rename(columns={user_id_col: 'user_id'})
        
        # Identify feature types
        feature_cols = [c for c in df.columns if c != 'user_id']
        
        # Save as .user file
        user_file = output_path / f"{self.dataset_name}.user"
        df.to_csv(user_file, sep='\t', index=False)
        self.logger.info(f"Saved RecBole user file: {user_file}")
        
        return {
            'user_file': f"{self.dataset_name}.user",
            'n_user_features': len(feature_cols),
            'feature_columns': feature_cols
        }
    
    def convert_item_features(self, item_features: pd.DataFrame,
                              output_dir: str) -> Dict[str, Any]:
        """Convert item features to RecBole .item format."""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        df = item_features.copy()
        
        item_id_col = detect_field(df, 'item_id')
        if not item_id_col:
            raise ValueError("item_id column required")
        if item_id_col != 'item_id':
            df = df.rename(columns={item_id_col: 'item_id'})
        
        feature_cols = [c for c in df.columns if c != 'item_id']
        
        # Save as .item file
        item_file = output_path / f"{self.dataset_name}.item"
        df.to_csv(item_file, sep='\t', index=False)
        self.logger.info(f"Saved RecBole item file: {item_file}")
        
        return {
            'item_file': f"{self.dataset_name}.item",
            'n_item_features': len(feature_cols),
            'feature_columns': feature_cols
        }
    
    def _save_recbole_config(self, path: Path, config: Dict[str, Any]) -> None:
        """Save RecBole dataset configuration."""
        content = f"""# RecBole Dataset Configuration
# Generated by RecSys Pipeline

dataset_name: {config['dataset_name']}
data_path: {config['data_path']}

# Data files
inter_file: {config.get('inter_file', '')}
{f"user_file: {config.get('user_file', '')}" if config.get('user_file') else ''}
{f"item_file: {config.get('item_file', '')}" if config.get('item_file') else ''}

# Dataset statistics
n_users: {config.get('n_users', 0)}
n_items: {config.get('n_items', 0)}
n_interactions: {config.get('n_interactions', 0)}
density: {config.get('density', 0):.6f}

# Field configuration
field_separator: "\\t"
seq_separator: " "

# Core fields
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
{f"TIME_FIELD: {config.get('time_field', 'timestamp')}" if config.get('time_field') else ''}

# Ignore columns
IGNORE_COLUMN: []

# Negative sampling
NEG_PREFIX: neg__

# Sequential settings (for SASRec)
ITEM_LIST_LENGTH: {self.config.get('sequential.max_seq_length', 50)}
MASK_ITEM_NUMBER: 1
"""
        with open(path, 'w') as f:
            f.write(content)
    
    def create_sequence_dataset(self, interactions: pd.DataFrame,
                                output_dir: str) -> Dict[str, Any]:
        """Create sequential dataset for SASRec."""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        max_seq_length = self.config.get('sequential.max_seq_length', 50)
        
        # Sort by user and timestamp
        df = interactions.sort_values(['user_id', 'timestamp'])
        
        # Build sequences
        sequences = []
        for user_id, group in df.groupby('user_id'):
            items = group['item_id'].tolist()
            timestamps = group['timestamp'].astype(np.int64) // 10**9
            
            # Create sliding windows
            for i in range(len(items)):
                if i >= max_seq_length - 1:
                    seq_items = items[i-max_seq_length+1:i+1]
                    seq_timestamps = timestamps[i-max_seq_length+1:i+1].tolist()
                else:
                    seq_items = items[:i+1]
                    seq_timestamps = timestamps[:i+1].tolist()
                
                # Pad if necessary
                if len(seq_items) < max_seq_length:
                    padding = [0] * (max_seq_length - len(seq_items))
                    seq_items = padding + seq_items
                    seq_timestamps = padding + seq_timestamps
                
                sequences.append({
                    'user_id': user_id,
                    'item_seq': ' '.join(map(str, seq_items)),
                    'time_seq': ' '.join(map(str, seq_timestamps)),
                    'target_item': items[i+1] if i+1 < len(items) else items[-1]
                })
        
        seq_df = pd.DataFrame(sequences)
        
        # Save sequence dataset
        seq_file = output_path / f"{self.dataset_name}_seq.inter"
        seq_df.to_csv(seq_file, sep='\t', index=False)
        self.logger.info(f"Saved sequential dataset: {seq_file} ({len(seq_df)} samples)")
        
        return {
            'sequence_file': f"{self.dataset_name}_seq.inter",
            'n_sequences': len(seq_df),
            'max_seq_length': max_seq_length
        }

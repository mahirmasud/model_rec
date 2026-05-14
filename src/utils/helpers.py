"""
Helper utilities for the recommendation system pipeline.
Provides common functions for data handling, file I/O, and preprocessing.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Save DataFrame to Parquet file.
    
    Args:
        df: DataFrame to save
        path: Output file path
        **kwargs: Additional arguments for to_parquet
    """
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, **kwargs)


def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from Parquet file.
    
    Args:
        path: Input file path
        **kwargs: Additional arguments for read_parquet
    
    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(path, **kwargs)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable)
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path], **kwargs) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Input file path
        **kwargs: Additional arguments for json.load
    
    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f, **kwargs)


def detect_field(
    df: pd.DataFrame,
    field_type: str,
    candidates: Optional[List[str]] = None
) -> Optional[str]:
    """
    Detect field name in DataFrame based on common naming conventions.
    
    Args:
        df: DataFrame to search
        field_type: Type of field to detect (user_id, item_id, timestamp, etc.)
        candidates: Optional list of candidate field names
    
    Returns:
        Detected field name or None
    """
    # Default candidates by field type
    default_candidates = {
        'user_id': ['USER_ID', 'user_id', 'uid', 'user', 'customer_id', 'client_id'],
        'item_id': ['ITEM_ID', 'item_id', 'iid', 'item', 'product_id', 'article_id'],
        'timestamp': ['TIMESTAMP', 'timestamp', 'time', 'date', 'created_at', 'event_time'],
        'interaction': ['INTERACTION', 'interaction', 'rating', 'label', 'click', 'purchase', 'score'],
        'session_id': ['SESSION_ID', 'session_id', 'sid', 'session'],
        'event_type': ['EVENT_TYPE', 'event_type', 'event', 'action', 'behavior']
    }
    
    candidates = candidates or default_candidates.get(field_type, [])
    columns_upper = {col.upper(): col for col in df.columns}
    columns_lower = {col.lower(): col for col in df.columns}
    
    for candidate in candidates:
        if candidate.upper() in columns_upper:
            return columns_upper[candidate.upper()]
    
    # Try fuzzy matching
    for col in df.columns:
        col_upper = col.upper()
        col_lower = col.lower()
        for candidate in candidates:
            if candidate.lower() in col_lower or candidate in col_upper:
                return col
    
    return None


def normalize_timestamps(
    df: pd.DataFrame,
    timestamp_col: str,
    output_format: str = '%Y-%m-%d %H:%M:%S'
) -> pd.DataFrame:
    """
    Normalize timestamp column to consistent format.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        output_format: Desired output format
    
    Returns:
        DataFrame with normalized timestamps
    """
    df = df.copy()
    
    # Convert to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Handle different input formats
    if df[timestamp_col].dtype == 'object':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
    
    # Fill NaT with current time or drop
    nat_count = df[timestamp_col].isna().sum()
    if nat_count > 0:
        df.loc[df[timestamp_col].isna(), timestamp_col] = datetime.now()
    
    return df


def split_dataframe(
    df: pd.DataFrame,
    ratios: List[float],
    strategy: str = 'random',
    sort_by: Optional[str] = None,
    group_by: Optional[str] = None,
    random_state: int = 42
) -> List[pd.DataFrame]:
    """
    Split DataFrame into multiple parts based on ratios.
    
    Args:
        df: DataFrame to split
        ratios: List of ratios (should sum to 1.0)
        strategy: Split strategy ('random', 'temporal', 'by_user')
        sort_by: Column to sort by for temporal splitting
        group_by: Column to group by for grouped splitting
        random_state: Random seed for reproducibility
    
    Returns:
        List of split DataFrames
    """
    assert abs(sum(ratios) - 1.0) < 0.01, "Ratios must sum to 1.0"
    n_splits = len(ratios)
    
    if strategy == 'random':
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(df_shuffled)
        splits = []
        start = 0
        for ratio in ratios:
            end = start + int(n * ratio)
            splits.append(df_shuffled.iloc[start:end].reset_index(drop=True))
            start = end
        return splits
    
    elif strategy == 'temporal':
        assert sort_by is not None, "sort_by required for temporal splitting"
        df_sorted = df.sort_values(sort_by).reset_index(drop=True)
        n = len(df_sorted)
        splits = []
        start = 0
        for ratio in ratios:
            end = start + int(n * ratio)
            splits.append(df_sorted.iloc[start:end].reset_index(drop=True))
            start = end
        return splits
    
    elif strategy == 'by_user':
        assert group_by is not None, "group_by required for user-based splitting"
        
        # Get unique groups
        groups = df[group_by].unique()
        np.random.seed(random_state)
        np.random.shuffle(groups)
        
        n_groups = len(groups)
        splits = []
        start = 0
        for ratio in ratios:
            end = start + int(n_groups * ratio)
            group_subset = groups[start:end]
            splits.append(df[df[group_by].isin(group_subset)].reset_index(drop=True))
            start = end
        return splits
    
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def get_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer semantic data types for each column.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary mapping column names to inferred types
    """
    type_mapping = {}
    
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_total = len(df)
        
        if dtype == 'object' or str(dtype) == 'category':
            if n_unique / n_total > 0.5:
                type_mapping[col] = 'text'
            else:
                type_mapping[col] = 'categorical'
        elif np.issubdtype(dtype, np.integer):
            if n_unique <= 10:
                type_mapping[col] = 'categorical'
            elif n_unique / n_total < 0.01:
                type_mapping[col] = 'id'
            else:
                type_mapping[col] = 'numerical'
        elif np.issubdtype(dtype, np.floating):
            type_mapping[col] = 'numerical'
        elif np.issubdtype(dtype, np.datetime64):
            type_mapping[col] = 'datetime'
        else:
            type_mapping[col] = 'unknown'
    
    return type_mapping


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': {},
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].notna().sum()),
            'null_count': int(df[col].isna().sum()),
            'null_percentage': float(df[col].isna().mean() * 100),
            'n_unique': int(df[col].nunique())
        }
        
        if df[col].dtype in [np.int64, np.float64]:
            col_stats['min'] = float(df[col].min()) if not df[col].empty else None
            col_stats['max'] = float(df[col].max()) if not df[col].empty else None
            col_stats['mean'] = float(df[col].mean()) if not df[col].empty else None
            col_stats['std'] = float(df[col].std()) if not df[col].empty else None
        
        stats['columns'][col] = col_stats
    
    return stats


def batch_iterator(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool = False,
    random_state: int = 42
):
    """
    Iterate over DataFrame in batches.
    
    Args:
        df: DataFrame to iterate
        batch_size: Number of rows per batch
        shuffle: Whether to shuffle before iterating
        random_state: Random seed for shuffling
    
    Yields:
        DataFrame batches
    """
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(df))
        yield df.iloc[start:end].reset_index(drop=True)


def safe_divide(numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safe division that handles zero denominators.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value when denominator is zero
    
    Returns:
        Division result with zeros handled
    """
    if isinstance(denominator, np.ndarray):
        result = np.zeros_like(numerator, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        result[~mask] = default
        return result
    else:
        return numerator / denominator if denominator != 0 else default

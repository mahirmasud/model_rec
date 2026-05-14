"""
Configuration loader for the recommendation system pipeline.
Handles YAML configuration loading, validation, and merging.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data path configurations."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    inter_dir: str = "data/inter"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    input_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Base model configuration."""
    model: str = ""
    embedding_dim: int = 64
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 50
    weight_decay: float = 1e-5


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    field_mapping: Dict[str, List[str]] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    sequential: Dict[str, Any] = field(default_factory=dict)
    ranking: Dict[str, Any] = field(default_factory=dict)
    reranking: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    serving: Dict[str, Any] = field(default_factory=dict)
    explainability: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    Configuration loader with support for YAML files, 
    environment variables, and programmatic overrides.
    """
    
    DEFAULT_CONFIG_PATH = "config/pipeline_config.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._loaded = False
    
    def load(self, config_path: Optional[str] = None) -> 'ConfigLoader':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Optional override for config path
        
        Returns:
            Self for method chaining
        """
        path = Path(config_path or self.config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        self._loaded = True
        return self
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> 'ConfigLoader':
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            Self for method chaining
        """
        self._config = config_dict
        self._loaded = True
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path (e.g., "retrieval.embedding_dim")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        if not self._loaded:
            self.load()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> 'ConfigLoader':
        """
        Set configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            self.load()
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        return self
    
    def merge(self, other_config: Union[Dict[str, Any], 'ConfigLoader']) -> 'ConfigLoader':
        """
        Merge another configuration into this one.
        
        Args:
            other_config: Configuration to merge (dict or ConfigLoader)
        
        Returns:
            Self for method chaining
        """
        if isinstance(other_config, ConfigLoader):
            other = other_config._config
        else:
            other = other_config
        
        self._config = self._deep_merge(self._config, other)
        return self
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Required sections
        required_sections = ['data', 'retrieval', 'sequential', 'ranking', 'reranking']
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")
        
        # Validate data paths
        data_config = self._config.get('data', {})
        if not data_config.get('raw_dir'):
            errors.append("Missing data.raw_dir configuration")
        
        # Validate model configurations
        for stage in ['retrieval', 'sequential', 'ranking']:
            stage_config = self._config.get(stage, {})
            if stage_config and 'model' not in stage_config:
                errors.append(f"Missing {stage}.model configuration")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        if not self._loaded:
            self.load()
        return self._config.copy()
    
    def save(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to output YAML file
        """
        if not self._loaded:
            self.load()
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @property
    def data(self) -> DataConfig:
        """Get data configuration as DataConfig object."""
        data_dict = self._config.get('data', {})
        return DataConfig(**data_dict)
    
    @property
    def retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return self._config.get('retrieval', {})
    
    @property
    def sequential_config(self) -> Dict[str, Any]:
        """Get sequential configuration."""
        return self._config.get('sequential', {})
    
    @property
    def ranking_config(self) -> Dict[str, Any]:
        """Get ranking configuration."""
        return self._config.get('ranking', {})
    
    @property
    def reranking_config(self) -> Dict[str, Any]:
        """Get reranking configuration."""
        return self._config.get('reranking', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get('evaluation', {})
    
    def __repr__(self) -> str:
        return f"ConfigLoader(loaded={self._loaded}, path={self.config_path})"
    
    def __str__(self) -> str:
        return yaml.dump(self._config, default_flow_style=False, sort_keys=False)

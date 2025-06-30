"""
Configuration for forest loss detection model training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "data"
    image_size: int = 256
    in_channels: int = 3
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ModelConfig:
    """Model configuration"""
    backbone: str = "efficientnet-b0"
    in_channels: int = 3
    out_channels: int = 1
    features: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    use_attention: bool = True
    use_contrastive: bool = True
    use_consistency: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    segmentation_weight: float = 1.0
    contrastive_weight: float = 0.5
    consistency_weight: float = 0.3

@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "outputs"
    model_save_dir: str = field(default_factory=lambda: str(Path("outputs") / "models"))
    log_dir: str = field(default_factory=lambda: str(Path("outputs") / "logs"))
    plot_dir: str = field(default_factory=lambda: str(Path("outputs") / "plots"))

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        """Create output directories"""
        for dir_path in [self.output.model_save_dir, self.output.log_dir, self.output.plot_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def update(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'output': self.output.__dict__
        } 
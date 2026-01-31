# src/config.py
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    p: int = 100
    k: int = 10
    sparsity: float = 0.1
    lr: float = 1e-2
    max_iter: int = 1000
    seed: int = 42

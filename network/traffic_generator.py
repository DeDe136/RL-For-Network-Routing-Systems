"""
network/traffic_generator.py

Sinh traffic demand (src, dst, volume_mbps) theo từng step.
"""

import numpy as np
from typing import Tuple


class TrafficGenerator:
    NUM_NODES = 8

    def __init__(self, mean_mbps: float = 10.0, seed: int = 42):
        self.mean_mbps = mean_mbps
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Tuple[int, int, float]:
        """Sinh 1 demand: (src, dst, volume_mbps)."""
        src = int(self.rng.integers(0, self.NUM_NODES))
        dst = int(self.rng.integers(0, self.NUM_NODES))
        while dst == src:
            dst = int(self.rng.integers(0, self.NUM_NODES))
        volume = float(max(1.0, self.rng.poisson(self.mean_mbps)))
        return src, dst, volume

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
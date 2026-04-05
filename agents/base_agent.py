from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseAgent(ABC):

    def __init__(self, n_states: int, n_actions: int, config: Dict[str, Any]):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.config    = config
        self.training  = True

    @abstractmethod
    def select_action(self, state: Any) -> int:
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        ...

    def train_mode(self):  self.training = True
    def eval_mode(self):   self.training = False

    def save(self, path: str): pass
    def load(self, path: str): pass
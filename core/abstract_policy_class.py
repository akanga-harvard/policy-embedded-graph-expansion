# Standard library imports
import copy

from abc import ABC, abstractmethod

# Third-party imports
import numpy as np

# Local imports
from core.binary_frontier_environment import BinaryFrontierEnv

class AbstractPolicyClass(ABC):
    def __init__(self, env: BinaryFrontierEnv, instance_hash: str) -> None:
        self.instance_hash = instance_hash
        self.env = copy.deepcopy(env)
        self.n = self.env.n
        self.discount_factor = self.env.discount_factor
        self.env.reset()
        self._setup_policy()
        self.train_time = 0
        self._train_policy()
    
    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @abstractmethod
    def _setup_policy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_policy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
        raise NotImplementedError
    
    def act(self, status: np.ndarray) -> int:
        valid_actions = self.env.get_frontier_actions(status)
        action = self._select_action(status, valid_actions)
        assert action in valid_actions
        return action
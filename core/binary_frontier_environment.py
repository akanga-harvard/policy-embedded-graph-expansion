# Standard library imports
import copy
from typing import Optional

# Third-party imports
import networkx as nx
import numpy as np

# Local imports
from core.abstract_joint_probability_class import AbstractJointProbabilityClass

class BinaryFrontierEnv:
    def __init__(
            self,
            G: nx.Graph,
            P: AbstractJointProbabilityClass,
            discount_factor: float,
            cc_dict: Optional[dict] = None,
            cc_root: Optional[list] = None,
            rng_seed: int = 314159
        ) -> None:
        assert 0 < discount_factor and discount_factor < 1
        self.P = copy.deepcopy(P)
        self.n = G.number_of_nodes()
        assert self.n == P.n
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(rng_seed)
        self.G = nx.relabel_nodes(G, {i: f"X{i}" for i in range(self.n)})
        self.tests_done = 0
        self.status = np.array([-1] * self.n)

        # Pre-process root of each connected component for frontier testing
        if cc_dict is not None and cc_root is not None:
            self.cc_dict = cc_dict
            self.cc_root = cc_root
        else:
            self.cc_dict = dict()
            self.cc_root = []
            # print("Preprocessing frontier...")
            for cc_nodes in nx.connected_components(self.G):
                self.cc_dict[frozenset(cc_nodes)] = len(self.cc_dict)
                indices = [int(v[1:]) for v in cc_nodes]
                marginal_prob1s = [(self.get_marginal_prob1(idx), idx) for idx in indices]
                self.cc_root.append(sorted(marginal_prob1s, reverse=True)[0][1])

    """
    Only DQN policy uses the output of reset
    """
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.tests_done = 0
        self.status = np.array([-1] * self.n)
        valid_actions = self.get_frontier_actions(self.status)
        valid_actions_array = np.array([1 if idx in valid_actions else 0 for idx in range(self.n)])
        return self.status.copy(), valid_actions_array

    """
    Used only by DQN policy
    """
    def get_status_and_factors(self) -> tuple[np.ndarray, dict, dict]:
        return self.status.copy(), self.P.unary_factors.copy(), self.P.pairwise_factors.copy()

    """
    Used only by DQN policy
    """
    def step(self, action_idx: int) -> tuple[np.ndarray, np.ndarray, float, bool]:
        assert isinstance(action_idx, int)
        assert action_idx in self.get_frontier_actions(self.status)

        # Compute log Pr[action_index = 1 | current statuses]
        p1 = self.get_marginal_prob1(action_idx)

        # Flip coin to realize X_action
        self.status[action_idx] = 1 if self.rng.random() <= p1 else 0

        # Update and output
        reward = self.status[action_idx]
        self.tests_done += 1
        valid_actions = self.get_frontier_actions(self.status)
        valid_actions_array = np.array([1 if idx in valid_actions else 0 for idx in range(self.n)])
        done = self.tests_done == self.n
        return self.status.copy(), valid_actions_array, reward, done
    
    """
    Computes log Pr(X_index = 1| observed statuses)
    """
    def get_marginal_prob1(self, index: int, observed_status: Optional[np.ndarray] = None) -> float:
        status = self.status if observed_status is None else observed_status
        if status[index] == 1:
            return 1.0
        elif status[index] == 0:
            return 0.0
        else:
            query_dict = {f"X{index}": 1}
            observation_dict = {f"X{idx}": status[idx] for idx in range(self.n) if status[idx] != -1}
            return self.compute_conditional_probability(query_dict, observation_dict)
    
    """
    Computes probability of given (partial) realization given observations
    Example format of dictionaries:
    - query_dict = {"X1": 1, "X3": 1}
    - observation_dict = {"X2": 0, "X4": 1}
    """
    def compute_conditional_probability(self, query_dict: dict, observation_dict: dict) -> float:
        assert len(set(query_dict.keys()).intersection(observation_dict.keys())) == 0
        return self.P.compute_conditional_probability(query_dict, observation_dict)
    
    """
    Returns set of frontier indices
    """
    def get_frontier_actions(self, status: np.ndarray) -> set[int]:
        assert len(status) == self.n
        tested = set([f"X{i}" for i in range(self.n) if status[i] != -1])
        frontier = set()
        for cc_nodes in nx.connected_components(self.G):
            if len(cc_nodes.intersection(tested)) == 0:
                # Add index with argmax marginal probability (pre-computed root) into frontier
                argmax_in_cc = self.cc_root[self.cc_dict[frozenset(cc_nodes)]]
                frontier.add(argmax_in_cc)
            else:
                # Add testing frontier
                for v in cc_nodes:
                    if v not in tested and len(set(self.G.neighbors(v)).intersection(tested)) > 0:
                        frontier.add(int(v[1:]))
        return frontier
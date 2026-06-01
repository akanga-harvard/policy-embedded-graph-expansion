from core.log_junction_tree import LogJunctionTree
import numpy as np
from core.abstract_joint_probability_class import AbstractJointProbabilityClass
import networkx as nx
from gittins_policy import GittinsPolicy
import time


def _form_distribution(full_G, full_covariates, theta: tuple[np.ndarray, np.ndarray]) -> AbstractJointProbabilityClass:
    theta_unary, theta_pairwise = theta
    args = {
        'G': full_G,
        'covariates': full_covariates,
        'theta_unary': theta_unary,
        'theta_pairwise': theta_pairwise
    }
    return LogJunctionTree([f"X{idx}" for idx in full_G.nodes()], args)

def _suggest_next_testing_index(G: nx.Graph, roots, incremental_statuses, P: AbstractJointProbabilityClass, discount_factor) -> int:
    # Run Gittins and act once
    s = time.time()
    gittins = GittinsPolicy(G, set(roots), P, discount_factor)
    e = time.time()
    # print(f'Gittins index policy took {e - s} seconds to run.')
    status = np.array([-1] * G.number_of_nodes())
    for idx in incremental_statuses.keys():
        status[idx] = incremental_statuses[idx]
    return gittins.suggest_next_index(status)
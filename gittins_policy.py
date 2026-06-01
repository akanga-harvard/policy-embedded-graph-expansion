# Standard library imports

# Third-party imports
import networkx as nx
import numpy as np

# Local imports
from core.linear_function import LinearFunction
from core.piecewise_linear_function import PiecewiseLinearFunction
from core.abstract_joint_probability_class import AbstractJointProbabilityClass

class GittinsPolicy:
    def __init__(self, G: nx.Graph, roots: set[int], P: AbstractJointProbabilityClass, discount_factor = 0.99) -> None:
        self.G = G
        self.roots = roots
        self.P = P
        self.n = self.G.number_of_nodes()
        # print(self.P.n, self.G.number_of_nodes())
        assert self.P.n == self.n
        
        # Setup Gittins indices using beta = 0.99
        self.discount_factor = discount_factor
        self._setup_policy()
        self._train_policy()

    def suggest_next_index(self, status: np.ndarray) -> int:
        tested = set([i for i in range(self.n) if status[i] != -1])

        # Frontier collects untested nodes that are either root or has tested parent
        frontier = set([
            node
            for node in range(self.n)
            if node not in tested and (node in self.roots or self.node_to_parent[node] in tested)
        ])

        index_to_test = self._select_action(status, frontier)
        return index_to_test

    def _setup_policy(self) -> None:
        self.memo_Phi = dict()
        self.memo_phi = dict()
        self.gittins_score = dict()

        # Pre-process: BFS projection for each connected component
        self.rooted_forest = []
        self.node_to_forest_idx = dict()
        self.node_to_parent = {v: -1 for v in range(self.n)}
        self.node_to_children = {v: [] for v in range(self.n)}
        for cc_nodes in nx.connected_components(self.G):
            # Consider current connected component
            cc = self.G.subgraph(cc_nodes)
            forest_idx = len(self.rooted_forest)
            overlap = self.roots.intersection(cc_nodes)
            assert len(overlap) == 1
            root_node = overlap.pop()

            # BFS projection
            rooted_tree = nx.bfs_tree(cc, root_node)
            assert nx.is_tree(rooted_tree)

            # Store information for computation of Gittins score
            forest_idx = len(self.rooted_forest)
            for parent, child in rooted_tree.edges():
                self.node_to_forest_idx[parent] = forest_idx
                self.node_to_forest_idx[child] = forest_idx
                self.node_to_parent[child] = parent
                self.node_to_children[parent].append(child)
            self.rooted_forest.append((root_node, rooted_tree))

    def _train_policy(self) -> None:
        for X in range(self.n):
            if self.node_to_parent[X] == -1:
                # Root node
                self.gittins_score[tuple([X, -1])] = self._compute_gittins_score(X, -1)
            else:
                # Non-root node
                self.gittins_score[tuple([X, 0])] = self._compute_gittins_score(X, 0)
                self.gittins_score[tuple([X, 1])] = self._compute_gittins_score(X, 1)

    def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
        # Gather Gittins scores
        score_node_list = []
        for node in valid_actions:
            parent_node = self.node_to_parent[node]
            if parent_node == -1:
                key = tuple([node, -1])
            else:
                parent_bit = status[parent_node]
                key = tuple([node, parent_bit])

            # NOTE: For non-tree inputs, MST parent may not have been tested yet
            if key in self.gittins_score.keys():
                score = self.gittins_score[key]
                score_node_list.append((score, node))

        # Return Gittins scores
        return sorted(score_node_list, reverse=True)

    def _compute_gittins_score(self, X: int, b: int) -> float:
        assert 0 <= X and X <= self.n
        assert b == -1 or b == 0 or b == 1
        phi = self._build_phi(X, b)
        score = phi.compute_fixed_point()
        return score

    def _build_phi(self, idx: int, b: int = -1) -> PiecewiseLinearFunction:
        key = tuple([idx, b])
        if key not in self.memo_phi.keys():
            query_dict = {f"X{idx}": 1}
            observation_dict = {}
            if b != -1:
                parent_node = self.node_to_parent[idx]
                observation_dict[f"X{parent_node}"] = b

            # phi = max(x, RHS), where RHS = p0 * (r(0) + beta * Phi0) + p1 * (r(1) + beta * Phi1)
            # Immediate rewards are r(0) = 0 and r(1) = 1
            p1 = self.P.compute_conditional_probability(query_dict, observation_dict)
            p0 = 1 - p1
            beta = self.discount_factor
            children = self.node_to_children[idx]
            Phi0 = self._build_Phi(children, 0)
            Phi1 = self._build_Phi(children, 1)
            RHS = Phi0.mult_by_const(beta).add_const(0).mult_by_const(p0) + Phi1.mult_by_const(beta).add_const(1).mult_by_const(p1)
            phi = RHS.max_with_linear()
            phi.clean()
            self.memo_phi[key] = phi
            self.memo_Phi[tuple([frozenset(children), 0])] = Phi0
            self.memo_Phi[tuple([frozenset(children), 1])] = Phi1

            startpoint_val = RHS.value_functions[0](0)
            endpoint_val = 1/(1 - beta)
            phi._validate(startpoint_val, endpoint_val)
        return self.memo_phi[key]

    """
    For any fixed x, the product of derivatives is a constant value between 0 and 1/(1-beta)
    Since each derivative is a piecewise constant function, their product is also a piecewise constant function
    Denote this by c(k), for 0 <= k <= 1/(1-beta)

    Then, Phi(x) = h(x) + Phi(0),
    where h(x) = int_{0}^x c(k) dk is a piecewise linear function and Phi(0) = 1 - h(1/(1-beta))
    """
    def _build_Phi(self, S: list[int], b: int) -> PiecewiseLinearFunction:
        assert b == 0 or b == 1
        beta = self.discount_factor
        if len(S) == 0:
            # Linear 1 * x + 0 function
            return PiecewiseLinearFunction([1/(1 - beta)], [LinearFunction(1.0, 0.0)])
        else:
            # Initialize as constant-1 function, then compute c(k) function
            product_of_derivatives = PiecewiseLinearFunction([1/(1 - beta)], [LinearFunction(0.0, 1.0)])
            for X in S:
                phi = self._build_phi(X, b)
                product_of_derivatives *= phi.derivative()

            # Create piecewise linear function h(x) by integrating product_of_derivatives    
            h = product_of_derivatives.integrate_piecewise_constant()

            # Compute Phi(x) = h(x) + Phi(0), where h(1/(1-beta)) = 1/(1 - beta) - Phi(0)
            Phi = h.add_const(1/(1 - beta) - h.value_functions[-1](1/(1 - beta)))
            assert np.isclose(h.value_functions[-1](1/(1 - beta)), 1/(1 - beta) - Phi.value_functions[0](0))
            return Phi
    
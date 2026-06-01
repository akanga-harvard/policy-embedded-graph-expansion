# Standard library imports
import itertools

# Third-party imports
import networkx as nx
from networkx.algorithms.chordal import complete_to_chordal_graph
import numpy as np
from scipy.special import logsumexp

# Local imports
from core.abstract_joint_probability_class import AbstractJointProbabilityClass

class LogJunctionTree(AbstractJointProbabilityClass):
    def __init__(self, variables: list[str], args: dict) -> None:
        super().__init__(variables, args)

    def _setup(self) -> None:
        G = self.args['G']
        covariates = self.args['covariates']
        theta_unary = self.args['theta_unary']
        theta_pairwise = self.args['theta_pairwise']
        
        assert G.number_of_nodes() == self.n
        assert len(covariates) == self.n
        covariate_length = len(covariates[0])
        assert len(theta_unary) == self.compute_theta_length(covariate_length, 1)
        assert len(theta_pairwise) == self.compute_theta_length(covariate_length, 2)

        """
        domains:         dict var→cardinality
        unary_factors:   { var: 1D array of size domains[var] }
        pairwise_factors:{ (u,v): 2D array of shape (dom[u], dom[v]) }
        """
        X = [f"X{i}" for i in range(self.n)]
        self.domains = {v: 2 for v in self.variables}
        self.factors = []

        # Add unary factors
        for i in range(self.n):
            c_i = covariates[i]
            log_phi_0 = theta_unary @ self.f_unary(0, c_i)
            log_phi_1 = theta_unary @ self.f_unary(1, c_i)
            self.unary_factors[X[i]] = np.array([log_phi_0, log_phi_1])
            self.factors.append(LogFactor([f"X{i}"], self.unary_factors[X[i]], is_log=True))

        # Add pairwise factors
        for i in range(self.n):
            c_i = covariates[i]
            for j in G.neighbors(i):
                if i < j:
                    c_j = covariates[j]
                    log_phi_00 = theta_pairwise @ self.f_pairwise(0, 0, c_i, c_j)
                    log_phi_01 = theta_pairwise @ self.f_pairwise(0, 1, c_i, c_j)
                    log_phi_10 = theta_pairwise @ self.f_pairwise(1, 0, c_i, c_j)
                    log_phi_11 = theta_pairwise @ self.f_pairwise(1, 1, c_i, c_j)
                    key = frozenset([X[i], X[j]])
                    self.pairwise_factors[key] = np.array(
                        [log_phi_00, log_phi_01, log_phi_10, log_phi_11]
                    ).reshape(self.domains[f"X{i}"], self.domains[f"X{j}"])
                    self.factors.append(LogFactor([f"X{i}", f"X{j}"], self.pairwise_factors[key], is_log=True))
        
        # Precompute elimination order by min-fill
        # Build adjacency among variables
        adj = {v:set() for v in self.variables}
        for (u, v) in self.pairwise_factors:
            adj[u].add(v)
            adj[v].add(u)
        # Min-fill heuristic
        elim_order = []
        adj_copy = {v:set(neis) for v, neis in adj.items()}
        while adj_copy:
            # pick v minimizing fill-in edges
            def fillin_cost(x):
                nbrs = adj_copy[x]
                return sum(1 for a,b in itertools.combinations(nbrs,2) if b not in adj_copy[a])
            v = min(adj_copy, key=lambda x: (fillin_cost(x), len(adj_copy[x])))
            elim_order.append(v)
            nbrs = adj_copy[v]
            # fill in clique among neighbors
            for a, b in itertools.combinations(nbrs, 2):
                adj_copy[a].add(b)
                adj_copy[b].add(a)
            # remove v
            for nbr in nbrs:
                adj_copy[nbr].remove(v)
            del adj_copy[v]
        self.elim_order = elim_order
        
    
    def compute_conditional_probability(self, query_dict: dict[str,int], evidence_dict: dict[str,int]) -> float:
        query_vars = list(query_dict.keys())
        query_vals = [query_dict[v] for v in query_vars]
        evidence_vars = list(evidence_dict.keys())
        evidence_vals = [evidence_dict[v] for v in evidence_vars]
        output = self.query(query_vars, query_vals, evidence_vars, evidence_vals)
        return output

    def query(self, query_vars, query_vals, evidence_vars, evidence_vals):
        """
        Return P(query_vars = query_vals | evidence_vars = evidence_vals).
        - query_vars, evidence_vars: lists of variable names
        - query_vals, evidence_vals: lists of the same length of integer assignments
        """
        # 1) Incorporate evidence
        factors = []
        for f in self.factors:
            f_red = f
            for v, val in zip(evidence_vars, evidence_vals):
                f_red = f_red.reduce(v, val)
            factors.append(f_red)

        # 2) Eliminate all other variables in elim_order
        to_eliminate = [
            v for v in self.elim_order
            if v not in query_vars and v not in evidence_vars
        ]
        for v in to_eliminate:
            # Gather factors involving v
            related = [f for f in factors if v in f.vars]
            if not related:
                continue
            # Multiply them and then marginalize out v
            f_prod = related[0]
            for f in related[1:]:
                f_prod = f_prod.multiply(f, self.domains)
            f_marg = f_prod.marginalize(v, self.domains)
            # Replace old factors
            factors = [f for f in factors if f not in related] + [f_marg]

        # 3) Multiply remaining factors → joint over query_vars
        joint = factors[0]
        for f in factors[1:]:
            joint = joint.multiply(f, self.domains)

        # 4) Extract the single log‐numerator entry
        #    Build index in the exact order of joint.vars
        idx = tuple(
            query_vals[query_vars.index(v)]
            for v in joint.vars
        )
        log_num = joint.table[idx]

        # 5) Compute log‐denominator over ALL entries
        log_den = logsumexp(joint.table, axis=None)  

        # 6) Exponentiate & return as Python float
        prob = np.exp(log_num - log_den)
        return prob.item()

class LogFactor:
    """
    A factor in log-domain over discrete variables.
    vars  : list of variable names (e.g. ['A','B'])
    table : numpy array of shape (card(A), card(B), …) already in log-space
    """
    def __init__(self, vars, table, is_log=False):
        self.vars = list(vars)
        # arr = np.array(table, copy=False)
        arr = np.array(table)
        self.table = arr if is_log else np.log(arr)

    def reduce(self, var, val):
        """Condition on var=val, slicing out that dimension."""
        if var not in self.vars:
            return self
        i = self.vars.index(var)
        slicer = [slice(None)] * len(self.vars)
        slicer[i] = val
        new_tbl = self.table[tuple(slicer)]
        new_vars = self.vars[:i] + self.vars[i+1:]
        return LogFactor(new_vars, new_tbl, is_log=True)

    def multiply(self, other, domains):
        """
        Multiply two log-factors → add their tables over a broadcast grid.
        domains: dict var→cardinality
        """

        def var_key(v):
            return int(v[1:])
        new_vars = sorted(set(self.vars + other.vars), key = var_key)
        # prepare shapes for broadcasting
        def reshape_for(f):
            shape = [(domains[v] if v in f.vars else 1) for v in new_vars]
            return f.table.reshape(shape)

        A = reshape_for(self)
        B = reshape_for(other)
        return LogFactor(new_vars, A + B, is_log=True)

    def marginalize(self, var, domains):
        """Sum-out (log-sum-exp) variable var."""
        if var not in self.vars:
            return self
        i = self.vars.index(var)
        new_tbl = logsumexp(self.table, axis=i)
        new_vars = self.vars[:i] + self.vars[i+1:]
        return LogFactor(new_vars, new_tbl, is_log=True)

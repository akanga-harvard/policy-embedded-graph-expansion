# External imports
import networkx as nx
import random
random.seed(1234)
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import numpy as np
from matplotlib import pyplot as plt
import copy
import torch
import os
from multiprocessing import Pool
import scipy
import hashlib

# Local imports
from core.abstract_joint_probability_class import AbstractJointProbabilityClass
import diffusion_model as dm
import gittins_no_statuses
from core.binary_frontier_environment import BinaryFrontierEnv
from dqn_policy import DQNPolicy



def _stable_seed(*parts) -> int:
	seed_text = ":".join(str(part) for part in parts)
	digest = hashlib.blake2b(seed_text.encode("utf-8"), digest_size=8).digest()
	return int.from_bytes(digest, byteorder="big") % (2**32)


class Agent:
	def __init__(self, policy: str, env, k=-1, d = -1, aggregation='mean') -> None:
		self.policy = policy
		
		# self.k, self.d, self.aggregation are only used in PEGE + DDB (Ours)
		self.k = k
		self.d = d
		self.aggregation = aggregation
		if self.policy in ['Greedy Classifier', 'DQN', 'PEGE + DDB (Ours)', 'PEGE + DDB (Full Theta)']:
			self.train_models(env)

	def get_action(self, env) -> int:
		# Call appropriate policy function
		if self.policy == 'Random':
			return self.random_policy(env)
		elif self.policy == 'Greedy Neighbor':
			return self.greedy_neighbor(env)
		elif self.policy == 'Greedy Classifier':
			return self.greedy_classifier(env)
		elif self.policy == 'DQN':
			return self.dqn_action(env)
		elif self.policy == 'PEGE + DDB (Ours)':
			return self.pege(env, mode = 'pege')
		elif self.policy == 'PEGE + DDB (Full Theta)':
			return self.pege(env, mode = 'pege')
		elif self.policy == 'Naive Gittins':
			return self.pege(env, mode = 'naive')
		elif self.policy == 'Fully Observable Gittins':
			return self.pege(env, mode = 'full')
		else:
			exit('Specified policy not in set of valid policies')

	def train_models(self, env) -> None:
		# Individual risk score classifier
		if self.policy == 'Greedy Classifier':
			X = np.asarray([env.true_covariates[node] for node in list(env.train_G.nodes())])
			y = np.asarray([env.true_statuses[node] for node in list(env.train_G.nodes())])
			self.clf = CalibratedClassifierCV(estimator=svm.LinearSVC(class_weight="balanced"))
			self.clf.fit(X, y)

		# Number of branches/child nodes predictor
		if self.policy == 'PEGE + DDB (Ours)' or self.policy == 'PEGE + DDB (Full Theta)':
			kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-6, 0.1))
			self.branch_regression = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state = 314)
			X = np.array([list(env.true_covariates[node]) for node in list(env.train_G.nodes())])
			y = np.array([len(list(env.overall_bfs_DG.successors(node))) for node in list(env.train_G.nodes())])
			self.branch_regression.fit(X, y)
			if not os.path.exists(f'checkpoints/diffusion_model/diffusion_model_split_{env.current_test_split}.pth'):
				dm.train_new_model(env)

		# Learn DQN
		if self.policy == 'DQN':
			train_cov = {}
			train_statuses = {}
			self.dqn_mapping = {}
			for i in range(env.test_G.number_of_nodes()):
				original_node = list(env.test_G.nodes())[i]
				self.dqn_mapping[original_node] = i
				train_cov[i] = env.true_covariates[original_node]
				train_statuses[i] = env.true_statuses[original_node]

			mapped_G = nx.relabel_nodes(env.test_G, self.dqn_mapping)
			P = gittins_no_statuses._form_distribution(mapped_G, train_cov, env.theta)
			cc_dict = {}
			cc_root = []
			relabeled_X_G = nx.relabel_nodes(mapped_G, {i: f"X{i}" for i in range(mapped_G.number_of_nodes())})
			for cc_nodes in nx.connected_components(relabeled_X_G):
				cc_dict[frozenset(cc_nodes)] = len(cc_dict)
				for root in env.test_roots:
					if f'X{self.dqn_mapping[root]}' in cc_nodes:
						cc_root.append(self.dqn_mapping[root])
			bfe = BinaryFrontierEnv(mapped_G, P, env.discount_factor, cc_dict = cc_dict, cc_root=cc_root)
			self.dqn = DQNPolicy(bfe, f'split_{env.current_test_split}')

	def random_policy(self, env) -> int:
		# Select random person to test
		return random.choice(self.get_eligible_nodes(env))

	def greedy_neighbor(self, env) -> int:
		# Look for a node with a parent that tested positive
		nodes = self.get_eligible_nodes(env)
		for node in nodes:
			predecessor = list(env.incremental_G.neighbors(node))
			if len(predecessor) > 0 and env.incremental_statuses[predecessor[0]] == 1:
				return node

		# If nobody has a parent that tested positive, return a random node
		return random.choice(nodes)

	def greedy_classifier(self, env) -> int:
		# Make predictions about nodes and take the node that is most likely to be positive
		nodes = self.get_eligible_nodes(env)
		X = np.asarray([env.incremental_covariates[node] for node in nodes])
		probabilities = self.clf.predict_proba(X)
		return nodes[np.argmax(probabilities[:, 1])]

	def fully_observable_solution(self, env, full_G, roots, statuses, covariates) -> int:
		# map to 0, ..., n
		mapped_statuses = {}
		mapped_covariates = {}
		mapping = {}
		mapped_roots = []
		for i in range(full_G.number_of_nodes()):
			original_node = list(full_G.nodes())[i]
			mapping[original_node] = i
			if original_node in statuses:
				mapped_statuses[i] = statuses[original_node]
			if original_node in roots:
				mapped_roots.append(i)
			mapped_covariates[i] = covariates[original_node]

		mapped_G = nx.relabel_nodes(full_G, mapping)

		# get gittins indexes for each node assuming full_G and covariates are correct
		P = gittins_no_statuses._form_distribution(mapped_G, mapped_covariates, env.theta)
		scores = gittins_no_statuses._suggest_next_testing_index(mapped_G, mapped_roots, mapped_statuses, P, env.discount_factor)
		scores_original = {}
		for score in scores:
			for k, v in mapping.items():
				if v == score[1]:
					scores_original[k] = score[0]
		return scores_original

	def simulate_graph(self, depth_limit: int, env, mode = 'pege', simulation_seed: int = None) -> list:
		simulation_seed = 0 if simulation_seed is None else simulation_seed
		simulation_rng = np.random.default_rng(simulation_seed)
		sampling_counter = 0

		# If fully observable, copy whole graph and covariates
		if mode == 'full':
			synthetic_G = copy.deepcopy(env.test_G)
			synthetic_covariates = {node: env.true_covariates[node] for node in list(env.test_G.nodes())}
		# Otherwise, start with just observable portion of the graph
		else:
			synthetic_G = copy.deepcopy(env.incremental_G)
			synthetic_covariates = copy.deepcopy(env.incremental_covariates)
		synthetic_roots = copy.deepcopy(env.test_roots)
		synthetic_frontier = list(self.get_eligible_nodes(env))
		synthetic_statuses = copy.deepcopy(env.incremental_statuses)

		# Build out graph (DDB)
		depth = 0
		frontier = synthetic_frontier
		while depth < depth_limit and len(frontier) > 0 and mode not in ['full', 'naive']:
			new_frontier = []
			for node in frontier:
				regression_X = np.array([synthetic_covariates[node]])
				prediction_means, prediction_stds = self.branch_regression.predict(regression_X, return_std = True)
				neighbors_predicted = int(np.rint(simulation_rng.normal(loc=prediction_means[0], scale=prediction_stds[0])))
				neighbors_predicted = min(neighbors_predicted, 25)

				if neighbors_predicted > 0:
					conditions = np.array([synthetic_covariates[node]]*neighbors_predicted)
					sampling_seed = _stable_seed(simulation_seed, "diffusion", sampling_counter)
					sampling_counter += 1
					diffusion_samples = dm.sampling(model_path = f"checkpoints/diffusion_model/diffusion_model_split_{env.current_test_split}.pth", 
							conditions = torch.tensor(conditions),
							diffusion_steps = 100,
							min_beta = 1e-4,
							max_beta = 0.02,
							seed = sampling_seed)
				for i in range(neighbors_predicted):
					neighbor = np.max(list(synthetic_G.nodes())) + 1
					synthetic_G.add_node(neighbor)
					synthetic_G.add_edge(node, neighbor)
					synthetic_covariates[neighbor] = diffusion_samples[i]
					new_frontier.append(neighbor)

			depth += 1
			frontier = new_frontier

		# Sanity checks
		assert nx.is_forest(synthetic_G)
		assert synthetic_roots == env.test_roots
		assert synthetic_statuses == env.incremental_statuses
		if mode == 'full':
			assert nx.utils.graphs_equal(synthetic_G, env.test_G)
			for k, v in synthetic_covariates.items():
				assert k in env.true_covariates.keys()
				assert np.all(np.array(v) == np.array(env.true_covariates[k]))

		# Get gittins indeces for each frontier node
		return self.fully_observable_solution(env, synthetic_G, synthetic_roots, synthetic_statuses, synthetic_covariates)

	def pege(self, env, mode = 'pege') -> int:
		# Run parallel graph expansions + node evaluations
		all_gittins_indeces = {}

		if mode == 'pege':
			num_simulations = self.k
		else:
			num_simulations = 1
		if mode == 'naive':
			depth_limit = 0
		else:
			depth_limit = self.d

		if num_simulations > 1:
			simulation_seeds = [
				_stable_seed("pege", env.current_test_split, len(env.incremental_statuses), i)
				for i in range(num_simulations)
			]
			with Pool() as p:
				gittins_arrs = p.starmap(
					self.simulate_graph,
					[(depth_limit, env, mode, simulation_seed) for simulation_seed in simulation_seeds]
				)
		else:

			gittins_arrs = [self.simulate_graph(depth_limit, env, mode)]
		
		for gittins_arr in gittins_arrs:
			for k, v in gittins_arr.items():
				if k in all_gittins_indeces:
					all_gittins_indeces[k].append(v)
				else:
					all_gittins_indeces[k] = [v]

		# Return action with best aggregated gittins index value
		max_gittins_index = -np.inf
		max_action = None
		if self.aggregation == 'mode':
			keys = list(all_gittins_indeces.keys())  # preserves insertion order
			mat = np.asarray([np.array(all_gittins_indeces[k_]) for k_ in keys], dtype=float)
			winners_row_idx = mat.argmax(axis=0)          # ties -> first row
			winners_keys = [keys[i] for i in winners_row_idx]
			if len(env.incremental_statuses) % 10 == 0:
				print(f'returned an action ({len(env.incremental_statuses)}/{env.test_G.number_of_nodes()} nodes tested)')
			return scipy.stats.mode(winners_keys).mode

		for k, v in all_gittins_indeces.items():
			if self.aggregation == 'mean':
				v_val = np.mean(v)
			elif self.aggregation == 'mean+var':
				v_val = np.mean(v) + np.var(v)
			elif self.aggregation == 'mean-var':
				v_val = np.mean(v) - np.var(v)

			if v_val > max_gittins_index:
				max_gittins_index = v_val
				max_action = k

		if len(env.incremental_statuses) % 10 == 0:
			print(f'returned an action ({len(env.incremental_statuses)}/{env.test_G.number_of_nodes()} nodes tested)')
		return max_action


	def get_eligible_nodes(self, env) -> list:
		# Get all nodes in the tree that have not been tested yet (i.e., the frontier)
		all_nodes = list(env.incremental_G.nodes())
		eligible_nodes = []
		for node in all_nodes:
			if node not in env.incremental_statuses:
				eligible_nodes.append(node)
		return eligible_nodes

	def dqn_action(self, env) -> list:
		# Return frontier node with highest predicted Q-value
		mapped_frontier = {self.dqn_mapping[node] for node in self.get_eligible_nodes(env)}
		mapped_statuses = np.array([-1] * env.test_G.number_of_nodes())
		for key, val in env.incremental_statuses.items():
			mapped_statuses[self.dqn_mapping[key]] = val
		action = self.dqn._select_action(mapped_statuses, mapped_frontier)
		for k, v in self.dqn_mapping.items():
			if v == action:
				return k
		

	



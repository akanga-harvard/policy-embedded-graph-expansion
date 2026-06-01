# External imports
import networkx as nx
from matplotlib import pyplot as plt
import random
random.seed(1234)
import copy
import algorithms
import pickle
from scipy import interpolate
import matplotlib as mpl

# Local imports
from ICPSR_22140_processor import *


# For keeping track of partitioning
splits_global = -1

class Simulator:
	def __init__(self, tsv_file1: str, tsv_file2: str, tsv_file3: str, pickle_filename: str, split: int,
				 discount_factor: float, num_splits: int) -> None:
		# Initialize ground truth data
		self.current_test_split = split
		self.proc = ICPSR22140Processor(tsv_file1, tsv_file2, tsv_file3, pickle_filename, filter_sex_only = True)
		self.true_statuses = self.proc.merged_datasets['HIV'][1]
		self.true_covariates = self.proc.merged_datasets['HIV'][0]
		self.discount_factor = discount_factor
		self.num_splits = num_splits

		# Get dataset from processor
		self.overall_G = self.proc.merged_datasets['HIV'][2]
		overall_DG = self.proc.merged_datasets['HIV'][3]
		self.overall_G.remove_edges_from(nx.selfloop_edges(self.overall_G))
		overall_DG.remove_edges_from(nx.selfloop_edges(overall_DG))

		# Pick one root per connected component
		comps = [set(c) for c in nx.connected_components(self.overall_G)]

		comp_roots = {}
		for i, C in enumerate(comps):
			zero_in = [n for n in C if overall_DG.in_degree(n) == 0]
			root = (min(zero_in, key=str) if zero_in else random.choice(tuple(C)))
			comp_roots[i] = root

		assert len(comp_roots) == len(list(nx.connected_components(self.overall_G)))

		# Build BFS projection per component -> forest
		self.overall_bfs_G = nx.Graph()
		self.overall_bfs_DG = nx.DiGraph()
		for i, C in enumerate(comps):
			r = comp_roots[i]
			assert r in C
			sub = self.overall_G.subgraph(C)
			T = nx.bfs_tree(sub, r)
			T_DG = nx.algorithms.traversal.breadth_first_search.bfs_tree(sub, r)
			
			self.overall_bfs_G.add_nodes_from(C)
			self.overall_bfs_G.add_edges_from(T.edges())
			self.overall_bfs_DG.add_nodes_from(C)
			self.overall_bfs_DG.add_edges_from(T_DG.edges())

		# Split data into train/test data
		self.partition_data()
		self.train_G, self.test_G = self.assign_train_test_splits()
		self.train_roots = []
		self.test_roots = []
		for i, C in enumerate(comps):
			if list(C)[0] in list(self.test_G.nodes()):
				self.test_roots.append(comp_roots[i])
			else:
				self.train_roots.append(comp_roots[i])

		# Train thetas if they don't exist
		if not os.path.exists(f'checkpoints/gittins_thetas/split_{self.current_test_split}.pkl'):
			theta_train_cov = {}
			theta_train_statuses = {}
			mapping = {}
			for i in range(self.train_G.number_of_nodes()):
				original_node = list(self.train_G.nodes())[i]
				mapping[original_node] = i
				theta_train_cov[i] = self.true_covariates[original_node]
				theta_train_statuses[i] = self.true_statuses[original_node]

			mapped_G = self.overall_G.subgraph(list(self.train_G.nodes()))
			mapped_G = nx.relabel_nodes(mapped_G, mapping)
			self.proc.set_dataset_info(theta_train_cov, theta_train_statuses, mapped_G)	
			self.proc.fit_theta_parameters(f'split_{self.current_test_split}')

		theta_unary, theta_pairwise = self.proc.get_theta_parameters(f'checkpoints/gittins_thetas/split_{self.current_test_split}')
		self.theta = (theta_unary, theta_pairwise)


		# final sanity checks
		assert nx.utils.misc.graphs_equal(self.overall_bfs_G, nx.Graph(self.overall_bfs_DG))
		assert nx.is_forest(self.overall_bfs_G)
		assert set(list(self.train_G.nodes())).issubset(set(list(self.overall_bfs_G.nodes())))
		assert set(list(self.test_G.nodes())).issubset(set(list(self.overall_bfs_G.nodes())))
		assert set(list(self.train_G.edges())).issubset(set(list(self.overall_bfs_G.edges())))
		assert set(list(self.test_G.edges())).issubset(set(list(self.overall_bfs_G.edges())))
		assert len(set(list(self.test_G.nodes())).intersection(set(list(self.train_G.nodes())))) == 0
		assert len(set(list(self.test_G.edges())).intersection(set(list(self.train_G.edges())))) == 0
		assert self.train_G.number_of_nodes() + self.test_G.number_of_nodes() == self.overall_bfs_G.number_of_nodes()
		assert self.train_G.number_of_edges() + self.test_G.number_of_edges() == self.overall_bfs_G.number_of_edges()

	def partition_data(self) -> None:
		global splits_global
		if splits_global == -1:
			# Split overall graph into connected components
			components_nodes = list(nx.connected_components(self.overall_bfs_G))
			component_subgraphs = [self.overall_bfs_G.subgraph(c).copy() for c in components_nodes]

			# Assign connected components to splits, respecting distribution of connected component sizes
			component_subgraphs.sort(key=lambda x: x.number_of_nodes(), reverse=True)
			self.splits = []
			for _ in range(self.num_splits):
				self.splits.append([])
			for i in range(0, len(component_subgraphs), self.num_splits):
				subgraphs = copy.deepcopy(component_subgraphs[i:i+self.num_splits])
				random.shuffle(subgraphs)
				for j in range(self.num_splits):
					if j < len(subgraphs):
						self.splits[j].append(subgraphs[j])
			splits_global = copy.deepcopy(self.splits)
		else:
			self.splits = copy.deepcopy(splits_global)

	def assign_train_test_splits(self, display_connected_component_distributions = False) -> [nx.Graph, nx.Graph]:
		# Build train/test sets
		train_graph = copy.deepcopy(self.overall_bfs_G)
		test_graph = copy.deepcopy(self.overall_bfs_G)

		for i in range(self.num_splits):
			if i == self.current_test_split:
				for connected_component in self.splits[i]:
					train_graph.remove_nodes_from(list(connected_component.nodes()))
			else:
				for connected_component in self.splits[i]:
					test_graph.remove_nodes_from(list(connected_component.nodes()))

		# Show train/test set connected component size distributions
		if display_connected_component_distributions:
			for graph in [self.overall_bfs_G, train_graph, test_graph]:
				connected_component_sizes = [len(x) for x in nx.connected_components(nx.Graph(graph))]
				values, bins, bars = plt.hist(connected_component_sizes, bins = 100)
				plt.show()

		return train_graph, test_graph

	def step(self, selected_node: int) -> None:
		# Check to make sure selected node is valid
		assert selected_node in self.incremental_G.nodes(), "Selected test recipient was not present in graph"
		assert selected_node not in self.incremental_statuses, "Selected test recipient already tested"

		# Update statistics
		self.tests_distributed += 1
		if self.true_statuses[selected_node] == 1:
			self.positive_tests_distributed += 1
			self.discounted_reward += self.discount_factor**len(self.incremental_statuses)

		# Update and verify incremental graph/data
		self.incremental_statuses[selected_node] = self.true_statuses[selected_node]
		successors = list(self.test_G.neighbors(selected_node))

		self.incremental_G.add_edges_from([(selected_node, successor) for successor in successors if successor not in list(self.incremental_G.nodes())])
		for successor in successors:
			self.incremental_covariates[successor] = self.true_covariates[successor]

		assert set(list(self.incremental_G.nodes())).issubset(set(list(self.test_G.nodes())))

	def simulation_complete(self) -> bool:
		# Check if all statuses have been discovered and verify forest completion
		complete = self.incremental_G.number_of_nodes() == self.test_G.number_of_nodes()
		if not complete:
			return False
		for node in list(self.incremental_G.nodes()):
			if node not in self.incremental_statuses:
				return False

		assert nx.utils.misc.graphs_equal(self.incremental_G, self.test_G)
		return True

	def reset(self, theta_all = False):
		# Reset incremental graph
		self.incremental_G = nx.Graph()
		self.incremental_G.add_nodes_from(self.test_roots)
		self.incremental_statuses = {}
		self.incremental_covariates = {}
		for node in self.incremental_G.nodes():
			self.incremental_covariates[node] = copy.deepcopy(self.true_covariates[node])

		if theta_all:
			theta_unary, theta_pairwise = self.proc.get_theta_parameters(f'checkpoints/gittins_thetas/all')
			self.theta = (theta_unary, theta_pairwise)
		else:
			theta_unary, theta_pairwise = self.proc.get_theta_parameters(f'checkpoints/gittins_thetas/split_{self.current_test_split}')
			self.theta = (theta_unary, theta_pairwise)

		# Reset statistics
		self.tests_distributed = 0
		self.positive_tests_distributed = 0
		self.discounted_reward = 0



if __name__ == '__main__':
	# Data files
	tsv_file1 = "data/22140-0001-Data.tsv"
	tsv_file2 = "data/22140-0002-Data.tsv"
	tsv_file3 = "data/22140-0003-Data.tsv"
	pickle_filename = "data/ICPSR_22140.pkl"
	experiment = 1
	
	# Initialize settings. k, d, and aggregation apply only to PEGE + DDB (Ours)
	num_splits = 5
	k = 24
	d = 3
	gamma = 0.99
	aggregation = 'mean'

	if experiment == 2:
		policies = ['Naive Gittins', 'PEGE + DDB (Full Theta)', 'PEGE + DDB (Ours)', 'Fully Observable Gittins']
	else:
		policies = ['Random', 'Greedy Neighbor', 'Greedy Classifier', 'DQN', 'PEGE + DDB (Ours)', 'Fully Observable Gittins']

	# Track performance
	all_results = []

	# Run experiments
	for s in range(0, num_splits):
		print(f'Split {s}')

		# Initialize simulator
		simulator = Simulator(tsv_file1 = tsv_file1, tsv_file2 = tsv_file2, tsv_file3 = tsv_file3, pickle_filename = pickle_filename, split = s,
							  discount_factor = gamma, num_splits = num_splits)

		# Track performance of each policy
		performance = {}
		for i in range(len(policies)):
			policy = policies[i]
			print(f'Policy: {policy}')

			# FOG uses theta parameters trained on entire dataset
			if policy == 'Fully Observable Gittins' or policy == 'PEGE + DDB (Full Theta)':
				simulator.reset(theta_all = True)
			else:
				simulator.reset()

			# Initialize performance array
			performance[policy] = [0]

			# Initialize agent
			agent = algorithms.Agent(policy = policy, env = simulator, k = k, d = d, aggregation = aggregation)

			# Step through simulation until complete
			while not simulator.simulation_complete():
				next_test = agent.get_action(simulator)
				simulator.step(next_test)
				performance[policy].append(simulator.discounted_reward)

			all_results.append(performance[policy])
			print(f'Total discounted reward: {simulator.discounted_reward}')
	
	# Save results
	file_path = f"results/results{experiment}.pkl"
	with open(file_path, "wb") as file:
		pickle.dump(all_results, file)

	# Load results
	# with open(file_path, 'rb') as file:
	# 	all_results = pickle.load(file)

	# Check to make sure results are in the right format
	num_policies = len(policies)
	assert len(all_results) % num_policies == 0
	fog_policy = 'Fully Observable Gittins'
	assert fog_policy in policies
	fog_policy_idx = policies.index(fog_policy)

	# Collect FOG rewards for normalization
	agg_weights = []
	fog_final_rewards = []
	for trial in range(num_splits):
		split_fog_row = all_results[int(trial*num_policies + fog_policy_idx)]
		split_size = len(split_fog_row) - 1
		fog_final_reward = split_fog_row[-1]
		if fog_final_reward <= 0:
			raise ValueError(f'{fog_policy} final discounted reward must be positive for split {trial}')
		for p in range(num_policies):
			assert len(all_results[int(trial*num_policies + p)]) == split_size + 1
		agg_weights.append(split_size)
		fog_final_rewards.append(fog_final_reward)

	def standard_error_across_splits(split_curves):
		if split_curves.shape[0] < 2:
			return np.zeros(split_curves.shape[1])
		return np.std(split_curves, axis = 0, ddof = 1) / np.sqrt(split_curves.shape[0])

	# Aggregate results
	policy_performances_discounted = {}
	policy_performances_undiscounted = {}
	policy_standard_errors_discounted = {}
	for policy in policies:
		policy_performances_discounted[policy] = []
		policy_performances_undiscounted[policy] = []

	x_new = np.arange(301)/300.0
	for trial in range(num_splits):
		for p in range(num_policies):
			data_row_discounted = all_results[int(trial*num_policies + p)]
			data_row_discounted = np.array(data_row_discounted, dtype = float) / fog_final_rewards[trial]
			data_row_undiscounted = [0]
			for i in range(1, len(data_row_discounted)):
				if data_row_discounted[i] > data_row_discounted[i-1]:
					data_row_undiscounted.append(data_row_undiscounted[i-1] + 1)
				else:
					data_row_undiscounted.append(data_row_undiscounted[i-1])

			f_discounted = interpolate.interp1d(np.arange(len(data_row_discounted))/(len(data_row_discounted)-1), np.array(data_row_discounted))
			f_undiscounted = interpolate.interp1d(np.arange(len(data_row_undiscounted))/(len(data_row_undiscounted)-1), np.array(data_row_undiscounted))

			policy_performances_discounted[policies[p]].append(f_discounted(x_new))
			policy_performances_undiscounted[policies[p]].append(f_undiscounted(x_new))

	mpl.rcParams['font.size'] = 19

	# Plot weighted algorithm means with split-level standard errors
	for policy in policies:
		split_curves_discounted = np.array(policy_performances_discounted[policy])
		split_curves_undiscounted = np.array(policy_performances_undiscounted[policy])
		policy_performances_discounted[policy] = np.average(split_curves_discounted, weights = agg_weights, axis = 0)
		policy_performances_undiscounted[policy] = np.average(split_curves_undiscounted, weights = agg_weights, axis = 0)
		policy_standard_errors_discounted[policy] = standard_error_across_splits(split_curves_discounted)

		print(f'{policy} AUC: {np.trapezoid(policy_performances_discounted[policy], x_new)}')
		for budget_fraction in [0.10, 0.25, 0.50, 0.75]:
			budget_idx = int(round(budget_fraction * (len(x_new) - 1)))
			print(f'{int(100*budget_fraction)} percent budget: {policy_performances_discounted[policy][budget_idx]} discounted, {policy_performances_undiscounted[policy][budget_idx]} undiscounted')
		line = plt.plot(x_new, policy_performances_discounted[policy], label = policy, linewidth = 1.75)[0]
		plt.fill_between(
			x_new,
			policy_performances_discounted[policy] - policy_standard_errors_discounted[policy],
			policy_performances_discounted[policy] + policy_standard_errors_discounted[policy],
			color = line.get_color(),
			alpha = 0.15,
			linewidth = 0
		)

	plt.xlabel('Fraction of population tested')
	plt.ylabel(f'Normalized cumulative discounted reward')
	plt.legend()
	plt.tight_layout()
	plt.show()

# Standard library imports
import os
import random
import timeit

from collections import deque

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from tqdm import tqdm

# Local imports
from core.binary_frontier_environment import BinaryFrontierEnv
from core.abstract_policy_class import AbstractPolicyClass

class DQNPolicy(AbstractPolicyClass):
    def __init__(self, env: BinaryFrontierEnv, instance_hash: str) -> None:
        self.rng_seed = 314
        random.seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        torch.manual_seed(self.rng_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.rng_seed)
            torch.cuda.manual_seed_all(self.rng_seed)
        if torch.backends.mps.is_available():
            torch.manual_seed(self.rng_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
            
        super().__init__(env, instance_hash)
        self.load_checkpoint()

    @staticmethod
    def name() -> str:
        return "DQN"

    def _setup_policy(self) -> None:
        self.buffer_capacity = 10000
        self.batch_size = 32
        self.gamma = self.env.discount_factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.update_target_every = 10
        self.num_episodes = 50
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.dqn_checkpoint_filename = f"checkpoints/dqn/{self.instance_hash}.pt"
        os.makedirs(os.path.dirname(self.dqn_checkpoint_filename), exist_ok=True)

        # Prepare GNN data representation
        status, unary_factors, pairwise_factors = self.env.get_status_and_factors()
        n = len(status)
        assert len(unary_factors) == n
        self.node_covariates = np.array([
            factor_array.flatten()
            for factor_array in unary_factors.values()
        ])
        assert self.node_covariates.shape == (len(unary_factors), 2)
        edge_log_factor_values = torch.from_numpy(np.array([
            factor_table.flatten()
            for factor_table in pairwise_factors.values()
        ])).float().to(self.device)
        assert edge_log_factor_values.shape == (len(pairwise_factors), 4)
        edge_list = [
            [min([int(Xidx[1:]) for Xidx in uv]), max([int(Xidx[1:]) for Xidx in uv])]
            for uv in pairwise_factors.keys()
        ]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(self.device)
        self.edge_attr = torch.cat([edge_log_factor_values, edge_log_factor_values], dim=0).to(self.device)

        self.q_net = EdgeFeatureGNN(node_in_dim=3, edge_in_dim=4, hidden_dim=64).to(self.device)
        self.target_net = EdgeFeatureGNN(node_in_dim=3, edge_in_dim=4, hidden_dim=64).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

    def _train_policy(self) -> None:
        buffer = ReplayBuffer(capacity=self.buffer_capacity)
        start_episode = self.load_checkpoint()
        start_time = timeit.default_timer()
        if start_episode < self.num_episodes:
            for episode in tqdm(range(start_episode+1, self.num_episodes+1), desc="Training DQN", leave=True):
                status, valid_actions_array = self.env.reset()
                total_reward = 0
                done = False
                while not done:
                    data = self.status_to_data(status)
                    q_values = self.q_net(data.x, data.edge_index, data.edge_attr)
                    action_mask = torch.tensor(valid_actions_array, dtype=torch.bool, device=self.device)
                    valid_indices = torch.nonzero(action_mask).squeeze(-1)
                    if len(valid_indices) == 0:
                        break
                    if torch.rand(1).item() < self.epsilon:
                        action = valid_indices[torch.randint(len(valid_indices), (1,), device=self.device)].item()
                    else:
                        masked_q = q_values.clone()
                        masked_q[~action_mask] = -float('inf')
                        action = masked_q.argmax().item()
                    
                    status_next, valid_actions_array_next, reward, done = self.env.step(int(action))
                    action_mask_next = torch.tensor(valid_actions_array_next, dtype=torch.bool, device=self.device)
                    buffer.push(status.copy(), action, reward, status_next.copy(), done, action_mask_next)
                    status = status_next
                    valid_actions_array = valid_actions_array_next
                    total_reward += reward

                    if len(buffer) >= self.batch_size:
                        statuses, actions, rewards, next_states, dones, masks = buffer.sample(self.batch_size)
                        q_preds = []
                        q_targets = []

                        for s, a, r, s_next, d, m_next in zip(statuses, actions, rewards, next_states, dones, masks):
                            s_data = self.status_to_data(s)
                            q_pred = self.q_net(s_data.x, s_data.edge_index, s_data.edge_attr)[a]
                            with torch.no_grad():
                                s_next_data = self.status_to_data(s_next)
                                q_next = self.target_net(s_next_data.x, s_next_data.edge_index, s_next_data.edge_attr)
                                q_next[~m_next] = -float('inf')
                                q_max = q_next.max() if not d else 0.0
                            q_target = r + self.gamma * q_max
                            q_preds.append(q_pred)
                            if isinstance(q_target, torch.Tensor):
                                q_targets.append(q_target.detach().clone().to(dtype=torch.float32, device=self.device))
                            else:
                                q_targets.append(torch.tensor(q_target, dtype=torch.float32, device=self.device))

                        q_preds = torch.stack(q_preds)
                        q_targets = torch.stack(q_targets)

                        loss = F.mse_loss(q_preds, q_targets)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                
                if episode % self.update_target_every == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                if episode % 10 == 0 or episode == self.num_episodes:
                    end_time = timeit.default_timer()
                    self.train_time += end_time - start_time
                    self.save_checkpoint(episode)
                    start_time = timeit.default_timer()

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        end_time = timeit.default_timer()
        self.train_time += end_time - start_time
            
    def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
        self.q_net.eval()
        valid_actions_array = np.array([1 if idx in valid_actions else 0 for idx in range(self.n)])
        with torch.no_grad():
            data = self.status_to_data(status)
            action_mask = torch.tensor(valid_actions_array, dtype=torch.bool, device=self.device)
            q_values = self.q_net(data.x, data.edge_index, data.edge_attr)
            q_values[~action_mask] = -float('inf')
            action = q_values.argmax().item()
        return action
    
    def save_checkpoint(self, episode):
        torch.save({
            'episode': episode,
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_time_so_far': self.train_time
        }, self.dqn_checkpoint_filename)

    def load_checkpoint(self) -> int:
        if not os.path.exists(self.dqn_checkpoint_filename):
            return 0
        checkpoint = torch.load(self.dqn_checkpoint_filename, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_time = checkpoint['train_time_so_far']
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return checkpoint['episode']
    
    def status_to_data(self, status):
        status_feat = status.astype(np.float32)[:, None]
        node_feat = np.hstack([self.node_covariates, status_feat])
        x = torch.tensor(node_feat, dtype=torch.float, device=self.device)
        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
    
class EdgeFeatureGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.edge_nn1 = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_in_dim * hidden_dim)
        )
        self.edge_nn2 = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv1 = NNConv(node_in_dim, hidden_dim, self.edge_nn1, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_nn2, aggr='mean')
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.q_head(x).squeeze(-1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.rng_seed = 314
        random.seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        torch.manual_seed(self.rng_seed)
        self.rng = np.random.default_rng(self.rng_seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask_next):
        self.buffer.append((state, action, reward, next_state, done, mask_next))

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)
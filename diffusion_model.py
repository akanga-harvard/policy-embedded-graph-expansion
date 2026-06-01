# External imports
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
random.seed(1234)
import math
import copy

# Local imports
import simulator


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sin/cos embedding for diffusion timesteps.

    Args:
        dim (int): Output embedding dimension (must be even).
        max_period (float): Controls the minimum frequency. 10k matches common practice.
        scale_by_2pi (bool): If True, multiplies by 2π (useful when t is normalized to [0,1]).

    Forward:
        t: Tensor of shape (B,) or (B,1). Can be integer (0..T-1) or float (e.g., t/T).

    Returns:
        Tensor of shape (B, dim)
    """
    def __init__(self, dim: int = 16, max_period: float = 10000.0, scale_by_2pi: bool = True):
        super().__init__()
        assert dim % 2 == 0 and dim >= 2, "dim must be even and >= 2"
        self.dim = dim
        self.max_period = float(max_period)
        self.scale_by_2pi = scale_by_2pi

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t -> (B,)
        if t.ndim == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        t = t.to(dtype=torch.float32)

        half = self.dim // 2
        # Exponential spacing of frequencies: [max_period^0, max_period^{-1/(half-1)}, ..., max_period^{-1}]
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )  # (half,)

        # Outer product: (B,1) * (1,half) -> (B,half)
        arg = t.unsqueeze(-1) * freqs.unsqueeze(0)
        if self.scale_by_2pi:
            arg = 2.0 * math.pi * arg

        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (B, dim)
        return emb

class SimpleNN(nn.Module):
    def __init__(self, input_dim=72*2, hidden_dim=512):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim//2)
        )

    def forward(self, x, t):
        t_e = self.time_embedding(t)
        input_data = torch.hstack([x, t_e])
        return self.net(input_data)

def _make_torch_generator(seed, device="cpu"):
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed) % (2**63))
    return generator

def _randn(shape, device, dtype=torch.float32, generator=None):
    if generator is None:
        return torch.randn(size=shape, dtype=dtype, device=device)
    try:
        return torch.randn(size=shape, dtype=dtype, device=device, generator=generator)
    except RuntimeError:
        return torch.randn(size=shape, dtype=dtype, generator=generator).to(device)

def _randn_like(x, generator=None):
    return _randn(x.shape, x.device, dtype=x.dtype, generator=generator)

def _randint(low, high, size, device, generator=None):
    if generator is None:
        return torch.randint(low, high, size=size, device=device)
    try:
        return torch.randint(low, high, size=size, device=device, generator=generator)
    except RuntimeError:
        return torch.randint(low, high, size=size, generator=generator).to(device)

def train(
    data,
    batch_size,
    device,
    epochs,
    diffusion_steps,
    min_beta,
    max_beta,
    learning_rate,
    output_model_path,
    seed=1234,
):
    # Initialize data loader, model, and optimizer
    device = torch.device(device)
    loader_generator = _make_torch_generator(seed, "cpu")
    train_generator = _make_torch_generator(None if seed is None else seed + 1, device)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, generator=loader_generator)
    if seed is None:
        model = SimpleNN().to(device)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) % (2**63))
            model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Initialize alpha and beta schedules
    beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
        diffusion_steps, min_beta, max_beta
    )
    bar_alpha_ts = bar_alpha_ts.to(device)

    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for x in data_loader:
            x = x.view(x.size(0), -1).to(device)

            # Noise data
            random_time_step = _randint(0, diffusion_steps, size=[len(x), 1], device=device, generator=train_generator)
            noised_x_t, eps = calculate_data_at_certain_time(
                x[:, 72:], bar_alpha_ts, random_time_step, generator=train_generator
            )

            # Normalize timestep
            random_time_step = random_time_step/(diffusion_steps - 1.0)

            # Condition on parent node covariates (first 72 entries)
            noised_x_t_conditioned = torch.cat((x[:,:72], noised_x_t), dim = 1)

            # Predict noise
            predicted_eps = model.forward(
                noised_x_t_conditioned, random_time_step
            )

            # Backpropagate
            loss = loss_fn(predicted_eps, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        if epoch % 100 == 0:
            print("Epoch {0}, Loss={1}".format(epoch, round(epoch_loss / count, 5)))

    print("Finished training!!")
    torch.save(model.state_dict(), output_model_path)
    print("Saved model: ", output_model_path)

def calculate_parameters(diffusion_steps, min_beta, max_beta):
    assert diffusion_steps > 1
    assert 0 < min_beta < max_beta < 1

    # Linear beta schedule scaled to the requested number of diffusion steps.
    step_scale = 1000.0 / diffusion_steps
    beta_ts = torch.linspace(min_beta * step_scale, max_beta * step_scale, diffusion_steps)
    beta_ts = torch.clamp(beta_ts, max=0.999)
    alpha_ts = 1.0 - beta_ts
    bar_alpha_ts = torch.cumprod(alpha_ts, dim=0)
    return beta_ts, alpha_ts, bar_alpha_ts

def calculate_data_at_certain_time(x_0, bar_alpha_ts, t, generator=None):
    # Add random noise to data
    eps = _randn_like(x_0, generator=generator)
    noised_x_t = (
        torch.sqrt(bar_alpha_ts[t]) * x_0 + torch.sqrt(1 - bar_alpha_ts[t]) * eps
    )
    return noised_x_t, eps

def sampling(model_path, conditions, diffusion_steps, min_beta, max_beta, device=None, seed=None):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)

    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    conditions = torch.as_tensor(conditions, dtype=torch.float32, device=device)
    sampling_generator = _make_torch_generator(seed, device)
    sample_num = conditions.shape[0]
    with torch.no_grad():
        # Start from pure noise
        x_t = _randn((sample_num, 72), device=device, generator=sampling_generator)
        beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(diffusion_steps, min_beta, max_beta)
        beta_ts = beta_ts.to(device)
        alpha_ts = alpha_ts.to(device)
        bar_alpha_ts = bar_alpha_ts.to(device)

        for t in range(diffusion_steps - 1, -1, -1):
            ts = torch.full((sample_num, 1), t/(diffusion_steps - 1), dtype=torch.float32, device=device)
            # predict epsilon from [cond || x_t, t]
            eps_hat = model.forward(torch.cat((conditions, x_t), dim=1), ts)

            # DDPM mean (epsilon parameterization)
            mu = (1.0 / torch.sqrt(alpha_ts[t])) * (
                    x_t - (1 - alpha_ts[t]) / torch.sqrt(1 - bar_alpha_ts[t]) * eps_hat
            )

            # Last reverse step maps x_1 to x_0 and is deterministic.
            if t > 0:
                # posterior variance (tilde beta)
                tilde_beta_t = (1 - bar_alpha_ts[t - 1]) / (1 - bar_alpha_ts[t]) * beta_ts[t]
                z = _randn_like(x_t, generator=sampling_generator)
                x_t = mu + torch.sqrt(tilde_beta_t) * z
            else:
                x_t = mu

    x_t = x_t.detach().cpu().numpy()

    # For each one-hot covariate, set the maximum value to 1 and the rest to 0 to ensure that output is valid covariates
    
    return one_hot(x_t)

def one_hot(x):
    x_one_hot = copy.deepcopy(x)
    for row in x_one_hot:
        row[0 + np.argmax(row[0:4])] = np.inf
        row[4 + np.argmax(row[4:11])] = np.inf
        row[11 + np.argmax(row[11:15])] = np.inf
        row[15 + np.argmax(row[15:18])] = np.inf
        row[18 + np.argmax(row[18:24])] = np.inf
        row[24 + np.argmax(row[24:27])] = np.inf
        row[27 + np.argmax(row[27:31])] = np.inf
        row[31 + np.argmax(row[31:35])] = np.inf
        row[35 + np.argmax(row[35:39])] = np.inf
        row[39 + np.argmax(row[39:43])] = np.inf
        row[43 + np.argmax(row[43:47])] = np.inf
        row[47 + np.argmax(row[47:51])] = np.inf
        row[51 + np.argmax(row[51:55])] = np.inf
        row[55 + np.argmax(row[55:59])] = np.inf
        row[59 + np.argmax(row[59:64])] = np.inf
        row[64 + np.argmax(row[64:68])] = np.inf
        row[68 + np.argmax(row[68:72])] = np.inf

    x_one_hot[x_one_hot != np.inf] = 0
    x_one_hot[x_one_hot == np.inf] = 1
    x_one_hot = x_one_hot.astype(int)
    return x_one_hot

def prepare_dataset(env):
    train_data = []
    test_data = []
    for node in list(env.train_G.nodes()):
        for successor in list(env.overall_G.neighbors(node)):
            parent_data = list(env.true_covariates[node])
            child_data = list(env.true_covariates[successor])
            train_data.append(np.array(parent_data + child_data))

    for node in list(env.test_G.nodes()):
        for successor in list(env.overall_bfs_DG.successors(node)):
            parent_data = list(env.true_covariates[node])
            child_data = list(env.true_covariates[successor])
            test_data.append(np.array(parent_data + child_data))

    train_data = torch.tensor(np.array(train_data), dtype = torch.float32)
    test_data = torch.tensor(np.array(test_data), dtype = torch.float32)
    return train_data, test_data

def train_new_model(env):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))

    train_data, test_data = prepare_dataset(env)

    batch_size = 32
    epochs = 4000
    diffusion_steps = 100
    min_beta = 1e-4
    max_beta = 0.02
    learning_rate = 1e-3
    output_model_path = f"checkpoints/diffusion_model/diffusion_model_split_{env.current_test_split}.pth"
    seed = getattr(env, "rng_seed", 1234) + 1000 * getattr(env, "current_test_split", 0)
    train(
        train_data,
        batch_size,
        device,
        epochs,
        diffusion_steps,
        min_beta,
        max_beta,
        learning_rate,
        output_model_path,
        seed=seed
    )

import torch
from torch import nn
from torch.nn import functional as F


class ReliabilityGraph(nn.Module):
    def __init__(self, distance, channels=1, width=64, horizon=12, neighbors=10, graph_window=12,
                 minimum_pairs=4, shrinkage=10, diffusion=2, variant="full", dropout=0.1):
        super().__init__()
        distance = torch.as_tensor(distance, dtype=torch.float32)
        nodes = len(distance)
        if distance.shape != (nodes, nodes) or (distance < 0).any():
            raise ValueError("Nonnegative node-aligned distance matrix required")
        distance = distance.clone()
        distance.fill_diagonal_(0)
        ordering = distance.clone()
        ordering.fill_diagonal_(-1)
        candidate = ordering.argsort(1)[:, :min(neighbors + 1, nodes)]
        distances = distance.gather(1, candidate)
        positive = distances[torch.isfinite(distances) & (distances > 0)]
        scale = positive.median() if len(positive) else torch.tensor(1.0)
        self.register_buffer("candidate", candidate)
        self.register_buffer("static_score", (-distances / scale).nan_to_num(neginf=-100))
        self.value = nn.Linear(channels, width)
        self.mask = nn.Linear(channels, width, bias=False)
        self.clock = nn.Linear(2, width, bias=False)
        self.identity = nn.Embedding(nodes, width)
        self.graph_maps = nn.ModuleList([nn.Linear(width, width) for _ in range(diffusion + 1)])
        self.cell = nn.GRUCell(width, width)
        self.dropout = nn.Dropout(dropout)
        self.impute = nn.Linear(width, channels)
        self.mean = nn.Linear(width, horizon * channels)
        self.variance = nn.Linear(width, horizon * channels)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.width, self.channels, self.horizon = width, channels, horizon
        self.graph_window, self.minimum_pairs, self.shrinkage = graph_window, minimum_pairs, shrinkage
        self.variant = variant

    @torch.no_grad()
    def correlation(self, x, mask):
        first = x[:, :, :, None, :]
        other = x[:, :, self.candidate, :]
        joint = mask[:, :, :, None, :] * mask[:, :, self.candidate, :]
        count = joint.sum((1, 4))
        mean_first = (first * joint).sum((1, 4)) / count.clamp_min(1)
        mean_other = (other * joint).sum((1, 4)) / count.clamp_min(1)
        centered_first = (first - mean_first[:, None, :, :, None]) * joint
        centered_other = (other - mean_other[:, None, :, :, None]) * joint
        var_first = centered_first.square().sum((1, 4))
        var_other = centered_other.square().sum((1, 4))
        covariance = (centered_first * centered_other).sum((1, 4))
        correlation = covariance / (var_first.sqrt() * var_other.sqrt() + 1e-6)
        valid = (count >= self.minimum_pairs) & (var_first > 1e-6) & (var_other > 1e-6)
        return valid * correlation.clamp(-1, 1) * count / (count + self.shrinkage)

    def forward(self, values, mask, clock):
        batch, steps, nodes, _ = values.shape
        hidden = values.new_zeros(batch * nodes, self.width)
        imputation, entropy = [], []
        for step in range(steps):
            left = max(0, step - self.graph_window + 1)
            prefix = values[:, left:step + 1]
            observed = mask[:, left:step + 1]
            correlation = self.correlation(prefix, observed)
            if self.variant in ("static", "no_correlation"):
                correlation = torch.zeros_like(correlation)
            affinity = F.softplus(self.alpha) * self.static_score + F.softplus(self.beta) * correlation
            affinity = (affinity - affinity.max(-1, keepdim=True).values).exp()
            reliability = observed.mean((1, 3))
            if self.variant in ("no_gate", "static"):
                reliability = torch.ones_like(reliability)
            weight = affinity * reliability[:, self.candidate]
            adjacency = weight / (weight.sum(-1, keepdim=True) + 1e-6)
            embedding = self.value(values[:, step] * mask[:, step]) + self.mask(mask[:, step])
            embedding = embedding + self.clock(clock[:, step])[:, None] + self.identity.weight[None]
            message = self.graph_maps[0](embedding)
            propagated = embedding
            for transform in self.graph_maps[1:]:
                propagated = (adjacency[..., None] * propagated[:, self.candidate]).sum(2)
                message = message + transform(propagated)
            hidden = self.cell(self.dropout(F.relu(message)).reshape(batch * nodes, self.width), hidden)
            state = hidden.reshape(batch, nodes, self.width)
            imputation.append(self.impute(state))
            entropy.append(-(adjacency * (adjacency + 1e-6).log()).sum(-1).mean())
        mean = self.mean(state).reshape(batch, nodes, self.horizon, self.channels).transpose(1, 2)
        variance = F.softplus(self.variance(state)).reshape(batch, nodes, self.horizon, self.channels).transpose(1, 2) + 1e-6
        return torch.stack(imputation, 1), mean, variance, torch.stack(entropy).mean()


def loss(model, x, available, hidden, clock, y, target_available, imputation_weight=1, entropy_weight=0.001):
    if hasattr(model,"objective"):
        return model.objective(x,available,hidden,clock,y,target_available,imputation_weight,entropy_weight)
    reconstruction, mean, variance, entropy = model(x, available * (1 - hidden), clock)
    reconstruction_mask = available * hidden
    reconstruction_loss = ((reconstruction - x).abs() * reconstruction_mask).sum() / reconstruction_mask.sum().clamp_min(1)
    nll = 0.5 * (torch.log(2 * torch.pi * variance) + (y - mean).square() / variance)
    nll = (nll * target_available).sum() / target_available.sum().clamp_min(1)
    if model.variant == "no_imputation":
        imputation_weight = 0
    return nll + imputation_weight * reconstruction_loss + entropy_weight * entropy

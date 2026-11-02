"""Frozen deterministic policy for the supplied evaluation checkpoint."""
import numpy as np
import torch
from torch import nn

class Actor(nn.Module):
    """Actor network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, s):
        x = self.net(s)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

class Policy:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.actor = Actor(17, 3).to(self.device)

    def act(self, state, deterministic=True):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, _ = self.actor(state)
            return torch.tanh(mean)

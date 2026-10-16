"""Inference-only Bayesian linear layer; parameter names match upstream weights."""
import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-.2, .2))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-.2, .2))
        self.bias_rho = nn.Parameter(torch.empty(out_features).uniform_(-5, -4))

    def forward(self, x, sample=False):
        weight, bias = self.weight_mu, self.bias_mu
        if sample:
            # Upstream draws on CPU then transfers noise to the parameter device.
            weight = weight + F.softplus(self.weight_rho) * torch.randn(weight.shape).to(weight.device)
            bias = bias + F.softplus(self.bias_rho) * torch.randn(bias.shape).to(bias.device)
        return F.linear(x, weight, bias)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFLayer(nn.Module):
    def __init__(self, units, gamma):
        super(RBFLayer, self).__init__()
        self.units = units
        self.gamma = gamma

    def initialize(self, input_shape):
        self.mu = nn.Parameter(torch.randn(int(input_shape[2]), self.units))
        self.w = nn.Parameter(torch.randn(int(input_shape[2]),))

    def forward(self, inputs):
        diff = inputs.unsqueeze(1) - self.mu
        l2 = torch.sum(torch.pow(diff, 2), dim=-1)
        res = self.w * torch.exp(-1 * self.gamma * l2)
        return res

    def extra_repr(self):
        return 'units={}, gamma={}'.format(self.units, self.gamma)

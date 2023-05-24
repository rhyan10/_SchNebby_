import torch
import torch.nn as nn
import schnetpack as spk
from schnetpack.nn.activations import shifted_softplus

class Critic_Model(nn.Module):
    def __init__(self, args):
        super(Critic_Model, self, ).__init__()
        self.n_layers = 8
        self.n_out = 1
        self.n_images = args.n_images
        self.outnet = spk.nn.build_mlp(
            n_in=args.basis,
            n_out=args.basis,
            n_layers=self.n_layers,
            activation=shifted_softplus,
        ).to(torch.float64)

        self.outnet2 = spk.nn.build_mlp(
            n_in=args.basis,
            n_out=self.n_out,
            n_layers=self.n_layers,
            activation=shifted_softplus,
        ).to(torch.float64)

        self.tangent_embedding = spk.nn.build_mlp(
             n_in=2,
             n_out=args.basis,
             n_layers=self.n_layers,
             activation=shifted_softplus,
        ).to(torch.float64)

    def forward(self, inputs, tangent_input):
        repr = inputs["scalar_representation"].detach()
        mol_splits = tuple(inputs["_n_atoms"].cpu().numpy())
        o0 = self.outnet(repr)
        tangent = tangent_input.detach()
        tangent_embedding = self.tangent_embedding(tangent)
        o1 = self.outnet2(o0 * tangent_embedding)
        o1 = torch.squeeze(o0)
        o3 = torch.split(o1, mol_splits)
        o4 = [torch.sum(val) for val in o3]
        o4 = torch.stack(o4)
        o4 = torch.split(o4, self.n_images)
        o5 = [-torch.sum(val) for val in o4]
        o5 = torch.stack(o5, dim=0)
        return o5

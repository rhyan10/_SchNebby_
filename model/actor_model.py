import torch
import torch.nn as nn
from schnetpack.atomistic import PairwiseDistances
import numpy as np
import schnetpack.transform as trn
from utils.loader import AtomsConverter
import schnetpack as spk
from schnetpack.nn.activations import shifted_softplus

class Actor_Model(nn.Module):
    def __init__(self, 
        n_atom_basis: int,
        n_images: int,
        n_bins: int,
        cutoff: int,
        batch_size: int,
        device: str,
        k: int,
    ):
        super(Actor_Model, self).__init__()
        self.Rn_layers = 6
        self.Rn_out = n_bins
        self.n_images = n_images
        self.limit = 0.5
        self.n_bins = n_bins
        self.device = device
        self.batch_size = batch_size
        self.n_atom_basis = n_atom_basis
        self.k = k

        self.force_outnet = spk.nn.build_mlp(
             n_in=3*n_atom_basis,
             n_out=n_bins,
             n_layers=self.Rn_layers,
             activation=shifted_softplus,
        ).to(torch.float64)

        self.tangent_outnet = spk.nn.build_mlp(
             n_in=3*n_atom_basis,
             n_out=n_bins,
             n_layers=self.Rn_layers,
             activation=shifted_softplus,
        ).to(torch.float64)
         
        self.tangent_embedding = spk.nn.build_mlp(
             n_in=2,
             n_out=3*n_atom_basis,
             n_layers=self.Rn_layers,
             activation=shifted_softplus,
        ).to(torch.float64)

        self.self_attn_force = nn.MultiheadAttention(n_atom_basis, 1, dropout=0.2, batch_first=True, dtype=torch.float64)
        self.self_attn_tangent = nn.MultiheadAttention(n_atom_basis, 1, dropout=0.2, batch_first=True, dtype=torch.float64)

        self.r_grid = torch.linspace(0, 100, self.n_bins).to(device)  
        self.t_grid = torch.linspace(-1.1, -0.9, self.n_bins).to(device)  

        self.atoms_converter = AtomsConverter(
            self.n_images,
            neighbor_list=trn.ASENeighborList(cutoff=cutoff),
            device=device
        )

    def forward(self, inputs, forces, energies, num_samples):

        """
        Forward inputs through output modules and representation.
        """
        repr_ = inputs["scalar_representation"].detach()
        
        #Mask tensors are calculated to support the relevant tangent vector calculation
        positions =  inputs["_positions"]
        mol_size = inputs["_n_atoms"]
        tangent = positions.reshape(self.batch_size,self.n_images,mol_size[0],3)[0][1].detach().cpu().numpy()
        mol_splits = tuple(inputs["_n_atoms"].cpu().numpy())
        energies = energies.reshape(self.batch_size, self.n_images)
        springs = torch.ones(self.batch_size, self.n_images, 3, self.n_atom_basis)
        mask = []
        for i, (path, springs_path, split) in enumerate(zip(energies, springs, mol_splits)):
            imax = torch.argmax(path)
            for j in range(imax+1,self.n_images):
                springs_path[j][-1] = springs_path[j][-1] - 1
            for j in range(imax):
                springs_path[j][0] = springs_path[j][0] - 1
            mol_spring_path = springs_path.repeat(split, 1, 1)
            mask.append(mol_spring_path)

        # The representations of neighbouring molecules are combined
        mask = torch.concat(mask, dim=0).to(self.device)
        repr = torch.split(repr_, mol_splits)
        repr = torch.stack(repr)
        repr = repr.chunk(int(repr.shape[0]/self.n_images))
        repr = torch.stack(repr)
        repr = repr.unsqueeze(-2)
        repr_b = torch.roll(repr, 1, 1)
        repr_f = torch.roll(repr, -1, 1) 
        full_repr = torch.cat((repr_b, repr, repr_f), -2)
        full_repr = full_repr.view(-1, full_repr.size(-2), full_repr.size(-1))

        #The representations are passed through self attention layers and atom-wise dense layers
        att_f_repr, _ = self.self_attn_force(full_repr, full_repr, full_repr)
        att_f_repr = att_f_repr.reshape(att_f_repr.shape[0], 3*att_f_repr.shape[-1])
        force_output = self.force_outnet(att_f_repr)
        force_output = torch.split(force_output, mol_splits)
        force_output = torch.abs(torch.stack(force_output))
        new_shape = (self.batch_size, self.n_images) + force_output.shape[1:]
        force_output = torch.sum(torch.reshape(force_output, new_shape), dim=-2)
        force_soft = force_output/torch.sum(force_output, dim=-1).unsqueeze(-1)
        force_distribution = torch.distributions.Categorical(force_soft)
        force_index = force_distribution.sample().detach()
        force_index = force_index.reshape(self.batch_size,10)
        chosen_values = torch.tensor(np.random.randint(0, self.n_bins, size=self.n_images*self.batch_size)).cuda()
        perm = torch.randperm(self.batch_size*self.n_images)
        perm = perm[:num_samples]
        force_index = force_index.flatten()
        force_index[perm] = chosen_values[perm]
        force_index = force_index.reshape(self.batch_size,10)
        force_index_ = force_index.unsqueeze(-1).repeat(1,1,23)
        force_log_soft = torch.log_softmax(force_output, dim=-1)
        
        #Full representation is passed through a self attention layer
        full_repr = full_repr * mask
        att_f_repr, _ = self.self_attn_tangent(full_repr, full_repr, full_repr)
        att_f_repr = att_f_repr.reshape(att_f_repr.shape[0], 3*att_f_repr.shape[-1])

        #Tangents for spring vectors are calculated using methodology outlined in supplementary material
        positions = torch.split(positions, mol_splits)
        positions = torch.stack(positions)
        positions = positions.chunk(int(positions.shape[0]/self.n_images))
        positions = torch.stack(positions)
        positions_b = torch.roll(positions, 1, 1).view(-1, 3)
        positions_f = torch.roll(positions, -1, 1).view(-1, 3)
        positions = positions.view(-1, 3)
        dr_b = positions - positions_b
        dr_f = positions_f - positions
        dr_b = dr_b.unsqueeze(1)
        dr_f = dr_f.unsqueeze(1)
        dr_bf = torch.concat((dr_b, dr_f), axis=1)
        forces = forces.reshape(self.batch_size,self.n_images,mol_size,3)
        forces = forces * force_index_.unsqueeze(-1)
        forces = forces.flatten()
        position_mask, _ = torch.mode(mask, dim=-1)
        position_mask = torch.stack((position_mask.T[0], position_mask.T[-1])).T.unsqueeze(-1)
        tangent = dr_bf * position_mask
        tangent = torch.sum(tangent, dim=1)
        tangent = tangent.reshape(self.batch_size,self.n_images,mol_size,3)
        tangent_mag = torch.sum(tangent*tangent, dim=-1)
        tangent_mag = torch.sum(tangent_mag, dim=-1)
        factor = tangent/tangent_mag.unsqueeze(-1).unsqueeze(-1)
        tangent_force = torch.sum(forces.reshape(self.batch_size,self.n_images,mol_size,3)*tangent, dim=-1)
        tangent_force = torch.sum(tangent_force, dim=-1)
        parallel_forces = tangent_force.unsqueeze(-1).unsqueeze(-1) * factor
        dr_b = dr_b.reshape(self.batch_size,self.n_images,mol_size,3)
        dr_f = dr_f.reshape(self.batch_size,self.n_images,mol_size,3)
        spring_force = torch.sum(self.k*(dr_b - dr_f) * tangent, dim=-1) 
        spring_force = torch.sum(spring_force, dim=-1).unsqueeze(-1).unsqueeze(-1)* factor
        total_tangent_force = torch.sum((parallel_forces + spring_force).view(-1,3), dim=-1).unsqueeze(-1)
        dot_prod = torch.sum(total_tangent_force*forces, dim=-1).unsqueeze(-1)

        #Relevant spring forces information is passed through a series of atom-wise dense layers and a distribution is created
        tangent_input = torch.cat((total_tangent_force,dot_prod), dim=-1)
        tangent_embedding = self.tangent_embedding(tangent_input)
        full_repr = full_repr.reshape(full_repr.shape[0], 3*full_repr.shape[-1])
        tangent_output = self.tangent_outnet(full_repr*tangent_embedding)
        tangent_output = torch.split(tangent_output, mol_splits)
        tangent_output = torch.abs(torch.stack(tangent_output))
        new_shape = (self.batch_size, self.n_images) + tangent_output.shape[1:]
        tangent_output = torch.sum(torch.reshape(tangent_output, new_shape), dim=-2)
        tangent_soft = tangent_output/torch.sum(tangent_output, dim=-1).unsqueeze(-1)
        tangent_distribution = torch.distributions.Categorical(tangent_soft)
        tangent_index = tangent_distribution.sample().detach()
        chosen_values = torch.tensor(np.random.randint(0, self.n_bins, size=self.n_images*self.batch_size)).cuda()
        perm = torch.randperm(self.batch_size*self.n_images)
        perm = perm[:num_samples]
        tangent_index = tangent_index.flatten()
        tangent_index[perm] = chosen_values[perm]
        tangent_index = tangent_index.reshape(self.batch_size,self.n_images)
        tangent_index_ = tangent_index.unsqueeze(-1).repeat(1,1,mol_size)
        tangent_log_soft = torch.log_softmax(tangent_soft, dim=-1)

        r_chosen = {"f": force_index, "t": tangent_index}
        r_dist = {"f": force_soft, "t": tangent_soft}
        r_log_dist = {"f": force_log_soft, "t": tangent_log_soft}

        positions = torch.split(positions, mol_splits)
        positions = torch.stack(positions)
        new_shape = (self.batch_size, self.n_images) + positions.shape[1:]
        positions = torch.reshape(positions, new_shape)

        return r_chosen, r_dist, r_log_dist, positions, tangent_input, self.r_grid[force_index_], self.r_grid[tangent_index_]

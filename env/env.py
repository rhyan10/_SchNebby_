import torch
import torch.nn.functional as F
import numpy as np
import dill
import schnetpack.transform as trn
import numpy as np
import ase.db
from schnetpack.interfaces import AtomsConverter as SchNetConverter
from ase.optimize import BFGS
from neb.neb import NEB
from math import sqrt

class Env():

    def __init__(self, batch_size, n_images, cutoff, spring_k, n_atom_basis, device, n_bins, model_location):
        self.batch_size = batch_size
        self.n_images = n_images
        self.cutoff = cutoff
        self.n_bins = n_bins
        self.k = spring_k
        self.basis = n_atom_basis
        self.device = device
        self.min_fmax = 10000
        self.schnetpack_model = torch.load(model_location)
        self.schnetpack_model.to(device)
        self.converter = SchNetConverter(neighbor_list=trn.ASENeighborList(cutoff=10), device=device)
        self.limit = 0.5
        self.min_activation = 10000
        self.opt_length = 50

    def step(self, forces, energies, force_index, tangent_index, inputs, atoms, epoch, episode):

        indices = tuple(inputs["_n_atoms"].cpu().numpy())
        atom_forces = torch.split(forces, indices)
        best_path = []
        total_fmax = []
        for j in range(self.batch_size):
            neb = NEB(atoms[j*self.n_images:(j+1)*self.n_images])
            energies_t = energies[j*self.n_images:(j+1)*self.n_images].cpu().detach().numpy()
            force_index_t = force_index[j][1:-1].unsqueeze(-1).cpu().numpy()
            tangent_index_t = tangent_index[j][1:-1].unsqueeze(-1).cpu().numpy()
            forces_t = atom_forces[j*self.n_images+1:(j+1)*self.n_images-1]
            forces_t = torch.stack(forces_t).cpu().detach().numpy()
            neb_forces = neb.get_forces(energies_t, forces_t, force_index_t, tangent_index_t)
            optimizer = BFGS(neb)
            optimizer.step(neb_forces)
            atoms[j*self.n_images:(j+1)*self.n_images] = optimizer.atoms.images
            neb = NEB(atoms[j*self.n_images:(j+1)*self.n_images])
            f_ones = np.ones(np.shape(force_index_t))
            t_ones = np.ones(np.shape(tangent_index_t))
            neb_forces = neb.get_forces(energies_t, forces_t, f_ones, t_ones)
            fmax = sqrt((neb_forces ** 2).sum(axis=1).max())
            total_fmax.append(fmax)
            best_path.append(atoms[j*self.n_images:(j+1)*self.n_images])

            if fmax<self.min_fmax :
                with open('./best_paths/energies_fmax.txt', 'a') as f:
                    f.write(str(energies_t) + '\n')
                with open('./best_paths/fmax.txt', 'a') as f:
                    f.write(str(fmax) + '\n')
                with open('./best_paths/indices_fmax.txt', 'a') as f:
                    f.write(str(epoch)+"-"+str(episode)+"-"+str(j) +'\n')
                with open("./best_paths/fmax_path.pkl", "wb") as f:
                    dill.dump(atoms[j*self.n_images:(j+1)*self.n_images], f)
                self.min_fmax = fmax

            max_energy = np.max(energies_t - energies_t[0])

            if max_energy<self.min_activation:
                with open('./best_paths/energies_act.txt', 'a') as f:
                    f.write(str(energies_t) + '\n')
                with open('./best_paths/indices_act.txt', 'a') as f:
                    f.write(str(epoch)+"-"+str(episode)+"-"+str(j)+'\n')
                with open('./best_paths/act_fmax.txt', 'a') as f:
                    f.write(str(fmax) + '\n')
                with open("./best_paths/act_path.pkl", "wb") as f:
                    dill.dump(atoms[j*self.n_images:(j+1)*self.n_images], f)
                self.min_activation = max_energy


        schnetpack_input = self.converter(atoms)

        results, inputs = self.schnetpack_model(schnetpack_input)

        forces = results["forces"].detach()
        energies = results["energy"].detach()

        total_fmax =  torch.tensor(total_fmax).to(torch.float64)

        return atoms, 1/total_fmax, forces, energies, best_path

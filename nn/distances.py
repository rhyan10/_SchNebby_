import numpy as np
from numpy import linalg as LA

def dist(r):

        ri = np.roll(r, 1, axis=0)
        diff = np.absolute(ri - r)
        dists = LA.norm(diff, axis=-1)
        dists = np.reshape(dists, (mol_size, 1))

        for i in range(2, mol_size):
            ri = np.roll(r, i, axis=0)
            diff = np.absolute(ri - r)
            dij = LA.norm(diff, axis=-1)
            dij = np.reshape(dij, (mol_size, 1))
            dists = np.concatenate((dists, dij), axis = 1)

        # padded_dists = []

        # for i in range(mol_size):
        #     pad = np.pad(dists[i], (0, max_molsize-mol_size+1), 'constant')
        #     padded_dists.append(pad)

        return np.array(dists) 


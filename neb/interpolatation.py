from ase.neb import NEB
from ase import Atoms
from geodesic_interpolate.interpolation import redistribute
from geodesic_interpolate.geodesic import Geodesic
import numpy as np
import logging
logging.disable(logging.INFO)

__all__ = ["get_interpolation"]


def get_interpolation(initial_image, final_image, n_images):

    interpolate_images = ase_geodesic_interpolate(initial_image, final_image, n_images)

    return interpolate_images

def ase_geodesic_interpolate(initial_mol,final_mol, n_images = 20, friction = 0.01, dist_cutoff = 3, scaling = 1.7, sweep = None, tol = 0.002, maxiter = 15, microiter = 20):
    atom_string = initial_mol.symbols
    
    atoms = list(atom_string)

    initial_pos = [initial_mol.positions]
    final_pos = [final_mol.positions]

    total_pos = initial_pos + final_pos

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(atoms, total_pos, n_images, tol=tol * 5)

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = Geodesic(atoms, raw, scaling, threshold=dist_cutoff, friction=friction)

    if sweep is None:
        sweep = len(atoms) > 35
    try:
        if sweep:
            smoother.sweep(tol=tol, max_iter=maxiter, micro_iter=microiter)
        else:
            smoother.smooth(tol=tol, max_iter=maxiter)
    finally:
        all_atoms = []
        for atoms in smoother.path:
            mol = Atoms(numbers=initial_mol.numbers, positions=atoms, cell=[0,0,0], pbc=False)
            all_atoms.append(mol)

        return all_atoms
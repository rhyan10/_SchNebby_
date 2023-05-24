import torch
import torch.nn.functional as F

__all__ = ["cdists"]

def cdists(mols, grid):
    '''
    Calculates the pairwise Euclidean distances between a set of molecules and a list
    of positions on a grid (uses inplace operations to minimize memory demands).
    Args:
        mols (torch.Tensor): data set (of molecules) with shape
            (batch_size x n_atoms x n_dims)
        grid (torch.Tensor): array (of positions) with shape (n_positions x n_dims)
    Returns:
        torch.Tensor: batch of distance matrices (batch_size x n_atoms x n_positions)
    '''
    if len(mols.size()) == len(grid.size())+1:
        grid = grid.unsqueeze(0)  # add batch dimension
    return F.relu(torch.sum((mols[:, :, None, :] - grid[:, None, :, :]).pow_(2), -1),
                  inplace=True).sqrt_()
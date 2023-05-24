import torch

__all__ = ["gen_grid"]

def gen_grid(radial_limit, n_bins):
    n_dims = 3
    coords = torch.linspace(-radial_limit, radial_limit, n_bins)
    grid = torch.meshgrid(*[coords for _ in range(n_dims)])
    grid = torch.stack(grid, axis=-1)
    return grid

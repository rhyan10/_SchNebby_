import os

import ase

import torch
import schnetpack
import logging
from copy import deepcopy
import numpy as np
import schnetpack.task
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import CastTo32, CastTo64

from typing import Optional, List, Union, Dict

log = logging.getLogger(__name__)

__all__ = ["AtomsConverter", "DatasetLoader", "collate_fn"]


class AtomsConverterError(Exception):
    pass


class AtomsConverter:
    """
    Convert ASE atoms to SchNetPack input batch format for model prediction.
    """

    def __init__(
        self,
        n_images: int,
        neighbor_list: Union[schnetpack.transform.Transform, None],
        transforms: Union[
            schnetpack.transform.Transform, List[schnetpack.transform.Transform]
        ] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        additional_inputs: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            neighbor_list (schnetpack.transform.Transform, None): neighbor list transform. Can be set to None incase
                that the neighbor list is contained in transforms.
            transforms: transforms for manipulating the neighbor lists. This can be either a single transform or a list
                of transforms that will be executed after the neighbor list is calculated. Such transforms may be
                useful, e.g., for filtering out certain neighbors. In case transforms are required before the neighbor
                list is calculated, neighbor_list argument can be set to None and a list of transforms including the
                neighbor list can be passed as transform argument. The transforms will be executed in the order of
                their appearance in the list.
            device (str, torch.device): device on which the model operates (default: cpu).
            dtype (torch.dtype): required data type for the model input (default: torch.float32).
            additional_inputs (dict): additional inputs required for some transforms.
                When setting up the AtomsConverter, those additional inputs will be
                stored to the input batch.
        """

        self.neighbor_list = deepcopy(neighbor_list)
        self.device = device
        self.dtype = dtype
        self.n_images = n_images
        self.additional_inputs = additional_inputs or {}

        # convert transforms and neighbor_list to list
        transforms = transforms or []
        if type(transforms) != list:
            transforms = [transforms]
        neighbor_list = [] if neighbor_list is None else [neighbor_list]

        # get transforms and initialize neighbor list
        self.transforms: List[schnetpack.transform.Transform] = (
            neighbor_list + transforms
        )

        # Set numerical precision
        if dtype == torch.float32:
            self.transforms.append(CastTo32())
        elif dtype == torch.float64:
            self.transforms.append(CastTo64())
        else:
            raise AtomsConverterError(f"Unrecognized precision {dtype}")

    def __call__(self, batch):
        """
        Args:
            atoms (list or ase.Atoms): list of ASE atoms objects or single ASE atoms object.
        Returns:
            dict[str, torch.Tensor]: input batch for model.
        """
        at_idx = 0
        cumulative_sum = 0
        inter_idx = []
        input_batch = []
        mol_idx = [0]

        for int_idx, interpolation in enumerate(batch):
            
            Z = interpolation['_atomic_numbers']
            cell = interpolation['_cell']
            pbc = interpolation['_pbc']
            inter_idx.append(Z.shape[0]*self.n_images)

            for mol in interpolation['_positions']:
                cumulative_sum += Z.shape[0]
                mol_idx.append(cumulative_sum)

                inputs = {
                            properties.n_atoms: torch.tensor([Z.shape[0]]),
                            properties.Z: torch.from_numpy(Z),
                            properties.R: torch.from_numpy(mol),
                            properties.cell: torch.from_numpy(cell.array).view(-1, 3, 3),
                            properties.pbc: torch.from_numpy(pbc).view(-1, 3),
                    }

                # specify sample index
                inputs.update({properties.idx: torch.tensor([at_idx])})

                # add additional inputs (specified in AtomsConverter __init__)
                inputs.update(self.additional_inputs)

                for transform in self.transforms:
                    inputs = transform(inputs)

                at_idx+=1
                input_batch.append(inputs)
    
        inputs = _atoms_collate_fn(input_batch)
        
        for entry in inputs.keys():
            inputs[entry] = inputs[entry].to(self.device)

        inputs["_inter_idx"] = tuple(inter_idx)

        inputs["_mols_idx"] = tuple(mol_idx)

        return inputs
    

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    return batch

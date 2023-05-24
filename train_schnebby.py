from A3C_agent import Agent
import ase.io
import argparse
from neb.interpolatation import get_interpolation
import logging
import dill as pickle
from nn.distances import dist
import warnings
from utils.loader import DatasetLoader, collate_fn
import torch
warnings.filterwarnings("ignore")
from tqdm import tqdm
from ase.visualize import view
import numpy as np
from ase import Atoms
logging.basicConfig(level=logging.INFO)
import torch
from ase import units

torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_dir',
                        help='Output directory for model and training log.', 
                        default="./weights")
    parser.add_argument('--batch_size', type=int, help='Batch size',
                        default=10)
    parser.add_argument('--cutoff', type=float, help='Distance cutoff',
                        default=10)
    parser.add_argument('--basis', type=int, help='Atomwise dense layer size',
                        default=64)
    parser.add_argument('--max_epochs', type=int, help='Number of steps',
                        default=50000)
    parser.add_argument('--lr', type=float, help='Actor learning rate',
                        default=1e-4)
    parser.add_argument('--n_images', type=int, help='Number of images',
                        default=10)                        
    parser.add_argument('--gap', help='',
                        default= 0.1)
    parser.add_argument('--k', help='Spring constant',
                        default=0.1 )
    parser.add_argument('--n_bins',default=10)
    parser.add_argument('--device', help='Add device to run model on CPU/GPU',
                        default='cuda')
    parser.add_argument('--reactant_file', help='Location of reactant file', 
    			 default='data/reactant.xyz')
    parser.add_argument('--product_file', help='Location of product file', 
    			 default='data/product.xyz')
    parser.add_argument('--preopt_length', 'Length of preoptimisation',
    			 default=0)
    parser.add_argument('--episode_length', 'Length of episode',
    			 default=10)
    args = parser.parse_args()

    reactant = ase.io.read(args.reactant_file)
    product = ase.io.read(args.product_file)

    logging.info("Loading in Dataset")
    
    logging.info("Creating initial interpolations")

    data_list = []

    for i in tqdm(range(args.batch_size)):
        interpolation = get_interpolation(reactant, product, args.n_images)
        element = {"_atomic_numbers": interpolation[0], "_positions": interpolation[1], "_cell": reactant.cell, "_pbc": reactant.pbc}
        data_list.append(element)

    dataset = DatasetLoader(data_list)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )

    if args.preopt_length != 0:
        agent = Agent(args, dataloader, preopt = True)
        logging.info("Pre-optimisation")
        agent.run()

        with open("./best_path/fmax_path.pkl", "r") as file:
            path = pickle.load(file)
        
        data_list = []
        for i in tqdm(range(args.batch_size)):
            data_list.append(path)

    agent = Agent(args, dataloader)
    logging.info("Training Model")
    agent.run()


if __name__ == "__main__":
    main()

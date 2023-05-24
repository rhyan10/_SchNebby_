import torch
import torch.nn.functional as F
import argparse
import numpy as np
import random
from model import Critic_Model, Actor_Model
import dill
import schnetpack.transform as trn
import schnetpack as spk
import numpy as np
import random
from schnetpack.interfaces import AtomsConverter as SchNetConverter
from torch.optim import Adam
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('logger')
file_handler = logging.FileHandler('./logging/rewards.txt')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)


class Actor:
    def __init__(self, args):
        self.repr = repr
        self.k = args.k
        self.n_images = args.n_images
        self.batch_size = args.batch_size
        self.entropy_beta = 1
        self.n_bins = args.n_bins 
        self.max_epochs = args.max_epochs
        self.device = args.device
        self.converter = SchNetConverter(neighbor_list=trn.ASENeighborList(cutoff=10), device=self.device)
        self.schnetpack_model = torch.load('best_inference_model')
        self.schnetpack_model.to(self.device)
        self.model = Actor_Model(args.basis, args.n_images, args.n_bins, args.cutoff, args.batch_size, args.device, args.k)
        self.model.to(args.device)
        self.optimizer = Adam(self.trainable_params, args.lr)
        self.idx_grid = torch.arange(0, self.n_bins)
        self.idx_grid = self.idx_grid.repeat(self.batch_size*self.n_images,1)
        self.idx_grid = self.idx_grid.reshape(self.batch_size, self.n_images, self.n_bins).to(args.device)

    def get_action(self, forces, energies, inputs, epoch):

        p = epoch/self.max_epochs * 0.5

        results = np.random.choice([0, 1], size = self.batch_size*self.n_images, p=[1, 0])

        num_samples = np.count_nonzero(results == 1)

        r_chosen, r_dist, r_log_dist, positions, tangent_input, force_index, tangent_index = self.model(inputs, forces, energies, num_samples)
            
        return r_chosen, r_dist, r_log_dist, positions, tangent_input, force_index, tangent_index

    def compute_loss(self, r_chosen_batch, r_dist_batch, r_log_batch, advantages, step):

        policy_loss = 0
        for r_chosen_t, r_dist_t, r_log_t, ad in zip(r_chosen_batch, r_dist_batch, r_log_batch, advantages):
            ad = ad.unsqueeze(-1)
            r_chosen_t['f'] = torch.unsqueeze(r_chosen_t['f'], -1)
            diff_grid = 1/(torch.abs(self.idx_grid - r_chosen_t['f']) + torch.ones(self.idx_grid.shape).to(self.device))
            r_log_t['f'] = diff_grid * r_log_t['f']
            r_log_t['f'] = torch.squeeze(torch.sum(r_log_t['f'], dim=-1))
            f_policy = - ad * r_log_t['f'] 
            f_policy = f_policy.sum()
            
            r_chosen_t['t'] = torch.unsqueeze(r_chosen_t['t'], -1)
            diff_grid = 1/(torch.abs(self.idx_grid - r_chosen_t['t']) + torch.ones(self.idx_grid.shape).to(self.device))
            r_log_t['t'] = diff_grid * r_log_t['t']
            r_log_t['t'] = torch.squeeze(torch.sum(r_log_t['t'], dim=-1))
            t_policy = - ad * r_log_t['t'] 
            t_policy = t_policy.sum()
            
            policy_loss += t_policy+f_policy

        return policy_loss

    def dict_list_to_batch(self, dicts):
        batch = dicts[0]
        for elem in dicts[1:]:
            for i, (k, v) in enumerate(elem.items()):
                batch[k] = np.append(batch[k], v, axis=0)
        return batch

    def loss(self, r_chosen_batch, r_dist_batch, r_log_batch, advantages, epoch):
        loss = self.compute_loss(r_chosen_batch, r_dist_batch, r_log_batch, advantages, epoch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Critic:
    def __init__(self, args):
        self.repr = repr
        self.model = Critic_Model(args)
        self.trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.model.to(args.device)
        self.optimizer = Adam(self.trainable_params, args.lr)

    def compute_loss(self, v_pred, td_targets):
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(td_targets, v_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(self, v_values, td_targets):
        loss = self.compute_loss(v_values, td_targets)
        return loss

class Agent:
    def __init__(self, args, dataloader):
        self.radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=args.cutoff)
        self.cut_off_func = spk.nn.CosineCutoff(args.cutoff)
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.dataloader = dataloader
        self.args = args
        self.steps_done = 0
        self.batch_size = args.batch_size
        self.n_images = args.n_images
        self.cutoff = args.cutoff
        self.gap = args.gap
        self.k = args.k
        self.basis = args.basis
        self.max_length = args.max_length
        self.episode_length = 10
        self.update_interval = 2
        self.gamma = 1.01
        self.max_epochs = args.max_epochs
        self.env = Env(self.batch_size, self.n_images, self.cutoff, self.k, self.basis, self.max_length, args.device, args.n_bins)
        self.converter = SchNetConverter(neighbor_list=trn.ASENeighborList(cutoff=10), device=args.device)
        self.logger = logging.getLogger('Logger')
        self.logger = logging.FileHandler('log_file.txt')
        self.schnetpack_model = torch.load('best_inference_model')
        self.schnetpack_model.to(args.device)
        self.max_reward = 0

    def n_step_td_target(self, rewards, next_v_value, done, td_targets_shape):
        td_targets = torch.zeros(td_targets_shape).to(torch.float64)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma * cumulative + rewards[k].cuda()
            td_targets[k] = cumulative

        return td_targets

    def advantage(self, td_targets, baselines):
        return (td_targets - baselines)

    def tuple_list_to_batch(self, tuples):
        batch = list(tuples[0])
        for elem in tuples[1:]:
            for i, value in enumerate(elem):
                batch[i] = np.append(batch[i], value, axis=0)
        return tuple(batch)

    def list_to_batch(self, list_):
        batch = []
        for elem in list_:
            split = torch.chunk(elem, self.batch_size)
            batch = batch + list(split) 
        return batch


    def run(self):

        for epoch in tqdm(range(self.max_epochs)):
            for initial_state in self.dataloader:            
                reward_batch = []
                v_values = []
                r_chosen_batch = []
                r_dist_batch = []
                r_log_batch = []
                done = False
                counter = 0

                atoms = [atoms_ for interpolation in initial_state for atoms_ in interpolation]

                schnetpack_input = self.converter(atoms)

                results, inputs = self.schnetpack_model(schnetpack_input)

                forces = results["forces"].detach()
                energies = results["energy"].detach()

                for counter in  (range(self.episode_length)):
                    r_chosen, r_dist, r_log_dist, tangent_input, force_index, tangent_index = self.actor.get_action(forces, energies, inputs, epoch)
                    v_value = self.critic.model(inputs, tangent_input)
                    atoms, reward, forces, energies, path_energies = self.env.step(forces, energies, force_index, tangent_index, inputs, atoms, epoch, counter)
                    r_chosen_batch.append(r_chosen)
                    r_dist_batch.append(r_dist)
                    r_log_batch.append(r_log_dist)
                    reward_batch.append(reward)
                    v_values.append(v_value)

                    if counter == self.episode_length - 1:
                        done = True
                        with open("actor_model.pkl", "wb") as f:
                            dill.dump(self.actor.model, f)
                        with open("critic_model.pkl", "wb") as f:
                            dill.dump(self.critic.model, f)

                    if len(reward_batch) >= self.update_interval or done == True:

                        if done == True:
                            logger.info('Reward: '+ str(np.sum(reward.numpy())))


                        td_targets_shape = torch.stack(reward_batch, dim=0).shape

                        v_values_batch = torch.stack(v_values, dim=0)

                        td_targets = self.n_step_td_target(
                            reward_batch, v_value, done, td_targets_shape).cuda()
                        advantages = (td_targets - v_values_batch).detach()
                        advantages = advantages.unsqueeze(-1)

                        actor_loss = self.actor.loss(r_chosen_batch, r_dist_batch, r_log_batch, advantages, epoch)
                        critic_loss = self.critic.loss(v_values_batch, td_targets.detach())

                        reward_batch = []
                        v_values = []
                        r_chosen_batch = []
                        r_dist_batch = []
                        r_log_batch = []


                    counter = counter + 1

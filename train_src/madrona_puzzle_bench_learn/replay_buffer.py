import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen = True)
class Rollouts:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    bootstrap_values: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]

class NStepReplay:
    def __init__(self, manager, buffer_size, device,
                 ):
        self.buffer_size = buffer_size
        ret = create_buffer(manager=manager,
                            buffer_size=buffer_size,
                            device=device)
        self.nstep_buf_obs, self.nstep_buf_action, self.nstep_buf_log_probs, self.nstep_buf_done, self.nstep_buf_reward, self.nstep_buf_return, self.nstep_buf_value, self.nstep_buf_bootstrap_value = ret #, self.nstep_buf_rnn_start_state = ret
        self.filled_elems = 0

    @torch.no_grad()
    def add_to_buffer(self, rollout):
        for i in range(len(rollout.obs)):
            self.nstep_buf_obs[i] = self.fifo_shift(self.nstep_buf_obs[i], rollout.obs[i])
        self.nstep_buf_action = self.fifo_shift(self.nstep_buf_action, rollout.actions)
        self.nstep_buf_log_probs = self.fifo_shift(self.nstep_buf_log_probs, rollout.log_probs)
        self.nstep_buf_done = self.fifo_shift(self.nstep_buf_done, rollout.dones)
        self.nstep_buf_reward = self.fifo_shift(self.nstep_buf_reward, rollout.rewards)
        self.nstep_buf_return = self.fifo_shift(self.nstep_buf_return, rollout.returns)
        self.nstep_buf_value = self.fifo_shift(self.nstep_buf_value, rollout.values)
        self.nstep_buf_bootstrap_value = self.fifo_shift(self.nstep_buf_bootstrap_value, rollout.bootstrap_values)
        # self.fifo_shift(self.nstep_buf_rnn_start_state, rollout.rnn_start_states)
        if self.filled_elems < self.buffer_size:
            self.filled_elems += 1
        return

    @torch.no_grad()
    def get_first(self, rollout):
        obs = []
        for i in range(len(rollout.obs)):
            obs.append(self.nstep_buf_obs[i][-1])
        # rollout.rnn_start_states = self.nstep_buf_rnn_start_state[self.filled_elems-1]
        # print("Obs", obs)
        return Rollouts(obs, self.nstep_buf_action[-1], self.nstep_buf_log_probs[-1], self.nstep_buf_done[-1], self.nstep_buf_reward[-1], self.nstep_buf_return[-1], self.nstep_buf_value[-1], self.nstep_buf_bootstrap_value[-1], rollout.rnn_start_states)

    @torch.no_grad()
    def get_last(self, rollout):
        obs = []
        for i in range(len(rollout.obs)):
            obs.append(self.nstep_buf_obs[i][-self.filled_elems])
        # rollout.rnn_start_states = self.nstep_buf_rnn_start_state[self.filled_elems-1]
        print("Obs", obs)
        return Rollouts(obs, self.nstep_buf_action[-self.filled_elems], self.nstep_buf_log_probs[-self.filled_elems], self.nstep_buf_done[-self.filled_elems], self.nstep_buf_reward[-self.filled_elems], self.nstep_buf_return[-self.filled_elems], self.nstep_buf_value[-self.filled_elems], self.nstep_buf_bootstrap_value[-self.filled_elems], rollout.rnn_start_states)
    
    @torch.no_grad()
    def get_multiple(self, rollout):
        # For each var, first dimension is the different buffer entries, second dimension is 1, third dimension is batch_size
        # We want to get a random portion of each filled element
        # Such that in total we get batch_size elements

        # Generate random indices corresponding to the elements of each of the self.filled_elems batches we should grab
        indices = []
        total_elems = 0
        for i in range(self.filled_elems):
            if i < self.filled_elems - 1:
                elems_from_batch = torch.randint(0, rollout.actions.shape[2], ((int)(rollout.actions.shape[2]/self.filled_elems),), device=rollout.actions.device)
                indices.append(elems_from_batch)
                total_elems += elems_from_batch.shape[0]
            else:
                elems_from_batch = torch.randint(0, rollout.actions.shape[2], (rollout.actions.shape[2] - total_elems,), device=rollout.actions.device)
                indices.append(elems_from_batch)
                total_elems += elems_from_batch.shape[0]
        print("indices", indices)

        # Now we have a list of indices, each of which is a list of indices corresponding to the elements of each batch we should grab
        obs = []
        action = None
        log_probs = None
        done = None
        reward = None
        return_ = None
        value = None
        bootstrap_value = None
        for j in range(len(indices)):
            if j == 0:
                for i in range(len(rollout.obs)):
                    obs.append(self.nstep_buf_obs[i][-j,:,:,indices[j]])
                action = self.nstep_buf_action[-j,:,:,indices[j]]
                log_probs = self.nstep_buf_log_probs[-j,:,:,indices[j]]
                done = self.nstep_buf_done[-j,:,:,indices[j]]
                reward = self.nstep_buf_reward[-j,:,:,indices[j]]
                return_ = self.nstep_buf_return[-j,:,:,indices[j]]
                value = self.nstep_buf_value[-j,:,:,indices[j]]
                bootstrap_value = self.nstep_buf_bootstrap_value[-j,indices[j]]
            else:
                for i in range(len(rollout.obs)):
                    obs[i] = torch.cat((obs[i], self.nstep_buf_obs[i][-j,:,:,indices[j]]), dim=2)
                action = torch.cat((action, self.nstep_buf_action[-j,:,:,indices[j]]), dim=2)
                log_probs = torch.cat((log_probs, self.nstep_buf_log_probs[-j,:,:,indices[j]]), dim=2)
                done = torch.cat((done, self.nstep_buf_done[-j,:,:,indices[j]]), dim=2)
                reward = torch.cat((reward, self.nstep_buf_reward[-j,:,:,indices[j]]), dim=2)
                return_ = torch.cat((return_, self.nstep_buf_return[-j,:,:,indices[j]]), dim=2)
                value = torch.cat((value, self.nstep_buf_value[-j,:,:,indices[j]]), dim=2)
                bootstrap_value = torch.cat((bootstrap_value, self.nstep_buf_bootstrap_value[-j,indices[j]]), dim=0)

        return Rollouts(obs, action, log_probs, done, reward, return_, value, bootstrap_value, rollout.rnn_start_states)

    def fifo_shift(self, queue, new_tensor):
        queue = torch.cat((queue[1:, :], new_tensor.unsqueeze(0)), dim=0)
        return queue

def create_buffer(manager, buffer_size, device='cuda'):
    buf_obs = []
    for i in range(len(manager.obs)):
        buf_obs.append(torch.empty((buffer_size, *manager.obs[i].shape),
                          dtype=torch.float32, device=device))
    buf_action = torch.empty((buffer_size, *manager.actions.shape),
                             dtype=torch.float32, device=device)
    buf_log_probs = torch.empty((buffer_size, *manager.log_probs.shape),
                                dtype=torch.float32, device=device)
    buf_done = torch.empty((buffer_size, *manager.dones.shape),
                            dtype=torch.bool, device=device)
    buf_reward = torch.empty((buffer_size, *manager.rewards.shape),
                                dtype=torch.float32, device=device)
    buf_return = torch.empty((buffer_size, *manager.returns.shape),
                                dtype=torch.float32, device=device) 
    buf_value = torch.empty((buffer_size, *manager.values.shape),
                                dtype=torch.float32, device=device)
    buf_bootstrap_value = torch.empty((buffer_size, *manager.bootstrap_values.shape),
                                dtype=torch.float32, device=device)
    #buf_rnn_start_state = torch.empty((buffer_size, *manager.rnn_start_states.shape),
    #                            dtype=torch.float32, device=device)
    return buf_obs, buf_action, buf_log_probs, buf_done, buf_reward, buf_return, buf_value, buf_bootstrap_value#, buf_rnn_start_state

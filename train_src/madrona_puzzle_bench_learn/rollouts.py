import torch
from time import time
from dataclasses import dataclass
from typing import List, Optional
from .amp import AMPState
from .cfg import SimInterface
from .actor_critic import ActorCritic, RecurrentStateConfig
from .profile import profile
from .moving_avg import EMANormalizer

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
    # Add intrinsic rewards and values
    rewards_intrinsic: Optional[torch.Tensor] = None
    values_intrinsic: Optional[torch.Tensor] = None
    bootstrap_values_intrinsic: Optional[torch.Tensor] = None

class RolloutManager:
    def __init__(
            self,
            dev : torch.device,
            sim : SimInterface,
            steps_per_update : int,
            num_bptt_chunks : int,
            amp : AMPState,
            recurrent_cfg : RecurrentStateConfig,
            intrinsic: bool,
        ):
        self.dev = dev
        self.steps_per_update = steps_per_update
        self.num_bptt_chunks = num_bptt_chunks
        assert(steps_per_update % num_bptt_chunks == 0)
        num_bptt_steps = steps_per_update // num_bptt_chunks
        self.num_bptt_steps = num_bptt_steps

        self.need_obs_copy = sim.obs[0].device != dev

        if dev.type == 'cuda':
            float_storage_type = torch.float16
        else:
            float_storage_type = torch.bfloat16

        self.actions = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.actions.shape),
            dtype=sim.actions.dtype, device=dev)

        self.log_probs = torch.zeros(
            self.actions.shape,
            dtype=float_storage_type, device=dev)

        self.dones = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.dones.shape),
            dtype=torch.bool, device=dev)

        self.rewards = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
            dtype=float_storage_type, device=dev)

        self.returns = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
            dtype=float_storage_type, device=dev)

        self.values = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
            dtype=float_storage_type, device=dev)

        self.bootstrap_values = torch.zeros(
            sim.rewards.shape, dtype=amp.compute_dtype, device=dev)
        
        if intrinsic:
            self.rewards_intrinsic = torch.zeros(
                (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
                dtype=float_storage_type, device=dev)
            self.values_intrinsic = torch.zeros(
                (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
                dtype=float_storage_type, device=dev)
            self.bootstrap_values_intrinsic = torch.zeros(
                sim.rewards.shape, dtype=amp.compute_dtype, device=dev)
        else:
            self.rewards_intrinsic = None
            self.values_intrinsic = None
            self.bootstrap_values_intrinsic = None

        self.obs = []

        for obs_tensor in sim.obs:
            print("Appending an obs tensor")
            self.obs.append(torch.zeros(
                (num_bptt_chunks, num_bptt_steps, *obs_tensor.shape),
                dtype=obs_tensor.dtype, device=dev))

        if self.need_obs_copy:
            self.final_obs = []

            for obs_tensor in sim.obs:
                self.final_obs.append(torch.zeros(
                    obs_tensor.shape, dtype=obs_tensor.dtype, device=dev))

        self.rnn_end_states = []
        self.rnn_alt_states = []
        self.rnn_start_states = []
        for rnn_state_shape in recurrent_cfg.shapes:
            # expand shape to batch size
            batched_state_shape = (*rnn_state_shape[0:2],
                sim.actions.shape[0], rnn_state_shape[2])

            rnn_end_state = torch.zeros(
                batched_state_shape, dtype=amp.compute_dtype, device=dev)
            rnn_alt_state = torch.zeros_like(rnn_end_state)

            self.rnn_end_states.append(rnn_end_state)
            self.rnn_alt_states.append(rnn_alt_state)

            bptt_starts_shape = (num_bptt_chunks, *batched_state_shape)

            rnn_start_state = torch.zeros(
                bptt_starts_shape, dtype=amp.compute_dtype, device=dev)

            self.rnn_start_states.append(rnn_start_state)

        self.rnn_end_states = tuple(self.rnn_end_states)
        self.rnn_alt_states = tuple(self.rnn_alt_states)
        self.rnn_start_states = tuple(self.rnn_start_states)

    def collect(
            self,
            amp : AMPState,
            sim : SimInterface,
            actor_critic : ActorCritic,
            value_normalizer: EMANormalizer,
            value_normalizer_intrinsic: Optional[EMANormalizer],
        ):
        rnn_states_cur_in = self.rnn_end_states
        rnn_states_cur_out = self.rnn_alt_states

        curr_returns = torch.clone(self.returns[0,0]) # Last bptt chunk, last step
        #print("Starting returns", curr_returns.mean())

        for bptt_chunk in range(0, self.num_bptt_chunks):
            with profile("Cache RNN state"):
                # Cache starting RNN state for this chunk
                for start_state, end_state in zip(
                        self.rnn_start_states, rnn_states_cur_in):
                    start_state[bptt_chunk].copy_(end_state)

            for slot in range(0, self.num_bptt_steps):
                cur_obs_buffers = [obs[bptt_chunk, slot] for obs in self.obs]

                with profile('Policy Infer', gpu=True):
                    for obs_idx, step_obs in enumerate(sim.obs):
                        cur_obs_buffers[obs_idx].copy_(step_obs, non_blocking=True)

                    cur_actions_store = self.actions[bptt_chunk, slot]

                    #print("len(cur_obs_buffers)", len(cur_obs_buffers))

                    with amp.enable():
                        actor_critic.fwd_rollout(
                            cur_actions_store,
                            self.log_probs[bptt_chunk, slot],
                            self.values[bptt_chunk, slot],
                            rnn_states_cur_out,
                            rnn_states_cur_in,
                            self.values_intrinsic[bptt_chunk, slot] if self.values_intrinsic is not None else None,
                            self.rewards_intrinsic[bptt_chunk, slot] if self.rewards_intrinsic is not None else None,
                            *cur_obs_buffers,
                        )

                    # Invert normalized values
                    self.values[bptt_chunk, slot] = \
                        value_normalizer.invert(amp, self.values[bptt_chunk, slot])
                    if self.values_intrinsic is not None:
                        self.values_intrinsic[bptt_chunk, slot] = \
                            value_normalizer_intrinsic.invert(amp, self.values_intrinsic[bptt_chunk, slot])

                    rnn_states_cur_in, rnn_states_cur_out = \
                        rnn_states_cur_out, rnn_states_cur_in

                    # This isn't non-blocking because if the sim is running in
                    # CPU mode, the copy needs to be finished before sim.step()
                    # FIXME: proper pytorch <-> madrona cuda stream integration

                    # For now, the Policy Infer profile block ends here to get
                    # a CPU synchronization
                    sim.actions.copy_(cur_actions_store)

                with profile('Simulator Step'):
                    sim.step()

                with profile('Post Step Copy'):
                    self.rewards[bptt_chunk, slot].copy_(
                        sim.rewards, non_blocking=True)

                    cur_dones_store = self.dones[bptt_chunk, slot]
                    cur_dones_store.copy_(
                        sim.dones, non_blocking=True)

                    for rnn_states in rnn_states_cur_in:
                        rnn_states.masked_fill_(cur_dones_store, 0)



                profile.gpu_measure(sync=True)

                # Tracking returns
                returns_store = self.returns[bptt_chunk, slot]
                returns_store.copy_(curr_returns)
                curr_returns = curr_returns*(1 - self.dones[bptt_chunk, slot].float()) + self.rewards[bptt_chunk, slot]
        self.returns[0, 0].copy_(curr_returns) # For tracking the next rollout's returns

        if self.need_obs_copy:
            final_obs = self.final_obs
            for obs_idx, step_obs in enumerate(sim.obs):
                final_obs[obs_idx].copy_(step_obs, non_blocking=True)
        else:
            final_obs = sim.obs

        # rnn_hidden_cur_in and rnn_hidden_cur_out are flipped after each
        # iter so rnn_hidden_cur_in is the final output
        self.rnn_end_states = rnn_states_cur_in
        self.rnn_alt_states = rnn_states_cur_out

        with amp.enable(), profile("Bootstrap Values"):
            actor_critic.fwd_critic(
                self.bootstrap_values, None, self.rnn_end_states, *final_obs)
            self.bootstrap_values = value_normalizer.invert(amp, self.bootstrap_values)

        # Right now this just returns the rollout manager's pointers,
        # but in the future could return only one set of buffers from a
        # double buffered store, etc

        #print("Rewards intrinsic", self.rewards_intrinsic.shape)

        return Rollouts(
            obs = self.obs,
            actions = self.actions,
            log_probs = self.log_probs,
            dones = self.dones,
            rewards = self.rewards,
            returns = self.returns,
            values = self.values,
            bootstrap_values = self.bootstrap_values,
            rnn_start_states = self.rnn_start_states,
            rewards_intrinsic = self.rewards_intrinsic,
            values_intrinsic = self.values_intrinsic,
            bootstrap_values_intrinsic = self.bootstrap_values_intrinsic,
        )

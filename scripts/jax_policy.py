import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import FrozenDict

import argparse
from functools import partial
import math

import madrona_learn
from madrona_learn import (
    Policy, ActorCritic, BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
    ObservationsEMANormalizer, ObservationsCaster,
)

from madrona_learn.models import (
    LayerNorm,
    MLP,
    EntitySelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
)
from madrona_learn.rnn import LSTM

def assert_valid_input(tensor):
    checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

def assert_valid_input(tensor):
    return None
    #checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    #checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

class PolicyRNN(nn.Module):
    rnn: nn.Module
    norm: nn.Module

    @staticmethod
    def create(num_hidden_channels, num_layers, dtype, rnn_cls = LSTM):
        return PolicyRNN(
            rnn = rnn_cls(
                num_hidden_channels = num_hidden_channels,
                num_layers = num_layers,
                dtype = dtype,
            ),
            norm = LayerNorm(dtype=dtype),
        )

    @nn.nowrap
    def init_recurrent_state(self, N):
        return self.rnn.init_recurrent_state(N)

    @nn.nowrap
    def clear_recurrent_state(self, rnn_states, should_clear):
        return self.rnn.clear_recurrent_state(rnn_states, should_clear)

    def setup(self):
        pass

    def __call__(
        self,
        cur_hiddens,
        x,
        train,
    ):
        out, new_hiddens = self.rnn(cur_hiddens, x, train)
        return self.norm(out), new_hiddens

    def sequence(
        self,
        start_hiddens,
        seq_ends,
        seq_x,
        train,
    ):
        return self.norm(
            self.rnn.sequence(start_hiddens, seq_ends, seq_x, train))

class PrefixCommon(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        jax.tree_map(lambda x: assert_valid_input(x), obs)

        obs, agent_txfm = obs.pop('agent_txfm')
        obs, agent_interact = obs.pop('agent_interact')
        obs, agent_level_type = obs.pop('agent_level_type')
        obs, agent_exit = obs.pop('agent_exit')
        obs, agent_lidar_depth = obs.pop('lidar_depth')
        obs, agent_lidar_hit_type = obs.pop('lidar_hit_type')
        obs, steps_remaining = obs.pop('steps_remaining')

        def flatten_lidar(lidar):
            return lidar.reshape(*lidar.shape[0:-2], -1)

        agent_lidar_depth = flatten_lidar(agent_lidar_depth)
        agent_lidar_hit_type = flatten_lidar(agent_lidar_hit_type)

        self_ob = jnp.concatenate([
            agent_txfm,
            agent_interact,
            agent_level_type,
            agent_exit,
            agent_lidar_depth,
            agent_lidar_hit_type,
            steps_remaining,
        ], axis=-1)
        
        obs, entity_physics_states = obs.pop('entity_physics_states')
        obs, entity_types  = obs.pop('entity_types')
        obs, entity_attrs  = obs.pop('entity_attrs')

        entities_ob = jnp.concatenate([
            entity_physics_states,
            entity_types,
            entity_attrs,
        ], axis=-1)

        assert len(obs) == 0

        return FrozenDict({
            'self': self_ob, 
            'entities': entities_ob, 
        })


class SimpleNet(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        num_batch_dims = len(obs['self'].shape) - 1
        obs = jax.tree_map(
            lambda o: o.reshape(*o.shape[0:num_batch_dims], -1), obs)

        flattened, _ = jax.tree_util.tree_flatten(obs)
        flattened = jnp.concatenate(flattened, axis=-1)

        return MLP(
                num_channels = 256,
                num_layers = 3,
                dtype = self.dtype,
            )(flattened, train)


class HashNet(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        @partial(jax.vmap, in_axes=(0, None), out_axes=0)
        def simhash(x, proj):
            ys = jnp.dot(proj, x)

            @partial(jax.vmap, in_axes=(-1, -1), out_axes=-1)
            def project(i, y):
                return jnp.where(y > 0, jnp.array(2**i, jnp.int32),
                                 jnp.array(0, jnp.int32))

            return project(jnp.arange(ys.shape[-1]), ys).sum(axis=-1)

        self_ob = obs['self']
        entities_ob = obs['entities']

        obs_concat = jnp.concatenate([
            self_ob,
            entities_ob.reshape(*entities_ob.shape[:-2], -1),
        ], axis=-1)

        hash_power = 16
        num_hash_bins = 2 ** hash_power
        feature_dim = 32
        num_hashes = 16

        proj_mat = self.param('proj_mat', 
            lambda rng, shape: random.normal(rng, shape, self.dtype),
            (num_hashes, hash_power, obs_concat.shape[-1]))

        hash_val = simhash(obs_concat, proj_mat)
        hash_val = lax.stop_gradient(hash_val)

        lookup_tbl = self.param('lookup',
            jax.nn.initializers.he_normal(dtype=self.dtype),
            (num_hashes, num_hash_bins, feature_dim))

        @partial(jax.vmap, in_axes=(-1, -3), out_axes=-2)
        def lookup(k, tbl):
            return tbl[k]

        features = lookup(hash_val, lookup_tbl)
        features = features.reshape(*features.shape[:-2], -1)
        return LayerNorm(dtype=self.dtype)(features)


class ActorNet(nn.Module):
    dtype: jnp.dtype
    use_simple: bool
    use_hash: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        if self.use_simple:
            return SimpleNet(dtype=self.dtype)(obs, train)
        elif self.use_hash:
            return HashNet(dtype=self.dtype)(obs, train)
        else:
            return EntitySelfAttentionNet(
                    num_embed_channels = 128,
                    num_out_channels = 256,
                    num_heads = 4,
                    dtype = self.dtype,
                )(obs, train=train)


class CriticNet(nn.Module):
    dtype: jnp.dtype
    use_simple: bool
    use_hash: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        if self.use_simple:
            return SimpleNet(dtype=self.dtype)(obs, train)
        elif self.use_hash:
            return HashNet(dtype=self.dtype)(obs, train)
        else:
            return EntitySelfAttentionNet(
                    num_embed_channels = 128,
                    num_out_channels = 256,
                    num_heads = 4,
                    dtype = self.dtype,
                )(obs, train=train)

def make_policy(dtype):
    #encoder = RecurrentBackboneEncoder(
    encoder = BackboneEncoder(
        net = ActorNet(dtype, use_simple=False, use_hash=False),
        #rnn = PolicyRNN.create(
        #    num_hidden_channels = 256,
        #    num_layers = 1,
        #    dtype = dtype,
        #),
    )

    backbone = BackboneShared(
        prefix = PrefixCommon(
            dtype = dtype,
        ),
        encoder = encoder,
    )

    actor_critic = ActorCritic(
        backbone = backbone,
        actor = DenseLayerDiscreteActor(
            actions_num_buckets = [4, 8, 5, 2],
            dtype = dtype,
        ),
        critic = DenseLayerCritic(dtype=dtype),
    )

    obs_preprocess = ObservationsEMANormalizer.create(
        decay = 0.99999,
        dtype = dtype,
        prep_fns = {},
        skip_normalization = {},
    )

    def get_episode_scores(episode_result):
        return episode_result[0]

    return Policy(
        actor_critic = actor_critic,
        obs_preprocess = obs_preprocess,
        get_episode_scores = get_episode_scores,
    )

    return policy

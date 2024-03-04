from madrona_puzzle_bench_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder, RNDModel
)

from madrona_puzzle_bench_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from madrona_puzzle_bench_learn.rnn import LSTM

import math
import torch

def setup_obs(sim):
    # First define all the observation tensors
    agent_txfm_obs_tensor = sim.agent_txfm_obs_tensor().to_torch() # Scalars
    agent_interact_obs_tensor = sim.agent_interact_obs_tensor().to_torch() # Bool, whether or not grabbing
    agent_level_type_obs_tensor = sim.agent_level_type_obs_tensor().to_torch() # Enum
    agent_exit_obs_tensor = sim.agent_exit_obs_tensor().to_torch() # Scalars
    entity_physics_state_obs_tensor = sim.entity_physics_state_obs_tensor().to_torch() # Scalars
    entity_type_obs_tensor = sim.entity_type_obs_tensor().to_torch() # Enum
    entity_attr_obs_tensor = sim.entity_attr_obs_tensor().to_torch() # Enum, but I don't know max
    lidar_depth_tensor = sim.lidar_depth_tensor().to_torch() # Scalars
    lidar_hit_type_tensor = sim.lidar_hit_type().to_torch() # Enum (EntityType)
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch() # Int but dont' need to convert

    # print all shapes for debugging purposes
    '''
    print("Agent txfm obs", agent_txfm_obs_tensor.shape)
    print("Agent interact obs", agent_interact_obs_tensor.shape)
    print("Agent level type obs", agent_level_type_obs_tensor.shape)
    print("Agent exit obs", agent_exit_obs_tensor.shape)
    print("Entity physics state obs", entity_physics_state_obs_tensor.shape)
    print("Entity type obs", entity_type_obs_tensor.shape)
    print("Entity attr obs", entity_attr_obs_tensor.shape)
    print("Lidar depth", lidar_depth_tensor.shape)
    print("Lidar hit type", lidar_hit_type_tensor.shape)
    print("Steps remaining", steps_remaining_tensor.shape)
    '''

    # Now process the tensors into a form good for the policy
    # Convert enums to one-hot
    agent_level_type_obs_tensor_onehot = torch.nn.functional.one_hot(agent_level_type_obs_tensor.to(torch.int64), num_classes=6)
    entity_type_obs_tensor_onehot = torch.nn.functional.one_hot(entity_type_obs_tensor.to(torch.int64), num_classes=11)
    entity_attr_obs_tensor_onehot = entity_attr_obs_tensor #torch.nn.functional.one_hot(entity_attr_obs_tensor.to(torch.int64), num_classes=2) # TODO: correct
    lidar_hit_type_tensor_onehot = torch.nn.functional.one_hot(lidar_hit_type_tensor.to(torch.int64), num_classes=11)

    # Print all the changed object shapes
    '''
    print("Agent level type obs", agent_level_type_obs_tensor.shape)
    print("Entity type obs", entity_type_obs_tensor.shape)
    print("Entity attr obs", entity_attr_obs_tensor.shape)
    print("Lidar hit type", lidar_hit_type_tensor.shape)
    '''

    # Reshape all non-entity observations to be (batch_size, -1)
    agent_txfm_obs_tensor = agent_txfm_obs_tensor.view(agent_txfm_obs_tensor.shape[0], -1)
    agent_interact_obs_tensor = agent_interact_obs_tensor.view(agent_interact_obs_tensor.shape[0], -1)
    agent_level_type_obs_tensor_onehot = agent_level_type_obs_tensor_onehot.view(agent_level_type_obs_tensor.shape[0], -1)
    agent_exit_obs_tensor = agent_exit_obs_tensor.view(agent_exit_obs_tensor.shape[0], -1)
    lidar_depth_tensor = lidar_depth_tensor.view(lidar_depth_tensor.shape[0], -1)
    lidar_hit_type_tensor_onehot = lidar_hit_type_tensor_onehot.view(lidar_hit_type_tensor.shape[0], -1)
    steps_remaining_tensor = steps_remaining_tensor.view(steps_remaining_tensor.shape[0], -1)

    # Reshape all entity observations to be (batch_size, num_entities, -1)
    entity_physics_state_obs_tensor = entity_physics_state_obs_tensor.view(entity_physics_state_obs_tensor.shape[0], entity_physics_state_obs_tensor.shape[1], -1)
    entity_type_obs_tensor_onehot = entity_type_obs_tensor_onehot.view(entity_type_obs_tensor.shape[0], entity_type_obs_tensor.shape[1], -1)
    entity_attr_obs_tensor_onehot = entity_attr_obs_tensor_onehot.view(entity_attr_obs_tensor.shape[0], entity_attr_obs_tensor.shape[1], -1)

    # Concatenate all non-entity observations
    obs_tensor = torch.cat([
        agent_txfm_obs_tensor,
        agent_interact_obs_tensor,
        agent_level_type_obs_tensor_onehot,
        agent_exit_obs_tensor,
        lidar_depth_tensor,
        lidar_hit_type_tensor_onehot,
        steps_remaining_tensor,
    ], dim=1)

    # Concatenate all entity observations
    entity_tensor = torch.cat([
        entity_physics_state_obs_tensor,
        entity_type_obs_tensor_onehot,
        entity_attr_obs_tensor_onehot,
    ], dim=2)

    num_obs_features = obs_tensor.shape[1] 
    num_entity_features = entity_tensor.shape[2]

    '''
    print("Obs tensor", obs_tensor.shape)
    print("Entity tensor", entity_tensor.shape)
    '''

    obs_list = [
        agent_txfm_obs_tensor,
        agent_interact_obs_tensor,
        agent_level_type_obs_tensor,
        agent_exit_obs_tensor,
        entity_physics_state_obs_tensor,
        entity_type_obs_tensor,
        entity_attr_obs_tensor,
        lidar_depth_tensor,
        lidar_hit_type_tensor,
        steps_remaining_tensor,
    ]

    return obs_list, num_obs_features + num_entity_features*entity_tensor.shape[1]

def process_obs(agent_txfm_obs_tensor, agent_interact_obs_tensor, agent_level_type_obs_tensor, agent_exit_obs_tensor, entity_physics_state_obs_tensor, entity_type_obs_tensor, entity_attr_obs_tensor, lidar_depth_tensor, lidar_hit_type_tensor, steps_remaining_tensor):

    agent_level_type_obs_tensor_onehot = torch.nn.functional.one_hot(agent_level_type_obs_tensor.to(torch.int64), num_classes=6)
    entity_type_obs_tensor_onehot = torch.nn.functional.one_hot(entity_type_obs_tensor.to(torch.int64), num_classes=11)
    entity_attr_obs_tensor_onehot = entity_attr_obs_tensor*0 + 1 #torch.nn.functional.one_hot(entity_attr_obs_tensor.to(torch.int64), num_classes=2) # TODO: correct
    lidar_hit_type_tensor_onehot = torch.nn.functional.one_hot(lidar_hit_type_tensor.to(torch.int64), num_classes=11)

    # Print all the changed object shapes
    '''
    print("Agent level type obs", agent_level_type_obs_tensor.shape)
    print("Entity type obs", entity_type_obs_tensor.shape)
    print("Entity attr obs", entity_attr_obs_tensor.shape)
    print("Lidar hit type", lidar_hit_type_tensor.shape)
    '''

    # Reshape all non-entity observations to be (batch_size, -1)
    agent_txfm_obs_tensor = agent_txfm_obs_tensor.view(agent_txfm_obs_tensor.shape[0], -1)
    agent_interact_obs_tensor = agent_interact_obs_tensor.view(agent_interact_obs_tensor.shape[0], -1)
    agent_level_type_obs_tensor_onehot = agent_level_type_obs_tensor_onehot.view(agent_level_type_obs_tensor.shape[0], -1)
    agent_exit_obs_tensor = agent_exit_obs_tensor.view(agent_exit_obs_tensor.shape[0], -1)
    lidar_depth_tensor = lidar_depth_tensor.view(lidar_depth_tensor.shape[0], -1)
    lidar_hit_type_tensor_onehot = lidar_hit_type_tensor_onehot.view(lidar_hit_type_tensor.shape[0], -1)
    steps_remaining_tensor = steps_remaining_tensor.view(steps_remaining_tensor.shape[0], -1)

    # Reshape all entity observations to be (batch_size, num_entities, -1)
    entity_physics_state_obs_tensor = entity_physics_state_obs_tensor.view(entity_physics_state_obs_tensor.shape[0], entity_physics_state_obs_tensor.shape[1], -1)
    entity_type_obs_tensor_onehot = entity_type_obs_tensor_onehot.view(entity_type_obs_tensor.shape[0], entity_type_obs_tensor.shape[1], -1)
    entity_attr_obs_tensor_onehot = entity_attr_obs_tensor_onehot.view(entity_attr_obs_tensor.shape[0], entity_attr_obs_tensor.shape[1], -1)

    # Concatenate all non-entity observations
    obs_tensor = torch.cat([
        agent_txfm_obs_tensor,
        agent_interact_obs_tensor,
        agent_level_type_obs_tensor_onehot,
        agent_exit_obs_tensor,
        lidar_depth_tensor,
        lidar_hit_type_tensor_onehot,
        steps_remaining_tensor,
    ], dim=1)

    # Concatenate all entity observations
    entity_tensor = torch.cat([
        entity_physics_state_obs_tensor,
        entity_type_obs_tensor_onehot,
        entity_attr_obs_tensor_onehot,
    ], dim=2)

    # Combine obs and entity tensors
    combined_tensor = torch.cat([obs_tensor, entity_tensor.view(entity_tensor.shape[0], -1)], dim=1)
    #print("Combined tensor", combined_tensor.shape)
    #print("Entity tensor shape", entity_tensor.shape)

    # If the combined tensor has nans, first print the positions of the nans, then raise an error
    if torch.isnan(combined_tensor).any():
        print("Nans in combined tensor", torch.where(torch.isnan(combined_tensor)))
        # Do this separately for obs and entity tensors
        print("Nans in obs tensor", torch.where(torch.isnan(obs_tensor)))
        print("Nans in entity tensor", torch.where(torch.isnan(entity_tensor)))
        raise ValueError("Nans in combined tensor")

    # Filter nans
    #combined_tensor[torch.isnan(combined_tensor)] = 0

    return combined_tensor

def make_policy(num_obs_features, num_channels, separate_value, intrinsic=False):
    #encoder = RecurrentBackboneEncoder(
    #    net = MLP(
    #        input_dim = num_obs_features,
    #        num_channels = num_channels,
    #        num_layers = 2,
    #    ),
    #    rnn = LSTM(
    #        in_channels = num_channels,
    #        hidden_channels = num_channels,
    #        num_layers = 1,
    #    ),
    #)

    encoder = BackboneEncoder(
        net = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 3,
        ),
    )

    if separate_value:
        backbone = BackboneSeparate(
            process_obs = process_obs,
            actor_encoder = encoder,
            critic_encoder = RecurrentBackboneEncoder(
                net = MLP(
                    input_dim = num_obs_features,
                    num_channels = num_channels,
                    num_layers = 2,
                ),
                rnn = LSTM(
                    in_channels = num_channels,
                    hidden_channels = num_channels,
                    num_layers = 1,
                ),
            )
        )
    else:
        backbone = BackboneShared(
            process_obs = process_obs,
            encoder = encoder,
        )

    if intrinsic:
        # Add the intrinsic reward module
        critic_intrinsic = LinearLayerCritic(num_channels)
        rnd_model = RNDModel(
            process_obs = process_obs,
            target_net = MLP(
                input_dim = num_obs_features,
                num_channels = num_channels,
                num_layers = 4, # VISHNU TODO: try options, original was convs + 1
            ),
            predictor_net = MLP(
                input_dim = num_obs_features,
                num_channels = num_channels,
                num_layers = 1, # VISHNU TODO: try options, original was convs + 3
            ),
        )
    else:
        critic_intrinsic = None
        rnd_model = None

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [4, 8, 5, 3],
            num_channels,
        ),
        critic = LinearLayerCritic(num_channels),
        critic_intrinsic = critic_intrinsic,
        rnd_model = rnd_model,
    )

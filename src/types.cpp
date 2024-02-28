#include "types.hpp"
#include "sim.hpp"

namespace madPuzzle {

using namespace madrona;

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);

    render::RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerComponent<Action>();

    registry.registerComponent<AgentTxfmObs>();
    registry.registerComponent<AgentInteractObs>();
    registry.registerComponent<AgentLevelTypeObs>();
    registry.registerComponent<AgentExitObs>();

    registry.registerComponent<EntityPhysicsStateObsArray>();
    registry.registerComponent<EntityTypeObsArray>();
    registry.registerComponent<EntityAttributesObsArray>();
    registry.registerComponent<LidarDepth>();
    registry.registerComponent<LidarHitType>();

    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<AgentPolicy>();

    registry.registerComponent<GrabState>();
    registry.registerComponent<Progress>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<OpenState>();
    registry.registerComponent<DoorButtons>();
    registry.registerComponent<DoorRooms>();
    registry.registerComponent<StepsRemainingObservation>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<EntityExtents>();

    registry.registerComponent<DeferredDelete>();

    registry.registerComponent<RoomAABB>();
    registry.registerComponent<RoomListElem>();

    registry.registerComponent<ButtonListElem>();

    registry.registerComponent<IsExit>();
    registry.registerComponent<IsLava>();
    registry.registerComponent<EnemyState>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<Level>();
    registry.registerSingleton<EpisodeState>();

    // Checkpoint state.
    registry.registerSingleton<Checkpoint>();
    registry.registerSingleton<CheckpointReset>();
    registry.registerSingleton<CheckpointSave>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<DoorEntity>();
    registry.registerArchetype<ButtonEntity>();
    registry.registerArchetype<RoomEntity>();
    registry.registerArchetype<ExitEntity>();
    registry.registerArchetype<EnemyEntity>();
    registry.registerArchetype<LavaEntity>();

    registry.registerArchetype<DeferredDeleteEntity>();

    registry.exportSingleton<Checkpoint>(
        (uint32_t)ExportID::Checkpoint);
    registry.exportSingleton<CheckpointReset>(
        (uint32_t)ExportID::CheckpointReset);
    registry.exportSingleton<CheckpointSave>(
        (uint32_t)ExportID::CheckpointSave);
    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);

    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);

    registry.exportColumn<Agent, AgentTxfmObs>(
        (uint32_t)ExportID::AgentTxfmObs);
    registry.exportColumn<Agent, AgentInteractObs>(
        (uint32_t)ExportID::AgentInteractObs);
    registry.exportColumn<Agent, AgentLevelTypeObs>(
        (uint32_t)ExportID::AgentLevelTypeObs);
    registry.exportColumn<Agent, AgentExitObs>(
        (uint32_t)ExportID::AgentExitObs);
    registry.exportColumn<Agent, StepsRemainingObservation>(
        (uint32_t)ExportID::StepsRemaining);

    registry.exportColumn<Agent, EntityPhysicsStateObsArray>(
        (uint32_t)ExportID::EntityPhysicsStateObsArray);
    registry.exportColumn<Agent, EntityTypeObsArray>(
        (uint32_t)ExportID::EntityTypeObsArray);
    registry.exportColumn<Agent, EntityAttributesObsArray>(
        (uint32_t)ExportID::EntityAttributesObsArray);

    registry.exportColumn<Agent, LidarDepth>(
        (uint32_t)ExportID::LidarDepth);
    registry.exportColumn<Agent, LidarHitType>(
        (uint32_t)ExportID::LidarHitType);

    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);

    registry.exportColumn<Agent, AgentPolicy>(
        (uint32_t)ExportID::AgentPolicy);
}

}

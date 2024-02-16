#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"

namespace madPuzzle {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::Query;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;
using madrona::math::Vector3;
using madrona::math::AABB;


// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

enum class LevelType : uint32_t {
    Chase,
    LavaPath,
    SingleButton,
    SingleBlockButton,
    NumTypes,
};

// Discrete action component. Ranges are defined by consts::numMoveBuckets (5),
// repeated here for clarity
struct Action {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t rotate; // [-2, 2]
    int32_t interact; // 0 = do nothing, 1 = grab / release, 2 = jump
};

// Per-agent reward
// Exported as an [N * A, 1] float tensor to training code
struct Reward {
    float v;
};

// Per-agent component that indicates that the agent's episode is finished
// This is exported per-agent for simplicity in the training code
struct Done {
    // Currently bool components are not supported due to
    // padding issues, so Done is an int32_t
    int32_t v;
};

struct AgentTxfmObs {
    Vector3 localRoomPos;
    AABB roomAABB;
    float theta;
};

struct AgentInteractObs {
    int32_t isGrabbing;
};

struct AgentLevelTypeObs {
    LevelType t;
};

// The state of the world is passed to each agent in terms of egocentric
// polar coordinates. theta is degrees off agent forward.
struct PolarObs {
    float r;
    float theta;
    float phi;
};

// Per-agent egocentric observations for the interactable entities
// in the current room.
struct EntityPhysicsStateObs {
    PolarObs positionPolar;
    PolarObs velocityPolar;
    Vector3 entityRotation;
};

struct EntityTypeObs {
    RoomEntityType entityType;
};

struct EntityAttributesObs {
    int32_t attr1;
    int32_t attr2;
};

struct EntityPhysicsStateObsArray {
    EntityPhysicsStateObs obs[consts::maxObservationsPerAgent];
};

struct EntityTypeObsArray {
    EntityTypeObs obs[consts::maxObservationsPerAgent];
};

struct EntityAttributesObsArray {
    EntityAttributesObs obs[consts::maxObservationsPerAgent];
};

// Linear depth values and entity type in a circle around the agent
struct LidarDepth {
    float samples[consts::numLidarSamples];
};

struct LidarHitType {
    RoomEntityType samples[consts::numLidarSamples];
};

// Number of steps remaining in the episode. Allows non-recurrent policies
// to track the progression of time.
struct StepsRemaining {
    uint32_t t;
};

// Tracks progress the agent has made through the challenge, used to add
// reward when more progress has been made
struct Progress {
    float maxY;
};

// Tracks if an agent is currently grabbing another entity
struct GrabState {
    Entity constraintEntity;
};

// This enum is used to track the type of each entity for the purposes of
// classifying the objects hit by each lidar sample.
enum class EntityType : uint32_t {
    None,
    Agent,
    Door,
    Block,
    Ramp,
    Lava,
    Button,
    Key,
    Enemy,
    Wall,
    NumTypes,
};

// A per-door component that tracks whether or not the door should be open.
struct OpenState {
    bool isOpen;
};

struct RoomAABB : AABB {
    inline RoomAABB(AABB aabb) : AABB(aabb) {}
}

// Linked buttons that control the door opening and whether or not the door
// should remain open after the buttons are pressed once.
struct DoorProperties {
    Entity linkedButton;
    Entity linkedRooms[2];
    bool    isPersistent;
};

// Similar to OpenState, true during frames where a button is pressed
struct ButtonState {
    bool isPressed;
};

struct Level {
    Entity startRoom;
};

struct EntityLinkedListElem {
    Entity next;
};

struct Checkpoint {
    // Checkpoint structs.
    struct ButtonSaveState {
        Position p;
        Rotation r;
        ButtonState b;
    };
    
    struct PhysicsEntityState {
        Position p;
        Rotation r;
        Velocity v;
    };
    
    struct DoorState {
        Position p;
        Rotation r;
        Velocity v;
        OpenState o;
    };
    
    struct AgentState {
        Position p;
        Rotation r;
        Velocity v;
        // Index within the checkpoint buffers of the
        // grabbed entity. -1 if not grabbing.
        int32_t grabIdx;
        Reward re;
        Done d;
        StepsRemaining s;
        Progress pr;
        madrona::phys::JointConstraint j;
        KeyCode k;
    };

    madrona::RandKey initRNDCounter;
    ButtonSaveState buttonStates[consts::maxRooms * 2 * 4];
    PhysicsEntityState cubeStates[consts::maxRooms * 3];
    DoorState doorStates[consts::maxRooms * 4];
    AgentState agentStates[consts::numAgents];
};

struct CheckpointReset {
    int32_t reset;
};

// For connection to the viewer.
struct CheckpointSave {
    int32_t save;
};

struct RoomEntity : public madrona::Archetype<
    RoomAABB,
    EntityLinkedListElem
> {};

/* ECS Archetypes for the game */

// There are 2 Agents in the environment trying to get to the destination
struct Agent : public madrona::Archetype<
    // Basic components required for physics. Note that the current physics
    // implementation requires archetypes to have these components first
    // in this exact order.
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,

    // Internal logic state.
    GrabState,
    Progress,
    EntityType,
    KeyCode,

    // Input
    Action,

    // Observations
    SelfObservation,
    RoomEntityObservations,
    RoomDoorObservations,
    Lidar,
    StepsRemaining,

    // Reward, episode termination
    Reward,
    Done,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    madrona::render::RenderCamera,
    // All entities with the Renderable component will be drawn by the
    // viewer and batch renderer
    madrona::render::Renderable
> {};

// Archetype for the doors blocking the end of each challenge room
struct DoorEntity : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    OpenState,
    DoorProperties,
    EntityType,
    EntityLinkedListElem,
    madrona::render::Renderable
> {};

// Archetype for the button objects that open the doors
// Buttons don't have collision but are rendered
struct ButtonEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    ButtonState,
    EntityType,
    EntityLinkedListElem,
    madrona::render::Renderable
> {};

// Generic archetype for entities that need physics but don't have custom
// logic associated with them.
struct PhysicsEntity : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    EntityType,
    EntityLinkedListElem,
    madrona::render::Renderable
> {};

}

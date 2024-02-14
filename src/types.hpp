#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"

namespace madEscape {

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


// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
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

// Observation state for the current agent.
// Positions are rescaled to the bounds of the play area to assist training.
struct SelfObservation {
    float roomX;
    float roomY;
    float globalX;
    float globalY;
    float globalZ;
    float maxY;
    float theta;
    float isGrabbing;
    float keyCode;
};

struct AgentID {
    int32_t id;
};

// The state of the world is passed to each agent in terms of egocentric
// polar coordinates. theta is degrees off agent forward.
struct PolarObservation {
    float r;
    float theta;
};

struct PartnerObservation {
    PolarObservation polar;
    float isGrabbing;
};

// Egocentric observations of other agents
struct PartnerObservations {
    PartnerObservation obs[consts::numAgents - 1];
};

// PartnerObservations is exported as a
// [N, A, consts::numAgents - 1, 3] // tensor to pytorch
static_assert(sizeof(PartnerObservations) == sizeof(float) *
    (consts::numAgents - 1) * 3);

// Per-agent egocentric observations for the interactable entities
// in the current room.
struct EntityObservation {
    PolarObservation polar;
    float encodedType;
};

struct RoomEntityObservations {
    EntityObservation obs[consts::maxObservationsPerAgent];
};

// RoomEntityObservations is exported as a
// [N, A, maxObservationsPerAgent, 3] tensor to pytorch
static_assert(sizeof(RoomEntityObservations) == sizeof(float) *
    consts::maxObservationsPerAgent * 3);

// Observation of the current room's door. It's relative position and
// whether or not it is ope
struct DoorObservation {
    PolarObservation polar;
    float isOpen; // 1.0 when open, 0.0 when closed, -1.0 when nonexistent.
};

struct RoomDoorObservations {
    DoorObservation obs[consts::doorsPerRoom];
};

struct LidarSample {
    float depth;
    float encodedType;
};

// Linear depth values and entity type in a circle around the agent
struct Lidar {
    LidarSample samples[consts::numLidarSamples];
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

// Per-agent component storing Entity IDs of the other agents. Used to
// build the egocentric observations of their state.
struct OtherAgents {
    madrona::Entity e[consts::numAgents - 1];
};

// Tracks if an agent is currently grabbing another entity
struct GrabState {
    Entity constraintEntity;
};

// This enum is used to track the type of each entity for the purposes of
// classifying the objects hit by each lidar sample.
enum class EntityType : uint32_t {
    None,
    Button,
    Cube,
    Wall,
    Agent,
    Door,
    Key,
    NumTypes,
};

// A per-door component that tracks whether or not the door should be open.
struct OpenState {
    bool isOpen;
};

// Linked buttons that control the door opening and whether or not the door
// should remain open after the buttons are pressed once.
struct DoorProperties {
    Entity  buttons[consts::maxEntitiesPerRoom];
    int32_t numButtons;
    bool    isPersistent;
    bool    isExit;
};

// Similar to OpenState, true during frames where a button is pressed
struct ButtonState {
    bool isPressed;
};

struct KeyCode {
    int32_t code;
};

// False before an agent claims the key, true after.
struct KeyState {
    bool claimed;
    KeyCode code; // Which door the key opens.
};

struct RoomRep {
    int32_t x;
    int32_t y;
    Entity door;
};

struct DoorRep {
    Entity door;
    int32_t roomIdx;
    int32_t code;
    bool exit;
};

// Room itself is not a component but is used by the singleton
// component "LevelState" (below) to represent the state of the full level
struct Room {
    // These are entities the agent will interact with
    Entity entities[consts::maxEntitiesPerRoom];

    // The walls that separate this room from the next
    Entity walls[consts::wallsPerRoom];

    // The door the agents need to figure out how to lower
    Entity door[consts::doorsPerRoom];
};

// A singleton component storing the state of all the rooms in the current
// randomly generated level
struct LevelState {
    Room rooms[consts::maxRooms];
};

// Checkpoint structs.
struct ButtonSaveState {
    Position p;
    Rotation r;
    ButtonState b;
};

struct KeySaveState {
    Position p;
    Rotation r;
    KeyState k;
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
    KeyCode k;
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

struct Checkpoint {
    madrona::RandKey initRNDCounter;
    ButtonSaveState buttonStates[consts::maxRooms * 2 * 4];
    PhysicsEntityState cubeStates[consts::maxRooms * 3];
    DoorState doorStates[consts::maxRooms * 4];
    AgentState agentStates[consts::numAgents];
    KeySaveState keyStates[consts::maxRooms * 4];
};

struct CheckpointReset {
    int32_t reset;
};

// For connection to the viewer.
struct CheckpointSave {
    int32_t save;
};

struct RoomCount {
    int32_t count;
};

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
    OtherAgents,
    EntityType,
    KeyCode,

    // Input
    Action,

    // Observations
    SelfObservation,
    AgentID,
    PartnerObservations,
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
    KeyCode,
    DoorProperties,
    EntityType,
    madrona::render::Renderable
> {};

// Archetype for the key objects that allow doors to be opened
// Keys don't have collision but are rendered
struct KeyEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    KeyState,
    EntityType,
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
    madrona::render::Renderable
> {};

}

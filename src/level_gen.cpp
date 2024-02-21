#include "level_gen.hpp"
#include <iostream>

namespace madPuzzle {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace {

enum class WallType : uint32_t {
    None,
    Solid,
    Entrance,
    Door,
};

struct WallConfig {
    WallType type;
    float entranceT = FLT_MAX;
};

struct RoomList {
    RoomListElem *cur;

    static inline RoomList init(RoomListElem *start)
    {
        start->next = Entity::none();

        return RoomList {
            .cur = start,
        };
    }

    inline Entity add(Engine &ctx)
    {
        Entity e = ctx.makeEntity<RoomEntity>();

        cur->next = e;

        cur = &ctx.get<RoomListElem>(e);
        cur->next = Entity::none();
        
        return e;
    }
};
}

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.sampleUniform() * range - range / 2.f;
}

static inline float randBetween(Engine &ctx, float min, float max)
{
    return ctx.data().rng.sampleUniform() * (max - min) + min;
}

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) =
        RigidBodyPhysicsSystem::registerEntity(ctx, e, obj_id);
}

// Agent entity persists across all episodes.
Entity createAgent(Engine &ctx)
{
    Entity agent = ctx.makeRenderableEntity<Agent>();

    // Create a render view for the agent
    if (ctx.data().enableRender) {
        render::RenderingSystem::attachEntityToView(ctx,
                agent,
                100.f, 0.001f,
                1.5f * math::up);
    }

    ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
    ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
    ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
    ctx.get<GrabState>(agent).constraintEntity = Entity::none();
    ctx.get<EntityType>(agent) = EntityType::Agent;

    return agent;
}

// Agents need to be re-registered them with the broadphase system and
// have their transform / physics state reset to spawn.
void resetAgent(Engine &ctx,
                Vector3 spawn_pos,
                float spawn_size,
                Vector3 exit_pos)
{
    Entity agent_entity = ctx.data().agent;
    registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

    float safe_spawn_range = spawn_size / 2.f - consts::agentRadius;
    spawn_pos.x += randInRangeCentered(ctx, safe_spawn_range);
    spawn_pos.y += randInRangeCentered(ctx, safe_spawn_range);

    ctx.get<Position>(agent_entity) = spawn_pos;
    ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
        randInRangeCentered(ctx, math::pi / 4.f),
        math::up);

    auto &grab_state = ctx.get<GrabState>(agent_entity);
    if (grab_state.constraintEntity != Entity::none()) {
        ctx.destroyEntity(grab_state.constraintEntity);
        grab_state.constraintEntity = Entity::none();
    }

    ctx.get<Progress>(agent_entity) = Progress {
        .minDistToExit = spawn_pos.distance(exit_pos),
    };

    ctx.get<Velocity>(agent_entity) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
    ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
    ctx.get<Action>(agent_entity) = Action {
        .moveAmount = 0,
        .moveAngle = 0,
        .rotate = consts::numTurnBuckets / 2,
        .interact = 0,
    };
}

static void linkDoorButtons(Engine &ctx,
                            Entity door,
                            Span<const Entity> buttons,
                            bool is_persistent)
{
    DoorButtons &door_buttons = ctx.get<DoorButtons>(door);

    ButtonListElem *button_list_elem = &door_buttons.linkedButton;

    auto linkButton = [&](Entity button) {
        button_list_elem->next = button;
        button_list_elem = &ctx.get<ButtonListElem>(button);
        button_list_elem->next = Entity::none();
    };

    for (CountT i = 0; i < buttons.size(); i++) {
        linkButton(buttons[i]);
    }
    door_buttons.isPersistent = is_persistent;
}

static Entity makeButton(Engine &ctx,
                         float button_x,
                         float button_y)
{
    const float button_width = ctx.data().buttonWidth;

    Entity button = ctx.makeRenderableEntity<ButtonEntity>();

    setupRigidBodyEntity(
        ctx,
        button,
        Vector3 {
            button_x,
            button_y,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Button,
        EntityType::Button,
        ResponseType::Static,
        Diag3x3 {
            button_width,
            button_width,
            1.f,
        });
    registerRigidBodyEntity(ctx, button, SimObject::Button);

    ctx.get<ButtonState>(button).isPressed = false;

    ctx.get<EntityExtents>(button) = Vector3 {
        .x = button_width,
        .y = button_width,
        .z = 0.25f,
    };

    return button;
}

static Entity makeBlock(Engine &ctx,
                       float block_x,
                       float block_y,
                       float block_size = 1.f)
{
    Entity block = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        block,
        Vector3 {
            block_x,
            block_y,
            block_size / 2.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Block,
        EntityType::Block,
        ResponseType::Dynamic,
        Diag3x3 {
            block_size,
            block_size,
            block_size,
        });
    registerRigidBodyEntity(ctx, block, SimObject::Block);

    ctx.get<EntityExtents>(block) = Vector3 {
        .x = block_size,
        .y = block_size,
        .z = block_size,
    };

    return block;
}

static Entity makeDoor(Engine &ctx,
                       Vector3 door_center,
                       Vector3 len_axis,
                       Vector3 width_axis,
                       const float door_height = 1.75f)

{
    const float door_obj_len = 0.8f * ctx.data().doorWidth;

    Vector3 door_dims = width_axis * consts::wallWidth +
        math::up * door_height + len_axis * door_obj_len;

    Entity door = ctx.makeRenderableEntity<DoorEntity>();

    setupRigidBodyEntity(
        ctx,
        door,
        door_center,
        Quat { 1, 0, 0, 0 },
        SimObject::Door,
        EntityType::Door,
        ResponseType::Static,
        Diag3x3::fromVec(door_dims));
    registerRigidBodyEntity(ctx, door, SimObject::Door);
    ctx.get<OpenState>(door).isOpen = false;
    ctx.get<DoorButtons>(door) = {
        .linkedButton = { Entity::none() },
        .isPersistent = true,
    };

    ctx.get<EntityExtents>(door) = door_dims;

    return door;
}

static Entity makeWall(Engine &ctx, Vector3 center, Diag3x3 scale)
{
    Entity wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        wall,
        center,
        Quat { 1, 0, 0, 0 },
        SimObject::Wall, 
        EntityType::Wall,
        ResponseType::Static,
        scale);
    registerRigidBodyEntity(ctx, wall, SimObject::Wall);

    return wall;
}

static void makeRoomWalls(Engine &ctx,
                          AABB room_aabb,
                          Span<const WallConfig> wall_cfgs,
                          Entity *door_entities,
                          Vector3 *entrance_positions)
{
    auto makeSolidWall = [
        &ctx
    ](Vector3 p0, Vector3 p1)
    {
        Vector3 diff = p1 - p0;
        assert(diff.z == 0);

        Diag3x3 scale;
        if (diff.x == 0) {
            scale.d0 = consts::wallWidth;
            scale.d1 = diff.y + consts::wallWidth;
        } else if (diff.y == 0) {
            scale.d0 = diff.x + consts::wallWidth;
            scale.d1 = consts::wallWidth;
        } else {
            assert(false);
        }
        scale.d2 = 2.f;

        Vector3 center = (p0 + p1) / 2.f;

        makeWall(ctx, center, scale);
    };

    const float entrance_width = ctx.data().doorWidth;
    auto makeEntranceWall = [
        &ctx, entrance_width
    ](Vector3 p0, Vector3 p1, float entrance_t,
      Entity *door_entity_out, Vector3 *entrance_pos_out)
    {
        Vector3 diff = p1 - p0;
        assert(diff.z == 0.f);

        float wall_len;
        Vector3 len_axis;
        Vector3 width_axis;
        if (diff.x == 0) {
            wall_len = diff.y;
            len_axis = math::fwd;
            width_axis = math::right;
        } else if (diff.y == 0) {
            wall_len = diff.x;
            len_axis = math::right;
            width_axis = math::fwd;
        } else {
            assert(false);
        }

        float entrance_padding = 0.65f * entrance_width;
        float safe_entrance_len = wall_len - 2.f * entrance_padding;

        float entrance_dist = entrance_padding + safe_entrance_len * entrance_t;

        Vector3 entrance_center = p0 + len_axis * entrance_dist;
        *entrance_pos_out = entrance_center;

        Vector3 before_entrance =
            entrance_center - len_axis * entrance_width / 2.f;
        Vector3 after_entrance =
            entrance_center + len_axis * entrance_width / 2.f;

        if (door_entity_out) {
            *door_entity_out = makeDoor(ctx, entrance_center, len_axis, width_axis);
        }

        Vector3 before_center = (p0 + before_entrance) / 2.f;
        Vector3 before_dims = consts::wallWidth * width_axis + 
            Diag3x3::fromVec(len_axis) * 
                (before_entrance - p0 + consts::wallWidth / 2.f) +
            2.f * math::up;

        before_center -= len_axis * consts::wallWidth / 4.f;

        makeWall(ctx, before_center, Diag3x3::fromVec(before_dims));

        Vector3 after_center = (after_entrance + p1) / 2.f;
        Vector3 after_dims = consts::wallWidth * width_axis + 
            Diag3x3::fromVec(len_axis) *
                (p1 - after_entrance + consts::wallWidth / 2.f) +
            2.f * math::up;

        after_center += len_axis * consts::wallWidth / 4.f;

        makeWall(ctx, after_center, Diag3x3::fromVec(after_dims));
    };

    const Vector3 corners[4] {
        Vector3(room_aabb.pMin.x, room_aabb.pMax.y, room_aabb.pMin.z),
        Vector3(room_aabb.pMax.x, room_aabb.pMax.y, room_aabb.pMin.z),
        Vector3(room_aabb.pMax.x, room_aabb.pMin.y, room_aabb.pMin.z),
        room_aabb.pMin,
    };

    auto makeSide = [&]
    (CountT side_idx, CountT corner_i, CountT corner_j)
    {
        WallConfig wall_cfg = wall_cfgs[side_idx];

        if (wall_cfg.type == WallType::None) {
            return;
        }

        Vector3 cur_corner = corners[corner_i];
        Vector3 next_corner = corners[corner_j];

        bool make_door = wall_cfg.type == WallType::Door;

        if (make_door || wall_cfg.type == WallType::Entrance) {
            makeEntranceWall(
                cur_corner, next_corner, wall_cfg.entranceT,
                make_door ? &door_entities[side_idx] : nullptr,
                &entrance_positions[side_idx]);
        } else {
            makeSolidWall(cur_corner, next_corner);
        }
    };

    makeSide(0, 0, 1);
    makeSide(1, 2, 1);
    makeSide(2, 3, 2);
    makeSide(3, 3, 0);
}

static void makeFloor(Engine &ctx)
{
    // Create the floor entity, just a simple static plane.
    Entity floor_plane = ctx.makeRenderableEntity<PhysicsEntity>();

    setupRigidBodyEntity(
        ctx,
        floor_plane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Plane,
        EntityType::None, // Floor plane type should never be queried
        ResponseType::Static);

    registerRigidBodyEntity(ctx, floor_plane, SimObject::Plane);
}

static void makeSpawn(Engine &ctx, float spawn_size, Vector3 spawn_pos)
{
    makeWall(ctx, 
        spawn_pos + Vector3 {
            0.f,
            -spawn_size / 2.f,
            0.f,
        },
        {
            spawn_size + consts::wallWidth,
            consts::wallWidth,
            2.f,
        });

    makeWall(ctx,
        spawn_pos + Vector3 {
            -spawn_size / 2.f,
            0.f,
            0.f,
        },
        {
            consts::wallWidth,
            spawn_size,
            2.f,
        });

    makeWall(ctx,
        spawn_pos + Vector3 {
            spawn_size / 2.f,
            0.f,
            0.f,
        },
        {
            consts::wallWidth,
            spawn_size,
            2.f,
        });
}

static Entity makeExit(Engine &ctx, float room_size, Vector3 exit_pos)
{
    Entity e = ctx.makeRenderableEntity<ExitEntity>();
    ctx.get<Position>(e) = exit_pos;
    ctx.get<Rotation>(e) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(e) = Diag3x3 { 1, 1, 1 };
    ctx.get<ObjectID>(e) = ObjectID { (int32_t)SimObject::Exit };

    makeWall(ctx,
        exit_pos + Vector3 {
            0.f,
            room_size / 2.f,
            0.f,
        },
        {
            room_size + consts::wallWidth,
            consts::wallWidth,
            2.f,
        });

    makeWall(ctx,
        exit_pos + Vector3 {
            -room_size / 2.f,
            0.f,
            0.f,
        },
        {
            consts::wallWidth,
            room_size,
            2.f,
        });

    makeWall(ctx,
        exit_pos + Vector3 {
            room_size / 2.f,
            0.f,
            0.f,
        },
        {
            consts::wallWidth,
            room_size,
            2.f,
        });

    return e;
}

static Entity makeEnemy(Engine &ctx, Vector3 position, 
                        float move_force = 200.f,
                        float move_torque = 200.f)
{
    Entity enemy = ctx.makeRenderableEntity<EnemyEntity>();
    setupRigidBodyEntity(
        ctx,
        enemy,
        position,
        Quat::angleAxis(randBetween(ctx, 0, 2.f * math::pi), math::up),
        SimObject::Enemy,
        EntityType::Enemy,
        ResponseType::Dynamic,
        Diag3x3 { 1, 1, 1 });
    registerRigidBodyEntity(ctx, enemy, SimObject::Enemy);

    ctx.get<EnemyState>(enemy) = EnemyState {
        .moveForce = move_force,
        .moveTorque = move_torque,
    };

    ctx.get<EntityExtents>(enemy) = Vector3 {
        2.f * consts::agentRadius,
        2.f * consts::agentRadius,
        2.f * consts::agentRadius,
    };

    return enemy;
}

static void setupSingleRoomLevel(Engine &ctx,
                                 float level_size,
                                 Entity *exit_door_out,
                                 AABB *room_aabb_out)
{
    const float spawn_size = 1.5f * ctx.data().doorWidth;
    const float half_wall_width = consts::wallWidth / 2.f;

    makeFloor(ctx);

    AABB room_aabb = {
        .pMin = Vector3 { -level_size / 2.f, 0.f, 0.f },
        .pMax = Vector3 { level_size / 2.f, level_size, 2.f },
    };

    float exit_t = ctx.data().rng.sampleUniform();
    float spawn_t = ctx.data().rng.sampleUniform();

    Entity doors[4];
    Vector3 entrance_positions[4];
    makeRoomWalls(ctx, room_aabb,
        {
            { exit_door_out ? WallType::Door : WallType::Entrance, exit_t },
            { WallType::Solid },
            { WallType::Entrance, spawn_t },
            { WallType::Solid },
        }, doors, entrance_positions);

    Vector3 exit_pos = entrance_positions[0];
    Vector3 spawn_pos = entrance_positions[2];

    spawn_pos.y -= spawn_size / 2.f;

    makeSpawn(ctx, spawn_size, spawn_pos);

    resetAgent(ctx, spawn_pos, spawn_size, exit_pos);

    exit_pos.y += spawn_size / 2.f;

    Level &level = ctx.singleton<Level>();
    level.exit = makeExit(ctx, spawn_size, exit_pos);

    RoomList room_list = RoomList::init(&level.rooms);

    // Save a larger AABB in the room list that contains
    // the exit and spawn rooms so they're picked up
    // for observations
    AABB obs_aabb = room_aabb;
    obs_aabb.pMax.y += spawn_size;
    obs_aabb.pMin.y -= spawn_size;

    // Grow downwards so closed doors are still in obs
    obs_aabb.pMin.z -= 1.f;

    Entity room = room_list.add(ctx);
    ctx.get<RoomAABB>(room) = obs_aabb;

    // Shrink returned AABB to account for walls
    AABB safe_aabb = room_aabb;
    safe_aabb.pMin.x += half_wall_width;
    safe_aabb.pMin.y += half_wall_width;
    safe_aabb.pMax.x -= half_wall_width;
    safe_aabb.pMax.y -= half_wall_width;

    *room_aabb_out = safe_aabb;

    if (exit_door_out) {
        Entity exit_door = doors[0];
        ctx.get<DoorRooms>(exit_door) = {
            .linkedRooms = { room, Entity::none() },
        };

        *exit_door_out = exit_door;
    }
}

static void chaseLevel(Engine &ctx)
{
    const float level_size = 20.f;

    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, nullptr, &room_aabb);

    Vector3 enemy_spawn = room_aabb.pMin;
    enemy_spawn.y += level_size / 2.f;
    enemy_spawn.y += randBetween(ctx, 0.f, room_aabb.pMax.y - enemy_spawn.y);
    enemy_spawn.x = randBetween(ctx, room_aabb.pMin.x, room_aabb.pMax.x);

    makeEnemy(ctx, enemy_spawn);
}

static void lavaPathLevel(Engine &ctx)
{
    const float level_size = 20.f;

    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, nullptr, &room_aabb);
}

static void singleButtonLevel(Engine &ctx)
{
    const float level_size = 20.f;

    const float button_width = ctx.data().buttonWidth;
    const float half_button_width = button_width / 2.f;

    Entity exit_door;
    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb);

    {
        float button_x = randBetween(ctx,
            room_aabb.pMin.x + half_button_width,
            room_aabb.pMax.x - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_button_width,
            room_aabb.pMax.y - half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);

        linkDoorButtons(ctx, exit_door, { button }, true);
    }
}

static void singleBlockButtonLevel(Engine &ctx)
{
    const float level_size = 20.f;
    const float block_size = 3;

    const float button_width = ctx.data().buttonWidth;

    const float half_button_width = button_width / 2.f;
    const float half_block_size = block_size / 2.f;

    Entity exit_door;
    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb);

    {
        float button_x = randBetween(ctx,
            room_aabb.pMin.x + half_button_width,
            room_aabb.pMax.x - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_button_width,
            room_aabb.pMax.y - half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);

        linkDoorButtons(ctx, exit_door, { button }, false);
    }

    {
        float block_x = randBetween(ctx,
            room_aabb.pMin.x + half_block_size,
            room_aabb.pMax.x - half_block_size);

        float block_y = randBetween(ctx,
            room_aabb.pMin.y + block_size,
            room_aabb.pMax.y - block_size);

        makeBlock(ctx, block_x, block_y, block_size);
    }
}

static void obstructedBlockButtonLevel(Engine &ctx)
{
    const float level_size = 20.f;
    const float block_size = 3.f;

    const float button_width = ctx.data().buttonWidth;

    const float half_button_width = button_width / 2.f;
    const float half_block_size = block_size / 2.f;

    Entity exit_door;
    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb);

    float button_min_x = 
        0.25f * room_aabb.pMin.x + 0.75f * room_aabb.pMax.x;
    {
        float button_x = randBetween(ctx,
            button_min_x,
            room_aabb.pMax.x - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_button_width,
            room_aabb.pMax.y - half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);

        linkDoorButtons(ctx, exit_door, { button }, false);
    }

    {
        float block_x = randBetween(ctx,
            // Agent needs to be able to get in to get block out
            room_aabb.pMin.x +
                consts::agentRadius * 2.5f + block_size, 
            button_min_x - half_block_size - consts::wallWidth);

        float block_y = randBetween(ctx,
            room_aabb.pMin.y + block_size + consts::wallWidth,
            room_aabb.pMax.y - block_size - consts::wallWidth);

        makeBlock(ctx, block_x, block_y, block_size);

        makeWall(ctx, {
            block_x + block_size / 2.f + consts::wallWidth / 2.f,
            block_y,
            0.f,
        },
        {
            consts::wallWidth,
            block_size + consts::wallWidth * 2.f,
            2.f,
        });

        makeWall(ctx, {
            block_x,
            block_y + block_size / 2.f + consts::wallWidth / 2.f,
            0.f,
        },
        {
            block_size,
            consts::wallWidth,
            2.f,
        });

        makeWall(ctx, {
            block_x,
            block_y - block_size / 2.f - consts::wallWidth / 2.f,
            0.f,
        },
        {
            block_size,
            consts::wallWidth,
            2.f,
        });
    }
}

LevelType generateLevel(Engine &ctx)
{
    LevelType level_type = (LevelType)(4); // (LevelType) (
        //ctx.data().rng.sampleI32(0, (uint32_t)LevelType::NumTypes));

    switch (level_type) {
    case LevelType::Chase: {
        chaseLevel(ctx);
    } break;
    case LevelType::LavaPath: {
        lavaPathLevel(ctx);
    } break;
    case LevelType::SingleButton: {
        singleButtonLevel(ctx);
    } break;
    case LevelType::SingleBlockButton: {
        singleBlockButtonLevel(ctx);
    } break;
    case LevelType::ObstructedBlockButton: {
        obstructedBlockButtonLevel(ctx);
    } break;
    default: MADRONA_UNREACHABLE();
    }

    return level_type;
}

void destroyLevel(Engine &ctx)
{
    ctx.iterateQuery(ctx.data().simEntityQuery, [
        &ctx
    ](Entity e, EntityType type)
    {
        if (type == EntityType::Agent) {
            return;
        }

        Loc l = ctx.makeTemporary<DeferredDeleteEntity>();
        ctx.get<DeferredDelete>(l).e = e;
    });

    Level &lvl = ctx.singleton<Level>();
    {
        Entity cur_room = lvl.rooms.next;

        while (cur_room != Entity::none()) {
            Entity next_room = ctx.get<RoomListElem>(cur_room).next;

            ctx.destroyEntity(cur_room);

            cur_room = next_room;
        }

        lvl.rooms.next = Entity::none();
    }

    ctx.destroyRenderableEntity(lvl.exit);
}

}

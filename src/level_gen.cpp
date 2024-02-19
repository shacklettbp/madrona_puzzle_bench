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
static void resetAgent(Engine &ctx, Vector3 spawn_pos, Vector3 exit_pos)
{
    Entity agent_entity = ctx.data().agent;
    registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

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
    button_list_elem->next = Entity::none();

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
    const float button_height = 0.2f;

    Entity button = ctx.makeRenderableEntity<ButtonEntity>();
    ctx.get<Position>(button) = Vector3 {
        button_x,
        button_y,
        0.f,
    };
    ctx.get<Rotation>(button) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(button) = Diag3x3 {
        button_width,
        button_width,
        button_height,
    };
    ctx.get<ObjectID>(button) = ObjectID { (int32_t)SimObject::Button };
    ctx.get<ButtonState>(button).isPressed = false;
    ctx.get<EntityType>(button) = EntityType::Button;

    ctx.get<EntityExtents>(button) = Vector3 {
        .x = button_width,
        .y = button_width,
        .z = button_height,
    };

    return button;
}

static Entity makeBlock(Engine &ctx,
                       float block_x,
                       float block_y,
                       float scale = 1.f)
{
    Entity block = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        block,
        Vector3 {
            block_x,
            block_y,
            1.f * scale,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Block,
        EntityType::Block,
        ResponseType::Dynamic,
        Diag3x3 {
            scale,
            scale,
            scale,
        });
    registerRigidBodyEntity(ctx, block, SimObject::Block);

    ctx.get<EntityExtents>(block) = Vector3 {
        .x = 2.f * scale,
        .y = 2.f * scale,
        .z = 2.f * scale,
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

    ctx.get<EntityExtents>(door) = door_dims;

    return door;
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
    };

    auto makeEntranceWall = [
        &ctx
    ](Vector3 p0, Vector3 p1, float entrance_t,
      Entity *door_entity_out, Vector3 *entrance_pos_out)
    {
        const float entrance_width = ctx.data().doorWidth;

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

        float entrance_padding = 0.55f * entrance_width;
        float safe_entrance_len = wall_len - 2 * entrance_padding;

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

        Entity before_wall = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            before_wall,
            before_center,
            Quat { 1, 0, 0, 0 },
            SimObject::Wall, 
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3::fromVec(before_dims));
        registerRigidBodyEntity(ctx, before_wall, SimObject::Wall);


        Vector3 after_center = (after_entrance + p1) / 2.f;
        Vector3 after_dims = consts::wallWidth * width_axis + 
            Diag3x3::fromVec(len_axis) *
                (p1 - after_entrance + consts::wallWidth / 2.f) +
            2.f * math::up;

        after_center += len_axis * consts::wallWidth / 4.f;

        Entity after_wall = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            after_wall,
            after_center,
            Quat { 1, 0, 0, 0 },
            SimObject::Wall, 
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3::fromVec(after_dims));
        registerRigidBodyEntity(ctx, after_wall, SimObject::Wall);
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
    Entity back_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        back_wall,
        spawn_pos + Vector3 {
            0.f,
            -spawn_size / 2.f,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            spawn_size + consts::wallWidth,
            consts::wallWidth,
            2.f,
        });
    registerRigidBodyEntity(ctx, back_wall, SimObject::Wall);

    Entity left_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        spawn_pos + Vector3 {
            -spawn_size / 2.f,
            0.f,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall, 
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            spawn_size,
            2.f,
        });
    registerRigidBodyEntity(ctx, left_wall, SimObject::Wall);

    Entity right_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        spawn_pos + Vector3 {
            spawn_size / 2.f,
            0.f,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall, 
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            spawn_size,
            2.f,
        });
    registerRigidBodyEntity(ctx, right_wall, SimObject::Wall);
}

static void chaseLevel(Engine &ctx)
{
    Vector3 exit_door_pos { 0, 5, 0 };

    Level &level = ctx.singleton<Level>();
    level.exitPos = exit_door_pos;

    makeFloor(ctx);
    resetAgent(ctx, Vector3::zero(), exit_door_pos);

    RoomList room_list = RoomList::init(&level.rooms);

    Entity room = room_list.add(ctx);
    ctx.get<RoomAABB>(room) = AABB::invalid();
}

static void lavaPathLevel(Engine &ctx)
{
    (void)ctx;
}

static void singleButtonLevel(Engine &ctx)
{
    (void)ctx;
}

static void singleBlockButtonLevel(Engine &ctx)
{
    const float level_size = 20.f;
    const float block_scale = 1.5f;
    const float block_size = 2.f * block_scale;

    const float half_wall_width = consts::wallWidth;
    const float button_width = ctx.data().buttonWidth;
    const float half_button_width = button_width / 2.f;

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
                      { WallType::Door, exit_t },
                      { WallType::Solid },
                      { WallType::Entrance, spawn_t },
                      { WallType::Solid },
                  }, doors, entrance_positions);

    Level &level = ctx.singleton<Level>();
    level.exitPos = entrance_positions[0];

    const float spawn_size = 1.2f * ctx.data().doorWidth;

    Vector3 spawn_pos = entrance_positions[2];
    spawn_pos.y -= spawn_size / 2.f;

    makeSpawn(ctx, spawn_size, spawn_pos);
    resetAgent(ctx, spawn_pos, level.exitPos);

    RoomList room_list = RoomList::init(&level.rooms);

    Entity room = room_list.add(ctx);
    ctx.get<RoomAABB>(room) = room_aabb;

    Entity exit_door = doors[0];
    ctx.get<DoorRooms>(exit_door) = {
        .linkedRooms = { room, Entity::none() },
    };

    {
        float button_x = randBetween(ctx,
            room_aabb.pMin.x + half_wall_width + half_button_width,
            room_aabb.pMax.x - half_wall_width - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_wall_width + half_button_width,
            room_aabb.pMax.y - half_wall_width - half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);

        linkDoorButtons(ctx, exit_door, { button }, false);
    }

    {
        float block_x = randBetween(ctx,
            room_aabb.pMin.x + half_wall_width + block_size / 2.f,
            room_aabb.pMax.x - half_wall_width - block_size / 2.f);

        float block_y = randBetween(ctx,
            room_aabb.pMin.y + half_wall_width + block_size / 2.f,
            room_aabb.pMax.y - half_wall_width - block_size / 2.f);

        makeBlock(ctx, block_x, block_y, 1.5f);
    }
}

LevelType generateLevel(Engine &ctx)
{
    LevelType level_type = (LevelType)(
        ctx.data().rng.sampleI32(0, (uint32_t)LevelType::NumTypes));

    level_type = LevelType::SingleBlockButton;

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

    {
        Entity cur_room = ctx.singleton<Level>().rooms.next;

        while (cur_room != Entity::none()) {
            Entity next_room = ctx.get<RoomListElem>(cur_room).next;

            ctx.destroyEntity(cur_room);

            cur_room = next_room;
        }
    }
}

}

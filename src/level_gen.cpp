#include "level_gen.hpp"
#include <iostream>

namespace madPuzzle {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

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

static void makeRoomWalls(Engine &ctx,
                          AABB room_aabb,
                          const float *n_door,
                          const float *e_door,
                          const float *s_door,
                          const float *w_door,
                          Vector3 *n_door_pos_out,
                          Vector3 *e_door_pos_out,
                          Vector3 *s_door_pos_out,
                          Vector3 *w_door_pos_out)
{
    Vector2 sw_corner = { room_aabb.pMin.x, room_aabb.pMin.y };
    Vector2 ne_corner = { room_aabb.pMax.x, room_aabb.pMax.y };

    auto makeFullWall = [
        &ctx
    ](Vector2 p0, Vector2 p1)
    {
        Vector2 diff = p1 - p0;

        Vector2 scale;
        if (diff.x == 0) {
            scale.x = consts::wallWidth;
            scale.y = diff.y;
        } else if (diff.y == 0) {
            scale.x = diff.x;
            scale.y = consts::wallWidth;
        }

        Vector2 center = (p0 + p1) / 2.f;

        Entity wall = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            wall,
            Vector3 {
                center.x,
                center.y,
                0.f,
            },
            Quat { 1, 0, 0, 0 },
            SimObject::Wall, 
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3 {
                scale.x,
                scale.y,
                2.f,
            });
        registerRigidBodyEntity(ctx, wall, SimObject::Wall);
    };

    auto makeDoorWall = [
        &ctx
    ](Vector2 p0, Vector2 p1, float door_t, Vector3 *out_door_center)
    {
        const float door_width = ctx.data().doorWidth;

        Vector2 diff = p1 - p0;

        float wall_len;
        bool y_axis;
        if (diff.x == 0) {
            wall_len = diff.y;
            y_axis = true;
        } else if (diff.y == 0) {
            wall_len = diff.x;
            y_axis = false;
        } else {
            assert(false);
        }

        float door_dist = 0.25f * door_width +
            door_t * (wall_len - 0.5f * door_width);

        float before_len = door_dist - 0.75f * door_width;
        float after_len = wall_len - door_dist - 0.5f * door_width;

        Vector2 door_center = p0;
        Vector2 before_center = p0;
        if (y_axis) {
            before_center.y += before_len / 2.f;
            door_center.y += door_dist;
        } else {
            before_center.x += before_len / 2.f;
            door_center.x += door_dist;
        }

        if (out_door_center != nullptr) {
            *out_door_center = {
                door_center.x,
                door_center.y,
                0.f,
            };
        }

        Entity before_wall = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            before_wall,
            Vector3 {
                before_center.x,
                before_center.y,
                0.f,
            },
            Quat { 1, 0, 0, 0 },
            SimObject::Wall, 
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3 {
                y_axis ? consts::wallWidth : before_len,
                y_axis ? before_len : consts::wallWidth,
                2.f,
            });
        registerRigidBodyEntity(ctx, before_wall, SimObject::Wall);

        Vector2 after_center = p0;
        if (y_axis) {
            after_center.y -= after_len / 2.f;
        } else {
            after_center.x -= after_len / 2.f;
        }

        Entity after_wall = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            after_wall,
            Vector3 {
                after_center.x,
                after_center.y,
                0.f,
            },
            Quat { 1, 0, 0, 0 },
            SimObject::Wall, 
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3 {
                y_axis ? consts::wallWidth : after_len,
                y_axis ? after_len : consts::wallWidth,
                2.f,
            });
        registerRigidBodyEntity(ctx, after_wall, SimObject::Wall);
    };

    Vector2 corners[4] {
        Vector2(sw_corner.x, ne_corner.y),
        ne_corner,
        Vector2(ne_corner.x, sw_corner.y),
        sw_corner,
    };

    if (n_door != nullptr) {
        makeDoorWall(corners[0], corners[1], *n_door,
                     n_door_pos_out);
    } else {
        makeFullWall(corners[0], corners[1]);
    }

    if (e_door != nullptr) {
        makeDoorWall(corners[1], corners[2], *e_door,
                     e_door_pos_out);
    } else {
        makeFullWall(corners[1], corners[2]);
    }

    if (s_door != nullptr) {
        makeDoorWall(corners[2], corners[3], *s_door,
                     s_door_pos_out);
    } else {
        makeFullWall(corners[2], corners[3]);
    }

    if (w_door != nullptr) {
        makeDoorWall(corners[3], corners[0], *w_door,
                     w_door_pos_out);
    } else {
        makeFullWall(corners[3], corners[0]);
    }
}

static Entity makeDoor(Engine &ctx,
                     Vector3 door_center,
                     bool y_axis)

{
    const float door_obj_width = 0.8f * ctx.data().doorWidth;

    const float door_x_len = 
        y_axis ? door_obj_width : consts::wallWidth;

    const float door_y_len = 
        y_axis ? consts::wallWidth : door_obj_width;

    const float door_height = 1.75f;

    Entity door = ctx.makeRenderableEntity<DoorEntity>();
    setupRigidBodyEntity(
        ctx,
        door,
        door_center,
        Quat { 1, 0, 0, 0 },
        SimObject::Door,
        EntityType::Door,
        ResponseType::Static,
        Diag3x3 {
            door_x_len,
            door_y_len,
            door_height,
        });
    registerRigidBodyEntity(ctx, door, SimObject::Door);
    ctx.get<OpenState>(door).isOpen = false;

    ctx.get<EntityExtents>(door) = Vector3 {
        .x = door_x_len,
        .y = door_y_len,
        .z = door_height,
    };

    return door;
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

static void makeSpawn(Engine &ctx, Vector3 spawn_pos)
{
    const float spawn_size = ctx.data().doorWidth + 1.5f;

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
            spawn_size,
            consts::wallWidth,
            2.f,
        });
    registerRigidBodyEntity(ctx, back_wall, SimObject::Wall);

    Entity left_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        spawn_pos + Vector3 {
            -spawn_size,
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
    registerRigidBodyEntity(ctx, back_wall, SimObject::Wall);

    Entity right_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        spawn_pos + Vector3 {
            spawn_size,
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
    registerRigidBodyEntity(ctx, back_wall, SimObject::Wall);
}

static void chaseLevel(Engine &ctx)
{
    (void)ctx;
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

    AABB room_aabb = {
        .pMin = Vector3 { -level_size / 2.f, 0.f, 0.f },
        .pMax = Vector3 { level_size / 2.f, level_size, 2.f },
    };

    float spawn_t = ctx.data().rng.sampleUniform();
    float exit_t = ctx.data().rng.sampleUniform();

    Vector3 spawn_door_pos;
    Vector3 exit_door_pos;
    makeRoomWalls(ctx,
                  room_aabb,
                  &spawn_t, nullptr, &exit_t, nullptr,
                  &spawn_door_pos, nullptr, &exit_door_pos, nullptr);

    Level &level = ctx.singleton<Level>();
    level.exitPos = exit_door_pos;

    Vector3 spawn_pos = spawn_door_pos;
    spawn_pos.y -= ctx.data().doorWidth / 2.f;

    makeSpawn(ctx, spawn_pos);
    resetAgent(ctx, spawn_pos, exit_door_pos);

    RoomList room_list = RoomList::init(&level.rooms);

    Entity room = room_list.add(ctx);
    ctx.get<RoomAABB>(room) = room_aabb;

    Entity exit_door = makeDoor(ctx, exit_door_pos, false);
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

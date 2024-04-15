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
    ctx.get<broadphase::LeafID>(e) = PhysicsSystem::registerEntity(ctx, e, obj_id);
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

    // if no PBT, policy needs to be set explicitly to 0 to pick up the fake
    // RewardHyperParams allocated
    ctx.get<AgentPolicy>(agent).policyIdx = 0;

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
        .minDistToButton = 0.f,
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

static Entity addButtonToList(Engine &ctx,
                              Entity tail,
                              Entity button)
{
    ButtonListElem &cur_elem = ctx.get<ButtonListElem>(tail);
    cur_elem.next = button;
    return button;
}

static void terminateButtonList(Engine &ctx,
                                Entity tail)
{
    ButtonListElem &cur_elem = ctx.get<ButtonListElem>(tail);
    cur_elem.next = Entity::none();
}

static Entity addPatternToList(Engine &ctx,
                              Entity tail,
                              Entity button)
{
    PatternListElem &cur_elem = ctx.get<PatternListElem>(tail);
    cur_elem.next = button;
    return button;
}

static void terminatePatternList(Engine &ctx,
                                Entity tail)
{
    PatternListElem &cur_elem = ctx.get<PatternListElem>(tail);
    cur_elem.next = Entity::none();
}

static Entity addChickenToList(Engine &ctx,
                              Entity tail,
                              Entity chicken)
{
    ChickenListElem &cur_elem = ctx.get<ChickenListElem>(tail);
    cur_elem.next = chicken;
    return chicken;
}

static void terminateChickenList(Engine &ctx,
                                Entity tail)
{
    ChickenListElem &cur_elem = ctx.get<ChickenListElem>(tail);
    cur_elem.next = Entity::none();
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

static Entity makePattern(Engine &ctx,
                           float x,
                           float y,
                           float z)
{
    Entity pattern = ctx.makeRenderableEntity<PatternEntity>();

    ctx.get<Position>(pattern) = Vector3 { x, y, z };
    ctx.get<Scale>(pattern) = Diag3x3 { 1, 1, 1 };

    ctx.get<PatternMatchState>(pattern).isMatched = false;

    return pattern;
}

static Entity makeBlock(Engine &ctx,
                       float block_x,
                       float block_y,
                       float block_size = 1.f,
                       bool is_fixed = false)
{
    Entity block = ctx.makeRenderableEntity<PhysicsEntity>();
    ResponseType response_type = is_fixed ? ResponseType::Static : ResponseType::Dynamic;
    EntityType entity_type = is_fixed ? EntityType::FixedBlock : EntityType::Block;
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
        entity_type,
        response_type,
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
                       const float door_height = 2.75f)

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

    ctx.get<DoorProperties>(door) = {
        .isPersistent = false,
    };

    // Initialize these linked lists as empty
    // so we don't need to worry about forgetting
    // to do this in one of the level types.
    terminateButtonList(ctx, door);
    terminatePatternList(ctx, door);
    terminateChickenList(ctx, door);

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

static Entity makeCoop(Engine &ctx, Vector3 center, Diag3x3 scale)
{
    Entity coop = ctx.makeRenderableEntity<CoopEntity>();

    setupRigidBodyEntity(
        ctx,
        coop,
        Vector3 {
            center.x,
            center.y,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Button,
        EntityType::Coop,
        ResponseType::Static,
        Diag3x3 {
            scale.d0,
            scale.d1,
            1.f,
        });
    registerRigidBodyEntity(ctx, coop, SimObject::Button);

    ctx.get<EntityExtents>(coop) = Vector3 {
        .x = scale.d0,
        .y = scale.d1,
        .z = 0.05f,
    };

    return coop;
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
        scale.d2 = 3.f;

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
            3.f * math::up;

        before_center -= len_axis * consts::wallWidth / 4.f;

        makeWall(ctx, before_center, Diag3x3::fromVec(before_dims));

        Vector3 after_center = (after_entrance + p1) / 2.f;
        Vector3 after_dims = consts::wallWidth * width_axis + 
            Diag3x3::fromVec(len_axis) *
                (p1 - after_entrance + consts::wallWidth / 2.f) +
            3.f * math::up;

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

static Entity makeLava(Engine &ctx, Vector3 position, 
                        Scale scale = Diag3x3{1.0f, 1.0f, 0.01f})
{
    Entity lava = ctx.makeRenderableEntity<LavaEntity>();
    setupRigidBodyEntity(
        ctx,
        lava,
        position,
        Quat { 1, 0, 0, 0 },
        SimObject::Lava,
        EntityType::Lava,
        ResponseType::Static,
        scale);
    registerRigidBodyEntity(ctx, lava, SimObject::Lava);

    ctx.get<EntityExtents>(lava) = Vector3 {
        scale.d0,
        scale.d1,
        scale.d2
    };

    return lava;
}

static Entity makeEnemy(Engine &ctx, Vector3 position, 
                        float move_force = 160.f,
                        float move_torque = 160.f,
                        bool isChicken = false)
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
        .isChicken = isChicken,
        .isDead = false,
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
                                 AABB *room_aabb_out,
                                 Vector3* entrance_positions,
                                 bool hop_wall = false,
                                 bool exit_door_is_persistent = false)
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
    Vector3 local_entrance_positions[4];

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
        scale.d2 = 3.f; // 1 higher than num blocks should work

        Vector3 center = (p0 + p1) / 2.f;

        makeWall(ctx, center, scale);
    };

    if (hop_wall){
        makeSolidWall(
            Vector3(room_aabb.pMin.x, room_aabb.pMax.y, room_aabb.pMin.z),
            Vector3(room_aabb.pMax.x, room_aabb.pMax.y, room_aabb.pMin.z)
        );
    } 
    makeRoomWalls(ctx, room_aabb,
        {
            { exit_door_out ? WallType::Door : WallType::Entrance, exit_t },
            { WallType::Solid },
            { WallType::Entrance, spawn_t },
            { WallType::Solid },
        }, doors, local_entrance_positions);


    if (entrance_positions != nullptr) {
        for (int i = 0; i < 4; ++i) {
            entrance_positions[i] = local_entrance_positions[i];
        }
    }

    Vector3 exit_pos = local_entrance_positions[0];
    Vector3 spawn_pos = local_entrance_positions[2];

    spawn_pos.y -= spawn_size / 2.f;

    makeSpawn(ctx, spawn_size, spawn_pos);

    exit_pos.y += spawn_size / 2.f;

    Level &level = ctx.singleton<Level>();
    level.exit = makeExit(ctx, spawn_size, exit_pos);

    resetAgent(ctx, spawn_pos, spawn_size, exit_pos);

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

        ctx.get<DoorProperties>(exit_door).isPersistent =
            exit_door_is_persistent;

        *exit_door_out = exit_door;
    }
}

static void chaseLevel(Engine &ctx)
{
    const float level_size = 20.f;

    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, nullptr, &room_aabb, nullptr);

    Vector3 enemy_spawn = room_aabb.pMin;
    enemy_spawn.y += level_size / 2.f;
    enemy_spawn.y += randBetween(ctx, 0.f, room_aabb.pMax.y - enemy_spawn.y);
    enemy_spawn.x = randBetween(ctx, room_aabb.pMin.x, room_aabb.pMax.x);

    makeEnemy(ctx, enemy_spawn, 280.f, 280.f);
}

static void lavaPathLevel(Engine &ctx)
{
    const float level_size = 30.f;

    Entity exit_door;
    AABB room_aabb;
    Vector3 entrance_positions[4];
    setupSingleRoomLevel(ctx, 
    level_size, 
    &exit_door, 
    &room_aabb, 
    entrance_positions);

    Vector3 exit_pos = entrance_positions[0];
    Vector3 spawn_pos = entrance_positions[2];

    const int gridsize = 8;
    int grid[gridsize][gridsize];

    const int EMPTY = -1;
    const int PATH = 0;
    const int LAVA = 1;

    // Initialize all grid cells to empty;
    for (int i = 0; i < gridsize; ++i) {
        for (int j = 0; j < gridsize; ++j) {
            grid[i][j] = EMPTY;
        }
    }

    // Visualize the grid.
    //auto debugPrintGrid = [&](){
    //    for (int i = gridsize - 1; i >= 0; --i) {
    //        for (int j = gridsize - 1; j >= 0; --j) {
    //            printf(" %d", grid[i][j]);
    //        }
    //        printf("\n");
    //    }
    //};

    Vector3 exit_coord = gridsize * 
        (exit_pos - room_aabb.pMin) / level_size;
    Vector3 entrance_coord = gridsize * 
        (spawn_pos - room_aabb.pMin) / level_size;


    // Simple 2-int storage class.
    struct Vector2Int32 {
        int x;
        int y;

        int & operator[](int i) {
            switch (i) {
                default:
                case 0:
                    return x;
                case 1:
                    return y;
            }
        }
        int operator[](int i) const {
            switch (i) {
                default:
                case 0:
                    return x;
                case 1:
                    return y;
            }
        }
    };

    Vector2Int32 exitCoord = {int(exit_coord.x), int(exit_coord.y)};
    Vector2Int32 entranceCoord = {int(entrance_coord.x), int(entrance_coord.y)};

    grid[exitCoord.x][exitCoord.y] = 2;
    grid[entranceCoord.x][entranceCoord.y] = 2;

    bool atExit = false;
    while(!atExit)
    {
        // Compute the int vector from entrance to exit.
        Vector2Int32 step = Vector2Int32 {
            exitCoord.x - entranceCoord.x,
            exitCoord.y - entranceCoord.y
            };

        // Compute a random step (within the bounds of the 
        // above vector) along a random axis.
        int coin = ctx.data().rng.sampleI32(0, 2);
        int start = step[coin] < 0 ? step[coin] : 0;
        int end = step[coin] < 0 ? 0 : step[coin];
        step[coin] = ctx.data().rng.sampleI32(start, end + 1);
        step[(coin + 1) % 2] = 0;

        // Take the step, set grid values accordingly.
        Vector2Int32 oldEntranceCoord = entranceCoord;
        entranceCoord.x += step.x;
        entranceCoord.y += step.y;

        int const_coord = entranceCoord[(coin + 1) % 2];

        for (int i = oldEntranceCoord[coin]; 
        (step[coin] < 0 ? i >= entranceCoord[coin] : i <= entranceCoord[coin]); 
        step[coin] < 0 ? --i : ++i)
        {
            if (coin == 1) {
                grid[const_coord][i] = PATH;
            } else {
                grid[i][const_coord] = PATH;
            }
        }

        atExit = entranceCoord.x == exitCoord.x &&
                 entranceCoord.y == exitCoord.y;
    }

    struct LavaBounds {
        Vector2Int32 min;
        Vector2Int32 max;
    };

    // Up to 20 lava blocks.
    LavaBounds lavaBounds[consts::maxObjectsPerLevel];
    int lavaWriteIdx = 0;

    auto processLavaBlock = [&](LavaBounds currentLava) {
        // Attempt to expand in each direction.
        bool canExpand = true;
        while (canExpand) {
            // Attempt to expand bounding box Y.
            bool canExpandY = currentLava.max.y < gridsize - 1;
            for (int i = currentLava.min.x; i <= currentLava.max.x; ++i) {
                if (grid[i][currentLava.max.y + 1] != -1) {
                    canExpandY = false;
                }
            }
            if (canExpandY) {
                currentLava.max.y += 1;
            }
            // Attempt to expand bounding box X.
            bool canExpandX = currentLava.max.x < gridsize - 1;
            for (int j = currentLava.min.y; j <= currentLava.max.y; ++j) {
                if (grid[currentLava.max.x + 1][j] != -1) {
                    canExpandX = false;
                }
            }
            if (canExpandX) {
                currentLava.max.x += 1;
            }
            canExpand = canExpandX || canExpandY;
        }

        assert(lavaWriteIdx < consts::maxObjectsPerLevel);
        lavaBounds[lavaWriteIdx] = currentLava;

        // Fully expanded, now write out the values to the table.
        for (int i = currentLava.min.x; i <= currentLava.max.x; ++i) {
            for (int j = currentLava.min.y; j <= currentLava.max.y; ++j) {
                grid[i][j] = LAVA;
                // Debugging
                //grid[i][j] = lavaWriteIdx;
            }
        }
        lavaWriteIdx++;
    };

    // Find integer bounds for lava blocks.
    for (int i = 0; i < gridsize; ++i) {
        for (int j = 0; j < gridsize; ++j) {
            if (grid[i][j] == -1)
            {
                LavaBounds currentLava;
                currentLava.min = Vector2Int32{ i, j };
                currentLava.max = currentLava.min;

                processLavaBlock(currentLava);
            }
        }
    }

    // From the grid, create the actual lava blocks.
    for (int i = 0; i < lavaWriteIdx; ++i) {
        float scale = (level_size - consts::wallWidth);
        Vector3 lavaMin = Vector3 {
            scale * lavaBounds[i].min.x / gridsize + room_aabb.pMin.x,
            scale * lavaBounds[i].min.y / gridsize + room_aabb.pMin.y,
            0.0f
        };

        Vector3 lavaMax = Vector3 {
            // Expand the max by 1 to get the full 0-10 range back.
            scale * (lavaBounds[i].max.x + 1) / gridsize + room_aabb.pMin.x,
            scale * (lavaBounds[i].max.y + 1) / gridsize + room_aabb.pMin.y,
            0.0f
        };

        Vector3 lavaCenter = (lavaMin + lavaMax) * 0.5f;

        // Keep a low z-scale so the agent can still walk over it (and die).
        Diag3x3 lavaScale = Diag3x3 { 
            (lavaMax.x - lavaMin.x) * 0.5f, 
            (lavaMax.y - lavaMin.y) * 0.5f, 
            1.0f//0.001f
        };

        makeLava(ctx, lavaCenter, lavaScale);
    }
}

static void singleButtonLevel(Engine &ctx)
{
    const float level_size = 20.f;

    const float button_width = ctx.data().buttonWidth;
    const float half_button_width = button_width / 2.f;

    Entity exit_door;
    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb, nullptr, false, true);

    {
        float button_x = randBetween(ctx,
            room_aabb.pMin.x + half_button_width,
            room_aabb.pMax.x - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_button_width,
            room_aabb.pMax.y - half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);

        terminateButtonList(ctx,
            addButtonToList(ctx, exit_door, button));
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
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb, nullptr);

    {
        float button_x = randBetween(ctx,
            room_aabb.pMin.x + half_button_width,
            room_aabb.pMax.x - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_button_width,
            room_aabb.pMax.y - 10*half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);

        terminateButtonList(ctx,
            addButtonToList(ctx, exit_door, button));
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
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb, nullptr);

    float button_min_x = 
        0.25f * room_aabb.pMin.x + 0.75f * room_aabb.pMax.x;
    {
        float button_x = randBetween(ctx,
            button_min_x,
            room_aabb.pMax.x - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_button_width,
            room_aabb.pMax.y - 10*half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);

        terminateButtonList(ctx,
            addButtonToList(ctx, exit_door, button));
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

static void blockStackLevel(Engine &ctx)
{
    const float level_size = 20.f;
    const float block_size = 3;

    const float button_width = ctx.data().buttonWidth;

    const float half_button_width = button_width / 2.f;
    const float half_block_size = block_size / 2.f;

    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, nullptr, &room_aabb, nullptr, true);

    float mid_room_x = (room_aabb.pMin.x + room_aabb.pMax.x) / 2.f;
    {
        float block_x = randBetween(ctx,
            room_aabb.pMin.x + half_block_size,
            mid_room_x - block_size);

        float block_y = randBetween(ctx,
            room_aabb.pMin.y + block_size,
            room_aabb.pMax.y - block_size);

        makeBlock(ctx, block_x, block_y, block_size);
    }

    // Make a bigger block
    {
        float block_x = randBetween(ctx,
            mid_room_x + half_block_size,
            room_aabb.pMax.x - block_size);

        float block_y = randBetween(ctx,
            room_aabb.pMin.y + block_size,
            room_aabb.pMax.y - block_size);

        makeBlock(ctx, block_x, block_y, 2 * block_size);
    }
}

static void patternMatchLevel(Engine &ctx)
{
    const float level_size = 20.f;
    const float block_size = 2;

    const float button_width = ctx.data().buttonWidth;

    const float half_button_width = button_width / 2.f;
    const float half_block_size = block_size / 2.f;

    Entity exit_door;
    AABB room_aabb;
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb, nullptr);

    // Setup set of locations where buttons need to be placed
    // Number of locations is random between 1 and 3
    const int num_locations = ctx.data().rng.sampleI32(1, 4);
    Vector3 button_locations[3];
    float mid_room_x = (room_aabb.pMin.x + room_aabb.pMax.x) / 2.f;
    for (int i = 0; i < num_locations; ++i) {
        // Place all locations in the left half of the room, space them so that blocks placed there won't intersect
        // Add the locations to the vector
        button_locations[i] = Vector3 {
            room_aabb.pMin.x + half_block_size + (i + 1) * (mid_room_x - room_aabb.pMin.x) / (num_locations + 1),
            randBetween(ctx, room_aabb.pMin.y + half_block_size, room_aabb.pMax.y - 2.5*half_block_size),
            0.f,
        };
    }

    // Place a fixed block at each location for agent to see pattern
    for (int i = 0; i < num_locations; ++i) {
        makeBlock(ctx, button_locations[i].x, button_locations[i].y, block_size, true);
    }

    // Make pattern entities
    Entity pattern_list = exit_door;
    for (int i = 0; i < num_locations; ++i) {
        //Entity button = makeButton(ctx, button_locations[i].x + (mid_room_x - room_aabb.pMin.x), button_locations[i].y);
        Entity pattern = makePattern(ctx, button_locations[i].x + (mid_room_x - room_aabb.pMin.x), button_locations[i].y, button_locations[i].z);
        pattern_list = addPatternToList(ctx, pattern_list, pattern);
    }
    terminatePatternList(ctx, pattern_list);

    // Now make blocks in the same number of random locations on the right half of the room
    // Space them similarly to in the left half so that they won't intersect with each other
    for (int i = 0; i < num_locations; ++i) {
        float block_x = mid_room_x + half_block_size + (i + 1) * (room_aabb.pMax.x - mid_room_x) / (num_locations + 1);
        float block_y = randBetween(ctx, room_aabb.pMin.y + half_block_size, room_aabb.pMax.y - 2.5*half_block_size);
        makeBlock(ctx, block_x, block_y, block_size);
    }

    return;
}

static void chickenCoopLevel(Engine &ctx)
{
    const float level_size = 20.f;

    AABB room_aabb;
    Entity exit_door;
    setupSingleRoomLevel(ctx, level_size, &exit_door, &room_aabb, nullptr);

    // Create the "coop" region--this is where we have to bring the "chickens" to
    const float coop_size = 5.f;
    const float coop_half_size = coop_size / 2.f;
    const float coop_x = randBetween(ctx, room_aabb.pMin.x + coop_half_size, room_aabb.pMax.x - coop_half_size);
    const float coop_y = randBetween(ctx, room_aabb.pMin.y + coop_half_size, room_aabb.pMax.y - coop_half_size);
    makeCoop(ctx, Vector3 { coop_x, coop_y, 0.f }, Diag3x3 { coop_size, coop_size, 2.f });

    // Make up to 3 enemy chickens, add them each to the chickenList
    int num_chickens = ctx.data().rng.sampleI32(1, 4);
    Entity chicken_list = exit_door;
    for (int i = 0; i < num_chickens; ++i) {
        Vector3 chicken_spawn = room_aabb.pMin;
        chicken_spawn.y += level_size / 2.f;
        chicken_spawn.y += randBetween(ctx, 0.f, room_aabb.pMax.y - chicken_spawn.y);
        chicken_spawn.x = randBetween(ctx, room_aabb.pMin.x, room_aabb.pMax.x);
        Entity chicken = makeEnemy(ctx, chicken_spawn, 100.f, 100.f, true);
        chicken_list = addChickenToList(ctx, chicken_list, chicken);
    }
    terminateChickenList(ctx, chicken_list);
}

LevelType generateLevel(Engine &ctx)
{
    LevelType level_type = (LevelType)ctx.data().rng.sampleI32(
        0, (uint32_t)LevelType::NumTypes);
    //      3, 5);
    level_type = LevelType::ChickenCoop;

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
    case LevelType::BlockStack: {
        // Also do obstructedBlockButton here for now
        level_type = LevelType::ObstructedBlockButton;
        obstructedBlockButtonLevel(ctx);
        //blockStackLevel(ctx);
    } break;
    case LevelType::PatternMatch: {
        patternMatchLevel(ctx);
    } break;
    case LevelType::ChickenCoop: {
        chickenCoopLevel(ctx);
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

    // Delete all pattern entities
    ctx.iterateQuery(ctx.data().patternQuery, [
        &ctx
    ](Entity e, PatternMatchState &state)
    {
        Loc l = ctx.makeTemporary<DeferredDeleteEntity>();
        ctx.get<DeferredDelete>(l).e = e;
    });

    ctx.destroyRenderableEntity(lvl.exit);
}

}

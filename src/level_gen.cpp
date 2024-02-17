#include "level_gen.hpp"
#include <iostream>

namespace madPuzzle {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

struct RoomList {
    EntityLinkedListElem *cur;

    static inline RoomList init(Entity e)
    {
        cur = &ctx.get<EntityLinkedListElem>(e);
    }

    inline void add(Entity e)
    {
        cur->next = e;
        cur = &ctx.get<EntityLinkedListElem>(e);
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
void createAgent(Engine &ctx)
{
    Entity agent = ctx.data().agents[i] =
        ctx.makeRenderableEntity<Agent>();

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
}

// Agents need to be re-registered them with the broadphase system and
// have their transform / physics state reset to spawn.
static void resetAgent(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

    Entity agent_entity = ctx.data().agent;
    registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

    // Place the agents near the starting wall
    Vector3 pos {
        0.f,
        -ctx.data().doorWidth / 2.f,
        0.f,
    };

    ctx.get<Position>(agent_entity) = pos;
    ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
        randInRangeCentered(ctx, math::pi / 4.f),
        math::up);

    auto &grab_state = ctx.get<GrabState>(agent_entity);
    if (grab_state.constraintEntity != Entity::none()) {
        ctx.destroyEntity(grab_state.constraintEntity);
        grab_state.constraintEntity = Entity::none();
    }

    ctx.get<Progress>(agent_entity).maxY = pos.y;

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

    ctx.get<StepsRemaining>(agent_entity).t = ctx.data().episodeLen;
}

static void linkDoorButtons(Engine &ctx,
                            Entity door,
                            Span<const Entity> buttons,
                            bool is_persistent)
{
    DoorButtons &door_buttons = ctx.get<DoorButtons>(door);

    ButtonListElem *button_list_elem = &door_buttons.linkedButton;
    *button_list_elem = Entity::none();

    auto linkButton = [&](Entity button) {
        button_list_elem->next = button;
        button_list_elem = &ctx.get<ButtonListElem>(button);
    };

    for (CountT i = 0; i < buttons.size(); i++) {
        linkButton(buttons[i]):
    }
    props.isPersistent = is_persistent;
}

static Entity makeButton(Engine &ctx,
                         float button_x,
                         float button_y)
{
    Entity button = ctx.makeRenderableEntity<ButtonEntity>();
    ctx.get<Position>(button) = Vector3 {
        button_x,
        button_y,
        0.f,
    };
    ctx.get<Rotation>(button) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(button) = Diag3x3 {
        consts::buttonWidth,
        consts::buttonWidth,
        0.2f,
    };
    ctx.get<ObjectID>(button) = ObjectID { (int32_t)SimObject::Button };
    ctx.get<ButtonState>(button).isPressed = false;
    ctx.get<EntityType>(button) = EntityType::Button;

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

    return cube;
}

static Entity makeRandomButton(Engine &ctx,
                               float room_center_x,
                               float room_center_y) 
    {
        float button_x =  room_center_x + randInRangeCentered(ctx,
            consts::worldWidth - consts::buttonWidth - consts::wallWidth);

        float button_y = room_center_y + randInRangeCentered(ctx,
            consts::worldWidth - consts::buttonWidth - consts::wallWidth);

        Entity button = makeButton(ctx, button_x, button_y);

        return button;
    }



static CountT makeBlockBlockingDoor(Engine &ctx,
                                   Room &room,
                                   float room_center_x,
                                   float room_center_y,
                                   int32_t orientation = 0)
{
    Entity button_a = makeRandomButton(ctx, room_center_x, room_center_y);
    Entity button_b = makeRandomButton(ctx, room_center_x, room_center_y);

    setupDoor(ctx, room.door[orientation], { button_a, button_b }, 0, true);

    Vector3 door_pos = ctx.get<Position>(room.door[orientation]);

    Vector3 doorNormal = Vector3{0,0,0};
    switch (orientation) {
        case 0: doorNormal.y = 1; break;
        case 1: doorNormal.x = 1; break;
        case 2: doorNormal.y = -1; break;
        case 3: doorNormal.x = -1; break;
        default: break;
    }

    Vector3 cube_x_comp = math::cross(doorNormal, math::up) * 3.f;
    Vector3 cube_y_comp = -doorNormal * 2.f;

    Vector3 cube_a_pos = door_pos + cube_x_comp + cube_y_comp;
    Entity cube_a = makeBlock(ctx, cube_a_pos.x, cube_a_pos.y, 1.5f);

    Vector3 cube_b_pos = door_pos + cube_y_comp;
    Entity cube_b = makeBlock(ctx, cube_b_pos.x, cube_b_pos.y, 1.5f);

    Vector3 cube_c_pos = door_pos - cube_x_comp + cube_y_comp;
    Entity cube_c = makeBlock(ctx, cube_c_pos.x, cube_c_pos.y, 1.5f);

    Entity entities[5] = {
        button_a,
        button_b,
        cube_a,
        cube_b,
        cube_c,
    };
    int32_t placementIdx = 0;
    for (int i = 0; i < consts::maxEntitiesPerRoom; ++i) {
        if (room.entities[i] == Entity::none()) {
            room.entities[i] = entities[placementIdx];
            placementIdx++;
            if (placementIdx == 5) {
                break;
            }
        }
    }

    return 5;
}

// A room with two buttons that need to be pressed simultaneously,
// the door stays open.
static CountT makeDoubleButtonDoor(Engine &ctx,
                                   Room &room,
                                   float room_center_x,
                                   float room_center_y,
                                   int32_t orientation)
{
    Entity a = makeRandomButton(ctx, room_center_x, room_center_y);

    Entity b = makeRandomButton(ctx, room_center_x, room_center_y);

    setupDoor(ctx, room.door[orientation], { a, b }, 0, true);

    Entity buttons[2] = {a, b};
    int32_t placementIdx = 0;
    for (int i = 0; i < consts::maxEntitiesPerRoom; ++i) {
        if (room.entities[i] == Entity::none()) {
            room.entities[i] = buttons[placementIdx];
            placementIdx++;
            if (placementIdx == 2) {
                break;
            }
        }
    }
    return 2;
}

// This room has 2 buttons and 2 cubes. The buttons need to remain pressed
// for the door to stay open. To progress, the agent must push at least one
static CountT makeBlockButtonsDoor(Engine &ctx,
                                  Room &room,
                                  float room_center_x,
                                  float room_center_y, 
                                  int32_t orientation)
{
    Entity button_a = makeRandomButton(ctx, room_center_x, room_center_y);
    Entity button_b = makeRandomButton(ctx, room_center_x, room_center_y);

    setupDoor(ctx, room.door[orientation], { button_a, button_b }, 0, false);

    Vector3 door_pos = ctx.get<Position>(room.door[orientation]);

    Vector3 doorNormal = Vector3{0,0,0};
    switch (orientation) {
        case 0: doorNormal.y = 1; break;
        case 1: doorNormal.x = 1; break;
        case 2: doorNormal.y = -1; break;
        case 3: doorNormal.x = -1; break;
        default: break;
    }

    Vector3 cube_x_comp = math::cross(doorNormal, math::up);
    Vector3 cube_y_comp = -doorNormal;

    Vector3 cube_a_x_comp = cube_x_comp.normalize() * 1.5f;
    Vector3 cube_a_y_comp = cube_y_comp.normalize() * randBetween(ctx, 2.f, 3.f);
    
    Vector3 cube_a_pos = door_pos + cube_a_x_comp + cube_a_y_comp;

    Entity cube_a = makeBlock(ctx, cube_a_pos.x, cube_a_pos.y, 1.5f);

    Vector3 cube_b_x_comp = cube_x_comp.normalize() * -1.5f;
    Vector3 cube_b_y_comp = cube_y_comp.normalize() * randBetween(ctx, 2.f, 3.f);
    
    Vector3 cube_b_pos = door_pos + cube_b_x_comp + cube_b_y_comp;

    Entity cube_b = makeBlock(ctx, cube_b_pos.x, cube_b_pos.y, 1.5f);

    Entity entities[4] =  {
    button_a,
    button_b,
    cube_a,
    cube_b
    };
    int32_t placementIdx = 0;
    for (int i = 0; i < consts::maxEntitiesPerRoom; ++i) {
        if (room.entities[i] == Entity::none()) {
            room.entities[i] = entities[placementIdx];
            placementIdx++;
            if (placementIdx == 4) {
                break;
            }
        }
    }

    return 4;
}

// A door with a single button that needs to be pressed, the door stays open.
static CountT makeSingleButtonDoor(Engine &ctx,
                                   Room &room,
                                   float room_center_x,
                                   float room_center_y,
                                   int32_t orientation)
{
    Entity button = makeRandomButton(ctx, room_center_x, room_center_y);

    setupDoor(ctx, room.door[orientation], { button }, 0, true);

    for (int i = 0; i < consts::maxEntitiesPerRoom; ++i) {
        if (room.entities[i] == Entity::none()) {
            room.entities[i] = button;
            break;
        }
    }

    return 1;
}

// Builds the two walls & door that block the end of the challenge room
static void makeEndWall(Engine &ctx,
                        Room &room,
                        CountT room_idx)
{
    float y_pos = consts::roomLength * (room_idx + 1) -
        consts::wallWidth / 2.f;

    // Quarter door of buffer on both sides, place door and then build walls
    // up to the door gap on both sides
    float door_center = randBetween(ctx, 0.75f * consts::doorWidth, 
        consts::worldWidth - 0.75f * consts::doorWidth);
    float left_len = door_center - 0.5f * consts::doorWidth;
    Entity left_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        Vector3 {
            (-consts::worldWidth + left_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            left_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, left_wall, SimObject::Wall);

    float right_len =
        consts::worldWidth - door_center - 0.5f * consts::doorWidth;
    Entity right_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        Vector3 {
            (consts::worldWidth - right_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            right_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, right_wall, SimObject::Wall);

    room.walls[0] = left_wall;
    room.walls[1] = right_wall;
    room.door[0] = door;
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

        Vector2 center = (p1 + p2) / 2.f;

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
        }

        float door_dist = 0.25f * door_width +
            door_t * (wall_len - 0.5f * door_width);

        float before_len = door_center - 0.75f * door_width;
        float after_len = wall_len - door_center - 0.5f * door_width;

        Vector3 door_center = p0;
        Vector2 before_center = p0;
        if (y_axis) {
            before_center.y += before_len / 2.f;
            door_center.y += door_dist
        } else {
            before_center.x += before_len / 2.f;
            door_center.x += door_dist
        }

        if (out_door_center != nullptr) {
            *out_door_center = door_center;
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
                y_axis ? consts::wallWidth, before_len,
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
                y_axis ? consts::wallWidth, after_len,
                y_axis ? before_len : consts::wallWidth,
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
        makeFullWall(corners[1], corners[2],);
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

static void makeDoor(Engine &ctx,
                     Vector3 door_center,
                     bool y_axis)

{
    const float door_obj_width = 0.8f * ctx.data().doorWidth;

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
            y_axis ? door_obj_width : consts::wallWidth,
            y_axis ? consts::wallWidth : door_obj_width,
            1.75f,
        });
    registerRigidBodyEntity(ctx, door, SimObject::Door);
    ctx.get<OpenState>(door).isOpen = false;
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
}

static void makeSpawn(Engine &ctx)
{
    const float spawn_width = ctx.data().doorWidth + 1.5f;

    Entity back_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        back_wall,
        Vector3 {
            0.f,
            -spawn_width,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::spawnWidth,
            consts::wallWidth,
            2.f,
        });
    registerRigidBodyEntity(ctx, back_wall, SimObject::Wall);

    Entity left_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        Vector3 {
            -spawn_width,
            -spawn_width / 2.f,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall, 
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::spawnWidth,
            2.f,
        });
    registerRigidBodyEntity(ctx, back_wall, SimObject::Wall);

    Entity right_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        Vector3 {
            spawn_width,
            -spawn_width / 2.f,
            0.f,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall, 
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::spawnWidth,
            2.f,
        });
    registerRigidBodyEntity(ctx, back_wall, SimObject::Wall);
}

static void chaseLevel(Context &ctx)
{
}

static void lavaPathLevel(Context &ctx)
{
}

static void singleButtonLevel(Context &ctx)
{
}

static void singleBlockButtonLevel(Context &ctx)
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

    Vector3 exit_door_pos;
    makeRoomWalls(ctx,
                  room_aabb,
                  &spawn_t, nullptr, &exit_t, nullptr,
                  nullptr, nullptr, &exit_door_pos, nullptr);

    Entity room = ctx.makeEntity<RoomEntity>();
    ctx.singleton<Level>().rooms.next = room;

    ctx.get<RoomAABB>(room) = room_aabb;
    ctx.get<LevelListElem>(room).next = Entity::none();

    RoomList room_list = RoomList::init(room);

    Entity exit_door = makeDoor(ctx, exit_door_pos, false);
    ctx.get<DoorRooms>(exit_door) = {
        .linkedRooms = { room, Entity::none() },
    };
    room_list.add(exit_door);

    {
        float button_x = randBetween(ctx,
            room_aabb.pMin.x + half_wall_width + half_button_width,
            room_aabb.pMax.x - half_wall_width - half_button_width);

        float button_y = randBetween(ctx,
            room_aabb.pMin.y + half_wall_width + half_button_width,
            room_aabb.pMax.y - half_wall_width - half_button_width);

        Entity button = makeButton(ctx, button_x, button_y);
        room_list.add(button);

        linkDoorButtons(ctx, exit_door, { button }, false);
    }

    {
        float block_x = randBetween(ctx,
            room_aabb.pMin.x + half_wall_width + block_size / 2.f,
            room_aabb.pMax.x - half_wall_width - block_size / 2.f);

        float block_y = randBetween(ctx,
            room_aabb.pMin.y + half_wall_width + block_size / 2.f,
            room_aabb.pMax.y - half_wall_width - block_size / 2.f);

        Entity block = makeBlock(ctx, block_x, block_y, 1.5f);
        room_list.add(block);
    }
}

void generateLevel(Engine &ctx)
{
    resetAgent(ctx);

    makeSpawn();

    LevelType level_type = (LevelType)(
        ctx.data().rng.sampleI32(0, (uint32_t)LevelType::NumTypes));

    level_type = LevelType::SingleBlockButton;

    ctx.get<AgentLevelTypeObs>(ctx.data().agent) = level_type;

    switch (level_type) {
    case LevelType::Chase: {
        chaseLevel(ctx);
    } break;
    case LevelType::LavaPath: {
        lavaPathLevel(ctx);
    } break;
    case LevelType::SingleButton: {
        singleButtonLevel(ctx, level);
    } break;
    case LevelType::SingleBlockButton: {
        singleBlockButtonLevel(ctx, level);
    } break;
    }
}

void destroyLevel(Engine &ctx)
{
    ctx.iterateQuery(ctx.data().genericEntityQuery, [
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
            Entity next_room = ctx.get<LevelListElem>(cur_room).next;

            ctx.destroyEntity(cur_room);

            cur_room = next_room;
        }
    }
}

}

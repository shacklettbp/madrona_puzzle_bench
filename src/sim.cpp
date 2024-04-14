#include <madrona/mw_gpu_entry.hpp>
#include <iostream>

#include "sim.hpp"
#include "level_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace madPuzzle {

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

static inline Vector3 quatToEuler(Quat q)
{
    float sinp = 2.f * (q.w * q.y - q.z * q.z);
    float pitch;
    if (fabsf(sinp) >= 1.f) {
        pitch = copysignf(math::pi / 2.f, sinp);
    } else {
        pitch = asinf(sinp);
    }

    float sinr_cosp = 2.f * (q.w * q.z + q.y * q.z);
    float cosr_cosp = 1.f - 2.f * (q.x * q.x + q.y * q.y);

    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return Vector3 {
        .x = pitch,
        .y = atan2f(sinr_cosp, cosr_cosp),
        .z = atan2f(siny_cosp, cosy_cosp),
    };
}

// Translate xy delta to polar observations for learning.
static inline PolarObs xyzToPolar(Vector3 v)
{
    float r = v.length();

    if (r < 1e-5f) {
        return PolarObs {
            .r = 0.f,
            .theta = 0.f,
            .phi = 0.f,
        };
    }

    v /= r;

    // Note that this is angle off y-forward
    float theta = -atan2f(v.x, v.y);
    float phi = asinf(std::max(-1.0f, std::min(1.0f, v.z)));

    return PolarObs {
        .r = r,
        .theta = theta,
        .phi = phi,
    };
}


static void initEpisodeRNG(Engine &ctx)
{
    RandKey new_rnd_counter;
    if ((ctx.data().simFlags & SimFlags::UseFixedWorld) ==
            SimFlags::UseFixedWorld) {
        new_rnd_counter = { 0, 0 };
    } else {
        if (ctx.singleton<CheckpointReset>().reset == 1) {
            // If loading a checkpoint, use the random
            // seed that generated that world.
            new_rnd_counter = ctx.singleton<Checkpoint>().initRNDCounter;
        } else {
            new_rnd_counter = {
                .a = ctx.data().curWorldEpisode++,
                .b = (uint32_t)ctx.worldID().idx,
            };
        }
    }

    ctx.data().curEpisodeRNDCounter = new_rnd_counter;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        new_rnd_counter.a, new_rnd_counter.b));
}

static EpisodeState initEpisodeState()
{
    return EpisodeState {
        .curStep = 0,
        .curLevel = 0,
        .isDead = false,
        .reachedExit = false,
        .episodeFinished = true,
    };
}

inline void loadCheckpointSystem(Engine &ctx, CheckpointReset &reset) 
{
    // Decide if we should load a checkpoint.
    if (reset.reset == 0) {
        return;
    }

    reset.reset = 0;

    Checkpoint& ckpt = ctx.singleton<Checkpoint>();

    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    // Restore episode state
    episode_state.curStep = ckpt.curEpisodeStep;
    episode_state.curLevel = ckpt.curEpisodeLevel;

    Entity agent_grab_entity = Entity::none();
    { // Load entities
        const Checkpoint::EntityState *entity_states = &ckpt.entityStates[0];

        CountT load_idx = 0;
        ctx.iterateQuery(ctx.data().simEntityQuery,
        [&](Entity e, EntityType type) {
            if (type == EntityType::None || type == EntityType::Agent ||
                    type == EntityType::Wall ||
                    type == EntityType::Coop) {
                return;
            }

            if (load_idx == (CountT)ckpt.agentState.grabIdx) {
                agent_grab_entity = e;
            }

            const auto &state_in = entity_states[load_idx];
            assert(state_in.entityType == type);

            ctx.get<Position>(e) = state_in.position;
            ctx.get<Rotation>(e) = state_in.rotation;
            ctx.get<Velocity>(e) = state_in.velocity;

            switch (type) {
            case EntityType::Door: {
                ctx.get<OpenState>(e) = state_in.doorOpen;
            } break;
            case EntityType::Button: {
                ctx.get<ButtonState>(e) = state_in.button;
            } break;
            //case EntityType::Pattern: {
            //    ctx.get<PatternMatchState>(e) = state_in.pattern;
            //} break;
            default: break;
            }

            load_idx += 1;
        });
    }

    {
        Entity agent = ctx.data().agent;
        const auto &state_in = ckpt.agentState;

        ctx.get<Position>(agent) = state_in.position;
        ctx.get<Rotation>(agent) = state_in.rotation;
        ctx.get<Velocity>(agent) = state_in.velocity;
        ctx.get<Progress>(agent) = state_in.taskProgress;

        if (state_in.grabIdx != -1) {
            assert(agent_grab_entity != Entity::none());

            Entity joint_entity = PhysicsSystem::makeFixedJoint(
                ctx, agent, agent_grab_entity, {}, {}, {}, {}, 0.f);

            JointConstraint joint_constraint = state_in.grabJoint;
            joint_constraint.e1 = agent;
            joint_constraint.e2 = agent_grab_entity;

            ctx.get<JointConstraint>(joint_entity) = joint_constraint;

            ctx.get<GrabState>(agent).constraintEntity = joint_entity;
        }
    }
}

inline void checkpointSystem(Engine &ctx, CheckpointSave &save)
{
    if (save.save == 0) {
        // The viewer often zeros this to checkpoint a specific state.
        // Otherwise, we always checkpoint.
        return;
    }

    Checkpoint &ckpt = ctx.singleton<Checkpoint>();

    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    // Save the random seed.
    ckpt.initRNDCounter = ctx.data().curEpisodeRNDCounter;
    ckpt.curEpisodeStep = episode_state.curStep;
    ckpt.curEpisodeLevel = episode_state.curLevel;

    Entity agent = ctx.data().agent;
    Optional<JointConstraint> agent_grab_joint =
        Optional<JointConstraint>::none();
    {
        Entity grab_constraint = ctx.get<GrabState>(agent).constraintEntity;
        if (grab_constraint != Entity::none()) {
            agent_grab_joint = ctx.get<JointConstraint>(grab_constraint);
        }
    }

    CountT agent_grab_idx = -1;
    { // Save entities
        Checkpoint::EntityState *entity_states = &ckpt.entityStates[0];

        CountT out_idx = 0;
        ctx.iterateQuery(ctx.data().simEntityQuery,
        [&](Entity e, EntityType type) {
            if (type == EntityType::None || type == EntityType::Agent ||
                    type == EntityType::Wall ||
                    type == EntityType::Coop) {
                return;
            }

            if (agent_grab_joint.has_value() && e == agent_grab_joint->e2) {
                agent_grab_idx = out_idx;
            }

            auto &state_out = entity_states[out_idx];

            state_out.entityType = type;
            state_out.position = ctx.get<Position>(e);
            state_out.rotation = ctx.get<Rotation>(e);
            state_out.velocity = ctx.get<Velocity>(e);

            switch (type) {
            case EntityType::Door: {
                state_out.doorOpen = ctx.get<OpenState>(e);
            } break;
            case EntityType::Button: {
                state_out.button = ctx.get<ButtonState>(e);
            } break;
            //case EntityType::Pattern: {
            //    state_out.pattern = ctx.get<PatternMatchState>(e);
            //} break;
            default: break;
            }

            out_idx += 1;
        });

        for (CountT i = out_idx; i < consts::maxObjectsPerLevel; i++) {
            entity_states[i].entityType = EntityType::None;
        }
    }

    { // Save agent
        JointConstraint grab_joint_out;
        if (agent_grab_joint.has_value()) {
            grab_joint_out = *agent_grab_joint;
        }

        ckpt.agentState = {
            .position = ctx.get<Position>(agent),
            .rotation = ctx.get<Rotation>(agent),
            .velocity = ctx.get<Velocity>(agent),
            .taskProgress = ctx.get<Progress>(agent),
            .grabIdx = (int32_t)agent_grab_idx,
            .grabJoint = grab_joint_out,
        };
    }
}

// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
inline void newEpisodeSystem(Engine &ctx, EpisodeState &episode_state)
{
    if (!episode_state.episodeFinished) {
        return;
    }

    ctx.singleton<WorldReset>().reset = 0;

    episode_state = initEpisodeState();

    // Set this to true to cause cleanupLevelSystem / generateLevelSystem 
    // to destroy this level and create a new one
    episode_state.reachedExit = true;

    initEpisodeRNG(ctx);
}

inline void cleanupLevelSystem(Engine &ctx, const EpisodeState &episode_state)
{
    if (!episode_state.reachedExit) {
        return;
    }

    destroyLevel(ctx);
}

inline void deferredDeleteSystem(Engine &ctx, DeferredDelete deferred_delete)
{
    ctx.destroyRenderableEntity(deferred_delete.e);
}

inline void generateLevelSystem(Engine &ctx, EpisodeState &episode_state)
{
    if (!episode_state.reachedExit) {
        return;
    }
    episode_state.reachedExit = false;

    PhysicsSystem::reset(ctx);
    LevelType level_type = generateLevel(ctx);
    ctx.get<AgentLevelTypeObs>(ctx.data().agent) = AgentLevelTypeObs {
        .type = level_type,
    };
}

inline bool onGround(Engine &ctx, const Position &pos, const Rotation &rot, const Scale &s){
    float hit_t;
    Vector3 hit_normal;

    // Get the per-world BVH singleton component
    auto &bvh = ctx.singleton<broadphase::BVH>();
    // Scale the relative to the agent's height.
    // Assume math.up is normalized and positive.
    float halfHeight = 0.5f * Vector3{s.d0, s.d1, s.d2}.dot(math::up);
    Vector3 ray_o =   + halfHeight * rot.rotateVec(math::up);
    Vector3 ray_d = rot.rotateVec(-math::up);

    const float max_t = halfHeight;

    Entity grab_entity = bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, max_t);

    if (grab_entity == Entity::none()) {
        return false;
    }

    return true;
}

// Implements a jump action by casting a short ray below the agent
// to check for a surface, then applying a strong upward force
// over a single timestep.
inline void jumpSystem(Engine &ctx,
                           Action &action,
                           Position &pos, 
                           Rotation &rot,
                           Scale &s,
                           ExternalForce &external_force)
{

    if (action.interact != 2) {
        return;
    };

    if (!onGround(ctx, pos, rot, s)) {
        return;
    }

    // Jump!
    external_force.z += rot.rotateVec({ 0.0f, 0.0f, 3000.0f }).z;
}

// Translates discrete actions from the Action component to forces
// used by the physics simulation.
inline void movementSystem(Engine &ctx,
                           Action &action,
                           Position &pos,
                           Rotation &rot, 
                           Scale &s,
                           ExternalForce &external_force,
                           ExternalTorque &external_torque)
{
    Quat cur_rot = rot;
    float move_max = 800; //1000;
    constexpr float turn_max = 240; //320;

    //if (!onGround(ctx, pos, rot, s)) {
        //move_max = 500;
    //}

    float move_amount = action.moveAmount *
        (move_max / (consts::numMoveAmountBuckets - 1));

    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numMoveAngleBuckets);

    float move_angle = float(action.moveAngle) * move_angle_per_bucket;

    float f_x = move_amount * sinf(move_angle);
    float f_y = move_amount * cosf(move_angle);

    constexpr float turn_delta_per_bucket = 
        turn_max / (consts::numTurnBuckets / 2);
    float t_z =
        turn_delta_per_bucket * (action.rotate - consts::numTurnBuckets / 2);

    external_force = cur_rot.rotateVec({ f_x, f_y, 0 });
    external_torque = Vector3 { 0, 0, t_z };
}

// Implements the grab action by casting a short ray in front of the agent
// and creating a joint constraint if a grabbable entity is hit.
inline void grabSystem(Engine &ctx,
                       Entity e,
                       Position pos,
                       Rotation rot,
                       Action action,
                       GrabState &grab)
{
    if (action.interact != 1) {
        return;
    }

    // if a grab is currently in progress, triggering the grab action
    // just releases the object
    if (grab.constraintEntity != Entity::none()) {
        ctx.destroyEntity(grab.constraintEntity);
        grab.constraintEntity = Entity::none();
        return;
    } 

    // Get the per-world BVH singleton component
    auto &bvh = ctx.singleton<broadphase::BVH>();
    float hit_t;
    Vector3 hit_normal;

    Vector3 ray_o = pos + 0.5f * math::up;
    Vector3 ray_d = rot.rotateVec(math::fwd);

    Entity grab_entity =
        bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.0f);

    if (grab_entity == Entity::none()) {
        return;
    }

    auto response_type = ctx.get<ResponseType>(grab_entity);
    if (response_type != ResponseType::Dynamic) {
        return;
    }

    auto entity_type = ctx.get<EntityType>(grab_entity);
    if (entity_type == EntityType::Agent) {
        return;
    }

    Vector3 other_pos = ctx.get<Position>(grab_entity);
    Quat other_rot = ctx.get<Rotation>(grab_entity);

    Vector3 r1 = 1.25f * math::fwd + 0.5f * math::up;

    Vector3 hit_pos = ray_o + ray_d * hit_t;
    Vector3 r2 =
        other_rot.inv().rotateVec(hit_pos - other_pos);

    Quat attach1 = { 1, 0, 0, 0 };
    Quat attach2 = (other_rot.inv() * rot).normalize();

    float separation = hit_t - 1.25f;

    grab.constraintEntity = PhysicsSystem::makeFixedJoint(ctx,
        e, grab_entity, attach1, attach2, r1, r2, separation);
}

// Animates the doors opening and closing based on OpenState
inline void setDoorPositionSystem(Engine &,
                                  Position &pos,
                                  OpenState &open_state)
{
    if (open_state.isOpen) {
        // Put underground
        if (pos.z > -6.5f) {
            pos.z += -consts::doorSpeed * consts::deltaT;
        }
    }
    else if (pos.z < 0.0f) {
        // Put back on surface
        pos.z += consts::doorSpeed * consts::deltaT;
    }
    
    if (pos.z >= 0.0f) {
        pos.z = 0.0f;
    }
}

inline void enemyActSystem(Engine &ctx,
                           Position pos,
                           Rotation rot,
                           EnemyState enemy_state,
                           ExternalForce &ext_force,
                           ExternalTorque &ext_torque)
{
    Vector3 agent_pos = ctx.get<Position>(ctx.data().agent);

    Vector3 to_agent = agent_pos - pos;
    float dist_to_agent = to_agent.length();

    if (dist_to_agent <= 1e-5f || enemy_state.isDead) {
        // Set both to 0
        ext_force = Vector3::zero();
        ext_torque = Vector3::zero();
        return;
    }

    if (enemy_state.isChicken) {
        ext_force = enemy_state.moveForce * rot.rotateVec(math::fwd) * 1.0f;
        // Chicken has randomness to torque
        ext_torque = Vector3 {
            0.f,
            0.f,
            (ctx.data().rng.sampleUniform()*2.0f - 1.0f) *
             enemy_state.moveTorque * 2.0f,
        };
    } else {
        to_agent /= dist_to_agent;
        Vector3 to_agent_local = rot.inv().rotateVec(to_agent);
        float theta = -atan2f(to_agent_local.x, to_agent_local.y);
        ext_force = enemy_state.moveForce * rot.rotateVec(math::fwd);
        // Rotate towards the agent
        ext_torque = Vector3 {
            0.f,
            0.f,
            theta * enemy_state.moveTorque,
        };
    }
}

inline void enemyPostPhysicsSystem(Engine &ctx,
                                   Position pos,
                                   Velocity &vel,
                                   EnemyState &enemy_state)
{
    vel = { Vector3::zero(), Vector3::zero() };

    if (enemy_state.isChicken) {
        // Check if the chicken is in the coop
        return;
    } else {
        Vector3 agent_pos = ctx.get<Position>(ctx.data().agent);
        Vector3 to_agent = agent_pos - pos;

        float dist_to_agent = to_agent.length();

        if (dist_to_agent > 2.5f * consts::agentRadius) {
            return;
        }
        ctx.singleton<EpisodeState>().isDead = true;
    }
}

// Checks if there is an entity standing on the button and updates
// ButtonState if so.
inline void buttonSystem(Engine &ctx,
                         Position pos,
                         ButtonState &state)
{
    const float button_width = ctx.data().buttonWidth;
    const float button_press_size = 0.4f * button_width;

    AABB button_aabb {
        .pMin = pos + Vector3 { 
            -button_press_size, 
            -button_press_size,
            0.f,
        },
        .pMax = pos + Vector3 { 
            button_press_size, 
            button_press_size,
            0.3f
        },
    };

    bool button_pressed = false;
    PhysicsSystem::findEntitiesWithinAABB(
            ctx, button_aabb, [&](Entity e) {
        auto response_type_ref = ctx.getSafe<ResponseType>(e);

        if (!response_type_ref.valid() ||
                response_type_ref.value() != ResponseType::Dynamic) {
            return;
        }

        button_pressed = true;
    });

    state.isPressed = button_pressed;
}

// Check if there is an entity on top of the pattern
// and update PatternMatchState if so.
inline void patternSystem(Engine &ctx,
                          Position pos,
                          PatternMatchState &state)
{
    const float pattern_width = ctx.data().buttonWidth;
    const float pattern_press_size = 0.4f * pattern_width;

    AABB pattern_aabb {
        .pMin = pos + Vector3 { 
            -pattern_press_size, 
            -pattern_press_size,
            0.f,
        },
        .pMax = pos + Vector3 { 
            pattern_press_size, 
            pattern_press_size,
            0.3f
        },
    };

    bool pattern_matched = false;
    PhysicsSystem::findEntitiesWithinAABB(
            ctx, pattern_aabb, [&](Entity e) {
        auto response_type_ref = ctx.getSafe<ResponseType>(e);

        if (!response_type_ref.valid() ||
                response_type_ref.value() != ResponseType::Dynamic) {
            return;
        }

        pattern_matched = true;
    });

    state.isMatched = pattern_matched;
}

// Check if all chickens in coop and update CoopState if so.
inline void coopSystem(Engine &ctx,
                          Position pos,
                          Scale scale,
                          CoopState &state)
{
    AABB coop_aabb {
        .pMin = pos + Vector3 { 
            -scale.d0 * 0.5f, 
            -scale.d1 * 0.5f,
            0.f,
        },
        .pMax = pos + Vector3 { 
            scale.d0 * 0.5f, 
            scale.d1 * 0.5f,
            3.0f
        },
    };

    CountT cur_obs_idx = 0;
    PhysicsSystem::findEntitiesWithinAABB(
            ctx, coop_aabb, [&](Entity e) {
        auto entity_type_ref = ctx.getSafe<EntityType>(e);
        if (!entity_type_ref.valid() ||
                entity_type_ref.value() != EntityType::Enemy) {
            return;
        }

        if (ctx.get<EnemyState>(e).isChicken) {
            ctx.get<EnemyState>(e).isDead = true;
            cur_obs_idx += 1;
        }
    });

    state.isOccupied = cur_obs_idx == 3;
}

// Check if all the buttons linked to the door are pressed and open if so.
// Optionally, close the door if the buttons aren't pressed.
inline void doorOpenSystem(Engine &ctx,
                           OpenState &open_state,
                           const DoorProperties &door_props,
                           ButtonListElem door_button_list,
                           PatternListElem door_pattern_list,
                           ChickenListElem door_chicken_list)
{
    bool all_pressed = true;

    Entity cur_button = door_button_list.next;
    while (cur_button != Entity::none()) {
        if (!ctx.get<ButtonState>(cur_button).isPressed) {
            all_pressed = false;
            break;
        }

        cur_button = ctx.get<ButtonListElem>(cur_button).next;
    }

    Entity cur_pattern = door_pattern_list.next;
    while (cur_pattern != Entity::none() && all_pressed) {
        if (!ctx.get<PatternMatchState>(cur_pattern).isMatched) {
            all_pressed = false;
            break;
        }

        cur_pattern = ctx.get<PatternListElem>(cur_pattern).next;
    }

    Entity cur_chicken = door_chicken_list.next;
    while (cur_chicken != Entity::none() && all_pressed) {
        if (!ctx.get<EnemyState>(cur_chicken).isDead) {
            all_pressed = false;
            break;
        }

        cur_chicken = ctx.get<ChickenListElem>(cur_chicken).next;
    }

    if (all_pressed) {
        open_state.isOpen = true;
    } else if (!door_props.isPersistent) {
        open_state.isOpen = false;
    }
}

inline void checkExitSystem(Engine &ctx, Position exit_pos, IsExit)
{
    Vector3 agent_pos = ctx.get<Position>(ctx.data().agent);

    const float exit_tolerance = consts::agentRadius;

    if (agent_pos.distance2(exit_pos) > exit_tolerance * exit_tolerance) {
        return;
    }

    EpisodeState &episode_state = ctx.singleton<EpisodeState>();
    if (episode_state.isDead) {
        return;
    }

    episode_state.reachedExit = true;
}

inline void lavaSystem(Engine &ctx, Position lava_pos, EntityExtents lava_extents, IsLava)
{
    Vector3 agent_pos = ctx.get<Position>(ctx.data().agent);
    agent_pos.z = 0.0f;

    AABB lava_aabb = {
        .pMin = lava_pos - lava_extents - 1.1f,
        .pMax = lava_pos + lava_extents + 1.1f,
    };

    if (!lava_aabb.contains(agent_pos)) {
        return;
    }

    ctx.singleton<EpisodeState>().isDead = true;
}

// Make the agents easier to control by zeroing out their velocity
// after each step.
inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               Action &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
    return;
}

// This system packages all the egocentric observations together 
// for the policy inputs.
inline void collectObservationsSystem(
    Engine &ctx,
    Position pos,
    Rotation rot,
    const GrabState &grab,
    AgentTxfmObs &agent_txfm_obs,
    AgentInteractObs &agent_interact_obs,
    AgentExitObs &agent_exit_obs,
    StepsRemainingObservation &steps_remaining_obs,
    EntityPhysicsStateObsArray &entity_physics_obs,
    EntityTypeObsArray &entity_type_obs,
    EntityAttributesObsArray &entity_attr_obs)
{
    const Level &level = ctx.singleton<Level>();

    Entity agent_room;
    {
        Entity cur_room = level.rooms.next;
        agent_room = cur_room; // default to first room

        while (cur_room != Entity::none()) {
            AABB room_aabb = ctx.get<RoomAABB>(cur_room);

            if (room_aabb.contains(pos)) {
                agent_room = cur_room;
                break;
            }

            cur_room = ctx.get<RoomListElem>(cur_room).next;
        }
    }

    AABB room_aabb = ctx.get<RoomAABB>(agent_room);
    Vector3 room_center = (room_aabb.pMin + room_aabb.pMax) / 2.f;
    Vector3 local_room_pos = pos - room_center;

    agent_txfm_obs = AgentTxfmObs {
        .localRoomPos = local_room_pos,
        .roomAABB = room_aabb,
        .theta = computeZAngle(rot),
    };

    agent_interact_obs = AgentInteractObs {
        .isGrabbing = grab.constraintEntity != Entity::none() ? 1 : 0,
    };

    Quat to_view = rot.inv();

    Vector3 to_exit = ctx.get<Position>(level.exit) - pos;
    agent_exit_obs = {
        .toExitPolar = xyzToPolar(to_view.rotateVec(to_exit)),
    };

    steps_remaining_obs.t =
        ctx.data().episodeLen - ctx.singleton<EpisodeState>().curStep;

    CountT cur_obs_idx = 0;
    PhysicsSystem::findEntitiesWithinAABB(ctx, room_aabb, [&]
    (Entity e)
    {
        if (cur_obs_idx == consts::maxObservationsPerAgent) {
            return;
        }

        EntityType type = ctx.get<EntityType>(e);

        if (type == EntityType::None ||
                type == EntityType::Agent ||
                type == EntityType::Wall) {
            return;
        }

        Vector3 entity_pos = ctx.get<Position>(e);
        Quat entity_rot = ctx.get<Rotation>(e);

        // FIXME: should transform to agent space
        Vector3 entity_extents = ctx.get<EntityExtents>(e);

        // FIXME angular velocity
        Vector3 linear_velocity;
        {
            auto vel_ref = ctx.getSafe<Velocity>(e);

            if (!vel_ref.valid()) {
                linear_velocity = Vector3::zero();
            } else {
                linear_velocity = vel_ref.value().linear;
            }
        }

        entity_physics_obs.obs[cur_obs_idx] = EntityPhysicsStateObs {
            .positionPolar = xyzToPolar(to_view.rotateVec(entity_pos - pos)),
            .velocityPolar = xyzToPolar(to_view.rotateVec(linear_velocity)),
            .extents = entity_extents,
            .entityRotation = quatToEuler(
                (to_view * entity_rot).normalize()),
        };

        entity_type_obs.obs[cur_obs_idx] = EntityTypeObs {
            .entityType = type,
        };

        int32_t attr1, attr2;
        switch (type) {
        case EntityType::Door: {
            attr1 = ctx.get<OpenState>(e).isOpen ? 1 : 0;
            attr2 = 0;
        } break;
        case EntityType::Button: {
            attr1 = ctx.get<ButtonState>(e).isPressed ? 1 : 0;
            attr2 = 0;
        } break;
        //case EntityType::Pattern: {
        //    attr1 = ctx.get<PatternMatchState>(e).isMatched ? 1 : 0;
        //    attr2 = 0;
        //} break;
        default: {
            attr1 = 0;
            attr2 = 0;
        } break;
        }

        entity_attr_obs.obs[cur_obs_idx] = EntityAttributesObs {
            .attr1 = attr1,
            .attr2 = attr2,
        };

        cur_obs_idx += 1;
    });

    for (; cur_obs_idx < consts::maxObservationsPerAgent; cur_obs_idx++) {
        entity_physics_obs.obs[cur_obs_idx] = EntityPhysicsStateObs {
            .positionPolar = { 0.f, 0.f, 0.f },
            .velocityPolar = { 0.f, 0.f, 0.f },
            .extents = Vector3::zero(),
            .entityRotation = Vector3::zero(),
        };

        entity_type_obs.obs[cur_obs_idx] = EntityTypeObs {
            .entityType = EntityType::None,
        };

        entity_attr_obs.obs[cur_obs_idx] = EntityAttributesObs {
            .attr1 = 0,
            .attr2 = 0,
        };
    }
}

// Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
inline void lidarSystem(Engine &ctx,
                        Entity e,
                        LidarDepth &lidar_depth,
                        LidarHitType &lidar_hit_type)
{
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (
            float(idx) / float(consts::numLidarSamples)) + math::pi / 2.f;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t,
                         &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar_depth.samples[idx] = 0.f;
            lidar_hit_type.samples[idx] = EntityType::None;
        } else {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            lidar_depth.samples[idx] = hit_t;
            lidar_hit_type.samples[idx] = entity_type;
        }
    };

    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for 
    // warp level programming
    int32_t idx = threadIdx.x % 32;

    if (idx < consts::numLidarSamples) {
        traceRay(idx);
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i);
    }
#endif
}

inline void dense1RewardSystem(Engine &ctx,
                               Position pos,
                               Progress &progress,
                               Reward &out_reward)
{ 
    const auto &lvl = ctx.singleton<Level>();
    const auto &episode_state = ctx.singleton<EpisodeState>();

    float dist_to_exit = pos.distance(ctx.get<Position>(lvl.exit));

    float min_dist = progress.minDistToExit;

    float reward = 0.f;

    if (dist_to_exit < min_dist) {
        float diff = min_dist - dist_to_exit;
        reward += diff * ctx.data().rewardPerDist;

        progress.minDistToExit = dist_to_exit;
    }

    if (episode_state.isDead) {
        reward -= 1.f;
    }

    if (episode_state.reachedExit) {
        reward += 1.f;

        if (episode_state.curLevel == ctx.data().levelsPerEpisode) {
            reward += 10.f;
        }
    }

    out_reward.v = reward;
}

// Dense reward for block-button puzzles
inline void dense2RewardSystem(Engine &ctx,
                               Position pos,
                               GrabState grab_state,
                               Progress &progress,
                               Reward &out_reward)
{ 
    const auto &lvl = ctx.singleton<Level>();
    const auto &episode_state = ctx.singleton<EpisodeState>();

    // This is super quick and dirty but because there is only one button right
    // now it will work fine. It will pick the info for the last button in
    // the world.
    Vector3 button_pos;
    bool button_is_pressed;
    ctx.iterateQuery(ctx.data().buttonQuery, [
        &button_pos, &button_is_pressed
    ](Position cur_button_pos, ButtonState cur_button_state) {
        button_pos = cur_button_pos;
        button_is_pressed = cur_button_state.isPressed;
    });

    bool is_grabbing = grab_state.constraintEntity != Entity::none();

    float reward = 0.f;
    if (button_is_pressed && progress.minDistToButton < 5.0f && !is_grabbing) {
        float dist_to_exit = pos.distance(ctx.get<Position>(lvl.exit));

        float min_dist = progress.minDistToExit;

        if (dist_to_exit < min_dist) {
            float diff = min_dist - dist_to_exit;
            reward += diff * ctx.data().rewardPerDist;

            progress.minDistToExit = dist_to_exit;
        }
        reward += 0.02f;
    } else {
        float dist_to_button = pos.distance(button_pos);
        if (is_grabbing) {
            // Track progress towards button
            float min_dist = progress.minDistToButton;

            if (dist_to_button < min_dist) {
                float diff = min_dist - dist_to_button;
                reward += diff * ctx.data().rewardPerDist;

                progress.minDistToButton = dist_to_button;
            }
            reward += 0.01f;
        } else {
            // Do nothing until grabbing block
            // Check if this is first step
            if (progress.minDistToButton == 0.f) {
                progress.minDistToButton = dist_to_button;
            }
        }
    }

    if (episode_state.isDead) {
        reward -= 1.f;
    }

    if (episode_state.reachedExit) {
        reward += 1.f;

        if (episode_state.curLevel == ctx.data().levelsPerEpisode) {
            reward += 10.f;
        }
    }

    out_reward.v = reward;
}

inline void perLevelRewardSystem(Engine &ctx,
                                 Reward &out_reward)
{ 
    const auto &episode_state = ctx.singleton<EpisodeState>();

    float reward = 0.f;

    if (episode_state.reachedExit) {
        reward += 1.f;

        if (episode_state.curLevel == ctx.data().levelsPerEpisode) {
            reward += 10.f;
        }
    }

    out_reward.v = reward;
}

inline void endOnlyRewardSystem(Engine &ctx,
                                Reward &out_reward)
{ 
    const auto &episode_state = ctx.singleton<EpisodeState>();

    float reward = 0.f;

    if (episode_state.reachedExit && 
            episode_state.curLevel == ctx.data().levelsPerEpisode) {
        reward += 10.f;
    }

    out_reward.v = reward;
}

inline void updateEpisodeStateSystem(Engine &ctx,
                                     EpisodeState &episode_state)
{
    episode_state.curStep += 1;

    if (episode_state.reachedExit) {
        episode_state.curLevel += 1;
    }

    WorldReset force_reset = ctx.singleton<WorldReset>();
    CheckpointReset ckpt_reset = ctx.singleton<CheckpointReset>();

    bool success = 
        episode_state.curLevel == ctx.data().levelsPerEpisode;

    episode_state.episodeFinished =
        force_reset.reset == 1 ||
        ckpt_reset.reset == 1 ||
        episode_state.isDead ||
        success
    ;

    if ((ctx.data().simFlags & SimFlags::IgnoreEpisodeLength) !=
            SimFlags::IgnoreEpisodeLength) {
        if (episode_state.curStep == ctx.data().episodeLen) {
            episode_state.episodeFinished = true;
        }
    }

    EpisodeResult &episode_result = ctx.singleton<EpisodeResult>();
    if (success) {
        episode_result.score = 
            float(ctx.data().episodeLen - episode_state.curStep) /
            ctx.data().episodeLen;
    } else {
        episode_result.score = 0.f;
    }
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void writeDoneSystem(Engine &ctx,
                            Done &done)
{
    const EpisodeState &episode_state = ctx.singleton<EpisodeState>();

    done.v = episode_state.episodeFinished ? 1 : 0;
}

static TaskGraphNodeID queueRewardSystem(const Sim::Config &cfg,
                                         TaskGraphBuilder &builder,
                                         Span<const TaskGraphNodeID> deps)
{
    TaskGraphNodeID reward_sys;
    switch (cfg.rewardMode) {
    case RewardMode::Dense1: {
        // Compute initial reward now that physics has updated the world state
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             dense1RewardSystem,
                Position,
                Progress,
                Reward
            >>(deps);
    } break;
    case RewardMode::Dense2: {
        // Compute initial reward now that physics has updated the world state
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             dense2RewardSystem,
                Position,
                GrabState,
                Progress,
                Reward
            >>(deps);
    } break;
    case RewardMode::PerLevel: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             perLevelRewardSystem,
                Reward
            >>(deps);
    } break;
    case RewardMode::EndOnly: {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
             endOnlyRewardSystem,
                Reward
            >>(deps);
    } break;
    default: MADRONA_UNREACHABLE();
    }

    return reward_sys;
}

// Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraphNodeID queueSortByWorld(TaskGraphBuilder &builder,
                                 Span<const TaskGraphNodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

static TaskGraphNodeID sortEntities(TaskGraphBuilder &builder,
                                    Span<const TaskGraphNodeID> deps)
{
    auto sort_sys = queueSortByWorld<PhysicsEntity>(
        builder, deps);
    sort_sys = queueSortByWorld<ButtonEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<DoorEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<RoomEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<ExitEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<EnemyEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<LavaEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<PatternEntity>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<CoopEntity>(
        builder, {sort_sys});

    return sort_sys;
}
#endif

static TaskGraphNodeID setupSimTasks(TaskGraphBuilder &builder,
                                     const Sim::Config &,
                                     Span<const TaskGraphNodeID> deps)
{
    // Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            Position,
            Rotation,
            Scale,
            ExternalForce,
            ExternalTorque
        >>(deps);

    // Scripted door behavior
    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine,
        setDoorPositionSystem,
            Position,
            OpenState
        >>({move_sys});
    
    // Build BVH for broadphase / raycasting
    auto broadphase_setup_sys = PhysicsSystem::setupBroadphaseTasks(
        builder, {set_door_pos_sys});

    // Grab action, post BVH build to allow raycasting
    auto grab_sys = builder.addToGraph<ParallelForNode<Engine,
        grabSystem,
            Entity,
            Position,
            Rotation,
            Action,
            GrabState
        >>({broadphase_setup_sys});

    // Jump action, post BVH build to allow raycasting
    auto jump_sys = builder.addToGraph<ParallelForNode<Engine,
        jumpSystem,
            Action,
            Position,
            Rotation,
            Scale,
            ExternalForce
        >>({grab_sys});

    auto enemy_act_sys = builder.addToGraph<ParallelForNode<Engine,
        enemyActSystem,
            Position,
            Rotation,
            EnemyState,
            ExternalForce,
            ExternalTorque
        >>({jump_sys});

    // Physics collision detection and solver
    auto substep_sys = PhysicsSystem::setupPhysicsStepTasks(builder,
        {enemy_act_sys}, consts::numPhysicsSubsteps);

    // Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {substep_sys});

    // Finalize physics subsystem work
    auto phys_done = PhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    auto enemy_post_phys_sys = builder.addToGraph<ParallelForNode<Engine,
        enemyPostPhysicsSystem, Position, Velocity, EnemyState>>(
            {phys_done});

    // Check buttons
    auto button_sys = builder.addToGraph<ParallelForNode<Engine,
        buttonSystem,
            Position,
            ButtonState
        >>({enemy_post_phys_sys});

    // Check patterns
    auto pattern_sys = builder.addToGraph<ParallelForNode<Engine,
        patternSystem,
            Position,
            PatternMatchState
        >>({button_sys});

    // Check chicken coop
    auto coop_sys = builder.addToGraph<ParallelForNode<Engine,
        coopSystem,
            Position,
            Scale, 
            CoopState
        >>({pattern_sys});
    
    // Set door to start opening if button conditions are met
    auto door_open_sys = builder.addToGraph<ParallelForNode<Engine,
        doorOpenSystem,
            OpenState,
            DoorProperties,
            ButtonListElem,
            PatternListElem,
            ChickenListElem
        >>({coop_sys});

    auto check_exit_sys = builder.addToGraph<ParallelForNode<Engine,
        checkExitSystem,
            Position,
            IsExit 
        >>({door_open_sys});

    auto lava_sys = builder.addToGraph<ParallelForNode<Engine,
        lavaSystem,
            Position,
            EntityExtents,
            IsLava 
        >>({check_exit_sys});

    return lava_sys;
}

static TaskGraphNodeID setupRewardAndDoneTasks(TaskGraphBuilder &builder,
                                               const Sim::Config &cfg,
                                               Span<const TaskGraphNodeID> deps)
{
    // Check if the episode is over
    auto update_episode_state_sys = builder.addToGraph<ParallelForNode<Engine,
        updateEpisodeStateSystem,
            EpisodeState
        >>(deps);

    TaskGraphNodeID reward_sys =
        queueRewardSystem(cfg, builder, {update_episode_state_sys});

    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        writeDoneSystem,
            Done
        >>({reward_sys});

    return done_sys;
}

static TaskGraphNodeID setupResetAndGenTasks(TaskGraphBuilder &builder,
                                             const Sim::Config &cfg,
                                             Span<const TaskGraphNodeID> deps)
{
    (void)cfg;

    // Conditionally reset the world if the episode is over
    auto new_episode_sys = builder.addToGraph<ParallelForNode<Engine,
        newEpisodeSystem,
           EpisodeState 
        >>(deps);

    // FIXME
    auto cleanup_level = builder.addToGraph<ParallelForNode<Engine,
        cleanupLevelSystem,
            EpisodeState 
        >>({new_episode_sys});

    cleanup_level = builder.addToGraph<ParallelForNode<Engine,
        deferredDeleteSystem,
            DeferredDelete
        >>({cleanup_level});

    cleanup_level = builder.addToGraph<
        ClearTmpNode<DeferredDeleteEntity>>({cleanup_level});

#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    cleanup_level = builder.addToGraph<RecycleEntitiesNode>({cleanup_level});
#endif

    auto gen_level_sys = builder.addToGraph<ParallelForNode<Engine,
        generateLevelSystem,
            EpisodeState
        >>({cleanup_level});

    auto post_gen = gen_level_sys;
#ifdef MADRONA_GPU_MODE
    post_gen = sortEntities(builder, {post_gen});
#endif

    // Conditionally load the checkpoint here including Done, Reward, 
    // and StepsRemaining. With Observations this should reconstruct 
    // all state that the training code needs.
    // This runs after the reset system resets the world.
    auto load_checkpoint_sys = builder.addToGraph<ParallelForNode<Engine,
        loadCheckpointSystem,
            CheckpointReset
        >>({post_gen});

    // Conditionally checkpoint the state of the system if we are on the Nth step.
    auto checkpoint_sys = builder.addToGraph<ParallelForNode<Engine,
        checkpointSystem,
            CheckpointSave
        >>({load_checkpoint_sys});

    // This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    auto post_reset_broadphase = PhysicsSystem::setupBroadphaseTasks(
        builder, {checkpoint_sys});

    return post_reset_broadphase;
}

static TaskGraphNodeID setupPostGenTasks(TaskGraphBuilder &builder,
                                         const Sim::Config &cfg,
                                         Span<const TaskGraphNodeID> deps)
{
    // The lidar system
#ifdef MADRONA_GPU_MODE
    // Note the use of CustomParallelForNode to create a taskgraph node
    // that launches a warp of threads (32) for each invocation (1).
    // The 32, 1 parameters could be changed to 32, 32 to create a system
    // that cooperatively processes 32 entities within a warp.
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            LidarDepth,
            LidarHitType
        >>(deps);

    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, deps);
    }


#ifndef MADRONA_GPU_MODE
    // Already sorted above.
    (void)lidar;
#endif

    // Finally, collect observations for the next step.
    return builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Rotation,
            GrabState,
            AgentTxfmObs,
            AgentInteractObs,
            AgentExitObs,
            StepsRemainingObservation,
            EntityPhysicsStateObsArray,
            EntityTypeObsArray,
            EntityAttributesObsArray
        >>({lidar});
}

static void setupInitTasks(TaskGraphBuilder &builder, const Sim::Config &cfg)
{
#ifdef MADRONA_GPU_MODE
    auto sort = queueSortByWorld<Agent>(builder, {});
    sort = sortEntities(builder, {sort});
#endif

    auto post_gen = setupResetAndGenTasks(builder, cfg, { 
#ifdef MADRONA_GPU_MODE
        sort,
#endif
    });

    setupPostGenTasks(builder, cfg, {post_gen});
}

static void setupStepTasks(TaskGraphBuilder &builder, const Sim::Config &cfg)
{
    auto sim_done = setupSimTasks(builder, cfg, {});
    auto reward_and_dones = setupRewardAndDoneTasks(builder, cfg, {sim_done});
    auto post_gen = setupResetAndGenTasks(builder, cfg, {reward_and_dones});
    setupPostGenTasks(builder, cfg, {post_gen});
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    TaskGraphBuilder &init_builder = taskgraph_mgr.init(TaskGraphID::Init);
    setupInitTasks(init_builder, cfg);

    TaskGraphBuilder &step_builder = taskgraph_mgr.init(TaskGraphID::Step);
    setupStepTasks(step_builder, cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    // Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    constexpr CountT max_total_entities = 
        consts::maxObjectsPerLevel + 10;

    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities);

    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;

    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    curWorldEpisode = 0;

    episodeLen = cfg.episodeLen;
    levelsPerEpisode = cfg.levelsPerEpisode;
    doorWidth = cfg.doorWidth;
    buttonWidth = cfg.buttonWidth;
    rewardPerDist = cfg.rewardPerDist;
    slackReward = cfg.slackReward;

    simFlags = cfg.simFlags;

    agent = createAgent(ctx);
    resetAgent(ctx, Vector3::zero(), 0.f, Vector3::zero());

    ctx.singleton<Level>() = {
        .rooms = { Entity::none() },
        .exit = ctx.makeRenderableEntity<ExitEntity>(),
    };

    ctx.singleton<EpisodeState>() = initEpisodeState();

    simEntityQuery = ctx.query<Entity, EntityType>();
    buttonQuery = ctx.query<Position, ButtonState>();
    patternQuery = ctx.query<Entity, PatternMatchState>();
    coopQuery = ctx.query<Position, CoopState>();

    ctx.singleton<CheckpointReset>().reset = 0;
    ctx.singleton<CheckpointSave>().save = 1;
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}

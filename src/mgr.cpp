#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace madPuzzle {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = consts::numAgents,
        .maxInstancesPerWorld = 1000,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    WorldReset *worldResetBuffer;
    CheckpointSave *worldSaveCheckpointBuffer;
    CheckpointReset *worldLoadCheckpointBuffer;
    Action *agentActionsBuffer;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                WorldReset *reset_buffer,
                CheckpointSave *checkpoint_save_buffer,
                CheckpointReset *checkpoint_load_buffer,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          worldResetBuffer(reset_buffer),
          worldSaveCheckpointBuffer(checkpoint_save_buffer),
          worldLoadCheckpointBuffer(checkpoint_load_buffer),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuRollout(cudaStream_t strm, void **buffers,
                            const TrainInterface &train_iface) = 0;
#endif

    virtual Tensor exportTensor(ExportID slot,
        Tensor::ElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   CheckpointSave *checkpoint_save_buffer,
                   CheckpointReset *checkpoint_load_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, checkpoint_save_buffer,
               checkpoint_load_buffer,action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final {}

    inline virtual void run()
    {
        cpuExec.run();
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuRollout(cudaStream_t strm, void **buffers,
                            const TrainInterface &train_iface)
    {
        (void)strm;
        (void)buffers;
        (void)train_iface;
        assert(false);
    }
#endif

    virtual inline Tensor exportTensor(ExportID slot,
        Tensor::ElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   CheckpointSave *checkpoint_save_buffer,
                   CheckpointReset *checkpoint_load_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, checkpoint_save_buffer,
               checkpoint_load_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void run()
    {
        gpuExec.run();
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuRollout(cudaStream_t strm, void **buffers,
                            const TrainInterface &train_iface)
    {
        auto numTensorBytes = [](const Tensor &t) {
            uint64_t num_items = 1;
            uint64_t num_dims = t.numDims();
            for (uint64_t i = 0; i < num_dims; i++) {
                num_items *= t.dims()[i];
            }

            return num_items * (uint64_t)t.numBytesPerItem();
        };

        auto copyToSim = [&strm, &numTensorBytes](const Tensor &dst, void *src) {
            uint64_t num_bytes = numTensorBytes(dst);

            REQ_CUDA(cudaMemcpyAsync(dst.devicePtr(), src, num_bytes,
                cudaMemcpyDeviceToDevice, strm));
        };

        auto copyFromSim = [&strm, &numTensorBytes](void *dst, const Tensor &src) {
            uint64_t num_bytes = numTensorBytes(src);

            REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
                cudaMemcpyDeviceToDevice, strm));
        };

        Span<const TrainInterface::NamedTensor> src_obs =
            train_iface.observations();
        Span<const TrainInterface::NamedTensor> src_stats =
            train_iface.stats();
        auto policy_assignments = train_iface.policyAssignments();

        void **input_buffers = buffers;
        void **output_buffers = buffers +
            src_obs.size() + src_stats.size() + 4;

        if (policy_assignments.has_value()) {
            output_buffers += 1;
        }

        CountT cur_idx = 0;

        copyToSim(train_iface.actions(), input_buffers[cur_idx++]);
        copyToSim(train_iface.resets(), input_buffers[cur_idx++]);

        gpuExec.runAsync(strm);

        copyFromSim(output_buffers[cur_idx++], train_iface.rewards());
        copyFromSim(output_buffers[cur_idx++], train_iface.dones());

        if (policy_assignments.has_value()) {
            copyFromSim(output_buffers[cur_idx++], *policy_assignments);
        }

        for (const TrainInterface::NamedTensor &t : src_obs) {
            copyFromSim(output_buffers[cur_idx++], t.hdl);
        }

        for (const TrainInterface::NamedTensor &t : src_stats) {
            copyFromSim(output_buffers[cur_idx++], t.hdl);
        }
    }
#endif

    virtual inline Tensor exportTensor(ExportID slot,
        Tensor::ElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static void loadRenderObjects(render::RenderManager &render_mgr)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::Door] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::PurpleDoor] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::BlueDoor] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::CyanDoor] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObject::Button] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)SimObject::Key] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::PurpleKey] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::BlueKey] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::CyanKey] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,},
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
        { render::rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f }, // Red
        { render::rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f }, // Yellow
        { render::rgb8ToFloat(20, 20, 230),   -1, 0.8f, 1.0f }, // blue
        { render::rgb8ToFloat(230, 20, 230),   -1, 0.8f, 1.0f }, // purple
        { render::rgb8ToFloat(20, 230, 230),   -1, 0.8f, 1.0f }, // cyan
    });

    // Override materials
    render_assets->objects[(CountT)SimObject::Cube].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Wall].meshes[0].materialIDX = 1;

    render_assets->objects[(CountT)SimObject::Door].meshes[0].materialIDX = 5;
    render_assets->objects[(CountT)SimObject::PurpleDoor].meshes[0].materialIDX = 8;
    render_assets->objects[(CountT)SimObject::BlueDoor].meshes[0].materialIDX = 7;
    render_assets->objects[(CountT)SimObject::CyanDoor].meshes[0].materialIDX = 9;

    render_assets->objects[(CountT)SimObject::Key].meshes[0].materialIDX = 5;
    render_assets->objects[(CountT)SimObject::PurpleKey].meshes[0].materialIDX = 8;
    render_assets->objects[(CountT)SimObject::BlueKey].meshes[0].materialIDX = 7;
    render_assets->objects[(CountT)SimObject::CyanKey].meshes[0].materialIDX = 9;

    render_assets->objects[(CountT)SimObject::Agent].meshes[0].materialIDX = 2;
    render_assets->objects[(CountT)SimObject::Agent].meshes[1].materialIDX = 3;
    render_assets->objects[(CountT)SimObject::Agent].meshes[2].materialIDX = 3;
    render_assets->objects[(CountT)SimObject::Button].meshes[0].materialIDX = 6;

    render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 4;

    render_mgr.loadObjects(render_assets->objects, materials, {
        { (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) /
           "smile.png").string().c_str() },
    });

    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    std::array<std::string, (size_t)SimObject::NumObjects - 1> asset_paths;
    asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObject::Door] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObject::PurpleDoor] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObject::BlueDoor] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObject::CyanDoor] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_collision_simplified.obj").string();
    asset_paths[(size_t)SimObject::Button] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::Key] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::PurpleKey] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::BlueKey] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::CyanKey] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();


    std::array<const char *, (size_t)SimObject::NumObjects - 1> asset_cstrs;
    for (size_t i = 0; i < asset_paths.size(); i++) {
        asset_cstrs[i] = asset_paths[i].c_str();
    }

    char import_err_buffer[4096];
    auto imported_src_hulls = imp::ImportedAssets::importFromDisk(
        asset_cstrs, import_err_buffer, true);

    if (!imported_src_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_src_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(
        (CountT)SimObject::NumObjects);

    auto setupHull = [&](SimObject obj_id,
                         float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)obj_id].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[(CountT)obj_id] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    setupHull(SimObject::Cube, 0.075f, {
        .muS = 0.5f,
        .muD = 0.75f,
    });

    setupHull(SimObject::Wall, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::Door, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::PurpleDoor, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::BlueDoor, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::CyanDoor, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::Agent, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::Button, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::Key, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::PurpleKey, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::BlueKey, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });
    setupHull(SimObject::CyanKey, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    src_objs[(CountT)SimObject::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };



    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // This is a bit hacky, but in order to make sure the agents
    // remain controllable by the policy, they are only allowed to
    // rotate around the Z axis (infinite inertia in x & y axes)
    rigid_body_assets.metadatas[
        (CountT)SimObject::Agent].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObject::Agent].mass.invInertiaTensor.y = 0.f;

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.simFlags = mgr_cfg.simFlags;
    sim_cfg.rewardMode = mgr_cfg.rewardMode;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);
    sim_cfg.buttonWidth = mgr_cfg.buttonWidth;
    sim_cfg.doorWidth = mgr_cfg.doorWidth;
    sim_cfg.rewardPerDist = mgr_cfg.rewardPerDist;
    sim_cfg.slackReward = mgr_cfg.slackReward;

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        // TODO: restore, 20
        PhysicsLoader phys_loader(ExecMode::CUDA, 13);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        CheckpointSave *checkpoint_save_buffer = 
            (CheckpointSave *)gpu_exec.getExported((uint32_t)ExportID::CheckpointSave);

        CheckpointReset *checkpoint_load_buffer = 
            (CheckpointReset *)gpu_exec.getExported((uint32_t)ExportID::CheckpointReset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            checkpoint_save_buffer,
            checkpoint_load_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        PhysicsLoader phys_loader(ExecMode::CPU, 13);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
        };

        WorldReset *world_reset_buffer = 
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        CheckpointSave *checkpoint_save_buffer = 
            (CheckpointSave *)cpu_exec.getExported((uint32_t)ExportID::CheckpointSave);

        CheckpointReset *checkpoint_load_buffer = 
            (CheckpointReset *)cpu_exec.getExported((uint32_t)ExportID::CheckpointReset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            checkpoint_save_buffer,
            checkpoint_load_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(cpu_exec),
        };

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.
    
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }

    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuRolloutStep(cudaStream_t strm, void **rollout_buffers)
{
    TrainInterface iface = trainInterface();
    impl_->gpuRollout(strm, rollout_buffers, iface);
}
#endif
Tensor Manager::checkpointResetTensor() const {
    return impl_->exportTensor(ExportID::CheckpointReset,
                               Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(CheckpointReset) / sizeof(int32_t)
                               });
}

Tensor Manager::checkpointTensor() const {
    return impl_->exportTensor(ExportID::Checkpoint,
                               Tensor::ElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(Checkpoint)
                               });
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(WorldReset) / sizeof(int32_t)
                               });
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, Tensor::ElementType::Int32,
        {
            impl_->cfg.numWorlds * consts::numAgents,
            sizeof(Action) / sizeof(int32_t),
        });
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   sizeof(Reward) / sizeof(float),
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   sizeof(Done) / sizeof(int32_t),
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   sizeof(SelfObservation) / sizeof(float),
                               });
}

Tensor Manager::partnerObservationsTensor() const
{
    return impl_->exportTensor(ExportID::PartnerObservations,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::numAgents - 1,
                                   sizeof(PartnerObservation) / sizeof(float),
                               });
}

Tensor Manager::roomEntityObservationsTensor() const
{
    return impl_->exportTensor(ExportID::RoomEntityObservations,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::maxObservationsPerAgent,
                                   sizeof(EntityObservation) / sizeof(float),
                               });
}

Tensor Manager::roomDoorObservationsTensor() const
{
    return impl_->exportTensor(ExportID::RoomDoorObservations,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::doorsPerRoom,
                                   sizeof(DoorObservation) / sizeof(float)
                               });
}

Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::numLidarSamples,
                                   sizeof(LidarSample) / sizeof(float),
                               });
}

Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   sizeof(StepsRemaining) / sizeof(int32_t),
                               });
}

Tensor Manager::agentIDTensor() const
{
    return impl_->exportTensor(ExportID::AgentID,
                               Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   sizeof(AgentID) / sizeof(int32_t),
                               });
}

TrainInterface Manager::trainInterface() const
{
    return TrainInterface {
        {
            { "self", selfObservationTensor() },
            { "partners", partnerObservationsTensor() },
            { "roomEntities", roomEntityObservationsTensor() },
            { "doors", roomDoorObservationsTensor() },
            { "lidar", lidarTensor() },
            { "stepsRemaining", stepsRemainingTensor() },
            { "agentID", agentIDTensor() },
        },
        actionTensor(),
        rewardTensor(),
        doneTensor(),
        resetTensor(),
        Optional<Tensor>::none(),
    };
}
Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, Tensor::ElementType::UInt8, {
        impl_->cfg.numWorlds,
        consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, Tensor::ElementType::Float32, {
        impl_->cfg.numWorlds,
        consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate,
                        int32_t interact)
{
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
        .interact = interact
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * consts::numAgents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

void Manager::setSaveCheckpoint(int32_t world_idx, int32_t value) 
{
    CheckpointSave save {
        value,
    };

    auto *save_ptr = impl_->worldSaveCheckpointBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(save_ptr, &save, sizeof(CheckpointSave),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *save_ptr = save;
    }
}

void Manager::triggerLoadCheckpoint(int32_t world_idx) 
{
    CheckpointReset load {
        1,
    };

    auto *load_ptr = impl_->worldLoadCheckpointBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(load_ptr, &load, sizeof(CheckpointReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *load_ptr = load;
    }
}
render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}

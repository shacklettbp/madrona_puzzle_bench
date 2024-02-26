#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>

#include <imgui.h>

using namespace madrona;
using namespace madrona::viz;

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace madPuzzle;

    constexpr int64_t num_views = 1;

    uint32_t num_worlds = 1;
    ExecMode exec_mode = ExecMode::CPU;

    auto usageErr = [argv]() {
        fprintf(stderr, "%s [NUM_WORLDS] [--backend cpu|cuda] [--record path] [--replay path] [--load-ckpt path] [--print-obs]\n", argv[0]);
        exit(EXIT_FAILURE);
    };

    bool num_worlds_set = false;

    char *record_log_path = nullptr;
    char *replay_log_path = nullptr;
    char *load_ckpt_path = nullptr;
    bool start_frozen = false;
    bool print_obs = false;

    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];

        if (arg[0] == '-' && arg[1] == '-') {
            arg += 2;

            if (!strcmp("backend", arg)) {
                i += 1;

                if (i == argc) {
                    usageErr();
                }

                char *value = argv[i];
                if (!strcmp("cpu", value)) {
                    exec_mode = ExecMode::CPU;
                } else if (!strcmp("cuda", value)) {
                    exec_mode = ExecMode::CUDA;
                } else {
                    usageErr();
                }
            } else if (!strcmp("record", arg)) {
                if (record_log_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                record_log_path = argv[i];
            } else if (!strcmp("replay", arg)) {
                if (replay_log_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                replay_log_path = argv[i];
            } else if (!strcmp("load-ckpt", arg)) {
                if (load_ckpt_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                load_ckpt_path = argv[i];
            } else if (!strcmp("freeze", arg)) {
                start_frozen = true;
            } else if (!strcmp("print-obs", arg)) {
                print_obs = true;
            } else {
                usageErr();
            }
        } else {
            if (num_worlds_set) {
                usageErr();
            }

            num_worlds_set = true;

            num_worlds = (uint32_t)atoi(arg);
        }
    }

    (void)record_log_path;

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 4);
    }

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif


    SimFlags flags = SimFlags::Default;

    if (!replay_log.has_value()) {
        flags |= SimFlags::IgnoreEpisodeLength;
    }

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Puzzle Bench", 2730, 1536);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .simFlags = flags,
        .rewardMode = RewardMode::Dense2,
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
        .episodeLen = 200,
        .levelsPerEpisode = 1,
        .buttonWidth = 1.3f,
        .doorWidth = 20.0f/3.0f,
        .rewardPerDist = 0.05f,
        .slackReward = -0.005f,
    });

    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, consts::worldWidth / 2.f, 30 };

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = start_frozen ? 0_u32 : 20_u32,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });

    // Replay step
    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            for (uint32_t j = 0; j < num_views; j++) {
                uint32_t base_idx = 0;
                base_idx = 4 * (cur_replay_step * num_views * num_worlds +
                    i * num_views + j);

                int32_t move_amount = (*replay_log)[base_idx];
                int32_t move_angle = (*replay_log)[base_idx + 1];
                int32_t turn = (*replay_log)[base_idx + 2];
                int32_t interact = (*replay_log)[base_idx + 3];

                printf("%d, %d: %d %d %d %d\n",
                       i, j, move_amount, move_angle, turn, interact);
                mgr.setAction(i, j, move_amount, move_angle, turn, interact);
            }
        }

        cur_replay_step++;

        return false;
    };

    // Printers
    //
    auto agent_txfm_tensor = mgr.agentTxfmObsTensor();
    auto agent_interact_tensor = mgr.agentInteractObsTensor();
    auto agent_lvl_type_tensor = mgr.agentLevelTypeObsTensor();
    auto agent_exit_tensor = mgr.agentExitObsTensor();

    auto entity_phys_tensor = mgr.entityPhysicsStateObsTensor();
    auto entity_type_tensor = mgr.entityTypeObsTensor();

    auto reward_tensor = mgr.rewardTensor();

    auto ckpt_tensor = mgr.checkpointTensor();
    
    auto agent_txfm_printer = agent_txfm_tensor.makePrinter();
    auto agent_interact_printer = agent_interact_tensor.makePrinter();
    auto agent_lvl_type_printer = agent_lvl_type_tensor.makePrinter();
    auto agent_exit_printer = agent_exit_tensor.makePrinter();

    auto entity_phys_printer = entity_phys_tensor.makePrinter();
    auto entity_type_printer = entity_type_tensor.makePrinter();

    auto reward_printer = reward_tensor.makePrinter();

    Checkpoint stashed_checkpoint;

    auto stashCheckpoint = [&](CountT world_idx)
    {
        auto dev_ptr = (Checkpoint *)ckpt_tensor.devicePtr();
        dev_ptr += world_idx;
        if (exec_mode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
            REQ_CUDA(cudaMemcpy(&stashed_checkpoint, dev_ptr,
                sizeof(Checkpoint), cudaMemcpyDeviceToHost));
#endif
        } else {
            stashed_checkpoint = *dev_ptr;
        }
    };

    auto loadStashedCheckpoint = [&](CountT world_idx)
    {
        auto dev_ptr = (Checkpoint *)ckpt_tensor.devicePtr();
        dev_ptr += world_idx;
        if (exec_mode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
            REQ_CUDA(cudaMemcpy(dev_ptr, &stashed_checkpoint,
                sizeof(Checkpoint), cudaMemcpyHostToDevice));
#endif
        } else {
            *dev_ptr = stashed_checkpoint;
        }
    };

    auto printObs = [&]() {
        if (!print_obs) {
            return;
        }

        printf("Agent Transform\n");
        agent_txfm_printer.print();

        printf("Agent Interact\n");
        agent_interact_printer.print();

        printf("Level Type\n");
        agent_lvl_type_printer.print();

        printf("To Exit\n");
        agent_exit_printer.print();

        printf("Entity Physics States\n");
        entity_phys_printer.print();

        printf("Entity Types\n");
        entity_type_printer.print();

        printf("Reward\n");
        reward_printer.print();

        printf("\n");
    };

    if (load_ckpt_path != nullptr) {
        std::ifstream ckpt_file(load_ckpt_path, std::ios::binary);
        assert(ckpt_file.is_open());
        HeapArray<Checkpoint> debug_ckpts(num_worlds);
        ckpt_file.read((char *)debug_ckpts.data(),
                       sizeof(Checkpoint) * num_worlds);

        for (CountT i = 0; i < num_worlds; i++) {
            mgr.triggerLoadCheckpoint(i);
            stashed_checkpoint = debug_ckpts[i];
            loadStashedCheckpoint(i);
        }

        mgr.step();

        printObs();
    }

    stashCheckpoint(0);

#ifdef MADRONA_CUDA_SUPPORT
    AgentTxfmObs *agent_txfm_readback = (AgentTxfmObs *)cu::allocReadback(
        sizeof(AgentTxfmObs) * num_views);

    Reward *reward_readback = (Reward *)cu::allocReadback(
        sizeof(Reward) * num_views);

    cudaStream_t readback_strm;
    REQ_CUDA(cudaStreamCreate(&readback_strm));
#endif

    // Main loop for the viewer viewer
    viewer.loop(
    [&](CountT world_idx, const Viewer::UserInput &input) 
    {
        using Key = Viewer::KeyboardKey;

        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx);
        }

        // Checkpointing
        if (input.keyHit(Key::Z)) {
            stashCheckpoint(world_idx);
            mgr.setSaveCheckpoint(world_idx, 1);
        }

        if (input.keyHit(Key::X)) {
            loadStashedCheckpoint(world_idx);
            mgr.triggerLoadCheckpoint(world_idx);
        }
    },
    [&](CountT world_idx, CountT agent_idx, const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 2;
        int32_t interact = 0;

        bool shift_pressed = input.keyPressed(Key::Shift);

        if (input.keyPressed(Key::W)) {
            y += 1;
        }
        if (input.keyPressed(Key::S)) {
            y -= 1;
        }

        if (input.keyPressed(Key::D)) {
            x += 1;
        }
        if (input.keyPressed(Key::A)) {
            x -= 1;
        }

        if (input.keyPressed(Key::Q)) {
            r += shift_pressed ? 2 : 1;
        }
        if (input.keyPressed(Key::E)) {
            r -= shift_pressed ? 2 : 1;
        }

        if (input.keyHit(Key::G)) {
            interact = 1;
        }

        if (input.keyPressed(Key::Space)) {
            interact = 2;
        }

        int32_t move_amount;
        if (x == 0 && y == 0) {
            move_amount = 0;
        } else if (shift_pressed) {
            move_amount = consts::numMoveAmountBuckets - 1;
        } else {
            move_amount = 1;
        }

        int32_t move_angle;
        if (x == 0 && y == 1) {
            move_angle = 0;
        } else if (x == 1 && y == 1) {
            move_angle = 1;
        } else if (x == 1 && y == 0) {
            move_angle = 2;
        } else if (x == 1 && y == -1) {
            move_angle = 3;
        } else if (x == 0 && y == -1) {
            move_angle = 4;
        } else if (x == -1 && y == -1) {
            move_angle = 5;
        } else if (x == -1 && y == 0) {
            move_angle = 6;
        } else if (x == -1 && y == 1) {
            move_angle = 7;
        } else {
            move_angle = 0;
        }

        mgr.setAction(world_idx, agent_idx, move_amount, move_angle, r, interact);
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        mgr.step();

        printObs();
    }, [&]() {
        CountT cur_world_id = viewer.getCurrentWorldID();
        CountT agent_world_offset = cur_world_id * num_views;

        AgentTxfmObs *agent_txfm_ptr =
            (AgentTxfmObs *)agent_txfm_tensor.devicePtr();

        Reward *reward_ptr = (Reward *)reward_tensor.devicePtr();

        agent_txfm_ptr += agent_world_offset;
        reward_ptr += agent_world_offset;

        if (exec_mode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
            cudaMemcpyAsync(agent_txfm_readback, agent_txfm_ptr,
                            sizeof(AgentTxfmObs) * num_views,
                            cudaMemcpyDeviceToHost, readback_strm);

            cudaMemcpyAsync(reward_readback, reward_ptr,
                            sizeof(Reward) * num_views,
                            cudaMemcpyDeviceToHost, readback_strm);

            REQ_CUDA(cudaStreamSynchronize(readback_strm));

            agent_txfm_ptr = agent_txfm_readback;
            reward_ptr = reward_readback;
#endif
        }

        for (int64_t i = 0; i < num_views; i++) {
            auto player_str = std::string("Agent ") + std::to_string(i);
            ImGui::Begin(player_str.c_str());

            const AgentTxfmObs &agent_txfm = agent_txfm_ptr[i];
            const Reward &reward = reward_ptr[i];

            ImGui::Text("Position:      (%.1f, %.1f, %.1f)",
                agent_txfm.localRoomPos.x,
                agent_txfm.localRoomPos.y,
                agent_txfm.localRoomPos.z);
            ImGui::Text("Rotation:      %.2f",
                agent_txfm.theta);
            ImGui::Text("Reward:    %.3f",
                reward.v);

            ImGui::End();
        }
    });

}

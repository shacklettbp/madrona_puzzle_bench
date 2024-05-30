#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

using namespace madrona::py;

namespace madPuzzle {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_puzzle_bench, m) {
    // Each simulator has a madrona submodule that includes base types
    // like Tensor and PyExecMode.
    setupMadronaSubmodule(m);

    nb::enum_<SimFlags>(m, "SimFlags", nb::is_arithmetic())
        .value("Default", SimFlags::Default)
        .value("UseFixedWorld", SimFlags::UseFixedWorld)
    ;

    nb::enum_<RewardMode>(m, "RewardMode")
        .value("Dense1", RewardMode::Dense1)
        .value("Dense2", RewardMode::Dense2)
        .value("PerLevel", RewardMode::PerLevel)
        .value("EndOnly", RewardMode::EndOnly)
        .value("LavaButton", RewardMode::LavaButton)
        .value("PerGoal", RewardMode::PerGoal)
    ;

    auto mgr_class = nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            uint32_t sim_flags,
                            RewardMode reward_mode,
                            bool enable_batch_renderer,
                            uint32_t episode_len,
                            uint32_t levels_per_episode,
                            float button_width,
                            float door_width,
                            float reward_per_dist,
                            float slack_reward,
                            uint32_t num_pbt_policies) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .simFlags = SimFlags(sim_flags),
                .rewardMode = reward_mode,
                .enableBatchRenderer = enable_batch_renderer,
                .episodeLen = episode_len,
                .levelsPerEpisode = levels_per_episode,
                .buttonWidth = button_width,
                .doorWidth = door_width,
                .rewardPerDist = reward_per_dist,
                .slackReward = slack_reward,
                .numPBTPolicies = num_pbt_policies,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("sim_flags"),
           nb::arg("reward_mode"),
           nb::arg("enable_batch_renderer") = false,
           nb::arg("episode_len"),
           nb::arg("levels_per_episode"),
           nb::arg("button_width"),
           nb::arg("door_width"),
           nb::arg("reward_per_dist"),
           nb::arg("slack_reward"),
           nb::arg("num_pbt_policies") = 0)
        .def("init", &Manager::init)
        .def("step", &Manager::step)
        .def("checkpoint_reset_tensor", &Manager::checkpointResetTensor)
        .def("checkpoint_tensor", &Manager::checkpointTensor)
        .def("reset_tensor", &Manager::resetTensor)
        .def("goal_tensor", &Manager::goalTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("agent_txfm_obs_tensor",
             &Manager::agentTxfmObsTensor)
        .def("agent_interact_obs_tensor",
             &Manager::agentInteractObsTensor)
        .def("agent_level_type_obs_tensor",
             &Manager::agentLevelTypeObsTensor)
        .def("agent_exit_obs_tensor",
             &Manager::agentExitObsTensor)
        .def("entity_physics_state_obs_tensor",
             &Manager::entityPhysicsStateObsTensor)
        .def("entity_type_obs_tensor",
             &Manager::entityTypeObsTensor)
        .def("entity_attr_obs_tensor",
             &Manager::entityAttributesObsTensor)
        .def("lidar_depth_tensor", &Manager::lidarDepthTensor)
        .def("lidar_hit_type", &Manager::lidarHitTypeTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("jax", JAXInterface::buildEntry<
                &Manager::trainInterface,
                &Manager::init,
                &Manager::step
#ifdef MADRONA_CUDA_SUPPORT
                ,
                &Manager::gpuStreamInit,
                &Manager::gpuStreamStep,
                &Manager::gpuStreamLoadCheckpoints,
                &Manager::gpuStreamGetCheckpoints
#endif
             >())
    ;
}

}

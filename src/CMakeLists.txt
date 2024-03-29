set(SIMULATOR_SRCS
    sim_flags.hpp sim_flags.inl
    sim.hpp sim.inl sim.cpp
    level_gen.hpp level_gen.cpp
    types.hpp types.cpp
)

add_library(mad_escape_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(mad_escape_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(mad_escape_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(mad_escape_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        mad_escape_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            mad_escape_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(mad_escape_mgr PRIVATE
        mad_escape_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(mad_escape_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_puzzle_bench
    bindings.cpp
)

target_link_libraries(madrona_puzzle_bench PRIVATE
    mad_escape_mgr
)

if (TARGET madrona_viz)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer PRIVATE
        madrona_mw_core mad_escape_mgr madrona_viz)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()

if (TARGET madrona_cuda)
    target_link_libraries(viewer PRIVATE madrona_cuda)
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless madrona_mw_core mad_escape_mgr)

# Shared by the full build and the dependency-free host-only build.
find_package(Threads REQUIRED)
# temp_file_test — TempFile is pure POSIX (mkstemp/pread/pwrite/fallocate/
# statfs) with no SYCL in it. It used to reach the class by linking the whole
# pos2_gpu_host library, which dragged AdaptiveCpp in and put the test out of
# reach of any environment without a SYCL toolchain — CI included. Compiling
# the one TU it actually needs keeps it runnable everywhere.
add_executable(temp_file_test tools/parity/temp_file_test.cpp
                              src/host/TempFile.cpp)
target_include_directories(temp_file_test PRIVATE src)
target_compile_features(temp_file_test PRIVATE cxx_std_20)
set_target_properties(temp_file_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# spill_engine_test — the disk-offload I/O engine's ticket protocol.
#
# The engine is the spill path's silent-corruption surface: a wait that returns
# one job early substitutes another chunk's bytes over a range SpillCoverage
# considers written, so nothing downstream can tell. It was untestable while it
# lived in an anonymous namespace inside GpuPipeline.cpp; SpillEngine.hpp now
# injects the two staging allocations and the one blocking copy it needed from
# SYCL (SpillHostOps), so the protocol runs here over malloc + memcpy with no
# GPU, no driver and no AdaptiveCpp. Threads and real temp-dir I/O, nothing
# else.
add_executable(spill_engine_test tools/parity/spill_engine_test.cpp
                                 src/host/TempFile.cpp)
target_include_directories(spill_engine_test PRIVATE src)
target_compile_features(spill_engine_test PRIVATE cxx_std_20)
target_link_libraries(spill_engine_test PRIVATE Threads::Threads)
set_target_properties(spill_engine_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# spill_coverage_test — the spill path's read guard. SpillCoverage.hpp is
# header-only and free of SYCL/GPU deps precisely so this builds and runs
# anywhere with no device present: it is pure interval arithmetic, and it is
# the only thing standing between a spill bug and a silently short .plot2.
add_executable(spill_coverage_test tools/parity/spill_coverage_test.cpp)
target_include_directories(spill_coverage_test PRIVATE src)
target_compile_features(spill_coverage_test PRIVATE cxx_std_20)
set_target_properties(spill_coverage_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# host_guard_test — the pinned-host redzone canaries. Same argument as
# spill_coverage_test: HostGuard.hpp is header-only and device-free, and a
# canary that silently never fires is indistinguishable from a clean run in
# any end-to-end plot test. The deliberate-overrun cases here are what makes
# a quiet soak under XCHPLOT2_HOST_GUARD mean anything.
add_executable(host_guard_test tools/parity/host_guard_test.cpp)
target_include_directories(host_guard_test PRIVATE src)
target_compile_features(host_guard_test PRIVATE cxx_std_20)
set_target_properties(host_guard_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# vram_probe_test — validation of a Level Zero Sysman memory reading.
# VramProbe.hpp is header-only and device-free so this runs anywhere, which is
# the entire point: the code it guards executes only on Intel hardware and
# parses a hand-declared ABI, so on any other box "it compiles" is the only
# feedback available. That is how the first version shipped a bound that
# rejected every reading on the target card. The B580 case is the real
# measurement from that host.
add_executable(vram_probe_test tools/parity/vram_probe_test.cpp)
target_include_directories(vram_probe_test PRIVATE src)
target_compile_features(vram_probe_test PRIVATE cxx_std_20)
set_target_properties(vram_probe_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# host_spill_policy_test — the host-RAM disk-offload budget policy. Same
# argument as the two above: HostRamPolicy is pure integer arithmetic with no
# SYCL/CUDA/device probe, so every branch is reachable here on any machine.
# Reaching them for real needs a GPU, a k=28-sized host and a specific free-RAM
# reading, which is why this policy shipped with no test until now — and it is
# the code that decides whether this process gets OOM-killed or plots.
add_executable(host_spill_policy_test tools/parity/host_spill_policy_test.cpp
                                      src/host/HostRamPolicy.cpp)
target_include_directories(host_spill_policy_test PRIVATE src)
target_compile_features(host_spill_policy_test PRIVATE cxx_std_20)
set_target_properties(host_spill_policy_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# bench_stats_test — the multi-worker bench arithmetic (per-worker warmup
# exclusion, drain-tail truncation, sum-of-rates aggregate). BenchStats.hpp is
# header-only and free of SYCL/GPU deps precisely so this can be built and run
# anywhere, with no device present: it is pure arithmetic over synthetic
# timelines, and it is the only thing standing between a plausible-looking
# throughput number and a wrong one.
add_executable(bench_stats_test tools/parity/bench_stats_test.cpp)
target_include_directories(bench_stats_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(bench_stats_test PRIVATE cxx_std_20)

# numa_topology_test — the sysfs cpulist grammar that decides which cores a CPU
# worker is pinned to. Same reasoning as bench_stats_test: NumaTopology is free
# of SYCL/GPU deps so this runs on any host, and it covers the multi-socket
# cpulist shapes ("0-15,32-47") that a single-socket dev box cannot produce and
# therefore cannot catch by running the real thing.
add_executable(numa_topology_test
    tools/parity/numa_topology_test.cpp
    src/host/NumaTopology.cpp)
target_include_directories(numa_topology_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(numa_topology_test PRIVATE cxx_std_20)
add_executable(vram_budget_test tools/parity/vram_budget_test.cpp)
target_include_directories(vram_budget_test PRIVATE src)
target_compile_features(vram_budget_test PRIVATE cxx_std_20)

add_executable(cli_host_test tools/parity/cli_host_test.cpp tools/xchplot2/cli.cpp
    src/host/ConfigFile.cpp src/host/BatchManifest.cpp src/host/NumaTopology.cpp src/host/Cancel.cpp)
target_include_directories(cli_host_test PRIVATE src keygen-rs/include)
target_compile_features(cli_host_test PRIVATE cxx_std_20)
target_compile_definitions(cli_host_test PRIVATE XCHPLOT2_VERSION="${PROJECT_VERSION}")
target_link_libraries(cli_host_test PRIVATE Threads::Threads)

add_executable(pipeline_control_test tools/parity/pipeline_control_test.cpp
    src/host/MultiGpuPipelineParallel.cpp src/host/Cancel.cpp)
target_include_directories(pipeline_control_test BEFORE PRIVATE tools/parity/pipeline_control_test src)
target_compile_features(pipeline_control_test PRIVATE cxx_std_20)
target_link_libraries(pipeline_control_test PRIVATE Threads::Threads)

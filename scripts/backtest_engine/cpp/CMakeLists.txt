# Standalone build of the header-only C++ engines: unit tests and micro-benchmarks.
#
# The Python extension is built by setup.py (pybind11); this project compiles the same headers without Python,
# with strict warnings and optional sanitizers, so that the C++ code is checked on its own.
#
#   cmake -S scripts/backtest_engine/cpp -B build/cpp -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   cmake --build build/cpp
#   ctest --test-dir build/cpp --output-on-failure
#   build/cpp/core_bench                      # timings (build with -DCMAKE_BUILD_TYPE=Release)
#
# Sanitizers: -DBT_SANITIZE=address,undefined (memory errors and undefined behaviour) or -DBT_SANITIZE=thread
# (data races in the parallel loops). They cannot be combined in one build.

cmake_minimum_required(VERSION 3.16)
project(backtest_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BT_WERROR "Treat compiler warnings as errors" ON)
set(BT_SANITIZE "" CACHE STRING "Comma-separated sanitizers passed to -fsanitize (for example address,undefined)")

find_package(Threads REQUIRED)

add_library(bt_engine INTERFACE)
target_include_directories(bt_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bt_engine INTERFACE Threads::Threads)

if(MSVC)
  target_compile_options(bt_engine INTERFACE /W4 /permissive- $<$<BOOL:${BT_WERROR}>:/WX>)
else()
  target_compile_options(bt_engine INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
                         -Wold-style-cast -Wnull-dereference -Wdouble-promotion
                         $<$<BOOL:${BT_WERROR}>:-Werror>)
  # Bounds-checked standard containers in debug-like builds.
  target_compile_definitions(bt_engine INTERFACE $<$<NOT:$<CONFIG:Release>>:_GLIBCXX_ASSERTIONS>)
endif()

if(BT_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "BT_SANITIZE is supported with GCC and Clang only")
  endif()
  target_compile_options(bt_engine INTERFACE -fsanitize=${BT_SANITIZE} -fno-omit-frame-pointer
                         -fno-sanitize-recover=all)
  target_link_options(bt_engine INTERFACE -fsanitize=${BT_SANITIZE})
endif()

enable_testing()

add_executable(core_tests tests/test_core.cpp)
target_link_libraries(core_tests PRIVATE bt_engine)
add_test(NAME core_tests COMMAND core_tests)

add_executable(core_bench bench/bench_core.cpp)
target_link_libraries(core_bench PRIVATE bt_engine)
add_test(NAME core_bench_smoke COMMAND core_bench --quick)

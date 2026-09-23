# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Layerwise split KV cache — migrated from upstream PR #2260
# Adapted for Iluvatar BI-V100 + Qwen3.5 (Qwen3.6-35B-A3B)
#
# Build options:
#   -DUSE_ILU=ON    Enable Iluvatar BI-V100 specific code paths
#   -DBUILD_TESTING Enable unit tests
#
# Without -DUSE_ILU, ILU-specific compilation units (mapping_ilu,
# layerwise_split_engine_ext, layerwise_split_master) are excluded
# from the build, matching the upstream pattern where device-specific
# code is guarded by #if defined(USE_NPU) || defined(USE_MLU).

cmake_minimum_required(VERSION 3.18)
project(layerwise_split_kv_cache LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED IMPORTED_TARGET libglog)
find_package(gflags REQUIRED)
find_package(GTest QUIET)

# ---------- common source files (all platforms) ----------
set(COMMON_SOURCES
  framework/config/parallel_config.cpp
  framework/kv_cache/kv_cache_estimation.cpp
  config/parallel_config_layerwise.cpp
  framework/kv_cache/kv_cache_layerwise.cpp
  framework/kv_cache/kv_cache_estimation_layerwise.cpp
  runtime/worker_layerwise_init.cpp
)

# ---------- ILU-specific source files ----------
# These files are wrapped in #if defined(USE_ILU) guards.
# Including them without -DUSE_ILU produces empty translation units
# (harmless but wasteful), so we conditionally add them.
set(ILU_SOURCES
  framework/parallel_state/mapping_ilu.cpp
  distributed_runtime/layerwise_split_engine_ext.cpp
  distributed_runtime/layerwise_split_master.cpp
)

# ---------- layerwise split KV cache library ----------
add_library(layerwise_split_kv STATIC
  ${COMMON_SOURCES}
  ${ILU_SOURCES}
)

target_include_directories(layerwise_split_kv PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(layerwise_split_kv PUBLIC
  gflags
  PkgConfig::GLOG
)

# Iluvatar BI-V100 build: define USE_ILU so the #if guards activate
# the ILU-specific code paths (mapping_ilu, engine_ext, master).
option(USE_ILU "Enable Iluvatar BI-V100 specific code paths" OFF)
if(USE_ILU)
  target_compile_definitions(layerwise_split_kv PUBLIC USE_ILU)
  message(STATUS "USE_ILU=ON: ILU-specific code paths enabled")
else()
  message(STATUS "USE_ILU=OFF: building without ILU device support")
endif()

# ---------- tests ----------
if(BUILD_TESTING)
  if(NOT GTest_FOUND)
    find_library(GTEST_LIB gtest)
    find_library(GTEST_MAIN_LIB gtest_main)
  endif()

  add_executable(test_layerwise_split
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/core/test_layerwise_split_kv_cache.cpp
  )

  if(GTest_FOUND)
    target_link_libraries(test_layerwise_split PRIVATE
      layerwise_split_kv GTest::gtest_main gflags PkgConfig::GLOG)
  else()
    target_link_libraries(test_layerwise_split PRIVATE
      layerwise_split_kv ${GTEST_LIB} ${GTEST_MAIN_LIB} gflags PkgConfig::GLOG pthread)
  endif()

  add_test(NAME LayerwiseSplitKVTests COMMAND test_layerwise_split)
endif()

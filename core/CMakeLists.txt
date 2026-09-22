# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Layerwise split KV cache — migrated from upstream PR #2260
# Adapted for Iluvatar BI-V100 + Qwen3.5 (Qwen3.6-35B-A3B)

cmake_minimum_required(VERSION 3.18)
project(layerwise_split_kv_cache LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED IMPORTED_TARGET libglog)
find_package(GTest QUIET)

# ---------- layerwise split KV cache library ----------
add_library(layerwise_split_kv STATIC
  # Upstream-migrated files
  framework/config/parallel_config.cpp
  framework/kv_cache/kv_cache_estimation.cpp
  # BI-V100 adaptation files
  config/parallel_config_layerwise.cpp
  framework/kv_cache/kv_cache_layerwise.cpp
  framework/kv_cache/kv_cache_estimation_layerwise.cpp
  framework/parallel_state/mapping_ilu.cpp
  distributed_runtime/layerwise_split_engine_ext.cpp
  distributed_runtime/layerwise_split_master.cpp
  runtime/worker_layerwise_init.cpp
)

target_include_directories(layerwise_split_kv PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(layerwise_split_kv PUBLIC
  gflags
  PkgConfig::GLOG
)

# Iluvatar BI-V100 build: define USE_ILU
if(USE_ILU)
  target_compile_definitions(layerwise_split_kv PUBLIC USE_ILU)
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

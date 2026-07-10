cmake_minimum_required(VERSION 3.15)
project(LBMSolver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Homebrew libomp path on Apple Silicon
if(APPLE)
  set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()

find_package(OpenMP REQUIRED)

# ------------------------------------------------------------------
# Main solver executable
# ------------------------------------------------------------------
add_executable(LBM_Engine src/main.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_Engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_Engine PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Unit tests (Google Test)
# ------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(LBM_Tests src/lbm_test.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_Tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_Tests PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Custom target: run simulation with default parameters
# ------------------------------------------------------------------
add_custom_target(sim
  COMMAND LBM_Engine
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ------------------------------------------------------------------
# Custom target: build docs site locally
# ------------------------------------------------------------------
add_custom_target(serve-docs
  COMMAND python3 -m http.server 8765 -d ${CMAKE_SOURCE_DIR}/docs
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

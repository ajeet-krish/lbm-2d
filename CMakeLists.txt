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
# Main solver executable (cylinder flow)
# ------------------------------------------------------------------
add_executable(LBM_Engine src/main.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_Engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_Engine PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Lid-driven cavity executable
# ------------------------------------------------------------------
add_executable(LBM_Cavity src/cavity.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_Cavity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_Cavity PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Backward-facing step executable
# ------------------------------------------------------------------
add_executable(LBM_Step src/step.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_Step PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_Step PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Urban canyon flow executable
# ------------------------------------------------------------------
add_executable(LBM_UrbanCanyon src/urban_canyon.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_UrbanCanyon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_UrbanCanyon PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Building downwash executable
# ------------------------------------------------------------------
add_executable(LBM_Downwash src/downwash.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_Downwash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_Downwash PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Square cylinder (ERCOFTAC 043)
# ------------------------------------------------------------------
add_executable(LBM_SquareCylinder src/square_cylinder.cpp src/lbm.hpp src/lbm_types.hpp src/geometry.hpp)
target_include_directories(LBM_SquareCylinder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_SquareCylinder PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Flat plate boundary layer (primary validation case)
# ------------------------------------------------------------------
add_executable(LBM_FlatPlate src/flat_plate.cpp src/lbm.hpp src/lbm_types.hpp src/geometry.hpp)
target_include_directories(LBM_FlatPlate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_FlatPlate PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# AMR test executable (cylinder with adaptive mesh refinement)
# ------------------------------------------------------------------
add_executable(LBM_AMR src/amr_test.cpp src/lbm.hpp src/lbm_types.hpp src/amr.hpp)
target_include_directories(LBM_AMR PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_AMR PRIVATE OpenMP::OpenMP_CXX)

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
# Custom targets
# ------------------------------------------------------------------
add_custom_target(sim
  COMMAND LBM_Engine
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(cavity
  COMMAND LBM_Cavity
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(step
  COMMAND LBM_Step
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(urban
  COMMAND LBM_UrbanCanyon
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(downwash
  COMMAND LBM_Downwash
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(amr
  COMMAND LBM_AMR
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(square_cylinder
  COMMAND LBM_SquareCylinder
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(flat_plate
  COMMAND LBM_FlatPlate
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ------------------------------------------------------------------
# Periodic hills (canonical LES benchmark)
# ------------------------------------------------------------------
add_executable(LBM_PeriodicHills src/periodic_hills.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_PeriodicHills PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_PeriodicHills PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Cylinder near a wall (ground effect)
# ------------------------------------------------------------------
add_executable(LBM_CylinderNearWall src/cylinder_near_wall.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_CylinderNearWall PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_CylinderNearWall PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Side-by-side cylinders (interference)
# ------------------------------------------------------------------
add_executable(LBM_SideBySide src/side_by_side_cylinders.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_SideBySide PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_SideBySide PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Rotating cylinder (Magnus effect)
# ------------------------------------------------------------------
add_executable(LBM_RotatingCylinder src/rotating_cylinder.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_RotatingCylinder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_RotatingCylinder PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Multi-stage orifice plate (staggered)
# ------------------------------------------------------------------
add_executable(LBM_OrificePlate src/orifice_plate.cpp src/lbm.hpp src/lbm_types.hpp)
target_include_directories(LBM_OrificePlate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_OrificePlate PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Heated cylinder (thermal LBM, Upgrade 4)
# ------------------------------------------------------------------
add_executable(LBM_HeatedCylinder src/heated_cylinder.cpp src/lbm.hpp src/lbm_types.hpp src/thermal.hpp src/ibm.hpp src/wall_functions.hpp)
target_include_directories(LBM_HeatedCylinder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_HeatedCylinder PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Airfoil with Immersed Boundary Method (Upgrade 1 Tier 3)
# ------------------------------------------------------------------
add_executable(LBM_AirfoilIBM src/airfoil_ibm.cpp src/lbm.hpp src/lbm_types.hpp src/thermal.hpp src/ibm.hpp src/wall_functions.hpp)
target_include_directories(LBM_AirfoilIBM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_AirfoilIBM PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------
# Natural convection cavity (Thermal LBM, Upgrade 4)
# ------------------------------------------------------------------
add_executable(LBM_ThermalCavity src/thermal_cavity.cpp src/lbm.hpp src/lbm_types.hpp src/thermal.hpp src/ibm.hpp src/wall_functions.hpp)
target_include_directories(LBM_ThermalCavity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(LBM_ThermalCavity PRIVATE OpenMP::OpenMP_CXX)

add_custom_target(orifice
  COMMAND LBM_OrificePlate
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(serve-docs
  COMMAND python3 -m http.server 8765 -d ${CMAKE_SOURCE_DIR}/docs
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(heated_cylinder
  COMMAND LBM_HeatedCylinder
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.24)

project(
  NESO-RNG-Toolkit-NP-Example
  VERSION 0.0.1
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# find MPI
find_package(MPI REQUIRED)
find_package(NESO-Particles REQUIRED)
find_package(NESO-RNG-Toolkit REQUIRED)

# put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
endif()

set(EXECUTABLE np-example)
set(EXECUTABLE_SOURCE main.cpp)

add_executable(${EXECUTABLE} ${EXECUTABLE_SOURCE})

target_include_directories(${EXECUTABLE} PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(
  ${EXECUTABLE}
  PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
target_link_libraries(${EXECUTABLE} PUBLIC NESO-Particles::NESO-Particles
    NESO-RNG-Toolkit::NESO-RNG-Toolkit)

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})

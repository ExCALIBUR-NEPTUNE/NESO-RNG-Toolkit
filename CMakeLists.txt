cmake_minimum_required(VERSION 3.24)

project(
  NESO-RNG-Toolkit
  VERSION 0.1.0
  LANGUAGES CXX C)

include(GNUInstallDirs)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-file-list.cmake)

option(
  NESO_RNG_TOOLKIT_ENABLE_FIND_SYCL
  "Enabling search for a SYCL implementation if add_sycl_to_target is not found."
  ON)

# This means that when the tests and lib are installed they have rpath set for
# the installed lib/binary.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Find SYCL
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SYCL.cmake)
if(NESO_RNG_TOOLKIT_ENABLE_FIND_SYCL)
  find_sycl_if_required()
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SRC_FILES ${SRC_DIR}/neso_rng_toolkit.cpp)
set(SRC_FILES_IGNORE "")
check_added_file_list(${SRC_DIR} cpp "${SRC_FILES}" "${SRC_FILES_IGNORE}")

# Create a list of the header files.
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INCLUDE_DIR_NESO_RNG_TOOLKIT ${INCLUDE_DIR}/neso_rng_toolkit)
set(HEADER_FILES ${INCLUDE_DIR}/neso_rng_toolkit.hpp)

# Check that the files added above are not missing any files in the include
# directory.
set(HEADER_FILES_IGNORE "")
check_added_file_list(${INCLUDE_DIR} hpp "${HEADER_FILES}"
                      "${HEADER_FILES_IGNORE}")

# Create shared library
add_library(NESO-RNG-Toolkit SHARED ${SRC_FILES} ${HEADER_FILES})
# Alias the name to the namespaces name. Can use in subdirectory or via Confiig
# files with namespace
add_library(NESO-RNG-Toolkit::NESO-RNG-Toolkit ALIAS NESO-RNG-Toolkit)

# Set standard
set_property(TARGET NESO-RNG-Toolkit PROPERTY CXX_STANDARD 17)

# Makes it easy to install + adds the files to the INCLUDE property of the lib
# i.e. don't need target_include_dir.. also no GLOBS
target_sources(
  NESO-RNG-Toolkit
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         ${INCLUDE_DIR}
         FILES
         ${HEADER_FILES})

# set the version infomation
target_compile_definitions(
  NESO-RNG-Toolkit
  PUBLIC NESO_RNG_TOOLKIT_VERSION_MAJOR=${CMAKE_PROJECT_VERSION_MAJOR})
target_compile_definitions(
  NESO-RNG-Toolkit
  PUBLIC NESO_RNG_TOOLKIT_VERSION_MINOR=${CMAKE_PROJECT_VERSION_MINOR})
target_compile_definitions(
  NESO-RNG-Toolkit
  PUBLIC NESO_RNG_TOOLKIT_VERSION_PATCH=${CMAKE_PROJECT_VERSION_PATCH})

option(NESO_RNG_TOOLKIT_ENABLE_TESTS "Build unit tests for this project." ON)

add_sycl_to_target(TARGET NESO-RNG-Toolkit SOURCES ${SRC_FILES})

# Include the tests
if(NESO_RNG_TOOLKIT_ENABLE_TESTS)
  # set build type
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
  endif()
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

# install the headers
install(
  TARGETS NESO-RNG-Toolkit
  EXPORT ${PROJECT_NAME}_Targets
  FILE_SET public_headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
# create the config.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)
# install the config scripts
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SYCL.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-file-list.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
export(
  EXPORT ${PROJECT_NAME}_Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake"
  NAMESPACE ${PROJECT_NAME}::)

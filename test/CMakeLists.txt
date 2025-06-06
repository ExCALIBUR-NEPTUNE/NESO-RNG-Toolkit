find_package(GTest QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        FALSE)
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_SRCS
    ${TEST_DIR}/test_version_print.cpp ${TEST_DIR}/test_platform_stdlib.cpp
    ${TEST_DIR}/test_platform_onedpl.cpp)

# Check that the files added above are not missing any files in the test
# directory.
set(TEST_SRCS_IGNORE ${TEST_DIR}/main.cpp)
check_added_file_list(${CMAKE_CURRENT_SOURCE_DIR} cpp "${TEST_SRCS}"
                      "${TEST_SRCS_IGNORE}")

set(EXECUTABLE testNESORNGToolkit)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# Build the tests individually
include(GoogleTest)
# Add a a target so we can build all the tests "make tests"
add_custom_target(tests)
foreach(TEST ${TEST_SRCS})
  get_filename_component(TEST_NAME ${TEST} NAME_WLE)
  message(STATUS "Found test - ${TEST_NAME}")
  set(TEST_LIST ${TEST_LIST} ${TEST})

  set(TEST_SOURCES ${TEST_MAIN} ${TEST})
  add_executable(${TEST_NAME} ${TEST_SOURCES})
  target_link_libraries(${TEST_NAME} PRIVATE NESO-RNG-Toolkit GTest::gtest)
  target_compile_definitions(${TEST_NAME}
                             PRIVATE NESO_RNG_TOOLKIT_TEST_COMPILATION)
  target_compile_definitions(
    ${TEST_NAME} PRIVATE CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})

  target_compile_options(${TEST_NAME} PRIVATE "-Wpedantic;-Wall;-Wextra")
  # exclude from all so indiviual tests aren't built for install
  set_target_properties(${TEST_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  # Make usre the dummy target depends on each tests
  add_dependencies(tests ${TEST_NAME})
  add_sycl_to_target(TARGET ${TEST_NAME} SOURCES ${TEST_SOURCES})
  gtest_add_tests(TARGET ${TEST_NAME})

  set_target_properties(
    ${TEST_NAME}
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)
endforeach()

# Build a global test suite
add_executable(${EXECUTABLE} ${TEST_MAIN} ${TEST_LIST})

target_link_libraries(${EXECUTABLE} PRIVATE NESO-RNG-Toolkit GTest::gtest)
target_compile_definitions(${EXECUTABLE}
                           PRIVATE NESO_RNG_TOOLKIT_TEST_COMPILATION)
target_compile_definitions(${EXECUTABLE}
                           PRIVATE CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})
target_compile_options(${EXECUTABLE} PRIVATE "-Wpedantic;-Wall;-Wextra")

set_target_properties(
  ${EXECUTABLE}
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN} ${TEST_LIST})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN})

# install the combined test binary
install(TARGETS ${EXECUTABLE} DESTINATION bin)

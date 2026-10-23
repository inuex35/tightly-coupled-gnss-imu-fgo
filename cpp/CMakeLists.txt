cmake_minimum_required(VERSION 3.16)
project(gnss_fgo_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# GTSAM with the custom GNSS factors (DD PR/CP, SD Doppler, NHC):
# inuex35/gtsam, branch custom/develop.
find_package(GTSAM REQUIRED)

# RTKLIB (rtklibexplorer demo5): RINEX input, broadcast ephemerides,
# satellite positions. Pinned; point RTKLIB_SOURCE_DIR at a checkout to
# build offline.
set(RTKLIB_SOURCE_DIR "" CACHE PATH "Local RTKLIB checkout (empty: fetch)")
set(RTKLIB_GIT_TAG 75a2e56275485b21a67bd35bc94bbeb8936e1a74 CACHE STRING "RTKLIB demo5 commit")
if(RTKLIB_SOURCE_DIR)
  set(rtklib_SOURCE_DIR ${RTKLIB_SOURCE_DIR})
else()
  include(FetchContent)
  # Sources only: RTKLIB's own CMake project is not built (the
  # SOURCE_SUBDIR does not exist).
  FetchContent_Declare(rtklib
    GIT_REPOSITORY https://github.com/rtklibexplorer/RTKLIB.git
    GIT_TAG ${RTKLIB_GIT_TAG}
    SOURCE_SUBDIR do-not-build)
  FetchContent_MakeAvailable(rtklib)
endif()

# The sources are copied next to a patched rtklib.h: demo5 stops BeiDou
# at C46, which drops the BeiDou-3 GEOs C59-C63 from both the observation
# and the navigation decoding.
set(RTKLIB_SRC ${CMAKE_BINARY_DIR}/rtklib_src)
set(RTKLIB_C_SOURCES rtkcmn.c rinex.c ephemeris.c preceph.c sbas.c trace.c)
foreach(f ${RTKLIB_C_SOURCES})
  configure_file(${rtklib_SOURCE_DIR}/src/${f} ${RTKLIB_SRC}/${f} COPYONLY)
endforeach()
file(READ ${rtklib_SOURCE_DIR}/src/rtklib.h RTKLIB_H)
string(REGEX REPLACE "#define MAXPRNCMP +46" "#define MAXPRNCMP   63" RTKLIB_H_PATCHED "${RTKLIB_H}")
if(RTKLIB_H_PATCHED STREQUAL RTKLIB_H)
  message(FATAL_ERROR "rtklib.h: MAXPRNCMP patch did not apply")
endif()
file(CONFIGURE OUTPUT ${RTKLIB_SRC}/rtklib.h CONTENT "${RTKLIB_H_PATCHED}" @ONLY)
list(TRANSFORM RTKLIB_C_SOURCES PREPEND ${RTKLIB_SRC}/)
add_library(rtklib STATIC ${RTKLIB_C_SOURCES})
target_include_directories(rtklib PUBLIC ${RTKLIB_SRC})
# All constellations; three frequency slots plus extended slots so that
# every code of a band survives decoding (the signal plan picks one).
target_compile_definitions(rtklib PUBLIC
  ENAGLO ENAGAL ENAQZS ENACMP ENAIRN NFREQ=3 NEXOBS=6)
set_target_properties(rtklib PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rtklib PRIVATE -w)
endif()
find_package(Threads REQUIRED)
target_link_libraries(rtklib PUBLIC m Threads::Threads)

add_library(gnss_fgo STATIC
  src/gnss_fgo/gnss.cpp
  src/gnss_fgo/rinex.cpp
  src/gnss_fgo/frontend.cpp
  src/gnss_fgo/config.cpp
  src/gnss_fgo/mlambda.cpp
  src/gnss_fgo/utils.cpp
  src/gnss_fgo/factors.cpp
  src/gnss_fgo/ar.cpp
  src/gnss_fgo/integrity.cpp
  src/gnss_fgo/pipeline.cpp
  src/gnss_fgo/initialization.cpp
  src/gnss_fgo/runner.cpp)
target_include_directories(gnss_fgo PUBLIC src)
target_link_libraries(gnss_fgo PUBLIC gtsam PRIVATE rtklib)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(gnss_fgo PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(run_imu_gnss_tc apps/run_imu_gnss_tc.cpp)
target_link_libraries(run_imu_gnss_tc PRIVATE gnss_fgo)

option(GNSS_FGO_BUILD_TESTS "Build unit tests" ON)
if(GNSS_FGO_BUILD_TESTS)
  enable_testing()
  add_executable(gnss_fgo_tests tests/test_main.cpp)
  target_link_libraries(gnss_fgo_tests PRIVATE gnss_fgo)
  add_test(NAME gnss_fgo_tests COMMAND gnss_fgo_tests)
endif()

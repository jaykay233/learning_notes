cmake_minimum_required(VERSION 3.16)
project(add_custom)

# Official AscendC kernel-launch style (CANN 8.x / 9.x):
#   cmake -B build -DRUN_MODE=cpu -DSOC_VERSION=Ascend910B1
#   cmake --build build -j

set(RUN_MODE "cpu" CACHE STRING "cpu | sim | npu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "e.g. Ascend910B1 / Ascend910B2")

if(DEFINED CMAKE_ASC_RUN_MODE AND NOT CMAKE_ASC_RUN_MODE STREQUAL "")
  set(RUN_MODE "${CMAKE_ASC_RUN_MODE}" CACHE STRING "" FORCE)
endif()
# Map old dav-* flag to a SOC_VERSION default when only arch was passed
if(DEFINED CMAKE_ASC_ARCHITECTURES AND CMAKE_ASC_ARCHITECTURES MATCHES "dav-2201")
  if(SOC_VERSION STREQUAL "Ascend910B1" OR SOC_VERSION STREQUAL "")
    set(SOC_VERSION "Ascend910B1" CACHE STRING "" FORCE)
  endif()
endif()

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR ASCEND_CANN_PACKAGE_PATH STREQUAL "")
  if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  elseif(DEFINED ENV{ASCEND_INSTALL_PATH} AND NOT "$ENV{ASCEND_INSTALL_PATH}" STREQUAL "")
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_INSTALL_PATH}")
  else()
    message(FATAL_ERROR "ASCEND_HOME_PATH / ASCEND_INSTALL_PATH empty; source set_env.sh first")
  endif()
endif()

# Locate ascendc_kernel_cmake (CANN 9.x prefers tools/ or aarch64-linux/)
set(ASCENDC_CMAKE_DIR "")
foreach(cand
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake"
    "$ENV{ASCENDC_CMAKE_DIR}")
  if(EXISTS "${cand}/ascendc.cmake")
    set(ASCENDC_CMAKE_DIR "${cand}")
    break()
  endif()
endforeach()

if(ASCENDC_CMAKE_DIR STREQUAL "")
  message(FATAL_ERROR
    "Cannot find ascendc_kernel_cmake/ascendc.cmake under ${ASCEND_CANN_PACKAGE_PATH}.\n"
    "On macOS: use Linux/Docker (see ../../README.md).")
endif()

message(STATUS "RUN_MODE=${RUN_MODE}")
message(STATUS "SOC_VERSION=${SOC_VERSION}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "ASCENDC_CMAKE_DIR=${ASCENDC_CMAKE_DIR}")

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels_${RUN_MODE} SHARED
  add_custom.cpp
)

add_executable(add_custom_${RUN_MODE}
  main.cpp
)

target_compile_options(add_custom_${RUN_MODE} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)

target_include_directories(add_custom_${RUN_MODE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(add_custom_${RUN_MODE} PRIVATE
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/tools/cpudebug/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib
  ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/${SOC_VERSION}
  ${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(add_custom_${RUN_MODE} PRIVATE
  ascendc_kernels_${RUN_MODE}
)

if(RUN_MODE STREQUAL "cpu")
  target_link_libraries(add_custom_${RUN_MODE} PRIVATE
    ascendc_acl_stub
    tikicpulib::${SOC_VERSION}
    cpudebug
    c_sec
    runtime
    register
    error_manager
    profapi
    ge_common_base
    unified_dlog
    mmpa
    dl
    ascend_dump
  )
elseif(RUN_MODE STREQUAL "npu")
  target_link_libraries(add_custom_${RUN_MODE} PRIVATE ascendcl)
endif()

set_target_properties(add_custom_${RUN_MODE} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

message(STATUS "Target: add_custom_${RUN_MODE}")

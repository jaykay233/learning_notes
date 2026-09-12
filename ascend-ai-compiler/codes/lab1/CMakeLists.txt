cmake_minimum_required(VERSION 3.20)
project(lab1-eliminate-identity LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_library(Lab1Dialect
  lib/LabDialect.cpp
  lib/EliminateIdentityPass.cpp
)
target_link_libraries(Lab1Dialect
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRArithDialect
)

add_executable(lab1-opt tools/lab1-opt/lab1-opt.cpp)
target_link_libraries(lab1-opt
  PRIVATE
  Lab1Dialect
  MLIROptLib
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
)

mlir_check_all_link_libraries(lab1-opt)

set_target_properties(lab1-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

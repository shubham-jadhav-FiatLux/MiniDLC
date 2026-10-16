cmake_minimum_required(VERSION 3.20)
project(MiniDLC VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Options
option(DLC_BUILD_TOOLS "Build MiniDLC compiler tools (e.g. dlc-opt)" ON)
option(DLC_BUILD_TESTS "Build MiniDLC tests" ON)
option(DLC_BUILD_PYTHON "Build Python bindings" OFF)

# 1. Core IR Library
add_library(dlc_ir
    lib/ir/Type.cpp
    lib/ir/Value.cpp
    lib/ir/Operation.cpp
    lib/ir/Ops.cpp
    lib/ir/Function.cpp
    lib/ir/Module.cpp
    lib/ir/Builder.cpp
    lib/ir/Verifier.cpp
    lib/ir/Printer.cpp
    lib/ir/Parser.cpp
)
target_include_directories(dlc_ir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 2. Passes Library
add_library(dlc_passes
    lib/passes/Pass.cpp
    lib/passes/ShapeInference.cpp
    lib/passes/Canonicalize.cpp
    lib/passes/CSE.cpp
    lib/passes/DCE.cpp
    lib/passes/Autodiff.cpp
    lib/passes/Fusion.cpp
    lib/passes/MemoryPlanning.cpp
)
target_link_libraries(dlc_passes PUBLIC dlc_ir)

# 3. Loop IR Library
add_library(dlc_loopir
    lib/loopir/LoopIR.cpp
    lib/loopir/Lowering.cpp
    lib/loopir/LoopPasses.cpp
)
target_link_libraries(dlc_loopir PUBLIC dlc_ir)

# 4. Runtime Library
add_library(dlc_runtime
    lib/runtime/Arena.cpp
    lib/runtime/ThreadPool.cpp
    lib/runtime/ExecutionPlan.cpp
)
target_include_directories(dlc_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 5. Backends Library
add_library(dlc_backends
    lib/backends/llvm/LLVMCodeGen.cpp
    lib/backends/llvm/LLVMBackend.cpp
    lib/backends/cuda/CUDABackend.cpp
)
target_link_libraries(dlc_backends PUBLIC dlc_loopir dlc_runtime ${CMAKE_DL_LIBS})


# 6. Tools: dlc-opt
if(DLC_BUILD_TOOLS)
    add_executable(dlc-opt
        tools/dlc-opt/dlc-opt.cpp
    )
    target_link_libraries(dlc-opt PRIVATE dlc_backends dlc_loopir dlc_passes dlc_runtime dlc_ir)
endif()

# 7. Unit Tests
if(DLC_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    # Prevent overriding parent compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    # Try finding system GTest or FetchContent
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        # Note: If offline or build without network, fallback to simple custom runner
        # or FetchContent when network is available
    endif()

    add_executable(dlc_unit_tests
        tests/unit/test_ir.cpp
        tests/unit/test_passes.cpp
        tests/unit/test_runtime.cpp
    )
    target_link_libraries(dlc_unit_tests PRIVATE dlc_backends dlc_passes dlc_runtime dlc_loopir dlc_ir)

    add_test(NAME dlc_unit_tests COMMAND dlc_unit_tests)
endif()

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
option(DLC_BUILD_PYTHON "Build Python bindings (not implemented yet)" OFF)
option(DLC_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(DLC_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

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
target_link_libraries(dlc_runtime PUBLIC Threads::Threads)

# 5. Backends Library
add_library(dlc_backends
    lib/backends/llvm/LLVMCodeGen.cpp
    lib/backends/llvm/LLVMBackend.cpp
    lib/backends/cuda/CUDABackend.cpp
    lib/backends/cuda/CUDACodeGen.cpp
)
target_link_libraries(dlc_backends PUBLIC dlc_loopir dlc_runtime ${CMAKE_DL_LIBS})


# 6. Tools: dlc-opt and Microbenchmarks
if(DLC_BUILD_TOOLS)
    add_executable(dlc-opt
        tools/dlc-opt/dlc-opt.cpp
    )
    target_link_libraries(dlc-opt PRIVATE dlc_backends dlc_loopir dlc_passes dlc_runtime dlc_ir)

    add_executable(dlc_fma_peak
        benchmarks/fma_peak.cpp
    )
    target_compile_options(dlc_fma_peak PRIVATE -mavx2 -mfma)
    target_link_libraries(dlc_fma_peak PRIVATE Threads::Threads)
endif()

# 7. Tests: GoogleTest unit tests and lit/FileCheck golden tests
if(DLC_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(dlc_unit_tests
        tests/unit/test_ir.cpp
        tests/unit/test_passes.cpp
        tests/unit/test_runtime.cpp
        tests/unit/test_cuda.cpp
        tests/unit/test_known_bugs.cpp
    )
    target_link_libraries(dlc_unit_tests PRIVATE dlc_backends dlc_passes dlc_runtime dlc_loopir dlc_ir GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(dlc_unit_tests)

    # Golden IR tests: lit drives dlc-opt and checks its output with LLVM's FileCheck
    find_program(LIT_EXECUTABLE NAMES lit llvm-lit)
    find_program(FILECHECK_EXECUTABLE NAMES FileCheck HINTS /usr/lib/llvm-23/bin /usr/lib/llvm-18/bin)
    if(DLC_BUILD_TOOLS AND LIT_EXECUTABLE AND FILECHECK_EXECUTABLE)
        add_test(NAME lit
            COMMAND ${LIT_EXECUTABLE} -sv ${CMAKE_CURRENT_SOURCE_DIR}/tests/lit
                    --param dlc_opt=$<TARGET_FILE:dlc-opt>
                    --param filecheck=${FILECHECK_EXECUTABLE})
    else()
        message(STATUS "lit tests disabled: they need dlc-opt, lit and FileCheck")
    endif()
endif()

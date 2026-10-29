cmake_minimum_required(VERSION 3.20)
project(MiniDLC VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# The libraries are also linked into the _minidlc Python module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Options
option(DLC_BUILD_TOOLS "Build MiniDLC compiler tools (e.g. dlc-opt)" ON)
option(DLC_BUILD_TESTS "Build MiniDLC tests" ON)
option(DLC_BUILD_PYTHON "Build the _minidlc Python module (needs Python and nanobind)" ON)
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
    lib/ir/OpRegistry.cpp
    lib/ir/SlotTracker.cpp
)
target_include_directories(dlc_ir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 2. Passes Library
add_library(dlc_passes
    lib/passes/Pass.cpp
    lib/passes/ShapeInference.cpp
    lib/passes/Canonicalize.cpp
    lib/passes/CSE.cpp
    lib/passes/DCE.cpp
    lib/passes/Activity.cpp
    lib/passes/Autodiff.cpp
    lib/passes/Optimizer.cpp
    lib/passes/Fusion.cpp
    lib/passes/CostModel.cpp
    lib/passes/BufferAssignment.cpp
)
target_link_libraries(dlc_passes PUBLIC dlc_ir)

# 3. Loop IR v2: lowering with the target schedules, and the interpreter
add_library(dlc_loopir
    lib/loopir/LoopIR.cpp
    lib/loopir/Lowering.cpp
    lib/loopir/Interpreter.cpp
    lib/loopir/Analysis.cpp
)
target_link_libraries(dlc_loopir PUBLIC dlc_passes)

# 4. Runtime: thread pool, plans and executables (no LLVM)
add_library(dlc_runtime
    lib/runtime/Arena.cpp
    lib/runtime/ThreadPool.cpp
    lib/runtime/Executable.cpp
)
target_include_directories(dlc_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dlc_runtime PUBLIC dlc_ir Threads::Threads)

# 5. LLVM backend (emitter, O3 pipeline, ORC LLJIT) and the compiler driver
find_package(LLVM REQUIRED CONFIG)
if(LLVM_VERSION_MAJOR LESS 23)
    message(FATAL_ERROR "MiniDLC needs LLVM 23 or newer, found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")
add_library(dlc_llvm
    lib/backends/llvm/Emitter.cpp
    lib/backends/llvm/CpuJit.cpp
)
target_include_directories(dlc_llvm SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(dlc_llvm PUBLIC ${LLVM_DEFINITIONS})
target_link_libraries(dlc_llvm PUBLIC dlc_loopir LLVM)

add_library(dlc_compiler lib/compiler/Compiler.cpp)
target_link_libraries(dlc_compiler PUBLIC dlc_llvm dlc_loopir dlc_runtime dlc_passes dlc_ir)


# 6. The _minidlc Python module (nanobind), written next to the pure-Python package
if(DLC_BUILD_PYTHON)
    find_package(Python 3.11 COMPONENTS Interpreter Development.Module)
    if(Python_FOUND)
        execute_process(COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
                        OUTPUT_VARIABLE DLC_NANOBIND_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    if(DLC_NANOBIND_DIR)
        list(APPEND CMAKE_PREFIX_PATH ${DLC_NANOBIND_DIR})
        find_package(nanobind CONFIG REQUIRED)
        nanobind_add_module(_minidlc NB_STATIC python/bindings/module.cpp)
        target_link_libraries(_minidlc PRIVATE dlc_compiler)
        set(DLC_PYTHON_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/minidlc CACHE PATH "Where _minidlc is written")
        set_target_properties(_minidlc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DLC_PYTHON_MODULE_DIR})
    else()
        message(STATUS "_minidlc disabled: it needs Python >= 3.11 with nanobind (pip install nanobind)")
    endif()
endif()

# 7. Tools: dlc-opt and Microbenchmarks
if(DLC_BUILD_TOOLS)
    add_executable(dlc-opt
        tools/dlc-opt/dlc-opt.cpp
    )
    target_link_libraries(dlc-opt PRIVATE dlc_compiler)

    add_executable(dlc_fma_peak
        benchmarks/fma_peak.cpp
    )
    target_compile_options(dlc_fma_peak PRIVATE -mavx2 -mfma)
    target_link_libraries(dlc_fma_peak PRIVATE Threads::Threads)
endif()

# 8. Tests: GoogleTest unit tests and lit/FileCheck golden tests
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
        tests/unit/test_loopir.cpp
        tests/unit/test_known_bugs.cpp
    )
    target_link_libraries(dlc_unit_tests PRIVATE dlc_compiler GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(dlc_unit_tests PROPERTIES TIMEOUT 120)

    # Golden IR tests: lit drives dlc-opt and checks its output with LLVM's FileCheck
    find_program(LIT_EXECUTABLE NAMES lit llvm-lit)
    find_program(FILECHECK_EXECUTABLE NAMES FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR} /usr/lib/llvm-23/bin)
    if(DLC_BUILD_TOOLS AND LIT_EXECUTABLE AND FILECHECK_EXECUTABLE)
        add_test(NAME lit
            COMMAND ${LIT_EXECUTABLE} -sv ${CMAKE_CURRENT_SOURCE_DIR}/tests/lit
                    --param dlc_opt=$<TARGET_FILE:dlc-opt>
                    --param filecheck=${FILECHECK_EXECUTABLE})
    else()
        message(STATUS "lit tests disabled: they need dlc-opt, lit and FileCheck")
    endif()

    # The Python suites that drive dlc-opt: translation validation of every graph pass, the C++ VJPs against finite
    # differences, and the training-step graph against eager training (all in the NumPy reference interpreter)
    find_package(Python3 COMPONENTS Interpreter)
    if(DLC_BUILD_TOOLS AND Python3_Interpreter_FOUND)
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, pytest"
                        RESULT_VARIABLE DLC_PYTHON_DEPS_MISSING OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(DLC_BUILD_TOOLS AND Python3_Interpreter_FOUND AND DLC_PYTHON_DEPS_MISSING EQUAL 0)
        add_test(NAME python_compiler_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
                    tests/python/test_translation_validation.py tests/python/test_training_graph.py
                    tests/python/test_cpu_backend.py tests/python/test_cpu_training.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(python_compiler_tests PROPERTIES ENVIRONMENT "DLC_OPT=$<TARGET_FILE:dlc-opt>")
    else()
        message(STATUS "Python compiler tests disabled: they need dlc-opt and a Python 3 with numpy and pytest")
    endif()
endif()

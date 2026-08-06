cmake_minimum_required(VERSION 3.10)

project(CpliceAI C)

enable_testing()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/sanitizers-cmake/cmake" ${CMAKE_MODULE_PATH})

include_directories(externals)

find_package(Sanitizers)

find_package(PkgConfig REQUIRED)
pkg_check_modules(HTSLIB REQUIRED htslib)

# Inference backend: ONNX Runtime (default, supports GPU via CUDA/TensorRT execution
# providers) or TensorFlow (kept buildable for parity testing against the ONNX Runtime
# migration; see tests/backend_parity.bats). Neither backend's headers leak into predict.h
# (Model is opaque), so this only affects the `predict` library and its two consumers below.
set(CPLICEAI_INFERENCE_BACKEND "onnxruntime" CACHE STRING "Inference backend: onnxruntime or tensorflow")
set_property(CACHE CPLICEAI_INFERENCE_BACKEND PROPERTY STRINGS onnxruntime tensorflow)

if(CPLICEAI_INFERENCE_BACKEND STREQUAL "onnxruntime")
    pkg_check_modules(ONNXRUNTIME QUIET onnxruntime)
    if(NOT ONNXRUNTIME_FOUND)
        # No .pc file to rely on here (unlike the prebuilt release tarball, which we write
        # one for by hand in the Dockerfile) -- fall back to a manual search, respecting an
        # ONNXRUNTIME_ROOT cache var/env var for wherever ORT was installed. Two layouts are
        # covered: the prebuilt release tarball's flat <root>/include/onnxruntime_c_api.h,
        # and a from-source `cmake --install`, which commonly nests headers under
        # <root>/include/onnxruntime/core/session/ instead -- and may install libraries to
        # lib64 rather than lib depending on the distro's install conventions.
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
            HINTS "${ONNXRUNTIME_ROOT}" "$ENV{ONNXRUNTIME_ROOT}" /opt/onnxruntime /usr/local
            PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
        find_library(ONNXRUNTIME_LIBRARY onnxruntime
            HINTS "${ONNXRUNTIME_ROOT}" "$ENV{ONNXRUNTIME_ROOT}" /opt/onnxruntime /usr/local
            PATH_SUFFIXES lib lib64)
        if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
            message(FATAL_ERROR "onnxruntime not found via pkg-config or ONNXRUNTIME_ROOT "
                "(searched ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}). Set "
                "-DONNXRUNTIME_ROOT=/path/to/onnxruntime, or -DONNXRUNTIME_INCLUDE_DIR=... "
                "and -DONNXRUNTIME_LIBRARY=... directly if your install layout is nonstandard "
                "(find the actual paths with: "
                "find <install-prefix> -iname onnxruntime_c_api.h -o -iname 'libonnxruntime.so*').")
        endif()
        set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
        set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
    endif()

    set(CPLICEAI_INFERENCE_SOURCES src/predict_ort.c src/predict_common.c)
    set(CPLICEAI_INFERENCE_LIBS ${ONNXRUNTIME_LIBRARIES})
    set(CPLICEAI_INFERENCE_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIRS})
    set(CPLICEAI_INFERENCE_LIBRARY_DIRS ${ONNXRUNTIME_LIBRARY_DIRS})
elseif(CPLICEAI_INFERENCE_BACKEND STREQUAL "tensorflow")
    pkg_check_modules(TENSORFLOW REQUIRED tensorflow)
    set(CPLICEAI_INFERENCE_SOURCES src/predict_tf.c src/predict_common.c)
    set(CPLICEAI_INFERENCE_LIBS ${TENSORFLOW_LIBRARIES})
    set(CPLICEAI_INFERENCE_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIRS})
    set(CPLICEAI_INFERENCE_LIBRARY_DIRS ${TENSORFLOW_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "Unknown CPLICEAI_INFERENCE_BACKEND: ${CPLICEAI_INFERENCE_BACKEND} (expected onnxruntime or tensorflow)")
endif()

# pkg_check_modules only gives bare `-l` names in *_LIBRARIES, not the -L search path
# (that's *_LIBRARY_DIRS) -- needed here since, unlike /usr/local/lib, ONNX Runtime's
# install location isn't necessarily on the linker's default search path.
link_directories(${CPLICEAI_INFERENCE_LIBRARY_DIRS})

# Shared static libraries used across the executables
add_library(logging STATIC src/logging/log.c)
add_library(predict STATIC ${CPLICEAI_INFERENCE_SOURCES})
target_include_directories(predict PRIVATE ${CPLICEAI_INFERENCE_INCLUDE_DIRS})
add_library(utils STATIC src/utils.c)
add_library(range STATIC src/range.c)
add_library(gene_regions STATIC src/gene_regions.c)
add_library(reference STATIC src/reference.c)
add_library(gene_reference STATIC src/gene_reference.c)
add_library(variant_io STATIC src/variant_input.c src/variant_output.c)

set(CPLICEAI_COMMON_LIBS logging predict utils range gene_regions reference)

# Executables
add_executable(cpliceai_reference src/bin/build_reference_scores.c)
target_link_libraries(cpliceai_reference PRIVATE
    ${CPLICEAI_COMMON_LIBS}
    ${CPLICEAI_INFERENCE_LIBS}
    ${HTSLIB_LIBRARIES})

add_executable(cpliceai_predict_variant src/bin/predict_variant.c)
target_link_libraries(cpliceai_predict_variant PRIVATE
    ${CPLICEAI_COMMON_LIBS}
    gene_reference
    variant_io
    ${CPLICEAI_INFERENCE_LIBS}
    ${HTSLIB_LIBRARIES})

add_executable(cpliceai_predict_gene src/bin/predict_gene.c)
target_link_libraries(cpliceai_predict_gene PRIVATE
    ${CPLICEAI_COMMON_LIBS}
    gene_reference
    variant_io
    ${CPLICEAI_INFERENCE_LIBS}
    ${HTSLIB_LIBRARIES})

foreach(cpliceai_target cpliceai_reference cpliceai_predict_variant cpliceai_predict_gene)
    add_sanitizers(${cpliceai_target})
endforeach()

# End-to-end tests (bats-core), wired into CTest and `make check`
find_program(BATS_EXECUTABLE bats REQUIRED)

if(CPLICEAI_INFERENCE_BACKEND STREQUAL "onnxruntime")
    set(CPLICEAI_TEST_MODEL_DIR "${CMAKE_SOURCE_DIR}/models/onnx")
else()
    set(CPLICEAI_TEST_MODEL_DIR "${CMAKE_SOURCE_DIR}/models/tf")
endif()

set(CPLICEAI_TEST_ENVIRONMENT
    "CPLICEAI_REFERENCE_BIN=$<TARGET_FILE:cpliceai_reference>"
    "CPLICEAI_PREDICT_VARIANT_BIN=$<TARGET_FILE:cpliceai_predict_variant>"
    "CPLICEAI_PREDICT_GENE_BIN=$<TARGET_FILE:cpliceai_predict_gene>"
    "MODEL_DIR=${CPLICEAI_TEST_MODEL_DIR}"
    "FIXTURES_DIR=${CMAKE_SOURCE_DIR}/tests/fixtures"
    # Force CPU EP for deterministic, reproducible test results regardless of what
    # hardware `ctest` happens to run on; GPU behavior is validated separately.
    "CPLICEAI_ORT_EP=cpu"
    # Multi-threaded conv reduction is not bit-reproducible run-to-run (thread completion
    # order affects floating-point summation order); pin to 1 thread so the 2dp assertions
    # below never flake on a rounding-boundary ULP difference.
    "CPLICEAI_ORT_INTRA_OP_THREADS=1")

# Fast tests: argument parsing only, binaries exit before loading any models.
foreach(cli_test reference predict_variant predict_gene)
    set(test_name "cli_${cli_test}")
    add_test(NAME ${test_name}
        COMMAND ${BATS_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/cli/${cli_test}.bats)
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "${CPLICEAI_TEST_ENVIRONMENT}"
        LABELS "cli"
        TIMEOUT 60)
endforeach()

# Slow test: runs real SpliceAI inference (model loading x3 invocations)
# over a tiny synthetic fixture, end to end across all three binaries.
add_test(NAME e2e_pipeline
    COMMAND ${BATS_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/pipeline.bats)
set_tests_properties(e2e_pipeline PROPERTIES
    ENVIRONMENT "${CPLICEAI_TEST_ENVIRONMENT}"
    LABELS "e2e"
    TIMEOUT 300)

# Chunking is only implemented in the onnxruntime backend's predict_ort.c (the tensorflow
# backend never hit the GPU OOM this mitigates), so only register this test for that backend.
# Slow test: cross-format equivalence across both binaries, both input formats and both
# output formats. Runs real inference like e2e_pipeline, over the same tiny fixture.
# Gene-boundary behaviour, especially indels running past a gene's edge.
add_test(NAME boundaries
    COMMAND ${BATS_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/boundaries.bats)
set_tests_properties(boundaries PROPERTIES
    ENVIRONMENT "${CPLICEAI_TEST_ENVIRONMENT}"
    LABELS "e2e"
    TIMEOUT 900)

add_test(NAME input_formats
    COMMAND ${BATS_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/input_formats.bats)
set_tests_properties(input_formats PROPERTIES
    ENVIRONMENT "${CPLICEAI_TEST_ENVIRONMENT}"
    LABELS "e2e"
    TIMEOUT 900)

if(CPLICEAI_INFERENCE_BACKEND STREQUAL "onnxruntime")
    add_test(NAME chunking_parity
        COMMAND ${BATS_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/chunking_parity.bats)
    set_tests_properties(chunking_parity PROPERTIES
        ENVIRONMENT "${CPLICEAI_TEST_ENVIRONMENT}"
        LABELS "e2e"
        TIMEOUT 300)
endif()

# Compares the onnxruntime and tensorflow backends against each other at full precision.
# A single build tree only has one backend's binaries, so this can't be auto-wired the way
# the tests above are -- it self-skips unless the CPLICEAI_ORT_*/CPLICEAI_TF_* env vars
# (see tests/backend_parity.bats's header comment) are exported in the invoking shell
# before running `ctest -R backend_parity`. Registered here so it's discoverable via
# `ctest -N`/`ctest -L parity`, and included in `check` (skips cleanly by default).
add_test(NAME backend_parity
    COMMAND ${BATS_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/backend_parity.bats)
set_tests_properties(backend_parity PROPERTIES
    ENVIRONMENT "FIXTURES_DIR=${CMAKE_SOURCE_DIR}/tests/fixtures"
    LABELS "parity"
    TIMEOUT 120)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS cpliceai_reference cpliceai_predict_variant cpliceai_predict_gene)

# Shared library
add_library(libcpliceai SHARED
    ${CPLICEAI_INFERENCE_SOURCES}
    src/utils.c
    src/range.c
    src/gene_regions.c
    src/reference.c
    src/gene_reference.c
    src/variant_input.c
    src/variant_output.c
    src/logging/log.c)
target_include_directories(libcpliceai PUBLIC src/)
target_include_directories(libcpliceai PRIVATE ${CPLICEAI_INFERENCE_INCLUDE_DIRS})
set_target_properties(libcpliceai PROPERTIES VERSION 1 SOVERSION 1 OUTPUT_NAME "cpliceai")
target_link_libraries(libcpliceai PRIVATE ${CPLICEAI_INFERENCE_LIBS} ${HTSLIB_LIBRARIES})

# Install library
install(TARGETS libcpliceai
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include)

# Install headers
install(FILES
    src/utils.h
    src/predict.h
    src/range.h
    src/gene_regions.h
    src/reference.h
    src/gene_reference.h
    src/variant_input.h
    src/variant_output.h
    DESTINATION include/cpliceai)

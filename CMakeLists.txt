cmake_minimum_required(VERSION 3.10)

project(CpliceAI C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/sanitizers-cmake/cmake" ${CMAKE_MODULE_PATH})

include_directories(externals)

find_package(Sanitizers)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TENSORFLOW REQUIRED tensorflow)
pkg_check_modules(HTSLIB REQUIRED htslib)

# Shared static libraries used across the executables
add_library(logging STATIC src/logging/log.c)
add_library(predict STATIC src/predict.c)
add_library(utils STATIC src/utils.c)
add_library(range STATIC src/range.c)
add_library(gene_regions STATIC src/gene_regions.c)
add_library(reference STATIC src/reference.c)
add_library(gene_reference STATIC src/gene_reference.c)

set(CPLICEAI_COMMON_LIBS logging predict utils range gene_regions reference)

# Executables
add_executable(cpliceai_reference src/bin/build_reference_scores.c)
target_link_libraries(cpliceai_reference PRIVATE
    ${CPLICEAI_COMMON_LIBS}
    ${TENSORFLOW_LIBRARIES}
    ${HTSLIB_LIBRARIES})

add_executable(cpliceai_predict_variant src/bin/predict_variant.c)
target_link_libraries(cpliceai_predict_variant PRIVATE
    ${CPLICEAI_COMMON_LIBS}
    gene_reference
    ${TENSORFLOW_LIBRARIES}
    ${HTSLIB_LIBRARIES})

add_executable(cpliceai_predict_gene src/bin/predict_gene.c)
target_link_libraries(cpliceai_predict_gene PRIVATE
    ${CPLICEAI_COMMON_LIBS}
    gene_reference
    ${TENSORFLOW_LIBRARIES}
    ${HTSLIB_LIBRARIES})

foreach(cpliceai_target cpliceai_reference cpliceai_predict_variant cpliceai_predict_gene)
    add_sanitizers(${cpliceai_target})
endforeach()

# Shared library
add_library(libcpliceai SHARED
    src/predict.c
    src/utils.c
    src/range.c
    src/gene_regions.c
    src/reference.c
    src/gene_reference.c
    src/logging/log.c)
target_include_directories(libcpliceai PUBLIC src/)
set_target_properties(libcpliceai PROPERTIES VERSION 1 SOVERSION 1 OUTPUT_NAME "cpliceai")
target_link_libraries(libcpliceai PRIVATE ${TENSORFLOW_LIBRARIES} ${HTSLIB_LIBRARIES})

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
    DESTINATION include/cpliceai)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
# set(CMAKE_C_FLAGS_DEBUG_INIT "-Wall -O2 -fsanitize=address,leak,undefined -g")
# set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -O2 -fsanitize=address,leak,undefined -g")
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

project(CpliceAI C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/sanitizers-cmake/cmake" ${CMAKE_MODULE_PATH}) 

include_directories(include)

find_package(Sanitizers)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TENSORFLOW REQUIRED tensorflow)
pkg_check_modules(HTSLIB REQUIRED htslib)

# Create executable
add_executable(cpliceai_reference src/bin/build_reference_scores.c)
add_executable(cpliceai_predict_variant src/bin/predict_variant.c)
add_executable(cpliceai_predict_gene src/bin/predict_gene.c)
# add_sanitizers(cpliceai_reference)

# Add logging module
add_library(logging STATIC src/logging/log.c)

target_include_directories(cpliceai_reference PRIVATE src/logging)
target_link_libraries(cpliceai_reference logging)

target_include_directories(cpliceai_predict_variant PRIVATE src/logging)
target_link_libraries(cpliceai_predict_variant logging)

target_include_directories(cpliceai_predict_gene PRIVATE src/logging)
target_link_libraries(cpliceai_predict_gene logging)


# Add predict module
add_library(predict STATIC src/predict.c)
target_link_libraries(cpliceai_reference predict)
target_link_libraries(cpliceai_predict_variant predict)
target_link_libraries(cpliceai_predict_gene predict)

# Add utils module
add_library(utils STATIC src/utils.c)
target_link_libraries(cpliceai_reference utils)
target_link_libraries(cpliceai_predict_variant utils)
target_link_libraries(cpliceai_predict_gene utils)

# Add range module
add_library(range STATIC src/range.c)
target_link_libraries(cpliceai_reference range)
target_link_libraries(cpliceai_predict_variant range)
target_link_libraries(cpliceai_predict_gene range)

# Add gene region module
add_library(gene_regions STATIC src/gene_regions.c)
target_link_libraries(cpliceai_reference gene_regions)
target_link_libraries(cpliceai_predict_variant gene_regions)
target_link_libraries(cpliceai_predict_gene gene_regions)

# Add reference module
add_library(reference STATIC src/reference.c)
target_link_libraries(cpliceai_reference reference)
target_link_libraries(cpliceai_predict_variant reference)
target_link_libraries(cpliceai_predict_gene reference)

# Add gene reference module
add_library(gene_reference STATIC src/gene_reference.c)
target_link_libraries(cpliceai_predict_variant gene_reference)
target_link_libraries(cpliceai_predict_gene gene_reference)

# Link external libraries
target_link_libraries(cpliceai_reference ${TENSORFLOW_LIBRARIES})
target_link_libraries(cpliceai_reference ${HTSLIB_LIBRARIES})

target_link_libraries(cpliceai_predict_variant ${TENSORFLOW_LIBRARIES})
target_link_libraries(cpliceai_predict_variant ${HTSLIB_LIBRARIES})

target_link_libraries(cpliceai_predict_gene ${TENSORFLOW_LIBRARIES})
target_link_libraries(cpliceai_predict_gene ${HTSLIB_LIBRARIES})


# Build SO
add_library(libcpliceai SHARED src/predict.c src/utils.c src/range.c src/gene_regions.c src/reference.c src/gene_reference.c src/logging/log.c)
target_include_directories(libcpliceai PUBLIC src/)
set_target_properties(libcpliceai PROPERTIES VERSION 1 SOVERSION 1 OUTPUT_NAME "cpliceai")
target_link_libraries(libcpliceai PRIVATE ${TENSORFLOW_LIBRARIES})
target_link_libraries(libcpliceai PRIVATE ${HTSLIB_LIBRARIES})

# Install library
install(TARGETS libcpliceai
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include)

# Include headers
install(FILES
    src/utils.h
    src/predict.h
    src/range.h
    src/gene_regions.h
    src/reference.h
    src/gene_reference.h
    DESTINATION include/cpliceai)


# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(cpliceai_reference PRIVATE -Wall -fsanitize=address,leak,undefined -g)
#     target_link_options(cpliceai_reference PRIVATE -fsanitize=address,leak,undefined)
# endif()

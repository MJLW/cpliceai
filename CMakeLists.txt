cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
# set(CMAKE_C_FLAGS_DEBUG_INIT "-Wall -fsanitize=address,leak,undefined -g")
# set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -fsanitize=address,leak,undefined -g")
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
add_executable(cpliceai src/cpliceai.c)
add_executable(cpliceai_reference src/build_reference_scores.c)
add_sanitizers(cpliceai_reference)

# Add logging module
add_library(logging STATIC src/logging/log.c)

target_include_directories(cpliceai PRIVATE src/logging)
target_link_libraries(cpliceai logging)

target_include_directories(cpliceai_reference PRIVATE src/logging)
target_link_libraries(cpliceai_reference logging)

# Add predict module
add_library(predict STATIC src/predict.c)
target_link_libraries(cpliceai predict)
target_link_libraries(cpliceai_reference predict)

# Add utils module
add_library(utils STATIC src/utils.c)
target_link_libraries(cpliceai utils)
target_link_libraries(cpliceai_reference utils)

# Add range module
add_library(range STATIC src/range.c)
target_link_libraries(cpliceai range)
target_link_libraries(cpliceai_reference range)

# Add gene region module
add_library(gene_regions STATIC src/gene_regions.c)
target_link_libraries(cpliceai_reference gene_regions)

# Add reference module
add_library(reference STATIC src/reference.c)
target_link_libraries(cpliceai_reference reference)
target_link_libraries(cpliceai reference)

# Link external libraries
target_link_libraries(cpliceai ${TENSORFLOW_LIBRARIES})
target_link_libraries(cpliceai ${HTSLIB_LIBRARIES})

target_link_libraries(cpliceai_reference ${TENSORFLOW_LIBRARIES})
target_link_libraries(cpliceai_reference ${HTSLIB_LIBRARIES})


# Build SO
add_library(libcpliceai SHARED src/predict.c src/utils.c src/range.c)
target_include_directories(libcpliceai PUBLIC src/)
set_target_properties(libcpliceai PROPERTIES VERSION 1 SOVERSION 1 OUTPUT_NAME "cpliceai")
target_link_libraries(libcpliceai PRIVATE ${TENSORFLOW_LIBRARIES})
target_link_libraries(libcpliceai PRIVATE ${HTSLIB_LIBRARIES})

target_compile_options(cpliceai_reference PRIVATE -mf16c)

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
    DESTINATION include/cpliceai)


# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(cpliceai_reference PRIVATE -Wall -fsanitize=address,leak,undefined -g)
#     target_link_options(cpliceai_reference PRIVATE -fsanitize=address,leak,undefined)
# endif()

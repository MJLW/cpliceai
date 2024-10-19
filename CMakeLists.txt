cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

project(CpliceAI C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

# Add tensorflow
#include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include)
#link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TENSORFLOW REQUIRED tensorflow)
pkg_check_modules(HTSLIB REQUIRED htslib)

# Create executable
add_executable(cpliceai src/cpliceai.c)

# Add logging module
add_library(logging STATIC src/logging/log.c)
target_include_directories(cpliceai PRIVATE src/logging)
target_link_libraries(cpliceai logging)

# Add predict module
add_library(predict STATIC src/predict.c)
target_link_libraries(cpliceai predict)

# Add utils module
add_library(utils STATIC src/utils.c)
target_link_libraries(cpliceai utils)

# Add range module
add_library(range STATIC src/range.c)
target_link_libraries(cpliceai range)

# Link external libraries
target_link_libraries(cpliceai ${TENSORFLOW_LIBRARIES})
target_link_libraries(cpliceai ${HTSLIB_LIBRARIES})

# Build SO
add_library(libcpliceai SHARED src/predict.c src/utils.c src/range.c)
target_include_directories(libcpliceai PUBLIC src/)
set_target_properties(libcpliceai PROPERTIES VERSION 1 SOVERSION 1 OUTPUT_NAME "cpliceai")
target_link_libraries(libcpliceai PRIVATE ${TENSORFLOW_LIBRARIES})
target_link_libraries(libcpliceai PRIVATE ${HTSLIB_LIBRARIES})


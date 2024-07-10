cmake_minimum_required(VERSION 3.10)

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

# Add logging
add_library(logging STATIC src/logging/log.c)
target_include_directories(cpliceai PRIVATE src/logging)
target_link_libraries(cpliceai logging)

# Link external libraries
target_link_libraries(cpliceai ${TENSORFLOW_LIBRARIES})
target_link_libraries(cpliceai ${HTSLIB_LIBRARIES})

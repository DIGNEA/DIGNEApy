cmake_minimum_required(VERSION 3.16.3)
project(par_ea_kp LANGUAGES CXX VERSION 1.0.0 DESCRIPTION "Parallel EA for Knapsack Problems")

set(CMAKE_CXX_STANDARD 20)
enable_language(C)
add_definitions(-DOS=LINUX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Path configuration
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

include(GNUInstallDirs)
include_directories(./)
link_directories(./)

# Libraries for compiling dignea
find_package(pybind11 REQUIRED)
find_package(OpenMP COMPONENTS C CXX REQUIRED)

# Set compiler flags for OpenMP
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")

    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}=-Xpreprocessor -fopenmp -lomp -I/usr/local/opt/libomp/include")
        set(CMAKE_CXX_FLAGS "{CMAKE_CXX_FLAGS} -OpenMP_CXX_LIB_NAMES=omp -OpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
endif()

# Always run with the optimisation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -isystem -fPIC -fpermissive -O3 -ffast-math -Wall -march=native -mtune=native")

pybind11_add_module(parallel_ea src/parallel_ea.cpp)
target_link_libraries(parallel_ea PUBLIC OpenMP::OpenMP_CXX)

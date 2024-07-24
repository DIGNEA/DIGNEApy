cmake_minimum_required(VERSION 3.16.3)
project(pisinger_cpp LANGUAGES CXX VERSION 1.0.0 DESCRIPTION "Pisinger Solvers Wrapper for Python code")

set(CMAKE_CXX_STANDARD 11)
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

# Libraries for compiling dignea
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Always run with the optimisation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem -fPIC -fpermissive -O3 -ffast-math -Wall -march=native -mtune=native -Wno-deprecated-register -Wno-writable-strings -Wno-unused-but-set-variable -Wno-format -Wno-uninitialized -Wno-unused-variable")

pybind11_add_module(pisinger_cpp src/pisinger.cpp src/minknap.cpp src/combo.cpp src/expknap.cpp)
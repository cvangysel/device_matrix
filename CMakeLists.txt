cmake_minimum_required (VERSION 3.5)
project (device_matrix)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/third_party)

find_package(CUDA REQUIRED)
find_package(Glog REQUIRED)

MESSAGE(STATUS "CUDA: " ${CUDA_LIBRARIES} " " ${CUDA_INCLUDE_DIRS})
MESSAGE(STATUS "GLog: " ${GLOG_LIBRARIES})

# Adapted from http://crascit.com/2015/07/25/cmake-gtest/.
configure_file(third_party/googletest-CMakeLists.txt googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

set(LIBRARIES
    -lglog
    -lcublas -lcudart -lnvToolsExt -lcnmem
    ${CUDA_LIBRARIES} ${GLOG_LIBRARIES})

set(TEST_LIBRARIES
    gtest_main gmock ${LIBRARIES})

set(CMAKE_CXX_FLAGS
    "-std=c++11 -march=native -O3 -funroll-loops")

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=sm_52 --relocatable-device-code=true --cudart=shared -use_fast_math -O3 -default-stream per-thread)
set(CUDA_SEPARABLE_COMPILATION ON)

# Circuimvent CMake here (include_directories) as it seems unable to pass SYSTEM include paths to NVCC.
SET(INCLUDE_DIRS
    "${gtest_SOURCE_DIR}/include"
    "${gmock_SOURCE_DIR}/include"
    ${GLOG_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
    ${GTEST_INCLUDE_DIRS}
)

foreach(INCLUDE_DIR ${INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -isystem ${INCLUDE_DIR})

    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -isystem ${INCLUDE_DIR}")
endforeach()

enable_testing()
subdirs(cpp examples)

file(GLOB_RECURSE DEVICE_MATRIX_HEADER_FILES
    "include/*.h"
)

# Installation rules for header files.
install(FILES ${DEVICE_MATRIX_HEADER_FILES} DESTINATION include/device_matrix)

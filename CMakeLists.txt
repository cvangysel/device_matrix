cmake_minimum_required (VERSION 3.5)
project (device_matrix)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/third_party)

find_package(CUDA 7.0 REQUIRED)
find_package(ExtraCUDA REQUIRED) # nvToolsExt
find_package(Glog 0.3.4 REQUIRED)

function(add_external_src TARGET_NAME MODULE_NAME)
    if(NOT TARGET ${TARGET_NAME})
        configure_file(third_party/${MODULE_NAME}-CMakeLists.txt
                       ${MODULE_NAME}-download/CMakeLists.txt)
        execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-download )
        execute_process(COMMAND ${CMAKE_COMMAND} --build .
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-download )

        # Prevent GoogleTest from overriding our compiler/linker options
        # when building with Visual Studio
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-src
                         ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-build
                         EXCLUDE_FROM_ALL)
    endif(NOT TARGET ${TARGET_NAME})
endfunction()

add_external_src(gtest googletest)
add_external_src(cnmem cnmem)

set(LIBRARIES
    cnmem
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${EXTRACUDA_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${GLOG_LIBRARIES})

set(TEST_LIBRARIES gtest_main gmock ${LIBRARIES})

set(CMAKE_CXX_FLAGS
    "-std=c++11 -march=native -O3 -funroll-loops")

# Circuimvent CMake here (include_directories) as it seems unable to pass SYSTEM include paths to NVCC.
SET(INCLUDE_DIRS
    "${gtest_SOURCE_DIR}/include"
    "${gmock_SOURCE_DIR}/include"
    "${cnmem_SOURCE_DIR}/include"
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
    ${GLOG_INCLUDE_DIRS}
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

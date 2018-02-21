SET(DEVICE_MATRIX_SRC_FILES device_matrix.cu runtime.cu)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories(BEFORE ${device_matrix_SOURCE_DIR}/include)

# Main library.
cuda_add_library(device_matrix
                 ${DEVICE_MATRIX_SRC_FILES}
                 OPTIONS -DFLOATING_POINT_TYPE=float32 -DNDEBUG=1)
target_compile_options(device_matrix PUBLIC -DFLOATING_POINT_TYPE=float32 -DNDEBUG=1)

# Tests.
cuda_add_library(device_matrix_debug
                 ${DEVICE_MATRIX_SRC_FILES}
                 OPTIONS -DFLOATING_POINT_TYPE=float64)
target_compile_options(device_matrix_debug PUBLIC -DFLOATING_POINT_TYPE=float64)

cuda_add_executable(device_matrix_tests device_matrix_tests.cu OPTIONS -DFLOATING_POINT_TYPE=float64)
target_link_libraries(device_matrix_tests device_matrix_debug ${TEST_LIBRARIES})

# Make test visible to CMake.
add_test(NAME device_matrix_tests COMMAND device_matrix_tests)

# Installation rules for library binary.
install(TARGETS device_matrix DESTINATION lib)
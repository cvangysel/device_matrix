include_directories(BEFORE ${device_matrix_SOURCE_DIR}/include)

cuda_add_executable(matrix_mult matrix_mult.cu OPTIONS -DFLOATING_POINT_TYPE=float32)
target_link_libraries(matrix_mult device_matrix ${LIBRARIES})

cuda_add_executable(kernel kernel.cu OPTIONS -DFLOATING_POINT_TYPE=float32)
target_link_libraries(kernel device_matrix ${LIBRARIES})

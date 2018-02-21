#ifndef DEVICE_MATRIX_BASE_H
#define DEVICE_MATRIX_BASE_H

#include <iomanip>
#include <iterator>
#include <ostream>
#include <vector>

namespace cuda {

typedef unsigned short uint16;

typedef long int32;
typedef unsigned long uint32;
typedef long long int64;
typedef unsigned long long uint64;

typedef float float32;
typedef double float64;

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << std::setprecision(20) << "[";
    copy(v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
    os << "]";

    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& v) {
    os << "[";
    for (const auto& e : v) {
        os << e << ", ";
    }
    os << "]";

    return os;
}

}  // namespace cuda

// Makes class instantiations non-copyable.
//
// From StackOverflow.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)

#define CCE(x) checkCudaErrors(x)

#define LAUNCH_KERNEL(...)\
    CCE(cudaGetLastError()); \
    VLOG(2) << "Launching " << #__VA_ARGS__ << "."; \
    __VA_ARGS__; \
    CCE(cudaGetLastError())

#define CHECK_DIMENSIONS_EQUAL(FIRST, SECOND) \
    CHECK_EQ((FIRST).getRows(), (SECOND).getRows());\
    CHECK_EQ((FIRST).getCols(), (SECOND).getCols())

#define CHECK_DIMENSIONS_MULT_COMPATIBLE(FIRST, SECOND) \
    CHECK_EQ((FIRST).getCols(), (SECOND).getRows())

#define CHECK_DIMENSIONS(MATRIX, NUM_ROWS, NUM_COLUMNS) \
    CHECK((MATRIX).getRows() == NUM_ROWS && \
          (MATRIX).getCols() == NUM_COLUMNS) \
        << "Tensor " << #MATRIX << " has shape " \
        << (MATRIX).getRows() << "-by-" << (MATRIX).getCols() << " " \
        << "instead of expected shape " << NUM_ROWS << "-by-" << NUM_COLUMNS << ".";

#define MAKE_MATRIX_NULL(MATRIX) (MATRIX).fillwith((MATRIX).getStream(), 0.0);

#if __DEVICE_MATRIX_DEBUG>0
#pragma message("Debug mode is enabled; everything will be slow!")

#define CHECK_MATRIX_FINITE(MATRIX) \
    cudaDeviceSynchronize();\
    CHECK(isfinite((MATRIX)))\
        << "Tensor " << #MATRIX << " is not finite."; \
    CHECK(isfinite_slow(MATRIX))\
        << "Tensor " << #MATRIX << " is not finite."

#define PRINT_MATRIX(MATRIX) cudaDeviceSynchronize(); VLOG(3) << #MATRIX; print_matrix((MATRIX))
#define PRINT_VECTOR(VECTOR) cudaDeviceSynchronize(); VLOG(3) << #VECTOR; print_vector((VECTOR))

#define CHECK_MATRIX_NORM(MATRIX)\
    LOG_IF(INFO, l2_norm((MATRIX)) < std::numeric_limits<float32>::epsilon()) \
        << "Matrix " << #MATRIX << " has zero l2 norm."

#else

#define CHECK_MATRIX_FINITE(matrix)
#define PRINT_MATRIX(matrix)
#define PRINT_VECTOR(vector)
#define CHECK_MATRIX_NORM(matrix)

#endif

#define CHECK_MATRIX(matrix) CHECK_MATRIX_FINITE(matrix); PRINT_MATRIX(matrix)

#endif /* DEVICE_MATRIX_BASE_H */
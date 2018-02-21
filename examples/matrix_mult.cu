#include <device_matrix/device_matrix.h>

#include <glog/logging.h>
#include <memory>

using namespace cuda;

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    const cudaStream_t stream = 0; // default CUDA stream.

    std::unique_ptr<device_matrix<float32>> a(
        device_matrix<float32>::create(
            stream,
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            2 /* num_rows */, 3 /* num_columns */));

    std::unique_ptr<device_matrix<float32>> b(
        device_matrix<float32>::create(
            stream,
            {7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
            3 /* num_rows */, 2 /* num_columns */));

    device_matrix<float32> c(
        2 /* num_rows */, 2 /* num_columns */, stream);

    matrix_mult(stream,
                *a, CUBLAS_OP_N,
                *b, CUBLAS_OP_N,
                &c);

    cudaDeviceSynchronize();

    print_matrix(c);
}

#include <device_matrix/device_matrix.h>

#include <glog/logging.h>
#include <memory>

using namespace cuda;

template <typename FloatT>
__global__
void inverse_kernel(FloatT* const input) {
    size_t offset = threadIdx.y * blockDim.x + threadIdx.x;
    input[offset] = -input[offset];
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    const cudaStream_t stream = 0; // default CUDA stream.

    std::unique_ptr<device_matrix<float32>> a(
        device_matrix<float32>::create(
            stream,
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            2 /* num_rows */, 3 /* num_columns */));

    LAUNCH_KERNEL(
        inverse_kernel
            <<<1, /* a single block */
               dim3(a->getRows(), a->getCols()), /* one thread per component */
               0,
               stream>>>(
            a->getData()));

    cudaDeviceSynchronize();

    print_matrix(*a);
}

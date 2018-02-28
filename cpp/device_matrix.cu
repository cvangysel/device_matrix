#include "device_matrix/device_matrix.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// From http://stackoverflow.com/questions/16077464/atomicadd-for-double-on-gpu.
//
// This is a hack that allows the tests to run in double precision.
// atomicAdd for doubles is available in CUDA 8 and onwards.
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace cuda {

cudaStream_t merge_streams(const cudaStream_t first,
                           const cudaStream_t second) {
    if (first == second) {
      return first;
    }

    cudaEvent_t first_stream_wait_on_second;
    CCE(cudaEventCreate(&first_stream_wait_on_second));
    CCE(cudaStreamWaitEvent(first, first_stream_wait_on_second, 0));
    CCE(cudaEventRecord(first_stream_wait_on_second, second));
    CCE(cudaEventDestroy(first_stream_wait_on_second));

    return first;
}

}  // namespace cuda

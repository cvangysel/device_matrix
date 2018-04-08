#include "device_matrix/device_matrix.h"

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

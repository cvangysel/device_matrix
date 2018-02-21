#include "device_matrix/runtime.h"

#include <glog/logging.h>

namespace cuda {

template <>
Runtime<FLOATING_POINT_TYPE>* Runtime<FLOATING_POINT_TYPE>::INSTANCE_ = new Runtime;

template <typename FloatT>
Runtime<FloatT>::Runtime() : ZERO(nullptr), ONE(nullptr) {
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    LOG_IF(FATAL, (device_count == 0)) << "Unable to find any CUDA-enabled device.";

    const int32 device_id = 0;

    CHECK_LT(device_id, device_count)
          << "Invalid CUDA device identifier "
          << "(" << device_count << " devices available).";

    CCE(cudaSetDevice(device_id));

    // Fresh start.
    CCE(cudaDeviceReset());

    // Hard-coded to run on device #0.
    CCE(cudaGetDeviceProperties(&props_, device_id));
    CCE(cublasCreate(&handle_));
    CCE(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE));

    LOG(INFO) << "Using device #" << device_id << ".";

    memset(&device_, 0, sizeof(device_));
    device_.device = device_id;
    device_.size = (size_t) (0.85 * props_.totalGlobalMem);
    CHECK_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device_, CNMEM_FLAGS_DEFAULT));

    CCE(cudaMalloc(const_cast<FloatT**>(&ZERO), sizeof(FloatT)));
    const FloatT zero = 0.0;
    CCE(cudaMemcpy(const_cast<FloatT*>(ZERO), &zero,
                   sizeof(FloatT),
                   cudaMemcpyHostToDevice));

    CCE(cudaMalloc(const_cast<FloatT**>(&ONE), sizeof(FloatT)));
    const FloatT one = 1.0;
    CCE(cudaMemcpy(const_cast<FloatT*>(ONE),
                   &one, sizeof(FloatT),
                   cudaMemcpyHostToDevice));
}

const decltype(&cublasSgemm) CuBLAS<float32>::gemm = &cublasSgemm;
const decltype(&cublasSgemv) CuBLAS<float32>::gemv = &cublasSgemv;
const decltype(&cublasSger) CuBLAS<float32>::ger = &cublasSger;

const decltype(&cublasDgemm) CuBLAS<float64>::gemm = &cublasDgemm;
const decltype(&cublasDgemv) CuBLAS<float64>::gemv = &cublasDgemv;
const decltype(&cublasDger) CuBLAS<float64>::ger = &cublasDger;

// Explicit instantiation.
template class Runtime<FLOATING_POINT_TYPE>;

}  // namespace cuda

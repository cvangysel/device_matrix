#ifndef DEVICE_MATRIX_RUNTIME_H
#define DEVICE_MATRIX_RUNTIME_H

#include "base.h"

#include <cnmem.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

namespace cuda {

template <typename FloatT>
class CuBLAS {};
template <>
class CuBLAS<float32> {
 public:
  static const decltype(&cublasSgemm) gemm;
  static const decltype(&cublasSgemv) gemv;
  static const decltype(&cublasSger) ger;
};
template <>
class CuBLAS<float64> {
 public:
  static const decltype(&cublasDgemm) gemm;
  static const decltype(&cublasDgemv) gemv;
  static const decltype(&cublasDger) ger;
};

#define MAX_THREADS_PER_BLOCK Runtime<FloatT>::getInstance()->props().maxThreadsPerBlock
#define MAX_SECONDARY_BLOCK_DIM Runtime<FloatT>::getInstance()->props().maxGridSize[1]

template <typename FloatT>
class Runtime {
 public:
  static Runtime* getInstance() {
      return INSTANCE_;
  }

  inline cublasHandle_t& handle() {
      return handle_;
  }

  inline cudaDeviceProp& props() {
      return props_;
  }

  const FloatT* const ZERO;
  const FloatT* const ONE;

 private:
  Runtime();

  ~Runtime() {
      CCE(cublasDestroy(handle_));

      // Kill it.
      cudaDeviceReset();
  }

  static Runtime* INSTANCE_;

  cudaDeviceProp props_;
  cublasHandle_t handle_;
  cnmemDevice_t device_;

  DISALLOW_COPY_AND_ASSIGN(Runtime);
};

}  // namespace cuda

#endif /* DEVICE_MATRIX_RUNTIME_H */
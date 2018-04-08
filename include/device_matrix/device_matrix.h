#ifndef DEVICE_MATRIX_H
#define DEVICE_MATRIX_H

#include "base.h"

// CUDA Math API
#include <math.h>

// For CUDA asserts.
#include <assert.h>

#include <glog/logging.h>

#include <thrust/version.h>
#ifndef THRUST_VERSION
#pragma error("Unable to determine Thrust version.")
#else
#if THRUST_VERSION <= 100802
#pragma error("Requires at least Thrust 1.9 and upwards.")
#endif
#endif

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cnmem.h>

#include "func.h"
#include "profile.h"
#include "streams.h"
#include "runtime.h"

namespace cuda {

// Forward declarations.
template <typename FloatT>
class device_matrix;

template <typename FloatT>
void print_matrix(const device_matrix<FloatT>& dmat,
                  const size_t start_row = 0,
                  size_t end_row = std::numeric_limits<size_t>::max(),
                  const bool strict = true);

template <typename FloatT>
void print_vector(const thrust::device_vector<FloatT>& vec);

template <typename FloatT>
FloatT get_scalar(const /* __device__ */ FloatT* const scalar_ptr);

// Forward declaration.
cudaStream_t merge_streams(const cudaStream_t first,
                           const cudaStream_t second);

// Forward declaration.
template <typename FloatT, typename IteratorT>
thrust::transform_iterator<func::scale_by_constant<FloatT>, IteratorT>
make_scalar_multiplication_iterator(IteratorT it, const FloatT scalar);

template <typename FloatT>
class device_matrix {
 public:
  static device_matrix<FloatT>* create(const cudaStream_t stream,
                                       const std::vector<FloatT>& values,
                                       const size_t num_rows, const size_t num_cols) {
      PROFILE_FUNCTION();

      device_matrix<FloatT>* matrix = new device_matrix<FloatT>(num_rows, num_cols, stream);
      matrix->fillwith(stream, values);

      return matrix;
  }

  static device_matrix<FloatT>* create_column(const cudaStream_t stream,
                                              const std::vector<FloatT>& values) {
      PROFILE_FUNCTION();

      return create(stream,
                    values,
                    values.size(), /* num_rows */
                    1 /* num_cols */);
  }

  static device_matrix<FloatT>* create(const cudaStream_t stream,
                                       /* __host__ */ FloatT* const begin,
                                       /* __host__ */ FloatT* const end,
                                       const size_t num_rows, const size_t num_cols) {
      PROFILE_FUNCTION();

      CHECK_EQ((end - begin) / num_cols, num_rows);
      CHECK_EQ((end - begin) % num_cols, 0);

      device_matrix<FloatT>* matrix = new device_matrix<FloatT>(
          num_rows, num_cols,
          stream);

      matrix->fillwith(stream, begin, end);

      return matrix;
  }

  static device_matrix<FloatT>* create_shape_as(const cudaStream_t stream,
                                                const device_matrix<FloatT>& matrix) {
      return new device_matrix<FloatT>(matrix.getRows(), matrix.getCols(),
                                       stream);
  }

  device_matrix(const size_t num_rows, const size_t num_cols,
                const cudaStream_t stream)
      : rows_(num_rows), cols_(num_cols), stream_(stream) {
      DCHECK_GT(rows_, 0);
      DCHECK_GT(cols_, 0);

      if (size() > 0) {
          CHECK_EQ(CNMEM_STATUS_SUCCESS,
                   cnmemMalloc((void**) &data_,
                               rows_ * cols_ * sizeof(FloatT),
                               stream_));
      } else {
          data_ = nullptr;
      }
  }

  virtual ~device_matrix() {
      if (data_ != nullptr) {
          CHECK_EQ(CNMEM_STATUS_SUCCESS,
                   cnmemFree((void*) data_,
                             stream_ /* stream */));
      }
  }

  thrust::device_ptr<FloatT> begin(const size_t column_idx=0) const {
      return thrust::device_pointer_cast(data_) + column_idx * getRows();
  }

  thrust::device_ptr<FloatT> end() const {
      return thrust::device_pointer_cast(data_ + size());
  }

  device_matrix<FloatT>* copy(const cudaStream_t stream) const {
      PROFILE_FUNCTION();

      device_matrix<FloatT>* const new_matrix =
          new device_matrix<FloatT>(getRows(), getCols(), stream);

      thrust::copy(
          thrust::cuda::par.on(stream),
          begin(), end(), /* source */
          new_matrix->begin() /* dst */);

      return new_matrix;
  }

  void fillwith(const cudaStream_t stream, const FloatT value) {
      PROFILE_FUNCTION();

      thrust::fill(
          thrust::cuda::par.on(stream),
          begin(), end(), value);
  }

  void fillwith(const cudaStream_t stream,
                const std::vector<FloatT>& values) {
      PROFILE_FUNCTION();

      CHECK_EQ(values.size(), size());

      cudaMemcpyAsync(raw_begin(*this), /* dst */
                      values.data(),
                      values.size() * sizeof(FloatT),
                      cudaMemcpyHostToDevice,
                      stream);
  }

  void fillwith(const cudaStream_t stream,
                /* __host__ */ const FloatT* const begin,
                /* __host__ */ const FloatT* const end) {
      PROFILE_FUNCTION();

      CHECK_EQ(end - begin, size());

      cudaMemcpyAsync(raw_begin(*this), /* dst */
                      begin,
                      (end - begin) * sizeof(FloatT),
                      cudaMemcpyHostToDevice,
                      stream);
  }

  void copyFrom(const cudaStream_t stream, const device_matrix<FloatT>& other) {
      PROFILE_FUNCTION();

      CHECK(hasSameShape(other));

      fillwith(merge_streams(stream, other.getStream()),
               raw_begin(other), raw_end(other));
  }

  void transfer(const cudaStream_t stream,
                /* __host__ */ FloatT* const dst,
                const size_t num_elements) const {
      CHECK_EQ(num_elements, size());

      cudaMemcpyAsync(dst,
                      raw_begin(*this),
                      num_elements * sizeof(FloatT),
                      cudaMemcpyDeviceToHost,
                      stream);
  }

  inline size_t size() const { return rows_ * cols_; }
  inline size_t getRows() const { return rows_; }
  inline size_t getCols() const { return cols_; }
  inline FloatT* getData() const { return data_; }

  inline cudaStream_t getStream() const { return stream_; }

  inline bool hasSameShape(const device_matrix<FloatT>& other) const {
      return (getRows() == other.getRows()) &&
          (getCols() == other.getCols());
  }

  template <typename TransformOp>
  void transform(const cudaStream_t stream,
                 const TransformOp op = TransformOp()) {
      thrust::transform(thrust::cuda::par.on(stream),
                        begin(), end(),
                        begin(),
                        op);
  }

  void square(const cudaStream_t stream) {
      PROFILE_FUNCTION();
      transform(stream, func::square<FloatT>());
  }

  void scale(const cudaStream_t stream, const FloatT alpha) {
      PROFILE_FUNCTION();
      transform(stream, func::scale_by_constant<FloatT>(alpha));
  }

 private:
  const size_t rows_;
  const size_t cols_;

  const cudaStream_t stream_;

  FloatT* data_;

 private:
  DISALLOW_COPY_AND_ASSIGN(device_matrix);
};

template <typename FloatT>
class device_matrix_view {
 public:
  device_matrix_view(const device_matrix<FloatT>& matrix,
                     const size_t start_col_idx,
                     const size_t num_cols)
      : matrix_(&matrix),
        start_col_idx_(start_col_idx),
        num_cols_(num_cols) {
      CHECK_GE(start_col_idx_, 0);
      CHECK_LE(start_col_idx_ + num_cols_, matrix_->getCols());
  }

  explicit device_matrix_view(const device_matrix<FloatT>& matrix)
      : matrix_(&matrix), start_col_idx_(0), num_cols_(matrix.getCols()) {
      CHECK_GE(start_col_idx_, 0);
      CHECK_LE(start_col_idx_ + num_cols_, matrix_->getCols());
  }

  inline FloatT* getData() const {
      return matrix_->getData() + matrix_->getRows() * start_col_idx_;
  }

  inline size_t getRows() const {
      return matrix_->getRows();
  }

  inline size_t getCols() const {
      return num_cols_;
  }

  inline size_t getLeadingDimension() const {
      return getRows();
  }

 private:
  const device_matrix<FloatT>* const matrix_;

  const size_t start_col_idx_;
  const size_t num_cols_;
};

// Caller takes ownership.
template <typename FloatT>
FloatT* get_array(const cudaStream_t stream,
                  const device_matrix<FloatT>& matrix);

template <typename FloatT>
std::ostream& operator<<(std::ostream& os,
                         const device_matrix<FloatT>& matrix);

template <typename FloatT>
device_matrix<FloatT>* hstack(const cudaStream_t stream,
                              const std::vector<std::pair<device_matrix<FloatT>*,
                                                          FloatT>>& pairs);

template <typename FloatT>
device_matrix<FloatT>* broadcast_columns(const cudaStream_t stream,
                                         const device_matrix<FloatT>& input,
                                         const size_t num_repeats);

template <typename FloatT, typename AggOp = thrust::plus<FloatT>>
device_matrix<FloatT>* fold_columns(const cudaStream_t stream,
                                    const device_matrix<FloatT>& input,
                                    const size_t cluster_size,
                                    const device_matrix<FloatT>* const weights = nullptr,
                                    const AggOp op = AggOp());

template <typename FloatT>
void matrix_mult(const cudaStream_t stream,
                 const device_matrix<FloatT>& first_op,
                 const cublasOperation_t first_op_trans,
                 const device_matrix<FloatT>& second_op,
                 const cublasOperation_t second_op_trans,
                 device_matrix<FloatT>* const dst,
                 const bool dst_contains_bias = false);

template <typename FloatT>
FloatT* raw_begin(const device_matrix<FloatT>& dmat);

template <typename FloatT>
FloatT* raw_end(const device_matrix<FloatT>& dmat);

template <typename FloatT>
thrust::device_ptr<FloatT> begin(const device_matrix<FloatT>& dmat);

template <typename FloatT>
thrust::device_ptr<FloatT> end(const device_matrix<FloatT>& dmat);

template <typename FloatT>
inline void flatten(const cudaStream_t stream,
                    const std::vector<std::vector<FloatT> >& iterable,
                    device_matrix<FloatT>* const flattened);

//
// Operations combining two device_matrix<FloatT>.
//

template <typename DevicePolicy, typename FloatT, typename BinaryFn, typename FirstOpOp = func::identity<FloatT>>
void elemwise_binary(const DevicePolicy& exec,
                     const device_matrix<FloatT>& first_op,
                     device_matrix<FloatT>* const second_op_and_dst,
                     const BinaryFn binary_fn = BinaryFn(),
                     const FirstOpOp first_op_op = func::identity<FloatT>());

template <typename DevicePolicy, typename FloatT, typename FirstOpOp = func::identity<FloatT>>
void elemwise_plus(const DevicePolicy& exec,
                   const device_matrix<FloatT>& first_op,
                   device_matrix<FloatT>* const second_op_and_dst,
                   const FirstOpOp first_op_op = func::identity<FloatT>());

template <typename DevicePolicy, typename FloatT, typename FirstOpOp = func::identity<FloatT>>
void hadamard_product(const DevicePolicy& exec,
                      const device_matrix<FloatT>& first_op,
                      device_matrix<FloatT>* const second_op_and_dst,
                      const FirstOpOp first_op_op = FirstOpOp());

// TODO(cvangysel): refactor this such that it uses the interface from above.
template <typename FloatT>
device_matrix<FloatT>* hadamard_product(const cudaStream_t stream,
                                        const device_matrix<FloatT>& first_op,
                                        const device_matrix<FloatT>& second_op);

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_elemwise(const DevicePolicy& exec,
                    device_matrix<FloatT>* const dmat,
                    const OpT op = OpT());

template <typename FloatT>
thrust::transform_iterator<func::divide_by_constant<size_t>, thrust::counting_iterator<size_t>>
make_matrix_column_iterator(const device_matrix<FloatT>& dmat);

template <typename FloatT, typename IteratorT>
thrust::transform_iterator<func::scale_by_constant<FloatT>, IteratorT>
make_scalar_multiplication_iterator(IteratorT it, const FloatT scalar);

// For every element in column c_i, apply the operation between that element
// and the element on the i'th position in the vector.
template <typename OpT, typename FirstOpOpT, typename SecondOpOpT, typename DevicePolicy, typename FloatT>
void apply_columnwise(const DevicePolicy& exec,
                      const device_matrix<FloatT>& src_matrix,
                      const device_matrix<FloatT>& vector,
                      device_matrix<FloatT>* const dst_matrix,
                      const FirstOpOpT first_op_op = FirstOpOpT(),
                      const SecondOpOpT second_op_op = SecondOpOpT());

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_columnwise(const DevicePolicy& exec,
                      const device_matrix<FloatT>& vector,
                      device_matrix<FloatT>* const dst_matrix);

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_except_every_Nth_column(const DevicePolicy& exec,
                                   const size_t col_idx,
                                   device_matrix<FloatT>* const matrix);

// Only every k-th row (including 0) will be updated; thus the passed iterator will not
// be applied everywhere.
template <typename OpT, typename DevicePolicy, typename FloatT, typename InputIterator>
void apply_every_Nth_column(const DevicePolicy& exec,
                            InputIterator begin_it,
                            InputIterator end_it,
                            const size_t col_idx,
                            device_matrix<FloatT>* const matrix,
                            OpT op = OpT());

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_every_Nth_column(const DevicePolicy& exec,
                            const size_t col_idx,
                            device_matrix<FloatT>* const matrix,
                            OpT op = OpT());

template <typename FloatT>
bool isfinite(const device_matrix<FloatT>& dmat);

template <typename FloatT>
bool isfinite_slow(const device_matrix<FloatT>& dmat);

enum AxisDesc {
    FIRST_AXIS = 1, // reduces the rows and leaves the columns.
    SECOND_AXIS = 2, // reduces the columns and leaves the rows.
};

// Sums the rows (SECOND_AXIS) or columns (FIRST_AXIS) of a matrix.
//
// Output does not need to be initialized to zero.
template <typename FloatT, typename TransformOp = func::identity<FloatT>>
void reduce_axis(const cudaStream_t stream,
                 const AxisDesc axis,
                 const device_matrix<FloatT>& src,
                 device_matrix<FloatT>* const output,
                 TransformOp op = TransformOp());

// TODO(cvangysel): currently this function is not very optimized.
template <typename FloatT>
device_matrix<FloatT>* repmat(const cudaStream_t stream,
                              const device_matrix<FloatT>& src,
                              const size_t num_repeats);

template <typename FloatT>
void flip_adjacent_columns(const cudaStream_t stream,
                           device_matrix<FloatT>* const matrix);

template <typename FloatT>
FloatT l2_norm(const device_matrix<FloatT>& dmat);

}  // namespace cuda

#include "device_matrix_inl.h"

#endif /* DEVICE_MATRIX_H */
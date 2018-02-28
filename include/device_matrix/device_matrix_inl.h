#ifndef DEVICE_MATRIX_INLINE_H
#define DEVICE_MATRIX_INLINE_H

namespace cuda {

template <typename FloatT>
void print_matrix(const device_matrix<FloatT>& dmat,
                  const size_t start_col,
                  size_t end_col,
                  const bool strict) {
    thrust::device_vector<FloatT> v(begin(dmat), end(dmat));

    std::vector<std::vector<FloatT>> matrix;

    end_col = std::min(dmat.getCols(), end_col);

    for (size_t j = start_col; j < end_col; ++j) {
        std::vector<FloatT> col;

        for (size_t i = 0; i < dmat.getRows(); ++i) {
            col.push_back(v[j * dmat.getRows() + i]);

            LOG_IF(FATAL, strict && !::isfinite(col.back()))
                << "Element at position (" << i << "," << j << ") "
                << "is not finite.";
        }

        matrix.push_back(col);
    }

    VLOG(3) << matrix;
}

template <typename FloatT>
void print_vector(const thrust::device_vector<FloatT>& vec) {
    std::vector<FloatT> row;

    for(size_t i = 0; i < vec.size(); i++) {
        row.push_back(vec[i]);
    }

    LOG(INFO) << "Vector: " << row;
}

// Repeating a whole matrix.

template <typename FloatT>
device_matrix<FloatT>* repmat(const cudaStream_t stream,
                              const device_matrix<FloatT>& src,
                              const size_t num_repeats) {
    PROFILE_FUNCTION();

    device_matrix<FloatT>* const repeated =
        new device_matrix<FloatT>(src.getRows(), src.getCols() * num_repeats,
                                  stream);

    for (size_t i = 0; i < num_repeats; ++i) {
        thrust::copy(
            thrust::cuda::par.on(repeated->getStream()),
            src.begin(), src.end(), /* source */
            repeated->begin() + i * src.size());
    }

    CHECK_MATRIX(*repeated);

    return repeated;
}

// Broadcasting (i.e. repeating each column of a matrix).

template <bool Transpose, typename FloatT>
__global__
void broadcast_kernel(const FloatT* const input, FloatT* const broadcasted) {
    const int32 num_repeats = gridDim.x;
    const int32 repeat_idx = blockIdx.x;

    int32 col_idx;

    int32 vector_size;
    int32 component_idx;

    // Branch will be optimized out at compile time.
    if (Transpose) {
        col_idx = blockIdx.z * blockDim.x + threadIdx.x;

        vector_size = gridDim.y;
        component_idx = blockIdx.y;
    } else {
        col_idx = blockIdx.y;

        vector_size = gridDim.z * blockDim.x;
        component_idx = blockIdx.z * blockDim.x + threadIdx.x;
    }

    broadcasted[((col_idx * num_repeats) + repeat_idx) * vector_size + component_idx] =
        input[col_idx * vector_size + component_idx];
}

template <typename FloatT>
device_matrix<FloatT>* broadcast_columns(const cudaStream_t stream,
                                         const device_matrix<FloatT>& input,
                                         const size_t num_repeats) {
    PROFILE_FUNCTION_WITH_STREAM(stream);

    CHECK_GT(num_repeats, 1);
    CHECK_MATRIX(input);

    std::unique_ptr<device_matrix<FloatT>> broadcasted(
        new device_matrix<FloatT>(input.getRows(),
                                  input.getCols() * num_repeats,
                                  stream));

    if (input.getRows() == 1 ||
            input.getRows() > MAX_THREADS_PER_BLOCK ||
            input.getCols() > MAX_SECONDARY_BLOCK_DIM) {
        uint32 num_threads;

        if (input.getCols() > MAX_THREADS_PER_BLOCK) {
            CHECK_EQ(input.getCols() % MAX_THREADS_PER_BLOCK, 0);

            num_threads = MAX_THREADS_PER_BLOCK;
        } else {
            num_threads = input.getCols();
        }

        LAUNCH_KERNEL(
            broadcast_kernel<true>
                <<<dim3(num_repeats, input.getRows(), input.getCols() / num_threads),
                   dim3(num_threads),
                   0,
                   stream>>>(
                input.getData(), broadcasted->getData()));
    } else {
        CHECK_LE(input.getRows(), MAX_THREADS_PER_BLOCK);
        LOG_IF_EVERY_N(WARNING, (input.getRows() % 32 != 0), 100)
            << "Row dimensionality should be a multiple of 32 to optimize efficiency.";

        LAUNCH_KERNEL(
            broadcast_kernel<false>
                <<<dim3(num_repeats, input.getCols(), 1),
                   dim3(input.getRows()),
                   0,
                   stream>>>(
                input.getData(), broadcasted->getData()));
    }

    CHECK_MATRIX(*broadcasted);

    return broadcasted.release();
}

template <typename FloatT>
void matrix_mult(const cudaStream_t stream,
                 const device_matrix<FloatT>& first_op,
                 const cublasOperation_t first_op_trans,
                 const device_matrix<FloatT>& second_op,
                 const cublasOperation_t second_op_trans,
                 device_matrix<FloatT>* const dst,
                 const bool dst_contains_bias) {
    PROFILE_FUNCTION_WITH_STREAM(stream);

    const size_t m = (first_op_trans == CUBLAS_OP_N) ?
        first_op.getRows() : first_op.getCols();
    CHECK_EQ(m, dst->getRows());

    const size_t n = (second_op_trans == CUBLAS_OP_N) ?
        second_op.getCols() : second_op.getRows();
    CHECK_EQ(n, dst->getCols());

    const size_t k_first_op = (first_op_trans == CUBLAS_OP_N) ?
        first_op.getCols() : first_op.getRows();
    const size_t k_second_op = (second_op_trans == CUBLAS_OP_N) ?
        second_op.getRows() : second_op.getCols();

    CHECK_EQ(k_first_op, k_second_op);
    const size_t k = k_first_op;

    const FloatT* const beta = dst_contains_bias ?
        Runtime<FloatT>::getInstance()->ONE : Runtime<FloatT>::getInstance()->ZERO;

    cudaStream_t prev_stream;
    CCE(cublasGetStream(Runtime<FloatT>::getInstance()->handle(),
                        &prev_stream));

    if (prev_stream != stream) {
        VLOG(2) << "Setting CuBLAS stream from " << prev_stream << " to " << stream << ".";
        CCE(cublasSetStream(Runtime<FloatT>::getInstance()->handle(),
                            stream));
    }

    CCE(CuBLAS<FloatT>::gemm(
        Runtime<FloatT>::getInstance()->handle(),
        first_op_trans,
        second_op_trans,
        m, n, k,
        Runtime<FloatT>::getInstance()->ONE, /* alpha */
        first_op.getData(), first_op.getRows(),
        second_op.getData(), second_op.getRows(),
        beta, /* beta */
        dst->getData(), dst->getRows()));

    // Bug in CuBLAS forces us to synchronize here.
    // cudaDeviceSynchronize();

    if (prev_stream != stream) {
        VLOG(2) << "Setting CuBLAS stream from " << stream << " to " << prev_stream << ".";
        CCE(cublasSetStream(Runtime<FloatT>::getInstance()->handle(),
                            prev_stream));
    }

    CHECK_MATRIX(*dst);
}

template <typename FloatT>
__global__
void flip_adjacent_columns_kernel(FloatT* const flipped) {
    const FloatT tmp = flipped[(2 * blockIdx.x) * blockDim.x + threadIdx.x];

    flipped[(2 * blockIdx.x) * blockDim.x + threadIdx.x] =
        flipped[(2 * blockIdx.x + 1) * blockDim.x + threadIdx.x];
    flipped[(2 * blockIdx.x + 1) * blockDim.x + threadIdx.x] = tmp;
}

template <typename FloatT>
void flip_adjacent_columns(const cudaStream_t stream,
                           device_matrix<FloatT>* const matrix) {
    CHECK_EQ(matrix->getCols() % 2, 0);

    LAUNCH_KERNEL(
        flip_adjacent_columns_kernel<FloatT>
            <<<matrix->getCols() / 2,
               matrix->getRows(),
               0,
               stream>>>(matrix->getData()));

    CHECK_MATRIX(*matrix);
}

template <typename FloatT>
FloatT l2_norm(const device_matrix<FloatT>& dmat) {
    PROFILE_FUNCTION();

    return sqrt(thrust::transform_reduce(
        begin(dmat),
        end(dmat),
        func::square<FloatT>(),
        0.0,
        thrust::plus<FloatT>()));
}

template <typename FloatT>
bool isfinite(const device_matrix<FloatT>& dmat) {
    return thrust::transform_reduce(
        begin(dmat),
        end(dmat),
        func::isfinite<FloatT>(),
        0,
        thrust::plus<bool>());
}

template <typename FloatT>
FloatT get_scalar(const /* __device__ */ FloatT* const scalar_ptr) {
    DLOG_EVERY_N(WARNING, 1000) << "Call to get_scalar can cripple performance.";

    FloatT scalar;
    cudaMemcpy(&scalar,
               scalar_ptr,
               sizeof(FloatT),
               cudaMemcpyDeviceToHost);

    return scalar;
}

template <typename FloatT>
bool isfinite_slow(const device_matrix<FloatT>& dmat) {
    for (size_t i = 0; i < dmat.size(); ++i) {
        const FloatT value = get_scalar(raw_begin(dmat) + i);

        if (!::isfinite(value)) {
            const size_t row_pos = i % dmat.getRows();
            const size_t col_pos = i / dmat.getRows();

            LOG(ERROR) << "Element at position (" << row_pos << "," << col_pos << ") "
                       << "is not finite.";

            return false;
        }
    }

    return true;
}

template <typename FloatT>
FloatT* get_array(const cudaStream_t stream,
                  const device_matrix<FloatT>& matrix) {
    cudaStreamSynchronize(stream);

    FloatT* const array = new FloatT[matrix.size()];
    matrix.transfer(stream, array, matrix.size());

    cudaStreamSynchronize(stream);

    return array;
}

template <typename FloatT>
device_matrix<FloatT>* hstack(const cudaStream_t stream,
                              const std::vector<std::pair<device_matrix<FloatT>*,
                                                          FloatT>>& pairs) {
    const size_t num_rows = std::get<0>(pairs.front())->getRows();
    size_t num_cols = 0;

    for (const auto& pair : pairs) {
        CHECK_EQ(std::get<0>(pair)->getRows(), num_rows) << *std::get<0>(pair);
        num_cols += std::get<0>(pair)->getCols();
    }

    device_matrix<FloatT>* const result =
        new device_matrix<FloatT>(num_rows, num_cols, stream);

    FloatT* current_start = raw_begin(*result);

    for (const auto& pair : pairs) {
        thrust::copy(
            thrust::cuda::par.on(stream),
            make_scalar_multiplication_iterator(
                std::get<0>(pair)->begin(),
                std::get<1>(pair)),
            make_scalar_multiplication_iterator(
                std::get<0>(pair)->end(),
                std::get<1>(pair)), /* source */
            current_start /* dst */);

        current_start += std::get<0>(pair)->getCols() * num_rows;
    }

    CHECK_EQ(current_start, raw_end(*result));

    return result;
}

template <typename FloatT>
inline void flatten(const cudaStream_t stream,
                    const std::vector<std::vector<FloatT> >& iterable,
                    device_matrix<FloatT>* const flattened) {
    /* __device__ */ FloatT* it = raw_begin(*flattened);
    size_t elements = 0;

    // TODO(cvangysel): there is more potential for parallelism here.
    for (const auto& instance : iterable) {
        elements += instance.size();
        CHECK_LE(elements, flattened->size());

        cudaMemcpyAsync(it, /* dst */
                        instance.data(),
                        instance.size() * sizeof(FloatT),
                        cudaMemcpyHostToDevice,
                        stream);

        it += instance.size();
    }
}

template <typename FloatT>
inline std::ostream& operator<<(std::ostream& os,
                                const device_matrix<FloatT>& matrix) {
    os << matrix.getRows() << "-by-" << matrix.getCols() << " matrix "
       << "on GPU at " << matrix.getData();

    return os;
}

template <typename FloatT>
FloatT* raw_begin(const device_matrix<FloatT>& dmat) {
    return thrust::raw_pointer_cast(dmat.getData());
}

template <typename FloatT>
FloatT* raw_end(const device_matrix<FloatT>& dmat) {
    return raw_begin(dmat) + dmat.getRows() * dmat.getCols();
}

template <typename FloatT>
thrust::device_ptr<FloatT> begin(const device_matrix<FloatT>& dmat) {
    return dmat.begin();
}

template <typename FloatT>
thrust::device_ptr<FloatT> end(const device_matrix<FloatT>& dmat) {
    return dmat.end();
}

// Folding.

template <typename FloatT, typename AggOp>
__global__
void fold_columns_kernel(const size_t cluster_size,
                         const FloatT* const input, const FloatT* const weights,
                         FloatT* const folded,
                         const AggOp op) {
    const size_t first_row_idx = blockIdx.x * cluster_size;
    const FloatT first_weight = (weights != nullptr ? weights[first_row_idx] : 1.0);

    FloatT agg = first_weight * input[first_row_idx * blockDim.x + threadIdx.x];

    // TODO(cvangysel): maybe avoid the loop and shell out for atomicAdd instead?
    for (size_t i = 1; i < cluster_size; ++i) {
        const size_t row_idx = first_row_idx + i;
        const FloatT weight = (weights != nullptr ? weights[row_idx] : 1.0);

        agg = op(agg, weight * input[row_idx * blockDim.x + threadIdx.x]);
    }

    folded[blockIdx.x * blockDim.x + threadIdx.x] = agg;
}


template <typename FloatT, typename AggOp>
device_matrix<FloatT>* fold_columns(const cudaStream_t stream,
                                    const device_matrix<FloatT>& input,
                                    const size_t cluster_size,
                                    const device_matrix<FloatT>* const weights,
                                    const AggOp op) {
    PROFILE_FUNCTION_WITH_STREAM(stream);

    CHECK_GT(cluster_size, 1);

    CHECK_LE(input.getRows(), MAX_THREADS_PER_BLOCK);
    LOG_IF_EVERY_N(WARNING, (input.getRows() % 32 != 0), 100)
        << "Row dimensionality should be a multiple of 32 to optimize efficiency.";

    CHECK_EQ(input.getCols() % cluster_size, 0);

    if (weights != nullptr) {
        CHECK_DIMENSIONS(*weights, 1, input.getCols());
    }

    std::unique_ptr<device_matrix<FloatT>> folded(
        new device_matrix<FloatT>(input.getRows(), input.getCols() / cluster_size,
                                  stream));

    LAUNCH_KERNEL(
        fold_columns_kernel<FloatT, AggOp><<<folded->getCols(), folded->getRows(), 0, stream>>>(
            cluster_size,
            input.getData(),
            (weights != nullptr ? raw_begin(*weights) : nullptr),
            folded->getData(),
            op));

    return folded.release();
}

// Reducing.

// Implements sequential addressing from
// https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf.
template <typename FloatT, typename TransformOp>
__global__
void reduce_axis_kernel(const FloatT* const input,
                        const uint32 num_reduce,
                        const uint32 idx_stride,
                        const uint32 reduce_stride,
                        FloatT* const reduced,
                        const TransformOp op,
                        const uint32 max_threads_per_block) {
    extern __shared__ FloatT sdata[];

    const uint32 elements_per_thread = (num_reduce > max_threads_per_block) ?
        num_reduce / max_threads_per_block : 1;

    FloatT agg = 0.0;

    // Every thread loads one value into shared memory.
    const uint32 lower = blockIdx.x * idx_stride;

    for (uint32 i = 0; i < elements_per_thread; ++i) {
        agg += op(input[lower + (threadIdx.x * elements_per_thread + i) * reduce_stride]);
    }

    sdata[threadIdx.x] = agg;

    // Barrier.
    __syncthreads();

    uint32 upper = 1;
    {
        uint32 block_dim = blockDim.x;
        while (block_dim >>= 1) upper <<= 1;
    }

    for (uint32 stride = upper; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < blockDim.x) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }

        __syncthreads();
    }

    // Write aggregate.
    if (threadIdx.x == 0) reduced[blockIdx.x] = sdata[0];
}

template <typename FloatT, typename TransformOp>
void reduce_axis(const cudaStream_t stream,
                 const AxisDesc axis,
                 const device_matrix<FloatT>& src,
                 device_matrix<FloatT>* const output,
                 const TransformOp op) {
    PROFILE_FUNCTION_WITH_STREAM(stream);

    CHECK_MATRIX(src);

    const size_t reduced_axis_size = (axis == FIRST_AXIS) ?
        src.getRows() : src.getCols();

    const size_t retained_axis_size = (axis == FIRST_AXIS) ?
        src.getCols() : src.getRows();

    if (reduced_axis_size > MAX_THREADS_PER_BLOCK) {
        CHECK_EQ(reduced_axis_size % MAX_THREADS_PER_BLOCK, 0);
    } else {
        LOG_IF_EVERY_N(WARNING, (reduced_axis_size % 32 != 0), 100)
            << "Row dimensionality should be a multiple of 32 to optimize efficiency.";
    }

    if (axis == FIRST_AXIS) {
        CHECK_EQ(output->getRows(), 1);
        CHECK_EQ(output->getCols(), retained_axis_size);
    } else if (axis == SECOND_AXIS) {
        CHECK_EQ(output->getRows(), retained_axis_size);
        CHECK_EQ(output->getCols(), 1);
    } else {
        LOG(FATAL) << "Unknown axis.";
    }

    const size_t idx_stride = (axis == FIRST_AXIS) ?
        reduced_axis_size : 1;

    const size_t reduce_stride = (axis == FIRST_AXIS) ?
        1 : retained_axis_size;

    if (reduced_axis_size > MAX_THREADS_PER_BLOCK) {
        CHECK_EQ(reduced_axis_size % MAX_THREADS_PER_BLOCK, 0);
    }

    const size_t num_threads = min(
        reduced_axis_size,
        static_cast<size_t>(MAX_THREADS_PER_BLOCK));

    LAUNCH_KERNEL(
        reduce_axis_kernel<FloatT, TransformOp>
            <<<retained_axis_size, /* grid dim */
               num_threads, /* block dim */
               num_threads * sizeof(FloatT), /* shared memory */
               stream>>>(
            src.getData(),
            reduced_axis_size,
            idx_stride,
            reduce_stride,
            output->getData(),
            op,
            MAX_THREADS_PER_BLOCK));
}

template <typename DevicePolicy, typename FloatT, typename BinaryFn, typename FirstOpOp>
void elemwise_binary(const DevicePolicy& exec,
                     const device_matrix<FloatT>& first_op,
                     device_matrix<FloatT>* const second_op_and_dst,
                     const BinaryFn binary_fn,
                     const FirstOpOp first_op_op) {
    PROFILE_FUNCTION();

    CHECK_DIMENSIONS_EQUAL(first_op, *second_op_and_dst);

    thrust::transform(
        exec,
        thrust::make_transform_iterator(
            begin(first_op),
            first_op_op),
        thrust::make_transform_iterator(
            end(first_op),
            first_op_op), /* first op */
        begin(*second_op_and_dst), /* second op */
        begin(*second_op_and_dst), /* output */
        binary_fn);
}

template <typename DevicePolicy, typename FloatT, typename FirstOpOp>
void elemwise_plus(const DevicePolicy& exec,
                   const device_matrix<FloatT>& first_op,
                   device_matrix<FloatT>* const second_op_and_dst,
                   const FirstOpOp first_op_op) {
    PROFILE_FUNCTION();

    elemwise_binary(exec, first_op, second_op_and_dst, thrust::plus<FloatT>(), first_op_op);
}

template <typename DevicePolicy, typename FloatT, typename FirstOpOp>
void hadamard_product(const DevicePolicy& exec,
                      const device_matrix<FloatT>& first_op,
                      device_matrix<FloatT>* const second_op_and_dst,
                      const FirstOpOp first_op_op) {
    PROFILE_FUNCTION();

    elemwise_binary(exec, first_op, second_op_and_dst, thrust::multiplies<FloatT>(), first_op_op);
}

// TODO(cvangysel): refactor this such that it uses the interface from above.
template <typename FloatT>
device_matrix<FloatT>* hadamard_product(const cudaStream_t stream,
                                        const device_matrix<FloatT>& first_op,
                                        const device_matrix<FloatT>& second_op) {
    PROFILE_FUNCTION();

    CHECK_DIMENSIONS_EQUAL(first_op, second_op);

    device_matrix<FloatT>* multiplied_matrices = new device_matrix<FloatT>(
        first_op.getRows(), first_op.getCols(), stream);

    thrust::transform(
        thrust::cuda::par.on(stream),
        begin(first_op), end(first_op), /* first op */
        begin(second_op), /* second op */
        begin(*multiplied_matrices), /* output */
        thrust::multiplies<FloatT>());

    return multiplied_matrices;
}

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_elemwise(const DevicePolicy& exec,
                    device_matrix<FloatT>* const dmat,
                    const OpT op) {
    PROFILE_FUNCTION();

    thrust::transform(exec,
                      begin(*dmat), end(*dmat), begin(*dmat), op);
}

template <typename FloatT>
thrust::transform_iterator<func::divide_by_constant<size_t>, thrust::counting_iterator<size_t>>
make_matrix_column_iterator(const device_matrix<FloatT>& dmat) {
    return thrust::transform_iterator<func::divide_by_constant<size_t>, thrust::counting_iterator<size_t>>(
        thrust::counting_iterator<size_t>(0),
        func::divide_by_constant<size_t>(dmat.getRows()));
}

template <typename FloatT, typename IteratorT>
thrust::transform_iterator<func::scale_by_constant<FloatT>, IteratorT>
make_scalar_multiplication_iterator(IteratorT it, const FloatT scalar) {
    return thrust::make_transform_iterator(it, func::scale_by_constant<FloatT>(scalar));
}

// For every element in column c_i, apply the operation between that element
// and the element on the i'th position in the vector.
template <typename OpT, typename FirstOpOpT, typename SecondOpOpT, typename DevicePolicy, typename FloatT>
void apply_columnwise(const DevicePolicy& exec,
                      const device_matrix<FloatT>& src_matrix,
                      const device_matrix<FloatT>& vector,
                      device_matrix<FloatT>* const dst_matrix,
                      const FirstOpOpT first_op_op,
                      const SecondOpOpT second_op_op) {
    PROFILE_FUNCTION();

    CHECK_DIMENSIONS_EQUAL(src_matrix, *dst_matrix);
    CHECK_DIMENSIONS(vector, 1, dst_matrix->getCols());

    thrust::transform(
        exec,
        thrust::make_transform_iterator(begin(src_matrix), first_op_op),
        thrust::make_transform_iterator(end(src_matrix), first_op_op), /* input */
        thrust::make_permutation_iterator(
            thrust::make_transform_iterator(begin(vector), second_op_op), /* elements */
            make_matrix_column_iterator(*dst_matrix) /* map */),
        begin(*dst_matrix), /* result */
        OpT());
}

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_columnwise(const DevicePolicy& exec,
                      const device_matrix<FloatT>& vector,
                      device_matrix<FloatT>* const dst_matrix) {
    apply_columnwise<OpT, thrust::identity<FloatT>, thrust::identity<FloatT>, DevicePolicy, FloatT>(
        exec, *dst_matrix, vector, dst_matrix);
}

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_except_every_Nth_column(const DevicePolicy& exec,
                                   const size_t col_idx,
                                   device_matrix<FloatT>* const matrix) {
    PROFILE_FUNCTION();

    thrust::transform_if(
        exec,
        begin(*matrix), end(*matrix), /* input */
        thrust::counting_iterator<size_t>(0), /* stencil (idx of element) */
        begin(*matrix), /* result */
        OpT(),
        thrust::not1(
            func::is_Nth_col(col_idx, matrix->getRows())));
}

// Only every k-th row (including 0) will be updated; thus the passed iterator will not
// be applied everywhere.
template <typename OpT, typename DevicePolicy, typename FloatT, typename InputIterator>
void apply_every_Nth_column(const DevicePolicy& exec,
                            InputIterator begin_it,
                            InputIterator end_it,
                            const size_t col_idx,
                            device_matrix<FloatT>* const matrix,
                            OpT op) {
    PROFILE_FUNCTION();

    CHECK_EQ(end_it - begin_it, matrix->size());

    thrust::transform_if(
        exec,
        begin_it, end_it,
        thrust::counting_iterator<size_t>(0), /* stencil (idx of element) */
        begin(*matrix), /* result */
        op,
        func::is_Nth_col(col_idx, matrix->getRows()));
}

template <typename OpT, typename DevicePolicy, typename FloatT>
void apply_every_Nth_column(const DevicePolicy& exec,
                            const size_t col_idx,
                            device_matrix<FloatT>* const matrix,
                            OpT op) {
    return apply_every_Nth_column<OpT>(
        exec,
        begin(*matrix), end(*matrix),
        col_idx,
        matrix,
        op);
}

}  // namespace cuda

#endif /* DEVICE_MATRIX_INLINE_H */
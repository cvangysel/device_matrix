#ifndef DEVICE_MATRIX_INLINE_H
#define DEVICE_MATRIX_INLINE_H

namespace cuda {

template <typename FloatT>
inline std::ostream& operator<<(std::ostream& os,
                                const device_matrix<FloatT>& matrix) {
    os << matrix.getRows() << "-by-" << matrix.getCols() << " matrix "
       << "on GPU at " << matrix.getData();

    return os;
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
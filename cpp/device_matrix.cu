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

// Explicit instantiation.
template decltype(begin<FLOATING_POINT_TYPE>) begin;
template decltype(broadcast_columns<FLOATING_POINT_TYPE>) broadcast_columns;
template decltype(end<FLOATING_POINT_TYPE>) end;
template decltype(flatten<FLOATING_POINT_TYPE>) flatten;
template decltype(flip_adjacent_columns<FLOATING_POINT_TYPE>) flip_adjacent_columns;
template decltype(get_array<FLOATING_POINT_TYPE>) get_array;
template decltype(get_scalar<FLOATING_POINT_TYPE>) get_scalar;
template decltype(hstack<FLOATING_POINT_TYPE>) hstack;
template decltype(isfinite<FLOATING_POINT_TYPE>) isfinite;
template decltype(isfinite_slow<FLOATING_POINT_TYPE>) isfinite_slow;
template decltype(l2_norm<FLOATING_POINT_TYPE>) l2_norm;
template decltype(matrix_mult<FLOATING_POINT_TYPE>) matrix_mult;
template decltype(repmat<FLOATING_POINT_TYPE>) repmat;
template decltype(print_matrix<FLOATING_POINT_TYPE>) print_matrix;
template decltype(print_vector<FLOATING_POINT_TYPE>) print_vector;
template decltype(raw_begin<FLOATING_POINT_TYPE>) raw_begin;
template decltype(raw_end<FLOATING_POINT_TYPE>) raw_end;

template class Runtime<FLOATING_POINT_TYPE>;

}  // namespace cuda

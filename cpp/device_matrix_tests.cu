#include "device_matrix/device_matrix.h"

#include <glog/logging.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace cuda {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

// Helpers.
template <typename FloatT>
std::vector<FloatT> to_host(const device_matrix<FloatT>& device_mat) {
    FloatT* data = get_array(device_mat.getStream(), device_mat);
    std::vector<FloatT> v(data, data + device_mat.size());
    delete [] data;

    return v;
}

template <typename FloatT>
void to_device(const std::vector<FloatT>& data, device_matrix<FloatT>* const matrix) {
    matrix->fillwith(matrix->getStream(), data);
}

template <typename FloatT>
inline std::vector<FloatT> range(size_t start, size_t end, size_t repeat = 1) {
    std::vector<FloatT> v;

    for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < repeat; ++j) v.push_back(i);
    }

    return v;
}

typedef FLOATING_POINT_TYPE FloatT;

class CudaUtilsTest : public ::testing::TestWithParam<size_t> {
 protected:
  CudaUtilsTest() {
      // Shifts the starting address of future allocations.
      filler_.reset(new device_matrix<FloatT>(
          GetParam(), GetParam() + GetParam(), NULL));
      filler_->fillwith(NULL, GetParam());

      // Initializes the GPU memory to something different than zero.
      device_matrix<FloatT> another_filler(
          GetParam(), GetParam() + GetParam(), NULL);
      another_filler.fillwith(NULL, GetParam());
  }

  virtual void TearDown() {
      CCE(cudaDeviceSynchronize());
      CCE(cudaGetLastError());
  }

  std::unique_ptr<device_matrix<FloatT>> filler_;
};

INSTANTIATE_TEST_CASE_P(MemoryFiller,
                        CudaUtilsTest,
                        ::testing::Range<size_t>(1 /* start, inclusive */,
                                                 11 /* end, exclusive */,
                                                 1 /* step */));

typedef CudaUtilsTest DeviceMatrixTest;

INSTANTIATE_TEST_CASE_P(MemoryFiller,
                        DeviceMatrixTest,
                        ::testing::Range<size_t>(1 /* start, inclusive */,
                                                 11 /* end, exclusive */,
                                                 1 /* step */));

TEST_P(DeviceMatrixTest, fillwith) {
    device_matrix<FloatT> matrix(3, 4, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    5.0);

    EXPECT_THAT(to_host(matrix),
                ElementsAreArray(std::vector<FloatT>(12 /* number */,
                                                     5.0 /* value */)));
}

TEST_P(DeviceMatrixTest, scale) {
    device_matrix<FloatT> matrix(3, 4, NULL /* stream */);
    to_device({1., 2., 3., 4., 5., 6.,
               7., 8., 9., 10., 11., 12.0},
              &matrix);

    matrix.scale(NULL, /* stream */
                 0.5);

    EXPECT_THAT(to_host(matrix),
                ElementsAreArray({0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                                  3.5, 4.0, 4.5, 5.0, 5.5, 6.0}));
}

TEST_P(DeviceMatrixTest, copy) {
    device_matrix<FloatT> matrix(100, 100, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    1.0);

    std::unique_ptr<device_matrix<FloatT>> new_matrix(matrix.copy(NULL /* stream */));

    EXPECT_EQ(to_host(matrix), to_host(*new_matrix));
}

TEST_P(DeviceMatrixTest, matrix_mult) {
    device_matrix<FloatT> op1(100, 300, NULL /* stream */);
    op1.fillwith(NULL, /* stream */
                 3.0);
    device_matrix<FloatT> op2(20, 300, NULL /* stream */);
    op2.fillwith(NULL, /* stream */
                 5.0);

    device_matrix<FloatT> result(100, 20, NULL /* stream */);
    result.fillwith(NULL, /* stream */
                    100.0);

    matrix_mult(NULL, /* stream */
                op1, CUBLAS_OP_N,
                op2, CUBLAS_OP_T,
                &result);

    EXPECT_THAT(to_host(result),
                ElementsAreArray(std::vector<FloatT>(100 * 20 /* number */,
                                                     15.0 * 300 /* value */)));
}

TEST_P(DeviceMatrixTest, flip_adjacent_columns) {
    device_matrix<FloatT> matrix(3, 4, NULL /* stream */);
    to_device({1.0, 2.0, 3.0,
               4.0, 5.0, 6.0,
               7.0, 8.0, 9.0,
               10.0, 11.0, 12.0},
              &matrix);

    flip_adjacent_columns(matrix.getStream(), &matrix);

    EXPECT_THAT(to_host(matrix),
                ElementsAreArray({
                    4.0, 5.0, 6.0,
                    1.0, 2.0, 3.0,
                    10.0, 11.0, 12.0,
                    7.0, 8.0, 9.0}));
}

TEST_P(DeviceMatrixTest, repmat) {
    device_matrix<FloatT> matrix(4, 3, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0},
              &matrix);

    std::unique_ptr<device_matrix<FloatT>> repeated(
        repmat(matrix.getStream(),
               matrix,
               3 /* num_repeats */));

    CHECK_EQ(repeated->getRows(), 4);
    CHECK_EQ(repeated->getCols(), 3 * 3);

    EXPECT_THAT(to_host(*repeated),
                ElementsAreArray({
                    1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0}));
}

TEST_P(DeviceMatrixTest, hstack_uniform) {
    device_matrix<FloatT> first(4, 2, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0},
              &first);

    device_matrix<FloatT> second(4, 1, NULL /* stream */);
    to_device({9.0, 10.0, 11.0, 12.0},
              &second);

    device_matrix<FloatT> third(4, 1, NULL /* stream */);
    to_device({13.0, 14.0, 15.0, 16.0},
              &third);

    device_matrix<FloatT> fourth(4, 4, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0,
               13.0, 14.0, 15.0, 16.0},
              &fourth);

    std::vector<std::pair<device_matrix<FloatT>*, FloatT>> pairs =
        {std::make_pair(&first, 1.0),
         std::make_pair(&second, 1.0),
         std::make_pair(&third, 1.0),
         std::make_pair(&fourth, 1.0)};

    std::unique_ptr<device_matrix<FloatT>> hstacked(
        hstack(NULL, /* stream */
               pairs));

    EXPECT_THAT(to_host(*hstacked),
                ElementsAreArray({1.0, 2.0, 3.0, 4.0,
                                  5.0, 6.0, 7.0, 8.0,
                                  9.0, 10.0, 11.0, 12.0,
                                  13.0, 14.0, 15.0, 16.0,
                                  1.0, 2.0, 3.0, 4.0,
                                  5.0, 6.0, 7.0, 8.0,
                                  9.0, 10.0, 11.0, 12.0,
                                  13.0, 14.0, 15.0, 16.0}));
}

TEST_P(DeviceMatrixTest, hstack_weighted) {
    device_matrix<FloatT> first(4, 2, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0},
              &first);

    device_matrix<FloatT> second(4, 1, NULL /* stream */);
    to_device({9.0, 10.0, 11.0, 12.0},
              &second);

    device_matrix<FloatT> third(4, 1, NULL /* stream */);
    to_device({13.0, 14.0, 15.0, 16.0},
              &third);

    std::vector<std::pair<device_matrix<FloatT>*, FloatT>> pairs =
        {std::make_pair(&first, 0.5),
         std::make_pair(&second, 1.0),
         std::make_pair(&third, -0.5)};

    std::unique_ptr<device_matrix<FloatT>> hstacked(
        hstack(NULL, /* stream */
               pairs));

    EXPECT_THAT(to_host(*hstacked),
                ElementsAreArray({0.5, 1.0, 1.5, 2.0,
                                  2.5, 3.0, 3.5, 4.0,
                                  9.0, 10.0, 11.0, 12.0,
                                  -6.5, -7.0, -7.5, -8.0}));
}

TEST_P(DeviceMatrixTest, broadcast_columns) {
    device_matrix<FloatT> matrix(4, 3, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0},
              &matrix);

    ASSERT_LE(matrix.getRows(), MAX_THREADS_PER_BLOCK);

    // Triggers second condition; as 1 < rows < MAX_THREADS_PER_BLOCK.
    std::unique_ptr<device_matrix<FloatT>> broadcasted(
        broadcast_columns(NULL, /* stream */
                          matrix,
                          5 /* num_repeats */));

    EXPECT_THAT(to_host(*broadcasted),
                ElementsAreArray({1.0, 2.0, 3.0, 4.0,
                                  1.0, 2.0, 3.0, 4.0,
                                  1.0, 2.0, 3.0, 4.0,
                                  1.0, 2.0, 3.0, 4.0,
                                  1.0, 2.0, 3.0, 4.0,
                                  5.0, 6.0, 7.0, 8.0,
                                  5.0, 6.0, 7.0, 8.0,
                                  5.0, 6.0, 7.0, 8.0,
                                  5.0, 6.0, 7.0, 8.0,
                                  5.0, 6.0, 7.0, 8.0,
                                  9.0, 10.0, 11.0, 12.0,
                                  9.0, 10.0, 11.0, 12.0,
                                  9.0, 10.0, 11.0, 12.0,
                                  9.0, 10.0, 11.0, 12.0,
                                  9.0, 10.0, 11.0, 12.0}));
}

TEST_P(DeviceMatrixTest, broadcast_columns_many_repeats) {
    device_matrix<FloatT> matrix(4, 3, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0},
              &matrix);

    ASSERT_LE(matrix.getRows(), MAX_THREADS_PER_BLOCK);

    // Triggers second condition; as 1 < rows < MAX_THREADS_PER_BLOCK.
    std::unique_ptr<device_matrix<FloatT>> broadcasted(
        broadcast_columns(NULL, /* stream */
                          matrix,
                          2048 /* num_repeats */));

    EXPECT_EQ(broadcasted->getCols(), 3 * 2048);
}

TEST_P(DeviceMatrixTest, broadcast_columns_single_row) {
    device_matrix<FloatT> matrix(1, 2048, NULL /* stream */);
    matrix.fillwith(NULL, range<FloatT>(1, 2049));

    // Triggers first condition; as rows = 1.
    std::unique_ptr<device_matrix<FloatT>> broadcasted(
        broadcast_columns(NULL, /* stream */
                          matrix,
                          2 /* num_repeats */));

    EXPECT_EQ(broadcasted->getRows(), 1);
    EXPECT_EQ(broadcasted->getCols(), 2 * 2048);

    EXPECT_THAT(to_host(*broadcasted),
                ElementsAreArray(range<FloatT>(1, 2049, 2 /* repeats */)));
}

std::vector<FloatT> simulate_broadcast_columns(const std::vector<FloatT>& data,
                                               const size_t num_rows, const size_t num_cols,
                                               const size_t num_repeats) {
    CHECK_EQ(data.size(), num_rows * num_cols);
    std::vector<FloatT> v;

    for (size_t i = 0; i < num_cols; ++i) {
        for (size_t j = 0; j < num_repeats; ++j) {
            v.insert(v.end(), &data[i * num_rows], &data[(i + 1) * num_rows]);
        }
    }

    return v;
}

TEST_P(DeviceMatrixTest, broadcast_columns_large_row) {
    const std::vector<FloatT> data = range<FloatT>(1, 1500 * 1024 + 1);

    device_matrix<FloatT> matrix(1500, 1024, NULL /* stream */);
    matrix.fillwith(NULL, data);

    // Triggers first condition; as 1 < MAX_THREADS_PER_BLOCK < rows.
    std::unique_ptr<device_matrix<FloatT>> broadcasted(
        broadcast_columns(NULL, /* stream */
                          matrix,
                          11 /* num_repeats */));

    EXPECT_EQ(broadcasted->getRows(), 1500);
    EXPECT_EQ(broadcasted->getCols(), 11 * 1024);

    EXPECT_THAT(to_host(*broadcasted),
                ElementsAreArray(simulate_broadcast_columns(data, 1500, 1024, 11)));
}

TEST_P(DeviceMatrixTest, fold_columns) {
    device_matrix<FloatT> matrix(3, 4, NULL /* stream */);
    to_device({1.0, 2.0, 3.0,
               4.0, 5.0, 6.0,
               7.0, 8.0, 9.0,
               10.0, 11.0, 12.0},
              &matrix);

    std::unique_ptr<device_matrix<FloatT>> folded(
        fold_columns(NULL, /* stream */
                     matrix,
                     2 /* cluster_size */));

    EXPECT_THAT(to_host(*folded),
                ElementsAreArray({5.0, 7.0, 9.0,
                                  17.0, 19.0, 21.0}));
}

TEST_P(DeviceMatrixTest, fold_columns_mult) {
    device_matrix<FloatT> matrix(3, 4, NULL /* stream */);
    to_device({1.0, 2.0, 3.0,
               4.0, 5.0, 6.0,
               7.0, 8.0, 9.0,
               10.0, 11.0, 12.0},
              &matrix);

    std::unique_ptr<device_matrix<FloatT>> folded(
        fold_columns<FloatT, thrust::multiplies<FloatT>>(
            NULL, /* stream */
            matrix,
            2 /* cluster_size */));

    EXPECT_THAT(to_host(*folded),
                ElementsAreArray({4.0, 10.0, 18.0,
                                  70.0, 88.0, 108.0}));
}

TEST_P(DeviceMatrixTest, fold_columns_weights) {
    device_matrix<FloatT> matrix(3, 4, NULL /* stream */);
    to_device({1.0, 2.0, 3.0,
               4.0, 5.0, 6.0,
               7.0, 8.0, 9.0,
               10.0, 11.0, 12.0},
              &matrix);

    device_matrix<FloatT> weights(1, 4, NULL /* stream */);
    to_device({0.5, 0.5, 1.0, 2.0}, &weights);

    std::unique_ptr<device_matrix<FloatT>> folded(
        fold_columns(NULL, /* stream */
                     matrix,
                     2, /* cluster_size */
                     &weights));

    EXPECT_THAT(to_host(*folded),
                ElementsAreArray({2.5, 3.5, 4.5,
                                  27.0, 30.0, 33.0}));
}

TEST_P(CudaUtilsTest, apply_columnwise) {
    device_matrix<FloatT> matrix(2, 4, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    1.0);

    device_matrix<FloatT> vector(1, 4, NULL /* stream */);
    to_device({0.5, 2.0, 3.5, 10.0}, &vector);

    apply_columnwise<thrust::multiplies<FloatT>>(
        thrust::device,
        vector, &matrix);

    thrust::host_vector<FloatT> result(begin(matrix), end(matrix));
    EXPECT_THAT(to_host(matrix), ElementsAre(0.5, 0.5,
                                             2.0, 2.0,
                                             3.5, 3.5,
                                             10.0, 10.0));
}

TEST_P(CudaUtilsTest, apply_except_every_2nd_column) {
    device_matrix<FloatT> matrix(2, 5, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    1.0);

    apply_except_every_Nth_column<thrust::negate<FloatT>>(
        thrust::device,
        2 /* col_idx */, &matrix);

    EXPECT_THAT(to_host(matrix), ElementsAre(1.0, 1.0,
                                             -1.0, -1.0,
                                             1.0, 1.0,
                                             -1.0, -1.0,
                                             1.0, 1.0));
}

TEST_P(CudaUtilsTest, apply_except_every_3rd_column) {
    device_matrix<FloatT> matrix(2, 5, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    1.0);

    apply_except_every_Nth_column<thrust::negate<FloatT>>(
        thrust::device,
        3 /* col_idx */, &matrix);

    EXPECT_THAT(to_host(matrix), ElementsAre(1.0, 1.0,
                                             -1.0, -1.0,
                                             -1.0, -1.0,
                                             1.0, 1.0,
                                             -1.0, -1.0));
}

TEST_P(CudaUtilsTest, apply_every_3rd_column) {
    device_matrix<FloatT> matrix(2, 5, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    1.0);

    apply_every_Nth_column<thrust::negate<FloatT>>(
        thrust::device,
        thrust::make_constant_iterator(10.0),
        thrust::make_constant_iterator(10.0) + matrix.size(),
        3, /* col_idx */
        &matrix);

    EXPECT_THAT(to_host(matrix), ElementsAre(-10.0, -10.0,
                                             1.0, 1.0,
                                             1.0, 1.0,
                                             -10.0, -10.0,
                                             1.0, 1.0));

    apply_every_Nth_column<thrust::negate<FloatT>>(
        thrust::device,
        3, /* col_idx */
        &matrix);

    EXPECT_THAT(to_host(matrix), ElementsAre(10.0, 10.0,
                                             1.0, 1.0,
                                             1.0, 1.0,
                                             10.0, 10.0,
                                             1.0, 1.0));
}

TEST_P(CudaUtilsTest, reduce_axis) {
    device_matrix<FloatT> matrix(2, 5, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    1.0);

    device_matrix<FloatT> reduced_first(1, 5, NULL /* stream */);
    device_matrix<FloatT> reduced_second(2, 1, NULL /* stream */);

    reduce_axis(
        NULL, /* stream */
        FIRST_AXIS,
        matrix,
        &reduced_first);

    reduce_axis(
        NULL, /* stream */
        SECOND_AXIS,
        matrix,
        &reduced_second);

    EXPECT_THAT(to_host(reduced_first), ElementsAre(2.0, 2.0, 2.0, 2.0, 2.0));
    EXPECT_THAT(to_host(reduced_second), ElementsAre(5.0, 5.0));
}

TEST_P(CudaUtilsTest, reduce_axis_square) {
    device_matrix<FloatT> matrix(2, 5, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    {1.0, 6.0,
                     2.0, 7.0,
                     3.0, 8.0,
                     4.0, 9.0,
                     5.0, 10.0});

    device_matrix<FloatT> reduced_first(1, 5, NULL /* stream */);

    reduce_axis<FloatT, func::square<FloatT>>(
        NULL, /* stream */
        FIRST_AXIS,
        matrix,
        &reduced_first);

    EXPECT_THAT(to_host(reduced_first),
                ElementsAre(37.0, 53.0, 73.0, 97.0, 125.0));
}

TEST_P(CudaUtilsTest, reduce_axis_vector) {
    std::unique_ptr<device_matrix<FloatT>> column(
        device_matrix<FloatT>::create_column(NULL, /* stream */
                                             {1.0, 2.0, 3.0, 10.0}));

    std::unique_ptr<device_matrix<FloatT>> row(
        new device_matrix<FloatT>(1, 4,
                                  NULL /* stream */));
    row->fillwith(NULL, /* stream */
                  {1.0, 2.0, 3.0, 10.0});

    device_matrix<FloatT> scalar_first(1, 1, NULL);
    device_matrix<FloatT> scalar_second(1, 1, NULL);

    reduce_axis(
        NULL, /* stream */
        FIRST_AXIS,
        *column,
        &scalar_first);

    reduce_axis(
        NULL, /* stream */
        SECOND_AXIS,
        *row,
        &scalar_second);

    EXPECT_THAT(to_host(scalar_first), ElementsAre(16.0));
    EXPECT_THAT(to_host(scalar_second), ElementsAre(16.0));
}

TEST_P(CudaUtilsTest, reduce_axis_large) {
    device_matrix<FloatT> matrix(2048, 4096, NULL /* stream */);
    matrix.fillwith(NULL, /* stream */
                    1.0);

    device_matrix<FloatT> reduced_first(1, 4096, NULL /* stream */);
    device_matrix<FloatT> reduced_second(2048, 1, NULL /* stream */);

    reduce_axis(
        NULL, /* stream */
        FIRST_AXIS,
        matrix,
        &reduced_first);

    reduce_axis(
        NULL, /* stream */
        SECOND_AXIS,
        matrix,
        &reduced_second);

    EXPECT_THAT(to_host(reduced_first),
                ElementsAreArray(std::vector<FloatT>(4096, 2048.0)));
    EXPECT_THAT(to_host(reduced_second),
                ElementsAreArray(std::vector<FloatT>(2048, 4096.0)));
}

TEST_P(CudaUtilsTest, make_scalar_multiplication_iterator) {
    device_matrix<FloatT> matrix(2 /* num_rows */, 5 /* num_cols */, NULL /* stream */);

    to_device(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        &matrix);

    auto it = make_scalar_multiplication_iterator(begin(matrix), 2.0);

    thrust::device_vector<FloatT> result(it, it + 10);
    EXPECT_THAT(result, ElementsAre(2, 4, 6, 8, 10, 12, 14, 16, 18, 20));
}

}  // namespace cuda

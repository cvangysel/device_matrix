#ifndef DEVICE_MATRIX_FUNC_H
#define DEVICE_MATRIX_FUNC_H

namespace cuda {
namespace func {

using thrust::identity;

template <typename FloatT>
struct divide_by_constant {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  explicit divide_by_constant(const FloatT c) : c_(c) {}
  __host__ __device__
  FloatT operator()(const FloatT idx) {
      return idx / c_;
  }
 private:
  const FloatT c_;
};

template <typename FloatT>
struct scale_by_constant {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  explicit scale_by_constant(const FloatT c) : c_(c) {}
  __host__ __device__
  FloatT operator()(const FloatT val) {
      return val * c_;
  }
 private:
  const FloatT c_;
};

template <typename FloatT>
struct add_constant {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  explicit add_constant(const FloatT c) : c_(c) {}
  __host__ __device__
  FloatT operator()(const FloatT val) {
      return val + c_;
  }
 private:
  const FloatT c_;
};

template <typename FloatT>
struct divides_tuple {
  typedef thrust::tuple<FloatT, FloatT> argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const thrust::tuple<FloatT, FloatT> tuple) {
      return thrust::get<0>(tuple) / thrust::get<1>(tuple);
  }
};

struct is_Nth_col {
  typedef size_t argument_type;
  typedef bool result_type;
  is_Nth_col(const size_t col_idx, const size_t num_rows)
      : col_idx_(col_idx), num_rows_(num_rows) {}
  __host__ __device__
  result_type operator()(const argument_type element_idx) {
      return (element_idx / num_rows_) % col_idx_ == 0;
  }
 private:
  const size_t col_idx_;
  const size_t num_rows_;
};

template <typename FloatT>
struct square {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  inline FloatT operator()(const FloatT& x) const {
      return x * x;
  }
};

template <typename FloatT>
struct power {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  explicit power(const FloatT exponent) : exponent_(exponent) {}
  __host__ __device__
  inline FloatT operator()(const FloatT& x) const {
      return ::pow(x, exponent_);
  }
 private:
  const FloatT exponent_;
};

template <typename FloatT>
struct isfinite {
    __host__ __device__
    bool operator()(const FloatT& x) const {
        return ::isfinite(x);
    }
};

template <typename FloatT>
struct tanh {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const FloatT x) {
      return ::tanh(x);
  }
};

template <typename FloatT>
struct log {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const FloatT x) {
      return ::log(x);
  }
};

}  // namespace func
}  // namespace cuda

#endif /* DEVICE_MATRIX_FUNC_H */
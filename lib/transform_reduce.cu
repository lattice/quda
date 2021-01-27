#include <reduce_helper.h>
#include <uint_to_char.h>
#include <tune_quda.h>
#include <transform_reduce.h>

namespace quda
{

  template <typename reduce_t, typename T, typename count_t, typename transformer, typename reducer>
  struct TransformReduceArg : public ReduceArg<reduce_t> {
    static constexpr int block_size = 512;
    static constexpr int n_batch_max = 8;
    const T *v[n_batch_max];
    count_t n_items;
    int n_batch;
    reduce_t init_value;
    reduce_t result[n_batch_max];
    transformer h;
    reducer r;
    TransformReduceArg(const std::vector<T *> &v, count_t n_items, transformer h, reduce_t init_value, reducer r) :
      ReduceArg<reduce_t>(v.size()),
      n_items(n_items),
      n_batch(v.size()),
      init_value(init_value),
      h(h),
      r(r)
    {
      if (n_batch > n_batch_max) errorQuda("Requested batch %d greater than max supported %d", n_batch, n_batch_max);
      for (size_t j = 0; j < v.size(); j++) this->v[j] = v[j];
    }

    __device__ __host__ reduce_t init() const { return init_value; }
  };

  template <typename Arg> void transform_reduce(Arg &arg)
  {
    using count_t = decltype(arg.n_items);

    for (int j = 0; j < arg.n_batch; j++) {
      auto v = arg.v[j];
      auto r_ = arg.init();
      for (count_t i = 0; i < arg.n_items; i++) {
        auto v_ = arg.h(v[i]);
        r_ = arg.r(r_, v_);
      }
      arg.result[j] = r_;
    }
  }

  template <typename Arg> __launch_bounds__(Arg::block_size) __global__ void transform_reduce_kernel(Arg arg)
  {
    using count_t = decltype(arg.n_items);

    count_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    auto v = arg.v[j];
    auto r_ = arg.init();

    while (i < arg.n_items) {
      auto v_ = arg.h(v[i]);
      r_ = arg.r(r_, v_);
      i += blockDim.x * gridDim.x;
    }

    reduce<Arg::block_size, 1, decltype(arg.r)>(arg, r_, j);
  }

  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  class TransformReduce : Tunable
  {
    using Arg = TransformReduceArg<reduce_t, T, I, transformer, reducer>;
    QudaFieldLocation location;
    std::vector<reduce_t> &result;
    const std::vector<T *> &v;
    I n_items;
    transformer &h;
    reduce_t init;
    reducer &r;

    bool tuneSharedBytes() const { return false; }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
    int blockMin() const { return Arg::block_size; }
    unsigned int maxBlockSize(const TuneParam &) const { return Arg::block_size; }

    bool advanceTuneParam(TuneParam &param) const // only do autotuning if we have device fields
    {
      return location == QUDA_CUDA_FIELD_LOCATION ? Tunable::advanceTuneParam(param) : false;
    }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.grid.y = v.size();
    }

  public:
    TransformReduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, I n_items,
                    transformer &h, reduce_t init, reducer &r) :
      location(location),
      result(result),
      v(v),
      n_items(n_items),
      h(h),
      init(init),
      r(r)
    {
      strcpy(aux, "batch_size=");
      u32toa(aux + 11, v.size());
      if (location == QUDA_CPU_FIELD_LOCATION) strcat(aux, ",cpu");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Arg arg(v, n_items, h, init, r);

      if (location == QUDA_CUDA_FIELD_LOCATION) {
        arg.launch_error = qudaLaunchKernel(transform_reduce_kernel<Arg>, tp, stream, arg);
        arg.complete(result, stream);
      } else {
        transform_reduce(arg);
        for (size_t j = 0; j < result.size(); j++) result[j] = arg.result[j];
      }
    }

    TuneKey tuneKey() const
    {
      char count[16];
      u32toa(count, n_items);
      return TuneKey(count, typeid(*this).name(), aux);
    }

    long long flops() const { return 0; } // just care about bandwidth
    long long bytes() const { return v.size() * n_items * sizeof(T); }
  };

  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Vector of results
     @param[in] v Vector of inputs
     @param[in] n_items Number of elements to be reduced in each input
     @param[in] transformer Functor that applies transform to each element
     @param[in] init The results are initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  void transform_reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, I n_items,
                        transformer h, reduce_t init, reducer r)
  {
    if (result.size() != v.size())
      errorQuda("result %lu and input %lu set sizes do not match", result.size(), v.size());
    TransformReduce<reduce_t, T, I, transformer, reducer> reduce(location, result, v, n_items, h, init, r);
  }

  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] transformer Functor that applies transform to each element
     @param[in] init Results is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  reduce_t transform_reduce(QudaFieldLocation location, const T *v, I n_items, transformer h, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce(location, result, v_, n_items, h, init, r);
    return result[0];
  }

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] init The results are initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  void reduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v, I n_items,
              reduce_t init, reducer r)
  {
    transform_reduce(location, result, v, n_items, identity<T>(), init, r);
  }

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] init Result is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element
   */
  template <typename reduce_t, typename T, typename I, typename reducer>
  reduce_t reduce(QudaFieldLocation location, const T *v, I n_items, reduce_t init, reducer r)
  {
    std::vector<reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce(location, result, v_, n_items, identity<T>(), init, r);
    return result[0];
  }

  // explicit instantiation list for transform_reduce

  template void transform_reduce<double, complex<float>, int, square_<double, float>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<float> *> const &, int, square_<double, float>,
    double, plus<double>);
  template void transform_reduce<double, complex<int>, int, abs_<double, int>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<int> *> const &, int, abs_<double, int>, double,
    plus<double>);
  template void transform_reduce<double, complex<double>, int, abs_<double, double>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, int, abs_<double, double>, double,
    plus<double>);
  template void transform_reduce<double, complex<float>, int, abs_<float, float>, minimum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<float> *> const &, int, abs_<float, float>, double,
    minimum<float>);
  template void transform_reduce<double, complex<int>, int, square_<double, int>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<int> *> const &, int, square_<double, int>, double,
    plus<double>);
  template void transform_reduce<double, complex<int>, int, abs_<float, int>, maximum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<int> *> const &, int, abs_<float, int>, double,
    maximum<float>);
  template void transform_reduce<double, complex<double>, int, abs_<double, double>, minimum<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, int, abs_<double, double>, double,
    minimum<double>);
  template void transform_reduce<double, complex<float>, int, abs_<float, float>, maximum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<float> *> const &, int, abs_<float, float>, double,
    maximum<float>);
  template void transform_reduce<double, complex<double>, int, square_<double, double>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, int, square_<double, double>,
    double, plus<double>);
  template void transform_reduce<double, complex<int>, int, abs_<float, int>, minimum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<int> *> const &, int, abs_<float, int>, double,
    minimum<float>);
  template void transform_reduce<double, complex<double>, int, abs_<double, double>, maximum<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, int, abs_<double, double>, double,
    maximum<double>);
  template void transform_reduce<double, complex<float>, int, abs_<double, float>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<float> *> const &, int, abs_<double, float>, double,
    plus<double>);
  template void transform_reduce<double, complex<signed char>, int, square_<double, signed char>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<signed char> *> const &, int,
    square_<double, signed char>, double, plus<double>);
  template void transform_reduce<double, complex<short>, int, abs_<float, short>, minimum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<short> *> const &, int, abs_<float, short>, double,
    minimum<float>);
  template void transform_reduce<double, complex<short>, int, square_<double, short>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<short> *> const &, int, square_<double, short>,
    double, plus<double>);
  template void transform_reduce<double, complex<signed char>, int, abs_<float, signed char>, maximum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<signed char> *> const &, int,
    abs_<float, signed char>, double, maximum<float>);
  template void transform_reduce<double, complex<signed char>, int, abs_<float, signed char>, minimum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<signed char> *> const &, int,
    abs_<float, signed char>, double, minimum<float>);
  template void transform_reduce<double, complex<signed char>, int, abs_<double, signed char>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<signed char> *> const &, int,
    abs_<double, signed char>, double, plus<double>);
  template void transform_reduce<double, complex<short>, int, abs_<double, short>, plus<double>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<short> *> const &, int, abs_<double, short>, double,
    plus<double>);
  template void transform_reduce<double, complex<short>, int, abs_<float, short>, maximum<float>>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<short> *> const &, int, abs_<float, short>, double,
    maximum<float>);

  template double transform_reduce<double, complex<double>, unsigned long, abs_<double, double>, minimum<double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, abs_<double, double>, double, minimum<double>);
  template double transform_reduce<double, complex<float>, int, abs_<double, float>, maximum<double>>(
    QudaFieldLocation, complex<float> const *, int, abs_<double, float>, double, maximum<double>);
  template double transform_reduce<double, complex<double>, unsigned long, square_<double, double>, plus<double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, square_<double, double>, double, plus<double>);
  template double transform_reduce<double, complex<double>, int, abs_<double, double>, maximum<double>>(
    QudaFieldLocation, complex<double> const *, int, abs_<double, double>, double, maximum<double>);
  template double transform_reduce<double, complex<float>, unsigned long, abs_<double, float>, plus<double>>(
    QudaFieldLocation, complex<float> const *, unsigned long, abs_<double, float>, double, plus<double>);
  template double transform_reduce<double, complex<double>, unsigned long, abs_<double, double>, plus<double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, abs_<double, double>, double, plus<double>);
  template double transform_reduce<double, complex<float>, unsigned long, abs_<float, float>, maximum<float>>(
    QudaFieldLocation, complex<float> const *, unsigned long, abs_<float, float>, double, maximum<float>);
  template double transform_reduce<double, complex<float>, unsigned long, square_<double, float>, plus<double>>(
    QudaFieldLocation, complex<float> const *, unsigned long, square_<double, float>, double, plus<double>);
  template double transform_reduce<double, complex<float>, unsigned long, abs_<float, float>, minimum<float>>(
    QudaFieldLocation, complex<float> const *, unsigned long, abs_<float, float>, double, minimum<float>);
  template double transform_reduce<double, complex<double>, unsigned long, abs_<double, double>, maximum<double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, abs_<double, double>, double, maximum<double>);
  template double transform_reduce<double, complex<float>, int, square_<double, float>, plus<double>>(
    QudaFieldLocation, complex<float> const *, int, square_<double, float>, double, plus<double>);
  template double transform_reduce<double, complex<short>, int, square_<double, short>, plus<double>>(
    QudaFieldLocation, complex<short> const *, int, square_<double, short>, double, plus<double>);

  template float reduce<float, float, int, maximum<float>>(QudaFieldLocation, float const *, int, float, maximum<float>);

} // namespace quda

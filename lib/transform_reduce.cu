#include <reduce_helper.h>
#include <transform_reduce.h>
#include <tunable_reduction.h>
#include <kernels/transform_reduce.cuh>

namespace quda
{

  template <typename reduce_t, typename T, typename I, typename transformer, typename reducer>
  class TransformReduce : TunableMultiReduction<1>
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
      TunableMultiReduction(n_items, v.size(), location),
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
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Arg arg(v, n_items, h, init, r);
      launch<transform_reducer>(result, tp, stream, arg);
    }

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
    if (result.size() != v.size()) errorQuda("result %lu and input %lu set sizes do not match", result.size(), v.size());
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

#include <reduce_helper.h>
#include <transform_reduce.h>
#include <tunable_reduction.h>
#include <kernels/transform_reduce.cuh>

namespace quda
{

  template <typename reducer, typename T, typename count_t, typename transformer, typename mapper>
  class TransformReduce : TunableMultiReduction
  {
    using reduce_t = typename reducer::reduce_t;
    using Arg = TransformReduceArg<reducer, T, count_t, transformer, mapper>;
    QudaFieldLocation location;
    std::vector<reduce_t> &result;
    const std::vector<T *> &v;
    count_t n_items;
    transformer &h;
    mapper &m;

    bool tuneSharedBytes() const { return false; }

  public:
    TransformReduce(QudaFieldLocation location, std::vector<reduce_t> &result, const std::vector<T *> &v,
                    count_t n_items, transformer &h, mapper &m) :
      TunableMultiReduction(n_items, v.size(), Arg::max_n_batch_block, location),
      location(location),
      result(result),
      v(v),
      n_items(n_items),
      h(h),
      m(m)
    {
      strcpy(aux, ",batch_size=");
      u32toa(aux + strlen(aux), v.size());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Arg arg(v, n_items, h, m);
      launch<transform_reducer, true>(result, tp, stream, arg);
    }

    long long bytes() const { return v.size() * n_items * sizeof(T); }
  };

  template <typename reducer, typename T, typename count_t, typename transformer, typename mapper>
  void transform_reduce(QudaFieldLocation location, std::vector<typename reducer::reduce_t> &result,
                        const std::vector<T *> &v, count_t n_items, transformer h, mapper m)
  {
    if (result.size() != v.size()) errorQuda("result %lu and input %lu set sizes do not match", result.size(), v.size());
    TransformReduce<reducer, T, count_t, transformer, mapper> reduce(location, result, v, n_items, h, m);
  }

  template <typename reducer, typename T, typename count_t, typename transformer>
  typename reducer::reduce_t transform_reduce(QudaFieldLocation location, const T *v, count_t n_items, transformer h)
  {
    std::vector<typename reducer::reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce<reducer>(location, result, v_, n_items, h, identity());
    return result[0];
  }

  template <typename reducer, typename T, typename count_t, typename transformer>
  void reduce(QudaFieldLocation location, std::vector<typename reducer::reduce_t> &result, const std::vector<T *> &v, count_t n_items)
  {
    transform_reduce<reducer>(location, result, v, n_items, identity());
  }

  template <typename reducer, typename T, typename count_t>
  typename reducer::reduce_t reduce(QudaFieldLocation location, const T *v, count_t n_items)
  {
    std::vector<typename reducer::reduce_t> result = {0.0};
    std::vector<const T *> v_ = {v};
    transform_reduce<reducer>(location, result, v_, n_items, identity());
    return result[0];
  }

  // explicit instantiation list for transform_reduce
  // abs
  template void transform_reduce<plus<double>, complex<double>, unsigned int, abs_<double, double>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, unsigned int,
    abs_<double, double>, identity);
  template void transform_reduce<plus<double>, complex<float>, unsigned int, abs_<double, float>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<float> *> const &, unsigned int, abs_<double, float>,
    identity);
  template void transform_reduce<plus<double>, complex<int>, unsigned int, abs_<double, int>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<int> *> const &, unsigned int,
    abs_<double, int>, identity);
  template void transform_reduce<plus<double>, complex<short>, unsigned int, abs_<double, short>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<short> *> const &, unsigned int, abs_<double, short>,
    identity);
  template void transform_reduce<plus<double>, complex<signed char>, unsigned int, abs_<double, signed char>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<signed char> *> const &, unsigned int,
    abs_<double, signed char>, identity);
  template double transform_reduce<plus<double>, complex<double>, unsigned long, abs_<double, double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, abs_<double, double>);
  template double transform_reduce<plus<double>, complex<float>, unsigned long, abs_<double, float>>(
    QudaFieldLocation, complex<float> const *, unsigned long, abs_<double, float>);
  // square
  template void transform_reduce<plus<double>, complex<double>, unsigned int, square_<double, double>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, unsigned int,
    square_<double, double>, identity);
  template void transform_reduce<plus<double>, complex<float>, unsigned int, square_<double, float>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<float> *> const &, unsigned int,
    square_<double, float>, identity);
  template void transform_reduce<plus<double>, complex<int>, unsigned int, square_<double, int>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<int> *> const &, unsigned int, square_<double, int>,
    identity);
  template void transform_reduce<plus<double>, complex<signed char>, unsigned int, square_<double, signed char>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<signed char> *> const &, unsigned int,
    square_<double, signed char>, identity);
  template void transform_reduce<plus<double>, complex<short>, unsigned int, square_<double, short>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<short> *> const &, unsigned int,
    square_<double, short>, identity);
  template double transform_reduce<plus<double>, complex<double>, unsigned long, square_<double, double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, square_<double, double>);
  template double transform_reduce<plus<double>, complex<float>, unsigned long, square_<double, float>>(
    QudaFieldLocation, complex<float> const *, unsigned long, square_<double, float>);
  template double transform_reduce<plus<double>, complex<float>, unsigned int, square_<double, float>>(
    QudaFieldLocation, complex<float> const *, unsigned int, square_<double, float>);
  template double transform_reduce<plus<double>, complex<short>, unsigned int, square_<double, short>>(
    QudaFieldLocation, complex<short> const *, unsigned int, square_<double, short>);
  // abs_max
  template void transform_reduce<maximum<double>, complex<double>, unsigned int, abs_max_<double, double>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, unsigned int,
    abs_max_<double, double>, identity);
  template void transform_reduce<maximum<float>, complex<int>, unsigned int, abs_max_<float, int>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<int> *> const &, unsigned int, abs_max_<float, int>,
    identity);
  template void transform_reduce<maximum<float>, complex<float>, unsigned int, abs_max_<float, float>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<float> *> const &, unsigned int,
    abs_max_<float, float>, identity);
  template void
  transform_reduce<maximum<float>, complex<signed char>, unsigned int, abs_max_<float, signed char>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<signed char> *> const &, unsigned int,
    abs_max_<float, signed char>, identity);
  template void transform_reduce<maximum<float>, complex<short>, unsigned int, abs_max_<float, short>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<short> *> const &, unsigned int,
    abs_max_<float, short>, identity);
  template double transform_reduce<maximum<double>, complex<double>, unsigned long, abs_max_<double, double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, abs_max_<double, double>);
  template double transform_reduce<maximum<double>, complex<double>, unsigned int, abs_max_<double, double>>(
    QudaFieldLocation, complex<double> const *, unsigned int, abs_max_<double, double>);
  template float transform_reduce<maximum<float>, complex<float>, unsigned long, abs_max_<float, float>>(
    QudaFieldLocation, complex<float> const *, unsigned long, abs_max_<float, float>);
  template float transform_reduce<maximum<float>, complex<float>, unsigned int, abs_max_<float, float>>(
    QudaFieldLocation, complex<float> const *, unsigned int, abs_max_<float, float>);
  template float transform_reduce<maximum<float>, complex<short>, unsigned long, abs_max_<float, short>>(
    QudaFieldLocation, complex<short> const *, unsigned long, abs_max_<float, short>);
  template float transform_reduce<maximum<float>, complex<short>, unsigned int, abs_max_<float, short>>(
    QudaFieldLocation, complex<short> const*, unsigned int, abs_max_<float, short>);
  // abs_min
  template void transform_reduce<minimum<double>, complex<double>, unsigned int, abs_min_<double, double>, identity>(
    QudaFieldLocation, std::vector<double> &, std::vector<complex<double> *> const &, unsigned int,
    abs_min_<double, double>, identity);
  template void transform_reduce<minimum<float>, complex<float>, unsigned int, abs_min_<float, float>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<float> *> const &, unsigned int,
    abs_min_<float, float>, identity);
  template void transform_reduce<minimum<float>, complex<int>, unsigned int, abs_min_<float, int>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<int> *> const &, unsigned int, abs_min_<float, int>,
    identity);
  template void transform_reduce<minimum<float>, complex<short>, unsigned int, abs_min_<float, short>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<short> *> const &, unsigned int,
    abs_min_<float, short>, identity);
  template void
  transform_reduce<minimum<float>, complex<signed char>, unsigned int, abs_min_<float, signed char>, identity>(
    QudaFieldLocation, std::vector<float> &, std::vector<complex<signed char> *> const &, unsigned int,
    abs_min_<float, signed char>, identity);
  template double transform_reduce<minimum<double>, complex<double>, unsigned long, abs_min_<double, double>>(
    QudaFieldLocation, complex<double> const *, unsigned long, abs_min_<double, double>);
  template float transform_reduce<minimum<float>, complex<float>, unsigned long, abs_min_<float, float>>(
    QudaFieldLocation, complex<float> const *, unsigned long, abs_min_<float, float>);

  template void quda::transform_reduce<plus<double>, complex<double>, unsigned int, square_<double, double>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<double> *> const &, unsigned int,
    square_<double, double>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<float>, unsigned int, square_<double, float>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<float> *> const &, unsigned int,
    square_<double, float>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<int>, unsigned int, square_<double, int>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<int> *> const &, unsigned int,
    square_<double, int>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<short>, unsigned int, square_<double, short>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<short> *> const &, unsigned int,
    square_<double, short>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<int8_t>, unsigned int, square_<double, int8_t>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<int8_t> *> const &, unsigned int,
    square_<double, int8_t>, milc_mapper);

  template void quda::transform_reduce<plus<double>, complex<double>, unsigned int, abs_<double, double>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<double> *> const &, unsigned int,
    abs_<double, double>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<float>, unsigned int, abs_<double, float>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<float> *> const &, unsigned int,
    abs_<double, float>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<int>, unsigned int, abs_<double, int>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<int> *> const &, unsigned int,
    abs_<double, int>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<short>, unsigned int, abs_<double, short>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<short> *> const &, unsigned int,
    abs_<double, short>, milc_mapper);
  template void quda::transform_reduce<plus<double>, complex<int8_t>, unsigned int, abs_<double, int8_t>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<int8_t> *> const &, unsigned int,
    abs_<double, int8_t>, milc_mapper);

  template void
  quda::transform_reduce<maximum<double>, complex<double>, unsigned int, abs_max_<double, double>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<double> *> const &, unsigned int,
    abs_max_<double, double>, milc_mapper);
  template void quda::transform_reduce<maximum<float>, complex<float>, unsigned int, abs_max_<float, float>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<float> *> const &, unsigned int,
    abs_max_<float, float>, milc_mapper);
  template void quda::transform_reduce<maximum<float>, complex<int>, unsigned int, abs_max_<float, int>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<int> *> const &, unsigned int,
    abs_max_<float, int>, milc_mapper);
  template void quda::transform_reduce<maximum<float>, complex<short>, unsigned int, abs_max_<float, short>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<short> *> const &, unsigned int,
    abs_max_<float, short>, milc_mapper);
  template void quda::transform_reduce<maximum<float>, complex<int8_t>, unsigned int, abs_max_<float, int8_t>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<int8_t> *> const &, unsigned int,
    abs_max_<float, int8_t>, milc_mapper);

  template void
  quda::transform_reduce<minimum<double>, complex<double>, unsigned int, abs_min_<double, double>, milc_mapper>(
    QudaFieldLocation, std::vector<double> &, std::vector<quda::complex<double> *> const &, unsigned int,
    abs_min_<double, double>, milc_mapper);
  template void quda::transform_reduce<minimum<float>, complex<float>, unsigned int, abs_min_<float, float>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<float> *> const &, unsigned int,
    abs_min_<float, float>, milc_mapper);
  template void quda::transform_reduce<minimum<float>, complex<int>, unsigned int, abs_min_<float, int>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<int> *> const &, unsigned int,
    abs_min_<float, int>, milc_mapper);
  template void quda::transform_reduce<minimum<float>, complex<short>, unsigned int, abs_min_<float, short>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<short> *> const &, unsigned int,
    abs_min_<float, short>, milc_mapper);
  template void quda::transform_reduce<minimum<float>, complex<int8_t>, unsigned int, abs_min_<float, int8_t>, milc_mapper>(
    QudaFieldLocation, std::vector<float> &, std::vector<quda::complex<int8_t> *> const &, unsigned int,
    abs_min_<float, int8_t>, milc_mapper);

} // namespace quda

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <kernel.h>

namespace quda {

  using namespace gauge;

  /**
     Kernel argument struct
   */
  template <typename store_out_t, typename store_in_t, int length_, bool fine_grain_, typename OutOrder, typename InOrder>
  struct CopyGaugeArg : kernel_param<> {
    using real_out_t  = typename mapper<store_out_t>::type;
    using real_in_t  = typename mapper<store_in_t>::type;
    static constexpr int length = length_;
    static constexpr int nColor = Ncolor(length);
    static constexpr bool fine_grain = fine_grain_;
    OutOrder out;
    const InOrder in;
    int volume;
    int faceVolumeCB[QUDA_MAX_DIM];
    int_fastdiv nDim;
    int_fastdiv geometry;
    int out_offset;
    int in_offset;
    CopyGaugeArg(const OutOrder &out, const InOrder &in, const GaugeField &meta) :
      kernel_param(dim3(1, 1, meta.Geometry() * 2)), // FIXME - need to set .x and .y components
      out(out),
      in(in),
      volume(meta.Volume()),
      nDim(meta.Ndim()),
      geometry(meta.Geometry()),
      out_offset(0),
      in_offset(0)
    {
      for (int d=0; d<nDim; d++) faceVolumeCB[d] = meta.SurfaceCB(d) * meta.Nface();
    }
  };

  /**
     Check whether the field contains Nans
  */
  template <typename Arg>
  void checkNan(const Arg &arg)
  {
    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.geometry; d++) {
	for (int x=0; x<arg.volume/2; x++) {
          if constexpr (Arg::fine_grain) {
            for (int i=0; i<Arg::nColor; i++)
              for (int j=0; j<Arg::nColor; j++) {
                complex<typename Arg::real_in_t> u = arg.in(d, parity, x, i, j);
                if (std::isnan(u.real()))
                  errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, 2*(i*Ncolor(Arg::length)+j));
                if (std::isnan(u.imag()))
                  errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, 2*(i*Ncolor(Arg::length)+j+1));
              }
          } else {
            Matrix<complex<typename Arg::real_in_t>, Arg::nColor> u = arg.in(d, x, parity);
            for (int i=0; i<Arg::length/2; i++)
              if (std::isnan(u(i).real()) || std::isnan(u(i).imag())) errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, i);
          }
	}
      }

    }
  }

  /**
     @brief Generic gauge reordering and packing
  */
  template <typename Arg>
  struct CopyGauge_ {
    const Arg &arg;
    constexpr CopyGauge_(const Arg &arg) :arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    template <bool fine_grain> __device__ __host__ inline std::enable_if_t<!fine_grain, void> copy(int d, int parity, int x, int)
    {
      Matrix<complex<typename Arg::real_in_t>, Arg::nColor> in = arg.in(d, x, parity);
      Matrix<complex<typename Arg::real_out_t>, Arg::nColor> out = in;
      arg.out(d, x, parity) = out;
    }

    template <bool fine_grain> __device__ __host__ inline std::enable_if_t<fine_grain, void> copy(int d, int parity, int x, int i)
    {
      for (int j = 0; j < Arg::nColor; j++) arg.out(d, parity, x, i, j) = arg.in(d, parity, x, i, j);
    }

    __device__ __host__ inline void operator()(int x, int i, int parity_d)
    {
      int parity = parity_d / arg.geometry;
      int d = parity_d % arg.geometry;
      copy<Arg::fine_grain>(d, parity, x, i);
    }
  };

  /**
     @brief Generic gauge ghost reordering and packing
  */
  template <typename Arg>
  struct CopyGhost_ {
    const Arg &arg;
    constexpr CopyGhost_(const Arg &arg) :arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    template <bool fine_grain> __device__ __host__ inline std::enable_if_t<!fine_grain, void> copy(int d, int parity, int x, int)
    {
      Matrix<complex<typename Arg::real_in_t>, Arg::nColor> in = arg.in.Ghost(d + arg.in_offset, x, parity);
      Matrix<complex<typename Arg::real_out_t>, Arg::nColor> out = in;
      arg.out.Ghost(d + arg.out_offset, x, parity) = out;
    }

    template <bool fine_grain> __device__ __host__ inline std::enable_if_t<fine_grain, void> copy(int d, int parity, int x, int i)
    {
      for (int j = 0; j < Arg::nColor; j++) {
        arg.out.Ghost(d+arg.out_offset, parity, x, i, j) = arg.in.Ghost(d+arg.in_offset, parity, x, i, j);
      }
    }

    __device__ __host__ inline void operator()(int x, int i, int parity_d)
    {
      int parity = parity_d / arg.nDim;
      int d = parity_d % arg.nDim;
      if (x < arg.faceVolumeCB[d]) copy<Arg::fine_grain>(d, parity, x, i);
    }
  };

} // namespace quda

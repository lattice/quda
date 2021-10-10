#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <fast_intdiv.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda {

  using namespace gauge;

  /**
     Kernel argument struct
   */
  template <typename store_out_t, typename store_in_t, int length, typename OutOrder, typename InOrder, bool regularToextended_>
  struct CopyGaugeExArg : kernel_param<> {
    using real_out_t = typename mapper<store_out_t>::type;
    using real_in_t = typename mapper<store_in_t>::type;
    static constexpr int nColor = gauge::Ncolor(length);
    static constexpr bool regularToextended = regularToextended_;
    OutOrder out;
    const InOrder in;
    int_fastdiv Xin[QUDA_MAX_DIM];
    int_fastdiv Xout[QUDA_MAX_DIM];
    int border[QUDA_MAX_DIM];
    int geometry;
    CopyGaugeExArg(GaugeField &out, const GaugeField &in, store_out_t *Out, store_in_t *In) :
      kernel_param(dim3(in.VolumeCB() == out.VolumeCB() ? in.VolumeCB() : in.LocalVolumeCB(), 2, 1)),
      out(out, Out),
      in(in, In),
      geometry(in.Geometry())
    {
      for (int d=0; d < in.Ndim(); d++) {
	Xout[d] = out.X()[d];
	Xin[d] = in.X()[d];
        // if volumes match then both extended and just regular copy
        border[d] = Xout[d] == Xin[d] ? 0 : regularToextended ? out.R()[d] : in.R()[d];
        // sanity check that if the volumes match, we have the same border dimensions
        if (in.VolumeCB() == out.VolumeCB() && out.R()[d] != in.R()[d])
          errorQuda("Cannot copy extended fields of differing borders d = %d out = %d in = %d", d, out.R()[d], in.R()[d]);
      }
    }
  };

  /**
     Copy a regular/extended gauge field into an extended/regular gauge field
  */
  template <typename Arg> struct CopyGaugeEx_ {
    const Arg &arg;
    constexpr CopyGaugeEx_(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      int x[4];
      int xin, xout;
      getCoords(x, x_cb, Arg::regularToextended ? arg.Xin : arg.Xout, parity);
      for (int d = 0; d < 4; d++) x[d] += arg.border[d];
      if (Arg::regularToextended) { // regular to extended
        // Y is the cb spatial index into the extended gauge field
        xout = linkIndex(x, arg.Xout);
        xin = x_cb;
      } else { // extended to regular gauge
        // Y is the cb spatial index into the extended gauge field
        xin = linkIndex(x, arg.Xin);
        xout = x_cb;
      }
      for (int d=0; d<arg.geometry; d++) {
        const Matrix<complex<typename Arg::real_in_t>, Arg::nColor> in = arg.in(d, xin, parity);
        Matrix<complex<typename Arg::real_out_t>, Arg::nColor> out = in;
        arg.out(d, xout, parity) = out;
      }//dir
    }
  };

}

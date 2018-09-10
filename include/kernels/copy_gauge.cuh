#include <gauge_field_order.h>

namespace quda {

  using namespace gauge;

  /**
     Kernel argument struct
   */
  template <typename OutOrder, typename InOrder>
  struct CopyGaugeArg {
    OutOrder out;
    const InOrder in;
    int volume;
    int faceVolumeCB[QUDA_MAX_DIM];
    int_fastdiv nDim;
    int_fastdiv geometry;
    int out_offset;
    int in_offset;
    CopyGaugeArg(const OutOrder &out, const InOrder &in, const GaugeField &meta)
      : out(out), in(in), volume(meta.Volume()), nDim(meta.Ndim()),
        geometry(meta.Geometry()), out_offset(0), in_offset(0) {
      for (int d=0; d<nDim; d++) faceVolumeCB[d] = meta.SurfaceCB(d) * meta.Nface();
    }
  };

  /**
     Generic CPU gauge reordering and packing
  */
  template <typename FloatOut, typename FloatIn, int length, typename Arg>
  void copyGauge(Arg &arg) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.geometry; d++) {
	for (int x=0; x<arg.volume/2; x++) {
#ifdef FINE_GRAINED_ACCESS
	  for (int i=0; i<Ncolor(length); i++)
	    for (int j=0; j<Ncolor(length); j++) {
	      arg.out(d, parity, x, i, j) = arg.in(d, parity, x, i, j);
	    }
#else
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  arg.in.load(in, x, d, parity);
	  for (int i=0; i<length; i++) out[i] = in[i];
	  arg.out.save(out, x, d, parity);
#endif
	}
      }

    }
  }

  /**
     Check whether the field contains Nans
  */
  template <typename Float, int length, typename Arg>
  void checkNan(Arg &arg) {
    typedef typename mapper<Float>::type RegType;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.geometry; d++) {
	for (int x=0; x<arg.volume/2; x++) {
#ifdef FINE_GRAINED_ACCESS
	  for (int i=0; i<Ncolor(length); i++)
	    for (int j=0; j<Ncolor(length); j++) {
              complex<Float> u = arg.in(d, parity, x, i, j);
	      if (isnan(u.real()))
	        errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, 2*(i*Ncolor(length)+j));
	      if (isnan(u.imag()))
		errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, 2*(i*Ncolor(length)+j+1));
	}
#else
	  RegType u[length];
	  arg.in.load(u, x, d, parity);
	  for (int i=0; i<length; i++)
	    if (isnan(u[i]))
	      errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, i);
#endif
	}
      }

    }
  }

  /**
      Generic CUDA gauge reordering and packing.  Adopts a similar form as
      the CPU version, using the same inlined functions.
  */
  template <typename FloatOut, typename FloatIn, int length, typename Arg>
  __global__ void copyGaugeKernel(Arg arg) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int parity_d = blockIdx.z * blockDim.z + threadIdx.z; //parity_d = parity*geometry + d
    int parity = parity_d / arg.geometry;
    int d = parity_d % arg.geometry;

    if (x >= arg.volume/2) return;
    if (parity_d >= 2 * arg.geometry) return;

#ifdef FINE_GRAINED_ACCESS
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Ncolor(length)) return;
    for (int j=0; j<Ncolor(length); j++) arg.out(d, parity, x, i, j) = arg.in(d, parity, x, i, j);
#else
    RegTypeIn in[length];
    RegTypeOut out[length];
    arg.in.load(in, x, d, parity);
    for (int i=0; i<length; i++) out[i] = in[i];
    arg.out.save(out, x, d, parity);
#endif
  }

  /**
     Generic CPU gauge ghost reordering and packing
  */
  template <typename FloatOut, typename FloatIn, int length, typename Arg>
  void copyGhost(Arg &arg) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.nDim; d++) {
        for (int x=0; x<arg.faceVolumeCB[d]; x++) {
#ifdef FINE_GRAINED_ACCESS
          for (int i=0; i<Ncolor(length); i++)
            for (int j=0; j<Ncolor(length); j++)
              arg.out.Ghost(d+arg.out_offset, parity, x, i, j) = arg.in.Ghost(d+arg.in_offset, parity, x, i, j);
#else
          RegTypeIn in[length];
          RegTypeOut out[length];
          arg.in.loadGhost(in, x, d+arg.in_offset, parity); // assumes we are loading
          for (int i=0; i<length; i++) out[i] = in[i];
          arg.out.saveGhost(out, x, d+arg.out_offset, parity);
#endif
        }
      }

    }
  }

  /**
     Generic CUDA kernel for copying the ghost zone.  Adopts a similar form as
     the CPU version, using the same inlined functions.
  */
  template <typename FloatOut, typename FloatIn, int length, typename Arg>
  __global__ void copyGhostKernel(Arg arg) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int parity_d = blockIdx.z * blockDim.z + threadIdx.z; //parity_d = parity*nDim + d
    int parity = parity_d / arg.nDim;
    int d = parity_d % arg.nDim;
    if (parity_d >= 2 * arg.nDim) return;

    if (x < arg.faceVolumeCB[d]) {
#ifdef FINE_GRAINED_ACCESS
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Ncolor(length)) return;
    for (int j=0; j<Ncolor(length); j++)
      arg.out.Ghost(d+arg.out_offset, parity, x, i, j) = arg.in.Ghost(d+arg.in_offset, parity, x, i, j);
#else
      RegTypeIn in[length];
      RegTypeOut out[length];
      arg.in.loadGhost(in, x, d+arg.in_offset, parity); // assumes we are loading
      for (int i=0; i<length; i++) out[i] = in[i];
	arg.out.saveGhost(out, x, d+arg.out_offset, parity);
#endif
    }

  }

} // namespace quda

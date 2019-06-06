#ifndef USE_LEGACY_DSLASH

#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>
#include <uint_to_char.h>

#include <dslash_policy.cuh>
#include <kernels/covDev.cuh>

/**
   This is the covariant derivative based on the basic gauged Laplace operator
*/

namespace quda
{

#ifdef GPU_COVDEV

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct CovDevLaunch {

    // kernel name for jit compilation
    static constexpr const char *kernel = "quda::covDevGPU";

    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      static_assert(xpay == false, "Covariant derivative operator only defined without xpay");
      static_assert(nParity == 2, "Covariant derivative operator only defined for full field");
      dslash.launch(covDevGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class CovDev : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    CovDev(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash<Float>(arg, out, in, "kernels/covDev.cuh"),
      arg(arg),
      in(in)
    {
    }

    virtual ~CovDev() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.xpay) errorQuda("Covariant derivative operator only defined without xpay");
      if (arg.nParity != 2) errorQuda("Covariant derivative operator only defined for full field");

      constexpr bool xpay = false;
      constexpr int nParity = 2;
      Dslash<Float>::template instantiate<CovDevLaunch, nDim, nColor, nParity, xpay>(tp, arg, stream);
    }

    long long flops() const
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int num_mv_multiply = in.Nspin();
      int ghost_flops = num_mv_multiply * mv_flops;
      int dim = arg.mu % 4;
      long long flops_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        if (arg.kernel_type != dim) break;
        flops_ = (ghost_flops)*in.GhostFace()[dim];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = in.GhostFace()[dim];
        flops_ = ghost_flops * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        flops_ = num_mv_multiply * mv_flops * sites; // SU(3) matrix-vector multiplies

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = arg.commDim[dim] ? in.GhostFace()[dim] : 0;
        flops_ -= ghost_flops * ghost_sites;

        break;
      }
      }

      return flops_;
    }

    long long bytes() const
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      int ghost_bytes = gauge_bytes + 3 * spinor_bytes; // 3 since we have to load the partial
      int dim = arg.mu % 4;
      long long bytes_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        if (arg.kernel_type != dim) break;
        bytes_ = ghost_bytes * in.GhostFace()[dim];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = in.GhostFace()[dim];
        bytes_ = ghost_bytes * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        bytes_ = (gauge_bytes + 2 * spinor_bytes) * sites;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = arg.commDim[dim] ? in.GhostFace()[dim] : 0;
        bytes_ -= ghost_bytes * ghost_sites;

        break;
      }
      }
      return bytes_;
    }

    TuneKey tuneKey() const
    {
      // add mu to the key
      char aux[TuneKey::aux_n];
      strcpy(aux, Dslash<Float>::aux[arg.kernel_type]);
      strcat(aux, ",mu=");
      char mu[8];
      u32toa(mu, arg.mu);
      strcat(aux, mu);
      return TuneKey(in.VolString(), typeid(*this).name(), aux);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct CovDevApply {

    inline CovDevApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int mu, int parity,
                       bool dagger, const int *comm_override, TimeProfile &profile)

    {
      constexpr int nDim = 4;
      CovDevArg<Float, nColor, recon> arg(out, in, U, mu, parity, dagger, comm_override);
      CovDev<Float, nDim, nColor, CovDevArg<Float, nColor, recon>> covDev(arg, out, in);

      dslash::DslashPolicyTune<decltype(covDev)> policy(
        covDev, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
        in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

#endif

  // Apply the covariant derivative operator
  // out(x) = U_{\mu}(x)in(x+mu) for mu = 0...3
  // out(x) = U^\dagger_mu'(x-mu')in(x-mu') for mu = 4...7 and we set mu' = mu-4
  void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int mu, int parity,
                   bool dagger, const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_COVDEV
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    pushKernelPackT(true); // non-spin projection requires kernel packing

    instantiate<CovDevApply>(out, in, U, mu, parity, dagger, comm_override, profile);

    popKernelPackT();
#else
    errorQuda("Covariant derivative kernels have not been built");
#endif
  }
} // namespace quda

// END #ifndef USE_LEGACY_DSLASH
#else

// BEGIN USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <tune_quda.h>
#include <worker.h>

/**
   This is the covariant derivative based on the basic gauged Laplace operator
*/

namespace quda {

#ifdef GPU_COVDEV

  /**
     @brief Parameter structure for driving the covariant derivative
   */

  template <typename Float, int nSpin, int nColor, QudaReconstructType reconstruct> struct CovDevArg {
    typedef typename colorspinor_mapper<Float, nSpin, nColor>::type F;
    typedef typename gauge_mapper<Float, reconstruct>::type G;

    F out;                // output vector field
    const F in;           // input vector field
    const G U;            // the gauge field
    const int parity;     // only use this for single parity fields
    const int nParity;    // number of parities we're working on
    const int nFace;      // hard code to 1 for now
    const int dim[5];     // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;   // checkerboarded volume
    const int mu;         // direction of the covariant derivative

    CovDevArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const int parity, const int mu) :
      out(out),
      in(in),
      U(U),
      parity(parity),
      mu(mu),
      nParity(in.SiteSubset()),
      nFace(1),
      dim {(3 - nParity) * in.X(0), in.X(1), in.X(2), in.X(3), 1},
      commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(in.VolumeCB())
    {
      if (!U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
    }
  };

  /**
     Applies the off-diagonal part of the Laplace operator
     @param[out] out The out result field
     @param[in] U The gauge field
     @param[in] kappa Kappa value
     @param[in] in The input field
     @param[in] parity The site parity
     @param[in] x_cb The checkerboarded site index
  */
  template <typename Float, int nDim, int nColor, int mu, typename Vector, typename Arg>
  __device__ __host__ inline void applyCovDev(Vector &out, Arg &arg, int x_cb, int parity) {
    typedef Matrix<complex<Float>,nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;

    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
    coord[4] = 0;

    const int d = mu%4;

    if (mu < 4) {
      //Forward gather - compute fwd offset for vector fetch
      const int fwd_idx = linkIndexP1(coord, arg.dim, d);

      if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

	const Link U = arg.U(d, x_cb, parity);
	const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);

	out += U * in;
      } else {

	const Link U = arg.U(d, x_cb, parity);
	const Vector in = arg.in(fwd_idx, their_spinor_parity);

	out += U * in;
      }
    } else {
      //Backward gather - compute back offset for spinor and gauge fetch
      const int back_idx = linkIndexM1(coord, arg.dim, d);
      const int gauge_idx = back_idx;

      if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

	const Link U = arg.U.Ghost(d, ghost_idx, 1-parity);
	const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);

	out += conj(U) * in;
      } else {

        const Link U = arg.U(d, gauge_idx, 1-parity);
	const Vector in = arg.in(back_idx, their_spinor_parity);

	out += conj(U) * in;
      }
    } // Forward/backward derivative

  }


  //out(x) = M*in
  template <typename Float, int nDim, int nSpin, int nColor, typename Arg>
  __device__ __host__ inline void covDev(Arg &arg, int x_cb, int parity)
  {
    typedef ColorSpinor<Float,nColor,nSpin> Vector;
    Vector out;

    switch (arg.mu) {
    case 0: applyCovDev<Float,nDim,nColor,0>(out, arg, x_cb, parity); break;
    case 1: applyCovDev<Float,nDim,nColor,1>(out, arg, x_cb, parity); break;
    case 2: applyCovDev<Float,nDim,nColor,2>(out, arg, x_cb, parity); break;
    case 3: applyCovDev<Float,nDim,nColor,3>(out, arg, x_cb, parity); break;
    case 4: applyCovDev<Float,nDim,nColor,4>(out, arg, x_cb, parity); break;
    case 5: applyCovDev<Float,nDim,nColor,5>(out, arg, x_cb, parity); break;
    case 6: applyCovDev<Float,nDim,nColor,6>(out, arg, x_cb, parity); break;
    case 7: applyCovDev<Float,nDim,nColor,7>(out, arg, x_cb, parity); break;
    }
    arg.out(x_cb, parity) = out;
  }

  // CPU kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nSpin, int nColor, typename Arg>
  void covDevCPU(Arg arg)
  {

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	covDev<Float,nDim,nSpin,nColor>(arg, x_cb, parity);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nSpin, int nColor, typename Arg>
  __global__ void covDevGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;

    // for full fields set parity from y thread index else use arg setting
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;

    covDev<Float,nDim,nSpin,nColor>(arg, x_cb, parity);
  }

  template <typename Float, int nDim, int nSpin, int nColor, typename Arg>
  class CovDev : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const
    {
      return 8*nColor*nColor*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const
    {
      return arg.out.Bytes() + arg.in.Bytes() + arg.nParity*arg.U.Bytes()*meta.VolumeCB();
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    CovDev(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
#ifdef MULTI_GPU
      char comm[5];
      comm[0] = (arg.commDim[0] ? '1' : '0');
      comm[1] = (arg.commDim[1] ? '1' : '0');
      comm[2] = (arg.commDim[2] ? '1' : '0');
      comm[3] = (arg.commDim[3] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux,",comm=");
      strcat(aux,comm);
#endif
    }
    virtual ~CovDev() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	covDevCPU<Float,nDim,nSpin,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	covDevGPU<Float,nDim,nSpin,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };


  template <typename Float, int nColor, QudaReconstructType recon>
    void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int parity, int mu)
  {
    constexpr int nDim = 4;
    if (in.Nspin() == 1) {
      constexpr int nSpin = 1;
      CovDevArg<Float,nSpin,nColor,recon> arg(out, in, U, parity, mu);
      CovDev<Float, nDim, nSpin, nColor, CovDevArg<Float, nSpin, nColor, recon>> myCovDev(arg, in);
      myCovDev.apply(0);
    } else if (in.Nspin() == 4) {
      constexpr int nSpin = 4;
      CovDevArg<Float,nSpin,nColor,recon> arg(out, in, U, parity, mu);
      CovDev<Float, nDim, nSpin, nColor, CovDevArg<Float, nSpin, nColor, recon>> myCovDev(arg, in);
      myCovDev.apply(0);
    } else {
      errorQuda("Unsupported nSpin=%d", in.Nspin());
    }
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
    void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int parity, int mu)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyCovDev<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, parity, mu);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyCovDev<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, parity, mu);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyCovDev<Float,nColor,QUDA_RECONSTRUCT_8> (out, in, U, parity, mu);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
    void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int parity, int mu)
  {
    if (in.Ncolor() == 3) {
      ApplyCovDev<Float,3>(out, in, U, parity, mu);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  // this is the Worker pointer that may have issue additional work
  // while we're waiting on communication to finish
  namespace dslash {
    extern Worker* aux_worker;
  }

#endif // GPU_COVDEV

  //Apply the covariant derivative operator
  //out(x) = U_{\mu}(x)in(x+mu) for mu = 0...3
  //out(x) = U^\dagger_mu'(x-mu')in(x-mu') for mu = 4...7 and we set mu' = mu-4
  void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int parity, int mu)
  {
#ifdef GPU_COVDEV
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precision match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    const int nFace = 1;
    in.exchangeGhost((QudaParity)(1-parity), nFace, 0); // last parameter is dummy

    if (dslash::aux_worker) dslash::aux_worker->apply(0);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyCovDev<double>(out, in, U, parity, mu);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCovDev<float>(out, in, U, parity, mu);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }

    in.bufferIndex = (1 - in.bufferIndex);
#else
    errorQuda("Covariant derivative kernels have not been built");
#endif
  }

} // namespace quda

#endif // END USE_LEGACY_DSLASH

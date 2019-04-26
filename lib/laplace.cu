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
#include <kernels/laplace.cuh>

/**
   This is the laplacian derivative based on the basic gauged differential operator
*/

namespace quda
{

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct LaplaceLaunch {

    // kernel name for jit compilation
    static constexpr const char *kernel = "quda::laplaceGPU";

    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      dslash.launch(laplaceGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class Laplace : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    Laplace(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash<Float>(arg, out, in, "kernels/laplace.cuh"),
      arg(arg),
      in(in)
    {
    }

    virtual ~Laplace() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      Dslash<Float>::template instantiate<LaplaceLaunch, nDim, nColor>(tp, arg, stream);
    }

    long long flops() const
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int num_mv_multiply = in.Nspin() == 4 ? 2 : 1;
      int ghost_flops = (num_mv_multiply * mv_flops + 2 * in.Ncolor() * in.Nspin());
      int xpay_flops = 2 * 2 * in.Ncolor() * in.Nspin(); // multiply and add per real component
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);      // 3D or 4D operator

      long long flops_ = 0;

      // FIXME - should we count the xpay flops in the derived kernels
      // since some kernels require the xpay in the exterior (preconditiond clover)

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * 2 * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        flops_ = (num_dir * (in.Nspin() / 4) * in.Ncolor() * in.Nspin() + // spin project (=0 for staggered)
                  num_dir * num_mv_multiply * mv_flops +                  // SU(3) matrix-vector multiplies
                  ((num_dir - 1) * 2 * in.Ncolor() * in.Nspin()))
          * sites; // accumulation
        if (arg.xpay) flops_ += xpay_flops * sites;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        flops_ -= ghost_flops * ghost_sites;

        break;
      }
      }

      return flops_;
    }

    virtual long long bytes() const
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      int proj_spinor_bytes = in.Nspin() == 4 ? spinor_bytes / 2 : spinor_bytes;
      int ghost_bytes = (proj_spinor_bytes + gauge_bytes) + 2 * spinor_bytes; // 2 since we have to load the partial
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);                           // 3D or 4D operator

      long long bytes_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: bytes_ = ghost_bytes * 2 * in.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        bytes_ = ghost_bytes * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        bytes_ = (num_dir * gauge_bytes + ((num_dir - 2) * spinor_bytes + 2 * proj_spinor_bytes) + spinor_bytes) * sites;
        if (arg.xpay) bytes_ += spinor_bytes;
	
        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        bytes_ -= ghost_bytes * ghost_sites;
	
        break;
      }
      }
      return bytes_;
    }
    
    TuneKey tuneKey() const
    {
      // add laplace transverse dir to the key
      char aux[TuneKey::aux_n];
      strcpy(aux, Dslash<Float>::aux[arg.kernel_type]);
      strcat(aux, ",laplace3D=");
      char laplace3D[32];
      u32toa(laplace3D, arg.dir);
      strcat(aux, laplace3D);
      return TuneKey(in.VolString(), typeid(*this).name(), aux);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct LaplaceApply {

    inline LaplaceApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir, double a,
                        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                        TimeProfile &profile)
    {

      constexpr int nDim = 4;
      LaplaceArg<Float, nColor, recon> arg(out, in, U, dir, a, x, parity, dagger, comm_override);
      Laplace<Float, nDim, nColor, LaplaceArg<Float, nColor, recon>> laplace(arg, out, in);

      dslash::DslashPolicyTune<decltype(laplace)> policy(
        laplace, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
        in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the Laplace operator
  // out(x) = M*in = - kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator.
  // Omits direction 'dir' from the operator.
  void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir, double kappa,
                    const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {

    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    instantiate<LaplaceApply>(out, in, U, dir, kappa, x, parity, dagger, comm_override, profile);
  }
} // namespace quda

#else

#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <color_spinor.h>
#include <worker.h>
#include <tune_quda.h>

/**
   This is a basic gauged Laplace operator
*/

namespace quda {

  /**
     @brief Parameter structure for driving the Laplace operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct, bool xpay>
  struct LaplaceArg {
    typedef typename colorspinor_mapper<Float,1,nColor>::type F;
    typedef typename gauge_mapper<Float,reconstruct>::type G;

    F out;                // output vector field
    const F in;           // input vector field
    const F x;            // input vector when doing xpay
    const G U;            // the gauge field
    const Float kappa;    // kappa parameter = 1/(8+m)
    const int parity;     // only use this for single parity fields
    const int nParity;    // number of parities we're working on
    const int nFace;      // hard code to 1 for now
    const int dim[5];     // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;   // checkerboarded volume

    __host__ __device__ static constexpr bool isXpay() { return xpay; }

    LaplaceArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
	       Float kappa, const ColorSpinorField *x, int parity)
      : out(out), in(in), U(U), kappa(kappa), x(xpay ? *x : in), parity(parity), nParity(in.SiteSubset()), nFace(1),
	dim{ (3-nParity) * in.X(0), in.X(1), in.X(2), in.X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(in.VolumeCB())
    {
      if (in.FieldOrder() != QUDA_FLOAT2_FIELD_ORDER || !U.isNative())
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
  template <typename Float, int nDim, int nColor, typename Vector, typename Arg>
  __device__ __host__ inline void applyLaplace(Vector &out, Arg &arg, int x_cb, int parity) {
    typedef Matrix<complex<Float>,nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;

    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
    coord[4] = 0;

#pragma unroll
    for (int d = 0; d<nDim; d++) // loop over dimension
    {
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
    } //nDim
  }


  //out(x) = M*in = (-D + m) * in(x-mu)
  template <typename Float, int nDim, int nColor, typename Arg>
  __device__ __host__ inline void laplace(Arg &arg, int x_cb, int parity)
  {
    typedef ColorSpinor<Float,nColor,1> Vector;
    Vector out;

    applyLaplace<Float,nDim,nColor>(out, arg, x_cb, parity);

    if (arg.isXpay()) {
      Vector x = arg.x(x_cb, parity);
      out = x + arg.kappa * out;
    }
    arg.out(x_cb, arg.nParity == 2 ? parity : 0) = out;
  }

  // CPU kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nColor, typename Arg>
  void laplaceCPU(Arg arg)
  {

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	laplace<Float,nDim,nColor>(arg, x_cb, parity);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nColor, typename Arg>
  __global__ void laplaceGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;

    // for full fields set parity from y thread index else use arg setting
    int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;

    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;

    laplace<Float,nDim,nColor>(arg, x_cb, parity);
  }

  template <typename Float, int nDim, int nColor, typename Arg>
  class Laplace : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const
    {
      return (2*nDim*(8*nColor*nColor)-2*nColor + (arg.isXpay() ? 2*2*nColor : 0) )*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const
    {
      return arg.out.Bytes() + 2*nDim*arg.in.Bytes() + arg.nParity*2*nDim*arg.U.Bytes()*meta.VolumeCB() +
	(arg.isXpay() ? arg.x.Bytes() : 0);
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    Laplace(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
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
      if (arg.isXpay()) strcat(aux,",xpay");
    }
    virtual ~Laplace() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	laplaceCPU<Float,nDim,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	laplaceGPU<Float,nDim,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };


  template <typename Float, int nColor, QudaReconstructType recon>
    void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		      double kappa, const ColorSpinorField *x, int parity)
  {
    constexpr int nDim = 4;
    if (x) {
      LaplaceArg<Float,nColor,recon,true> arg(out, in, U, kappa, x, parity);
      Laplace<Float,nDim,nColor,LaplaceArg<Float,nColor,recon,true> > laplace(arg, in);
      laplace.apply(0);
    } else {
      LaplaceArg<Float,nColor,recon,false> arg(out, in, U, kappa, x, parity);
      Laplace<Float,nDim,nColor,LaplaceArg<Float,nColor,recon,false> > laplace(arg, in);
      laplace.apply(0);
    }
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
    void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		      double kappa, const ColorSpinorField *x, int parity)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyLaplace<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, kappa, x, parity);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyLaplace<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, kappa, x, parity);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyLaplace<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, kappa, x, parity);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
    void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		      double kappa, const ColorSpinorField *x, int parity)
  {
    if (in.Ncolor() == 3) {
      ApplyLaplace<Float,3>(out, in, U, kappa, x, parity);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  // this is the Worker pointer that may have issue additional work
  // while we're waiting on communication to finish
  namespace dslash {
    extern Worker* aux_worker;
  }

  //Apply the Laplace operator
  //out(x) = M*in = - kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		    double kappa, const ColorSpinorField *x, int parity)		    
  {
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    const int nFace = 1;
    in.exchangeGhost((QudaParity)(1-parity), nFace, 0); // last parameter is dummy

    if (dslash::aux_worker) dslash::aux_worker->apply(0);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyLaplace<double>(out, in, U, kappa, x, parity);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyLaplace<float>(out, in, U, kappa, x, parity);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }

    in.bufferIndex = (1 - in.bufferIndex);
  }
} // namespace quda

#endif

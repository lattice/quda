#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <multigrid_helper.cuh>

namespace quda {

#ifdef GPU_MULTIGRID
  using namespace quda::colorspinor;
  
  /** 
      Kernel argument struct
  */
  template <typename Float, typename vFloat, int fineSpin, int fineColor, int coarseSpin, int coarseColor, QudaFieldOrder order>
  struct ProlongateArg {
    FieldOrderCB<Float,fineSpin,fineColor,1,order> out;
    const FieldOrderCB<Float,coarseSpin,coarseColor,1,order> in;
    const FieldOrderCB<Float,fineSpin,fineColor,coarseColor,order,vFloat> V;
    const int *geo_map;  // need to make a device copy of this
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the output field (if single parity)
    const int nParity; // number of parities of input fine field

    ProlongateArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &V,
                  const int *geo_map,  const int parity)
      : out(out), in(in), V(V), geo_map(geo_map), spin_map(), parity(parity), nParity(out.SiteSubset()) { }

    ProlongateArg(const ProlongateArg<Float,vFloat,fineSpin,fineColor,coarseSpin,coarseColor,order> &arg)
      : out(arg.out), in(arg.in), V(arg.V), geo_map(arg.geo_map), spin_map(),
        parity(arg.parity), nParity(arg.nParity) { }
  };

  /**
     Applies the grid prolongation operator (coarse to fine)
  */
  template <typename Float, int fineSpin, int coarseColor, class Coarse, typename S>
  __device__ __host__ inline void prolongate(complex<Float> out[fineSpin*coarseColor], const Coarse &in, 
                                             int parity, int x_cb, const int *geo_map, const S& spin_map, int fineVolumeCB) {
    int x = parity*fineVolumeCB + x_cb;
    int x_coarse = geo_map[x];
    int parity_coarse = (x_coarse >= in.VolumeCB()) ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*in.VolumeCB();

#pragma unroll
    for (int s=0; s<fineSpin; s++) {
#pragma unroll
      for (int c=0; c<coarseColor; c++) {
        out[s*coarseColor+c] = in(parity_coarse, x_coarse_cb, spin_map(s,parity), c);
      }
    }
  }

  /**
     Rotates from the coarse-color basis into the fine-color basis.  This
     is the second step of applying the prolongator.
  */
  template <typename Float, int fineSpin, int fineColor, int coarseColor, int fine_colors_per_thread,
            class FineColor, class Rotator>
  __device__ __host__ inline void rotateFineColor(FineColor &out, const complex<Float> in[fineSpin*coarseColor],
                                                  const Rotator &V, int parity, int nParity, int x_cb, int fine_color_block) {
    const int spinor_parity = (nParity == 2) ? parity : 0;
    const int v_parity = (V.Nparity() == 2) ? parity : 0;

    constexpr int color_unroll = 2;

#pragma unroll
    for (int s=0; s<fineSpin; s++)
#pragma unroll
      for (int fine_color_local=0; fine_color_local<fine_colors_per_thread; fine_color_local++)
        out(spinor_parity, x_cb, s, fine_color_block+fine_color_local) = 0.0; // global fine color index
    
#pragma unroll
    for (int s=0; s<fineSpin; s++) {
#pragma unroll
      for (int fine_color_local=0; fine_color_local<fine_colors_per_thread; fine_color_local++) {
        int i = fine_color_block + fine_color_local; // global fine color index

        complex<Float> partial[color_unroll];
#pragma unroll
        for (int k=0; k<color_unroll; k++) partial[k] = 0.0;

#pragma unroll
        for (int j=0; j<coarseColor; j+=color_unroll) {
          // V is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
#pragma unroll
          for (int k=0; k<color_unroll; k++)
            partial[k] += V(v_parity, x_cb, s, i, j+k) * in[s*coarseColor + j + k];
        }

#pragma unroll
        for (int k=0; k<color_unroll; k++) out(spinor_parity, x_cb, s, i) += partial[k];
      }
    }

  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int fine_colors_per_thread, typename Arg>
  void Prolongate(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb=0; x_cb<arg.out.VolumeCB(); x_cb++) {
        complex<Float> tmp[fineSpin*coarseColor];
        prolongate<Float,fineSpin,coarseColor>(tmp, arg.in, parity, x_cb, arg.geo_map, arg.spin_map, arg.out.VolumeCB());
        for (int fine_color_block=0; fine_color_block<fineColor; fine_color_block+=fine_colors_per_thread) {
          rotateFineColor<Float,fineSpin,fineColor,coarseColor,fine_colors_per_thread>
            (arg.out, tmp, arg.V, parity, arg.nParity, x_cb, fine_color_block);
        }
      }
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int fine_colors_per_thread, typename Arg>
  __global__ void ProlongateKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = arg.nParity == 2 ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
    if (x_cb >= arg.out.VolumeCB()) return;

    int fine_color_block = (blockDim.z*blockIdx.z + threadIdx.z) * fine_colors_per_thread;
    if (fine_color_block >= fineColor) return;

    complex<Float> tmp[fineSpin*coarseColor];
    prolongate<Float,fineSpin,coarseColor>(tmp, arg.in, parity, x_cb, arg.geo_map, arg.spin_map, arg.out.VolumeCB());
    rotateFineColor<Float,fineSpin,fineColor,coarseColor,fine_colors_per_thread>
      (arg.out, tmp, arg.V, parity, arg.nParity, x_cb, fine_color_block);
  }
  
  template <typename Float, typename vFloat, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int fine_colors_per_thread>
  class ProlongateLaunch : public TunableVectorYZ {

  protected:
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &V;
    const int *fine_to_coarse;
    int parity;
    QudaFieldLocation location;
    char vol[TuneKey::volume_n];

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return out.VolumeCB(); } // fine parity is the block y dimension

  public:
    ProlongateLaunch(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &V,
                     const int *fine_to_coarse, int parity)
      : TunableVectorYZ(out.SiteSubset(), fineColor/fine_colors_per_thread), out(out), in(in), V(V),
        fine_to_coarse(fine_to_coarse), parity(parity), location(checkLocation(out, in, V))
    {
      strcpy(vol, out.VolString());
      strcat(vol, ",");
      strcat(vol, in.VolString());

      strcpy(aux, out.AuxString());
      strcat(aux, ",");
      strcat(aux, in.AuxString());
    }

    virtual ~ProlongateLaunch() { }

    void apply(const qudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          ProlongateArg<Float,vFloat,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
            arg(out, in, V, fine_to_coarse, parity);
          Prolongate<Float,fineSpin,fineColor,coarseSpin,coarseColor,fine_colors_per_thread>(arg);
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      } else {
        if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          ProlongateArg<Float,vFloat,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_FLOAT2_FIELD_ORDER>
            arg(out, in, V, fine_to_coarse, parity);
          ProlongateKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,fine_colors_per_thread>
            <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      }
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }

    long long flops() const { return 8 * fineSpin * fineColor * coarseColor * out.SiteSubset()*(long long)out.VolumeCB(); }

    long long bytes() const {
      size_t v_bytes = V.Bytes() / (V.SiteSubset() == out.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + v_bytes + out.SiteSubset()*out.VolumeCB()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, int parity) {

    // for all grids use 1 color per thread
    constexpr int fine_colors_per_thread = 1;

    if (v.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      ProlongateLaunch<Float, short, fineSpin, fineColor, coarseSpin, coarseColor, fine_colors_per_thread>
      prolongator(out, in, v, fine_to_coarse, parity);
      prolongator.apply(0);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (v.Precision() == in.Precision()) {
      ProlongateLaunch<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor, fine_colors_per_thread>
      prolongator(out, in, v, fine_to_coarse, parity);
      prolongator.apply(0);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }

    if (checkLocation(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }


  template <typename Float, int fineSpin>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int nVec, const int *fine_to_coarse, const int * const * spin_map, int parity) {

    if (in.Nspin() != 2) errorQuda("Coarse spin %d is not supported", in.Nspin());
    const int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++) 
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (out.Ncolor() == 3) {
      const int fineColor = 3;
#ifdef NSPIN4
      if (nVec == 6) { // Free field Wilson
        Prolongate<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, parity);
      } else
#endif // NSPIN4
      if (nVec == 24) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, parity);
#ifdef NSPIN4
      } else if (nVec == 32) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, parity);
#endif // NSPIN4
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#ifdef NSPIN4
    } else if (out.Ncolor() == 6) { // for coarsening coarsened Wilson free field.
      const int fineColor = 6;
      if (nVec == 6) { // these are probably only for debugging only
        Prolongate<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#endif // NSPIN4
    } else if (out.Ncolor() == 24) {
      const int fineColor = 24;
      if (nVec == 24) { // to keep compilation under control coarse grids have same or more colors
        Prolongate<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, parity);
#ifdef NSPIN4
      } else if (nVec == 32) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, parity);
#endif // NSPIN4
#ifdef NSPIN1
      } else if (nVec == 64) { 
        Prolongate<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, parity);
      } else if (nVec == 96) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, parity);
#endif // NSPIN1
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#ifdef NSPIN4
    } else if (out.Ncolor() == 32) {
      const int fineColor = 32;
      if (nVec == 32) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#endif // NSPIN4
#ifdef NSPIN1
    } else if (out.Ncolor() == 64) {
      const int fineColor = 64;
      if (nVec == 64) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, parity);
      } else if (nVec == 96) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (out.Ncolor() == 96) {
      const int fineColor = 96;
      if (nVec == 96) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#endif // NSPIN1
    } else {
      errorQuda("Unsupported nColor %d", out.Ncolor());
    }
  }

  template <typename Float>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int Nvec, const int *fine_to_coarse, const int * const * spin_map, int parity) {

    if (out.Nspin() == 2) {
      Prolongate<Float,2>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#ifdef NSPIN4
    } else if (out.Nspin() == 4) {
      Prolongate<Float,4>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#endif
#if 0 // Not needed until we have Laplace MG or staggered MG Lanczos
//#ifdef NSPIN1
    } else if (out.Nspin() == 1) {
      Prolongate<Float,1>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#endif
    } else {
      errorQuda("Unsupported nSpin %d", out.Nspin());
    }
  }

#endif // GPU_MULTIGRID

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int Nvec, const int *fine_to_coarse, const int * const * spin_map, int parity) {
#ifdef GPU_MULTIGRID
    if (out.FieldOrder() != in.FieldOrder() || out.FieldOrder() != v.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d, v=%d)", 
                out.FieldOrder(), in.FieldOrder(), v.FieldOrder());

    QudaPrecision precision = checkPrecision(out, in);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      Prolongate<double>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      Prolongate<float>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }

    if (checkLocation(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
#else
    errorQuda("Multigrid has not been built");
#endif
  }

} // end namespace quda

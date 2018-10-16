#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <typeinfo>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>

namespace quda {

#ifdef GPU_MULTIGRID

  using namespace quda::colorspinor;
  
  /** 
      Kernel argument struct
  */
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, QudaFieldOrder order>
  struct StaggeredProlongateArg {
    FieldOrderCB<Float,fineSpin,fineColor,1,order> out;
    const FieldOrderCB<Float,coarseSpin,coarseColor,1,order> in;
    const int *geo_map;  // need to make a device copy of this
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the output field (if single parity)
    const int nParity; // number of parities of input fine field
    const int X[4]; // spatial volume

    StaggeredProlongateArg(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *geo_map, const int parity)
      : out(out), in(in), geo_map(geo_map), spin_map(), parity(parity),
        nParity(out.SiteSubset()), X{out.X()[0],out.X()[1],out.X()[2],out.X()[3]}
    {;}

    StaggeredProlongateArg(const StaggeredProlongateArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,order> &arg)
      : out(arg.out), in(arg.in), geo_map(arg.geo_map), spin_map(),
        parity(arg.parity), nParity(arg.nParity), X{arg.X[0],arg.X[1],arg.X[2],arg.X[3]}
    {;}
  };

  /**
     Performs the permutation from a coarse degree of freedom to a 
     fine degree of freedom
  */
  template <class FineColor, class Coarse, typename S>
  __device__ __host__ inline void staggeredProlongate(FineColor& out, const Coarse &in, 
                                            int parity, int x_cb, int c, const int *geo_map, const S& spin_map, int fineVolumeCB, const int X[]) {
    int x = parity*fineVolumeCB + x_cb;
    int x_coarse = geo_map[x];
    int parity_coarse = (x_coarse >= in.VolumeCB()) ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*in.VolumeCB();

    // coarse_color = 8*fine_color + corner of the hypercube
    int fineCoords[4];
    getCoords(fineCoords,x_cb,X,parity);
    int hyperCorner = 4*(fineCoords[3]%2)+2*(fineCoords[2]%2)+(fineCoords[1]%2);

    out(parity,x_cb,0,c) = in(parity_coarse, x_coarse_cb, spin_map(0,parity), 8*c+hyperCorner);
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void StaggeredProlongate(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb=0; x_cb<arg.out.VolumeCB(); x_cb++) {
        for (int c=0; c<fineColor; c++) {
          staggeredProlongate(arg.out, arg.in, parity, x_cb, c, arg.geo_map, arg.spin_map, arg.out.VolumeCB(), arg.X);
        }
      }
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void StaggeredProlongateKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = arg.nParity == 2 ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
    if (x_cb >= arg.out.VolumeCB()) return;

    int c = blockDim.z*blockIdx.z + threadIdx.z;
    if (c >= fineColor) return;

    staggeredProlongate(arg.out, arg.in, parity, x_cb, c, arg.geo_map, arg.spin_map, arg.out.VolumeCB(), arg.X);
  }
  
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  class StaggeredProlongateLaunch : public TunableVectorYZ {

  protected:
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const int *fine_to_coarse;
    int parity;
    QudaFieldLocation location;
    char vol[TuneKey::volume_n];

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return out.VolumeCB(); } // fine parity is the block y dimension

  public:
    StaggeredProlongateLaunch(ColorSpinorField &out, const ColorSpinorField &in,
                     const int *fine_to_coarse, int parity)
      : TunableVectorYZ(out.SiteSubset(), fineColor), out(out), in(in),
        fine_to_coarse(fine_to_coarse), parity(parity), location(checkLocation(out, in))
    {
      strcpy(vol, out.VolString());
      strcat(vol, ",");
      strcat(vol, in.VolString());

      strcpy(aux, out.AuxString());
      strcat(aux, ",");
      strcat(aux, in.AuxString());
    }

    virtual ~StaggeredProlongateLaunch() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          StaggeredProlongateArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
            arg(out, in, fine_to_coarse, parity);
          StaggeredProlongate<Float,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      } else {
        if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
          //TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

          // ESW version
          TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

          StaggeredProlongateArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_FLOAT2_FIELD_ORDER>
            arg(out, in, fine_to_coarse, parity);
          StaggeredProlongateKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor>
            <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      }
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }

    long long flops() const { return 0; }

    long long bytes() const {
      return in.Bytes() + out.Bytes() + out.SiteSubset()*out.VolumeCB()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  void StaggeredProlongate(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, int parity) {

    StaggeredProlongateLaunch<Float, fineSpin, fineColor, coarseSpin, coarseColor>
    staggered_prolongator(out, in, fine_to_coarse, parity);
    staggered_prolongator.apply(0);
    
    if (checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }


  template <typename Float>
  void StaggeredProlongate(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, const int * const * spin_map, int parity) {

    if (out.Nspin() != 1) errorQuda("Unsupported nSpin %d", out.Nspin());
    const int fineSpin = 1;

    if (in.Nspin() != 2) errorQuda("Coarse spin %d is not supported", in.Nspin());
    const int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++) 
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (out.Ncolor() != 3) errorQuda("Unsupported fine nColor %d", out.Ncolor());
    const int fineColor = 3;

    if (in.Ncolor() != 8*fineColor) errorQuda("Unsupported coarse nColor %d", in.Ncolor());
    const int coarseColor = 8*fineColor;

    StaggeredProlongate<Float,fineSpin,fineColor,coarseSpin,coarseColor>(out, in, fine_to_coarse, parity);

  }

#endif // GPU_MULTIGRID

  void StaggeredProlongate(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, const int * const * spin_map, int parity) {
#if defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)
    if (out.FieldOrder() != in.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d)", 
                out.FieldOrder(), in.FieldOrder());

    QudaPrecision precision = checkPrecision(out, in);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      StaggeredProlongate<double>(out, in, fine_to_coarse, spin_map, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      StaggeredProlongate<float>(out, in, fine_to_coarse, spin_map, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }

    if (checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
#else
    errorQuda("Staggered multigrid has not been built");
#endif
  }

} // end namespace quda

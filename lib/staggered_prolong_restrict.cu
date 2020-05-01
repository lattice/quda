#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>

namespace quda {

#if defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)

  enum class StaggeredTransferType {
    STAGGERED_TRANSFER_PROLONG,
    STAGGERED_TRANSFER_RESTRICT,
    STAGGERED_TRANSFER_INVALID = QUDA_INVALID_ENUM
  };

  using namespace quda::colorspinor;

  // Use a trait to define whether the "out" spin is the fine or coarse spin
  template<int fineSpin, int coarseSpin, StaggeredTransferType transferType> struct StaggeredTransferOutSpin { static constexpr int outSpin = -1; };
  template<int fineSpin, int coarseSpin> struct StaggeredTransferOutSpin<fineSpin,coarseSpin,StaggeredTransferType::STAGGERED_TRANSFER_PROLONG> { static constexpr int outSpin = fineSpin; };
  template<int fineSpin, int coarseSpin> struct StaggeredTransferOutSpin<fineSpin,coarseSpin,StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT> { static constexpr int outSpin = coarseSpin; };

  // Use a trait to define whether the "in" spin is the fine or coarse spin
  template<int fineSpin, int coarseSpin, StaggeredTransferType transferType> struct StaggeredTransferInSpin { static constexpr int inSpin = -1; };
  template<int fineSpin, int coarseSpin> struct StaggeredTransferInSpin<fineSpin,coarseSpin,StaggeredTransferType::STAGGERED_TRANSFER_PROLONG> { static constexpr int inSpin = coarseSpin; };
  template<int fineSpin, int coarseSpin> struct StaggeredTransferInSpin<fineSpin,coarseSpin,StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT> { static constexpr int inSpin = fineSpin; };

  // Use a trait to define whether the "out" color is the fine or coarse color
  template<int fineColor, int coarseColor, StaggeredTransferType transferType> struct StaggeredTransferOutColor { static constexpr int outColor = -1; };
  template<int fineColor, int coarseColor> struct StaggeredTransferOutColor<fineColor,coarseColor,StaggeredTransferType::STAGGERED_TRANSFER_PROLONG> { static constexpr int outColor = fineColor; };
  template<int fineColor, int coarseColor> struct StaggeredTransferOutColor<fineColor,coarseColor,StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT> { static constexpr int outColor = coarseColor; };

  // Use a trait to define whether the "in" color is the fine or coarse color
  template<int fineColor, int coarseColor, StaggeredTransferType transferType> struct StaggeredTransferInColor { static constexpr int inColor = -1; };
  template<int fineColor, int coarseColor> struct StaggeredTransferInColor<fineColor,coarseColor,StaggeredTransferType::STAGGERED_TRANSFER_PROLONG> { static constexpr int inColor = coarseColor; };
  template<int fineColor, int coarseColor> struct StaggeredTransferInColor<fineColor,coarseColor,StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT> { static constexpr int inColor = fineColor; };

  // Function to return the fine ColorSpinorField
  template<StaggeredTransferType transferType>
  inline const ColorSpinorField& fineColorSpinorField(const ColorSpinorField& quoteIn, const ColorSpinorField& quoteOut) {
    errorQuda("Invalid transfer type %d for fineColorSpinorField", (int)transferType);
    return quoteIn; 
  }

  // on prolong, the out vector is the fine vector
  template<>
  inline const ColorSpinorField& fineColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_PROLONG>(const ColorSpinorField& quoteIn, const ColorSpinorField& quoteOut) {
    return quoteOut;
  }

  // on restrict, the in vector is the fine vector
  template<>
  inline const ColorSpinorField& fineColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT>(const ColorSpinorField& quoteIn, const ColorSpinorField& quoteOut) {
    return quoteIn;
  }

  // Function to return the coarse ColorSpinorField
  template<StaggeredTransferType transferType>
  inline const ColorSpinorField& coarseColorSpinorField(const ColorSpinorField& quoteIn, const ColorSpinorField& quoteOut) {
    errorQuda("Invalid transfer type %d for coarseColorSpinorField", (int)transferType);
    return quoteIn; 
  }

  // on prolong, the out vector is the fine vector
  template<>
  inline const ColorSpinorField& coarseColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_PROLONG>(const ColorSpinorField& quoteIn, const ColorSpinorField& quoteOut) {
    return quoteIn;
  }

  // on restrict, the in vector is the fine vector
  template<>
  inline const ColorSpinorField& coarseColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT>(const ColorSpinorField& quoteIn, const ColorSpinorField& quoteOut) {
    return quoteOut;
  }
  /** 
      Kernel argument struct
  */
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, QudaFieldOrder order, StaggeredTransferType theTransferType>
  struct StaggeredProlongRestrictArg {
    FieldOrderCB<Float, StaggeredTransferOutSpin<fineSpin,coarseSpin,theTransferType>::outSpin, StaggeredTransferOutColor<fineColor,coarseColor,theTransferType>::outColor,1,order> out;
    const FieldOrderCB<Float, StaggeredTransferInSpin<fineSpin,coarseSpin,theTransferType>::inSpin, StaggeredTransferInColor<fineColor,coarseColor,theTransferType>::inColor,1,order> in;
    const int *geo_map;  // need to make a device copy of this
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the output field (if single parity)
    const int nParity; // number of parities of input fine field
    const int fineX[4]; // fine spatial volume
    const int fineVolumeCB; // fine spatial volume
    const int coarseVolumeCB; // coarse spatial volume
    static constexpr StaggeredTransferType transferType = theTransferType;

    StaggeredProlongRestrictArg(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *geo_map, const int parity)
      : out(out), in(in), geo_map(geo_map), spin_map(), parity(parity),
        nParity(fineColorSpinorField<transferType>(in,out).SiteSubset()),
        fineX{fineColorSpinorField<transferType>(in,out).X()[0],
          fineColorSpinorField<transferType>(in,out).X()[1],
          fineColorSpinorField<transferType>(in,out).X()[2],
          fineColorSpinorField<transferType>(in,out).X()[3]},
        fineVolumeCB(fineColorSpinorField<transferType>(in,out).VolumeCB()),
        coarseVolumeCB(coarseColorSpinorField<transferType>(in,out).VolumeCB())
    {;}

    StaggeredProlongRestrictArg(const StaggeredProlongRestrictArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,order,theTransferType> &arg)
      : out(arg.out), in(arg.in), geo_map(arg.geo_map), spin_map(),
        parity(arg.parity), nParity(arg.nParity), fineX{arg.fineX[0],arg.fineX[1],arg.fineX[2],arg.fineX[3]},
        fineVolumeCB(arg.fineVolumeCB), coarseVolumeCB(arg.coarseVolumeCB)
    {;}
  };

  /**
     Performs the permutation from a coarse degree of freedom to a 
     fine degree of freedom
  */
  template <StaggeredTransferType transferType, class OutAccessor, class InAccessor, typename S>
  __device__ __host__ inline void staggeredProlongRestrict(OutAccessor& out, const InAccessor &in,
                                            int parity, int x_cb, int c, const int *geo_map, const S& spin_map, const int fineVolumeCB, const int coarseVolumeCB, const int X[]) {
    int x = parity*fineVolumeCB + x_cb;
    int x_coarse = geo_map[x];
    int parity_coarse = (x_coarse >= coarseVolumeCB) ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*coarseVolumeCB;

    // coarse_color = 8*fine_color + corner of the hypercube
    int fineCoords[5];
    fineCoords[4] = 0;
    getCoords(fineCoords,x_cb,X,parity);
    int hyperCorner = 4*(fineCoords[3]%2)+2*(fineCoords[2]%2)+(fineCoords[1]%2);

    if (transferType == StaggeredTransferType::STAGGERED_TRANSFER_PROLONG) {
      out(parity,x_cb,0,c) = in(parity_coarse, x_coarse_cb, spin_map(0,parity), 8*c+hyperCorner);
    } else { 
      out(parity_coarse, x_coarse_cb, spin_map(0,parity), 8*c+hyperCorner) = in(parity,x_cb,0,c);
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void StaggeredProlongRestrict(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;

      // We don't actually have to loop over spin because fineSpin = 1, coarseSpin = fine parity
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
        for (int c=0; c<fineColor; c++) {
          staggeredProlongRestrict<Arg::transferType>(arg.out, arg.in, parity, x_cb, c, arg.geo_map, arg.spin_map, arg.fineVolumeCB, arg.coarseVolumeCB, arg.fineX);
        }
      }
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void StaggeredProlongRestrictKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = arg.nParity == 2 ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
    if (x_cb >= arg.fineVolumeCB) return;

    int c = blockDim.z*blockIdx.z + threadIdx.z;
    if (c >= fineColor) return;

    staggeredProlongRestrict<Arg::transferType>(arg.out, arg.in, parity, x_cb, c, arg.geo_map, arg.spin_map, arg.fineVolumeCB, arg.coarseVolumeCB, arg.fineX);
  }
  
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, StaggeredTransferType transferType>
  class StaggeredProlongRestrictLaunch : public TunableVectorYZ {

  protected:
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const int *fine_to_coarse;
    int parity;
    QudaFieldLocation location;
    char vol[TuneKey::volume_n];

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return fineColorSpinorField<transferType>(in,out).VolumeCB(); } // fine parity is the block y dimension

  public:
    StaggeredProlongRestrictLaunch(ColorSpinorField &out, const ColorSpinorField &in,
                     const int *fine_to_coarse, int parity)
      : TunableVectorYZ(fineColorSpinorField<transferType>(in,out).SiteSubset(), fineColor), out(out), in(in),
        fine_to_coarse(fine_to_coarse), parity(parity), location(checkLocation(out, in))
    {
      strcpy(vol, fineColorSpinorField<transferType>(in,out).VolString());
      strcat(vol, ",");
      strcat(vol, coarseColorSpinorField<transferType>(in,out).VolString());

      strcpy(aux, fineColorSpinorField<transferType>(in,out).AuxString());
      strcat(aux, ",");
      strcat(aux, coarseColorSpinorField<transferType>(in,out).AuxString());
    }

    virtual ~StaggeredProlongRestrictLaunch() { }

    void apply(const qudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          StaggeredProlongRestrictArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,transferType>
            arg(out, in, fine_to_coarse, parity);
          StaggeredProlongRestrict<Float,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      } else {
        if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

          StaggeredProlongRestrictArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_FLOAT2_FIELD_ORDER,transferType>
            arg(out, in, fine_to_coarse, parity);
          StaggeredProlongRestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor>
            <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      }
    }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }

    long long flops() const { return 0; }

    long long bytes() const {
      return in.Bytes() + out.Bytes() + fineColorSpinorField<transferType>(in,out).SiteSubset()*fineColorSpinorField<transferType>(in,out).VolumeCB()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, StaggeredTransferType transferType>
  void StaggeredProlongRestrict(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, int parity) {

    StaggeredProlongRestrictLaunch<Float, fineSpin, fineColor, coarseSpin, coarseColor, transferType>
    staggered_prolong_restrict(out, in, fine_to_coarse, parity);
    staggered_prolong_restrict.apply(0);
    
    if (checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  template <int fineSpin, int fineColor, int coarseSpin, int coarseColor, StaggeredTransferType transferType>
  void StaggeredProlongRestrict(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, int parity) {
    // check precision
    QudaPrecision precision = checkPrecision(out, in);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      StaggeredProlongRestrict<double,fineSpin,fineColor,coarseSpin,coarseColor,transferType>(out, in, fine_to_coarse, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      StaggeredProlongRestrict<float,fineSpin,fineColor,coarseSpin,coarseColor,transferType>(out, in, fine_to_coarse, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }

  }

  template <StaggeredTransferType transferType>
  void StaggeredProlongRestrict(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, const int * const * spin_map, int parity) {

    if (out.FieldOrder() != in.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d)", 
                out.FieldOrder(), in.FieldOrder());

    if (fineColorSpinorField<transferType>(in,out).Nspin() != 1) errorQuda("Fine spin %d is not supported", fineColorSpinorField<transferType>(in,out).Nspin());
    const int fineSpin = 1;

    if (coarseColorSpinorField<transferType>(in,out).Nspin() != 2) errorQuda("Coarse spin %d is not supported", coarseColorSpinorField<transferType>(in,out).Nspin());
    const int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++) 
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (fineColorSpinorField<transferType>(in,out).Ncolor() != 3) errorQuda("Unsupported fine nColor %d",fineColorSpinorField<transferType>(in,out).Ncolor());
    const int fineColor = 3;

    if (coarseColorSpinorField<transferType>(in,out).Ncolor() != 8*fineColor) errorQuda("Unsupported coarse nColor %d", coarseColorSpinorField<transferType>(in,out).Ncolor());
    const int coarseColor = 8*fineColor;

    StaggeredProlongRestrict<fineSpin,fineColor,coarseSpin,coarseColor,transferType>(out, in, fine_to_coarse, parity);
  }

#endif // defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)

  void StaggeredProlongate(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, const int * const * spin_map, int parity) {
#if defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)

    StaggeredProlongRestrict<StaggeredTransferType::STAGGERED_TRANSFER_PROLONG>(out, in, fine_to_coarse, spin_map, parity);

#else
    errorQuda("Staggered multigrid has not been build");
#endif
  }

  void StaggeredRestrict(ColorSpinorField &out, const ColorSpinorField &in,
                  const int *fine_to_coarse, const int * const * spin_map, int parity) {
#if defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)
 

    StaggeredProlongRestrict<StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT>(out, in, fine_to_coarse, spin_map, parity);

#else
    errorQuda("Staggered multigrid has not been build");
#endif
  }



} // end namespace quda

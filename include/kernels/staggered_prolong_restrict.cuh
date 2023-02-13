#pragma once

#include <color_spinor_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda {

  using namespace quda::colorspinor;

  enum class StaggeredTransferType {
    STAGGERED_TRANSFER_PROLONG,
    STAGGERED_TRANSFER_RESTRICT,
    STAGGERED_TRANSFER_INVALID = QUDA_INVALID_ENUM
  };

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
  inline const ColorSpinorField& fineColorSpinorField(const ColorSpinorField& quoteIn, const ColorSpinorField&)
  {
    errorQuda("Invalid transfer type %d for fineColorSpinorField", (int)transferType);
    return quoteIn;
  }

  // on prolong, the out vector is the fine vector
  template<>
  inline const ColorSpinorField& fineColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_PROLONG>(const ColorSpinorField&, const ColorSpinorField& quoteOut) {
    return quoteOut;
  }

  // on restrict, the in vector is the fine vector
  template<>
  inline const ColorSpinorField& fineColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT>(const ColorSpinorField& quoteIn, const ColorSpinorField&) {
    return quoteIn;
  }

  // Function to return the coarse ColorSpinorField
  template<StaggeredTransferType transferType>
  inline const ColorSpinorField& coarseColorSpinorField(const ColorSpinorField& quoteIn, const ColorSpinorField&)
  {
    errorQuda("Invalid transfer type %d for coarseColorSpinorField", (int)transferType);
    return quoteIn;
  }

  // on prolong, the out vector is the fine vector
  template<>
  inline const ColorSpinorField& coarseColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_PROLONG>(const ColorSpinorField& quoteIn, const ColorSpinorField&) {
    return quoteIn;
  }

  // on restrict, the in vector is the fine vector
  template<>
  inline const ColorSpinorField& coarseColorSpinorField<StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT>(const ColorSpinorField&, const ColorSpinorField& quoteOut) {
    return quoteOut;
  }

  /**
      Kernel argument struct
  */
  template <typename Float, int fineSpin, int fineColor_, int coarseSpin, int coarseColor, StaggeredTransferType theTransferType, bool native>
  struct StaggeredProlongRestrictArg : kernel_param<> {
    static constexpr int fineColor = fineColor_;
    static constexpr int outSpin = StaggeredTransferOutSpin<fineSpin,coarseSpin,theTransferType>::outSpin;
    static constexpr int inSpin = StaggeredTransferInSpin<fineSpin,coarseSpin,theTransferType>::inSpin;
    static constexpr int outColor = StaggeredTransferOutColor<fineColor,coarseColor,theTransferType>::outColor;
    static constexpr int inColor = StaggeredTransferInColor<fineColor,coarseColor,theTransferType>::inColor;
    static constexpr QudaFieldOrder outOrder = native ? colorspinor::getNative<Float>(outSpin) : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    static constexpr QudaFieldOrder inOrder = native ? colorspinor::getNative<Float>(inSpin) : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    FieldOrderCB<Float, outSpin, outColor, 1, outOrder> out;
    const FieldOrderCB<Float, inSpin, inColor, 1, inOrder> in;
    const int *geo_map;  // need to make a device copy of this
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the output field (if single parity)
    const int nParity; // number of parities of input fine field
    const int fineX[4]; // fine spatial volume
    const int fineVolumeCB; // fine spatial volume
    const int coarseVolumeCB; // coarse spatial volume
    static constexpr StaggeredTransferType transferType = theTransferType;

    StaggeredProlongRestrictArg(ColorSpinorField &out, const ColorSpinorField &in,
                                const int *geo_map, const int parity) :
      out(out), in(in), geo_map(geo_map), spin_map(), parity(parity),
      nParity(fineColorSpinorField<transferType>(in,out).SiteSubset()),
      fineX{fineColorSpinorField<transferType>(in,out).X()[0],
        fineColorSpinorField<transferType>(in,out).X()[1],
        fineColorSpinorField<transferType>(in,out).X()[2],
        fineColorSpinorField<transferType>(in,out).X()[3]},
      fineVolumeCB(fineColorSpinorField<transferType>(in,out).VolumeCB()),
      coarseVolumeCB(coarseColorSpinorField<transferType>(in,out).VolumeCB())
    {
      this->threads = dim3(fineVolumeCB, nParity, fineColor);
    }
  };

  /**
     Performs the permutation from a coarse degree of freedom to a
     fine degree of freedom
  */
  template <typename Arg> struct StaggeredProlongRestrict {
    const Arg &arg;
    constexpr StaggeredProlongRestrict(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int c)
    {
      parity = arg.nParity == 2 ? parity : arg.parity;

      int x = parity*arg.fineVolumeCB + x_cb;
      int x_coarse = arg.geo_map[x];
      int parity_coarse = (x_coarse >= arg.coarseVolumeCB) ? 1 : 0;
      int x_coarse_cb = x_coarse - parity_coarse*arg.coarseVolumeCB;

      // coarse_color = 8*fine_color + corner of the hypercube
      int fineCoords[5];
      fineCoords[4] = 0;
      getCoords(fineCoords,x_cb,arg.fineX,parity);
      int hyperCorner = 4*(fineCoords[3]%2)+2*(fineCoords[2]%2)+(fineCoords[1]%2);

      if (Arg::transferType == StaggeredTransferType::STAGGERED_TRANSFER_PROLONG) {
        arg.out(parity,x_cb,0,c) = arg.in(parity_coarse, x_coarse_cb, arg.spin_map(0,parity), 8*c+hyperCorner);
      } else {
        arg.out(parity_coarse, x_coarse_cb, arg.spin_map(0,parity), 8*c+hyperCorner) = arg.in(parity,x_cb,0,c);
      }
    }
  };

}

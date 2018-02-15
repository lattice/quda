#include <color_spinor_field.h>
#include <tune_quda.h>
#include <multigrid_helper.cuh>
#include <fast_intdiv.h>
#include <cub_helper.cuh>
#include <uint_to_char.h>
#include <typeinfo>
#include <vector>
#include <assert.h>

#include <color_spinor_field_order.h>

namespace quda{

  using namespace quda::colorspinor;


//Field order for composite ColorSpinorField
  template<typename real, int nSpin, int nColor, QudaFieldOrder order>
  struct FillBArg {

    FieldOrderCB<real,nSpin,nColor,1,order,real,real,true> B;//composite color spinor!
    FieldOrderCB<real,nSpin,nColor,1,order,real,real,true> b;
    int *index_map;

    FillBArg(ColorSpinorField &B, const ColorSpinorField &b, int *index_map)
      : B(B), b(b), index_map(index_map) { }

  };

  // CPU routine to copy the null-space vectors into the V-field
  template <typename Float, int nSpin, int nColor, typename Arg>
  void FillBCPU(Arg &arg) {

    for (int parity=0; parity<arg.b.Nparity(); parity++) {
      for (int x_cb=0; x_cb<arg.b.VolumeCB(); x_cb++) {
        int x = parity*arg.b.VolumeCB() + x_cb;
        int component_id = arg.index_map[x];
	for (int s=0; s<nSpin; s++) {
	  for (int c=0; c<nColor; c++) {
	    arg.B(parity, x_cb*component_id, s, c) = arg.b(parity, x_cb, s, c);
	  }
	}
      }
    }

  }

  // GPU kernel to copy the null-space vectors into the V-field
  template <typename Float, int nSpin, int nColor, typename Arg>
  __global__ void FillBGPU(Arg arg) {

    int x_cb   = threadIdx.x + blockDim.x*blockIdx.x;
    int parity = threadIdx.y + blockDim.y*blockIdx.y;
    if (x_cb >= arg.b.VolumeCB()) return;
    int x = parity*arg.b.VolumeCB()+x_cb;
    int component_id = arg.index_map[x];

    for (int s=0; s<nSpin; s++) {
      for (int c=0; c<nColor; c++) {
	arg.B(parity, x_cb*component_id, s, c) = arg.b(parity, x_cb, s, c);
      }
    }

  }

  template <typename real, int nSpin, int nColor>
  class FillBLaunch : public TunableVectorY {

    ColorSpinorField    &B;//must be transfered to 5d
    const ColorSpinorField &b;
    //const std::vector<ColorSpinorField*> &B;
    int *index_map;
    unsigned int minThreads() const { return b.VolumeCB(); }
    bool tuneGridDim() const { return false; }

  public:
    FillBLaunch(ColorSpinorField &B, const ColorSpinorField &b, int *index_map)
      : TunableVectorY(2), B(B), b(b), index_map(index_map) {
      (b.Location() == QUDA_CPU_FIELD_LOCATION) ? strcpy(aux,"CPU") : strcpy(aux,"GPU");
    }
    virtual ~FillBLaunch() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (b.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (b.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	  FillBArg<real,nSpin,nColor,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> arg(B,b,index_map);
	  FillBCPU<real,nSpin,nColor>(arg);
	} else {
	  errorQuda("Field order not implemented %d", b.FieldOrder());
	}
      } else {
	if (b.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
	  FillBArg<real,nSpin,nColor,QUDA_FLOAT2_FIELD_ORDER> arg(B,b,index_map);
	  FillBGPU<real,nSpin,nColor> <<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	} else {
	  errorQuda("Field order not implemented %d", b.FieldOrder());
	}
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (b.Location() == QUDA_CUDA_FIELD_LOCATION) {
	return advanceSharedBytes(param) || advanceBlockDim(param);
      } else {
	return false;
      }
    }

    TuneKey tuneKey() const { return TuneKey(b.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; }
    long long bytes() const { return 2ll*b.Bytes(); }
  };


  template <typename real, int nSpin, int nColor>
  void FillB(ColorSpinorField &B, const ColorSpinorField &b, int *index_map) {
    {
      FillBLaunch<real,nSpin,nColor> f(B,b,index_map);
      f.apply(0);
    }
  }

  template <typename Float, int nSpin>
  void FillB(ColorSpinorField &B, const ColorSpinorField &b, int *index_map) {
    if(b.Ncolor() == 3) {
      FillB<Float,nSpin,3>(B,b,index_map);
    } else {
      errorQuda("Unsupported nColor %d", b.Ncolor());
    }
  }

  template <typename Float>
  void FillB(ColorSpinorField &B, const ColorSpinorField &b, int *index_map) {
    if (b.Nspin() == 4) {
      FillB<Float,4>(B,b,index_map);
#ifdef GPU_STAGGERED_DIRAC
    } else if (b.Nspin() == 1) {
      FillB<Float,1>(B,b,index_map);
#endif
    } else {
      errorQuda("Unsupported nSpin %d", b.Nspin());
    }
  }

  void FillB(ColorSpinorField &B, const ColorSpinorField &b, int *index_map) {

    if( !B.IsComposite() && (B.CompositeDim() != B.X(4))) errorQuda("Something is wrong...\n");

    if (b.Precision() == QUDA_DOUBLE_PRECISION) {
      FillB<double>(B,b,index_map);
    } else if (b.Precision() == QUDA_SINGLE_PRECISION) {
      FillB<float >(B,b,index_map);
    } else {
      errorQuda("Unsupported precision %d", b.Precision());
    }
  }

}

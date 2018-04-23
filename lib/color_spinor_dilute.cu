/*
  Spinor reordering and copying routines.  These are implemented to
  un on both CPU and GPU.  Here we are templating on the following:
  - input precision
  - output precision
  - number of colors
  - number of spins
  - field ordering
*/

#define DISABLE_GHOST

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <integer_sequence.hpp> // C++11 version of this C++14 feature

namespace quda {

  using namespace colorspinor;

  template<typename real, int Ns, int Nc, int Nvec, QudaFieldOrder order>
  struct Arg {
    typedef typename colorspinor::FieldOrderCB<real,Ns,Nc,1,order> V;
    V out[Nvec];
    const V in;
    const int nParity;
    const int volumeCB;
    template <std::size_t... S>
    Arg(std::vector<ColorSpinorField*> &out, const ColorSpinorField &in, std::index_sequence<S...>)
      : out{*out[S]...}, in(in), nParity(in.SiteSubset()), volumeCB(in.VolumeCB()) { }
  };

  template<typename real, int Ns, int Nc, QudaDiluteType type, typename Arg> // Gauss
  __device__ __host__ inline void dilute(Arg &arg, int parity, int x_cb) {

    if (type == QUDA_DILUTE_COLOR) {
      for (int s=0; s<Ns; s++) {
        for (int c=0; c<Nc; c++) {
          arg.out[c](parity, x_cb, s, c) = arg.in(parity, x_cb, s, c);
        }
      }
    } else if (type == QUDA_DILUTE_SPIN) {
      for (int s=0; s<Ns; s++) {
        for (int c=0; c<Nc; c++) {
          arg.out[s](parity, x_cb, s, c) = arg.in(parity, x_cb, s, c);
        }
      }
    } else if (type == QUDA_DILUTE_SPIN_COLOR) {
      for (int s=0; s<Ns; s++) {
        for (int c=0; c<Nc; c++) {
          arg.out[s*Nc+c](parity, x_cb, s, c) = arg.in(parity, x_cb, s, c);
        }
      }
    }

  }

  // CPU function to dilute color-spinor fields
  template <typename real, int Ns, int Nc, QudaDiluteType type, typename Arg>
  void dilutionCPU(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++) {
        dilute<real,Ns,Nc,type>(arg, parity, x_cb);
      }
    }
  }

  // CUDA kernel to dilute color-spinor fields
  template <typename real, int Ns, int Nc, QudaDiluteType type, typename Arg>
  __global__ void dilutionGPU(Arg arg) {

    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;

    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    if (parity >= arg.nParity) return;

    for (int parity=0; parity<arg.nParity; parity++) {
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++) {
        dilute<real,Ns,Nc,type>(arg, parity, x_cb);
      }
    }
  }

  template <typename real, int Ns, int Nc, QudaDiluteType type, typename Arg>
  class Diluter : TunableVectorY {
    Arg &arg;
    const ColorSpinorField &meta; // this reference is for meta data only

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return meta.VolumeCB(); }

  public:
    Diluter(Arg &arg, const ColorSpinorField &meta)
      : TunableVectorY(meta.SiteSubset()), arg(arg), meta(meta) {
      strcpy(aux, meta.AuxString());
      strcat(aux, meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU" : ",CPU");
    }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	dilutionCPU<real, Ns, Nc, type>(arg);
      } else {
	dilutionGPU<real, Ns, Nc, type> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 0; }
    long long bytes() const { return 2*meta.Bytes(); } // perfect dilution should be one read and one write
  };

  template <typename real, int Ns, int Nc, QudaFieldOrder order>
  void dilute(std::vector<ColorSpinorField*> &out, const ColorSpinorField &in, QudaDiluteType type)
  {
    switch (type) {
    case QUDA_DILUTE_COLOR:
      {
        if ((int)out.size() != in.Ncolor()) errorQuda("out.size (%lu) does not match number of dilutions %d", out.size(), in.Ncolor());
        Arg<real, Ns, Nc, Nc, order> arg(out, in, std::make_index_sequence<Nc>());
        Diluter<real, Ns, Nc, QUDA_DILUTE_COLOR, Arg<real, Ns, Nc, Nc, order> > diluter(arg, in);
        diluter.apply(0);
        break;
      }
    case QUDA_DILUTE_SPIN:
      {
        if ((int)out.size() != in.Nspin()) errorQuda("out.size (%lu) does not match number of dilutions %d", out.size(), in.Nspin());
        Arg<real, Ns, Nc, Ns, order> arg(out, in, std::make_index_sequence<Ns>());
        Diluter<real, Ns, Nc, QUDA_DILUTE_SPIN, Arg<real, Ns, Nc, Ns, order> > diluter(arg, in);
        diluter.apply(0);
        break;
      }
    case QUDA_DILUTE_SPIN_COLOR:
      {
        if ((int)out.size() != in.Nspin()*in.Ncolor()) errorQuda("out.size (%lu) does not match number of dilutions %d", out.size(), in.Nspin()*in.Ncolor());
        Arg<real, Ns, Nc, Ns*Nc, order> arg(out, in, std::make_index_sequence<Ns*Nc>());
        Diluter<real, Ns, Nc, QUDA_DILUTE_SPIN_COLOR, Arg<real, Ns, Nc, Ns*Nc, order> > diluter(arg, in);
        diluter.apply(0);
        break;
      }
    default:
      errorQuda("Dilution type %d not implemented", type);
    }
  }

  /** Decide on the input order*/
  template <typename real, int Ns, int Nc>
  void dilute(std::vector<ColorSpinorField*> &out, const ColorSpinorField &in, QudaDiluteType type)
  {
    switch (in.FieldOrder()) {
    case QUDA_FLOAT2_FIELD_ORDER: dilute<real,Ns,Nc,QUDA_FLOAT2_FIELD_ORDER>(out, in, type); break;
    case QUDA_FLOAT4_FIELD_ORDER: dilute<real,Ns,Nc,QUDA_FLOAT4_FIELD_ORDER>(out, in, type); break;
    case QUDA_SPACE_SPIN_COLOR_FIELD_ORDER: dilute<real,Ns,Nc,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(out, in, type); break;
    default: errorQuda("Order %d not defined (Ns=%d, Nc=%d)", in.FieldOrder(), Ns, Nc);
    }
  }

  template <typename real>
  void dilute(std::vector<ColorSpinorField*> out, const ColorSpinorField &in, QudaDiluteType type)
  {
    switch (in.Nspin()) {
    case 4:
      switch (in.Ncolor()) {
      case  3: dilute<real,4, 3>(out, in, type); break;
      default: errorQuda("Ncolor = %d not implemented", in.Ncolor());
      }
      break;
    case 2:
      switch (in.Ncolor()) {
      case 24: dilute<real,2,24>(out, in, type); break;
      case 32: dilute<real,2,32>(out, in, type); break;
      default: errorQuda("Ncolor = %d not implemented", in.Ncolor());
      }
      break;
    case 1:
      switch (in.Ncolor()) {
      case  3: dilute<real,1, 3>(out, in, type); break;
      default: errorQuda("Ncolor = %d not implemented", in.Ncolor());
      }
      break;
    default: errorQuda("Nspin = %d not implemented", in.Nspin());
    }
  }

  void dilute(std::vector<ColorSpinorField*> &out, const ColorSpinorField &in, QudaDiluteType type)
  {
    switch (in.Precision()) {
    case QUDA_DOUBLE_PRECISION: dilute<double>(out, in, type); break;
    case QUDA_SINGLE_PRECISION: dilute< float>(out, in, type); break;
    default: errorQuda("Precision %d not implemented", in.Precision());
    }
  }

} // namespace quda

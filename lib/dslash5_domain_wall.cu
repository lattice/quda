#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <inline_ptx.h>

namespace quda {

#ifdef GPU_DOMAIN_WALL_DIRAC

  /**
     @brief Parameter structure for applying the Dslash
   */
  template <typename Float, int nColor>
  struct Dslash5Arg {
    typedef typename colorspinor_mapper<Float,4,nColor>::type F;
    typedef typename mapper<Float>::type real;

    F out;                  // output vector field
    const F in;             // input vector field
    const F x;              // auxiliary input vector field
    const int nParity;      // number of parities we're working on
    const int volume_cb;    // checkerboarded volume
    const int volume_4d_cb; // 4-d checkerboarded volume
    const int_fastdiv Ls;   // length of 5th dimension

    const real m_f;         // fermion mass parameter
    const real m_5;         // Wilson mass shift

    const bool dagger;      // dagger
    const bool xpay;        // whether we are doing xpay or not

    // zMobius / Zolotarev coefficients
    complex<real> b_5[QUDA_MAX_DWF_LS];
    complex<real> c_5[QUDA_MAX_DWF_LS];

    // real constant Mobius coefficient
    double b;
    double c;

    // xpay coefficients
    real a;
    complex<real> a_5[QUDA_MAX_DWF_LS];

    Dslash5Type type;

    Dslash5Arg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
               double m_f, double m_5, const Complex *b_5_, const Complex *c_5_,
               double a, bool dagger, Dslash5Type type)
      : out(out), in(in), x(x), nParity(in.SiteSubset()),
	volume_cb(in.VolumeCB()), volume_4d_cb(volume_cb/in.X(4)), Ls(in.X(4)),
	m_f(m_f), m_5(m_5), a(a), dagger(dagger), xpay(in.V() == x.V() ? false: true), type(type)
    {
      if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

      switch(type) {
      case DSLASH5_DWF:
	// xpay
	for (int s=0; s<Ls; s++) {
	  a_5[s] = a;
	}
	break;
      case DSLASH5_MOBIUS_PRE:
	for (int s=0; s<Ls; s++) {
	  b_5[s] = b_5_[s].real();
	  c_5[s] = 0.5*c_5_[s].real();

	  // xpay
	  a_5[s] = (0.5/(b_5_[s]*(m_5+4.0) + 1.0)).real();
	  a_5[s] *= a_5[s] * static_cast<real>(a);
        }
	break;
      case DSLASH5_MOBIUS:
	for (int s=0; s<Ls; s++) {
	  b_5[s] = 1.0;
	  c_5[s] = (0.5 * (c_5_[s] * (m_5 + 4.0) - 1.0) / (b_5_[s] * (m_5 + 4.0) + 1.0)).real();

	  // axpy
	  a_5[s] = (0.5 / (b_5_[s] * (m_5 + 4.0) + 1.0)).real();
	  a_5[s] *= a_5[s] * static_cast<real>(a);
	}
	break;
      default:
	errorQuda("Unknown Dslash5Type %d", type);
      }
      b = b_5[0].real();
      c = c_5[0].real();
    }
  };

  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  __device__ __host__ inline void dslash5(Arg &arg, int parity, int x_cb, int s) {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,4> Vector;

    Vector out;

    { // forwards direction
      const int fwd_idx = ((s + 1) % arg.Ls) * arg.volume_4d_cb + x_cb;
      const Vector in = arg.in(fwd_idx, parity);
      constexpr int proj_dir = dagger ? +1 : -1;
      if (s == arg.Ls-1) {
	out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
      } else {
	out += in.project(4, proj_dir).reconstruct(4, proj_dir);
      }
    }

    { // backwards direction
      const int back_idx = ((s + arg.Ls - 1) % arg.Ls) * arg.volume_4d_cb + x_cb;
      const Vector in = arg.in(back_idx, parity);
      constexpr int proj_dir = dagger ? -1 : +1;
      if (s == 0) {
	out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
      } else {
	out += in.project(4, proj_dir).reconstruct(4, proj_dir);
      }
    }

    if (type == DSLASH5_DWF && xpay) {
      Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
      out = x + arg.a*out;
    } else if (type == DSLASH5_MOBIUS_PRE) {
      Vector diagonal = arg.in(s*arg.volume_4d_cb + x_cb, parity);
      const complex<real> b = arg.b; // arg.b_5[s]
      const complex<real> c = arg.c; // arg.c_5[s]
      out = c * out + b * diagonal;

      if (xpay) {
	Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
        complex<real> a = arg.a; // arg.a_5[s]
	out = x + a*out;
      }
    } else if (type == DSLASH5_MOBIUS) {
      Vector diagonal = arg.in(s*arg.volume_4d_cb + x_cb, parity);
      const complex<real> c = arg.c; // arg.c_5[s]
      out = c * out + diagonal;

      if (xpay) { // really axpy
	Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
        complex<real> a = arg.a; // arg.a_5[s]
	out = a*x + out;
      }
    }

    arg.out(s*arg.volume_4d_cb + x_cb, parity) = out;
  }

  // CPU kernel for applying the dslash operator
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  void dslash5CPU(Arg &arg)
  {
    for (int parity= 0; parity < arg.nParity; parity++) {
      for (int s=0; s < arg.Ls; s++) {
	for (int x_cb = 0; x_cb < arg.volume_4d_cb; x_cb++) { // 4-d volume
	  dslash5<Float,nColor,dagger,xpay,type>(arg, parity, x_cb, s);
	}  // 4-d volumeCB
      } // ls
    } // parity

  }

  // GPU Kernel for applying the dslash operator
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  __global__ void dslash5GPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int s = blockIdx.y*blockDim.y + threadIdx.y;
    int parity = blockIdx.z*blockDim.z + threadIdx.z;

    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    dslash5<Float,nColor,dagger,xpay,type>(arg, parity, x_cb, s);
  }

  template <typename Float, int nColor, typename Arg>
  class Dslash5 : public TunableVectorYZ {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const {
      long long Ls = meta.X(4);
      long long bulk = (Ls-2)*(meta.Volume()/Ls);
      long long wall = 2*meta.Volume()/Ls;
      int n = meta.Ncolor() * meta.Nspin();
      bool zMobius = false; // set to true when we have complexity

      long long flops_ = 0;
      switch (arg.type) {
      case DSLASH5_DWF:
        flops_ = n * (8ll*bulk + 10ll*wall + (arg.xpay ? 4ll * meta.Volume() : 0) );
        break;
      case DSLASH5_MOBIUS_PRE:
        flops_ = n * (8ll*bulk + 10ll*wall + (zMobius ? 14ll : 6ll) * meta.Volume() +
                      (arg.xpay ? (zMobius ? 8ll : 4ll) * meta.Volume() : 0) );
        break;
      case DSLASH5_MOBIUS:
        flops_ = n * (8ll*bulk + 10ll*wall + (zMobius ? 8ll : 4ll) * meta.Volume() +
                      (arg.xpay ? (zMobius ? 8ll : 4ll) * meta.Volume() : 0) );
        break;
      default:
	errorQuda("Unknown Dslash5Type %d", arg.type);
      }

      return flops_;
    }

    long long bytes() const {
      switch (arg.type) {
      case DSLASH5_DWF:        return arg.out.Bytes() + 2*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case DSLASH5_MOBIUS_PRE: return arg.out.Bytes() + 3*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case DSLASH5_MOBIUS:     return arg.out.Bytes() + 3*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      default: errorQuda("Unknown Dslash5Type %d", arg.type);
      }
      return 0ll;
    }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volume_4d_cb; }
    int blockStep() const { return 8; }
    int blockMin() const { return 8; }

  public:
    Dslash5(Arg &arg, const ColorSpinorField &meta)
      : TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
      if (arg.dagger) strcat(aux, ",Dagger");
      if (arg.xpay) strcat(aux,",xpay");
      strcat(aux, arg.type == DSLASH5_DWF ? ",DSLASH5_DWF" :
             arg.type == DSLASH5_MOBIUS_PRE ? ",DSLASH5_MOBIUS_PRE" : ",DSLASH5_MOBIUS");
    }
    virtual ~Dslash5() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.type == DSLASH5_DWF) {
	  if (arg.xpay) arg.dagger ?
			  dslash5CPU<Float,nColor, true,true,DSLASH5_DWF>(arg) :
			  dslash5CPU<Float,nColor,false,true,DSLASH5_DWF>(arg);
	  else          arg.dagger ?
			  dslash5CPU<Float,nColor, true,false,DSLASH5_DWF>(arg) :
			  dslash5CPU<Float,nColor,false,false,DSLASH5_DWF>(arg);
	} else if (arg.type == DSLASH5_MOBIUS_PRE) {
	  if (arg.xpay) arg.dagger ?
			  dslash5CPU<Float,nColor, true, true,DSLASH5_MOBIUS_PRE>(arg) :
			  dslash5CPU<Float,nColor,false, true,DSLASH5_MOBIUS_PRE>(arg);
	  else          arg.dagger ?
			  dslash5CPU<Float,nColor, true,false,DSLASH5_MOBIUS_PRE>(arg) :
			  dslash5CPU<Float,nColor,false,false,DSLASH5_MOBIUS_PRE>(arg);
	} else if (arg.type == DSLASH5_MOBIUS) {
	  if (arg.xpay) arg.dagger ?
			  dslash5CPU<Float,nColor, true, true,DSLASH5_MOBIUS>(arg) :
			  dslash5CPU<Float,nColor,false, true,DSLASH5_MOBIUS>(arg);
	  else          arg.dagger ?
			  dslash5CPU<Float,nColor, true,false,DSLASH5_MOBIUS>(arg) :
			  dslash5CPU<Float,nColor,false,false,DSLASH5_MOBIUS>(arg);
	}
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (arg.type == DSLASH5_DWF) {
	  if (arg.xpay) arg.dagger ?
			  dslash5GPU<Float,nColor, true, true,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false, true,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5GPU<Float,nColor, true,false,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false,false,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	} else if (arg.type == DSLASH5_MOBIUS_PRE) {
	  if (arg.xpay) arg.dagger ?
			  dslash5GPU<Float,nColor, true, true,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false, true,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5GPU<Float,nColor, true,false,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false,false,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	} else if (arg.type == DSLASH5_MOBIUS) {
	  if (arg.xpay) arg.dagger ?
			  dslash5GPU<Float,nColor, true, true,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false, true,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5GPU<Float,nColor, true,false,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false,false,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	}
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };


  template <typename Float, int nColor>
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
		    double m_f, double m_5, const Complex *b_5, const Complex *c_5,
		    double a, bool dagger, Dslash5Type type)
  {
    Dslash5Arg<Float,nColor> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
    Dslash5<Float,nColor,Dslash5Arg<Float,nColor> > dslash(arg, in);
    dslash.apply(streams[Nstream-1]);
  }

  // template on the number of colors
  template <typename Float>
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
		    double m_f, double m_5, const Complex *b_5, const Complex *c_5,
		    double a, bool dagger, Dslash5Type type)
  {
    switch(in.Ncolor()) {
    case 3: ApplyDslash5<Float,3>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    default: errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

#endif

  //Apply the 5th dimension dslash operator to a colorspinor field
  //out = Dslash5*in
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
		    double m_f, double m_5, const Complex *b_5, const Complex *c_5,
		    double a, bool dagger, Dslash5Type type)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    if (in.DWFPCtype() != QUDA_4D_PC) errorQuda("Only 4-d preconditioned fields are supported");
    checkLocation(out, in);     // check all locations match

    switch(checkPrecision(out,in)) {
    case QUDA_DOUBLE_PRECISION: ApplyDslash5<double>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    case QUDA_SINGLE_PRECISION: ApplyDslash5<float> (out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    case QUDA_HALF_PRECISION:   ApplyDslash5<short> (out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    default: errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }

} // namespace quda


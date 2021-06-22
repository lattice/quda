#include <color_spinor_field.h>
#include <contract_quda.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <kernels/color_contract.cuh>
//#include <kernels/contraction.cuh>

namespace quda {

  // DMH TODO: Conglomerate these into a single class with a QUDA
  // param to denote either inner product or color contract.

  // Inner Product
  //----------------------------------------------------------------------------
  template <typename Float, int nColor> class InnerProduct : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    complex<Float> *result;

    unsigned int minThreads() const { return x.VolumeCB(); }
    
  public:
    InnerProduct(const ColorSpinorField &x, const ColorSpinorField &y, void *result) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      result(static_cast<complex<Float>*>(result))
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorContractArg<Float, nColor> arg(x, y, result);
      launch<InnerProd>(tp, stream, arg);
    }
    
    long long flops() const
    {
      // 1 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 1 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(Complex);
    }
  };

#ifdef GPU_CONTRACT
  void innerProductQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result)
  {
    checkPrecision(x, y);
    if (x.Nspin() != 1 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    instantiate<InnerProduct>(x, y, result);
  }
#else
  void innerProductQuda(const ColorSpinorField &, const ColorSpinorField &, void *)
  {
    errorQuda("Contraction code has not been built");
  }
#endif
  //----------------------------------------------------------------------------

  
  // Color Contract
  //----------------------------------------------------------------------------
  template <typename Float, int nColor> class ColorContract : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    complex<Float> *result;

    unsigned int minThreads() const { return x.VolumeCB(); }
    
  public:
    ColorContract(const ColorSpinorField &x, const ColorSpinorField &y, void *result) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      result(static_cast<complex<Float>*>(result))
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorContractArg<Float, nColor> arg(x, y, result);
      launch<ColorContraction>(tp, stream, arg);
    }

    long long flops() const
    {
      // 1 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 1 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(Complex);
    }
  };

#ifdef GPU_CONTRACT
  void colorContractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result)
  {
    checkPrecision(x, y);
    if (x.Nspin() != 1 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
   instantiate<ColorContract>(x, y, result);
  }
#else
  void colorContractQuda(const ColorSpinorField &, const ColorSpinorField &, void *)
  {
    errorQuda("Contraction code has not been built");
  }
#endif
  //----------------------------------------------------------------------------

  // Color Cross Product
  //----------------------------------------------------------------------------
  template <typename Float, int nColor> class ColorCross : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    ColorSpinorField &result;

    unsigned int minThreads() const { return x.VolumeCB(); }

  public:
    ColorCross(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      result(result)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorCrossArg<Float, nColor> arg(x, y, result);      
      launch<ColorCrossCompute>(tp, stream, arg);
    }
    
    long long flops() const
    {
      // 1 spin, 3 color, 6 complex, lattice volume
      return 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };

#if defined(GPU_CONTRACT) && (N_COLORS == 3)  
  void colorCrossQuda(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result)
  {
    checkPrecision(x, y);
    checkPrecision(result, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3 || result.Ncolor() != 3) errorQuda("Unexpected number of colors x = %d y = %d result = %d", x.Ncolor(), y.Ncolor(), result.Ncolor());
    if (x.Nspin() != 1 || y.Nspin() != 1 || result.Nspin() != 1) errorQuda("Unexpected number of spins x = %d y = %d result = %d", x.Nspin(), y.Nspin(), result.Nspin());

    instantiate<ColorCross>(x, y, result);

  }
#else
  void colorCrossQuda(const ColorSpinorField &, const ColorSpinorField &, ColorSpinorField &)
  {
    errorQuda("Contraction code has not been built");
  }
#endif
  //----------------------------------------------------------------------------

    // Color Contract with FT
  //----------------------------------------------------------------------------
  template <typename Float, int nColor> class MomentumProject : TunableMultiReduction<1>
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    complex<Float> *cc_array; // device vector with color contracted data
    std::vector<Complex> &mom_proj; // host vector for momentum mode data
    std::vector<int> &momenta;// host vector with momentum modes
    const int n_mom;

  public:
    MomentumProject(const ColorSpinorField &x, const ColorSpinorField &y,
		    void *cc_array, std::vector<Complex> &mom_proj,
		    std::vector<int> &momenta, const int n_mom) :
      TunableMultiReduction(x, n_mom * x.X()[3]),
      x(x),
      y(y),
      cc_array(static_cast<complex<Float>*>(cc_array)),
      mom_proj(mom_proj),
      momenta(momenta),
      n_mom(n_mom)
    {
      apply(device::get_default_stream());
    }
    
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      int t_dim = x.X()[3];
      std::vector<double> mom_proj_local(2 * t_dim);

      for(int k=0; k<n_mom; k++) {
        //printfQuda("Computing mom_mode k=%d\n", k);	
	// Zero out the local results.
	if(!activeTuning()) {
	  for(int i=0; i<2 * t_dim; i++) mom_proj_local[i] = 0.0;
	}
	const int mom[3] = {momenta[3*k + 0], momenta[3*k + 1], momenta[3*k + 2]};
	
	MomentumProjectArg<Float, nColor, 1> arg(x, cc_array, mom);
	launch<MomProj>(mom_proj_local, tp, stream, arg);
	
	// Copy results back to host array
	if(!activeTuning()) {
	  for(int i=0; i<t_dim; i++) {
	    mom_proj[k*t_dim + i].real(mom_proj_local[2*i]);
	    mom_proj[k*t_dim + i].imag(mom_proj_local[2*i+1]);
	  }
	}      
      }
    }    
    
    long long flops() const
    {
      // 1 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 1 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(Complex);
    }
  };

  //#ifdef GPU_CONTRACT
  void momentumProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y,
			   void *cc_array, std::vector<Complex> &mom_proj,
			   std::vector<int> &momenta, const int n_mom)
  {
    checkPrecision(x, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x = %d y = %d", x.Ncolor(), y.Ncolor());
    
    instantiate<InnerProduct>(x, y, cc_array);
    instantiate<MomentumProject>(x, y, cc_array, mom_proj, momenta, n_mom);
  }
  //#else
  //void momentumProjectQuda(const ColorSpinorField &, const ColorSpinorField &,
  //void *, const std::vector<int> &)
  //{
  //errorQuda("Contraction code has not been built");
  //}
  //#endif
  //----------------------------------------------------------------------------

}// namespace quda

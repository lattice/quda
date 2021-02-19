#include <color_spinor_field.h>
#include <contract_quda.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <kernels/contraction.cuh>

namespace quda {
  
  template <typename Float, int nColor> class EvecProject : TunableMultiReduction<1>
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    std::vector<Complex> &result;

  public:
    EvecProject(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result) :
      TunableMultiReduction(x, x.X()[3]),
      x(x),
      y(y),
      result(result)
    {
      apply(device::get_default_stream());
    }
    
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      const int array_size = x.Nspin() * x.X()[3];
      std::vector<double> result_local(2*array_size);
      
      // Zero out the local results.
      if(!activeTuning()) {
	for(int i=0; i<array_size; i++) {
	  result_local[2*i] = 0.0;
	  result_local[2*i+1] = 0.0;
	}
      }

      EvecProjectionArg<Float, nColor, 3> arg(x, y);
      launch<EvecProjection>(result_local, tp, stream, arg);
      
      // Copy results back to host array
      if(!activeTuning()) {
	for(int i=0; i<array_size; i++) {
	  result_global[comm_coord(3) * array_size + i].real(result_local[2*i]);
	  result_global[comm_coord(3) * array_size + i].imag(result_local[2*i+1]);
	}
      }
    }

    long long flops() const
    {
      // 4 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 4 * 3 * 6ll * x.Volume();
    }
    
    long long bytes() const
    {
      return x.Bytes() + y.Bytes();
    }
  };
  
  
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::complex<double> &result)
  {
    checkPrecision(x, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    instantiate<EvecProject>(x, y, result);
  }
} // namespace quda

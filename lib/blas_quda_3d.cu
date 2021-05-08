#include <color_spinor_field.h>
#include <contract_quda.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <kernels/blas_3d.cuh>

namespace quda {

  namespace blas3d {

    template <typename Float, int nColor> class copy3D : TunableKernel2D
    {
    protected:
      ColorSpinorField &x;
      const ColorSpinorField &y;
      const int reduction_dim;
      const int dim_slice;
      
      unsigned int minThreads() const { return x.VolumeCB(); }
      
    public:
      copy3D(ColorSpinorField &x, const ColorSpinorField &y, const int reduction_dim, const int dim_slice) :
	TunableKernel2D(x, x.SiteSubset()),
	x(x),
	y(y),
	reduction_dim(reduction_dim),
	dim_slice(dim_slice)
      {
	apply(device::get_default_stream());
      }
      
      void apply(const qudaStream_t &stream)
      {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	switch (reduction_dim) {
	case 3:
	  launch<copy3d>(tp, stream, copy3dArg<Float, nColor, 3>(x, y, dim_slice));
	  break;
	case 2:
	  launch<copy3d>(tp, stream, copy3dArg<Float, nColor, 2>(x, y, dim_slice));
	  break;
	case 1:
	  launch<copy3d>(tp, stream, copy3dArg<Float, nColor, 1>(x, y, dim_slice));
	  break;
	case 0:
	  launch<copy3d>(tp, stream, copy3dArg<Float, nColor, 0>(x, y, dim_slice));
	  break;
	default: errorQuda("Undefined reduction dim %d", reduction_dim);
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
    
    //#ifdef GPU_BLAS_3D
    void copy(const int dim, const int slice, ColorSpinorField &x, const ColorSpinorField &y)
    {      
      checkPrecision(x, y);
      
      // Check spins
      if (x.Nspin() != y.Nspin()) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
      
      // Check colors
      if (x.Ncolor() != y.Ncolor()) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
      
      // Check reduction dim
      if (x.X()[dim] != 1) errorQuda("Unexpected dimensions in x[%d]=%d", dim, x.X()[dim]);

      // Check slice value
      if (slice >= y.X()[dim]) errorQuda("Unexpected slice in y[%d]=%d", dim, y.X()[dim]);
      
      // We must give a Lattice field as the first argument
      instantiate<copy3D>(x, y, dim, slice);
    }
    
    // #else
    //     void axpby(const int, std::vector<double> &, ColorSpinorField &, std::vector<double> &, ColorSpinorField &) 
    //     {
    //       errorQuda("BLAS 3D code has not been built"); 
    //     }
    // #endif

    
    template <typename Float, int nColor> class axpby3D : TunableKernel2D
    {
    protected:
      ColorSpinorField &x;
      ColorSpinorField &y;
      void *a;
      void *b;
      const int reduction_dim;
      
      unsigned int minThreads() const { return x.VolumeCB(); }
      
    public:
      axpby3D(ColorSpinorField &x, ColorSpinorField &y, void *a, void *b, const int reduction_dim) :
	TunableKernel2D(x, x.SiteSubset()),
	x(x),
	y(y),
	a(a),
	b(b),
	reduction_dim(reduction_dim)
      {
	apply(device::get_default_stream());
      }
      
      void apply(const qudaStream_t &stream)
      {
	size_t data_bytes = x.X()[reduction_dim] *  x.Precision();
	void *d_a = pool_device_malloc(data_bytes);
	void *d_b = pool_device_malloc(data_bytes);
	qudaMemcpy(d_a, a, data_bytes, qudaMemcpyHostToDevice);
	qudaMemcpy(d_b, b, data_bytes, qudaMemcpyHostToDevice);
	
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	switch (reduction_dim) {
	case 3:
	  launch<axpby3d>(tp, stream, axpby3dArg<Float, nColor, 3>((Float*)d_a, x, (Float*)d_b, y));
	  break;
	case 2:
	  launch<axpby3d>(tp, stream, axpby3dArg<Float, nColor, 2>((Float*)d_a, x, (Float*)d_b, y));
	  break;
	case 1:
	  launch<axpby3d>(tp, stream, axpby3dArg<Float, nColor, 1>((Float*)d_a, x, (Float*)d_b, y));
	  break;
	case 0:
	  launch<axpby3d>(tp, stream, axpby3dArg<Float, nColor, 0>((Float*)d_a, x, (Float*)d_b, y));
	  break;
	default: errorQuda("Undefined reduction dim %d", reduction_dim);
	}
	
	qudaMemcpy(a, d_a, data_bytes, qudaMemcpyDeviceToHost);
	qudaMemcpy(b, d_b, data_bytes, qudaMemcpyDeviceToHost);
	pool_device_free(d_b);
	pool_device_free(d_a);
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
    
    //#ifdef GPU_BLAS_3D
    void axpby(const int dim, std::vector<double> &a, ColorSpinorField &x, std::vector<double> &b, ColorSpinorField &y)
    {
      
      checkPrecision(x, y);
      
      // Check spins
      if (x.Nspin() != y.Nspin()) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
      
      // Check colors
      if (x.Ncolor() != y.Ncolor()) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    
      // Check reduction dim
      if (x.X()[dim] != y.X()[dim]) errorQuda("Unexpected dimensions in x[%d]=%d and y[%d]=%d", dim, x.X()[dim], dim, y.X()[dim]);

      // Check coefficients
      if (a.size() != b.size() && a.size() != (unsigned int)x.X()[dim]) errorQuda("Unexpected coeff array sizes a=%lu b=%lu", a.size(), b.size());
      
      // We must give a Lattice field as the first argument
      instantiate<axpby3D>(x, y, a.data(), b.data(), dim);
    }
    
    // #else
    //     void axpby(const int, std::vector<double> &, ColorSpinorField &, std::vector<double> &, ColorSpinorField &) 
    //     {
    //       errorQuda("BLAS 3D code has not been built"); 
    //     }
    // #endif

    template <typename Float, int nColor> class caxpby3D : TunableKernel2D
    {
    protected:
      ColorSpinorField &x;
      ColorSpinorField &y;
      void *a;
      void *b;
      const int reduction_dim;
      
      unsigned int minThreads() const { return x.VolumeCB(); }
      
    public:
      caxpby3D(ColorSpinorField &x, ColorSpinorField &y, void *a, void *b, const int reduction_dim) :
	TunableKernel2D(x, x.SiteSubset()),
	x(x),
	y(y),
	a(a),
	b(b),
	reduction_dim(reduction_dim)
      {
	apply(device::get_default_stream());
      }
      
      void apply(const qudaStream_t &stream)
      {
	size_t data_bytes = 2 * x.X()[reduction_dim] *  x.Precision();
	void *d_a = pool_device_malloc(data_bytes);
	void *d_b = pool_device_malloc(data_bytes);
	qudaMemcpy(d_a, a, data_bytes, qudaMemcpyHostToDevice);
	qudaMemcpy(d_b, b, data_bytes, qudaMemcpyHostToDevice);
	
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	switch (reduction_dim) {
	case 3:
	  launch<caxpby3d>(tp, stream, caxpby3dArg<Float, nColor, 3>((complex<Float>*)d_a, x, (complex<Float>*)d_b, y));
	  break;
	case 2:
	  launch<caxpby3d>(tp, stream, caxpby3dArg<Float, nColor, 2>((complex<Float>*)d_a, x, (complex<Float>*)d_b, y));
	  break;
	case 1:
	  launch<caxpby3d>(tp, stream, caxpby3dArg<Float, nColor, 1>((complex<Float>*)d_a, x, (complex<Float>*)d_b, y));
	  break;
	case 0:
	  launch<caxpby3d>(tp, stream, caxpby3dArg<Float, nColor, 0>((complex<Float>*)d_a, x, (complex<Float>*)d_b, y));
	  break;
	default: errorQuda("Undefined reduction dim %d", reduction_dim);
	}
	
	qudaMemcpy(a, d_a, data_bytes, qudaMemcpyDeviceToHost);
	qudaMemcpy(b, d_b, data_bytes, qudaMemcpyDeviceToHost);
	pool_device_free(d_b);
	pool_device_free(d_a);
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
    
    //#ifdef GPU_BLAS_3D
    void caxpby(const int dim, std::vector<Complex> &a, ColorSpinorField &x, std::vector<Complex> &b, ColorSpinorField &y)
    {
      
      checkPrecision(x, y);
      
      // Check spins
      if (x.Nspin() != y.Nspin()) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
      
      // Check colors
      if (x.Ncolor() != y.Ncolor()) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    
      // Check reduction dim
      if (x.X()[dim] != y.X()[dim]) errorQuda("Unexpected dimensions in x[%d]=%d and y[%d]=%d", dim, x.X()[dim], dim, y.X()[dim]);

      // Check coefficients
      if (a.size() != b.size() && a.size() != (unsigned int)x.X()[dim]) errorQuda("Unexpected coeff array sizes a=%lu b=%lu", a.size(), b.size());
      
      // We must give a Lattice field as the first argument
      instantiate<caxpby3D>(x, y, a.data(), b.data(), dim);
    }
    
    // #else
    //     void axpby(const int, std::vector<double> &, ColorSpinorField &, std::vector<double> &, ColorSpinorField &) 
    //     {
    //       errorQuda("BLAS 3D code has not been built"); 
    //     }
    // #endif


    
    template <typename Float, int nColor> class reDotProduct3D : TunableMultiReduction<1>
    {
    protected:
      const ColorSpinorField &x;
      const ColorSpinorField &y;
      std::vector<double> &result;
      const int reduction_dim;
      
    public:
      reDotProduct3D(const ColorSpinorField &x, const ColorSpinorField &y,
		     std::vector<double> &result,
		     const int reduction_dim) :
	TunableMultiReduction(x, x.X()[reduction_dim]),
	x(x),
	y(y),
	result(result),
	reduction_dim(reduction_dim)
      {
	apply(device::get_default_stream());
      }
      
      void apply(const qudaStream_t &stream)
      {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	
	// Zero out the local results.
	std::vector<double> result_local(x.X()[reduction_dim]);
	if(!activeTuning()) {
	  for(int i=0; i<x.X()[reduction_dim]; i++) result_local[i] = 0.0;
	}
	
	// Pass the integer value of the redection dim as a template arg
	reDotProduct3dArg<Float, nColor, 3> arg(x, y);
	launch<reDotProduct3d>(result_local, tp, stream, arg);
	
	// Copy results back to host array
	if(!activeTuning()) {
	  for(int i=0; i<x.X()[reduction_dim]; i++) {
	    result[comm_coord(reduction_dim) * x.X()[reduction_dim] + i] = result_local[i];
	  }
	}
      }
    
      long long flops() const
      {
	return ((x.Nspin() * x.Nspin() * x.Ncolor() * 6ll) + (x.Nspin() * x.Nspin() * (x.Nspin() + x.Nspin()*x.Ncolor()))) * x.Volume();
      }
      
      long long bytes() const
      {
	return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
      }
    };
    
    //#ifdef GPU_CONTRACT
    void reDotProduct(const int dim, std::vector<double> &result, const ColorSpinorField &x, const ColorSpinorField &y)
    {
      // Check spins
      if (x.Nspin() != y.Nspin()) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
      
      // Check colors
      if (x.Ncolor() != y.Ncolor()) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
      
      // Check reduction dim
      if (x.X()[dim] != y.X()[dim]) errorQuda("Unexpected dimensions in x[%d]=%d and y[%d]=%d", dim, x.X()[dim], dim, y.X()[dim]);
      
      // Check coefficients
      if (result.size() != (unsigned int)x.X()[dim]) errorQuda("Unexpected coeff array size a=%lu", result.size());
      
      // We must give a Lattice field as the first argument
      instantiate<reDotProduct3D>(x, y, result, dim);
    }
    // #else
    //     void reDotProduct(const int, std::vector<double> &, const ColorSpinorField &, void *, const ColorSpinorField &) 
    //     {
    //       errorQuda("BLAS 3D code has not been built"); 
    //     }
    // #endif

    // #else
    //     void reDotProduct(const int, std::vector<double> &, const ColorSpinorField &, void *, const ColorSpinorField &) 
    //     {
    //       errorQuda("BLAS 3D code has not been built"); 
    //     }
    // #endif

    template <typename Float, int nColor> class cDotProduct3D : TunableMultiReduction<1>
    {
    protected:
      const ColorSpinorField &x;
      const ColorSpinorField &y;
      std::vector<Complex> &result;
      const int reduction_dim;
      
    public:
      cDotProduct3D(const ColorSpinorField &x, const ColorSpinorField &y,
		    std::vector<Complex> &result,
		    const int reduction_dim) :
	TunableMultiReduction(x, x.X()[reduction_dim]),
	x(x),
	y(y),
	result(result),
	reduction_dim(reduction_dim)
      {
	apply(device::get_default_stream());
      }
      
      void apply(const qudaStream_t &stream)
      {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	
	// Zero out the local results.
	std::vector<double> result_local(2*x.X()[reduction_dim]);
	if(!activeTuning()) {
	  for(int i=0; i<x.X()[reduction_dim]; i++) {
	    result_local[2*i] = 0.0;
	    result_local[2*i+1] = 0.0;
	  }
	}
	
	// Pass the integer value of the redection dim as a template arg
	cDotProduct3dArg<Float, nColor, 3> arg(x, y);
	launch<cDotProduct3d>(result_local, tp, stream, arg);
	
	// Copy results back to host array
	if(!activeTuning()) {
	  for(int i=0; i<x.X()[reduction_dim]; i++) {
	    result[comm_coord(reduction_dim) * x.X()[reduction_dim] + i].real(result_local[2*i]);
	    result[comm_coord(reduction_dim) * x.X()[reduction_dim] + i].imag(result_local[2*i+1]);
	  }
	}
      }
    
      long long flops() const
      {
	return ((x.Nspin() * x.Nspin() * x.Ncolor() * 6ll) + (x.Nspin() * x.Nspin() * (x.Nspin() + x.Nspin()*x.Ncolor()))) * x.Volume();
      }
      
      long long bytes() const
      {
	return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
      }
    };
    
    //#ifdef GPU_CONTRACT
    void cDotProduct(const int dim, std::vector<Complex> &result, const ColorSpinorField &x, const ColorSpinorField &y)
    {
      // Check spins
      if (x.Nspin() != y.Nspin()) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
      
      // Check colors
      if (x.Ncolor() != y.Ncolor()) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
      
      // Check reduction dim
      if (x.X()[dim] != y.X()[dim]) errorQuda("Unexpected dimensions in x[%d]=%d and y[%d]=%d", dim, x.X()[dim], dim, y.X()[dim]);
      
      // Check coefficients
      if (result.size() != (unsigned int)x.X()[dim]) errorQuda("Unexpected coeff array size a=%lu", result.size());
      
      // We must give a Lattice field as the first argument
      instantiate<cDotProduct3D>(x, y, result, dim);
    }
    
  } // blas3d
  
} // namespace quda


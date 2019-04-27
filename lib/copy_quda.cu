#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <register_traits.h>

// For kernels with precision conversion built in
#define checkSpinorLength(a, b)						\
  {									\
    if (a.Length() != b.Length())					\
      errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length()); \
    if (a.Stride() != b.Stride())					\
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());	\
    if (a.GammaBasis() != b.GammaBasis())				\
      errorQuda("gamma basis does not match: %d %d", a.GammaBasis(), b.GammaBasis());	\
  }

namespace quda {

  namespace blas {
    cudaStream_t* getStream();
    
    namespace copy_ns {

#include <texture.h>

    static struct {
      const char *vol_str;
      const char *aux_str;      
    } blasStrings;

    template <typename FloatN, int N, typename Output, typename Input>
    __global__ void copyKernel(Output Y, Input X, int length) {
      unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
      unsigned int parity = blockIdx.y;
      unsigned int gridSize = gridDim.x*blockDim.x;

      while (i < length) {
        FloatN x[N];
        X.load(x, i, parity);
        Y.save(x, i, parity);
        i += gridSize;
      }
    }
      
      template <typename FloatN, int N, typename Output, typename Input>
      class CopyCuda : public Tunable {

      private:
	Input &X;
	Output &Y;
	const int length;
	const int nParity;

	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

	virtual bool advanceSharedBytes(TuneParam &param) const
	{
	  TuneParam next(param);
	  advanceBlockDim(next); // to get next blockDim
	  int nthreads = next.block.x * next.block.y * next.block.z;
	  param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ?
	    sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
	  return false;
	}

    public:
	CopyCuda(Output &Y, Input &X, int length, int nParity)
	  : X(X), Y(Y), length(length/nParity), nParity(nParity) { }
      virtual ~CopyCuda() { ; }

      inline TuneKey tuneKey() const {
	return TuneKey(blasStrings.vol_str, "copyKernel", blasStrings.aux_str); 
      }

      inline void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	copyKernel<FloatN, N><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(Y, X, length);
      }

	void preTune() { ; } // no need to save state for copy kernels
	void postTune() { ; } // no need to restore state for copy kernels

	void initTuneParam(TuneParam &param) const {
	  Tunable::initTuneParam(param);
	  param.grid.y = nParity;
	}

	void defaultTuneParam(TuneParam &param) const {
	  Tunable::defaultTuneParam(param);
	  param.grid.y = nParity;
	}

	long long flops() const { return 0; }
	long long bytes() const { 
	  const int Ninternal = (sizeof(FloatN)/sizeof(((FloatN*)0)->x))*N;
	  size_t bytes = (X.Precision() + Y.Precision())*Ninternal;
	  if (X.Precision() == QUDA_HALF_PRECISION || X.Precision() == QUDA_QUARTER_PRECISION) bytes += sizeof(float);
	  if (Y.Precision() == QUDA_HALF_PRECISION || Y.Precision() == QUDA_QUARTER_PRECISION) bytes += sizeof(float);
	  return bytes*length*nParity;
	}
	int tuningIter() const { return 3; }
      };

      void copy(cudaColorSpinorField &dst, const cudaColorSpinorField &src) {
	if (&src == &dst) return; // aliasing fields
	
	if (src.SiteSubset() != dst.SiteSubset())
	  errorQuda("Spinor fields do not have matching subsets dst=%d src=%d\n", src.SiteSubset(), dst.SiteSubset());

	checkSpinorLength(dst, src);

	blasStrings.vol_str = src.VolString();
	char tmp[256];
	strcpy(tmp, "dst=");
	strcat(tmp, dst.AuxString());
	strcat(tmp, ",src=");
	strcat(tmp, src.AuxString());
	blasStrings.aux_str = tmp;
	
	if (dst.Nspin() != src.Nspin())
	  errorQuda("Spins (%d,%d) do not match", dst.Nspin(), src.Nspin());

	// For a given dst precision, there are two non-trivial possibilities for the
	// src precision.

	blas::bytes += (unsigned long long)src.RealLength()*(src.Precision() + dst.Precision());

	int partitions = (src.IsComposite() ? src.CompositeDim() : 1) * (src.SiteSubset());

	if (dst.Precision() == src.Precision()) {
	  if (src.Bytes() != dst.Bytes()) errorQuda("Precisions match, but bytes do not");
	  qudaMemcpyAsync(dst.V(), src.V(), dst.Bytes(), cudaMemcpyDeviceToDevice, *blas::getStream());
	  if (dst.Precision() == QUDA_HALF_PRECISION || dst.Precision() == QUDA_QUARTER_PRECISION) {
	    qudaMemcpyAsync(dst.Norm(), src.Norm(), dst.NormBytes(), cudaMemcpyDeviceToDevice, *blas::getStream());
	    blas::bytes += 2*(unsigned long long)dst.RealLength()*sizeof(float);
	  }
	} else if (dst.Precision() == QUDA_DOUBLE_PRECISION && src.Precision() == QUDA_SINGLE_PRECISION) {
	  if (src.Nspin() == 4){
	    Spinor<float4, float4, 6, 0> src_tex(src);
	    Spinor<float4, double2, 6, 1> dst_spinor(dst);
	    CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
              copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());	
	  } else if (src.Nspin() == 2) {
	    if (src.Length() != src.RealLength() || dst.Length() != dst.RealLength())
	      errorQuda("Non-zero stride not supported"); // we need to know how many colors to set "M" (requires JIT)
	    Spinor<float2, float2, 1, 0> src_tex(src);
	    Spinor<float2, double2, 1, 1> dst_spinor(dst);
	    CopyCuda<float2, 1, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Length()/2, partitions);
	    copy.apply(*blas::getStream());
	  } else if (src.Nspin() == 1) {
	    Spinor<float2, float2, 3, 0> src_tex(src);
	    Spinor<float2, double2, 3, 1> dst_spinor(dst);
	    CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());	
	  } else {
	    errorQuda("Nspin(%d) is not supported", src.Nspin());
	  }
	} else if (dst.Precision() == QUDA_SINGLE_PRECISION && src.Precision() == QUDA_DOUBLE_PRECISION) {
	  if (src.Nspin() == 4){
	    Spinor<float4, double2, 6, 0> src_tex(src);
	    Spinor<float4, float4, 6, 1> dst_spinor(dst);
	    CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	      copy.apply(*blas::getStream());
	  } else if (src.Nspin() == 2) {
	    if (src.Length() != src.RealLength() || dst.Length() != dst.RealLength())
	      errorQuda("Non-zero stride not supported"); // we need to know how many colors to set "M" (requires JIT)
	    Spinor<float2, double2, 1, 0> src_tex(src);
	    Spinor<float2, float2, 1, 1> dst_spinor(dst);
	    CopyCuda<float2, 1, decltype(dst_spinor), decltype(src_tex)>
              copy(dst_spinor, src_tex, src.Length()/2, partitions);
	    copy.apply(*blas::getStream());
	  } else if (src.Nspin() == 1) {
	    Spinor<float2, double2, 3, 0> src_tex(src);
	    Spinor<float2, float2, 3, 1> dst_spinor(dst);
	    CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());	
	  } else {
	    errorQuda("Nspin(%d) is not supported", src.Nspin());
	  }
	} else if (dst.Precision() == QUDA_SINGLE_PRECISION && src.Precision() == QUDA_HALF_PRECISION) {
	  blas::bytes += (unsigned long long)src.Volume()*sizeof(float);
	  if (src.Nspin() == 4){      
	    Spinor<float4, short4, 6, 0> src_tex(src);
	    Spinor<float4, float4, 6, 1> dst_spinor(dst);
	    CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	      copy.apply(*blas::getStream());
	  } else if (src.Nspin() == 1) {
	    Spinor<float2, short2, 3, 0> src_tex(src);
	    Spinor<float2, float2, 3, 1> dst_spinor(dst);
	    CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());
	  } else {
	    errorQuda("Nspin(%d) is not supported", src.Nspin());
	  }
	} else if (dst.Precision() == QUDA_HALF_PRECISION && src.Precision() == QUDA_SINGLE_PRECISION) {
	  blas::bytes += (unsigned long long)dst.Volume()*sizeof(float);
	  if (src.Nspin() == 4){
	    Spinor<float4, float4, 6, 0> src_tex(src);
	    Spinor<float4, short4, 6, 1> dst_spinor(dst);
	    CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());
	  } else if (src.Nspin() == 1) {
	    Spinor<float2, float2, 3, 0> src_tex(src);
	    Spinor<float2, short2, 3, 1> dst_spinor(dst);
	    CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());	
	  } else {
	    errorQuda("Nspin(%d) is not supported", src.Nspin());
	  }
	} else if (dst.Precision() == QUDA_DOUBLE_PRECISION && src.Precision() == QUDA_HALF_PRECISION) {
	  blas::bytes += (unsigned long long)src.Volume()*sizeof(float);
	  if (src.Nspin() == 4){
	    Spinor<double2, short4, 12, 0> src_tex(src);
	    Spinor<double2, double2, 12, 1> dst_spinor(dst);
	    CopyCuda<double2, 12, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());
	  } else if (src.Nspin() == 1) {
	    Spinor<double2, short2, 3, 0> src_tex(src);
	    Spinor<double2, double2, 3, 1> dst_spinor(dst);
	    CopyCuda<double2, 3, decltype(dst_spinor), decltype(src_tex)>
	    copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());
	  } else {
	    errorQuda("Nspin(%d) is not supported", src.Nspin());
	  }
	} else if (dst.Precision() == QUDA_HALF_PRECISION && src.Precision() == QUDA_DOUBLE_PRECISION) {
	  blas::bytes += (unsigned long long)dst.Volume()*sizeof(float);
	  if (src.Nspin() == 4){
	    Spinor<double2, double2, 12, 0> src_tex(src);
	    Spinor<double2, short4, 12, 1> dst_spinor(dst);
	    CopyCuda<double2, 12, decltype(dst_spinor), decltype(src_tex)>
	      copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());
	  } else if (src.Nspin() == 1) {
	    Spinor<double2, double2, 3, 0> src_tex(src);
	    Spinor<double2, short2, 3, 1> dst_spinor(dst);
	    CopyCuda<double2, 3, decltype(dst_spinor), decltype(src_tex)>
	    copy(dst_spinor, src_tex, src.Volume(), partitions);
	    copy.apply(*blas::getStream());
	  } else {
	    errorQuda("Nspin(%d) is not supported", src.Nspin());
	  }


  } else if (dst.Precision() == QUDA_HALF_PRECISION && src.Precision() == QUDA_QUARTER_PRECISION) {
    blas::bytes += (unsigned long long)src.Volume()*sizeof(float)*2;
    if (src.Nspin() == 4){      
      Spinor<float4, char4, 6, 0> src_tex(src);
      Spinor<float4, short4, 6, 1> dst_spinor(dst);
      CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
        copy.apply(*blas::getStream());
    } else if (src.Nspin() == 1) {
      Spinor<float2, char2, 3, 0> src_tex(src);
      Spinor<float2, short2, 3, 1> dst_spinor(dst);
      CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else {
      errorQuda("Nspin(%d) is not supported", src.Nspin());
    }
  } else if (dst.Precision() == QUDA_QUARTER_PRECISION && src.Precision() == QUDA_HALF_PRECISION) {
    blas::bytes += (unsigned long long)dst.Volume()*sizeof(float)*2;
    if (src.Nspin() == 4){
      Spinor<float4, short4, 6, 0> src_tex(src);
      Spinor<float4, char4, 6, 1> dst_spinor(dst);
      CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else if (src.Nspin() == 1) {
      Spinor<float2, short2, 3, 0> src_tex(src);
      Spinor<float2, char2, 3, 1> dst_spinor(dst);
      CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream()); 
    } else {
      errorQuda("Nspin(%d) is not supported", src.Nspin());
    }
  } else if (dst.Precision() == QUDA_SINGLE_PRECISION && src.Precision() == QUDA_QUARTER_PRECISION) {
    blas::bytes += (unsigned long long)src.Volume()*sizeof(float);
    if (src.Nspin() == 4){      
      Spinor<float4, char4, 6, 0> src_tex(src);
      Spinor<float4, float4, 6, 1> dst_spinor(dst);
      CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
        copy.apply(*blas::getStream());
    } else if (src.Nspin() == 1) {
      Spinor<float2, char2, 3, 0> src_tex(src);
      Spinor<float2, float2, 3, 1> dst_spinor(dst);
      CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else {
      errorQuda("Nspin(%d) is not supported", src.Nspin());
    }
  } else if (dst.Precision() == QUDA_QUARTER_PRECISION && src.Precision() == QUDA_SINGLE_PRECISION) {
    blas::bytes += (unsigned long long)dst.Volume()*sizeof(float);
    if (src.Nspin() == 4){
      Spinor<float4, float4, 6, 0> src_tex(src);
      Spinor<float4, char4, 6, 1> dst_spinor(dst);
      CopyCuda<float4, 6, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else if (src.Nspin() == 1) {
      Spinor<float2, float2, 3, 0> src_tex(src);
      Spinor<float2, char2, 3, 1> dst_spinor(dst);
      CopyCuda<float2, 3, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream()); 
    } else {
      errorQuda("Nspin(%d) is not supported", src.Nspin());
    }
  } else if (dst.Precision() == QUDA_DOUBLE_PRECISION && src.Precision() == QUDA_QUARTER_PRECISION) {
    blas::bytes += (unsigned long long)src.Volume()*sizeof(float);
    if (src.Nspin() == 4){
      Spinor<double2, char4, 12, 0> src_tex(src);
      Spinor<double2, double2, 12, 1> dst_spinor(dst);
      CopyCuda<double2, 12, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else if (src.Nspin() == 1) {
      Spinor<double2, char2, 3, 0> src_tex(src);
      Spinor<double2, double2, 3, 1> dst_spinor(dst);
      CopyCuda<double2, 3, decltype(dst_spinor), decltype(src_tex)>
      copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else {
      errorQuda("Nspin(%d) is not supported", src.Nspin());
    }
  } else if (dst.Precision() == QUDA_QUARTER_PRECISION && src.Precision() == QUDA_DOUBLE_PRECISION) {
    blas::bytes += (unsigned long long)dst.Volume()*sizeof(float);
    if (src.Nspin() == 4){
      Spinor<double2, double2, 12, 0> src_tex(src);
      Spinor<double2, char4, 12, 1> dst_spinor(dst);
      CopyCuda<double2, 12, decltype(dst_spinor), decltype(src_tex)>
        copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else if (src.Nspin() == 1) {
      Spinor<double2, double2, 3, 0> src_tex(src);
      Spinor<double2, char2, 3, 1> dst_spinor(dst);
      CopyCuda<double2, 3, decltype(dst_spinor), decltype(src_tex)>
      copy(dst_spinor, src_tex, src.Volume(), partitions);
      copy.apply(*blas::getStream());
    } else {
      errorQuda("Nspin(%d) is not supported", src.Nspin());
    }
  } else {
	  errorQuda("Invalid precision combination dst=%d and src=%d", dst.Precision(), src.Precision());
	}
      
	checkCudaError();
      }

    } // namespace copy_nw

    void copy(ColorSpinorField &dst, const ColorSpinorField &src) {
      if (dst.Location() == QUDA_CUDA_FIELD_LOCATION &&
	  src.Location() == QUDA_CUDA_FIELD_LOCATION) {
	copy_ns::copy(static_cast<cudaColorSpinorField&>(dst), 
		      static_cast<const cudaColorSpinorField&>(src));
      } else {
	dst = src;
      }
    }
  
  } // namespace blas
} // namespace quda

#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <clover_field.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>

namespace quda {

#ifdef GPU_CLOVER_DIRAC

  template<typename Float, typename Clover1, typename Clover2, typename Gauge>
  struct CloverTraceArg {
    Clover1 clover1;
    Clover2 clover2;
    Gauge gauge;
    Float coeff;

    CloverTraceArg(Clover1 &clover1, Clover2 &clover2, Gauge &gauge, Float coeff)
      : clover1(clover1), clover2(clover2), gauge(gauge), coeff(coeff) {}
  };


  template <typename Float, typename Arg>
    __device__ __host__ void cloverSigmaTraceCompute(Arg & arg, const int x, int parity)
    {

      Float A[72];
      if (parity==0) arg.clover1.load(A,x,parity);
      else arg.clover2.load(A,x,parity);

      // load the clover term into memory
      for (int mu=0; mu<4; mu++) {
	for (int nu=0; nu<mu; nu++) {

	  Matrix<complex<Float>,3> mat;
	  setZero(&mat);

	  Float diag[2][6];
	  complex<Float> tri[2][15];
	  const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
	  complex<Float> ctmp;

	  for (int ch=0; ch<2; ++ch) {
	    // factor of two is inherent to QUDA clover storage
	    for (int i=0; i<6; i++) diag[ch][i] = 2.0*A[ch*36+i];
	    for (int i=0; i<15; i++) tri[ch][idtab[i]] = complex<Float>(2.0*A[ch*36+6+2*i], 2.0*A[ch*36+6+2*i+1]);
	  }

	  // X, Y
	  if (nu == 0) {
	    if (mu == 1) {
	      for (int j=0; j<3; ++j) {
		mat(j,j).y = diag[0][j+3] + diag[1][j+3] - diag[0][j] - diag[1][j];
	      }

	      // triangular part
	      int jk=0;
	      for (int j=1; j<3; ++j) {
		int jk2 = (j+3)*(j+2)/2 + 3;
		for (int k=0; k<j; ++k) {
		  ctmp = tri[0][jk2] + tri[1][jk2] - tri[0][jk] - tri[1][jk];

		  mat(j,k).x = -ctmp.imag();
		  mat(j,k).y =  ctmp.real();
		  mat(k,j).x =  ctmp.imag();
		  mat(k,j).y =  ctmp.real();

		  jk++; jk2++;
		}
	      } // X Y

	    } else if (mu == 2) {

	      for (int j=0; j<3; ++j) {
		int jk = (j+3)*(j+2)/2;
		for (int k=0; k<3; ++k) {
		  int kj = (k+3)*(k+2)/2 + j;
		  mat(j,k) = conj(tri[0][kj]) - tri[0][jk] + conj(tri[1][kj]) - tri[1][jk];
		  jk++;
		}
	      } // X Z

	    } else if (mu == 3) {
	      for (int j=0; j<3; ++j) {
		int jk = (j+3)*(j+2)/2;
		for (int k=0; k<3; ++k) {
		  int kj = (k+3)*(k+2)/2 + j;
		  ctmp = conj(tri[0][kj]) + tri[0][jk] - conj(tri[1][kj]) - tri[1][jk];
		  mat(j,k).x = -ctmp.imag();
		  mat(j,k).y =  ctmp.real();
		  jk++;
		}
	      }
	    } // mu == 3 // X T
	  } else if (nu == 1) {
	    if (mu == 2) { // Y Z
	      for (int j=0; j<3; ++j) {
		int jk = (j+3)*(j+2)/2;
		for (int k=0; k<3; ++k) {
		  int kj = (k+3)*(k+2)/2 + j;
		  ctmp = conj(tri[0][kj]) + tri[0][jk] + conj(tri[1][kj]) + tri[1][jk];
		  mat(j,k).x =  ctmp.imag();
		  mat(j,k).y = -ctmp.real();
		  jk++;
		}
	      }
	    } else if (mu == 3){ // Y T
	      for (int j=0; j<3; ++j) {
		int jk = (j+3)*(j+2)/2;
		for (int k=0; k<3; ++k) {
		  int kj = (k+3)*(k+2)/2 + j;
		  mat(j,k) = conj(tri[0][kj]) - tri[0][jk] - conj(tri[1][kj]) + tri[1][jk];
		  jk++;
		}
	      }
	    } // mu == 3
	  } // nu == 1
	  else if (nu == 2){
	    if (mu == 3) {
	      for (int j=0; j<3; ++j) {
		mat(j,j).y = diag[0][j] - diag[0][j+3] - diag[1][j] + diag[1][j+3];
	      }
	      int jk=0;
	      for (int j=1; j<3; ++j) {
		int jk2 = (j+3)*(j+2)/2 + 3;
		for (int k=0; k<j; ++k) {
		  ctmp = tri[0][jk] - tri[0][jk2] - tri[1][jk] + tri[1][jk2];
		  mat(j,k).x = -ctmp.imag();
		  mat(j,k).y =  ctmp.real();

		  mat(k,j).x = ctmp.imag();
		  mat(k,j).y = ctmp.real();
		  jk++; jk2++;
		}
	      }
	    }
	  }

	  mat *= arg.coeff;
	  arg.gauge((mu-1)*mu/2 + nu, x, parity) = mat;
	} // nu
      } // mu

      return;
    }

  template<typename Float, typename Arg>
    void cloverSigmaTrace(Arg &arg)
    {
      for (int x=0; x<arg.clover1.volumeCB; x++) {
        cloverSigmaTraceCompute<Float,Arg>(arg, x, 1);
      }
      return;
    }


  template<typename Float, typename Arg>
    __global__ void cloverSigmaTraceKernel(Arg arg)
    {
      int idx = blockIdx.x*blockDim.x + threadIdx.x;
      if (idx >= arg.clover1.volumeCB) return;
      // odd parity
      cloverSigmaTraceCompute<Float,Arg>(arg, idx, 1);
    }

  template<typename Float, typename Arg>
    class CloverSigmaTrace : Tunable {
      Arg &arg;
      const GaugeField &meta;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune the shared memory
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.clover1.volumeCB; }

      public: 
      CloverSigmaTrace(Arg &arg, const GaugeField &meta)
        : arg(arg), meta(meta) {
	writeAuxString("stride=%d", arg.clover1.stride);
      }
      virtual ~CloverSigmaTrace() {;}

      void apply(const qudaStream_t &stream){
        if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	  TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          cloverSigmaTraceKernel<Float,Arg><<<tp.grid,tp.block,0>>>(arg);
        } else {
          cloverSigmaTrace<Float,Arg>(arg);
        }
      }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

      long long flops() const { return 0; } // Fix this
      long long bytes() const { return (arg.clover1.Bytes() + 6*arg.gauge.Bytes()) * arg.clover1.volumeCB; }

    }; // CloverSigmaTrace


  template<typename Float, typename Clover1, typename Clover2, typename Gauge>
  void computeCloverSigmaTrace(Clover1 clover1, Clover2 clover2, Gauge gauge,
			       const GaugeField &meta, Float coeff)
  {
    typedef CloverTraceArg<Float, Clover1, Clover2, Gauge> Arg;
    Arg arg(clover1, clover2, gauge, coeff);
    CloverSigmaTrace<Float, Arg> traceCompute(arg, meta);
    traceCompute.apply(0);
    return;
  }

  template<typename Float>
  void computeCloverSigmaTrace(GaugeField& gauge, const CloverField& clover, Float coeff){

    if(clover.isNative()) {
      typedef typename clover_mapper<Float>::type C;
      if (gauge.isNative()) {
	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  computeCloverSigmaTrace<Float>( C(clover,0), C(clover,1), G(gauge), gauge, coeff);
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  computeCloverSigmaTrace<Float>( C(clover,0), C(clover,1), G(gauge), gauge, coeff);
	} else {
	  errorQuda("Reconstruction type %d not supported", gauge.Reconstruct());
	}
      } else {
	errorQuda("Gauge order %d not supported", gauge.Order());
      }
    } else {
      errorQuda("clover order %d not supported", clover.Order());
    } // clover order

  }

#endif

  void computeCloverSigmaTrace(GaugeField& output, const CloverField& clover, double coeff) {

#ifdef GPU_CLOVER_DIRAC
    if (clover.Precision() == QUDA_SINGLE_PRECISION) {
      computeCloverSigmaTrace<float>(output, clover, static_cast<float>(coeff));
    } else if (clover.Precision() == QUDA_DOUBLE_PRECISION){
      computeCloverSigmaTrace<double>(output, clover, coeff);
    } else {
      errorQuda("Precision %d not supported", clover.Precision());
    }
#else
    errorQuda("Clover has not been built");
#endif

  }     


} // namespace quda

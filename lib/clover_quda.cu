#include <quda_matrix.h>
#include <tune_quda.h>
#include <clover_field.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <instantiate.h>

namespace quda {

  template <typename store_t_>
  struct CloverArg {
    using store_t = store_t_;
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = 3;
    static constexpr int nSpin = 4;

    using Clover = typename clover_mapper<store_t>::type;
    using Fmunu = typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO>::type;

    Clover clover;
    const Fmunu f;
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    real coeff;
    
    CloverArg(CloverField &clover, const GaugeField &f, double coeff) :
      clover(clover, 0),
      f(f),
      threads(f.VolumeCB()),
      coeff(coeff)
    { 
      for (int dir=0; dir<4; ++dir) X[dir] = f.X()[dir];
    }
  };

  /*
    Put into clover order
    Upper-left block (chirality index 0)
       /                                                                                \
       |  1 + c*I*(F[0,1] - F[2,3]) ,     c*I*(F[1,2] - F[0,3]) + c*(F[0,2] + F[1,3])   |
       |                                                                                |
       |  c*I*(F[1,2] - F[0,3]) - c*(F[0,2] + F[1,3]),   1 - c*I*(F[0,1] - F[2,3])      |
       |                                                                                |
       \                                                                                /

       /
       | 1 - c*I*(F[0] - F[5]),   -c*I*(F[2] - F[3]) - c*(F[1] + F[4])
       |
       |  -c*I*(F[2] -F[3]) + c*(F[1] + F[4]),   1 + c*I*(F[0] - F[5])
       |
       \

     Lower-right block (chirality index 1)

       /                                                                  \
       |  1 - c*I*(F[0] + F[5]),  -c*I*(F[2] + F[3]) - c*(F[1] - F[4])    |
       |                                                                  |
       |  -c*I*(F[2]+F[3]) + c*(F[1]-F[4]),     1 + c*I*(F[0] + F[5])     |
       \                                                                  /
  */
  // Core routine for constructing clover term from field strength
  template <typename Arg>
  __device__ __host__ void cloverComputeCore(Arg &arg, int x_cb, int parity)
  {
    constexpr int N = Arg::nColor*Arg::nSpin / 2;
    using real = typename Arg::real;
    using Complex = complex<real>;
    using Link = Matrix<Complex, Arg::nColor>;

    // Load the field-strength tensor from global memory
    Link F[6];
#pragma unroll
    for (int i=0; i<6; ++i) F[i] = arg.f(i, x_cb, parity);

    Complex I(0.0,1.0);
    Complex coeff(0.0, arg.coeff);
    Link block1[2], block2[2];
    block1[0] = coeff*(F[0]-F[5]); // (18 + 6*9=) 72 floating-point ops
    block1[1] = coeff*(F[0]+F[5]); // 72 floating-point ops
    block2[0] = arg.coeff*(F[1]+F[4] - I*(F[2]-F[3])); // 126 floating-point ops
    block2[1] = arg.coeff*(F[1]-F[4] - I*(F[2]+F[3])); // 126 floating-point ops

    // This uses lots of unnecessary memory
#pragma unroll
    for (int ch=0; ch<2; ++ch) {
      HMatrix<real,N> A;
      // c = 0(1) => positive(negative) chiral block
      // Compute real diagonal elements
#pragma unroll
      for (int i=0; i<N/2; ++i) {
	A(i+0,i+0) = 1.0 - block1[ch](i,i).real();
	A(i+3,i+3) = 1.0 + block1[ch](i,i).real();
      }

      // Compute off diagonal components
      // First row
      A(1,0) = -block1[ch](1,0);
      // Second row
      A(2,0) = -block1[ch](2,0);
      A(2,1) = -block1[ch](2,1);
      // Third row
      A(3,0) =  block2[ch](0,0);
      A(3,1) =  block2[ch](0,1);
      A(3,2) =  block2[ch](0,2);
      // Fourth row
      A(4,0) =  block2[ch](1,0);
      A(4,1) =  block2[ch](1,1);
      A(4,2) =  block2[ch](1,2);
      A(4,3) =  block1[ch](1,0);
      // Fifth row
      A(5,0) =  block2[ch](2,0);
      A(5,1) =  block2[ch](2,1);
      A(5,2) =  block2[ch](2,2);
      A(5,3) =  block1[ch](2,0);
      A(5,4) =  block1[ch](2,1);
      A *= static_cast<real>(0.5);

      arg.clover(x_cb, parity, ch) = A;
    } // ch
    // 84 floating-point ops
  }

  template <typename Arg> __global__ void cloverComputeKernel(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    if (x_cb >= arg.threads) return;
    cloverComputeCore(arg, x_cb, parity);
  }

  template <typename Arg> void cloverComputeCPU(Arg arg)
  {
    for (int parity = 0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.threads; x_cb++){
	cloverComputeCore(arg, x_cb, parity);
      }
    }
  }

  template <typename store_t>
  class CloverCompute : TunableVectorY {
    CloverArg<store_t> arg;
    const GaugeField &meta;
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

  public:
    CloverCompute(CloverField &clover, const GaugeField& f, double coeff) :
      TunableVectorY(2),
      arg(clover, f, coeff),
      meta(f)
    {
      checkNative(clover, f);
      writeAuxString("threads=%d,stride=%d,prec=%lu",arg.threads,arg.clover.stride,sizeof(store_t));

      apply(0);
      qudaDeviceSynchronize();
    }

    void apply(const qudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        qudaLaunchKernel(cloverComputeKernel<decltype(arg)>, tp, stream, arg);
      } else { // run the CPU code
        errorQuda("Not implemented");
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 2*arg.threads*480ll; }
    long long bytes() const { return 2*arg.threads*(6*arg.f.Bytes() + arg.clover.Bytes()); }
  };

  void computeClover(CloverField &clover, const GaugeField& f, double coeff)
  {
#ifdef GPU_CLOVER_DIRAC
    instantiate<CloverCompute>(clover, f, coeff);
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda


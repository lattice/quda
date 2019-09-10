#include <cstdio>
#include <cstdlib>
#include <staggered_oprod.h>

#include <tune_quda.h>
#include <quda_internal.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <dslash_quda.h>

namespace quda {

#ifdef GPU_STAGGERED_DIRAC

  namespace { // anonymous
#include <texture.h>
  }

  enum OprodKernelType { OPROD_INTERIOR_KERNEL, OPROD_EXTERIOR_KERNEL };

  template <typename Float, typename Output, typename InputA, typename InputB> struct StaggeredOprodArg {
    unsigned int length;
    const int parity;
    int dir;
    int displacement;
    OprodKernelType kernelType;
    const int nFace;
    const InputA inA;
    const InputB inB;
    Output outA;
    Output outB;
    Float coeff[2];
    int X[4];
    unsigned int ghostOffset[4];
    bool partitioned[4];

    StaggeredOprodArg(int parity, int dir, const unsigned int *ghostOffset, int displacement,
                      const OprodKernelType &kernelType, int nFace, const double coeff[2], InputA &inA, InputB &inB,
                      Output &outA, Output &outB, GaugeField &meta) :
      length(meta.VolumeCB()),
      parity(parity),
      dir(dir),
      displacement(displacement),
      kernelType(kernelType),
      nFace(nFace),
      inA(inA),
      inB(inB),
      outA(outA),
      outB(outB)
    {
      this->coeff[0] = coeff[0];
      this->coeff[1] = coeff[1];
      for (int i = 0; i < 4; ++i) this->X[i] = meta.X()[i];
      for (int i = 0; i < 4; ++i) this->ghostOffset[i] = ghostOffset[i];
      for (int i = 0; i < 4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
    }
  };

  enum IndexType {
    EVEN_X = 0,
    EVEN_Y = 1,
    EVEN_Z = 2,
    EVEN_T = 3
  };

  template <IndexType idxType>
  __device__ inline void coordsFromIndex(int &idx, int c[4], unsigned int cb_idx, int parity, const int X[4])
  {
      const int &LX = X[0];
      const int &LY = X[1];
      const int &LZ = X[2];
      const int XYZ = X[2]*X[1]*X[0];
      const int XY = X[1]*X[0];

      idx = 2*cb_idx;

      int x, y, z, t;

      if (idxType == EVEN_X /*!(LX & 1)*/) { // X even
        //   t = idx / XYZ;
        //   z = (idx / XY) % Z;
        //   y = (idx / X) % Y;
        //   idx += (parity + t + z + y) & 1;
        //   x = idx % X;
        // equivalent to the above, but with fewer divisions/mods:
        int aux1 = idx / LX;
        x = idx - aux1 * LX;
        int aux2 = aux1 / LY;
        y = aux1 - aux2 * LY;
        t = aux2 / LZ;
        z = aux2 - t * LZ;
        aux1 = (parity + t + z + y) & 1;
        x += aux1;
        idx += aux1;
      } else if (idxType == EVEN_Y /*!(LY & 1)*/) { // Y even
        t = idx / XYZ;
        z = (idx / XY) % LZ;
        idx += (parity + t + z) & 1;
        y = (idx / LX) % LY;
        x = idx % LX;
      } else if (idxType == EVEN_Z /*!(LZ & 1)*/) { // Z even
        t = idx / XYZ;
        idx += (parity + t) & 1;
        z = (idx / XY) % LZ;
        y = (idx / LX) % LY;
        x = idx % LX;
      } else {
        idx += parity;
        t = idx / XYZ;
        z = (idx / XY) % LZ;
        y = (idx / LX) % LY;
        x = idx % LX;
      }

      c[0] = x;
      c[1] = y;
      c[2] = z;
      c[3] = t;
    }
  

  // Get the  coordinates for the exterior kernels
    __device__ inline void coordsFromIndex(int x[4], unsigned int cb_idx, const int X[4], int dir, int displacement,
                                           int parity)
    {
      int Xh[2] = {X[0] / 2, X[1] / 2};
      switch (dir) {
      case 0:
        x[2] = cb_idx / Xh[1] % X[2];
        x[3] = cb_idx / (Xh[1] * X[2]) % X[3];
        x[0] = cb_idx / (Xh[1] * X[2] * X[3]);
        x[0] += (X[0] - displacement);
        x[1] = 2 * (cb_idx % Xh[1]) + ((x[0] + x[2] + x[3] + parity) & 1);
        break;

      case 1:
        x[2] = cb_idx / Xh[0] % X[2];
        x[3] = cb_idx / (Xh[0] * X[2]) % X[3];
        x[1] = cb_idx / (Xh[0] * X[2] * X[3]);
        x[1] += (X[1] - displacement);
        x[0] = 2 * (cb_idx % Xh[0]) + ((x[1] + x[2] + x[3] + parity) & 1);
        break;

      case 2:
        x[1] = cb_idx / Xh[0] % X[1];
        x[3] = cb_idx / (Xh[0] * X[1]) % X[3];
        x[2] = cb_idx / (Xh[0] * X[1] * X[3]);
        x[2] += (X[2] - displacement);
        x[0] = 2 * (cb_idx % Xh[0]) + ((x[1] + x[2] + x[3] + parity) & 1);
        break;

      case 3:
        x[1] = cb_idx / Xh[0] % X[1];
        x[2] = cb_idx / (Xh[0] * X[1]) % X[2];
        x[3] = cb_idx / (Xh[0] * X[1] * X[2]);
        x[3] += (X[3] - displacement);
        x[0] = 2 * (cb_idx % Xh[0]) + ((x[1] + x[2] + x[3] + parity) & 1);
        break;
      }
      return;
  }

  __device__ inline int neighborIndex(unsigned int cb_idx, const int shift[4], const bool partitioned[4], int parity,
                                      const int X[4])
  {
    int full_idx;
    int x[4]; 

    coordsFromIndex<EVEN_X>(full_idx, x, cb_idx, parity, X);
    
    for(int dim = 0; dim<4; ++dim){
      if( partitioned[dim] )
	if( (x[dim]+shift[dim])<0 || (x[dim]+shift[dim])>=X[dim]) return -1;
    }

    for(int dim=0; dim<4; ++dim){
      x[dim] = shift[dim] ? (x[dim]+shift[dim] + X[dim]) % X[dim] : x[dim];
    }
    return (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  }

  template<typename real, typename Output, typename InputA, typename InputB>
  __global__ void interiorOprodKernel(StaggeredOprodArg<real, Output, InputA, InputB> arg)
    {
      unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
      const unsigned int gridSize = gridDim.x*blockDim.x;

      typedef complex<real> Complex;
      Complex x[3];
      Complex y[3];
      Complex z[3];
      Matrix<Complex,3> result;
      Matrix<Complex,3> tempA, tempB; // input

      while(idx<arg.length){
        arg.inA.load(x, idx);

#pragma unroll
        for(int dim=0; dim<4; ++dim){
          int shift[4] = {0,0,0,0};
          shift[dim] = 1;
          const int first_nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
          if(first_nbr_idx >= 0){
            arg.inB.load(y, first_nbr_idx);
            outerProd(y,x,&result);
            arg.outA.load(reinterpret_cast<real*>(tempA.data), idx, dim, arg.parity); 
            result = tempA + result*arg.coeff[0];
    
	    arg.outA.save(reinterpret_cast<real*>(result.data), idx, dim, arg.parity); 

	    if (arg.nFace == 3) {
	      shift[dim] = 3;
	      const int third_nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
	      if(third_nbr_idx >= 0){
		arg.inB.load(z, third_nbr_idx);
		outerProd(z, x, &result);
		arg.outB.load(reinterpret_cast<real*>(tempB.data), idx, dim, arg.parity); 
		result = tempB + result*arg.coeff[1];
		arg.outB.save(reinterpret_cast<real*>(result.data), idx, dim, arg.parity); 
	      }
	    }
          }
        } // dim

        idx += gridSize;
      }
      return;
    } // interiorOprodKernel


  template<int dim, typename real, typename Output, typename InputA, typename InputB>
  __global__ void exteriorOprodKernel(StaggeredOprodArg<real, Output, InputA, InputB> arg)
    {
      typedef complex<real> Complex;

      unsigned int cb_idx = blockIdx.x*blockDim.x + threadIdx.x;
      const unsigned int gridSize = gridDim.x*blockDim.x;

      Complex a[3];
      Complex b[3];
      Matrix<Complex,3> result;
      Matrix<Complex,3> inmatrix; // input

      Output& out = (arg.displacement == 1) ? arg.outA : arg.outB;
      real coeff = (arg.displacement == 1) ? arg.coeff[0] : arg.coeff[1];

      int x[4];
      while(cb_idx<arg.length){
        coordsFromIndex(x, cb_idx, arg.X, arg.dir, arg.displacement, arg.parity); 
        const unsigned int bulk_cb_idx = ((((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0]) >> 1);

        out.load(reinterpret_cast<real*>(inmatrix.data), bulk_cb_idx, arg.dir, arg.parity); 
        arg.inA.load(a, bulk_cb_idx);

        const unsigned int ghost_idx = arg.ghostOffset[dim] + cb_idx;
        arg.inB.loadGhost(b, ghost_idx, arg.dir);

        outerProd(b,a,&result);
        result = inmatrix + result*coeff; 
        out.save(reinterpret_cast<real*>(result.data), bulk_cb_idx, arg.dir, arg.parity); 

        cb_idx += gridSize;
      }
      return;
    }


  template<typename Float, typename Output, typename InputA, typename InputB>
  class StaggeredOprodField : public Tunable {

  private:
    StaggeredOprodArg<Float,Output,InputA,InputB> &arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.outA.volumeCB; }
    bool tunedGridDim() const { return false; }

  public:
    StaggeredOprodField(StaggeredOprodArg<Float,Output,InputA,InputB> &arg, const GaugeField &meta)
      : arg(arg), meta(meta) {
      writeAuxString("threads=%d,prec=%lu,stride=%d",arg.length,sizeof(Complex)/2,arg.inA.Stride());
      // this sets the communications pattern for the packing kernel
      int comms[QUDA_MAX_DIM] = { commDimPartitioned(0), commDimPartitioned(1), commDimPartitioned(2), commDimPartitioned(3) };
      setPackComms(comms);
    }

    virtual ~StaggeredOprodField() {}

    void apply(const cudaStream_t &stream){
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	// Disable tuning for the time being
	TuneParam tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());
	if (arg.kernelType == OPROD_INTERIOR_KERNEL) {
	  interiorOprodKernel<<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	} else if (arg.kernelType == OPROD_EXTERIOR_KERNEL) {
	       if (arg.dir == 0) exteriorOprodKernel<0><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 1) exteriorOprodKernel<1><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 2) exteriorOprodKernel<2><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 3) exteriorOprodKernel<3><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	} else {
	  errorQuda("Kernel type not supported\n");
	}
      } else { // run the CPU code
	errorQuda("No CPU support for staggered outer-product calculation\n");
      }
    } // apply

    void preTune(){ this->arg.outA.save(); this->arg.outB.save(); }
    void postTune(){ this->arg.outA.load(); this->arg.outB.load(); }
  
    long long flops() const { return 0; } // FIXME
    long long bytes() const { return 0; } // FIXME
    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux);}
  }; // StaggeredOprodField


  void exchangeGhost(int nFace, cudaColorSpinorField &a, int parity, int dag) {
    // need to enable packing in temporal direction to get spin-projector correct
    pushKernelPackT(true);

    // first transfer src1
    qudaDeviceSynchronize();

    MemoryLocation location[2*QUDA_MAX_DIM] = {Device, Device, Device, Device, Device, Device, Device, Device};
    a.pack(nFace, 1-parity, dag, Nstream-1, location, Device);

    qudaDeviceSynchronize();

    for(int i=3; i>=0; i--){
      if(commDimPartitioned(i)){
	// Initialize the host transfer from the source spinor
	a.gather(nFace, dag, 2*i);
      } // commDim(i)
    } // i=3,..,0

    qudaDeviceSynchronize(); comm_barrier();

    for (int i=3; i>=0; i--) {
      if(commDimPartitioned(i)) {
	a.commsStart(nFace, 2*i, dag);
      }
    }

    for (int i=3; i>=0; i--) {
      if(commDimPartitioned(i)) {
	a.commsWait(nFace, 2*i, dag);
	a.scatter(nFace, dag, 2*i);
      }
    }

    qudaDeviceSynchronize();
    popKernelPackT(); // restore packing state

    a.bufferIndex = (1 - a.bufferIndex);
    comm_barrier();
  }

  template <typename Float, typename Output, typename InputA, typename InputB>
  void computeStaggeredOprodCuda(Output outA, Output outB, GaugeField &outFieldA, GaugeField &outFieldB, InputA &inA,
                                 InputB &inB, cudaColorSpinorField &src, int parity, const int faceVolumeCB[4],
                                 const double coeff[2], int nFace)
  {
    unsigned int ghostOffset[4] = {0, 0, 0, 0};
    for (int dir = 0; dir < 4; ++dir)
      ghostOffset[dir] = src.GhostOffset(dir, 1) / src.FieldOrder(); // offset we want is the forwards one

    // Create the arguments for the interior kernel
    StaggeredOprodArg<Float, Output, InputA, InputB> arg(parity, 0, ghostOffset, 1, OPROD_INTERIOR_KERNEL, nFace, coeff,
                                                         inA, inB, outA, outB, outFieldA);
    StaggeredOprodField<Float, Output, InputA, InputB> oprod(arg, outFieldA);

    arg.kernelType = OPROD_INTERIOR_KERNEL;
    arg.length = src.VolumeCB();
    oprod.apply(streams[Nstream - 1]);

    for (int i = 3; i >= 0; i--) {
      if (commDimPartitioned(i)) {
        // update parameters for this exterior kernel
        arg.kernelType = OPROD_EXTERIOR_KERNEL;
        arg.dir = i;

        // First, do the one hop term
        {
          arg.displacement = 1;
          arg.length = faceVolumeCB[i];
          oprod.apply(streams[Nstream - 1]);
        }

        // Now do the 3 hop term
        if (nFace == 3) {
          arg.displacement = 3;
          arg.length = arg.displacement * faceVolumeCB[i];
          oprod.apply(streams[Nstream - 1]);
        }
      }
    } // i=3,..,0

    checkCudaError();
    } // computeStaggeredOprodCuda

#endif // GPU_STAGGERED_DIRAC

    void computeStaggeredOprod(GaugeField &outA, GaugeField &outB, ColorSpinorField &inEven, ColorSpinorField &inOdd,
                               int parity, const double coeff[2], int nFace)
    {
#ifdef GPU_STAGGERED_DIRAC
    if(outA.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", outA.Order());    

    if(outB.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", outB.Order());    

    if(inEven.Precision() != outA.Precision()) errorQuda("Mixed precision not supported: %d %d\n", inEven.Precision(), outA.Precision());

    cudaColorSpinorField &inA = (parity&1) ? static_cast<cudaColorSpinorField&>(inOdd) : static_cast<cudaColorSpinorField&>(inEven);
    cudaColorSpinorField &inB = (parity&1) ? static_cast<cudaColorSpinorField&>(inEven) : static_cast<cudaColorSpinorField&>(inOdd);

    inA.allocateGhostBuffer(nFace);
    inB.allocateGhostBuffer(nFace);

    if (inEven.Precision() == QUDA_DOUBLE_PRECISION) {
      Spinor<double2, double2, 3, 0> spinorA(inA, nFace);
      Spinor<double2, double2, 3, 0> spinorB(inB, nFace);
      exchangeGhost(nFace,static_cast<cudaColorSpinorField&>(inB), parity, 0);

      computeStaggeredOprodCuda<double>(gauge::FloatNOrder<double, 18, 2, 18>(outA), gauge::FloatNOrder<double, 18, 2, 18>(outB),
					outA, outB, spinorA, spinorB, inB, parity, inB.GhostFace(), coeff, nFace);
    } else if (inEven.Precision() == QUDA_SINGLE_PRECISION) {
      Spinor<float2, float2, 3, 0> spinorA(inA, nFace);
      Spinor<float2, float2, 3, 0> spinorB(inB, nFace);
      exchangeGhost(nFace,static_cast<cudaColorSpinorField&>(inB), parity, 0);

      computeStaggeredOprodCuda<float>(gauge::FloatNOrder<float, 18, 2, 18>(outA), gauge::FloatNOrder<float, 18, 2, 18>(outB),
				       outA, outB, spinorA, spinorB, inB, parity, inB.GhostFace(), coeff, nFace);
    } else {
      errorQuda("Unsupported precision: %d\n", inEven.Precision());
    }

#else // GPU_STAGGERED_DIRAC not defined
    errorQuda("Staggered Outer Product has not been built!"); 
#endif

    return;
  } // computeStaggeredOprod

  void computeStaggeredOprod(GaugeField *out[], ColorSpinorField& in, const double coeff[], int nFace)
  {
    if (nFace == 1) {
      computeStaggeredOprod(*out[0], *out[0], in.Even(), in.Odd(), 0, coeff, nFace);
      double coeff_[2] = {-coeff[0],0.0}; // need to multiply by -1 on odd sites
      computeStaggeredOprod(*out[0], *out[0], in.Even(), in.Odd(), 1, coeff_, nFace);
    } else if (nFace == 3) {
      computeStaggeredOprod(*out[0], *out[1], in.Even(), in.Odd(), 0, coeff, nFace);
      computeStaggeredOprod(*out[0], *out[1], in.Even(), in.Odd(), 1, coeff, nFace);
    } else {
      errorQuda("Invalid nFace=%d", nFace);
    }
  }

} // namespace quda

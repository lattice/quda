#include <cstdlib>
#include <cstdio>
#include <string>

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>

#define PRESERVE_SPINOR_NORM

#ifdef PRESERVE_SPINOR_NORM // Preserve the norm regardless of basis
#define kP (1.0/sqrt(2.0))
#define kU (1.0/sqrt(2.0))
#else // More numerically accurate not to preserve the norm between basis
#define kP (0.5)
#define kU (1.0)
#endif



namespace quda {

  using namespace colorspinor;
  
  void exchangeExtendedGhost(cudaColorSpinorField* spinor, int R[], int parity, qudaStream_t *stream_p)
  {
#ifdef MULTI_GPU
    int nFace = 0;
    for(int i=0; i<4; i++){
      if(R[i] > nFace) nFace = R[i];
    }

    int dagger = 0;

    int gatherCompleted[2] = {0,0};
    int commsCompleted[2] = {0,0};

    cudaEvent_t gatherEnd[2];
    for(int dir=0; dir<2; dir++) cudaEventCreate(&gatherEnd[dir], cudaEventDisableTiming);

    for(int dim=3; dim<=0; dim--){
      if(!commDim(dim)) continue;

      spinor->packExtended(nFace, R, parity, dagger, dim, stream_p); // packing in the dim dimension complete
      qudaDeviceSynchronize(); // Need this since packing is performed in stream[Nstream-1]
      for(int dir=1; dir<=0; dir--){
        spinor->gather(nFace, dagger, 2*dim + dir);
        qudaEventRecord(gatherEnd[dir], streams[2*dim+dir]); // gatherEnd[1], gatherEnd[0]
      } 

      int completeSum = 0;
      int dir = 1;
      while(completeSum < 2){
        if(!gatherCompleted[dir]){
          if(cudaSuccess == cudaEventQuery(gatherEnd[dir])){
            spinor->commsStart(nFace, 2*dim+dir, dagger);
            completeSum++;
            gatherCompleted[dir--] = 1;
          }
        }
      }
      gatherCompleted[0] = gatherCompleted[1] = 0;

      // Query if comms has completed
      dir = 1;
      while(completeSum < 4){
        if(!commsCompleted[dir]){
          if(spinor->commsQuery(nFace, 2*dim+dir, dagger)){
            spinor->scatterExtended(nFace, parity, dagger, 2*dim+dir);
            completeSum++;
            commsCompleted[dir--] = 1;
          }
        }
      } 
      commsCompleted[0] = commsCompleted[1] = 0;
      qudaDeviceSynchronize(); // Wait for scatters to complete before next iteration
    } // loop over dim

    for(int dir=0; dir<2; dir++) cudaEventDestroy(gatherEnd[dir]);
#endif
    return;
  }


  /** Straight copy with no basis change */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    class PreserveBasis {
      typedef typename mapper<FloatIn>::type RegTypeIn;
      typedef typename mapper<FloatOut>::type RegTypeOut;
      public:
      __device__ __host__ inline void operator()(ColorSpinor<RegTypeOut,Nc,Ns> &out, const ColorSpinor<RegTypeIn,Nc,Ns> &in) {
        for (int s=0; s<Ns; s++) {
          for (int c=0; c<Nc; c++) {
            out(s,c) = in(s,c);
          }
        }
      }
    };

  /** Transform from relativistic into non-relavisitic basis */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    struct NonRelBasis {
      typedef typename mapper<FloatIn>::type RegTypeIn;
      typedef typename mapper<FloatOut>::type RegTypeOut;
      __device__ __host__ inline void operator()(ColorSpinor<RegTypeOut,Nc,Ns> &out, const ColorSpinor<RegTypeIn,Nc,Ns> &in) {
        int s1[4] = {1, 2, 3, 0};
        int s2[4] = {3, 0, 1, 2};
        RegTypeOut K1[4] = {static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(-kP),
			    static_cast<RegTypeOut>(-kP), static_cast<RegTypeOut>(-kP)};
        RegTypeOut K2[4] = {static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(-kP),
			    static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(kP)};
        for (int s=0; s<Ns; s++) {
          for (int c=0; c<Nc; c++) {
            out(s,c).real(K1[s]*in(s1[s],c).real() + K2[s]*in(s2[s],c).real());
            out(s,c).imag(K1[s]*in(s1[s],c).imag() + K2[s]*in(s2[s],c).imag());
          }
        }
      }
    };

  /** Transform from non-relativistic into relavisitic basis */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    struct RelBasis {
      typedef typename mapper<FloatIn>::type RegTypeIn;
      typedef typename mapper<FloatOut>::type RegTypeOut;
      __device__ __host__ inline void operator()(ColorSpinor<RegTypeOut,Nc,Ns> &out, const ColorSpinor<RegTypeIn,Nc,Ns> &in) {
        int s1[4] = {1, 2, 3, 0};
        int s2[4] = {3, 0, 1, 2};
        RegTypeOut K1[4] = {static_cast<RegTypeOut>(-kU), static_cast<RegTypeOut>(kU),
			    static_cast<RegTypeOut>(kU), static_cast<RegTypeOut>(kU)};
        RegTypeOut K2[4] = {static_cast<RegTypeOut>(-kU), static_cast<RegTypeOut>(kU),
			    static_cast<RegTypeOut>(-kU), static_cast<RegTypeOut>(-kU)};
        for (int s=0; s<Ns; s++) {
          for (int c=0; c<Nc; c++) {
            out(s,c).real(K1[s]*in(s1[s],c).real() + K2[s]*in(s2[s],c).real());
            out(s,c).imag(K1[s]*in(s1[s],c).imag() + K2[s]*in(s2[s],c).imag());
          }
        }
      }
    };

  template<typename OutOrder, typename InOrder, typename Basis>
    struct CopySpinorExArg{
      OutOrder out;
      const InOrder in;
      Basis basis;
      int E[QUDA_MAX_DIM];
      int X[QUDA_MAX_DIM];
      int length;
      int parity;

      CopySpinorExArg(const OutOrder &out, const InOrder &in, const Basis& basis, const int *E, const int *X, const int parity)
        : out(out), in(in), basis(basis), parity(parity) 
      {
        this->length = 1;
        for(int d=0; d<4; d++){
          this->E[d] = E[d];
          this->X[d] = X[d];
          this->length *= X[d]; // smaller volume
        }
      }
    };


  template<typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis, bool extend>
    __device__ __host__ void copyInterior(CopySpinorExArg<OutOrder,InOrder,Basis>& arg, int X)
    {
      int x[4];
      int R[4];
      for(int d=0; d<4; d++) R[d] = (arg.E[d] - arg.X[d]) >> 1;

      int za = X/(arg.X[0]/2);
      int x0h = X - za*(arg.X[0]/2);
      int zb = za/arg.X[1];
      x[1] = za - zb*arg.X[1];
      x[3] = zb / arg.X[2];
      x[2] = zb - x[3]*arg.X[2];
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + arg.parity) & 1);

      // Y is the cb spatial index into the extended gauge field
      int Y = ((((x[3]+R[3])*arg.E[2] + (x[2]+R[2]))*arg.E[1] + (x[1]+R[1]))*arg.E[0]+(x[0]+R[0])) >> 1;

      typedef typename mapper<FloatIn>::type RegTypeIn;
      typedef typename mapper<FloatOut>::type RegTypeOut;

      ColorSpinor<RegTypeIn,Nc,Ns> in;
      ColorSpinor<RegTypeOut,Nc,Ns> out;
      int parity = 0;

      if(extend){
        in = arg.in(X, parity);
        arg.basis(out, in);
        arg.out(Y, parity) = out;
      }else{
        in = arg.in(Y, parity);
        arg.basis(out, in);
        arg.out(Y, parity) = out;
      }
    }


  template<typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis, bool extend>
    __global__ void copyInteriorKernel(CopySpinorExArg<OutOrder,InOrder,Basis> arg)
    {
      int cb_idx = blockIdx.x*blockDim.x + threadIdx.x;

      while(cb_idx < arg.length){
        copyInterior<FloatOut,FloatIn,Ns,Nc,OutOrder,InOrder,Basis,extend>(arg,cb_idx);
        cb_idx += gridDim.x*blockDim.x;
      }
    }

  /*
     Host function
   */
  template<typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis, bool extend>
    void copyInterior(CopySpinorExArg<OutOrder,InOrder,Basis>& arg)
    {
      for(int cb_idx=0; cb_idx<arg.length; cb_idx++){
        copyInterior<FloatOut,FloatIn,Ns,Nc,OutOrder,InOrder,Basis,extend>(arg, cb_idx);
      }
    }




  template<typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis, bool extend>
    class CopySpinorEx : Tunable {

      CopySpinorExArg<OutOrder,InOrder,Basis> arg;
      const ColorSpinorField &meta;
      QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
      bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.length; }

      public: 
      CopySpinorEx(CopySpinorExArg<OutOrder,InOrder,Basis> &arg, const ColorSpinorField &meta, QudaFieldLocation location)
        : arg(arg), meta(meta), location(location) {
	writeAuxString("out_stride=%d,in_stride=%d",arg.out.stride,arg.in.stride);
      }
      virtual ~CopySpinorEx() {}

      void apply(const qudaStream_t &stream){
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if(location == QUDA_CPU_FIELD_LOCATION){
          copyInterior<FloatOut,FloatIn,Ns,Nc,OutOrder,InOrder,Basis,extend>(arg);    
        }else if(location == QUDA_CUDA_FIELD_LOCATION){
          copyInteriorKernel<FloatOut,FloatIn,Ns,Nc,OutOrder,InOrder,Basis,extend>
            <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);    
        }
      } 

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

      long long flops() const { return 0; }
      long long bytes() const {
        return arg.length*2*Nc*Ns*(sizeof(FloatIn) + sizeof(FloatOut));
      }

    }; // CopySpinorEx



  template<typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
    void copySpinorEx(OutOrder outOrder, const InOrder inOrder, const Basis basis, const int *E, 
		      const int *X, const int parity, const bool extend, const ColorSpinorField &meta, QudaFieldLocation location)
    {
      CopySpinorExArg<OutOrder,InOrder,Basis> arg(outOrder, inOrder, basis, E, X, parity);
      if(extend){
        CopySpinorEx<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, Basis, true> copier(arg, meta, location);
        copier.apply(0);
      }else{
        CopySpinorEx<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, Basis, false> copier(arg, meta, location);
        copier.apply(0);
      }
      if(location == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
    }

  template<typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder>
    void copySpinorEx(OutOrder outOrder, InOrder inOrder, const QudaGammaBasis outBasis, const QudaGammaBasis inBasis,
		      const int* E, const int* X, const int parity, const bool extend, 
		      const ColorSpinorField &meta, QudaFieldLocation location)
    {
      if(inBasis == outBasis){
        PreserveBasis<FloatOut,FloatIn,Ns,Nc> basis;
        copySpinorEx<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, PreserveBasis<FloatOut,FloatIn,Ns,Nc> >
          (outOrder, inOrder, basis, E, X, parity, extend, meta, location);
      }else if(outBasis == QUDA_UKQCD_GAMMA_BASIS && inBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS){
        if(Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
        NonRelBasis<FloatOut,FloatIn,4,Nc> basis;
        copySpinorEx<FloatOut, FloatIn, 4, Nc, OutOrder, InOrder, NonRelBasis<FloatOut,FloatIn,4,Nc> >
          (outOrder, inOrder, basis, E, X, parity, extend, meta, location);
      }else if(inBasis == QUDA_UKQCD_GAMMA_BASIS && outBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS){
        if(Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
        RelBasis<FloatOut,FloatIn,4,Nc> basis;
        copySpinorEx<FloatOut, FloatIn, 4, Nc, OutOrder, InOrder, RelBasis<FloatOut,FloatIn,4,Nc> >
          (outOrder, inOrder, basis, E, X, parity, extend, meta, location);
      }else{
        errorQuda("Basis change not supported");
      }
    }


  // Need to rewrite the following two functions...
  // Decide on the output order 
  template<typename FloatOut, typename FloatIn, int Ns, int Nc, typename InOrder>
    void extendedCopyColorSpinor(InOrder &inOrder, ColorSpinorField &out,
        QudaGammaBasis inBasis, const int *E, const int *X,  const int parity, const bool extend,
        QudaFieldLocation location, FloatOut *Out, float *outNorm){

    if (out.isNative()) {
      typedef typename colorspinor_mapper<FloatOut,Ns,Nc>::type ColorSpinor;
      ColorSpinor outOrder(out, 1, Out, outNorm);
      copySpinorEx<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.GammaBasis(), inBasis, E, X, parity, extend, out, location);
    } else {
      errorQuda("Order not defined");
    }

  }

  template<typename FloatOut, typename FloatIn, int Ns, int Nc> 
    void extendedCopyColorSpinor(ColorSpinorField &out, const ColorSpinorField &in, 
        const int parity, const QudaFieldLocation location, FloatOut *Out, FloatIn *In, 
        float* outNorm, float *inNorm){

    int E[4];
    int X[4];
    const bool extend = (out.Volume() >= in.Volume());
    if (extend) {
      for (int d=0; d<4; d++) {
	E[d] = out.X()[d];
	X[d] = in.X()[d];
      }
    } else {
      for (int d=0; d<4; d++) {
	E[d] = in.X()[d];
	X[d] = out.X()[d];
      }
    }
    X[0] *= 2; E[0] *= 2; // Since we consider only a single parity at a time

    if (in.isNative()) {
      typedef typename colorspinor_mapper<FloatIn,Ns,Nc>::type ColorSpinor;
      ColorSpinor inOrder(in, 1, In, inNorm);
      extendedCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in.GammaBasis(), E, X, parity, extend, location, Out, outNorm);
    } else {
      errorQuda("Order not defined");
    }

  }

  template<int Ns, typename dstFloat, typename srcFloat>
    void copyExtendedColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
        const int parity, const QudaFieldLocation location, dstFloat *Dst, srcFloat *Src,
        float *dstNorm, float *srcNorm) {


      if(dst.Ndim() != src.Ndim())
        errorQuda("Number of dimensions %d %d don't match", dst.Ndim(), src.Ndim());

      if(!(dst.SiteOrder() == src.SiteOrder() ||
            (dst.SiteOrder() == QUDA_EVEN_ODD_SITE_ORDER &&
             src.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER) ||
            (dst.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER &&
             src.SiteOrder() == QUDA_EVEN_ODD_SITE_ORDER) ) ){

        errorQuda("Subset orders %d %d don't match", dst.SiteOrder(), src.SiteOrder());
      }

      if(dst.SiteSubset() != src.SiteSubset())
        errorQuda("Subset types do not match %d %d", dst.SiteSubset(), src.SiteSubset());

      if(dst.Ncolor() != 3 || src.Ncolor() != 3) errorQuda("Nc != 3 not yet supported");

      const int Nc = 3;

      // We currently only support parity-ordered fields; even-odd or odd-even
      if(dst.SiteOrder() == QUDA_LEXICOGRAPHIC_SITE_ORDER){
        errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
      }

      if(dst.SiteSubset() == QUDA_FULL_SITE_SUBSET){
        if(src.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER ||
            dst.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER){
          errorQuda("QDPJIT field ordering not supported for full site fields");
        }

        // set for the source subset ordering
        srcFloat *srcEven = Src ? Src : (srcFloat*)src.V();
        srcFloat* srcOdd = (srcFloat*)((char*)srcEven + src.Bytes()/2);
        float *srcNormEven = srcNorm ? srcNorm : (float*)src.Norm();
        float *srcNormOdd = (float*)((char*)srcNormEven + src.NormBytes()/2);
        if(src.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER){
          std::swap<srcFloat*>(srcEven, srcOdd);
          std::swap<float*>(srcNormEven, srcNormOdd);
        }

        // set for the destination subset ordering
        dstFloat *dstEven = Dst ? Dst : (dstFloat*)dst.V();
        dstFloat *dstOdd = (dstFloat*)((char*)dstEven + dst.Bytes()/2);
        float *dstNormEven = dstNorm ? dstNorm : (float*)dst.Norm();
        float *dstNormOdd = (float*)((char*)dstNormEven + dst.NormBytes()/2);
        if(dst.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER){
          std::swap<dstFloat*>(dstEven, dstOdd);
          std::swap<float*>(dstNormEven, dstNormOdd);
        }

        // should be able to apply to select either even or odd parity at this point as well.
        extendedCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>
          (dst, src, 0, location, dstEven, srcEven, dstNormEven, srcNormEven);
        extendedCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>
          (dst, src, 1, location, dstOdd, srcOdd, dstNormOdd, srcNormOdd); 
      }else{
        extendedCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>
          (dst, src, parity, location, Dst, Src, dstNorm, srcNorm);
      } // N.B. Need to update this to account for differences in parity
    }


  template<typename dstFloat, typename srcFloat>
    void CopyExtendedColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
        const int parity, const QudaFieldLocation location, dstFloat *Dst, srcFloat *Src,
        float *dstNorm=0, float *srcNorm=0)
    {
      if(dst.Nspin() != src.Nspin())
        errorQuda("source and destination spins must match");

      if(dst.Nspin() == 4){
#ifdef NSPIN4
        copyExtendedColorSpinor<4>(dst, src, parity, location, Dst, Src, dstNorm, srcNorm);
#else
	errorQuda("Extended copy has not been built for Nspin=%d fields",dst.Nspin());
#endif
      }else if(dst.Nspin() == 1){
#ifdef NSPIN1
        copyExtendedColorSpinor<1>(dst, src, parity, location, Dst, Src, dstNorm, srcNorm);
#else
	errorQuda("Extended copy has not been built for Nspin=%d fields", dst.Nspin());
#endif
      }else{
        errorQuda("Nspin=%d unsupported", dst.Nspin());
      }
    }


  // There's probably no need to have the additional Dst and Src arguments here!
  void copyExtendedColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
      QudaFieldLocation location, const int parity, void *Dst, void *Src, 
      void *dstNorm, void *srcNorm){

#if 0
    if(dst.Precision() == QUDA_DOUBLE_PRECISION){
      if(src.Precision() == QUDA_DOUBLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<double*>(Dst), static_cast<double*>(Src));
      }else if(src.Precision() == QUDA_SINGLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location,  static_cast<double*>(Dst), static_cast<float*>(Src));
      }else if(src.Precision() == QUDA_HALF_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<double*>(Dst), static_cast<short*>(Src), 0, static_cast<float*>(srcNorm));
      } else {
        errorQuda("Unsupported Precision %d", src.Precision());
      }
    } else if (dst.Precision() == QUDA_SINGLE_PRECISION){
      if(src.Precision() == QUDA_DOUBLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<float*>(Dst), static_cast<double*>(Src));
      }else if(src.Precision() == QUDA_SINGLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<float*>(Dst), static_cast<float*>(Src));
      }else if(src.Precision() == QUDA_HALF_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<float*>(Dst), static_cast<short*>(Src), 0, static_cast<float*>(srcNorm));
      }else{
        errorQuda("Unsupported Precision %d", src.Precision());
      }
    } else if (dst.Precision() == QUDA_HALF_PRECISION){
      if(src.Precision() == QUDA_DOUBLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<short*>(Dst), static_cast<double*>(Src), static_cast<float*>(dstNorm), 0);
      }else if(src.Precision() == QUDA_SINGLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<short*>(Dst), static_cast<float*>(Src), static_cast<float*>(dstNorm), 0);
      }else if(src.Precision() == QUDA_HALF_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<short*>(Dst), static_cast<short*>(Src), static_cast<float*>(dstNorm), static_cast<float*>(srcNorm));
      }else{
        errorQuda("Unsupported Precision %d", src.Precision());
      }
    } else if (dst.Precision() == QUDA_QUARTER_PRECISION){
      if(src.Precision() == QUDA_DOUBLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<char*>(Dst), static_cast<double*>(Src), static_cast<float*>(dstNorm), 0);
      }else if(src.Precision() == QUDA_SINGLE_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<char*>(Dst), static_cast<float*>(Src), static_cast<float*>(dstNorm), 0);
      }else if(src.Precision() == QUDA_HALF_PRECISION){
        CopyExtendedColorSpinor(dst, src, parity, location, static_cast<char*>(Dst), static_cast<short*>(Src), static_cast<float*>(dstNorm), static_cast<float*>(srcNorm));
      }else{
        errorQuda("Unsupported Precision %d", src.Precision());
      }
    }else{
      errorQuda("Unsupported Precision %d", dst.Precision());
    }
#else
    errorQuda("Disabled");
#endif
  }

} // quda

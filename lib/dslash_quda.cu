#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#include <color_spinor_field.h>
#include <clover_field.h>

// these control the Wilson-type actions
#ifdef GPU_WILSON_DIRAC
//#define DIRECT_ACCESS_LINK
//#define DIRECT_ACCESS_WILSON_SPINOR
//#define DIRECT_ACCESS_WILSON_ACCUM
//#define DIRECT_ACCESS_WILSON_INTER
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR
//#define DIRECT_ACCESS_CLOVER
#endif // GPU_WILSON_DIRAC

//these are access control for staggered action
#ifdef GPU_STAGGERED_DIRAC
#if (__COMPUTE_CAPABILITY__ >= 300) // Kepler works best with texture loads only
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#elif (__COMPUTE_CAPABILITY__ >= 200)
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#else
#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#endif
#endif // GPU_STAGGERED_DIRAC

#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>
#include <blas_quda.h>
#include <face_quda.h>

#include <inline_ptx.h>

namespace quda {

  namespace dslash_aux {
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

#include <tm_core.h>              // solo twisted mass kernel
#include <tmc_core.h>              // solo twisted mass kernel
#include <clover_def.h>           // kernels for applying the clover term alone
  }

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#ifndef CLOVER_SHARED_FLOATS_PER_THREAD
#define CLOVER_SHARED_FLOATS_PER_THREAD 0
#endif

#ifndef NDEGTM_SHARED_FLOATS_PER_THREAD
#define NDEGTM_SHARED_FLOATS_PER_THREAD 0
#endif

  // these should not be namespaced!!
  // determines whether the temporal ghost zones are packed with a gather kernel,
  // as opposed to multiple calls to cudaMemcpy()
  static bool kernelPackT = false;

  void setKernelPackT(bool packT) { kernelPackT = packT; }

  bool getKernelPackT() { return kernelPackT; }


  //these params are needed for twisted mass (in particular, for packing twisted spinor)
  static bool twistPack = false;

  void setTwistPack(bool flag) { twistPack = flag; }
  bool getTwistPack() { return twistPack; }

  namespace dslash {
    int it = 0;

#ifdef PTHREADS
    cudaEvent_t interiorDslashEnd;
#endif
    cudaEvent_t packEnd[Nstream];
    cudaEvent_t gatherStart[Nstream];
    cudaEvent_t gatherEnd[Nstream];
    cudaEvent_t scatterStart[Nstream];
    cudaEvent_t scatterEnd[Nstream];
    cudaEvent_t dslashStart;
    cudaEvent_t dslashEnd;
  }

  void createDslashEvents()
  {
    using namespace dslash;
    // add cudaEventDisableTiming for lower sync overhead
    for (int i=0; i<Nstream; i++) {
      cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
      cudaEventCreate(&gatherStart[i], cudaEventDisableTiming);
      cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterStart[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterEnd[i], cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&dslashStart, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&dslashEnd, cudaEventDisableTiming);
#ifdef PTHREADS
    cudaEventCreateWithFlags(&interiorDslashEnd, cudaEventDisableTiming);
#endif

    checkCudaError();
  }


  void destroyDslashEvents()
  {
    using namespace dslash;
    for (int i=0; i<Nstream; i++) {
      cudaEventDestroy(packEnd[i]);
      cudaEventDestroy(gatherStart[i]);
      cudaEventDestroy(gatherEnd[i]);
      cudaEventDestroy(scatterStart[i]);
      cudaEventDestroy(scatterEnd[i]);
    }

    cudaEventDestroy(dslashStart);
    cudaEventDestroy(dslashEnd);
#ifdef PTHREADS
    cudaEventDestroy(interiorDslashEnd);
#endif

    checkCudaError();
  }

  using namespace dslash_aux;

template <typename sFloat, typename cFloat>
class CloverCuda : public Tunable {
  private:
    cudaColorSpinorField *out;
    float *outNorm;
    char *saveOut, *saveOutNorm;
    const cFloat *clover;
    const float *cloverNorm;
    const cudaColorSpinorField *in;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return CLOVER_SHARED_FLOATS_PER_THREAD * reg_size;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in->VolumeCB(); }

  public:
    CloverCuda(cudaColorSpinorField *out, const cFloat *clover, const float *cloverNorm, 
	       int cl_stride, const cudaColorSpinorField *in)
      : out(out), clover(clover), cloverNorm(cloverNorm), in(in)
    {
      bindSpinorTex<sFloat>(in);
      dslashParam.sp_stride = in->Stride();
#ifdef GPU_CLOVER_DIRAC
      dslashParam.cl_stride = cl_stride;
#endif
    }
    virtual ~CloverCuda() { unbindSpinorTex<sFloat>(in); }
    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      cloverKernel<<<gridDim, tp.block, tp.shared_bytes, stream>>>
        ((sFloat*)out->V(), (float*)out->Norm(), clover, cloverNorm, 
         (sFloat*)in->V(), (float*)in->Norm(), dslashParam);
    }
    virtual TuneKey tuneKey() const { return TuneKey(in->VolString(), typeid(*this).name()); }

    // Need to save the out field if it aliases the in field
    void preTune() {
      if (in == out) {
        saveOut = new char[out->Bytes()];
        cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
        if (typeid(sFloat) == typeid(short4)) {
          saveOutNorm = new char[out->NormBytes()];
          cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
        }
      }
    }

    // Restore if the in and out fields alias
    void postTune() {
      if (in == out) {
        cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
        delete[] saveOut;
        if (typeid(sFloat) == typeid(short4)) {
          cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
          delete[] saveOutNorm;
        }
      }
    }

    std::string paramString(const TuneParam &param) const // Don't bother printing the grid dim.
    {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 504ll * in->VolumeCB(); }
};


void cloverCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover clover, 
		const cudaColorSpinorField *in, const int parity) {

  dslashParam.parity = parity;
  dslashParam.threads = in->Volume();

#ifdef GPU_CLOVER_DIRAC
  Tunable *clov = 0;
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);

  if (in->Precision() != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    clov = new CloverCuda<double2, double2>(out, (double2*)cloverP, (float*)cloverNormP, clover.stride, in);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    clov = new CloverCuda<float4, float4>(out, (float4*)cloverP, (float*)cloverNormP, clover.stride, in);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    clov = new CloverCuda<short4, short4>(out, (short4*)cloverP, (float*)cloverNormP, clover.stride, in);
  }
  clov->apply(0);

  unbindCloverTex(clover);
  checkCudaError();

  delete clov;
#else
  errorQuda("Clover dslash has not been built");
#endif
}


template <typename sFloat>
class TwistGamma5Cuda : public Tunable {

  private:
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;
    double a;
    double b;
    double c;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in->X(0) * in->X(1) * in->X(2) * in->X(3); }

    char *saveOut, *saveOutNorm;

  public:
    TwistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
        double kappa, double mu, double epsilon, const int dagger, QudaTwistGamma5Type twist) :
      out(out), in(in) 
  {
    bindSpinorTex<sFloat>(in);
    dslashParam.sp_stride = in->Stride();
    if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) {
      setTwistParam(a, b, kappa, mu, dagger, twist);
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
      dslashParam.fl_stride = in->VolumeCB();
#endif
    } else {//twist doublet
      a = kappa, b = mu, c = epsilon;
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
      dslashParam.fl_stride = in->VolumeCB()/2;
#endif
    } 
  }

    virtual ~TwistGamma5Cuda() {
      unbindSpinorTex<sFloat>(in);    
    }

    TuneKey tuneKey() const { return TuneKey(in->VolString(), typeid(*this).name(), in->AuxString()); }

    void apply(const cudaStream_t &stream) 
    {
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) {
        twistGamma5Kernel<<<gridDim, tp.block, tp.shared_bytes, stream>>> 
          ((sFloat*)out->V(), (float*)out->Norm(), a, b, 
           (sFloat*)in->V(), (float*)in->Norm(), dslashParam);
      } else {
        twistGamma5Kernel<<<gridDim, tp.block, tp.shared_bytes, stream>>>
          ((sFloat*)out->V(), (float*)out->Norm(), a, b, c, 
           (sFloat*)in->V(), (float*)in->Norm(), dslashParam);
      }
#endif
    }

    void preTune() {
      saveOut = new char[out->Bytes()];
      cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
        saveOutNorm = new char[out->NormBytes()];
        cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
      }
    }

    void postTune() {
      cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
        cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
        delete[] saveOutNorm;
      }
    }

    std::string paramString(const TuneParam &param) const {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 24ll * in->VolumeCB(); }
    long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
};

//!ndeg tm: 
void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
    const int dagger, const double &kappa, const double &mu, const double &epsilon,   const QudaTwistGamma5Type twist)
{
  if(in->TwistFlavor() == QUDA_TWIST_PLUS || in->TwistFlavor() == QUDA_TWIST_MINUS)
    dslashParam.threads = in->Volume();
  else //twist doublet    
    dslashParam.threads = in->Volume() / 2;

#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
  Tunable *twistGamma5 = 0;

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    twistGamma5 = new TwistGamma5Cuda<double2>(out, in, kappa, mu, epsilon, dagger, twist);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    twistGamma5 = new TwistGamma5Cuda<float4>(out, in, kappa, mu, epsilon, dagger, twist);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    twistGamma5 = new TwistGamma5Cuda<short4>(out, in, kappa, mu, epsilon, dagger, twist);
  }

  twistGamma5->apply(streams[Nstream-1]);
  checkCudaError();

  delete twistGamma5;
#else
  errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
}

#if (__COMPUTE_CAPABILITY__ >= 200) && defined(GPU_TWISTED_CLOVER_DIRAC)
#include "dslash_core/tmc_gamma_core.h"
#endif

template <typename cFloat, typename sFloat>
class TwistCloverGamma5Cuda : public Tunable {
  private:
    const cFloat *clover;
    const float *cNorm;
    const cFloat *cloverInv;
    const float *cNrm2;
    QudaTwistGamma5Type twist;
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;
    double a;
    double b;
    double c;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in->X(0) * in->X(1) * in->X(2) * in->X(3); }
    char *saveOut, *saveOutNorm;

  public:
    TwistCloverGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
        double kappa, double mu, double epsilon, const int dagger, QudaTwistGamma5Type tw,
			  cFloat *clov, const float *cN, cFloat *clovInv, const float *cN2, int cl_stride) :
      out(out), in(in)
  {
    bindSpinorTex<sFloat>(in);
    dslashParam.sp_stride = in->Stride();
#ifdef GPU_TWISTED_CLOVER_DIRAC
    dslashParam.cl_stride = cl_stride;
    dslashParam.fl_stride = in->VolumeCB();
#endif
    twist = tw;
    clover = clov;
    cNorm = cN;
    cloverInv = clovInv;
    cNrm2 = cN2;

    if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS))
      setTwistParam(a, b, kappa, mu, dagger, tw);
    else{//twist doublet
      errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
    } 
  }
    virtual ~TwistCloverGamma5Cuda() {
      unbindSpinorTex<sFloat>(in);    
    }

    TuneKey tuneKey() const {
      return TuneKey(in->VolString(), typeid(*this).name(), in->AuxString());
    }  

    void apply(const cudaStream_t &stream)
    {
#if (__COMPUTE_CAPABILITY__ >= 200) && defined(GPU_TWISTED_CLOVER_DIRAC)
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) {	//Idea for the kernel, two spinor inputs (IN and clover applied IN), on output (Clover applied IN + ig5IN)
        if (twist == QUDA_TWIST_GAMMA5_DIRECT)
          twistCloverGamma5Kernel<<<gridDim, tp.block, tp.shared_bytes, stream>>> 
            ((sFloat*)out->V(), (float*)out->Norm(), a, 
             (sFloat*)in->V(), (float*)in->Norm(), dslashParam,
             clover, cNorm, cloverInv, cNrm2);
        else if (twist == QUDA_TWIST_GAMMA5_INVERSE)
          twistCloverGamma5InvKernel<<<gridDim, tp.block, tp.shared_bytes, stream>>> 
            ((sFloat*)out->V(), (float*)out->Norm(), a, 
             (sFloat*)in->V(), (float*)in->Norm(), dslashParam,
             clover, cNorm, cloverInv, cNrm2);
      } else {
        errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
      }
#endif
    }

    void preTune() {
      saveOut = new char[out->Bytes()];
      cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
        saveOutNorm = new char[out->NormBytes()];
        cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
      }
    }

    void postTune() {
      cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
        cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
        delete[] saveOutNorm;
      }
    }

    std::string paramString(const TuneParam &param) const {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 24ll * in->VolumeCB(); }	//TODO FIX THIS NUMBER!!!
    long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
};

void twistCloverGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in, const int dagger, const double &kappa, const double &mu,
    const double &epsilon, const QudaTwistGamma5Type twist, const FullClover *clov, const FullClover *clovInv, const int parity)
{
  if(in->TwistFlavor() == QUDA_TWIST_PLUS || in->TwistFlavor() == QUDA_TWIST_MINUS)
    dslashParam.threads = in->Volume();
  else //twist doublet    
    errorQuda("Twisted doublet not supported in twisted clover dslash");

#ifdef GPU_TWISTED_CLOVER_DIRAC
  Tunable *tmClovGamma5 = 0;

  void *clover, *cNorm, *cloverInv, *cNorm2;
  QudaPrecision clover_prec = bindTwistedCloverTex(*clov, *clovInv, parity, &clover, &cNorm, &cloverInv, &cNorm2);

  if (in->Precision() != clover_prec)
    errorQuda("ERROR: Clover precision and spinor precision do not match\n");

  if (clov->stride != clovInv->stride) 
    errorQuda("clover and cloverInv must have matching strides (%d != %d)", clov->stride, clovInv->stride);
    

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
    tmClovGamma5 = new TwistCloverGamma5Cuda<double2,double2>
      (out, in, kappa, mu, epsilon, dagger, twist, (double2 *) clover, (float *) cNorm, (double2 *) cloverInv, (float *) cNorm2, clov->stride);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    tmClovGamma5 = new TwistCloverGamma5Cuda<float4,float4>
      (out, in, kappa, mu, epsilon, dagger, twist, (float4 *) clover, (float *) cNorm, (float4 *) cloverInv, (float *) cNorm2, clov->stride);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    tmClovGamma5 = new TwistCloverGamma5Cuda<short4,short4>
      (out, in, kappa, mu, epsilon, dagger, twist, (short4 *) clover, (float *) cNorm, (short4 *) cloverInv, (float *) cNorm2, clov->stride);
  }

  tmClovGamma5->apply(streams[Nstream-1]);
  checkCudaError();

  delete tmClovGamma5;
  unbindTwistedCloverTex(*clov);
#else
  errorQuda("Twisted clover dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
}

} // namespace quda

#ifdef GPU_CONTRACT
#include "contract.cu"
#endif

#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#include <color_spinor_field.h>
//#include <clover_field.h>

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
#else // fermi
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
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
#include <io_spinor.h>

//#include <tm_core.h>              // solo twisted mass kernel
//#include <tmc_core.h>              // solo twisted mass kernel
//#include <clover_def.h>           // kernels for applying the clover term alone
  }

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

  // these should not be namespaced!!
  // determines whether the temporal ghost zones are packed with a gather kernel,
  // as opposed to multiple calls to cudaMemcpy()
  static bool kernelPackT = false;

  void setKernelPackT(bool packT) { kernelPackT = packT; }

  bool getKernelPackT() { return kernelPackT; }

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

    // FIX this is a hack from hell
    // Auxiliary work that can be done while waiting on comms to finis
    Worker *aux_worker;

#if CUDA_VERSION >= 8000
    cuuint32_t *commsEnd_h;
    CUdeviceptr commsEnd_d[Nstream];
#endif
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

    aux_worker = NULL;

#if CUDA_VERSION >= 8000
    commsEnd_h = static_cast<cuuint32_t*>(mapped_malloc(Nstream*sizeof(int)));
    for (int i=0; i<Nstream; i++) {
      cudaHostGetDevicePointer((void**)&commsEnd_d[i], commsEnd_h+i, 0);
      commsEnd_h[i] = 0;
    }
#endif

    checkCudaError();
  }


  void destroyDslashEvents()
  {
    using namespace dslash;

#if CUDA_VERSION >= 8000
    host_free(commsEnd_h);
    commsEnd_h = 0;
#endif

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

} // namespace quda

#include "contract.cu"

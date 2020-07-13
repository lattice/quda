#pragma once

#if CUDA_VERSION >= 10200

#define FAST_REDUCE
#include <cuda/std/atomic>

namespace quda {

  namespace blas {
    cuda::atomic<unsigned int, cuda::thread_scope_device>* getReduceCount();
    cuda::atomic<unsigned int, cuda::thread_scope_system>* getReduceSignal();
  }

}

#endif

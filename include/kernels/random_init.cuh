#pragma once

#include <random_helper.h>
#include <lattice_field.h>
#include <index_helper.cuh>
#include <comm_quda.h>
#include <kernel.h>

namespace quda {

  struct rngArg : kernel_param<> {
    int commCoord[QUDA_MAX_DIM];
    int X[QUDA_MAX_DIM];
    int X_global[QUDA_MAX_DIM];
    RNGState *state;
    unsigned long long seed;
    rngArg(RNGState *state, unsigned long long seed, const LatticeField &meta) :
      kernel_param(dim3(meta.LocalVolumeCB(), meta.SiteSubset(), 1)),
      state(state),
      seed(seed)
    {
      for (int i=0; i<4; i++) {
        commCoord[i] = comm_coord(i);
        X[i] = meta.LocalX()[i];
        X_global[i] = X[i] * comm_dim(i);
      }
    }
  };

  /**
     @brief functor to initialize the RNG states
     @param state RNG state array
     @param seed initial seed for RNG
     @param size size of the RNG state array
     @param arg Metadata needed for computing multi-gpu offsets
  */
  template <typename Arg>
  struct init_random {
    const Arg &arg;
    __device__ constexpr init_random(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()(int id, int parity)
    {
      // Each thread gets same seed, a different sequence number, no offset
      int x[4];
      getCoords(x, id, arg.X, parity);
      for (int i = 0; i < 4; i++) x[i] += arg.commCoord[i] * arg.X[i];
      int idd = (((x[3] * arg.X_global[2] + x[2]) * arg.X_global[1]) + x[1]) * arg.X_global[0] + x[0];
      random_init(arg.seed, idd, 0, arg.state[parity * arg.threads.x + id]);
    }
  };

}

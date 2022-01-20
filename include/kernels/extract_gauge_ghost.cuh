#pragma once

#include <comm_quda.h>
#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <kernel.h>

namespace quda {

  using namespace gauge;

  template <typename Float, int nColor_, typename Gauge, bool extract_>
  struct ExtractGhostArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nDim = 4;
    static constexpr int nColor = nColor_;
    static constexpr bool extract = extract_;
    Gauge u;
    const int nFace;
    int X[nDim];
    int A[nDim];
    int B[nDim];
    int C[nDim];
    int f[nDim][nDim];
    bool localParity[nDim];
    int faceVolumeCB[nDim];
    int comm_dim[QUDA_MAX_DIM];
    const int offset;
    ExtractGhostArg(const GaugeField &u, Float **Ghost, int offset, uint64_t size) :
      kernel_param(dim3(size, 1, 1)),
      u(u, 0, Ghost),
      nFace(u.Nface()),
      offset(offset)
    {
      for (int d=0; d<nDim; d++) {
	X[d] = u.X()[d];
	comm_dim[d] = comm_dim_partitioned(d);
	faceVolumeCB[d] = u.SurfaceCB(d)*u.Nface();
      }

      //loop variables: a, b, c with a the most signifcant and c the least significant
      //A, B, C the maximum value
      //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
      A[0] = X[3]; B[0] = X[2]; C[0] = X[1]; // X dimension face
      A[1] = X[3]; B[1] = X[2]; C[1] = X[0]; // Y dimension face
      A[2] = X[3]; B[2] = X[1]; C[2] = X[0]; // Z dimension face
      A[3] = X[2]; B[3] = X[1]; C[3] = X[0]; // T dimension face

      //multiplication factor to compute index in original cpu memory
      int f_[nDim][nDim] = {
        {X[0]*X[1]*X[2],  X[0]*X[1], X[0],               1},
        {X[0]*X[1]*X[2],  X[0]*X[1],    1,            X[0]},
        {X[0]*X[1]*X[2],       X[0],    1,       X[0]*X[1]},
        {     X[0]*X[1],       X[0],    1,  X[0]*X[1]*X[2]}
      };
      memcpy(f, f_, sizeof(f));

      //set the local processor parity
      //switching odd and even ghost gauge when that dimension size is odd
      //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      // FIXME - I don't understand this, shouldn't it be commDim(dim) == 0 ?
      for (int dim=0; dim<nDim; dim++)
        //localParity[dim] = (X[dim]%2==0 || commDim(dim)) ? 0 : 1;
        localParity[dim] = ((X[dim] % 2 == 1) && (commDim(dim) > 1)) ? 1 : 0;
    }
  };

  template <typename Arg> struct GhostExtractor {
    const Arg &arg;
    constexpr GhostExtractor(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int X, int, int parity_dim)
    {
      using real = typename Arg::real;
      constexpr int nColor = Arg::nColor;

      int parity = parity_dim / Arg::nDim;
      int dim = parity_dim % Arg::nDim;

      // for now we never inject unless we have partitioned in that dimension
      if (!arg.comm_dim[dim] && !Arg::extract) return;

      // linear index used for writing into ghost buffer
      if (X >= 2*arg.faceVolumeCB[dim]) return;

      // X = ((d * A + a)*B + b)*C + c
      int dab = X/arg.C[dim];
      int c = X - dab*arg.C[dim];
      int da = dab/arg.B[dim];
      int b = dab - da*arg.B[dim];
      int d = da / arg.A[dim];
      int a = da - d * arg.A[dim];
      d += arg.X[dim]-arg.nFace;

      // index is a checkboarded spacetime coordinate
      int indexCB = (a*arg.f[dim][0] + b*arg.f[dim][1] + c*arg.f[dim][2] + d*arg.f[dim][3]) >> 1;
      // we only do the extraction for parity we are currently working on
      int oddness = (a+b+c+d)&1;
      if (oddness == parity) {
        if (Arg::extract) {
          // load the ghost element from the bulk
          Matrix<complex<real>, nColor> u = arg.u(dim+arg.offset, indexCB, parity);
          arg.u.Ghost(dim, X>>1, (parity+arg.localParity[dim])&1) = u;
        } else { // injection
          Matrix <complex<real>, nColor> u = arg.u.Ghost(dim, X>>1, (parity+arg.localParity[dim])&1);
          arg.u(dim+arg.offset, indexCB, parity) = u; // save the ghost element to the bulk
        }
      } // oddness == parity
    }
  };

  /**
     Generic GPU gauge ghost extraction and packing using fine-grained accessors
     NB This routines is specialized to four dimensions
     FIXME this implementation will have two-way warp divergence
  */
  template <typename Arg> struct GhostExtractorFineGrained {
    const Arg &arg;
    constexpr GhostExtractorFineGrained(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int X, int i, int parity_dim)
    {
      constexpr int nColor = Arg::nColor;

      int parity = parity_dim / Arg::nDim;
      int dim = parity_dim % Arg::nDim;

      // for now we never inject unless we have partitioned in that dimension
      if (!arg.comm_dim[dim] && !Arg::extract) return;

      // linear index used for writing into ghost buffer
      if (X >= 2*arg.faceVolumeCB[dim]) return;

      // X = ((d * A + a)*B + b)*C + c
      int dab = X/arg.C[dim];
      int c = X - dab*arg.C[dim];
      int da = dab/arg.B[dim];
      int b = dab - da*arg.B[dim];
      int d = da / arg.A[dim];
      int a = da - d * arg.A[dim];
      d += arg.X[dim]-arg.nFace;

      // index is a checkboarded spacetime coordinate
      int indexCB = (a*arg.f[dim][0] + b*arg.f[dim][1] + c*arg.f[dim][2] + d*arg.f[dim][3]) >> 1;
      // we only do the extraction for parity we are currently working on
      int oddness = (a+b+c+d)&1;
      if (oddness == parity) {
        for (int j=0; j<nColor; j++) {
          if (Arg::extract) {
            arg.u.Ghost(dim, (parity+arg.localParity[dim])&1, X>>1, i, j)
              = arg.u(dim+arg.offset, parity, indexCB, i, j);
          } else { // injection
            arg.u(dim+arg.offset, parity, indexCB, i, j)
              = arg.u.Ghost(dim, (parity+arg.localParity[dim])&1, X>>1, i, j);
          }
        }
      } // oddness == parity
    }
  };  
  
}

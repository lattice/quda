#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <kernel.h>

namespace quda {

  using namespace gauge;

  template <typename Gauge, int dim_, bool extract_>
  struct ExtractGhostExArg : kernel_param<> {
    static constexpr int nDim = 4;
    static constexpr int dim = dim_;
    static constexpr bool extract = extract_;
    static constexpr int nColor = gauge::Ncolor(Gauge::length);
    using real = typename Gauge::real;
    Gauge u;
    int E[nDim];
    int X[nDim];
    int R[nDim];
    int surfaceCB[nDim];
    int A0[nDim];
    int A1[nDim];
    int B0[nDim];
    int B1[nDim];
    int C0[nDim];
    int C1[nDim];
    int fBody[nDim][nDim];
    int fBuf[nDim][nDim];
    int localParity[nDim];
    ExtractGhostExArg(const GaugeField &u, const lat_dim_t &R_, void **ghost) :
      kernel_param(dim3(0, 2, 2)),
      u(u, 0, reinterpret_cast<typename Gauge::store_t**>(ghost))
    {
      for (int d=0; d<nDim; d++) {
	E[d] = u.X()[d];
	this->R[d] = R_[d];
	X[d] = u.X()[d] - 2*R[d];
	surfaceCB[d] = u.SurfaceCB(d);
      }

      //set the local processor parity
      //switching odd and even ghost gauge when that dimension size is odd
      //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      // FIXME - I don't understand this, shouldn't it be commDim(dim) == 0 ?
      for (int d=0; d<nDim; d++)
        localParity[dim] = ((X[dim] % 2 ==1) && (commDim(dim) > 1)) ? 1 : 0;
      //      localParity[dim] = (X[dim]%2==0 || commDim(dim)) ? 0 : 1;

      //loop variables: a, b, c with a the most signifcant and c the least significant
      //A0, B0, C0 the minimum value
      //A0, B0, C0 the maximum value

      int A0_[] {R[3], R[3], R[3], 0};
      int A1_[] = {X[3]+R[3], X[3]+R[3],   X[3]+R[3],    X[2]+2*R[2]};
      int B0_[] = {R[2], R[2], 0, 0};
      int B1_[] = {X[2]+R[2], X[2]+R[2],   X[1]+2*R[1],  X[1]+2*R[1]};
      int C0_[] = {R[1], 0, 0, 0};
      int C1_[] = {X[1]+R[1], X[0]+2*R[0], X[0]+2*R[0],  X[0]+2*R[0]};
      memcpy(A0, A0_, sizeof(A0));
      memcpy(A1, A1_, sizeof(A1));
      memcpy(B0, B0_, sizeof(B0));
      memcpy(B1, B1_, sizeof(B1));
      memcpy(C0, C0_, sizeof(C0));
      memcpy(C1, C1_, sizeof(C1));

      auto dA = A1[dim]-A0[dim];
      auto dB = B1[dim]-B0[dim];
      auto dC = C1[dim]-C0[dim];
      this->threads.x = R[dim]*dA*dB*dC*u.Geometry();

      int fBody_[nDim][nDim] = {
        {E[2]*E[1]*E[0], E[1]*E[0], E[0],              1},
        {E[2]*E[1]*E[0], E[1]*E[0],    1,           E[0]},
        {E[2]*E[1]*E[0],      E[0],    1,      E[1]*E[0]},
        {E[1]*E[0],           E[0],    1, E[2]*E[1]*E[0]}
      };
      memcpy(fBody, fBody_, sizeof(fBody_));

      int fBuf_[nDim][nDim]={
        {E[2]*E[1], E[1], 1, E[3]*E[2]*E[1]},
        {E[2]*E[0], E[0], 1, E[3]*E[2]*E[0]},
        {E[1]*E[0], E[0], 1, E[3]*E[1]*E[0]},
        {E[1]*E[0], E[0], 1, E[2]*E[1]*E[0]}
      };
      memcpy(fBuf, fBuf_, sizeof(fBuf));

    }

  };

  template <typename Arg>
  __device__ __host__ void extractor(const Arg &arg, int dir, int a, int b,
				     int c, int d, int g, int parity)
  {
    auto dim = Arg::dim;
    int srcIdx = (a*arg.fBody[dim][0] + b*arg.fBody[dim][1] +
		  c*arg.fBody[dim][2] + d*arg.fBody[dim][3]) >> 1;

    int dstIdx = (a*arg.fBuf[dim][0] + b*arg.fBuf[dim][1] +
		  c*arg.fBuf[dim][2] + (d-(dir?arg.X[dim]:arg.R[dim]))*arg.fBuf[dim][3]) >> 1;

    // load the ghost element from the bulk
    Matrix<complex<typename Arg::real>, Arg::nColor> u = arg.u(g, srcIdx, parity);

    // need dir dependence in write
    // srcIdx is used here to determine boundary condition
    arg.u.saveGhostEx(u.data, dstIdx, srcIdx, dir, dim, g, (parity+arg.localParity[dim])&1, arg.R);
  }


  template <typename Arg>
  __device__ __host__ void injector(const Arg &arg, int dir, int a, int b,
				    int c, int d, int g, int parity)
  {
    auto dim = Arg::dim;
    int srcIdx = (a*arg.fBuf[dim][0] + b*arg.fBuf[dim][1] +
		  c*arg.fBuf[dim][2] + (d-dir*(arg.X[dim]+arg.R[dim]))*arg.fBuf[dim][3]) >> 1;

    int dstIdx = (a*arg.fBody[dim][0] + b*arg.fBody[dim][1] +
		  c*arg.fBody[dim][2] + d*arg.fBody[dim][3]) >> 1;

    int oddness = (parity+arg.localParity[dim])&1;

    Matrix<complex<typename Arg::real>, Arg::nColor> u;

    // need dir dependence in read
    // dstIdx is used here to determine boundary condition
    arg.u.loadGhostEx(u.data, srcIdx, dstIdx, dir, dim, g, oddness, arg.R);

    arg.u(g, dstIdx, parity) = u; // save the ghost element into the bulk
  }

  /**
     Generic extended gauge ghost extraction and packing
     NB This routines is specialized to four dimensions
     FIXME this implementation will have two-way warp divergence
  */
  template <typename Arg> struct GhostExtractorEx {
    const Arg &arg;
    constexpr GhostExtractorEx(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int X, int parity, int dir)
    {
      // this will have two-warp divergence since we only do work on
      // one parity but parity alternates between threads
      // linear index used for writing into ghost buffer
      auto dim = Arg::dim;
      int dA = arg.A1[dim]-arg.A0[dim];
      int dB = arg.B1[dim]-arg.B0[dim];
      int dC = arg.C1[dim]-arg.C0[dim];
      int D0 = Arg::extract ? dir*arg.X[dim] + (1-dir)*arg.R[dim] : dir*(arg.X[dim] + arg.R[dim]);

      // thread order is optimized to maximize coalescing
      // X = (((g*R + d) * dA + a)*dB + b)*dC + c
      int gdab = X / dC;
      int c    = arg.C0[dim] + X    - gdab*dC;
      int gda  = gdab / dB;
      int b    = arg.B0[dim] + gdab - gda *dB;
      int gd   = gda / dA;
      int a    = arg.A0[dim] + gda  - gd  *dA;
      int g    = gd / arg.R[dim];
      int d    = D0          + gd   - g   *arg.R[dim];

      // we only do the extraction for parity we are currently working on
      int oddness = (a+b+c+d) & 1;
      if (oddness == parity) {
        if (Arg::extract) extractor(arg, dir, a, b, c, d, g, parity);
        else injector(arg, dir, a, b, c, d, g, parity);
      } // oddness == parity
    }
  };

}

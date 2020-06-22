#include <tune_quda.h>
#include <quda_matrix.h>

#pragma once

namespace quda {

  using namespace gauge;

  template <typename Float, int nColor_, typename Order, int nDim>
  struct ExtractGhostArg {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    Order order;
    const unsigned char nFace;
    unsigned short X[nDim];
    unsigned short A[nDim];
    unsigned short B[nDim];
    unsigned short C[nDim];
    int f[nDim][nDim];
    bool localParity[nDim];
    int faceVolumeCB[nDim];
    int commDim[QUDA_MAX_DIM];
    const int offset;
    ExtractGhostArg(const Order &order, const GaugeField &u, const int *A_,
		    const int *B_, const int *C_, const int f_[nDim][nDim], const int *localParity_, int offset)
      : order(order), nFace(u.Nface()), offset(offset) {
      for (int d=0; d<nDim; d++) {
	X[d] = u.X()[d];
	A[d] = A_[d];
	B[d] = B_[d];
	C[d] = C_[d];
	for (int e=0; e<nDim; e++) f[d][e] = f_[d][e];
	localParity[d] = localParity_[d];
	commDim[d] = comm_dim_partitioned(d);
	faceVolumeCB[d] = u.SurfaceCB(d)*u.Nface();
      }
    }
  };

  /**
     Generic CPU gauge ghost extraction and packing
     NB This routines is specialized to four dimensions
  */
  template <int nDim, bool extract, typename Arg>
  void extractGhost(Arg &arg)
  {
    using real = typename Arg::real;
    constexpr int nColor = Arg::nColor;

    for (int parity=0; parity<2; parity++) {

      for (int dim=0; dim<nDim; dim++) {

	// for now we never inject unless we have partitioned in that dimension
	if (!arg.commDim[dim] && !extract) continue;

	// linear index used for reading/writing into ghost buffer
	int indexGhost = 0;
	// the following 4-way loop means this is specialized for 4 dimensions

	// FIXME redefine a, b, c, d such that we always optimize for locality
	for (int d=arg.X[dim]-arg.nFace; d<arg.X[dim]; d++) { // loop over last nFace faces in this dimension
	  for (int a=0; a<arg.A[dim]; a++) { // loop over the surface elements of this face
	    for (int b=0; b<arg.B[dim]; b++) { // loop over the surface elements of this face
	      for (int c=0; c<arg.C[dim]; c++) { // loop over the surface elements of this face
		// index is a checkboarded spacetime coordinate
		int indexCB = (a*arg.f[dim][0] + b*arg.f[dim][1] + c*arg.f[dim][2] + d*arg.f[dim][3]) >> 1;
		// we only do the extraction for parity we are currently working on
		int oddness = (a+b+c+d) & 1;
		if (oddness == parity) {
#ifdef FINE_GRAINED_ACCESS
		  for (int i=0; i<nColor; i++) {
		    for (int j=0; j<nColor; j++) {
		      if (extract) {
			arg.order.Ghost(dim, (parity+arg.localParity[dim])&1, indexGhost, i, j)
			  = arg.order(dim+arg.offset, parity, indexCB, i, j);
		      } else { // injection
			arg.order(dim+arg.offset, parity, indexCB, i, j)
			  = arg.order.Ghost(dim, (parity+arg.localParity[dim])&1, indexGhost, i, j);
		      }
		    }
		  }
#else
		  if (extract) {
                    // load the ghost element from the bulk
                    Matrix<complex<real>, nColor> u = arg.order(dim+arg.offset, indexCB, parity);
		    arg.order.Ghost(dim, indexGhost, (parity+arg.localParity[dim])&1) = u;
		  } else { // injection
		    Matrix <complex<real>, nColor> u = arg.order.Ghost(dim, indexGhost, (parity+arg.localParity[dim])&1);
		    arg.order(dim+arg.offset, indexCB, parity) = u; // save the ghost element to the bulk
		  }
#endif
		  indexGhost++;
		} // oddness == parity
	      } // c
	    } // b
	  } // a
	} // d

	assert(indexGhost == arg.faceVolumeCB[dim]);
      } // dim

    } // parity

  }

  /**
     Generic GPU gauge ghost extraction and packing
     NB This routines is specialized to four dimensions
     FIXME this implementation will have two-way warp divergence
  */
  template <int nDim, bool extract, typename Arg>
  __global__ void extractGhostKernel(Arg arg)
  {
    using real = typename Arg::real;
    constexpr int nColor = Arg::nColor;

    int parity_dim = blockIdx.z * blockDim.z + threadIdx.z; //parity_dim = parity*nDim + dim
    int parity = parity_dim / nDim;
    int dim = parity_dim % nDim;
    if (parity_dim >= 2 * nDim) return;

    // for now we never inject unless we have partitioned in that dimension
    if (!arg.commDim[dim] && !extract) return;

    // linear index used for writing into ghost buffer
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    //if (X >= 2*arg.nFace*arg.surfaceCB[dim]) continue;
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
#ifdef FINE_GRAINED_ACCESS
      int i = blockIdx.y * blockDim.y + threadIdx.y;
      if (i >= nColor) return;
      for (int j=0; j<nColor; j++) {
	if (extract) {
	  arg.order.Ghost(dim, (parity+arg.localParity[dim])&1, X>>1, i, j)
	    = arg.order(dim+arg.offset, parity, indexCB, i, j);
	} else { // injection
	  arg.order(dim+arg.offset, parity, indexCB, i, j)
	    = arg.order.Ghost(dim, (parity+arg.localParity[dim])&1, X>>1, i, j);
	}
      }
#else
      if (extract) {
        // load the ghost element from the bulk
        Matrix<complex<real>, nColor> u = arg.order(dim+arg.offset, indexCB, parity);
        arg.order.Ghost(dim, X>>1, (parity+arg.localParity[dim])&1) = u;
      } else { // injection
        Matrix <complex<real>, nColor> u = arg.order.Ghost(dim, X>>1, (parity+arg.localParity[dim])&1);
        arg.order(dim+arg.offset, indexCB, parity) = u; // save the ghost element to the bulk
      }
#endif
    } // oddness == parity
  }

  template <int nDim, typename Arg>
  class ExtractGhost : TunableVectorYZ {
    Arg &arg;
    int size;
    const GaugeField &meta;
    QudaFieldLocation location;
    bool extract;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return size; }

  public:
    ExtractGhost(Arg &arg, const GaugeField &meta, QudaFieldLocation location, bool extract)
#ifndef FINE_GRAINED_ACCESS
      : TunableVectorYZ(1, 2*nDim), arg(arg), meta(meta), location(location), extract(extract) {
#else
      : TunableVectorYZ(Arg::nColor, 2*nDim), arg(arg), meta(meta), location(location), extract(extract) {
#endif
      int faceMax = 0;
      for (int d=0; d<nDim; d++)
	faceMax = (arg.faceVolumeCB[d] > faceMax ) ? arg.faceVolumeCB[d] : faceMax;
      size = 2 * faceMax; // factor of comes from parity

#ifndef FINE_GRAINED_ACCESS
      writeAuxString("stride=%d", arg.order.stride);
#else
      writeAuxString("fine-grained");
#endif
    }

    virtual ~ExtractGhost() { ; }

    void apply(const qudaStream_t &stream) {
      if (location==QUDA_CPU_FIELD_LOCATION) {
	if (extract) extractGhost<nDim,true>(arg);
	else extractGhost<nDim,false>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (extract) {
	  extractGhostKernel<nDim, true> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	} else {
	  extractGhostKernel<nDim, false> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	}
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; }
    long long bytes() const {
      int sites = 0;
      for (int d=0; d<nDim; d++) sites += arg.faceVolumeCB[d];
      return 2 * sites * 2 * arg.order.Bytes(); // parity * sites * i/o * vec size
    }
  };


  /**
     Generic gauge ghost extraction and packing (or the converse)
     NB This routines is specialized to four dimensions
  */
  template <typename Float, int length, typename Order>
  void extractGhost(Order order, const GaugeField &u, QudaFieldLocation location, bool extract, int offset) {
    const int *X = u.X();
    constexpr int nDim = 4;
    //loop variables: a, b, c with a the most signifcant and c the least significant
    //A, B, C the maximum value
    //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
    int A[nDim], B[nDim], C[nDim];
    A[0] = X[3]; B[0] = X[2]; C[0] = X[1]; // X dimension face
    A[1] = X[3]; B[1] = X[2]; C[1] = X[0]; // Y dimension face
    A[2] = X[3]; B[2] = X[1]; C[2] = X[0]; // Z dimension face
    A[3] = X[2]; B[3] = X[1]; C[3] = X[0]; // T dimension face

    //multiplication factor to compute index in original cpu memory
    int f[nDim][nDim]={
      {X[0]*X[1]*X[2],  X[0]*X[1], X[0],               1},
      {X[0]*X[1]*X[2],  X[0]*X[1],    1,            X[0]},
      {X[0]*X[1]*X[2],       X[0],    1,       X[0]*X[1]},
      {     X[0]*X[1],       X[0],    1,  X[0]*X[1]*X[2]}
    };

    //set the local processor parity
    //switching odd and even ghost gauge when that dimension size is odd
    //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
    // FIXME - I don't understand this, shouldn't it be commDim(dim) == 0 ?
    int localParity[nDim];
    for (int dim=0; dim<nDim; dim++)
      //localParity[dim] = (X[dim]%2==0 || commDim(dim)) ? 0 : 1;
      localParity[dim] = ((X[dim] % 2 ==1) && (commDim(dim) > 1)) ? 1 : 0;

    ExtractGhostArg<Float, gauge::Ncolor(length), Order, nDim> arg(order, u, A, B, C, f, localParity, offset);
    ExtractGhost<nDim, decltype(arg)> extractor(arg, u, location, extract);
    extractor.apply(0);
  }

} // namespace quda

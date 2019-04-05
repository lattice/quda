#include <gauge_field_order.h>
#include <index_helper.cuh>

namespace quda {

  template <typename PreconditionedGauge, typename Gauge, int n>
  struct CalculateYhatArg {
    PreconditionedGauge Yhat;
    const Gauge Y;
    const Gauge Xinv;
    int dim[QUDA_MAX_DIM];
    int comm_dim[QUDA_MAX_DIM];
    int nFace;
    const int coarseVolumeCB;   /** Coarse grid volume */

    CalculateYhatArg(const PreconditionedGauge &Yhat, const Gauge Y, const Gauge Xinv, const int *dim, const int *comm_dim, int nFace)
      : Yhat(Yhat), Y(Y), Xinv(Xinv), nFace(nFace), coarseVolumeCB(Y.VolumeCB()) {
      for (int i=0; i<4; i++) {
	this->comm_dim[i] = comm_dim[i];
	this->dim[i] = dim[i];
      }
    }
  };

  // complex multiply-add with optimal use of fma
  template<typename Float>
  inline __device__ __host__ void caxpy(const complex<Float> &a, const complex<Float> &x, complex<Float> &y) {
    y.x += a.x*x.x;
    y.x -= a.y*x.y;
    y.y += a.y*x.x;
    y.y += a.x*x.y;
  }

  template<typename Float, int n, typename Arg>
  inline __device__ __host__ void computeYhat(Arg &arg, int d, int x_cb, int parity, int i, int j) {

    constexpr int nDim = 4;
    int coord[nDim];
    getCoords(coord, x_cb, arg.dim, parity);

    const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

    // first do the backwards links Y^{+\mu} * X^{-\dagger}
    if ( arg.comm_dim[d] && (coord[d] - arg.nFace < 0) ) {

      complex<Float> yHat = 0.0;
#pragma unroll
      for(int k = 0; k<n; k++) {
        caxpy(arg.Y.Ghost(d,1-parity,ghost_idx,i,k), conj(arg.Xinv(0,parity,x_cb,j,k)), yHat);
      }
      arg.Yhat.Ghost(d,1-parity,ghost_idx,i,j) = yHat;

    } else {
      const int back_idx = linkIndexM1(coord, arg.dim, d);

      complex<Float> yHat = 0.0;
#pragma unroll
      for (int k = 0; k<n; k++) {
        caxpy(arg.Y(d,1-parity,back_idx,i,k), conj(arg.Xinv(0,parity,x_cb,j,k)), yHat);
      }
      arg.Yhat(d,1-parity,back_idx,i,j) = yHat;

    }

    // now do the forwards links X^{-1} * Y^{-\mu}
    complex<Float> yHat = 0.0;
#pragma unroll
    for (int k = 0; k<n; k++) {
      caxpy(arg.Xinv(0,parity,x_cb,i,k), arg.Y(d+4,parity,x_cb,k,j), yHat);
    }
    arg.Yhat(d+4,parity,x_cb,i,j) = yHat;

  }

  template<typename Float, int n, typename Arg>
  void CalculateYhatCPU(Arg &arg) {

    for (int d=0; d<4; d++) {
      for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
	for (int x_cb=0; x_cb<arg.Y.VolumeCB(); x_cb++) {
	  for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)
              computeYhat<Float,n>(arg, d, x_cb, parity, i, j);
	} // x_cb
      } //parity
    } // dimension
  }

  template<typename Float, int n, typename Arg>
  __global__ void CalculateYhatGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int i_parity = blockDim.y*blockIdx.y + threadIdx.y;
    if (i_parity >= 2*n) return;
    int j_d = blockDim.z*blockIdx.z + threadIdx.z;
    if (j_d >= 4*n) return;

    int i = i_parity % n;
    int parity = i_parity / n;
    int j = j_d % n;
    int d = j_d / n;
    
    computeYhat<Float,n>(arg, d, x_cb, parity, i, j);
  }

} // namespace quda

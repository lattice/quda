#include <gauge_field_order.h>
#include <index_helper.cuh>

namespace quda {

  template <typename Float_, typename PreconditionedGauge, typename Gauge, int n_> struct CalculateYhatArg {
    using Float = Float_;
    static constexpr int n = n_;
    PreconditionedGauge Yhat;
    const Gauge Y;
    const Gauge Xinv;
    int dim[QUDA_MAX_DIM];
    int comm_dim[QUDA_MAX_DIM];
    int nFace;
    const int coarseVolumeCB;   /** Coarse grid volume */

    Float *max_h;  // host scalar that stores the maximum element of Yhat. Pointer b/c pinned.
    Float *max_d; // device scalar that stores the maximum element of Yhat

    CalculateYhatArg(const PreconditionedGauge &Yhat, const Gauge Y, const Gauge Xinv, const int *dim,
                     const int *comm_dim, int nFace) :
      Yhat(Yhat),
      Y(Y),
      Xinv(Xinv),
      nFace(nFace),
      coarseVolumeCB(Y.VolumeCB()),
      max_h(nullptr),
      max_d(nullptr)
    {
      for (int i=0; i<4; i++) {
        this->comm_dim[i] = comm_dim[i];
        this->dim[i] = dim[i];
      }
    }
  };

  template <bool compute_max_only, typename Arg>
  inline __device__ __host__ typename Arg::Float computeYhat(Arg &arg, int d, int x_cb, int parity, int i, int j)
  {
    using Float = typename Arg::Float;
    constexpr int nDim = 4;
    int coord[nDim];
    getCoords(coord, x_cb, arg.dim, parity);

    const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

    Float yHatMax = 0.0;

    // first do the backwards links Y^{+\mu} * X^{-\dagger}
    if ( arg.comm_dim[d] && (coord[d] - arg.nFace < 0) ) {

      complex<Float> yHat = 0.0;
#pragma unroll
      for (int k = 0; k<Arg::n; k++) {
        yHat = cmac(arg.Y.Ghost(d,1-parity,ghost_idx,i,k), conj(arg.Xinv(0,parity,x_cb,j,k)), yHat);
      }
      if (compute_max_only) {
        yHatMax = fmax(fabs(yHat.x), fabs(yHat.y));
      } else {
        arg.Yhat.Ghost(d, 1 - parity, ghost_idx, i, j) = yHat;
      }

    } else {
      const int back_idx = linkIndexM1(coord, arg.dim, d);

      complex<Float> yHat = 0.0;
#pragma unroll
      for (int k = 0; k<Arg::n; k++) {
        yHat = cmac(arg.Y(d,1-parity,back_idx,i,k), conj(arg.Xinv(0,parity,x_cb,j,k)), yHat);
      }
      if (compute_max_only) {
        yHatMax = fmax(fabs(yHat.x), fabs(yHat.y));
      } else {
        arg.Yhat(d, 1 - parity, back_idx, i, j) = yHat;
      }
    }

    // now do the forwards links X^{-1} * Y^{-\mu}
    complex<Float> yHat = 0.0;
#pragma unroll
    for (int k = 0; k<Arg::n; k++) {
      yHat = cmac(arg.Xinv(0,parity,x_cb,i,k), arg.Y(d+4,parity,x_cb,k,j), yHat);
    }
    if (compute_max_only) {
      yHatMax = fmax(yHatMax, fmax(fabs(yHat.x), fabs(yHat.y)));
    } else {
      arg.Yhat(d + 4, parity, x_cb, i, j) = yHat;
    }

    return yHatMax;
  }

  template <bool compute_max_only, typename Arg> void CalculateYhatCPU(Arg &arg)
  {
    typename Arg::Float max = 0.0;
    for (int d=0; d<4; d++) {
      for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
        for (int x_cb = 0; x_cb < arg.Y.VolumeCB(); x_cb++) {
          for (int i = 0; i < Arg::n; i++)
            for (int j = 0; j < Arg::n; j++) {
              typename Arg::Float max_x = computeYhat<compute_max_only>(arg, d, x_cb, parity, i, j);
              if (compute_max_only) max = max > max_x ? max : max_x;
            }
        }
      } //parity
    } // dimension
    if (compute_max_only) *arg.max_h = max;
  }

  template <bool compute_max_only, typename Arg> __global__ void CalculateYhatGPU(Arg arg)
  {
    constexpr auto n = Arg::n;
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

    typename Arg::Float max = computeYhat<compute_max_only>(arg, d, x_cb, parity, i, j);
    if (compute_max_only) atomicAbsMax(arg.max_d, max);
  }

} // namespace quda

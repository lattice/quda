#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <kernel.h>
#include <reduction_kernel.h>

namespace quda {

  /**
   * @brief container to pass parameters for the gauge fixing quality kernel
   */
  template <typename store_t, QudaReconstructType recon_, int gauge_dir_>
  struct GaugeFixQualityOVRArg : public ReduceArg<double2> {
    using real = typename mapper<store_t>::type;
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<store_t, recon>::type;
    static constexpr int gauge_dir = gauge_dir_;

    int X[4]; // grid dimensions
    int border[4];
    Gauge data;
    double2 result;
    dim3 threads; // number of active threads required

    GaugeFixQualityOVRArg(const GaugeField &data) :
      ReduceArg<double2>(1, true), // reset = true
      threads(data.LocalVolumeCB(), 2, 1),
      data(data)
    {
      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        border[dir] = data.R()[dir];
      }
    }

    __device__ __host__ double2 init() const { return zero<double2>(); }
    double getAction(){ return result.x; }
    double getTheta(){ return result.y; }
  };

  template <typename Arg> struct FixQualityOVR {

    using reduce_t = double2;
    Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr FixQualityOVR(Arg &arg) : arg(arg) {}

    /**
     * @brief Measure gauge fixing quality
     */
    template <typename Reducer>
    __device__ __host__ inline reduce_t operator()(reduce_t &value, Reducer &r, int x_cb, int parity)
    {
      reduce_t data;
      using Link = Matrix<complex<typename Arg::real>, 3>;

      int X[4];
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      Link delta;
      setZero(&delta);
      //load upward links
      for ( int mu = 0; mu < Arg::gauge_dir; mu++ ) {
        Link U = arg.data(mu, linkIndex(x, X), parity);
        delta -= U;
      }
      //18*gauge_dir
      data.x = -delta(0, 0).real() - delta(1, 1).real() - delta(2, 2).real();
      //2
      //load downward links
      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        Link U = arg.data(mu, linkIndexM1(x,X,mu), 1 - parity);
        delta += U;
      }
      //18*gauge_dir
      delta -= conj(delta);
      //18
      SubTraceUnit(delta);
      //12
      data.y = getRealTraceUVdagger(delta, delta);
      //35
      //T=36*gauge_dir+65

      return r(data, value);
    }
  };

  /**
   * @brief container to pass parameters for the gauge fixing kernel
   */
  template <typename Float, typename Gauge>
  struct GaugeFixArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    int border[4];
    Gauge dataOr;
    GaugeField &data;
    const Float relax_boost;

    GaugeFixArg(Gauge & dataOr, GaugeField & data, const Float relax_boost)
      : dataOr(dataOr), data(data), relax_boost(relax_boost) {

      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        border[dir] = data.R()[dir];
      }
      threads = X[0] * X[1] * X[2] * X[3] >> 1;
    }
  };


  /**
   * @brief Kernel to perform gauge fixing with overrelaxation for single-GPU
   */
  template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFix(GaugeFixArg<Float, Gauge> arg, int parity)
  {
    typedef complex<Float> Cmplx;
    int tid = (threadIdx.x + blockSize) % blockSize;
    int idx = blockIdx.x * blockSize + tid;

    if ( idx >= arg.threads ) return;

    // 8 threads per lattice site
    if ( ImplementationType < 3 ) {
      int X[4];
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, idx, X, parity);
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      int mu = (threadIdx.x / blockSize);
      int oddbit = parity;
      if ( threadIdx.x >= blockSize * 4 ) {
        mu -= 4;
        x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
        oddbit = 1 - parity;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, oddbit);
      // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 8x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 0 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 threads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 1 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 2 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      arg.dataOr(mu, idx, oddbit) = link;
    }
    // 4 threads per lattice site
    else{
      int X[4];
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, idx, X, parity);
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      int mu = (threadIdx.x / blockSize);
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      //load upward link
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, parity);

      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      int idx1 = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      //load downward link
      Matrix<Cmplx,3> link1 = arg.dataOr(mu, idx1, 1 - parity);

      // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 4x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 3 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 threads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 4 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 5 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

      arg.dataOr(mu, idx, parity) = link;
      arg.dataOr(mu, idx1, 1 - parity) = link1;
    }
  }

  template <typename Float, typename Gauge>
  struct GaugeFixInteriorPointsArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    int border[4];
    Gauge dataOr;
    GaugeField &data;
    const Float relax_boost;
    GaugeFixInteriorPointsArg(Gauge & dataOr, GaugeField & data, const Float relax_boost)
      : dataOr(dataOr), data(data), relax_boost(relax_boost)
    {
      for ( int dir = 0; dir < 4; ++dir ) {
        if ( comm_dim_partitioned(dir)) border[dir] = data.R()[dir] + 1;  //skip BORDER_RADIUS + face border point
        else border[dir] = 0;
      }
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir] - border[dir] * 2;
      threads = X[0] * X[1] * X[2] * X[3] >> 1;
    }
  };


  /**
   * @brief Kernel to perform gauge fixing with overrelaxation in the interior points for multi-GPU implementation
   */
  template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFixInteriorPoints(GaugeFixInteriorPointsArg<Float, Gauge> arg, int parity){
    int tid = (threadIdx.x + blockSize) % blockSize;
    int idx = blockIdx.x * blockSize + tid;
    if ( idx >= arg.threads ) return;
    typedef complex<Float> Complex;
    int X[4];
#pragma unroll
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];

    int za = (idx / (X[0] / 2));
    int zb =  (za / X[1]);
    x[1] = za - zb * X[1];
    x[3] = (zb / X[2]);
    x[2] = zb - x[3] * X[2];
    int p = 0; for ( int dr = 0; dr < 4; ++dr ) p += arg.border[dr];
    p = p & 1;
    int x1odd = (x[1] + x[2] + x[3] + parity + p) & 1;
    //int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    x[0] = (2 * idx + x1odd)  - za * X[0];
    for ( int dr = 0; dr < 4; ++dr ) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }

    int mu = (threadIdx.x / blockSize);

    // 8 threads per lattice site
    if ( ImplementationType < 3 ) {
      if ( threadIdx.x >= blockSize * 4 ) {
        mu -= 4;
        x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
        parity = 1 - parity;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Complex,3> link = arg.dataOr(mu, idx, parity);
      // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 8x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 0 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 threads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 1 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 2 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      arg.dataOr(mu, idx, parity) = link;
    }
    // 4 threads per lattice site
    else{
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Complex,3> link = arg.dataOr(mu, idx, parity);


      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      int idx1 = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Complex,3> link1 = arg.dataOr(mu, idx1, 1 - parity);

      // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 4x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 3 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 threads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 4 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 5 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

      arg.dataOr(mu, idx, parity) = link;
      arg.dataOr(mu, idx1, 1 - parity) = link1;
    }
  }

  void PreCalculateLatticeIndices(size_t faceVolume[4], size_t faceVolumeCB[4], int X[4], int border[4],
                                  int &threads, int *borderpoints[2]);

  template <typename Float, typename Gauge>
  struct GaugeFixBorderPointsArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    int border[4];
    int *borderpoints[2];
    int *faceindicessize[2];
    size_t faceVolume[4];
    size_t faceVolumeCB[4];
    Gauge dataOr;
    GaugeField &data;
    const Float relax_boost;

    GaugeFixBorderPointsArg(Gauge & dataOr, GaugeField & data, const Float relax_boost, size_t faceVolume_[4], size_t faceVolumeCB_[4])
      : dataOr(dataOr), data(data), relax_boost(relax_boost)
    {
      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        border[dir] = data.R()[dir];
      }

      for ( int dir = 0; dir < 4; ++dir ) {
        faceVolume[dir] = faceVolume_[dir];
        faceVolumeCB[dir] = faceVolumeCB_[dir];
      }
      if (comm_partitioned()) PreCalculateLatticeIndices(faceVolume, faceVolumeCB, X, border, threads, borderpoints);
    }
  };

  /**
   * @brief Kernel to perform gauge fixing with overrelaxation in the border points for multi-GPU implementation
  */
  template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFixBorderPoints(GaugeFixBorderPointsArg<Float, Gauge> arg, int parity){
    typedef complex<Float> Cmplx;

    int tid = (threadIdx.x + blockSize) % blockSize;
    int idx = blockIdx.x * blockSize + tid;
    if ( idx >= arg.threads ) return;
    int mu = (threadIdx.x / blockSize);
    idx = arg.borderpoints[parity][idx];
    int X[4], x[4];
    x[3] = idx / (arg.X[0] * arg.X[1]  * arg.X[2]);
    x[2] = (idx / (arg.X[0] * arg.X[1])) % arg.X[2];
    x[1] = (idx / arg.X[0]) % arg.X[1];
    x[0] = idx % arg.X[0];
  #pragma unroll
    for ( int dr = 0; dr < 4; ++dr ) x[dr] += arg.border[dr];
  #pragma unroll
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr] + 2 * arg.border[dr];

    // 8 threads per lattice site
    if ( ImplementationType < 3 ) {
      if ( threadIdx.x >= blockSize * 4 ) {
        mu -= 4;
        x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
        parity = 1 - parity;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, parity);
      // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 8x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 0 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 threads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 1 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 2 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      arg.dataOr(mu, idx, parity) = link;
    }
    // 4 threads per lattice site
    else{
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, parity);


      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      int idx1 = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link1 = arg.dataOr(mu, idx1, 1 - parity);

      // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 4x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 3 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 threads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 4 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 5 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

      arg.dataOr(mu, idx, parity) = link;
      arg.dataOr(mu, idx1, 1 - parity) = link1;
    }
  }

  template <int NElems_, typename Gauge>
  struct GaugeFixUnPackArg {
    static constexpr int NElems = NElems_;
    int X[4]; // grid dimensions
    int border[4];
    Gauge dataOr;
    GaugeFixUnPackArg(Gauge & dataOr, GaugeField & data)
      : dataOr(dataOr) {
      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        border[dir] = data.R()[dir];
      }
    }
  };

  template <typename Float, bool pack, typename Arg>
  __global__ void Kernel_UnPackGhost(int size, Arg arg, complex<Float> *array, int parity, int face, int dir)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= size ) return;
    int X[4];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];
    int za, xodd;
    int borderid = 0;
    parity = 1 - parity;
    switch ( face ) {
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = borderid;
      xodd = (borderid + x[2] + x[3] + parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
      break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = borderid;
      xodd = (borderid  + x[2] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = borderid;
      xodd = (borderid  + x[1] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = borderid;
      xodd = (borderid  + x[1] + x[2] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    }
    for ( int dr = 0; dr < 4; ++dr ) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }
    x[face] -= 1;
    parity = 1 - parity;
    int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    typedef complex<Float> Cmplx;
    typedef typename mapper<Float>::type RegType;
    RegType tmp[Arg::NElems];
    Cmplx data[9];
    if ( pack ) {
      arg.dataOr.load(data, id, dir, parity);
      arg.dataOr.reconstruct.Pack(tmp, data, id);
      for ( int i = 0; i < Arg::NElems / 2; ++i ) {
        array[idx + size * i] = Cmplx(tmp[2*i+0], tmp[2*i+1]);
      }
    } else {
      for ( int i = 0; i < Arg::NElems / 2; ++i ) {
        tmp[2*i+0] = array[idx + size * i].real();
        tmp[2*i+1] = array[idx + size * i].imag();
      }
      arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0, arg.dataOr.X, arg.dataOr.R);
      arg.dataOr.save(data, id, dir, parity);
    }
  }

  template <typename Float, bool pack, typename Arg>
  __global__ void Kernel_UnPackTop(int size, Arg arg, complex<Float> *array, int parity, int face, int dir)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= size ) return;
    int X[4];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];
    int za, xodd;
    int borderid = arg.X[face] - 1;
    switch ( face ) {
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = borderid;
      xodd = (borderid + x[2] + x[3] + parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
      break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = borderid;
      xodd = (borderid  + x[2] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = borderid;
      xodd = (borderid  + x[1] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = borderid;
      xodd = (borderid  + x[1] + x[2] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    }
    for ( int dr = 0; dr < 4; ++dr ) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }
    int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    typedef complex<Float> Cmplx;
    typedef typename mapper<Float>::type RegType;
    RegType tmp[Arg::NElems];
    Cmplx data[9];
    if ( pack ) {
      arg.dataOr.load(data, id, dir, parity);
      arg.dataOr.reconstruct.Pack(tmp, data, id);
      for ( int i = 0; i < Arg::NElems / 2; ++i ) array[idx + size * i] = Cmplx(tmp[2*i+0], tmp[2*i+1]);
    }
    else{
      for ( int i = 0; i < Arg::NElems / 2; ++i ) {
        tmp[2*i+0] = array[idx + size * i].real();
        tmp[2*i+1] = array[idx + size * i].imag();
      }
      arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0, arg.dataOr.X, arg.dataOr.R);
      arg.dataOr.save(data, id, dir, parity);
    }
  }

  struct BorderIdArg {
    int X[4]; // grid dimensions
    int border[4];
    BorderIdArg(int X_[4], int border_[4]) {
      for ( int dir = 0; dir < 4; ++dir ) border[dir] = border_[dir];
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = X_[dir];
    }
  };

  __global__ void ComputeBorderPointsActiveFaceIndex(BorderIdArg arg, int *faceindices, int facesize, int faceid, int parity)
  {
    int idd = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idd < facesize ) {
      int borderid = 0;
      int idx = idd;
      if ( idx >= facesize / 2 ) {
        borderid = arg.X[faceid] - 1;
        idx -= facesize / 2;
      }
      int X[4];
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
      int x[4];
      int za, xodd;
      switch ( faceid ) {
      case 0: //X FACE
        za = idx / ( X[1] / 2);
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[0] = borderid;
        xodd = (borderid + x[2] + x[3] + parity) & 1;
        x[1] = (2 * idx + xodd)  - za * X[1];
        break;
      case 1: //Y FACE
        za = idx / ( X[0] / 2);
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[1] = borderid;
        xodd = (borderid + x[2] + x[3] + parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      case 2: //Z FACE
        za = idx / ( X[0] / 2);
        x[3] = za / X[1];
        x[1] = za - x[3] * X[1];
        x[2] = borderid;
        xodd = (borderid + x[1] + x[3] + parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      case 3: //T FACE
        za = idx / ( X[0] / 2);
        x[2] = za / X[1];
        x[1] = za - x[2] * X[1];
        x[3] = borderid;
        xodd = (borderid + x[1] + x[2] + parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]);;
      faceindices[idd] = idx;
    }
  }

}

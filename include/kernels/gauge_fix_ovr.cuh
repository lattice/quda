#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <array.h>
#include <kernel.h>
#include <reduction_kernel.h>
#include <gauge_fix_ovr_hit_devf.cuh>

namespace quda {

  /**
   * @brief container to pass parameters for the gauge fixing quality kernel
   */
  template <typename store_t, QudaReconstructType recon_, int gauge_dir_>
  struct GaugeFixQualityOVRArg : public ReduceArg<array<double, 2>> {
    using real = typename mapper<store_t>::type;
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<store_t, recon>::type;
    static constexpr int gauge_dir = gauge_dir_;

    int X[4]; // grid dimensions
    int border[4];
    Gauge data;
    reduce_t result;

    GaugeFixQualityOVRArg(const GaugeField &data) :
      ReduceArg<reduce_t>(dim3(data.LocalVolumeCB(), 2, 1), 1, true), // reset = true
      data(data),
      result{0, 0}
    {
      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        border[dir] = data.R()[dir];
      }
    }

    double getAction(){ return result[0]; }
    double getTheta(){ return result[1]; }
  };

  template <typename Arg> struct FixQualityOVR : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr FixQualityOVR(const Arg &arg) : arg(arg) {}

    /**
     * @brief Measure gauge fixing quality
     */
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      reduce_t data{0, 0};
      using Link = Matrix<complex<typename Arg::real>, 3>;

      int X[4];
#pragma unroll
      for (int dr = 0; dr < 4; dr++) X[dr] = arg.X[dr];

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
#pragma unroll
      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        Link U = arg.data(mu, linkIndex(x, X), parity);
        delta -= U;
      }
      //18*gauge_dir
      data[0] = -delta(0, 0).real() - delta(1, 1).real() - delta(2, 2).real();
      //2
      //load downward links
#pragma unroll
      for (int mu = 0; mu < Arg::gauge_dir; mu++) {
        Link U = arg.data(mu, linkIndexM1(x,X,mu), 1 - parity);
        delta += U;
      }
      //18*gauge_dir
      delta -= conj(delta);
      //18
      SubTraceUnit(delta);
      //12
      data[1] = getRealTraceUVdagger(delta, delta);
      //35
      //T=36*gauge_dir+65

      return operator()(data, value);
    }
  };

  /**
   * @brief container to pass parameters for the gauge fixing kernel
   */
  template <typename store_t, QudaReconstructType recon, int gauge_dir_, bool halo_, int type_>
  struct GaugeFixArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int gauge_dir = gauge_dir_;
    static constexpr bool halo = halo_;
    static constexpr int type = type_;
    typename gauge_mapper<store_t, recon>::type u;
    const real relax_boost;
    int parity;
    int X[4]; // grid dimensions
    int border[4];
    int *borderpoints[2];

    GaugeFixArg(const GaugeField &u, const double relax_boost, int parity, int *borderpoints[2], unsigned threads) :
      kernel_param(dim3(threads, type < 3 ? 8 : 4, 1)),
      u(u),
      relax_boost(static_cast<real>(relax_boost)),
      parity(parity),
      borderpoints{ borderpoints[0], borderpoints[1] }
    {
      for (int dir = 0; dir < 4; dir++) {
        border[dir] = halo ? u.R()[dir] : comm_dim_partitioned(dir) ? u.R()[dir] + 1 : 0;
        X[dir] = u.X()[dir] - border[dir] * 2;
      }
    }
  };

  /**
   * @brief Perform gauge fixing with overrelaxation
   */
  template <typename Arg> struct computeFix {
    const Arg &arg;
    constexpr computeFix(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()(int idx, int mu)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, 3>;
      int parity = arg.parity;

      int X[4], x[4];
#pragma unroll
      for (int dr = 0; dr < 4; dr++) X[dr] = arg.X[dr];

      if (!Arg::halo) {
        int p = 0;
        for (int dr = 0; dr < 4; dr++) p += arg.border[dr];
        getCoords(x, idx, arg.X, p + parity);
      } else {
        idx = arg.borderpoints[parity][idx];  // load the lattice site assigment
        x[3] = idx / (X[0] * X[1]  * X[2]);
        x[2] = (idx / (X[0] * X[1])) % X[2];
        x[1] = (idx / X[0]) % X[1];
        x[0] = idx % X[0];
      }

#pragma unroll
      for (int dr = 0; dr < 4; dr++) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      if (Arg::type < 3) {
        // 8 threads per lattice site
        int dim = mu;
        if (dim >= 4) {
          dim -= 4;
          switch (dim) {
          case 0: x[0] = (x[0] - 1 + X[0]) % X[0]; break;
          case 1: x[1] = (x[1] - 1 + X[1]) % X[1]; break;
          case 2: x[2] = (x[2] - 1 + X[2]) % X[2]; break;
          case 3: x[3] = (x[3] - 1 + X[3]) % X[3]; break;
          }
          parity = 1 - parity;
        }
        idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
        Link link = arg.u(dim, idx, parity);

        switch (Arg::type) {
          // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
          // this implementation needs 8x more shared memory than the implementation using atomicadd
        case 0: GaugeFixHit_NoAtomicAdd<real, Arg::gauge_dir, 3>(link, arg.relax_boost, mu); break;
          // 8 threads per lattice site, the reduction is performed by shared memory using atomicadd
        case 1: GaugeFixHit_AtomicAdd<real, Arg::gauge_dir, 3>(link, arg.relax_boost, mu); break;
          // 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
          // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
        case 2: GaugeFixHit_NoAtomicAdd_LessSM<real, Arg::gauge_dir, 3>(link, arg.relax_boost, mu); break;
        default: break;
        }

        arg.u(dim, idx, parity) = link;
      } else {
        // 4 threads per lattice site
        idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
        Link link = arg.u(mu, idx, parity);

        switch (mu) {
        case 0: x[0] = (x[0] - 1 + X[0]) % X[0]; break;
        case 1: x[1] = (x[1] - 1 + X[1]) % X[1]; break;
        case 2: x[2] = (x[2] - 1 + X[2]) % X[2]; break;
        case 3: x[3] = (x[3] - 1 + X[3]) % X[3]; break;
        }
        int idx1 = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
        Link link1 = arg.u(mu, idx1, 1 - parity);

        switch (Arg::type) {
          // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
          // this implementation needs 4x more shared memory than the implementation using atomicadd
        case 3: GaugeFixHit_NoAtomicAdd<real, Arg::gauge_dir, 3>(link, link1, arg.relax_boost, mu); break;
          // 4 threads per lattice site, the reduction is performed by shared memory using atomicadd
        case 4: GaugeFixHit_AtomicAdd<real, Arg::gauge_dir, 3>(link, link1, arg.relax_boost, mu); break;
          // 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
          // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
        case 5: GaugeFixHit_NoAtomicAdd_LessSM<real, Arg::gauge_dir, 3>(link, link1, arg.relax_boost, mu); break;
        default: break;
        }

        arg.u(mu, idx, parity) = link;
        arg.u(mu, idx1, 1 - parity) = link1;
      }
    }
  };

  template <typename store_t_, QudaReconstructType recon, bool pack_, bool top_>
  struct GaugeFixPackArg : kernel_param<> {
    using store_t = store_t_;
    using real = typename mapper<store_t>::type;
    static constexpr int NElems = recon;
    static constexpr bool pack = pack_;
    static constexpr bool top = top_;
    typename gauge_mapper<store_t, recon>::type u;
    complex<store_t> *array;
    int parity;
    int dim;
    int X[4]; // grid dimensions
    int border[4];

    GaugeFixPackArg(GaugeField &u, complex<store_t> *array, int parity, int dim) :
      kernel_param(dim3(u.LocalSurfaceCB(dim), 1, 1)),
      u(u),
      array(array),
      parity(parity),
      dim(dim)
    {
      for (int dir = 0; dir < 4; dir++) {
        X[dir] = u.X()[dir] - u.R()[dir] * 2;
        border[dir] = u.R()[dir];
      }
    }
  };

  template <typename Arg> struct Packer
  {
    const Arg &arg;
    constexpr Packer(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      int X[4];
      for (int dr = 0; dr < 4; dr++) X[dr] = arg.X[dr];
      int x[4];
      int za, xodd;
      int borderid = Arg::top ? arg.X[arg.dim] - 1 : 0;
      int parity = Arg::top ? arg.parity : 1 - arg.parity;
      switch (arg.dim) {
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
      if (!Arg::top) {
        switch (arg.dim) {
        case 0: x[0] -= 1; break;
        case 1: x[1] -= 1; break;
        case 2: x[2] -= 1; break;
        case 3: x[3] -= 1; break;
        }
        parity = 1 - parity;
      }
      int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      using complex = complex<typename Arg::store_t>;
      typename Arg::real tmp[Arg::NElems];
      complex data[9];
      if (Arg::pack) {
        arg.u.load(data, id, arg.dim, parity);
        arg.u.reconstruct.Pack(tmp, data);
        for ( int i = 0; i < Arg::NElems / 2; ++i ) arg.array[idx + arg.threads.x * i] = complex(tmp[2*i+0], tmp[2*i+1]);
      } else {
        for ( int i = 0; i < Arg::NElems / 2; ++i ) {
          tmp[2*i+0] = arg.array[idx + arg.threads.x * i].real();
          tmp[2*i+1] = arg.array[idx + arg.threads.x * i].imag();
        }
        arg.u.reconstruct.Unpack(data, tmp, id, arg.dim, 0, arg.u.X, arg.u.R);
        arg.u.save(data, id, arg.dim, parity);
      }
    }
  };

  struct BorderIdArg : kernel_param<> {
    int X[4]; // grid dimensions
    int border[4];
    int *faceindices[8];
    int offset[5];
    int surface_cb[4];
    BorderIdArg(const GaugeField &u, int *faceindices[2]) :
      kernel_param(dim3(0, 2, 2)),
      faceindices{faceindices[0], faceindices[1]}
    {
      offset[0] = 0;
      for (int dim = 0; dim < 4; dim++) {
        border[dim] = u.R()[dim];
        X[dim] = u.LocalX()[dim];
        if (comm_dim_partitioned(dim)) threads.x += u.LocalSurfaceCB(dim);
        offset[dim+1] = threads.x;
        surface_cb[dim] = u.LocalSurfaceCB(dim);
      }
    }
  };

  template <typename Arg> struct BorderPointsCompute {
    const Arg &arg;
    constexpr BorderPointsCompute(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int idd, int parity, int dir)
    {
      int dim = 0;
      for (int d = 0; d < 4; d++) if (idd >= arg.offset[d] && idd < arg.offset[d+1]) { dim = d; break; }

      int idx = idd - arg.offset[dim];
      int borderid = dir == 0 ? 0 : arg.X[dim] - 1;

      int X[4];
      for (int dr = 0; dr < 4; dr++) X[dr] = arg.X[dr];
      int x[4];
      int za, xodd;
      switch (dim) {
      case 0: //X FACE
        za = idx / ( X[1] / 2);
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[0] = borderid;
        xodd = (borderid + x[2] + x[3] + parity) & 1;
        x[1] = (2 * idx + xodd) - za * X[1];
        break;
      case 1: //Y FACE
        za = idx / ( X[0] / 2);
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[1] = borderid;
        xodd = (borderid + x[2] + x[3] + parity) & 1;
        x[0] = (2 * idx + xodd) - za * X[0];
        break;
      case 2: //Z FACE
        za = idx / ( X[0] / 2);
        x[3] = za / X[1];
        x[1] = za - x[3] * X[1];
        x[2] = borderid;
        xodd = (borderid + x[1] + x[3] + parity) & 1;
        x[0] = (2 * idx + xodd) - za * X[0];
        break;
      case 3: //T FACE
        za = idx / ( X[0] / 2);
        x[2] = za / X[1];
        x[1] = za - x[2] * X[1];
        x[3] = borderid;
        xodd = (borderid + x[1] + x[2] + parity) & 1;
        x[0] = (2 * idx + xodd) - za * X[0];
        break;
      }

      int bulk_idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]);
      arg.faceindices[parity][arg.offset[dim] * 2 + arg.surface_cb[dim] * dir + idx] = bulk_idx;
    }
  };

}

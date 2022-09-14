#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <thread_array.h>
#include <kernel.h>
#include <gauge_path_helper.cuh>

namespace quda {

  template <typename store_t, int nColor_, QudaReconstructType recon_u, QudaReconstructType recon_m, bool force_>
  struct GaugeForceArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static constexpr bool compute_force = force_;
    using Link = Matrix<complex<real>, nColor>;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    using Gauge = typename gauge_mapper<real,recon_u>::type;
    using Mom = typename gauge_mapper<real,recon_m>::type;

    Mom mom;
    const Gauge u;

    int X[4]; // the regular volume parameters
    int E[4]; // the extended volume parameters
    int border[4]; // radius of border

    real epsilon; // stepsize and any other overall scaling factor
    const paths<4> p;

    GaugeForceArg(GaugeField &mom, const GaugeField &u, double epsilon, const paths<4> &p) :
      kernel_param(dim3(mom.VolumeCB(), 2, 4)),
      mom(mom),
      u(u),
      epsilon(epsilon),
      p(p)
    {
      for (int i=0; i<4; i++) {
        X[i] = mom.X()[i];
        E[i] = u.X()[i];
        border[i] = (E[i] - X[i])/2;
      }
    }
  };

  template <typename Arg> struct GaugeForce
  {
    const Arg &arg;
    constexpr GaugeForce(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }    

    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = typename Arg::Link;

      int x[4] = {0, 0, 0, 0};
      getCoords(x, x_cb, arg.X, parity);
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      // prod: current matrix product
      // accum: accumulator matrix
      Link link_prod, accum;
      thread_array<int, 4> dx{0};

      for (int i=0; i<arg.p.num_paths; i++) {
        real coeff = arg.p.path_coeff[i];
        if (coeff == 0) continue;

        const int* path = arg.p.input_path[dir] + i*arg.p.max_length;

        // the gauge path starts pre-shifted, so we need to do the shift + update the parity
        dx[dir]++;
        int nbr_oddbit = (parity ^ 1);

        // compute the path
        link_prod = computeGaugePath(arg, x, nbr_oddbit, path, arg.p.length[i], dx);

        accum = accum + coeff * link_prod;
      } //i

      // multiply by U(x)
      link_prod = arg.u(dir, linkIndex(x,arg.E), parity);
      link_prod = link_prod * accum;

      // update mom(x)
      Link mom = arg.mom(dir, x_cb, parity);
      if (arg.compute_force) {
        mom = mom - arg.epsilon * link_prod;
        makeAntiHerm(mom);
      } else {
        mom = mom + arg.epsilon * link_prod;
      }
      arg.mom(dir, x_cb, parity) = mom;
    }
  };

}

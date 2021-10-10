#pragma once

#include <index_helper.cuh>
#include <gauge_field_order.h>
#include <fast_intdiv.h>
#include <kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon>
  struct LinkArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type Link;
    typedef typename gauge_mapper<Float, recon, 18, QUDA_STAGGERED_PHASE_MILC>::type Gauge;

    Link link;
    Gauge u;
    Float coeff;

    int_fastdiv X[4];
    int_fastdiv E[4];
    int border[4];

    /** This keeps track of any parity changes that result in using a
    radius of 1 for the extended border (the staple computations use
    such an extension, and if an odd number of dimensions are
    partitioned then we have to correct for this when computing the local index */
    int odd_bit;

    LinkArg(GaugeField &link, const GaugeField &u, Float coeff) :
      kernel_param(dim3(link.VolumeCB(), 2, 4)),
      link(link),
      u(u),
      coeff(coeff)
    {
      if (u.StaggeredPhase() != QUDA_STAGGERED_PHASE_MILC && u.Reconstruct() != QUDA_RECONSTRUCT_NO)
        errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
      for (int d=0; d<4; d++) {
        X[d] = link.X()[d];
        E[d] = u.X()[d];
        border[d] = (E[d] - X[d]) / 2;
      }
    }
  };

  template <int dir, typename Arg>
  __device__ void longLinkDir(const Arg &arg, int idx, int parity) {
    int x[4];
    int dx[4] = {0, 0, 0, 0};

    getCoords(x, idx, arg.X, parity);
    for (int d=0; d<4; d++) x[d] += arg.border[d];

    using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;

    Link a = arg.u(dir, linkIndex(x, arg.E), parity);

    dx[dir]++;
    Link b = arg.u(dir, linkIndexShift(x, dx, arg.E), 1-parity);

    dx[dir]++;
    Link c = arg.u(dir, linkIndexShift(x, dx, arg.E), parity);
    dx[dir]-=2;

    arg.link(dir, idx, parity) = arg.coeff * a * b * c;
  }

  template <typename Arg> struct ComputeLongLink
  {
    const Arg &arg;
    constexpr ComputeLongLink(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      switch(dir) {
      case 0: longLinkDir<0>(arg, x_cb, parity); break;
      case 1: longLinkDir<1>(arg, x_cb, parity); break;
      case 2: longLinkDir<2>(arg, x_cb, parity); break;
      case 3: longLinkDir<3>(arg, x_cb, parity); break;
      }
    }
  };

  template <typename Arg> struct ComputeOneLink
  {
    const Arg &arg;
    constexpr ComputeOneLink(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int d=0; d<4; d++) x[d] += arg.border[d];

      using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;

      Link a = arg.u(dir, linkIndex(x,arg.E), parity);
      arg.link(dir, x_cb, parity) = arg.coeff*a;
    }
  };

  template <typename Float_, int nColor_, QudaReconstructType recon, QudaReconstructType recon_mu, bool save_staple_>
  struct StapleArg : kernel_param<> {
    using Float = Float_;
    using Link = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    using Gauge = typename gauge_mapper<Float, recon, 18, QUDA_STAGGERED_PHASE_MILC>::type;
    using MuLink = typename gauge_mapper<Float, recon_mu, 18, QUDA_STAGGERED_PHASE_MILC>::type;
    static constexpr int nColor = nColor_;
    static constexpr bool save_staple = save_staple_;

    int_fastdiv X[4];
    int_fastdiv E[4];
    int border[4];

    int_fastdiv inner_X[4];
    int inner_border[4];

    Link fat;
    Link staple;
    MuLink mulink;
    Gauge u;
    Float coeff;

    int nu;
    int mu_map[4];

    /** This keeps track of any parity changes that result in using a
    radius of 1 for the extended border (the staple computations use
    such an extension, and if an odd number of dimensions are
    partitioned then we have to correct for this when computing the local index */
    int odd_bit;

    StapleArg(GaugeField &fat, GaugeField &staple, const GaugeField &mulink, const GaugeField &u,
              Float coeff, int nu, int mu_map[4]) :
      kernel_param(dim3(1, 2, 1)),
      fat(fat),
      staple(staple),
      mulink(mulink),
      u(u),
      coeff(coeff),
      nu(nu),
      odd_bit( (commDimPartitioned(0)+commDimPartitioned(1) +
                commDimPartitioned(2)+commDimPartitioned(3))%2 )
    {
      for (int d=0; d<4; d++) {
        X[d] = (fat.X()[d] + u.X()[d]) / 2;
        E[d] = u.X()[d];
        border[d] = (E[d] - X[d]) / 2;
        this->threads.x *= X[d];

        inner_X[d] = fat.X()[d];
        inner_border[d] = (E[d] - inner_X[d]) / 2;

        this->mu_map[d] = mu_map[d];
      }
      this->threads.x /= 2; // account for parity in y dimension
    }
  };

  template <int mu, int nu, typename Arg>
  __device__ inline void computeStaple(Matrix<complex<typename Arg::Float>, Arg::nColor> &staple, const Arg &arg, int x[], int parity)
  {
    using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;
    int dx[4] = {0, 0, 0, 0};

    /* Computes the upper staple :
     *                 mu (B)
     *               +-------+
     *       nu	   |	   |
     *	     (A)   |	   |(C)
     *		   X	   X
     */
    {
      /* load matrix A*/
      Link a = arg.u(nu, linkIndex(x, arg.E), parity);

      /* load matrix B*/
      dx[nu]++;
      Link b = arg.mulink(mu, linkIndexShift(x, dx, arg.E), 1-parity);
      dx[nu]--;

      /* load matrix C*/
      dx[mu]++;
      Link c = arg.u(nu, linkIndexShift(x, dx, arg.E), 1-parity);
      dx[mu]--;

      staple = a * b * conj(c);
    }

    /* Computes the lower staple :
     *                 X       X
     *           nu    |       |
     *	         (A)   |       | (C)
     *		       +-------+
     *                  mu (B)
     */
    {
      /* load matrix A*/
      dx[nu]--;
      Link a = arg.u(nu, linkIndexShift(x, dx, arg.E), 1-parity);

      /* load matrix B*/
      Link b = arg.mulink(mu, linkIndexShift(x, dx, arg.E), 1-parity);

      /* load matrix C*/
      dx[mu]++;
      Link c = arg.u(nu, linkIndexShift(x, dx, arg.E), parity);
      dx[mu]--;
      dx[nu]++;

      staple = staple + conj(a)*b*c;
    }
  }

  template <typename Arg> struct ComputeStaple
  {
    const Arg &arg;
    constexpr ComputeStaple(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity, int mu_idx)
    {
      int mu;
      switch(mu_idx) {
      case 0: mu = arg.mu_map[0]; break;
      case 1: mu = arg.mu_map[1]; break;
      case 2: mu = arg.mu_map[2]; break;
      }

      int x[4];
      getCoords(x, x_cb, arg.X, (parity+arg.odd_bit)%2);
      for (int d=0; d<4; d++) x[d] += arg.border[d];

      using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;
      Link staple;
      switch (mu) {
      case 0:
        switch (arg.nu) {
        case 1: computeStaple<0,1>(staple, arg, x, parity); break;
        case 2: computeStaple<0,2>(staple, arg, x, parity); break;
        case 3: computeStaple<0,3>(staple, arg, x, parity); break;
        } break;
      case 1:
        switch (arg.nu) {
        case 0: computeStaple<1,0>(staple, arg, x, parity); break;
        case 2: computeStaple<1,2>(staple, arg, x, parity); break;
        case 3: computeStaple<1,3>(staple, arg, x, parity); break;
        } break;
      case 2:
        switch (arg.nu) {
        case 0: computeStaple<2,0>(staple, arg, x, parity); break;
        case 1: computeStaple<2,1>(staple, arg, x, parity); break;
        case 3: computeStaple<2,3>(staple, arg, x, parity); break;
      } break;
      case 3:
        switch (arg.nu) {
        case 0: computeStaple<3,0>(staple, arg, x, parity); break;
        case 1: computeStaple<3,1>(staple, arg, x, parity); break;
        case 2: computeStaple<3,2>(staple, arg, x, parity); break;
        } break;
      }

      // exclude inner halo
      if ( !(x[0] < arg.inner_border[0] || x[0] >= arg.inner_X[0] + arg.inner_border[0] ||
             x[1] < arg.inner_border[1] || x[1] >= arg.inner_X[1] + arg.inner_border[1] ||
             x[2] < arg.inner_border[2] || x[2] >= arg.inner_X[2] + arg.inner_border[2] ||
             x[3] < arg.inner_border[3] || x[3] >= arg.inner_X[3] + arg.inner_border[3]) ) {
        // convert to inner coords
        int inner_x[] = {x[0]-arg.inner_border[0], x[1]-arg.inner_border[1], x[2]-arg.inner_border[2], x[3]-arg.inner_border[3]};
        Link fat = arg.fat(mu, linkIndex(inner_x, arg.inner_X), parity);
        fat += arg.coeff * staple;
        arg.fat(mu, linkIndex(inner_x, arg.inner_X), parity) = fat;
      }

      if (arg.save_staple) arg.staple(mu, linkIndex(x, arg.E), parity) = staple;
    }
  };

}

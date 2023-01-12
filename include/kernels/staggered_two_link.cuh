#pragma once

#include <quda_matrix.h>
#include <index_helper.cuh>
#include <gauge_field_order.h>
#include <kernel.h>

namespace quda
{

  namespace staggered_quark_smearing
  {

    template <typename real_, int nColor_, QudaReconstructType reconstruct = QUDA_RECONSTRUCT_NO>
    struct BaseArg : kernel_param<> {
      using real = real_;
      static constexpr int nColor = nColor_;
      typedef typename gauge_mapper<real, reconstruct>::type G;
      const G link;
      int X[4]; // regular grid dims
      int E[4]; // extended grid dims

      int commDim[4];
      int border[4];

      /**
         @param[in] link Gauge field
       */
      BaseArg(const GaugeField &link) :
        kernel_param(dim3(1, 2, 4)),
        link(link),
        commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)}
      {
        for (int d = 0; d < 4; d++) {
          E[d] = link.X()[d];
          border[d] = link.R()[d];
          X[d] = E[d] - 2 * border[d];
          this->threads.x *= X[d];
        }
        this->threads.x /= 2;
      }
    };

    template <typename real, int nColor, QudaReconstructType reconstruct = QUDA_RECONSTRUCT_NO>
    struct TwoLinkArg : public BaseArg<real, nColor, reconstruct> {

      typedef typename gauge_mapper<real, QUDA_RECONSTRUCT_NO>::type F;
      F outA;

      TwoLinkArg(GaugeField &twoLink, const GaugeField &link) : BaseArg<real, nColor, reconstruct>(link), outA(twoLink)
      {
      }
    };

    template <typename Arg> struct TwoLink {
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      constexpr TwoLink(const Arg &arg) : arg(arg) { }
      constexpr static const char *filename() { return KERNEL_FILE; }

      // Flops count, in two-number pair (matrix_mult, matrix_add)
      // 				   (4, 4)
      // 198 Flops per direction
      __device__ __host__ void operator()(int x_cb, int parity, int mu)
      {
        int x[4];
        int dx[4] = {0, 0, 0, 0};
        dx[mu] = 1;

        getCoords(x, x_cb, arg.X, parity);

        for (int i = 0; i < 4; i++) x[i] += arg.border[i];
        int e_cb = linkIndex(x, arg.E);

        /*
         *
         *    C    D
         *    ---- ----
         *
         *   ---> sig direction
         *
         *   C is the current point (sid)
         *
         */

        // compute the forward two links
        int point_c = e_cb;
        int point_d = linkIndexShift(x, dx, arg.E);

        Link Ucd = arg.link(mu, point_c, parity);
        Link Ude = arg.link(mu, point_d, 1 - parity);

        arg.outA(mu, x_cb, parity) = Ucd * Ude;
      }
    };

  } // namespace staggered_quark_smearing
} // namespace quda

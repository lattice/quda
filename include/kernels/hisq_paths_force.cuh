#pragma once

#include <quda_matrix.h>
#include <index_helper.cuh>
#include <gauge_field_order.h>
#include <kernel.h>

namespace quda {

  namespace fermion_force {

    enum {
      XUP = 0,
      YUP = 1,
      ZUP = 2,
      TUP = 3,
      TDOWN = 4,
      ZDOWN = 5,
      YDOWN = 6,
      XDOWN = 7
    };

    constexpr int opp_dir(int signed_dir) { return 7 - signed_dir; }
    constexpr int goes_forward(int signed_dir) { return signed_dir <= 3; }
    constexpr int goes_backward(int signed_dir) { return signed_dir > 3; }
    constexpr int coeff_sign(int pos_dir, int odd_lattice) { return 2*((pos_dir + odd_lattice + 1) & 1) - 1; }
    constexpr int parity_sign(int parity) { return parity ? -1 : 1; }
    constexpr int pos_dir(int signed_dir) { return (signed_dir >= 4) ? 7 - signed_dir : signed_dir; }

    __host__ __device__ int updateCoordsIndexMILCDir(int x[], const int X[], int signed_dir) {
      switch (signed_dir) {
      case 0: x[0] = (x[0] + 1 + X[0]) % X[0]; break;
      case 1: x[1] = (x[1] + 1 + X[1]) % X[1]; break;
      case 2: x[2] = (x[2] + 1 + X[2]) % X[2]; break;
      case 3: x[3] = (x[3] + 1 + X[3]) % X[3]; break;
      case 4: x[3] = (x[3] - 1 + X[3]) % X[3]; break;
      case 5: x[2] = (x[2] - 1 + X[2]) % X[2]; break;
      case 6: x[1] = (x[1] - 1 + X[1]) % X[1]; break;
      case 7: x[0] = (x[0] - 1 + X[0]) % X[0]; break;
      }
      int idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      return idx;
    }

    //struct for holding the fattening path coefficients
    template <typename real>
    struct PathCoefficients {
      const real one;
      const real naik;
      const real three;
      const real five;
      const real seven;
      const real lepage;
      PathCoefficients(const double *path_coeff_array)
        : one(path_coeff_array[0]), naik(path_coeff_array[1]),
          three(path_coeff_array[2]), five(path_coeff_array[3]),
          seven(path_coeff_array[4]), lepage(path_coeff_array[5]) { }
    };

    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct BaseForceArg : kernel_param<> {
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      const Gauge link;
      int X[4]; // regular grid dims
      int D[4]; // working set grid dims
      int E[4]; // extended grid dims

      int commDim[4];
      int border[4];
      int base_idx[4]; // the offset into the extended field
      int oddness_change;

      // for readability, we explicitly set the different directions
      int mu;
      int nu;
      int rho;
      int sig;
      int compute_lepage;

      /**
         @param[in] link Gauge field
         @param[in] overlap Radius of additional redundant computation to do
       */
      BaseForceArg(const GaugeField &link, int overlap) :
        kernel_param(dim3(1, 2, 1)),
        link(link),
        commDim{ comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3) },
        mu(-1), nu(-1), rho(-1), sig(-1), compute_lepage(-1)
      {
        for (int d=0; d<4; d++) {
          E[d] = link.X()[d];
          border[d] = link.R()[d];
          X[d] = E[d] - 2*border[d];
          D[d] = comm_dim_partitioned(d) ? X[d]+overlap*2 : int(X[d]);
          base_idx[d] = comm_dim_partitioned(d) ? border[d]-overlap : 0;
          this->threads.x *= D[d];
        }
        this->threads.x /= 2;
        oddness_change = (base_idx[0] + base_idx[1] + base_idx[2] + base_idx[3])&1;
      }
    };

    template <typename Arg_, int sig_positive_, int mu_positive_ = -1, int nu_positive_ = -1, int compute_lepage_ = -1>
    struct FatLinkParam : kernel_param<> {
      // whether the sig direction, if relevant, is forwards or backwards
      static constexpr int sig_positive = sig_positive_;

      // whether the mu direction, if relevant, is forwards or backwards
      static constexpr int mu_positive = mu_positive_;

      // whether the nu direction, if relevant, is forwards or backwards
      static constexpr int nu_positive = nu_positive_;

      // whether or not to compute the lepage contribution
      static constexpr int compute_lepage = compute_lepage_;

      // base argument structure
      using Arg = Arg_;
      Arg arg;
      FatLinkParam(Arg &arg) :
        kernel_param<>(arg.threads),
        arg(arg) {}
    };

    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct OneLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge force;

      const Gauge oProd;
      const real coeff_one;

      static constexpr int overlap = 0;

      OneLinkArg(GaugeField &force, const GaugeField &oProd, const GaugeField &link, const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), oProd(oProd), coeff_one(act_path_coeff.one) {
          this->threads.z = 4;
        }

    };

    template <typename Arg> struct OneLinkTerm
    {
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      constexpr OneLinkTerm(const Arg &arg) : arg(arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity, int sig)
      {
        int x[4];
        getCoords(x, x_cb, arg.X, parity);
#pragma unroll
        for (int d=0; d<4; d++) x[d] += arg.border[d];
        int e_cb = linkIndex(x,arg.E);

        Link w = arg.oProd(sig, e_cb, parity);
        Link force = arg.force(sig, e_cb, parity);
        force += arg.coeff_one * w;
        arg.force(sig, e_cb, parity) = force;
      }
    };

    /**************************middleThreeLinkKernel*****************************
     *
     *
     * Generally we need
     * READ
     *    3 LINKS:         ab_link,     bc_link,    ad_link
     *    2 COLOR MATRIX:  newOprod_at_A, oprod_at_C
     * WRITE
     *    4 COLOR MATRIX:  newOprod_at_A, P3_at_A, Pmu_at_B, Qmu_at_A
     *
     *   In all three above case, if the direction sig is negative, newOprod_at_A is
     *   not read in or written out.
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *   Call 1:  (called 48 times, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 6)
     *             else               :     (3, 4)
     *
     * note: oprod_at_C could actually be read in from D when it is the fresh outer product
     *       and we call it oprod_at_C to simply naming. This does not affect our data traffic analysis
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *   call 1:     if (sig is positive)  (3, 1)
     *               else                  (2, 0)
     *
     ****************************************************************************/
    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct MiddleThreeLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge force;
      Gauge pMu;
      Gauge p3;
      Gauge qMu;

      const Gauge oProd;
      const real coeff_three;

      static constexpr int overlap = 2;

      MiddleThreeLinkArg(GaugeField &force, GaugeField &pMu, GaugeField &P3, GaugeField &qMu,
                 const GaugeField &oProd, const GaugeField &link,
                  const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), pMu(pMu), p3(P3), qMu(qMu),
        oProd(oProd), coeff_three(act_path_coeff.three)
      { }

    };

    template <typename Param> struct MiddleThreeLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;

      static constexpr int sig_positive = Param::sig_positive;
      static constexpr int mu_positive = Param::mu_positive;
      static_assert(Param::nu_positive == -1, "nu_positive should be set to -1 for MiddleThreeLink");
      static_assert(Param::compute_lepage == -1, "compute_lepage should be set to -1 for MiddleThreeLink");

      constexpr MiddleThreeLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);

        /*        A________B
         *   mu   |        |
         *       D|        |C
         *
         *    A is the current point (sid)
         *
         * Variables have been named to reflection dimensionality for
         * mu_positive == true, sig_positive == true
         */

#pragma unroll
        for (int d = 0; d < 4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity ^ arg.oddness_change;
        int y[4] = {x[0], x[1], x[2], x[3]};

        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int ad_link_nbr_idx = mu_positive ? point_d : e_cb;

        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);

#pragma unroll
        for (int d = 0; d < 4; d++) y[d] = x[d];
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.sig);

        int bc_link_nbr_idx = mu_positive ? point_c : point_b;
        int ab_link_nbr_idx = sig_positive ? e_cb : point_b;

        // load the link variable connecting a and b (need a -> b for sig positive)
        Link Uab = arg.link(pos_dir(arg.sig), ab_link_nbr_idx, sig_positive ^ (1 - parity));

        // load the link variable connecting b and c (need b -> c for mu positive)
        Link Ucb = arg.link(pos_dir(arg.mu), bc_link_nbr_idx, mu_positive ^ (1 - parity));

        // load the input oprod connecting c and d (need d -> c for sig positive)
        Link Odc = arg.oProd(pos_dir(arg.sig), sig_positive ? point_d : point_c, sig_positive ^ parity);
        if constexpr (!sig_positive) Odc = conj(Odc);

        // for sig_positive, mu_positive, this is U_{b -> c} * O_{d -> c}
        Link UbcOdc = !mu_positive ? Ucb * Odc : conj(Ucb) * Odc;

        arg.pMu(0, point_b, 1 - parity) = UbcOdc;

        arg.p3(0, e_cb, parity) = sig_positive ? Uab * UbcOdc : conj(Uab) * UbcOdc;

        // load the link variable connecting a and d (need d -> a for mu positive)
        Link Uda = arg.link(pos_dir(arg.mu), ad_link_nbr_idx, mu_positive ^ parity);
        if (!mu_positive) Uda = conj(Uda);

        arg.qMu(0, e_cb, parity) = Uda;

        if constexpr (sig_positive) {
          Link UbcOdcUda = UbcOdc * Uda;
          Link oprod = arg.force(arg.sig, e_cb, parity);
          oprod -= arg.coeff_three * UbcOdcUda;
          arg.force(arg.sig, e_cb, parity) = oprod;
        }

      }
    };


    /**************************middleFiveLinkKernel*****************************
     *
     *
     * Generally we need
     * READ
     *    3 LINKS:         ab_link,     bc_link,    ad_link
     *    3 COLOR MATRIX:  newOprod_at_A, Pmu_at_C,  Qmu_at_D
     * WRITE
     *    4 COLOR MATRIX:  newOprod_at_A, P3_at_A, Pnumu_at_B, Qnumu_at_A
     *
     * Three call variations:
     *   2. full read/write
     *
     *   In all three above case, if the direction sig is negative, newOprod_at_A is
     *   not read in or written out.
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *   Call 2:  (called 192 time, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 7)
     *             else               :     (3, 5)
     *
     * note: Pmu_at_C could actually be read in from D when it is the fresh outer product
     *       and we call it Pmu_at_C to simply naming. This does not affect our data traffic analysis
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *   call 2:     if (sig is positive)  (4, 1)
     *               else                  (3, 0)
     *
     ****************************************************************************/
    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct MiddleFiveLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge force;
      Gauge pNuMu;
      Gauge p5;
      Gauge qNuMu;

      const Gauge pMu;
      const Gauge qMu;
      const real coeff_five;

      static constexpr int overlap = 2;

      MiddleFiveLinkArg(GaugeField &force, GaugeField &pNuMu, GaugeField &P5, GaugeField &qNuMu,
                 const GaugeField &pMu, const GaugeField &qMu, const GaugeField &link,
                  const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), pNuMu(pNuMu), p5(P5), qNuMu(qNuMu),
        pMu(pMu), qMu(qMu), coeff_five(act_path_coeff.five)
      { }

    };

    template <typename Param> struct MiddleFiveLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;

      static constexpr int sig_positive = Param::sig_positive;
      static_assert(Param::mu_positive == -1, "mu_positive should be set to -1 for MiddleFiveLink");
      static constexpr int nu_positive = Param::nu_positive;
      static_assert(Param::compute_lepage == -1, "compute_lepage should be set to -1 for MiddleFiveLink");

      constexpr MiddleFiveLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);

        /*        A________B
         *   nu   |        |
         *       D|        |C
         *
         *    A is the current point (sid)
         *
         */

        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity ^ arg.oddness_change;
        int y[4] = {x[0], x[1], x[2], x[3]};

        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.nu));
        int ad_link_nbr_idx = nu_positive ? point_d : e_cb;

        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);

        for (int d=0; d<4; d++) y[d] = x[d];
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.sig);

        int bc_link_nbr_idx = nu_positive ? point_c : point_b;
        int ab_link_nbr_idx = sig_positive ? e_cb : point_b;

        // load the link variable connecting a and b
        Link Uab = arg.link(pos_dir(arg.sig), ab_link_nbr_idx, sig_positive^(1-parity));

        // load the link variable connecting b and c
        Link Ubc = arg.link(pos_dir(arg.nu), bc_link_nbr_idx, nu_positive^(1-parity));

        Link Oy = arg.pMu(0, point_c, parity);

        Link Ow = !nu_positive ? Ubc*Oy : conj(Ubc)*Oy;

        arg.pNuMu(0, point_b, 1-parity) = Ow;

        arg.p5(0, e_cb, parity) = sig_positive ? Uab*Ow : conj(Uab)*Ow;

        Link Uad = arg.link(pos_dir(arg.nu), ad_link_nbr_idx, nu_positive^parity);
        if (!nu_positive)  Uad = conj(Uad);

        Oy = arg.qMu(0, point_d, 1-parity);
        Link Ox = Oy*Uad;
        arg.qNuMu(0, e_cb, parity) = Ox;

        if constexpr (sig_positive) {
          Oy = Ow * Ox;
          Link oprod = arg.force(arg.sig, e_cb, parity);
          oprod += arg.coeff_five * Oy;
          arg.force(arg.sig, e_cb, parity) = oprod;
        }

      }
    };

    /********************************allSevenSideFiveLinkKernel*********************************************
     * Note: this kernel names points differently than the other kernels, there's a diagram
     *   within the kernel code
     *
     * In this function we need
     *   READ
     *     8 LINKS:         ab_link, af_link, be_link, fe_link, dc_link, da_link, cb_link, ah_link
     *     12 COLOR MATRIX:  qnumu_at_{a, f, d}, oprod_at_{e, b, c}, newOprod_{rho,sig,nu}_at_a,
     *                      p5_at_a, shortP_at_h, qProd_at_h
     *   WRITE:
     *     4 COLOR MATRIX:  newOprod_{rho,sig,nu}_at_a, shortP_at_h
     *
     * If sig is negative, then we don't need to read qnumu_at_c, oprod_at_b, read/write
     *             newOprod_sig_at_a
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *
     *             if (sig is positive):    (8, 16)
     *             else               :     (8, 12)
     *
     * This function is called 192 times, half positive sig, half negative sig
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *             if(sig is positive)      (17,11)
     *             else                     (12,8)
     *
     ************************************************************************************************/
    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct AllSevenSideFiveLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge force;
      Gauge shortP;
      Gauge p5;

      const Gauge oProd;
      const Gauge qNuMu;
      const Gauge qProd;

      const real coeff_five;
      const real accumu_coeff_five;
      const real coeff_seven;
      const real accumu_coeff_seven;

      static constexpr int overlap = 2;

      AllSevenSideFiveLinkArg(GaugeField &force, GaugeField &shortP, GaugeField &P5, const GaugeField &oProd, const GaugeField &qNuMu,
                 const GaugeField &qProd, const GaugeField &link, const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), shortP(shortP), p5(P5),
          oProd(oProd), qNuMu(qNuMu), qProd(qProd),
          coeff_five(act_path_coeff.five), accumu_coeff_five(act_path_coeff.three != 0 ? act_path_coeff.five / act_path_coeff.three : 0),
          coeff_seven(act_path_coeff.seven), accumu_coeff_seven(act_path_coeff.five != 0 ? act_path_coeff.seven / act_path_coeff.five : 0)
      { }

    };

    template <typename Param> struct AllSevenSideFiveLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;

      static constexpr int sig_positive = Param::sig_positive;
      static_assert(Param::mu_positive == -1, "mu_positive should be set to -1 for AllSevenSideFiveLink");
      static constexpr int nu_positive = Param::nu_positive;
      static_assert(Param::compute_lepage == -1, "compute_lepage should be set to -1 for AllSevenSideFiveLink");

      constexpr AllSevenSideFiveLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ Link all_link(int x[4], int e_cb, int parity) {
        auto mycoeff_seven = coeff_sign(sig_positive,parity)*arg.coeff_seven;

        // Intermediate accumulators; force_sig is only needed when sig is positive
        Link force_sig, force_rho, p5_sig;

        // Link product intermediates
        Link Oy, Oz;

        /*            sig
         *         F        E
         *          |      |
         *         A|______|B
         *      rho |      |
         *        D |      |C
         *
         *   A is the current point (sid)
         *
         */

        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_a = e_cb;
        int parity_a = parity;
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_b = 1 - parity;
        int ab_link_nbr_idx = (sig_positive) ? point_a : point_b;
        int ab_link_nbr_parity = (sig_positive) ? parity_a : parity_b;

        for (int d=0; d<4; d++) y[d] = x[d];
        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.rho));
        int parity_d = 1 - parity;
        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;
        int dc_link_nbr_idx = (sig_positive) ? point_d : point_c;
        int dc_link_nbr_parity = (sig_positive) ? parity_d : parity_c;

        for (int d = 0; d < 4; d++) y[d] = x[d];
        int point_f = updateCoordsIndexMILCDir(y, arg.E, arg.rho);
        int parity_f = 1 - parity;
        int point_e = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_e = parity;
        int fe_link_nbr_idx = (sig_positive) ? point_f : point_e;
        int fe_link_nbr_parity = (sig_positive) ? parity_f : parity_e;

        // Compute the force_rho (and force_sig, when sig is positive) contribution
        // from the negative rho direction
        Link Uab = arg.link(pos_dir(arg.sig), ab_link_nbr_idx, ab_link_nbr_parity);
        Link Uaf = arg.link(pos_dir(arg.rho), point_a, parity_a);
        Link Ube = arg.link(pos_dir(arg.rho), point_b, parity_b);
        Link Of = arg.qNuMu(0, point_f, parity_f);
        Link Oe = arg.oProd(0, point_e, parity_e);
        Oz = Ube * Oe;
        Oy = (sig_positive ? Uab : conj(Uab)) * Oz;
        force_rho += (parity_sign(parity_a) * mycoeff_seven) * conj(Of) * conj(Oy);
        if constexpr (sig_positive) {
          force_sig += (parity_sign(parity_a) * mycoeff_seven) * Oz * Of * conj(Uaf);
        }

        // Compute the force_rho and p5 contribution from the positive rho direction
        Link Ufe = arg.link(pos_dir(arg.sig), fe_link_nbr_idx, fe_link_nbr_parity);
        Link Oa = arg.qNuMu(0, point_a, parity_a);
        Link Ob = arg.oProd(0, point_b, parity_b);
        Oz = conj(Ube) * Ob;
        Oy = (sig_positive ? Ufe : conj(Ufe)) * Oz;
        force_rho -= (parity_sign(parity_a) * mycoeff_seven) * Oy * Oa;
        p5_sig += arg.accumu_coeff_seven * Uaf * Oy;

        // Compute the p5 contribution from the negative rho direction
        Link Udc = arg.link(pos_dir(arg.sig), dc_link_nbr_idx, dc_link_nbr_parity);
        Link Uda = arg.link(pos_dir(arg.rho), point_d, parity_d);
        Link Ucb = arg.link(pos_dir(arg.rho), point_c, parity_c);
        Oz = Ucb * Ob;
        Oy = (sig_positive ? Udc : conj(Udc)) * Oz;
        p5_sig += arg.accumu_coeff_seven * conj(Uda) * Oy;

        // When sig is positive, compute the force_sig contribution from the
        // positive rho direction
        if constexpr (sig_positive) {
          Link Od = arg.qNuMu(0, point_d, parity_d);
          Link Oc = arg.oProd(0, point_c, parity_c);
          Oz = conj(Ucb) * Oc;
          force_sig += (parity_sign(parity_a) * mycoeff_seven) * Oz * Od * Uda;
        }

        // update the force in the rho direction
        Link force = arg.force(pos_dir(arg.rho), point_a, parity_a);
        force += force_rho;
        arg.force(pos_dir(arg.rho), point_a, parity_a) = force;

        // update the force in the sigma direction
        if constexpr (sig_positive) {
          Link force = arg.force(arg.sig, point_a, parity_a);
          force += force_sig;
          arg.force(arg.sig, point_a, parity_a) = force;
        }

        return p5_sig;

      }

      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);
        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity^arg.oddness_change;

        /*      compute the side link contribution to the momentum
         *
         *             sig
         *          A________B
         *           |       |   nu
         *         H |       |G
         *
         *      A is the current point (x_cb)
         *
         */

        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_a = e_cb;
        int parity_a = parity;
        int point_h = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.nu));
        int parity_h = 1 - parity;
        int ah_link_nbr_idx = nu_positive ? point_h : point_a;
        int ah_link_nbr_parity = nu_positive ? parity_h : parity_a;

        // calculate p5_sig
        Link p5_sig = all_link(x, e_cb, parity);

        // load and accumulate p5
        Link Oy = arg.p5(0, point_a, parity_a);
        Oy += p5_sig;

        Link Uah = arg.link(pos_dir(arg.nu), ah_link_nbr_idx, ah_link_nbr_parity);
        Link Ow = nu_positive ? Uah * Oy : conj(Uah) * Oy;

        Link shortP = arg.shortP(0, point_h, parity_h);
        shortP += arg.accumu_coeff_five * Ow;
        arg.shortP(0, point_h, parity_h) = shortP;

        Link Ox = arg.qProd(0, point_h, parity_h);
        Ow = nu_positive ? Oy * Ox : conj(Ox) * conj(Oy);

        auto mycoeff_five = -coeff_sign(goes_forward(arg.sig), parity_a)*coeff_sign(goes_forward(arg.nu),parity_a)*arg.coeff_five;

        Link oprod = arg.force(pos_dir(arg.nu), ah_link_nbr_idx, ah_link_nbr_parity);
        oprod += mycoeff_five * Ow;
        arg.force(pos_dir(arg.nu), ah_link_nbr_idx, ah_link_nbr_parity) = oprod;
      }
    };


    /************************AllLepageSideThreeLink*****************************
     * Note: this kernel names points differently than the other kernels, there's a diagram
     *   within the kernel code
     *
     * If we are only computing the side three link term (when the Lepage coefficient
     * is zero), we only need:
     * READ
     *    0 LINKS
     *    2 COLOR MATRIX:  p3_at_{F for mu_positive, A for sig_positive},
     *                     newOprod_at_A (mu direction)
     *  WRITE
     *    1 COLOR MATRIX:  newOprod_at_A (mu direction)
     *
     * Therefore the data traffic, in two-number pair (num of link, num_of_color_matrix), is
     *      (sig positive or negative):   (0, 3)
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add) is (0, 1)
     *
     * If we are are computing the Lepage middle and all link, this gets complicated.
     * We have an additional:
     * READ
     *    5 LINKS:          hg_link, eg_link, fh_link, fe_link, be_link
     *    3 COLOR MATRIX:   oprod_at_e, oprod_at_b, qProd_at_d
     *    additionally, if sig positive:
     *      1 LINK:         af_link
     *      1 COLOR MATRIX: qProd_at_a, newOprod_at_f (sig direction)
     * WRITE if sig is positive
     *    1 COLOR MATRIX:   newOprod_at_f (sig_direction)
     *
     * If mu is negative, there is no link load when sig is positive. Further, many
     * of the sites that components are loaded from are shifted --- the code uses
     * self-documenting variables, so check there for the shifts.
     *
     * The TOTAL data traffic (including side link), in two-number pair (num_of_link, num_of_color_matrix)
     *   if (mu is positive):
     *     if (sig is positive):   (6, 9)
     *     else:                   (6, 6)
     *   else (mu is negative):
     *     if (sig is positive):   (6, 8)
     *     else:                   (6, 6)
     *
     * This kernel is called 48 times, 24 for sig positive and 24 for sig negative
     *
     * The TOTAL flop count, in two-number pair (matrix_multi, matrix_add)
     *   if (sig is positive):     (8, 3)
     *   else:                     (6, 2)
     *
     ****************************************************************************/
    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct AllLepageSideThreeLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge force;

      const Gauge p3;
      const Gauge oProd;
      const Gauge qProd;

      const real coeff_three;
      const real coeff_lepage;
      const real accumu_coeff_lepage;

      static constexpr int overlap = 2;

      AllLepageSideThreeLinkArg(GaugeField &force, const GaugeField &P3, const GaugeField &oProd,
                 const GaugeField &qProd, const GaugeField &link,
                 const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), p3(P3),
        oProd(oProd), qProd(qProd),
        coeff_three(act_path_coeff.three), coeff_lepage(act_path_coeff.lepage),
        accumu_coeff_lepage(act_path_coeff.three != 0 ? act_path_coeff.lepage / act_path_coeff.three : 0)
      { }

    };

    template <typename Param> struct AllLepageSideThreeLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;

      static_assert( (Param::compute_lepage == 1 && Param::sig_positive != -1) ||
                     (Param::compute_lepage == 0 && Param::sig_positive == -1),
                     "sig_positive should only be set when compute_lepage == 1 for AllLepageSideThreeLink");
      static constexpr int sig_positive = Param::sig_positive;
      static constexpr int mu_positive = Param::mu_positive;
      static_assert(Param::nu_positive == -1, "nu_positive should be set to -1 for AllLepageSideThreeLink");
      static constexpr int compute_lepage = Param::compute_lepage;

      constexpr AllLepageSideThreeLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity)
      {

        int x[4];
        getCoords(x, x_cb, arg.D, parity);

        /*
         * Very top points are for the Lepage term, to prepare p3 for the 3-link side link
         * 
         *  
         *            sig
         *         H|      |G
         *          |      |
         *         F        E
         *          |      |
         *         A|______|B
         *       mu |      |
         *        D |      |C
         *
         *   A is the current point (sid)
         *
         */

        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity ^ arg.oddness_change;

        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_a = e_cb;
        int parity_a = parity;
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_b = 1 - parity;

        for (int d=0; d<4; d++) y[d] = x[d];
        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int parity_d = 1 - parity;
        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;

        for (int d=0; d<4; d++) y[d] = x[d];
        int point_f = updateCoordsIndexMILCDir(y, arg.E, arg.mu);
        int parity_f = 1 - parity;
        int point_e = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_e = parity;

        int point_g = updateCoordsIndexMILCDir(y, arg.E, arg.mu);
        int parity_g = 1 - parity;
        int point_h = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.sig));
        int parity_h = parity;

        int ab_link_nbr_idx = (sig_positive) ? point_a : point_b;
        int ab_link_nbr_parity = (sig_positive) ? parity_a : parity_b;

        int fe_link_nbr_idx = (sig_positive) ? point_f : point_e;
        int fe_link_nbr_parity = (sig_positive) ? parity_f : parity_e;

        int hg_link_nbr_idx = (sig_positive) ? point_h : point_g;
        int hg_link_nbr_parity = (sig_positive) ? parity_h : parity_g;

        Link p3;
        if constexpr ( mu_positive ) {
          p3 = arg.p3(0, point_f, parity_f);
        } else {
          p3 = arg.p3(0, point_a, parity_a);
        }

        Link force_mu = arg.force(pos_dir(arg.mu), point_a, parity_a);

        if constexpr ( compute_lepage ) {

          if constexpr ( mu_positive ) {

            // Accumulate p3
            {
              Link Uhg = arg.link(pos_dir(arg.sig), hg_link_nbr_idx, hg_link_nbr_parity);
              Link Ueg = arg.link(pos_dir(arg.mu), point_e, parity_e);
              Link Ufh = arg.link(pos_dir(arg.mu), point_f, parity_f);
              Link Oe = arg.oProd(0, point_e, parity_e);

              Link Ow = conj(Ueg) * Oe;
              Link Oy = sig_positive ? Uhg * Ow : conj(Uhg) * Ow;
              Link Ox = Ufh * Oy;

              p3 += arg.accumu_coeff_lepage * Ox;
            }

            // Accumulate force_mu
            Link Ufe = arg.link(pos_dir(arg.sig), fe_link_nbr_idx, fe_link_nbr_parity);
            Link Ube = arg.link(pos_dir(arg.mu), point_b, parity_b);
            Link Ob = arg.oProd(0, point_b, parity_b);
            Link Qd = arg.qProd(0, point_a, parity_a);

            Link Ow = conj(Ube) * Ob;
            Link Oy = sig_positive ? Ufe * Ow : conj(Ufe) * Ow;
            Link Ox = Oy * Qd;

            auto mycoeff_lepage = -coeff_sign(goes_forward(arg.sig), parity)*coeff_sign(goes_forward(arg.mu),parity)*arg.coeff_lepage;

            force_mu += mycoeff_lepage * Ox;

            // Update force_sig if sig is positive
            if constexpr ( sig_positive ) {
              Link Uaf = arg.link(pos_dir(arg.mu), point_a, parity_a);
              Link Qa = arg.qProd(0, point_a, parity_a);
              Link Ox = Qa * Uaf;
              Link Oy = Ow * Ox;

              Link oprod = arg.force(arg.sig, point_f, parity_f);
              oprod += arg.coeff_lepage * Oy;
              arg.force(arg.sig, point_f, parity_f) = oprod;
            }

          } else {
            // mu negative

            // Accumulate p3
            {
              Link Ufe = arg.link(pos_dir(arg.sig), fe_link_nbr_idx, fe_link_nbr_parity);
              Link Ueg = arg.link(pos_dir(arg.mu), point_e, parity_e);
              Link Ufh = arg.link(pos_dir(arg.mu), point_f, parity_f);
              Link Ob = arg.oProd(0, point_b, parity_b);

              Link Ow = Ueg * Ob;
              Link Oy = sig_positive ? Ufe * Ow : conj(Ufe) * Ow;
              Link Ox = conj(Ufh) * Oy;

              p3 += arg.accumu_coeff_lepage * Ox;
            }

            // Accumulate force_mu, update force_sig if sig is positive
            {
              Link Uab = arg.link(pos_dir(arg.sig), ab_link_nbr_idx, ab_link_nbr_parity);
              Link Ubc = arg.link(pos_dir(arg.mu), point_b, parity_b);
              Link Oc = arg.oProd(0, point_c, parity_c);
              Link Qd = arg.qProd(0, point_d, parity_d);

              Link Ow = Ubc * Oc;
              Link Oy = sig_positive ? Uab*Ow : conj(Uab)*Ow;
              Link Ox = conj(Qd) * conj(Oy);

              auto mycoeff_lepage = -coeff_sign(goes_forward(arg.sig), parity)*coeff_sign(goes_forward(arg.mu),parity)*arg.coeff_lepage;
              force_mu += mycoeff_lepage * Ox;

              if constexpr ( sig_positive ) {
                Link Uad = arg.link(pos_dir(arg.mu), point_a, parity_a);
                Link Ox = Qd * conj(Uad);
                Link Oy = Ow * Ox;

                Link oprod = arg.force(arg.sig, point_a, parity_a);
                oprod += arg.coeff_lepage*Oy;
                arg.force(arg.sig, point_a, parity_a) = oprod;
              }
            }
          }
        }

        // Update force_mu
        auto mycoeff_three = coeff_sign(goes_forward(arg.sig),parity)*coeff_sign(goes_forward(arg.mu),parity)*arg.coeff_three;
        force_mu += mycoeff_three * (mu_positive ? p3 : conj(p3));
        arg.force(pos_dir(arg.mu), point_a, parity_a) = force_mu;

      }
    };

    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct CompleteForceArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge outA;        // force output accessor
      const Gauge oProd; // force input accessor
      const real coeff;

      CompleteForceArg(GaugeField &force, const GaugeField &link)
        : BaseForceArg(link, 0), outA(force), oProd(force), coeff(0.0)
      { }

    };

    template <typename Arg> struct CompleteForce
    {
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      constexpr CompleteForce(const Arg &arg) : arg(arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      // Flops count: 4 matrix multiplications per lattice site = 792 Flops per site
      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        getCoords(x, x_cb, arg.X, parity);

        for (int d=0; d<4; d++) x[d] += arg.border[d];
        int e_cb = linkIndex(x,arg.E);

#pragma unroll
        for (int sig=0; sig<4; ++sig) {
          Link Uw = arg.link(sig, e_cb, parity);
          Link Ox = arg.oProd(sig, e_cb, parity);
          Link Ow = Uw*Ox;

          makeAntiHerm(Ow);

          typename Arg::real coeff = (parity==1) ? -1.0 : 1.0;
          arg.outA(sig, e_cb, parity) = coeff*Ow;
        }
      }
    };

    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct LongLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge outA;
      const Gauge oProd;
      const real coeff;

      LongLinkArg(GaugeField &newOprod, const GaugeField &link, const GaugeField &oprod, real coeff)
        : BaseForceArg(link,0), outA(newOprod), oProd(oprod), coeff(coeff)
      { }

    };

    template <typename Arg> struct LongLink
    {
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      constexpr LongLink(const Arg &arg) : arg(arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      // Flops count, in two-number pair (matrix_mult, matrix_add)
      // 				   (24, 12)
      // 4968 Flops per site in total
      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        int dx[4] = {0,0,0,0};

        getCoords(x, x_cb, arg.X, parity);

        for (int i=0; i<4; i++) x[i] += arg.border[i];
        int e_cb = linkIndex(x,arg.E);

        /*
         *
         *    A   B    C    D    E
         *    ---- ---- ---- ----
         *
         *   ---> sig direction
         *
         *   C is the current point (sid)
         *
         */

        // compute the force for forward long links
#pragma unroll
        for (int sig=0; sig<4; sig++) {
          int point_c = e_cb;

          dx[sig]++;
          int point_d = linkIndexShift(x,dx,arg.E);

          dx[sig]++;
          int point_e = linkIndexShift(x,dx,arg.E);

          dx[sig] = -1;
          int point_b = linkIndexShift(x,dx,arg.E);

          dx[sig]--;
          int point_a = linkIndexShift(x,dx,arg.E);
          dx[sig] = 0;

          Link Uab = arg.link(sig, point_a, parity);
          Link Ubc = arg.link(sig, point_b, 1-parity);
          Link Ude = arg.link(sig, point_d, 1-parity);
          Link Uef = arg.link(sig, point_e, parity);

          Link Oz = arg.oProd(sig, point_c, parity);
          Link Oy = arg.oProd(sig, point_b, 1-parity);
          Link Ox = arg.oProd(sig, point_a, parity);

          Link temp = Ude*Uef*Oz - Ude*Oy*Ubc + Ox*Uab*Ubc;

          Link force = arg.outA(sig, e_cb, parity);
          arg.outA(sig, e_cb, parity) = force + arg.coeff*temp;
        } // loop over sig
      }
    };

  }
}

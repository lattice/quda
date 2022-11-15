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

    // for readability
    enum {
      SIG_POSITIVE = 1,
      SIG_NEGATIVE = 0,
      SIG_IGNORED = -1
    };

    enum {
      MU_POSITIVE = 1,
      MU_NEGATIVE = 0,
      MU_IGNORED = -1
    };

    enum {
      NU_POSITIVE = 1,
      NU_NEGATIVE = 0,
      NU_IGNORED = -1
    };

    enum {
      NU_NEXT_POSITIVE = 1,
      NU_NEXT_NEGATIVE = 0,
      NU_NEXT_IGNORED = -1
    };

    enum {
      COMPUTE_LEPAGE_YES = 1,
      COMPUTE_LEPAGE_NO = 0,
      COMPUTE_LEPAGE_IGNORED = -1
    };

    constexpr int opp_dir(int signed_dir) { return 7 - signed_dir; }
    constexpr int goes_forward(int signed_dir) { return signed_dir <= 3; }
    constexpr int goes_backward(int signed_dir) { return signed_dir > 3; }
    constexpr int coeff_sign(int pos_dir, int odd_lattice) { return 2*((pos_dir + odd_lattice + 1) & 1) - 1; }
    constexpr int parity_sign(int parity) { return parity ? -1 : 1; }
    constexpr int pos_dir(int signed_dir) { return (signed_dir >= 4) ? 7 - signed_dir : signed_dir; }

    /**
      @brief Compute the checkerboard 1-d index for the nearest neighbor in a MILC-convention
             signed direction, updating the lattice coordinates in-place
      @param[in/out] x Local coordinate, which is returned shifted
      @param[in] X Full lattice dimensions
      @param[in] signed_dir Signed MILC direction
      @return Shifted 1-d checkboard index
    */
    __host__ __device__ int updateCoordsIndexMILCDir(int x[], const int_fastdiv X[], int signed_dir) {
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

    /**
      @brief Compute the checkerboard 1-d index for the nearest neighbor in a MILC-convention
             signed direction
      @param[in/out] x Local coordinate
      @param[in] X Full lattice dimensions
      @param[in] signed_dir Signed MILC direction
      @return Shifted 1-d checkboard index
    */
    __host__ __device__ int getIndexMILCDir(const int x[], const int_fastdiv X[], int signed_dir) {
      int y[4] = {x[0], x[1], x[2], x[3]};
      return updateCoordsIndexMILCDir(y, X, signed_dir);
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
      using Link = Matrix<complex<real>, nColor>;

      const Gauge link;
      int_fastdiv X[4]; // regular grid dims
      int_fastdiv D[4]; // working set grid dims
      int_fastdiv E[4]; // extended grid dims

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

      // for fused all 5 all seven
      int nu_next;

      /**
         @param[in] link Gauge field
         @param[in] overlap Radius of additional redundant computation to do
       */
      BaseForceArg(const GaugeField &link, int overlap) :
        kernel_param(dim3(1, 2, 1)),
        link(link),
        commDim{ comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3) },
        mu(-1), nu(-1), rho(-1), sig(-1), compute_lepage(-1), nu_next(-1)
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

    template <typename Arg_, int sig_positive_, int mu_positive_ = -1, int nu_positive_ = -1, int nu_side_positive_ = -1, int compute_lepage_ = -1>
    struct FatLinkParam : kernel_param<> {
      // whether the sig direction, if relevant, is forwards or backwards
      static constexpr int sig_positive = sig_positive_;

      // whether the mu direction, if relevant, is forwards or backwards
      static constexpr int mu_positive = mu_positive_;

      // whether the nu direction, if relevant, is forwards or backwards
      static constexpr int nu_positive = nu_positive_;

      // for fused AllFiveAllSeven, whether the nu direction for the side 5 is forwards or backwards
      // I guess this is always technically the opposite of nu_positive...
      static constexpr int nu_side_positive = nu_side_positive_;

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
     *    2 LINKS:         ab_link,     bc_link
     *    1 COLOR MATRIX:  oprod_at_C
     * WRITE
     *    2 COLOR MATRIX:  P3_at_A, Pmu_at_B
     *
     * If sig is positive, we additionally:
     * READ
     *    1 LINK:          da_link
     *    1 COLOR MATRIX:  newOprod_{sig}_at_A
     * WRITE
     *    1 COLOR MATRIX:  newOprod_{sig}_at_A
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *   Call 1:  (called 48 times, half positive sig, half negative sig)
     *             if (sig is positive):    (2, 5)
     *             else                :    (3, 3)
     *
     * Flop count, in three-number pair (matrix_multi, matrix_add, matrix_rescale)
     *   call 1:     if (sig is positive)  (3, 1, 1)
     *               else                  (2, 0, 0)
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

      const Gauge oProd;
      const real coeff_three;

      static constexpr int overlap = 2;

      MiddleThreeLinkArg(GaugeField &force, GaugeField &pMu, GaugeField &P3,
                 const GaugeField &oProd, const GaugeField &link,
                  const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), pMu(pMu), p3(P3),
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
        int point_a = e_cb;
        int parity_a = parity;
        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int parity_d = 1 - parity;

        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.mu);
        int parity_b = 1 - parity;

        int da_link_nbr_idx = mu_positive ? point_d : point_a;
        int da_link_nbr_parity = mu_positive ? parity_d : parity_a;

        int cb_link_nbr_idx = mu_positive ? point_c : point_b;
        int cb_link_nbr_parity = mu_positive ? parity_c : parity_b;

        int ab_link_nbr_idx = sig_positive ? point_a : point_b;
        int ab_link_nbr_parity = sig_positive ? parity_a : parity_b;

        int dc_link_nbr_idx = sig_positive ? point_d : point_c;
        int dc_link_nbr_parity = sig_positive ? parity_d : parity_c;

        // Load link and outer product contributions for Pmu and P3, as well as
        // force_sig when sig is positive
        Link Uab = arg.link(pos_dir(arg.sig), ab_link_nbr_idx, ab_link_nbr_parity);
        Link Ucb = arg.link(pos_dir(arg.mu), cb_link_nbr_idx, cb_link_nbr_parity);
        Link Uda = arg.link(pos_dir(arg.mu), da_link_nbr_idx, da_link_nbr_parity);
        Link Od = arg.oProd(pos_dir(arg.sig), dc_link_nbr_idx, dc_link_nbr_parity);

        if constexpr (!sig_positive) Od = conj(Od);
        if constexpr (!mu_positive) Uda = conj(Uda);

        Link Oz = !mu_positive ? Ucb * Od : conj(Ucb) * Od;
        arg.pMu(0, point_b, parity_b) = Oz;
        arg.p3(0, e_cb, parity) = sig_positive ? Uab * Oz : conj(Uab) * Oz;

        // Update the force in the sigma direction
        if constexpr (sig_positive) {
          Link Oy = Oz * Uda;
          Link oprod = arg.force(arg.sig, point_a, parity_a);
          oprod -= arg.coeff_three * Oy;
          arg.force(arg.sig, point_a, parity_a) = oprod;
        }

      }
    };


    /**************************middleFiveLinkKernel*****************************
     *
     *
     * Generally we need
     * READ
     *    4 LINKS:         ab_link, cb_link, da_link, qd_link
     *    1 COLOR MATRIX:  Pmu_at_C
     * WRITE
     *    3 COLOR MATRIX:  P5_at_A, Pnumu_at_B, Qnumu_at_A
     *
     * If sig is positive, we additionally:
     * READ
     *    1 COLOR MATRIX:  newOprod_{sig}_at_A
     * WRITE
     *    1 COLOR MATRIX:  newOprod_{sig}_at_A
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *   Call 2:  (called 192 time, half positive sig, half negative sig)
     *             if (sig is positive):    (4, 4)
     *             else               :     (4, 6)
     *
     *
     * Flop count, in three-number pair (matrix_multi, matrix_add, matrix_rescale)
     *   call 2:     if (sig is positive)  (4, 1, 1)
     *               else                  (3, 0, 0)
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
      const real coeff_five;

      static constexpr int overlap = 2;

      MiddleFiveLinkArg(GaugeField &force, GaugeField &pNuMu, GaugeField &P5, GaugeField &qNuMu,
                 const GaugeField &pMu, const GaugeField &link,
                  const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), pNuMu(pNuMu), p5(P5), qNuMu(qNuMu),
        pMu(pMu), coeff_five(act_path_coeff.five)
      { }

    };

    template <typename Param> struct MiddleFiveLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;

      static constexpr int sig_positive = Param::sig_positive;
      static constexpr int mu_positive = Param::mu_positive;
      static constexpr int nu_positive = Param::nu_positive;
      static_assert(Param::compute_lepage == -1, "compute_lepage should be set to -1 for MiddleFiveLink");

      constexpr MiddleFiveLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      /**
        @brief Compute the middle five link contribution to the HISQ force
        @param[in] x Local coordinate
        @param[in] e_cb 1-d checkerboard index in the full extended lattice
        @param[in] parity Parity of the coordinate x
        @return Contribution to the force in the sigma direction (when sigma is positive)
      */
      __device__ __host__ Link middle_five(int x[4], int e_cb, int parity) {

        /*        A________B
         *   nu   |        |
         *       H|        |C
         *
         *    A is the current point (sid)
         *
         * This kernel also depends on U_{mu} flowing *into* point H,
         * formerly referred to as "qMu" or "qProd". We denote the point
         * in the negative "mu" direction from H as "Q".
         */

        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_a = e_cb;
        int parity_a = parity;
        int point_h = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.nu));
        int parity_h = 1 - parity;

        int point_q = getIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int parity_q = parity;

        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.nu);
        int parity_b = 1 - parity;

        int ha_link_nbr_idx = nu_positive ? point_h : point_a;
        int ha_link_nbr_parity = nu_positive ? parity_h : parity_a;

        int qh_link_nbr_idx = mu_positive ? point_q : point_h;
        int qh_link_nbr_parity = mu_positive ? parity_q : parity_h;

        int cb_link_nbr_idx = nu_positive ? point_c : point_b;
        int cb_link_nbr_parity = nu_positive ? parity_c : parity_b;

        int ab_link_nbr_idx = sig_positive ? point_a : point_b;
        int ab_link_nbr_parity = sig_positive ? parity_a : parity_b;

        // Load link and outer product contributions for pNuMu, P5, qNuMu
        Link Uab = arg.link(pos_dir(arg.sig), ab_link_nbr_idx, ab_link_nbr_parity);
        Link Ubc = arg.link(pos_dir(arg.nu), cb_link_nbr_idx, cb_link_nbr_parity);
        Link Uha = arg.link(pos_dir(arg.nu), ha_link_nbr_idx, ha_link_nbr_parity);
        Link Uqh = arg.link(pos_dir(arg.mu), qh_link_nbr_idx, qh_link_nbr_parity);
        Link Oc = arg.pMu(0, point_c, parity_c);

        Link Ow = !nu_positive ? Ubc * Oc : conj(Ubc) * Oc;
        if constexpr (!nu_positive) Uha = conj(Uha);
        if constexpr (!mu_positive) Uqh = conj(Uqh);

        arg.pNuMu(0, point_b, parity_b) = Ow;
        arg.p5(0, point_a, parity_a) = sig_positive ? Uab * Ow : conj(Uab) * Ow;

        Link Ox = Uqh * Uha;
        arg.qNuMu(0, point_a, parity_a) = Ox;

        // compute the force in the sigma direction if sig is positive
        Link force_sig;
        if constexpr (sig_positive) {
          force_sig = arg.coeff_five * (Ow * Ox);
        }

        return force_sig;
      }

      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);

        /*        A________B
         *   nu   |        |
         *       H|        |C
         *
         *    A is the current point (sid)
         *
         * This kernel also depends on U_{mu} flowing *into* point H,
         * formerly referred to as "qMu" or "qProd". We denote the point
         * in the negative "mu" direction from H as "Q".
         */

#pragma unroll
        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity ^ arg.oddness_change;

        Link force_sig = middle_five(x, e_cb, parity);

        // Update the force in the sigma direction
        if constexpr (sig_positive) {
          int point_a = e_cb;
          int parity_a = parity;
          Link oprod = arg.force(arg.sig, point_a, parity_a);
          oprod += force_sig;
          arg.force(arg.sig, point_a, parity_a) = oprod;
        }

      }
    };

    /********************************allSevenSideFiveLinkKernel*********************************************
     * Note: this kernel names points differently than the other kernels, there's a diagram
     *   within the kernel code
     *
     * Generally we need
     * READ
     *    9 LINKS:         ab_link, af_link, be_link, fe_link, dc_link,
     *                     da_link, cb_link, ah_link, qh_link
     *    8 COLOR MATRIX:  qnumu_at_{F, A}, oProd_at_{E, B}, p5_at_A, shortP_at_H,
     *                     newOprod_{rho, nu}_at_A
     * WRITE
     *    3 COLOR MATRIX:  newOprod_{rho, nu}_at_A, shortP_at_H
     *
     * If sig is positive, we additionally:
     * READ
     *    3 COLOR MATRIX:  qnumu_at_D, oProd_at_C, newOprod_{sig}_at_A
     * WRITE
     *    1 COLOR MATRIX:  newOprod_{sig}_at_A
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *
     *             if (sig is positive):    (9, 15)
     *             else               :     (9, 11)
     *
     * This function is called 192 times, half positive sig, half negative sig
     *
     * Flop count, in three-number pair (matrix_multi, matrix_add, matrix_rescale)
     *             if (sig is positive)     (17, 11, 8)
     *             else                     (12,  8, 6)
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

      const Gauge p5;
      const Gauge pNuMu;
      const Gauge qNuMu;

      const real coeff_five;
      const real accumu_coeff_five;
      const real coeff_seven;
      const real accumu_coeff_seven;

      static constexpr int overlap = 2;

      AllSevenSideFiveLinkArg(GaugeField &force, GaugeField &shortP, const GaugeField &P5, const GaugeField &pNuMu, const GaugeField &qNuMu,
                 const GaugeField &link, const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), shortP(shortP), p5(P5),
          pNuMu(pNuMu), qNuMu(qNuMu),
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
      static constexpr int mu_positive = Param::mu_positive;
      //static_assert(Param::mu_positive == -1, "mu_positive should be set to -1 for AllSevenSideFiveLink");
      static constexpr int nu_positive = Param::nu_positive;
      static_assert(Param::compute_lepage == -1, "compute_lepage should be set to -1 for AllSevenSideFiveLink");

      constexpr AllSevenSideFiveLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      /**
        @brief Compute the all seven link contribution to the HISQ force
        @param[in] x Local coordinate
        @param[in] e_cb 1-d checkerboard index in the full extended lattice
        @param[in] parity Parity of the coordinate x
        @param[in/out] p5_sig Contribution to the P5 field from the seven link calculation
        @param[in/out] force_sig Contribution to the force in the sigma direction (when sigma is positive)
      */
      __device__ __host__ void all_link(int x[4], int e_cb, int parity, Link &p5_sig, Link &force_sig) {
        auto mycoeff_seven = coeff_sign(sig_positive,parity)*arg.coeff_seven;

        // Intermediate accumulators; force_sig is only needed when sig is positive
        Link force_rho;

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

#pragma unroll
        for (int d=0; d<4; d++) y[d] = x[d];
        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.rho));
        int parity_d = 1 - parity;
        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;
        int dc_link_nbr_idx = (sig_positive) ? point_d : point_c;
        int dc_link_nbr_parity = (sig_positive) ? parity_d : parity_c;

#pragma unroll
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
        Link Oe = arg.pNuMu(0, point_e, parity_e);
        Oz = Ube * Oe;
        Oy = (sig_positive ? Uab : conj(Uab)) * Oz;
        force_rho += (parity_sign(parity_a) * mycoeff_seven) * conj(Of) * conj(Oy);
        if constexpr (sig_positive) {
          force_sig += (parity_sign(parity_a) * mycoeff_seven) * Oz * Of * conj(Uaf);
        }

        // Compute the force_rho and p5 contribution from the positive rho direction
        Link Ufe = arg.link(pos_dir(arg.sig), fe_link_nbr_idx, fe_link_nbr_parity);
        Link Oa = arg.qNuMu(0, point_a, parity_a);
        Link Ob = arg.pNuMu(0, point_b, parity_b);
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
          Link Oc = arg.pNuMu(0, point_c, parity_c);
          Oz = conj(Ucb) * Oc;
          force_sig += (parity_sign(parity_a) * mycoeff_seven) * Oz * Od * Uda;
        }

        // update the force in the rho direction
        Link force = arg.force(pos_dir(arg.rho), point_a, parity_a);
        force += force_rho;
        arg.force(pos_dir(arg.rho), point_a, parity_a) = force;

      }

      /**
        @brief Compute the side five link contribution to the HISQ force
        @param[in] x Local coordinate
        @param[in] e_cb 1-d checkerboard index in the full extended lattice
        @param[in] parity Parity of the coordinate x
        @param[in] P5 Full P5 contribution summed from the previous middle five and all seven
      */
      __device__ __host__ void side_link(int x[4], int e_cb, int parity, const Link &P5) {
        /*      compute the side link contribution to the momentum
         *
         *             sig
         *          A________B
         *           |       |   nu
         *         H |       |G
         *
         *      A is the current point (x_cb)
         *
         * This kernel also depends on U_{mu} flowing *into* point H,
         * formerly referred to as "qMu" or "qProd". We denote the point
         * in the negative "mu" direction from H as "Q".
         */

        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_a = e_cb;
        int parity_a = parity;
        int point_h = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.nu));
        int parity_h = 1 - parity;

        int point_q = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int parity_q = parity;

        int ha_link_nbr_idx = nu_positive ? point_h : point_a;
        int ha_link_nbr_parity = nu_positive ? parity_h : parity_a;

        int qh_link_nbr_idx = mu_positive ? point_q : point_h;
        int qh_link_nbr_parity = mu_positive ? parity_q : parity_h;

        Link Uah = arg.link(pos_dir(arg.nu), ha_link_nbr_idx, ha_link_nbr_parity);
        Link Ow = nu_positive ? Uah * P5 : conj(Uah) * P5;
        Link Uqh = arg.link(pos_dir(arg.mu), qh_link_nbr_idx, qh_link_nbr_parity);
        if constexpr (!mu_positive) Uqh = conj(Uqh);

        Link shortP = arg.shortP(0, point_h, parity_h);
        shortP += arg.accumu_coeff_five * Ow;
        arg.shortP(0, point_h, parity_h) = shortP;

        Ow = nu_positive ? P5 * Uqh : conj(Uqh) * conj(P5);

        auto mycoeff_five = -coeff_sign(goes_forward(arg.sig), parity_a)*coeff_sign(goes_forward(arg.nu),parity_a)*arg.coeff_five;

        Link oprod = arg.force(pos_dir(arg.nu), ha_link_nbr_idx, ha_link_nbr_parity);
        oprod += mycoeff_five * Ow;
        arg.force(pos_dir(arg.nu), ha_link_nbr_idx, ha_link_nbr_parity) = oprod;
      }

      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);
        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity^arg.oddness_change;

        int point_a = e_cb;
        int parity_a = parity;
        
        // calculate p5_sig
        Link force_sig;
        if constexpr (sig_positive) {
          force_sig = arg.force(arg.sig, point_a, parity_a);
        }
        Link P5 = arg.p5(0, point_a, parity_a);

        // accumulate into P5, force_sig
        all_link(x, e_cb, parity, P5, force_sig);

        // update the force in the sigma direction
        if constexpr (sig_positive) {
          arg.force(arg.sig, point_a, parity_a) = force_sig;
        }

        side_link(x, e_cb, parity, P5);

      }
    };

    /********************************allFiveAllSevenLinkKernel*********************************************
     * Note: this kernel names points differently than the other kernels, there's a diagram
     *   within the kernel code
     *
     * In lieu of a lot of copy and paste, we refer to the documentation for AllSevenSideFive and
     * MiddleFive. There is no math reuse so the flops counts for each component are unchanged,
     * and the only reuse in load/store is reusing link_ab and, when sig is positive, the read/write
     * of newOprod_sig_at_A.
     *
     * This kernel uses a double buffer for P5, Pnumu, and Qnumu to safely read from multiple sites
     * in each field during the all_link portion and write to each field during the middle_five portion.
     ************************************************************************************************/
    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct AllFiveAllSevenLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge force;
      Gauge shortP;
      
      const Gauge pMu;

      // double-buffer: read p5, pNuMu, qNuMu for side 5, middle 7
      const Gauge p5;
      const Gauge pNuMu;
      const Gauge qNuMu;

      // write the other p5, pNuMu, qNuMu for next middle 5
      Gauge p5_next;
      Gauge pNuMu_next;
      Gauge qNuMu_next;

      const real coeff_five;
      const real accumu_coeff_five;
      const real coeff_seven;
      const real accumu_coeff_seven;

      static constexpr int overlap = 2;

      AllFiveAllSevenLinkArg(GaugeField &force, GaugeField &shortP, const Gauge &pMu,
                 const GaugeField &P5, const GaugeField &pNuMu, const GaugeField &qNuMu,
                 const GaugeField &P5_next, const GaugeField &pNuMu_next, const GaugeField &qNuMu_next,
                 const GaugeField &link, const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), shortP(shortP), pMu(pMu),
          p5(P5), pNuMu(pNuMu), qNuMu(qNuMu),
          p5_next(P5_next), pNuMu_next(pNuMu_next), qNuMu_next(qNuMu_next),
          coeff_five(act_path_coeff.five), accumu_coeff_five(act_path_coeff.three != 0 ? act_path_coeff.five / act_path_coeff.three : 0),
          coeff_seven(act_path_coeff.seven), accumu_coeff_seven(act_path_coeff.five != 0 ? act_path_coeff.seven / act_path_coeff.five : 0)
      { }

    };

    template <typename Param> struct AllFiveAllSevenLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;

      static constexpr int sig_positive = Param::sig_positive;
      static constexpr int mu_positive = Param::mu_positive;
      //static_assert(Param::mu_positive == -1, "mu_positive should be set to -1 for AllSevenSideFiveLink");
      static constexpr int nu_positive = Param::nu_positive; // if nu_positive == -1, skip
      static constexpr int nu_side_positive = Param::nu_side_positive; // if nu_side_positive == -1, skip
      static_assert(Param::compute_lepage == -1, "compute_lepage should be set to -1 for AllFiveAllSevenLink");

      constexpr AllFiveAllSevenLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      /**
        @brief Compute the all seven link contribution to the HISQ force
        @param[in] x Local coordinate
        @param[in] e_cb 1-d checkerboard index in the full extended lattice
        @param[in] parity Parity of the coordinate x
        @param[in/out] p5_sig Contribution to the P5 field from the seven link calculation
        @param[in/out] force_sig Contribution to the force in the sigma direction (when sigma is positive)
        @param[in] Uab Gauge link going from site a to site b
      */
      __device__ __host__ void all_link(int x[4], int e_cb, int parity, Link &p5_sig, Link &force_sig, const Link &Uab) {
        auto mycoeff_seven = coeff_sign(sig_positive,parity)*arg.coeff_seven;

        // Intermediate accumulators; force_sig is only needed when sig is positive
        Link force_rho;

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

#pragma unroll
        for (int d=0; d<4; d++) y[d] = x[d];
        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.rho));
        int parity_d = 1 - parity;
        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;
        int dc_link_nbr_idx = (sig_positive) ? point_d : point_c;
        int dc_link_nbr_parity = (sig_positive) ? parity_d : parity_c;

#pragma unroll
        for (int d = 0; d < 4; d++) y[d] = x[d];
        int point_f = updateCoordsIndexMILCDir(y, arg.E, arg.rho);
        int parity_f = 1 - parity;
        int point_e = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_e = parity;
        int fe_link_nbr_idx = (sig_positive) ? point_f : point_e;
        int fe_link_nbr_parity = (sig_positive) ? parity_f : parity_e;

        // Compute the force_rho (and force_sig, when sig is positive) contribution
        // from the negative rho direction
        Link Uaf = arg.link(pos_dir(arg.rho), point_a, parity_a);
        Link Ube = arg.link(pos_dir(arg.rho), point_b, parity_b);
        Link Of = arg.qNuMu(0, point_f, parity_f);
        Link Oe = arg.pNuMu(0, point_e, parity_e);
        Oz = Ube * Oe;
        Oy = (sig_positive ? Uab : conj(Uab)) * Oz;
        force_rho += (parity_sign(parity_a) * mycoeff_seven) * conj(Of) * conj(Oy);
        if constexpr (sig_positive) {
          force_sig += (parity_sign(parity_a) * mycoeff_seven) * Oz * Of * conj(Uaf);
        }

        // Compute the force_rho and p5 contribution from the positive rho direction
        Link Ufe = arg.link(pos_dir(arg.sig), fe_link_nbr_idx, fe_link_nbr_parity);
        Link Oa = arg.qNuMu(0, point_a, parity_a);
        Link Ob = arg.pNuMu(0, point_b, parity_b);
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
          Link Oc = arg.pNuMu(0, point_c, parity_c);
          Oz = conj(Ucb) * Oc;
          force_sig += (parity_sign(parity_a) * mycoeff_seven) * Oz * Od * Uda;
        }

        // update the force in the rho direction
        Link force = arg.force(pos_dir(arg.rho), point_a, parity_a);
        force += force_rho;
        arg.force(pos_dir(arg.rho), point_a, parity_a) = force;

      }

      /**
        @brief Compute the side five link contribution to the HISQ force
        @param[in] x Local coordinate
        @param[in] e_cb 1-d checkerboard index in the full extended lattice
        @param[in] parity Parity of the coordinate x
        @param[in] P5 Full P5 contribution summed from the previous middle five and all seven
      */
      __device__ __host__ void side_link(int x[4], int e_cb, int parity, const Link &P5) {
        /*      compute the side link contribution to the momentum
         *
         *             sig
         *          A________B
         *           |       |   nu
         *         H |       |G
         *
         *      A is the current point (x_cb)
         *
         * This kernel also depends on U_{mu} flowing *into* point H,
         * formerly referred to as "qMu" or "qProd". We denote the point
         * in the negative "mu" direction from H as "Q".
         */

        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_a = e_cb;
        int parity_a = parity;
        int point_h = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.nu));
        int parity_h = 1 - parity;

        int point_q = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int parity_q = parity;

        int ha_link_nbr_idx = nu_positive ? point_h : point_a;
        int ha_link_nbr_parity = nu_positive ? parity_h : parity_a;

        int qh_link_nbr_idx = mu_positive ? point_q : point_h;
        int qh_link_nbr_parity = mu_positive ? parity_q : parity_h;

        Link Uah = arg.link(pos_dir(arg.nu), ha_link_nbr_idx, ha_link_nbr_parity);
        Link Ow = nu_positive ? Uah * P5 : conj(Uah) * P5;
        Link Uqh = arg.link(pos_dir(arg.mu), qh_link_nbr_idx, qh_link_nbr_parity);
        if constexpr (!mu_positive) Uqh = conj(Uqh);

        Link shortP = arg.shortP(0, point_h, parity_h);
        shortP += arg.accumu_coeff_five * Ow;
        arg.shortP(0, point_h, parity_h) = shortP;

        Ow = nu_positive ? P5 * Uqh : conj(Uqh) * conj(P5);

        auto mycoeff_five = -coeff_sign(goes_forward(arg.sig), parity_a)*coeff_sign(goes_forward(arg.nu),parity_a)*arg.coeff_five;

        Link oprod = arg.force(pos_dir(arg.nu), ha_link_nbr_idx, ha_link_nbr_parity);
        oprod += mycoeff_five * Ow;
        arg.force(pos_dir(arg.nu), ha_link_nbr_idx, ha_link_nbr_parity) = oprod;
      }

      /**
        @brief Compute the middle five link contribution to the HISQ force
        @param[in] x Local coordinate
        @param[in] e_cb 1-d checkerboard index in the full extended lattice
        @param[in] parity Parity of the coordinate x
        @param[in/out] force_sig Contribution to the force in the sigma direction (when sigma is positive)
        @param[in] Uab Gauge link going from site a to site b
      */
      __device__ __host__ void middle_five(int x[4], int e_cb, int parity, Link &force_sig, const Link &Uab) {

        /*        A________B
         *   nu   |        |
         *       H|        |C
         *
         *    A is the current point (sid)
         *
         * This kernel also depends on U_{mu} flowing *into* point H,
         * formerly referred to as "qMu" or "qProd". We denote the point
         * in the negative "mu" direction from H as "Q".
         */

        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_a = e_cb;
        int parity_a = parity;
        int point_h = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.nu_next));
        int parity_h = 1 - parity;

        int point_q = getIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int parity_q = parity;

        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.nu_next);
        int parity_b = 1 - parity;

        int ha_link_nbr_idx = nu_side_positive ? point_h : point_a;
        int ha_link_nbr_parity = nu_side_positive ? parity_h : parity_a;

        int qh_link_nbr_idx = mu_positive ? point_q : point_h;
        int qh_link_nbr_parity = mu_positive ? parity_q : parity_h;

        int cb_link_nbr_idx = nu_side_positive ? point_c : point_b;
        int cb_link_nbr_parity = nu_side_positive ? parity_c : parity_b;

        // Load link and outer product contributions for pNuMu, P5, qNuMu
        Link Ubc = arg.link(pos_dir(arg.nu_next), cb_link_nbr_idx, cb_link_nbr_parity);
        Link Uha = arg.link(pos_dir(arg.nu_next), ha_link_nbr_idx, ha_link_nbr_parity);
        Link Uqh = arg.link(pos_dir(arg.mu), qh_link_nbr_idx, qh_link_nbr_parity);
        Link Oc = arg.pMu(0, point_c, parity_c);

        Link Ow = !nu_side_positive ? Ubc * Oc : conj(Ubc) * Oc;
        if constexpr (!nu_side_positive) Uha = conj(Uha);
        if constexpr (!mu_positive) Uqh = conj(Uqh);

        arg.pNuMu_next(0, point_b, parity_b) = Ow;
        arg.p5_next(0, point_a, parity_a) = sig_positive ? Uab * Ow : conj(Uab) * Ow;

        Link Ox = Uqh * Uha;
        arg.qNuMu_next(0, point_a, parity_a) = Ox;

        // compute the force in the sigma direction if sig is positive
        if constexpr (sig_positive) {
          force_sig += arg.coeff_five * (Ow * Ox);
        }
      }

      __device__ __host__ void operator()(int x_cb, int parity)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);
        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity^arg.oddness_change;

        int point_a = e_cb;
        int parity_a = parity;
        
        // calculate p5_sig
        Link force_sig;
        if constexpr (sig_positive) {
          force_sig = arg.force(arg.sig, point_a, parity_a);
        }

        // Link Uab can be reused, nothing else can
        int y[4] = {x[0], x[1], x[2], x[3]};
        int point_b = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_b = 1 - parity;
        int ab_link_nbr_idx = (sig_positive) ? point_a : point_b;
        int ab_link_nbr_parity = (sig_positive) ? parity_a : parity_b;
        Link Uab = arg.link(pos_dir(arg.sig), ab_link_nbr_idx, ab_link_nbr_parity);

        if constexpr (nu_positive != NU_IGNORED) {
          Link P5 = arg.p5(0, point_a, parity_a);
          // accumulate into P5, force_sig
          all_link(x, e_cb, parity, P5, force_sig, Uab);
          side_link(x, e_cb, parity, P5);
        }

        if constexpr (nu_side_positive != NU_NEXT_IGNORED) {
          middle_five(x, e_cb, parity, force_sig, Uab);
        }

        // update the force in the sigma direction
        if constexpr (sig_positive) {
          arg.force(arg.sig, point_a, parity_a) = force_sig;
        }

      }
    };


    /************************AllLepageSideThreeLink*****************************
     * Note: this kernel names points differently than the other kernels, there's a diagram
     *   within the kernel code. The site names printed below assume sig and mu are positive.
     *   Many of the load/store locations get shifted if mu is negative; this code uses relatively
     *  self-documenting variables, so check there for the shifted sites.
     *
     * If we are only computing the side three link term (when the Lepage coefficient
     * is zero), we only need:
     * READ
     *    0 LINKS
     *    2 COLOR MATRIX:  p3_at_F, newOprod_{mu}_at_A
     * WRITE
     *    1 COLOR MATRIX:  newOprod_{mu}_at_A
     *
     * Therefore the data traffic, in two-number pair (num of link, num_of_color_matrix), is
     *      (sig positive or negative):   (0, 3)
     *
     * Flop count, in three-number set (matrix_multi, matrix_add, matrix_rescale) is (0, 1, 1)
     *
     * If we are are computing the Lepage middle and all link, this gets complicated.
     * We have an additional:
     * READ
     *    6 LINKS:          hg_link, eg_link, fh_link, fe_link, be_link, ud_link
     *    2 COLOR MATRIX:   oprod_at_e, oprod_at_b
     * If sig is positive, we additionally:
     * READ
     *    1 LINK:           af_link
     *    1 COLOR MATRIX:   newOprod_{sig}_at_F
     * WRITE
     *    1 COLOR MATRIX:   newOprod_{sig}_at_F
     *
     * The TOTAL data traffic (including side link), in two-number pair (num_of_link, num_of_color_matrix)
     *    if (sig is positive):   (7, 7)
     *    else:                   (6, 5)
     *
     * This kernel is called 48 times, 24 for sig positive and 24 for sig negative
     *
     * The TOTAL flop count, in three-number set (matrix_multi, matrix_add, matrix_rescale)
     *   if (sig is positive):     (8, 4, 4)
     *   else:                     (6, 3, 3)
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

      const real coeff_three;
      const real coeff_lepage;
      const real accumu_coeff_lepage;

      static constexpr int overlap = 2;

      AllLepageSideThreeLinkArg(GaugeField &force, const GaugeField &P3, const GaugeField &oProd,
                 const GaugeField &link,
                 const PathCoefficients<real> &act_path_coeff)
        : BaseForceArg(link, overlap), force(force), p3(P3),
        oProd(oProd),
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
        int ab_link_nbr_idx = (sig_positive) ? point_a : point_b;
        int ab_link_nbr_parity = (sig_positive) ? parity_a : parity_b;

#pragma unroll
        for (int d=0; d<4; d++) y[d] = x[d];
        int point_d = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.mu));
        int parity_d = 1 - parity;
        int point_c = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_c = parity;

#pragma unroll
        for (int d=0; d<4; d++) y[d] = x[d];
        int point_f = updateCoordsIndexMILCDir(y, arg.E, arg.mu);
        int parity_f = 1 - parity;
        int point_e = updateCoordsIndexMILCDir(y, arg.E, arg.sig);
        int parity_e = parity;
        int fe_link_nbr_idx = (sig_positive) ? point_f : point_e;
        int fe_link_nbr_parity = (sig_positive) ? parity_f : parity_e;

        int point_g = updateCoordsIndexMILCDir(y, arg.E, arg.mu);
        int parity_g = 1 - parity;
        int point_h = updateCoordsIndexMILCDir(y, arg.E, opp_dir(arg.sig));
        int parity_h = parity;
        int hg_link_nbr_idx = (sig_positive) ? point_h : point_g;
        int hg_link_nbr_parity = (sig_positive) ? parity_h : parity_g;

        int fa_link_nbr_idx = (mu_positive) ? point_f : point_a;
        int fa_link_nbr_parity = (mu_positive) ? parity_f : parity_a;

        Link p3 = arg.p3(0, fa_link_nbr_idx, fa_link_nbr_parity);
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
            Link Uda = arg.link(pos_dir(arg.mu), point_d, parity_d);

            Link Ow = conj(Ube) * Ob;
            Link Oy = sig_positive ? Ufe * Ow : conj(Ufe) * Ow;
            Link Ox = Oy * Uda;

            auto mycoeff_lepage = -coeff_sign(goes_forward(arg.sig), parity)*coeff_sign(goes_forward(arg.mu),parity)*arg.coeff_lepage;

            force_mu += mycoeff_lepage * Ox;

            // Update force_sig if sig is positive
            if constexpr ( sig_positive ) {
              Link Uaf = arg.link(pos_dir(arg.mu), point_a, parity_a);
              Link Ox = Uda * Uaf;
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
              Link Uda = arg.link(pos_dir(arg.mu), point_d, parity_d);

              Link Ow = Ubc * Oc;
              Link Oy = sig_positive ? Uab * Ow : conj(Uab) * Ow;
              Link Ox = Uda * conj(Oy);

              auto mycoeff_lepage = -coeff_sign(goes_forward(arg.sig), parity)*coeff_sign(goes_forward(arg.mu),parity)*arg.coeff_lepage;
              force_mu += mycoeff_lepage * Ox;

              if constexpr ( sig_positive ) {
                Link Uaf = arg.link(pos_dir(arg.mu), point_a, parity_a);
                Link Ox = conj(Uda) * conj(Uaf);
                Link Oy = Ow * Ox;

                Link oprod = arg.force(arg.sig, point_a, parity_a);
                oprod += arg.coeff_lepage * Oy;
                arg.force(arg.sig, point_a, parity_a) = oprod;
              }
            }
          }
        } // if (compute_lepage)

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

      Gauge force;        // force output accessor
      const Gauge oProd; // force input accessor
      const real coeff;

      static constexpr int overlap = 0;

      CompleteForceArg(GaugeField &force, const GaugeField &link)
        : BaseForceArg(link, overlap), force(force), oProd(force), coeff(0.0)
      { }

    };

    template <typename Arg> struct CompleteForce
    {
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      constexpr CompleteForce(const Arg &arg) : arg(arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

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
          Link Ow = Uw * Ox;

          makeAntiHerm(Ow);

          typename Arg::real coeff = (parity==1) ? -1.0 : 1.0;
          arg.force(sig, e_cb, parity) = coeff * Ow;
        }
      }
    };

    template <typename store_t, int nColor_, QudaReconstructType recon>
    struct LongLinkArg : public BaseForceArg<store_t, nColor_, recon> {
      using BaseForceArg = BaseForceArg<store_t, nColor_, recon>;
      using real = typename mapper<store_t>::type;
      static constexpr int nColor = nColor_;
      using Gauge = typename gauge_mapper<real, recon>::type;

      Gauge force;
      const Gauge oProd;
      const real coeff;

      static constexpr int overlap = 0;

      LongLinkArg(GaugeField &force, const GaugeField &link, const GaugeField &oprod, real coeff)
        : BaseForceArg(link, overlap), force(force), oProd(oprod), coeff(coeff)
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
          int parity_c = parity;

          dx[sig]++;
          int point_d = linkIndexShift(x,dx,arg.E);
          int parity_d = 1 - parity;

          dx[sig]++;
          int point_e = linkIndexShift(x,dx,arg.E);
          int parity_e = parity;

          dx[sig] = -1;
          int point_b = linkIndexShift(x,dx,arg.E);
          int parity_b = 1 - parity;

          dx[sig]--;
          int point_a = linkIndexShift(x,dx,arg.E);
          int parity_a = parity;
          dx[sig] = 0;

          Link Uab = arg.link(sig, point_a, parity_a);
          Link Ubc = arg.link(sig, point_b, parity_b);
          Link Ude = arg.link(sig, point_d, parity_d);
          Link Uef = arg.link(sig, point_e, parity_e);

          Link Oz = arg.oProd(sig, point_c, parity_c);
          Link Oy = arg.oProd(sig, point_b, parity_b);
          Link Ox = arg.oProd(sig, point_a, parity_a);

          Link temp = Ude*Uef*Oz - Ude*Oy*Ubc + Ox*Uab*Ubc;

          Link force = arg.force(sig, point_c, parity_c);
          arg.force(sig, point_c, parity_c) = force + arg.coeff * temp;
        } // loop over sig
      }
    };

  }
}

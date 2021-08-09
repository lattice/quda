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

    enum HisqForceType {
      FORCE_ALL_LINK,
      FORCE_MIDDLE_LINK,
      FORCE_LEPAGE_MIDDLE_LINK,
      FORCE_SIDE_LINK,
      FORCE_SIDE_LINK_SHORT,
      FORCE_LONG_LINK,
      FORCE_COMPLETE,
      FORCE_ONE_LINK,
      FORCE_INVALID
    };

    constexpr int opp_dir(int dir) { return 7-dir; }
    constexpr int goes_forward(int dir) { return dir<=3; }
    constexpr int goes_backward(int dir) { return dir>3; }
    constexpr int CoeffSign(int pos_dir, int odd_lattice) { return 2*((pos_dir + odd_lattice + 1) & 1) - 1; }
    constexpr int Sign(int parity) { return parity ? -1 : 1; }
    constexpr int posDir(int dir) { return (dir >= 4) ? 7-dir : dir; }

    template <int dir, typename Arg>
    constexpr void updateCoords(int x[], int shift, const Arg &arg) {
      x[dir] = (x[dir] + shift + arg.E[dir]) % arg.E[dir];
    }

    template <typename Arg>
    constexpr void updateCoords(int x[], int dir, int shift, const Arg &arg) {
      switch (dir) {
      case 0: updateCoords<0>(x, shift, arg); break;
      case 1: updateCoords<1>(x, shift, arg); break;
      case 2: updateCoords<2>(x, shift, arg); break;
      case 3: updateCoords<3>(x, shift, arg); break;
      }
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

    template <typename real_, int nColor_, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct BaseForceArg : kernel_param<> {
      using real = real_;
      static constexpr int nColor = nColor_;
      typedef typename gauge_mapper<real,reconstruct>::type G;
      const G link;
      int X[4]; // regular grid dims
      int D[4]; // working set grid dims
      int E[4]; // extended grid dims

      int commDim[4];
      int border[4];
      int base_idx[4]; // the offset into the extended field
      int oddness_change;
      int mu;
      int sig;

      /**
         @param[in] link Gauge field
         @param[in] overlap Radius of additional redundant computation to do
       */
      BaseForceArg(const GaugeField &link, int overlap) :
        kernel_param(dim3(1, 2, 1)),
        link(link),
        commDim{ comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3) }
      {
        for (int d=0; d<4; d++) {
          E[d] = link.X()[d];
          border[d] = link.R()[d];
          X[d] = E[d] - 2*border[d];
          D[d] = comm_dim_partitioned(d) ? X[d]+overlap*2 : X[d];
          base_idx[d] = comm_dim_partitioned(d) ? border[d]-overlap : 0;
          this->threads.x *= D[d];
        }
        this->threads.x /= 2;
        oddness_change = (base_idx[0] + base_idx[1] + base_idx[2] + base_idx[3])&1;
      }
    };

    template <typename real, int nColor, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct FatLinkArg : public BaseForceArg<real, nColor, reconstruct> {
      using BaseForceArg = BaseForceArg<real, nColor, reconstruct>;
      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F outA;
      F outB;
      F pMu;
      F p3;
      F qMu;

      const F oProd;
      const F qProd;
      const F qPrev;
      const real coeff;
      const real accumu_coeff;

      const bool p_mu;
      const bool q_mu;
      const bool q_prev;

      FatLinkArg(GaugeField &force, const GaugeField &oProd, const GaugeField &link, real coeff, HisqForceType type)
        : BaseForceArg(link, 0), outA(force), outB(force), pMu(oProd), p3(oProd), qMu(oProd),
        oProd(oProd), qProd(oProd), qPrev(oProd), coeff(coeff), accumu_coeff(0),
        p_mu(false), q_mu(false), q_prev(false)
      { if (type != FORCE_ONE_LINK) errorQuda("This constructor is for FORCE_ONE_LINK"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &pMu, GaugeField &P3, GaugeField &qMu,
                 const GaugeField &oProd, const GaugeField &qPrev, const GaugeField &link,
                 real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), outA(newOprod), outB(newOprod), pMu(pMu), p3(P3), qMu(qMu),
        oProd(oProd), qProd(oProd), qPrev(qPrev), coeff(coeff), accumu_coeff(0), p_mu(true), q_mu(true), q_prev(true)
      { if (type != FORCE_MIDDLE_LINK) errorQuda("This constructor is for FORCE_MIDDLE_LINK"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &pMu, GaugeField &P3, GaugeField &qMu,
                 const GaugeField &oProd, const GaugeField &link,
                 real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), outA(newOprod), outB(newOprod), pMu(pMu), p3(P3), qMu(qMu),
        oProd(oProd), qProd(oProd), qPrev(qMu), coeff(coeff), accumu_coeff(0), p_mu(true), q_mu(true), q_prev(false)
      { if (type != FORCE_MIDDLE_LINK) errorQuda("This constructor is for FORCE_MIDDLE_LINK"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &P3, const GaugeField &oProd,
                 const GaugeField &qPrev, const GaugeField &link,
                 real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), outA(newOprod), outB(newOprod), pMu(P3), p3(P3), qMu(qPrev),
        oProd(oProd), qProd(oProd), qPrev(qPrev), coeff(coeff), accumu_coeff(0), p_mu(false), q_mu(false), q_prev(true)
      { if (type != FORCE_LEPAGE_MIDDLE_LINK) errorQuda("This constructor is for FORCE_LEPAGE_MIDDLE_LINK"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &shortP, const GaugeField &P3,
                 const GaugeField &qProd, const GaugeField &link, real coeff, real accumu_coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), outA(newOprod), outB(shortP), pMu(P3), p3(P3), qMu(qProd), oProd(qProd), qProd(qProd),
        qPrev(qProd), coeff(coeff), accumu_coeff(accumu_coeff),
        p_mu(false), q_mu(false), q_prev(false)
      { if (type != FORCE_SIDE_LINK) errorQuda("This constructor is for FORCE_SIDE_LINK"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &P3, const GaugeField &link,
                 real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), outA(newOprod), outB(newOprod),
        pMu(P3), p3(P3), qMu(P3), oProd(P3), qProd(P3), qPrev(P3), coeff(coeff), accumu_coeff(0.0),
        p_mu(false), q_mu(false), q_prev(false)
      { if (type != FORCE_SIDE_LINK_SHORT) errorQuda("This constructor is for FORCE_SIDE_LINK_SHORT"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &shortP, const GaugeField &oProd, const GaugeField &qPrev,
                 const GaugeField &link, real coeff, real accumu_coeff, int overlap, HisqForceType type, bool)
        : BaseForceArg(link, overlap), outA(newOprod), outB(shortP), pMu(shortP),
          p3(shortP), qMu(qPrev), oProd(oProd), qProd(qPrev), qPrev(qPrev),
          coeff(coeff), accumu_coeff(accumu_coeff), p_mu(false), q_mu(false), q_prev(false)
      { if (type != FORCE_ALL_LINK) errorQuda("This constructor is for FORCE_ALL_LINK"); }

    };

    template <typename Arg_, int mu_positive_ = 0, int sig_positive_ = 0,
              bool pMu_ = false, bool qMu_ = false, bool qPrev_ = false>
    struct FatLinkParam : kernel_param<> {
      static constexpr int mu_positive = mu_positive_;
      static constexpr int sig_positive = sig_positive_;
      static constexpr bool pMu = pMu_;
      static constexpr bool qMu = qMu_;
      static constexpr bool qPrev = qPrev_;
      using Arg = Arg_;
      Arg arg;
      FatLinkParam(Arg &arg) :
        kernel_param<>(arg.threads),
        arg(arg) {}
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
        Link force = arg.outA(sig, e_cb, parity);
        force += arg.coeff * w;
        arg.outA(sig, e_cb, parity) = force;
      }
    };

    /********************************allLinkKernel*********************************************
     *
     * In this function we need
     *   READ
     *     3 LINKS:         ad_link, ab_link, bc_link
     *     5 COLOR MATRIX:  Qprev_at_D, oprod_at_C, newOprod_at_A(sig), newOprod_at_D/newOprod_at_A(mu), shortP_at_D
     *   WRITE:
     *     3 COLOR MATRIX:  newOprod_at_A(sig), newOprod_at_D/newOprod_at_A(mu), shortP_at_D,
     *
     * If sig is negative, then we don't need to read/write the color matrix newOprod_at_A(sig)
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *
     *             if (sig is positive):    (3, 8)
     *             else               :     (3, 6)
     *
     * This function is called 384 times, half positive sig, half negative sig
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *             if(sig is positive)      (6,3)
     *             else                     (4,2)
     *
     ************************************************************************************************/
    template <typename Param> struct AllLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      static constexpr int mu_positive = Param::mu_positive;
      static constexpr int sig_positive = Param::sig_positive;

      constexpr AllLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity, int)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);
        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity^arg.oddness_change;

        auto mycoeff = CoeffSign(sig_positive,parity)*arg.coeff;

        int y[4] = {x[0], x[1], x[2], x[3]};
        int mysig = posDir(arg.sig);
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
        int point_b = linkIndex(y,arg.E);
        int ab_link_nbr_idx = (sig_positive) ? e_cb : point_b;

        for (int d=0; d<4; d++) y[d] = x[d];

        /*            sig
         *         A________B
         *      mu  |      |
         *        D |      |C
         *
         *   A is the current point (sid)
         *
         */

        int mu = mu_positive ? arg.mu : opp_dir(arg.mu);
        int dir = mu_positive ? -1 : 1;

        updateCoords(y, mu, dir, arg);
        int point_d = linkIndex(y,arg.E);
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
        int point_c = linkIndex(y,arg.E);

        Link Uab = arg.link(posDir(arg.sig), ab_link_nbr_idx, sig_positive^(1-parity));
        Link Uad = arg.link(mu, mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity);
        Link Ubc = arg.link(mu, mu_positive ? point_c : point_b, mu_positive ? parity : 1-parity);
        Link Ox = arg.qPrev(0, point_d, 1-parity);
        Link Oy = arg.oProd(0, point_c, parity);
        Link Oz = mu_positive ? conj(Ubc)*Oy : Ubc*Oy;

        if (sig_positive) {
          Link force = arg.outA(arg.sig, e_cb, parity);
          force += Sign(parity)*mycoeff*Oz*Ox* (mu_positive ? Uad : conj(Uad));
          arg.outA(arg.sig, e_cb, parity) = force;
          Oy = Uab*Oz;
        } else {
          Oy = conj(Uab)*Oz;
        }

        Link force = arg.outA(mu, mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity);
        force += Sign(mu_positive ? 1-parity : parity)*mycoeff* (mu_positive ? Oy*Ox : conj(Ox)*conj(Oy));
        arg.outA(mu, mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity) = force;

        Link shortP = arg.outB(0, point_d, 1-parity);
        shortP += arg.accumu_coeff* (mu_positive ? Uad : conj(Uad)) *Oy;
        arg.outB(0, point_d, 1-parity) = shortP;
      }
    };


    /**************************middleLinkKernel*****************************
     *
     *
     * Generally we need
     * READ
     *    3 LINKS:         ab_link,     bc_link,    ad_link
     *    3 COLOR MATRIX:  newOprod_at_A, oprod_at_C,  Qprod_at_D
     * WRITE
     *    4 COLOR MATRIX:  newOprod_at_A, P3_at_A, Pmu_at_B, Qmu_at_A
     *
     * Three call variations:
     *   1. when Qprev == NULL:   Qprod_at_D does not exist and is not read in
     *   2. full read/write
     *   3. when Pmu/Qmu == NULL,   Pmu_at_B and Qmu_at_A are not written out
     *
     *   In all three above case, if the direction sig is negative, newOprod_at_A is
     *   not read in or written out.
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *   Call 1:  (called 48 times, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 6)
     *             else               :     (3, 4)
     *   Call 2:  (called 192 time, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 7)
     *             else               :     (3, 5)
     *   Call 3:  (called 48 times, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 5)
     *             else               :     (3, 2) no need to loadQprod_at_D in this case
     *
     * note: oprod_at_C could actually be read in from D when it is the fresh outer product
     *       and we call it oprod_at_C to simply naming. This does not affect our data traffic analysis
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *   call 1:     if (sig is positive)  (3, 1)
     *               else                  (2, 0)
     *   call 2:     if (sig is positive)  (4, 1)
     *               else                  (3, 0)
     *   call 3:     if (sig is positive)  (4, 1)
     *   (Lepage)    else                  (2, 0)
     *
     ****************************************************************************/
    template <typename Param> struct MiddleLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      static constexpr int mu_positive = Param::mu_positive;
      static constexpr int sig_positive = Param::sig_positive;
      static constexpr bool pMu = Param::pMu;
      static constexpr bool qMu = Param::qMu;
      static constexpr bool qPrev = Param::qPrev;

      constexpr MiddleLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity, int)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);

        /*        A________B
         *   mu   |        |
         *       D|        |C
         *
         *	  A is the current point (sid)
         *
         */

        for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity ^ arg.oddness_change;
        int y[4] = {x[0], x[1], x[2], x[3]};

        int mymu = posDir(arg.mu);
        updateCoords(y, mymu, (mu_positive ? -1 : 1), arg);

        int point_d = linkIndex(y, arg.E);
        int ad_link_nbr_idx = mu_positive ? point_d : e_cb;

        int mysig = posDir(arg.sig);
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
        int point_c = linkIndex(y, arg.E);

        for (int d=0; d<4; d++) y[d] = x[d];
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
        int point_b = linkIndex(y, arg.E);

        int bc_link_nbr_idx = mu_positive ? point_c : point_b;
        int ab_link_nbr_idx = sig_positive ? e_cb : point_b;

        // load the link variable connecting a and b
        Link Uab = arg.link(mysig, ab_link_nbr_idx, sig_positive^(1-parity));

        // load the link variable connecting b and c
        Link Ubc = arg.link(mymu, bc_link_nbr_idx, mu_positive^(1-parity));

        Link Oy;
        if (!qPrev) {
          Oy = arg.oProd(posDir(arg.sig), sig_positive ? point_d : point_c, sig_positive^parity);
          if (!sig_positive) Oy = conj(Oy);
        } else { // QprevOdd != NULL
          Oy = arg.oProd(0, point_c, parity);
        }

        Link Ow = !mu_positive ? Ubc*Oy : conj(Ubc)*Oy;

        if (pMu) arg.pMu(0, point_b, 1-parity) = Ow;

        arg.p3(0, e_cb, parity) = sig_positive ? Uab*Ow : conj(Uab)*Ow;

        Link Uad = arg.link(mymu, ad_link_nbr_idx, mu_positive^parity);
        if (!mu_positive)  Uad = conj(Uad);

        if (!qPrev) {
          if (sig_positive) Oy = Ow*Uad;
          if ( qMu ) arg.qMu(0, e_cb, parity) = Uad;
        } else {
          Link Ox;
          if ( qMu || sig_positive ) {
            Oy = arg.qPrev(0, point_d, 1-parity);
            Ox = Oy*Uad;
          }
          if ( qMu ) arg.qMu(0, e_cb, parity) = Ox;
          if (sig_positive) Oy = Ow*Ox;
        }

        if (sig_positive) {
          Link oprod = arg.outA(arg.sig, e_cb, parity);
          oprod += arg.coeff*Oy;
          arg.outA(arg.sig, e_cb, parity) = oprod;
        }

      }
    };

    /***********************************sideLinkKernel***************************
     *
     * In general we need
     * READ
     *    1  LINK:          ad_link
     *    4  COLOR MATRIX:  shortP_at_D, newOprod, P3_at_A, Qprod_at_D,
     * WRITE
     *    2  COLOR MATRIX:  shortP_at_D, newOprod,
     *
     * Two call variations:
     *   1. full read/write
     *   2. when shortP == NULL && Qprod == NULL:
     *          no need to read ad_link/shortP_at_D or write shortP_at_D
     *          Qprod_at_D does not exit and is not read in
     *
     *
     * Therefore the data traffic, in two-number pair (num_of_links, num_of_color_matrix)
     *   Call 1:   (called 192 times)
     *                           (1, 6)
     *
     *   Call 2:   (called 48 times)
     *                           (0, 3)
     *
     * note: newOprod can be at point D or A, depending on if mu is postive or negative
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *   call 1:       (2, 2)
     *   call 2:       (0, 1)
     *
     *********************************************************************************/
    template <typename Param> struct SideLink
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      static constexpr int mu_positive = Param::mu_positive;

      constexpr SideLink(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity, int)
      {
        int x[4];
        getCoords(x, x_cb ,arg.D, parity);
        for (int d=0; d<4; d++) x[d] = x[d] + arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity ^ arg.oddness_change;

        /*      compute the side link contribution to the momentum
         *
         *             sig
         *          A________B
         *           |       |   mu
         *         D |       |C
         *
         *      A is the current point (x_cb)
         *
         */

        int mymu = posDir(arg.mu);
        int y[4] = {x[0], x[1], x[2], x[3]};
        updateCoords(y, mymu, (mu_positive ? -1 : 1), arg);
        int point_d = linkIndex(y,arg.E);

        Link Oy = arg.p3(0, e_cb, parity);

        {
          int ad_link_nbr_idx = mu_positive ? point_d : e_cb;

          Link Uad = arg.link(mymu, ad_link_nbr_idx, mu_positive^parity);
          Link Ow = mu_positive ? Uad*Oy : conj(Uad)*Oy;

          Link shortP = arg.outB(0, point_d, 1-parity);
          shortP += arg.accumu_coeff * Ow;
          arg.outB(0, point_d, 1-parity) = shortP;
        }

        {
          Link Ox = arg.qProd(0, point_d, 1-parity);
          Link Ow = mu_positive ? Oy*Ox : conj(Ox)*conj(Oy);

          auto mycoeff = CoeffSign(goes_forward(arg.sig), parity)*CoeffSign(goes_forward(arg.mu),parity)*arg.coeff;

          Link oprod = arg.outA(mu_positive ? arg.mu : opp_dir(arg.mu), mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity);
          oprod += mycoeff * Ow;
          arg.outA(mu_positive ? arg.mu : opp_dir(arg.mu), mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity) = oprod;
        }
      }
    };

    // Flop count, in two-number pair (matrix_mult, matrix_add)
    // 		(0,1)
    template <typename Param> struct SideLinkShort
    {
      using Arg = typename Param::Arg;
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      const Arg &arg;
      static constexpr int mu_positive = Param::mu_positive;

      constexpr SideLinkShort(const Param &param) : arg(param.arg) {}
      constexpr static const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity, int)
      {
        int x[4];
        getCoords(x, x_cb, arg.D, parity);
        for (int d=0; d<4; d++) x[d] = x[d] + arg.base_idx[d];
        int e_cb = linkIndex(x,arg.E);
        parity = parity ^ arg.oddness_change;

        /*      compute the side link contribution to the momentum
         *
         *             sig
         *          A________B
         *           |       |   mu
         *         D |       |C
         *
         *      A is the current point (x_cb)
         *
         */
        int mymu = posDir(arg.mu);
        int y[4] = {x[0], x[1], x[2], x[3]};
        updateCoords(y, mymu, (mu_positive ? -1 : 1), arg);
        int point_d = mu_positive ? linkIndex(y,arg.E) : e_cb;

        int parity_ = mu_positive ? 1-parity : parity;
        auto mycoeff = CoeffSign(goes_forward(arg.sig),parity)*CoeffSign(goes_forward(arg.mu),parity)*arg.coeff;

        Link Oy = arg.p3(0, e_cb, parity);
        Link oprod = arg.outA(posDir(arg.mu), point_d, parity_);
        oprod += mu_positive ? mycoeff * Oy : mycoeff * conj(Oy);
        arg.outA(posDir(arg.mu), point_d, parity_) = oprod;
      }
    };

    template <typename real, int nColor, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct CompleteForceArg : public BaseForceArg<real, nColor, reconstruct> {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F outA;        // force output accessor
      const F oProd; // force input accessor
      const real coeff;

      CompleteForceArg(GaugeField &force, const GaugeField &link)
        : BaseForceArg<real, nColor, reconstruct>(link, 0), outA(force), oProd(force), coeff(0.0)
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

    template <typename real, int nColor, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct LongLinkArg : public BaseForceArg<real, nColor, reconstruct> {

      typedef typename gauge::FloatNOrder<real,18,2,11> M;
      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F outA;
      const F oProd;
      const real coeff;

      LongLinkArg(GaugeField &newOprod, const GaugeField &link, const GaugeField &oprod, real coeff)
        : BaseForceArg<real, nColor, reconstruct>(link,0), outA(newOprod), oProd(oprod), coeff(coeff)
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

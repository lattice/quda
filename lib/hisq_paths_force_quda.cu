#include <utility>
#include <quda_internal.h>
#include <gauge_field.h>
#include <ks_improved_force.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <index_helper.cuh>
#include <gauge_field_order.h>
#include <instantiate.h>

#ifdef GPU_HISQ_FORCE

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
      const real three;
      const real five;
      const real seven;
      const real naik;
      const real lepage;
      PathCoefficients(const double *path_coeff_array)
        : one(path_coeff_array[0]), naik(path_coeff_array[1]),
          three(path_coeff_array[2]), five(path_coeff_array[3]),
          seven(path_coeff_array[4]), lepage(path_coeff_array[5]) { }
    };

    template <typename real_, int nColor_, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct BaseForceArg {
      using real = real_;
      static constexpr int nColor = nColor_;
      typedef typename gauge_mapper<real,reconstruct>::type G;
      const G link;
      int threads;
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
      BaseForceArg(const GaugeField &link, int overlap) : link(link), threads(1),
        commDim{ comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3) }
      {
        for (int d=0; d<4; d++) {
          E[d] = link.X()[d];
          border[d] = link.R()[d];
          X[d] = E[d] - 2*border[d];
          D[d] = comm_dim_partitioned(d) ? X[d]+overlap*2 : X[d];
          base_idx[d] = comm_dim_partitioned(d) ? border[d]-overlap : 0;
          threads *= D[d];
        }
        threads /= 2;
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
      { if (type != FORCE_LEPAGE_MIDDLE_LINK) errorQuda("This constructor is for FORCE_MIDDLE_LINK"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &shortP, const GaugeField &P3,
                 const GaugeField &qProd, const GaugeField &link, real coeff, real accumu_coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), outA(newOprod), outB(shortP), pMu(P3), p3(P3), qMu(qProd), oProd(qProd), qProd(qProd),
        qPrev(qProd), coeff(coeff), accumu_coeff(accumu_coeff),
        p_mu(false), q_mu(false), q_prev(false)
      { if (type != FORCE_SIDE_LINK) errorQuda("This constructor is for FORCE_SIDE_LINK or FORCE_ALL_LINK"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &P3, const GaugeField &link,
                 real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), outA(newOprod), outB(newOprod),
        pMu(P3), p3(P3), qMu(P3), oProd(P3), qProd(P3), qPrev(P3), coeff(coeff), accumu_coeff(0.0),
        p_mu(false), q_mu(false), q_prev(false)
      { if (type != FORCE_SIDE_LINK_SHORT) errorQuda("This constructor is for FORCE_SIDE_LINK_SHORT"); }

      FatLinkArg(GaugeField &newOprod, GaugeField &shortP, const GaugeField &oProd, const GaugeField &qPrev,
                 const GaugeField &link, real coeff, real accumu_coeff, int overlap, HisqForceType type, bool dummy)
        : BaseForceArg(link, overlap), outA(newOprod), outB(shortP), oProd(oProd), qPrev(qPrev),
        pMu(shortP), p3(shortP), qMu(qPrev), qProd(qPrev), // dummy
        coeff(coeff), accumu_coeff(accumu_coeff), p_mu(false), q_mu(false), q_prev(false)
      { if (type != FORCE_ALL_LINK) errorQuda("This constructor is for FORCE_ALL_LINK"); }

    };

    template <typename Arg>
    __global__ void oneLinkTermKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;
      int sig = blockIdx.z * blockDim.z + threadIdx.z;
      if (sig >= 4) return;

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
    template <int sig_positive, int mu_positive, typename Arg>
    __global__ void allLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;

      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

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
    template <int sig_positive, int mu_positive, bool pMu, bool qMu, bool qPrev, typename Arg>
    __global__ void middleLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;

      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

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
    template <int mu_positive, typename Arg>
    __global__ void sideLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

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

    // Flop count, in two-number pair (matrix_mult, matrix_add)
    // 		(0,1)
    template <int mu_positive, typename Arg>
    __global__ void sideLinkShortKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

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

    template <typename Arg>
    class FatLinkForce : public TunableVectorYZ {

      Arg &arg;
      const GaugeField &meta;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      FatLinkForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorYZ(2,type == FORCE_ONE_LINK ? 4 : 1), arg(arg), meta(meta), type(type) {
        arg.sig = sig;
        arg.mu = mu;
      }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << meta.AuxString() << comm_dim_partitioned_string() << ",threads=" << arg.threads;
        if (type == FORCE_MIDDLE_LINK || type == FORCE_LEPAGE_MIDDLE_LINK)
          aux << ",sig=" << arg.sig << ",mu=" << arg.mu << ",pMu=" << arg.p_mu << ",q_muu=" << arg.q_mu << ",q_prev=" << arg.q_prev;
        else if (type != FORCE_ONE_LINK)
          aux << ",mu=" << arg.mu; // no sig dependence needed for side link

        switch (type) {
        case FORCE_ONE_LINK:           aux << ",ONE_LINK";           break;
        case FORCE_ALL_LINK:           aux << ",ALL_LINK";           break;
        case FORCE_MIDDLE_LINK:        aux << ",MIDDLE_LINK";        break;
        case FORCE_LEPAGE_MIDDLE_LINK: aux << ",LEPAGE_MIDDLE_LINK"; break;
        case FORCE_SIDE_LINK:          aux << ",SIDE_LINK";          break;
        case FORCE_SIDE_LINK_SHORT:    aux << ",SIDE_LINK_SHORT";    break;
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const qudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        switch (type) {
        case FORCE_ONE_LINK:
          oneLinkTermKernel<Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        case FORCE_ALL_LINK:
          if (goes_forward(arg.sig) && goes_forward(arg.mu))
            allLinkKernel<1,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_forward(arg.sig) && goes_backward(arg.mu))
            allLinkKernel<1,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_backward(arg.sig) && goes_forward(arg.mu))
            allLinkKernel<0,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else
            allLinkKernel<0,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        case FORCE_MIDDLE_LINK:
          if (!arg.p_mu || !arg.q_mu) errorQuda("Expect p_mu=%d and q_mu=%d to both be true", arg.p_mu, arg.q_mu);
          if (arg.q_prev) {
            if (goes_forward(arg.sig) && goes_forward(arg.mu))
              middleLinkKernel<1,1,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_forward(arg.sig) && goes_backward(arg.mu))
              middleLinkKernel<1,0,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_backward(arg.sig) && goes_forward(arg.mu))
              middleLinkKernel<0,1,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else
              middleLinkKernel<0,0,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          } else {
            if (goes_forward(arg.sig) && goes_forward(arg.mu))
              middleLinkKernel<1,1,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_forward(arg.sig) && goes_backward(arg.mu))
              middleLinkKernel<1,0,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_backward(arg.sig) && goes_forward(arg.mu))
              middleLinkKernel<0,1,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else
              middleLinkKernel<0,0,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          }
          break;
        case FORCE_LEPAGE_MIDDLE_LINK:
          if (arg.p_mu || arg.q_mu || !arg.q_prev)
            errorQuda("Expect p_mu=%d and q_mu=%d to both be false and q_prev=%d true", arg.p_mu, arg.q_mu, arg.q_prev);
          if (goes_forward(arg.sig) && goes_forward(arg.mu))
            middleLinkKernel<1,1,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_forward(arg.sig) && goes_backward(arg.mu))
            middleLinkKernel<1,0,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_backward(arg.sig) && goes_forward(arg.mu))
            middleLinkKernel<0,1,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else
            middleLinkKernel<0,0,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        case FORCE_SIDE_LINK:
          if (goes_forward(arg.mu)) sideLinkKernel<1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else                      sideLinkKernel<0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        case FORCE_SIDE_LINK_SHORT:
          if (goes_forward(arg.mu)) sideLinkShortKernel<1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else                      sideLinkShortKernel<0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_ONE_LINK:
          arg.outA.save();
          break;
        case FORCE_ALL_LINK:
          arg.outA.save();
          arg.outB.save();
          break;
        case FORCE_MIDDLE_LINK:
          arg.pMu.save();
          arg.qMu.save();
        case FORCE_LEPAGE_MIDDLE_LINK:
          arg.outA.save();
          arg.p3.save();
          break;
        case FORCE_SIDE_LINK:
          arg.outB.save();
        case FORCE_SIDE_LINK_SHORT:
          arg.outA.save();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_ONE_LINK:
          arg.outA.load();
          break;
        case FORCE_ALL_LINK:
          arg.outA.load();
          arg.outB.load();
          break;
        case FORCE_MIDDLE_LINK:
          arg.pMu.load();
          arg.qMu.load();
        case FORCE_LEPAGE_MIDDLE_LINK:
          arg.outA.load();
          arg.p3.load();
          break;
        case FORCE_SIDE_LINK:
          arg.outB.load();
        case FORCE_SIDE_LINK_SHORT:
          arg.outA.load();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_ONE_LINK:
          return 2*4*arg.threads*36ll;
        case FORCE_ALL_LINK:
          return 2*arg.threads*(goes_forward(arg.sig) ? 1242ll : 828ll);
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads*(2 * 198 +
                                (!arg.q_prev && goes_forward(arg.sig) ? 198 : 0) +
                                (arg.q_prev && (arg.q_mu || goes_forward(arg.sig) ) ? 198 : 0) +
                                ((arg.q_prev && goes_forward(arg.sig) ) ?  198 : 0) +
                                ( goes_forward(arg.sig) ? 216 : 0) );
        case FORCE_SIDE_LINK:       return 2*arg.threads*2*234;
        case FORCE_SIDE_LINK_SHORT: return 2*arg.threads*36;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_ONE_LINK:
          return 2*4*arg.threads*( arg.oProd.Bytes() + 2*arg.outA.Bytes() );
        case FORCE_ALL_LINK:
          return 2*arg.threads*( (goes_forward(arg.sig) ? 4 : 2)*arg.outA.Bytes() + 3*arg.link.Bytes()
                                 + arg.oProd.Bytes() + arg.qPrev.Bytes() + 2*arg.outB.Bytes());
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads*( ( goes_forward(arg.sig) ? 2*arg.outA.Bytes() : 0 ) +
                                 (arg.p_mu ? arg.pMu.Bytes() : 0) +
                                 (arg.q_mu ? arg.qMu.Bytes() : 0) +
                                 ( ( goes_forward(arg.sig) || arg.q_mu ) ? arg.qPrev.Bytes() : 0) +
                                 arg.p3.Bytes() + 3*arg.link.Bytes() + arg.oProd.Bytes() );
        case FORCE_SIDE_LINK:
          return 2*arg.threads*( 2*arg.outA.Bytes() + 2*arg.outB.Bytes() +
                                 arg.p3.Bytes() + arg.link.Bytes() + arg.qProd.Bytes() );
        case FORCE_SIDE_LINK_SHORT:
          return 2*arg.threads*( 2*arg.outA.Bytes() + arg.p3.Bytes() );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqStaplesForce {
      HisqStaplesForce(GaugeField &Pmu, GaugeField &P3, GaugeField &P5, GaugeField &Pnumu,
                       GaugeField &Qmu, GaugeField &Qnumu, GaugeField &newOprod,
                       const GaugeField &oprod, const GaugeField &link,
                       const double *path_coeff_array)
      {
        PathCoefficients<real> act_path_coeff(path_coeff_array);
        real OneLink = act_path_coeff.one;
        real ThreeSt = act_path_coeff.three;
        real mThreeSt = -ThreeSt;
        real FiveSt  = act_path_coeff.five;
        real mFiveSt  = -FiveSt;
        real SevenSt = act_path_coeff.seven;
        real Lepage  = act_path_coeff.lepage;
        real mLepage  = -Lepage;

        FatLinkArg<real, nColor> arg(newOprod, oprod, link, OneLink, FORCE_ONE_LINK);
        FatLinkForce<decltype(arg)> oneLink(arg, link, 0, 0, FORCE_ONE_LINK);
        oneLink.apply(0);

        for (int sig=0; sig<8; sig++) {
          for (int mu=0; mu<8; mu++) {
            if ( (mu == sig) || (mu == opp_dir(sig))) continue;

            //3-link
            //Kernel A: middle link
            FatLinkArg<real, nColor> middleLinkArg( newOprod, Pmu, P3, Qmu, oprod, link, mThreeSt, 2, FORCE_MIDDLE_LINK);
            FatLinkForce<decltype(arg)> middleLink(middleLinkArg, link, sig, mu, FORCE_MIDDLE_LINK);
            middleLink.apply(0);

            for (int nu=0; nu < 8; nu++) {
              if (nu == sig || nu == opp_dir(sig) || nu == mu || nu == opp_dir(mu)) continue;

              //5-link: middle link
              //Kernel B
              FatLinkArg<real, nColor> middleLinkArg( newOprod, Pnumu, P5, Qnumu, Pmu, Qmu, link, FiveSt, 1, FORCE_MIDDLE_LINK);
              FatLinkForce<decltype(arg)> middleLink(middleLinkArg, link, sig, nu, FORCE_MIDDLE_LINK);
              middleLink.apply(0);

              for (int rho = 0; rho < 8; rho++) {
                if (rho == sig || rho == opp_dir(sig) || rho == mu || rho == opp_dir(mu) || rho == nu || rho == opp_dir(nu)) continue;

                //7-link: middle link and side link
                FatLinkArg<real, nColor> arg(newOprod, P5, Pnumu, Qnumu, link, SevenSt, FiveSt != 0 ? SevenSt/FiveSt : 0, 1, FORCE_ALL_LINK, true);
                FatLinkForce<decltype(arg)> all(arg, link, sig, rho, FORCE_ALL_LINK);
                all.apply(0);

              }//rho

              //5-link: side link
              FatLinkArg<real, nColor> arg(newOprod, P3, P5, Qmu, link, mFiveSt, (ThreeSt != 0 ? FiveSt/ThreeSt : 0), 1, FORCE_SIDE_LINK);
              FatLinkForce<decltype(arg)> side(arg, link, sig, nu, FORCE_SIDE_LINK);
              side.apply(0);

            } //nu

            //lepage
            if (Lepage != 0.) {
              FatLinkArg<real, nColor> middleLinkArg( newOprod, P5, Pmu, Qmu, link, Lepage, 2, FORCE_LEPAGE_MIDDLE_LINK);
              FatLinkForce<decltype(arg)> middleLink(middleLinkArg, link, sig, mu, FORCE_LEPAGE_MIDDLE_LINK);
              middleLink.apply(0);

              FatLinkArg<real, nColor> arg(newOprod, P3, P5, Qmu, link, mLepage, (ThreeSt != 0 ? Lepage/ThreeSt : 0), 2, FORCE_SIDE_LINK);
              FatLinkForce<decltype(arg)> side(arg, link, sig, mu, FORCE_SIDE_LINK);
              side.apply(0);
            } // Lepage != 0.0

            // 3-link side link
            FatLinkArg<real, nColor> arg(newOprod, P3, link, ThreeSt, 1, FORCE_SIDE_LINK_SHORT);
            FatLinkForce<decltype(arg)> side(arg, P3, sig, mu, FORCE_SIDE_LINK_SHORT);
            side.apply(0);
          }//mu
        }//sig
      }
    };

    void hisqStaplesForce(GaugeField &newOprod, const GaugeField &oprod, const GaugeField &link, const double path_coeff_array[6])
    {
      if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());
      if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
      if (checkLocation(newOprod,oprod,link) == QUDA_CPU_FIELD_LOCATION) errorQuda("CPU not implemented");

      // create color matrix fields with zero padding
      GaugeFieldParam gauge_param(link);
      gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
      gauge_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      gauge_param.geometry = QUDA_SCALAR_GEOMETRY;

      cudaGaugeField Pmu(gauge_param);
      cudaGaugeField P3(gauge_param);
      cudaGaugeField P5(gauge_param);
      cudaGaugeField Pnumu(gauge_param);
      cudaGaugeField Qmu(gauge_param);
      cudaGaugeField Qnumu(gauge_param);

      QudaPrecision precision = checkPrecision(oprod, link, newOprod);
      instantiate<HisqStaplesForce, ReconstructNone>(Pmu, P3, P5, Pnumu, Qmu, Qnumu, newOprod, oprod, link, path_coeff_array);

      cudaDeviceSynchronize();
      checkCudaError();
    }

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

    // Flops count: 4 matrix multiplications per lattice site = 792 Flops per site
    template <typename Arg>
    __global__ void completeForceKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

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

    // Flops count, in two-number pair (matrix_mult, matrix_add)
    // 				   (24, 12)
    // 4968 Flops per site in total
    template <typename Arg>
    __global__ void longLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

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

    template <typename Arg>
    class HisqForce : public TunableVectorY {

      Arg &arg;
      const GaugeField &meta;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      HisqForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorY(2), arg(arg), meta(meta), type(type) {
        arg.sig = sig;
        arg.mu = mu;
      }

      void apply(const qudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        switch (type) {
        case FORCE_LONG_LINK:
          longLinkKernel<Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
        case FORCE_COMPLETE:
          completeForceKernel<Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << meta.AuxString() << comm_dim_partitioned_string() << ",threads=" << arg.threads;
        switch (type) {
        case FORCE_LONG_LINK: aux << ",LONG_LINK"; break;
        case FORCE_COMPLETE:  aux << ",COMPLETE";  break;
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void preTune() {
        switch (type) {
        case FORCE_LONG_LINK:
        case FORCE_COMPLETE:
          arg.outA.save(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_LONG_LINK:
        case FORCE_COMPLETE:
          arg.outA.load(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_LONG_LINK: return 2*arg.threads*4968ll;
        case FORCE_COMPLETE:  return 2*arg.threads*792ll;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_LONG_LINK: return 4*2*arg.threads*(2*arg.outA.Bytes() + 4*arg.link.Bytes() + 3*arg.oProd.Bytes());
        case FORCE_COMPLETE:  return 4*2*arg.threads*(arg.outA.Bytes() + arg.link.Bytes() + arg.oProd.Bytes());
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqLongLinkForce {
      HisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
      {
        LongLinkArg<real, nColor, recon> arg(newOprod, link, oldOprod, coeff);
        HisqForce<decltype(arg)> longLink(arg, link, 0, 0, FORCE_LONG_LINK);
        longLink.apply(0);
        cudaDeviceSynchronize();
        checkCudaError();
      }
    };

    void hisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
    {
      if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      if (!oldOprod.isNative()) errorQuda("Unsupported gauge order %d", oldOprod.Order());
      if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
      if (checkLocation(newOprod,oldOprod,link) == QUDA_CPU_FIELD_LOCATION) errorQuda("CPU not implemented");
      checkPrecision(newOprod, link, oldOprod);
      instantiate<HisqLongLinkForce, ReconstructNone>(newOprod, oldOprod, link, coeff);
    }

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqCompleteForce {
      HisqCompleteForce(GaugeField &force, const GaugeField &link)
      {
        CompleteForceArg<real, nColor, recon> arg(force, link);
        HisqForce<decltype(arg)> completeForce(arg, link, 0, 0, FORCE_COMPLETE);
        completeForce.apply(0);
        cudaDeviceSynchronize();
        checkCudaError();
      }
    };

    void hisqCompleteForce(GaugeField &force, const GaugeField &link)
    {
      if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
      if (checkLocation(force,link) == QUDA_CPU_FIELD_LOCATION) errorQuda("CPU not implemented");
      checkPrecision(link, force);
      instantiate<HisqCompleteForce, ReconstructNone>(force, link);
    }

  } // namespace fermion_force

} // namespace quda

#endif // GPU_HISQ_FORCE

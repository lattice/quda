#include <quda_internal.h>
#include <gauge_field.h>
#include <ks_improved_force.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <index_helper.cuh>
#include <gauge_field_order.h>

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

    __device__ __host__ constexpr inline int opp_dir(int dir) { return 7-dir; }
    __device__ __host__ constexpr inline int goes_forward(int dir) { return dir<=3; }
    __device__ __host__ constexpr inline int goes_backward(int dir) { return dir>3; }
    __device__ __host__ constexpr inline int CoeffSign(int pos_dir, int odd_lattice) { return 2*((pos_dir + odd_lattice + 1) & 1) - 1; }
    __device__ __host__ constexpr inline int Sign(int parity) { return parity ? -1 : 1; }
    __device__ __host__ constexpr inline int posDir(int dir) { return (dir >= 4) ? 7-dir : dir; }

    template <int dir, typename Arg>
    inline __device__ __host__ void updateCoords(int x[], int shift, const Arg &arg) {
      x[dir] = (x[dir] + shift + arg.E[dir]) % arg.E[dir];
    }

    template <typename Arg>
    inline __device__ __host__ void updateCoords(int x[], int dir, int shift, const Arg &arg) {
      switch (dir) {
        case 0:
	  updateCoords<0>(x, shift, arg);
	  break;
        case 1:
	  updateCoords<1>(x, shift, arg);
          break;
        case 2:
	  updateCoords<2>(x, shift, arg);
	  break;
        case 3:
	  updateCoords<3>(x, shift, arg);
	  break;
      }
      return;
    }

    //struct for holding the fattening path coefficients
    template<class Real>
      struct PathCoefficients
      {
        Real one;
        Real three;
        Real five;
        Real seven;
        Real naik;
        Real lepage;
      };

    struct BaseForceArg {
      int threads;
      int X[4]; // regular grid dims
      int D[4]; // working set grid dims
      int E[4]; // extended grid dims

      int commDim[4];
      int border[4];
      int base_idx[4];
      int oddness_change;

      int mu;
      int sig;

      BaseForceArg(const GaugeField &meta, int overlap) : threads(1),
        commDim{ comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3) }
      {
        for (int d=0; d<4; d++) {
          E[d] = meta.X()[d];
          border[d] = meta.R()[d];
          X[d] = E[d] - 2*border[d];
          D[d] = comm_dim_partitioned(d) ? X[d]+overlap*2 : X[d];
          base_idx[d] = comm_dim_partitioned(d) ? border[d]-overlap : 0;
          threads *= D[d];
        }
        threads /= 2;
        oddness_change = (base_idx[0] + base_idx[1] + base_idx[2] + base_idx[3])&1;
      }
    };

    template <typename real, QudaReconstructType reconstruct>
    struct CompleteForceArg : public BaseForceArg {

      typedef typename gauge::FloatNOrder<real,18,2,11> M;
      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      typedef typename gauge_mapper<real,reconstruct>::type G;

      M force;
      const F oprod;
      const G link;
      const real coeff;

      CompleteForceArg(GaugeField &force, const GaugeField &link, const GaugeField &oprod)
        : BaseForceArg(link, 0), force(force), link(link), oprod(oprod), coeff(0.0)
      {
        if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
        if (!link.isNative())  errorQuda("Unsupported gauge order %d", link.Order());
        if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());
      }

    };

    // Flops count: 4 matrix multiplications per lattice site = 792 Flops per site
    template <typename real, typename Arg>
    __global__ void completeForceKernel(Arg arg)
    {
      typedef Matrix<complex<real>,3> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      getCoords(x, x_cb, arg.X, parity);

      for (int d=0; d<4; d++) x[d] += arg.border[d];
      int e_cb = linkIndex(x,arg.E);

      for (int sig=0; sig<4; ++sig) {
        Link Uw = arg.link(sig, e_cb, parity);
        Link Ox = arg.oprod(sig, e_cb, parity);
        Link Ow = Uw*Ox;

        makeAntiHerm(Ow);

        real coeff = (parity==1) ? -1.0 : 1.0;
        arg.force(sig, x_cb, parity) = coeff*Ow;
      }
    }

    template <typename real, QudaReconstructType reconstruct>
    struct LongLinkArg : public BaseForceArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      typedef typename gauge_mapper<real,reconstruct>::type G;

      F force;
      const F oprod;
      const G link;
      const real coeff;

      LongLinkArg(GaugeField &force, const GaugeField &link, const GaugeField &oprod, real coeff)
        : BaseForceArg(link,0), force(force), link(link), oprod(oprod), coeff(coeff)
      {
        if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
        if (!link.isNative())  errorQuda("Unsupported gauge order %d", link.Order());
        if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());
      }

    };

    // Flops count, in two-number pair (matrix_mult, matrix_add)
    // 				   (24, 12)
    // 4968 Flops per site in total
    template <typename real, typename Arg>
    __global__ void longLinkKernel(Arg arg)
    {
      typedef Matrix<complex<real>,3> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      int dx[4] = {0,0,0,0};

      getCoords(x, x_cb, arg.X, parity);

      for (int i=0; i<4; i++) x[i] += arg.border[i];
      int e_cb = linkIndexShift(x,dx,arg.E);

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

        Link Oz = arg.oprod(sig, point_c, parity);
        Link Oy = arg.oprod(sig, point_b, 1-parity);
        Link Ox = arg.oprod(sig, point_a, parity);

        Link temp = Ude*Uef*Oz - Ude*Oy*Ubc + Ox*Uab*Ubc;

        Link force = arg.force(sig, e_cb, parity);
        arg.force(sig, e_cb, parity) = force + arg.coeff*temp;
      } // loop over sig

    }

    template <typename real, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct HisqForceArg : public BaseForceArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      typedef typename gauge_mapper<real,reconstruct>::type G;

      F force;
      F shortP;
      const F oprod;
      const F qPrev;
      const G link;

      const real coeff;
      const real accumu_coeff;

      HisqForceArg(GaugeField &force, GaugeField &shortP, const GaugeField &link, const GaugeField &oprod,
                   const GaugeField &qPrev, real coeff, real accumu_coeff, int overlap)
        : BaseForceArg(link, overlap), force(force), shortP(shortP), link(link), oprod(oprod), qPrev(qPrev),
          coeff(coeff), accumu_coeff(accumu_coeff)
      {
        if (link.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct type %d", link.Reconstruct());
        if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
        if (!shortP.isNative()) errorQuda("Unsupported gauge order %d", shortP.Order());
        if (!link.isNative())  errorQuda("Unsupported gauge order %d", link.Order());
        if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());
        if (!qPrev.isNative()) errorQuda("Unsupported gauge order %d", qPrev.Order());
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

    // 198 flops per matrix multiply
    // 18 flops per matrix addition
    // if(sig is positive) 1242 Flops per lattice site
    // else 828 Flops per lattice site
    //
    // Aggregate Flops per site
    // 1242*192 + 828*192
    // = 397440 Flops per site
    template<typename real, int sig_positive, int mu_positive, typename Arg>
    __global__ void allLinkKernel(Arg arg)
    {
      typedef Matrix<complex<real>,3> Link;

      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      getCoords(x, x_cb, arg.D, parity);
      for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
      int e_cb = linkIndex(x,arg.E);
      parity = parity^arg.oddness_change;

      real mycoeff = CoeffSign(sig_positive,parity)*arg.coeff;

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
      if (mu_positive) { //positive mu
        updateCoords(y, arg.mu, -1, arg);
        int point_d = linkIndex(y,arg.E);
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
        int point_c = linkIndex(y,arg.E);

        Link Uab = arg.link(posDir(arg.sig), ab_link_nbr_idx, sig_positive^(1-parity));
        Link Uad = arg.link(arg.mu, point_d, 1-parity);
        Link Ubc = arg.link(arg.mu, point_c, parity);
        Link Ox = arg.qPrev(0, point_d, 1-parity);
        Link Oy = arg.oprod(0, point_c, parity);
        Link Oz = conj(Ubc)*Oy;

        if (sig_positive) {
          Link force = arg.force(arg.sig, e_cb, parity);
          force += Sign(parity)*mycoeff*Oz*Ox*Uad;
          arg.force(arg.sig, e_cb, parity) = force;
          Oy = Uab*Oz;
        } else {
          Oy = conj(Uab)*Oz;
        }

        Link force = arg.force(arg.mu, point_d, 1-parity);
        force += -Sign(parity)*mycoeff*Oy*Ox;
        arg.force(arg.mu, point_d, 1-parity) = force;

        Link shortP = arg.shortP(0, point_d, 1-parity);
        shortP += arg.accumu_coeff*Uad*Oy;
        arg.shortP(0, point_d, 1-parity) = shortP;
      } else { //negative mu

        int mu = opp_dir(arg.mu);
        updateCoords(y, mu, 1, arg);
        int point_d = linkIndex(y,arg.E);
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
        int point_c = linkIndex(y,arg.E);

        Link Uab = arg.link(posDir(arg.sig), ab_link_nbr_idx, sig_positive^(1-parity));
        Link Uad = arg.link(mu, e_cb, parity);
        Link Ubc = arg.link(mu, point_b, 1-parity);
        Link Ox = arg.qPrev(0, point_d, 1-parity);
        Link Oy = arg.oprod(0, point_c, parity);
        Link Oz = Ubc*Oy;

        if (sig_positive) {
          Oy = Oz*Ox*conj(Uad);
          Link force = arg.force(arg.sig, e_cb, parity);
          force += Sign(parity)*mycoeff*Oy;
          arg.force(arg.sig, e_cb, parity) = force;
          Oy = Uab*Oz;
        } else {
          Oy = conj(Uab)*Oz;
        }

        Link force = arg.force(mu, e_cb, parity);
        force += Sign(parity)*mycoeff*conj(Ox)*conj(Oy);
        arg.force(mu, e_cb, parity) = force;

        Link shortP = arg.shortP(0, point_d, 1-parity);
        shortP += arg.accumu_coeff*conj(Uad)*Oy;
        arg.shortP(0, point_d, 1-parity) = shortP;
      }
    }


    template <typename real, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct MiddleLinkArg : public BaseForceArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      typedef typename gauge_mapper<real,reconstruct>::type G;
      F newOprod;
      F pMu;
      F p3;
      F qMu;
      const F oProd;
      const F qPrev;
      const G link;
      const real coeff;

      const bool p_mu;
      const bool q_mu;
      const bool q_prev;

      MiddleLinkArg(GaugeField &newOprod, GaugeField &pMu, GaugeField &P3, GaugeField &qMu,
                    const GaugeField &oProd, const GaugeField &qPrev, const GaugeField &link,
                    real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), newOprod(newOprod), pMu(pMu), p3(P3), qMu(qMu),
          oProd(oProd), qPrev(qPrev), link(link), coeff(coeff), p_mu(true), q_mu(true), q_prev(true)
      {
        if (type != FORCE_MIDDLE_LINK) errorQuda("This constructor is for FORCE_MIDDLE_LINK");
        if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
        if (!qPrev.isNative()) errorQuda("Unsupported gauge order %d", qPrev.Order());
        if (!qMu.isNative()) errorQuda("Unsupported gauge order %d", qMu.Order());
        if (!pMu.isNative()) errorQuda("Unsupported gauge order %d", pMu.Order());
        if (!P3.isNative()) errorQuda("Unsupported gauge order %d", P3.Order());
        if (!oProd.isNative()) errorQuda("Unsupported gauge order %d", oProd.Order());
        if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      }

      MiddleLinkArg(GaugeField &newOprod, GaugeField &pMu, GaugeField &P3, GaugeField &qMu,
                    const GaugeField &oProd, const GaugeField &link,
                    real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), newOprod(newOprod), pMu(pMu), p3(P3), qMu(qMu),
          oProd(oProd), qPrev(qMu), link(link), coeff(coeff), p_mu(true), q_mu(true), q_prev(false)
      {
        if (type != FORCE_MIDDLE_LINK) errorQuda("This constructor is for FORCE_MIDDLE_LINK");
        if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
        if (!qMu.isNative()) errorQuda("Unsupported gauge order %d", qMu.Order());
        if (!pMu.isNative()) errorQuda("Unsupported gauge order %d", pMu.Order());
        if (!P3.isNative()) errorQuda("Unsupported gauge order %d", P3.Order());
        if (!oProd.isNative()) errorQuda("Unsupported gauge order %d", oProd.Order());
        if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      }

      MiddleLinkArg(GaugeField &newOprod, GaugeField &P3, const GaugeField &oProd,
                    const GaugeField &qPrev, const GaugeField &link,
                    real coeff, int overlap, HisqForceType type)
        : BaseForceArg(link, overlap), newOprod(newOprod), pMu(P3), p3(P3), qMu(qPrev),
          oProd(oProd), qPrev(qPrev), link(link), coeff(coeff), p_mu(false), q_mu(false), q_prev(true)
      {
        if (type != FORCE_LEPAGE_MIDDLE_LINK) errorQuda("This constructor is for FORCE_MIDDLE_LINK");
        if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
        if (!qPrev.isNative()) errorQuda("Unsupported gauge order %d", qPrev.Order());
        if (!P3.isNative()) errorQuda("Unsupported gauge order %d", P3.Order());
        if (!oProd.isNative()) errorQuda("Unsupported gauge order %d", oProd.Order());
        if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      }

    };

    /**************************do_middle_link_kernel*****************************
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
     *               else                  (2, 0)
     *
     ****************************************************************************/
    // call 1: if (sig is positive) 612 Flops per site
    // 	   else 		396 Flops per site
    //
    // call 2: if (sig is positive) 810 Flops per site
    // 	   else 		594 Flops per site
    //
    // call 3: if (sig is positive) 810 Flops per site
    // 	   else			396 Flops per site
    //
    // call 1: 24 times with +ve sig and 24 times with -ve sig
    // 	   24192 Flops per site for the full 48 calls
    //
    // call 2: 96 times with +ve sig and 96 times with -ve sig
    // 	   134784 Flops per site in total
    //
    // call 3 (Lepage)
    // 	: 24 times with +ve sig and 24 times with -ve sig
    //	28944 Flops per site in total
    //
    template <typename real, int sig_positive, int mu_positive, bool pMu, bool qMu, bool qPrev, typename Arg>
    __global__ void middleLinkKernel(Arg arg)
    {
      typedef Matrix<complex<real>,3> Link;

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
        Link oprod = arg.newOprod(arg.sig, e_cb, parity);
        oprod += arg.coeff*Oy;
        arg.newOprod(arg.sig, e_cb, parity) = oprod;
      }

    }

    template <typename real, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct SideLinkArg : public BaseForceArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      typedef typename gauge_mapper<real,reconstruct>::type G;
      F newOprod;
      F shortP;
      const F p3;
      const F qProd;
      const G link;
      const real coeff;
      const real accumu_coeff;

      SideLinkArg(GaugeField &newOprod, GaugeField &shortP, const GaugeField &P3,
                  const GaugeField &qProd, const GaugeField &link, real coeff, real accumu_coeff, int overlap)
        : BaseForceArg(link, overlap), newOprod(newOprod), shortP(shortP), p3(P3), qProd(qProd), link(link),
          coeff(coeff), accumu_coeff(accumu_coeff)
      {
        if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
        if (!shortP.isNative()) errorQuda("Unsupported gauge order %d", shortP.Order());
        if (!P3.isNative()) errorQuda("Unsupported gauge order %d", P3.Order());
        if (!qProd.isNative()) errorQuda("Unsupported gauge order %d", qProd.Order());
        if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      }

    };

    template <typename real>
    struct SideLinkShortArg : public BaseForceArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F newOprod;
      const F p3;
      const real coeff;

      SideLinkShortArg(GaugeField &newOprod, GaugeField &P3, const GaugeField &link,
                       real coeff, int overlap)
        : BaseForceArg(link, overlap), newOprod(newOprod), p3(P3), coeff(coeff)
      {
        if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
        if (!P3.isNative()) errorQuda("Unsupported gauge order %d", P3.Order());
      }

    };

    /***********************************do_side_link_kernel***************************
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

    // Flop count, in two-number pair (matrix_mult, matrix_add)
    // 		(2,2)
    // call 1: 432 Flops per site
    // call 2 (short)
    // 	: 18 Flops per site
    //
    // call 1: 240 calls
    // call 2: 48 calls
    //
    // Aggregate Flops:
    // call 1: 103680
    // call 2: 864

    template <typename real, int sig_positive, int mu_positive, typename Arg>
    __global__ void sideLinkKernel(Arg arg)
    {
      typedef Matrix<complex<real>, 3> Link;
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

        Link shortP = arg.shortP(0, point_d, 1-parity);
        shortP += arg.accumu_coeff * Ow;
        arg.shortP(0, point_d, 1-parity) = shortP;
      }

      {
        Link Ox = arg.qProd(0, point_d, 1-parity);
        Link Ow = mu_positive ? Oy*Ox : conj(Ox)*conj(Oy);

        real mycoeff = CoeffSign(sig_positive, parity)*arg.coeff;
        if ( (mu_positive && !parity) || (!mu_positive && parity) ) mycoeff = -mycoeff;

        Link oprod = arg.newOprod(mu_positive ? arg.mu : opp_dir(arg.mu), mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity);
        oprod += mycoeff * Ow;
        arg.newOprod(mu_positive ? arg.mu : opp_dir(arg.mu), mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity) = oprod;
      }
    }

    // Flop count, in two-number pair (matrix_mult, matrix_add)
    // 		(0,1)
    template<typename real, int sig_positive, int mu_positive, typename Arg>
    __global__ void sideLinkShortKernel(Arg arg)
    {
      typedef Matrix<complex<real>,3> Link;
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
      int point_d = linkIndex(y,arg.E);

      Link Oy = arg.p3(0, e_cb, parity);

      real mycoeff = CoeffSign(sig_positive,parity)*arg.coeff;

      if (mu_positive) {
        if (!parity) { mycoeff = -mycoeff;} // need to change this to get away from parity
        Link oprod = arg.newOprod(arg.mu, point_d, 1-parity);
        oprod += mycoeff * Oy;
        arg.newOprod(arg.mu, point_d, 1-parity) = oprod;
      } else {
        if (parity) mycoeff = -mycoeff;

        Link oprod = arg.newOprod(opp_dir(arg.mu), e_cb, parity);
        oprod += mycoeff * conj(Oy);
        arg.newOprod(opp_dir(arg.mu), e_cb, parity) = oprod;
      }
    }


    template <typename real, typename Arg>
    class HisqForce : public TunableVectorY {

    private:
      Arg &arg;
      const GaugeField &meta;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      HisqForce(Arg &arg, const GaugeField &meta, HisqForceType type)
        : TunableVectorY(2), arg(arg), meta(meta), type(type) { }
      virtual ~HisqForce() { }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << "threads=" << arg.threads << ",prec=" << sizeof(real);
        switch (type) {
        case FORCE_LONG_LINK: aux << ",LONG_LINK"; break; // FIXME presently uses extended fields only so tuneKey is messed up
        case FORCE_COMPLETE:  aux << ",COMPLETE";  break;
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        switch (type) {
        case FORCE_LONG_LINK:
          longLinkKernel<real,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        case FORCE_COMPLETE:
          completeForceKernel<real,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_LONG_LINK: arg.force.save(); break;
        case FORCE_COMPLETE:
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_LONG_LINK: arg.force.load(); break;
        case FORCE_COMPLETE:
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_LONG_LINK: return 2*arg.threads*4968ll;
        case FORCE_COMPLETE: return 2*arg.threads*792ll;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_LONG_LINK: return 4*2*arg.threads*(2*arg.force.Bytes() + 4*arg.link.Bytes() + 3*arg.oprod.Bytes());
        case FORCE_COMPLETE: return 4*2*arg.threads*(arg.force.Bytes() + arg.link.Bytes() + arg.oprod.Bytes());
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, typename Arg>
    class AllForce : public TunableVectorY {

    private:
      Arg &arg;
      const GaugeField &meta;
      int mu;
      int sig;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      AllForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorY(2), arg(arg), meta(meta), sig(sig), mu(mu), type(type) { }
      virtual ~AllForce() { }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << "threads=" << arg.threads << ",sig=" << sig << ",mu=" << mu;
        switch (type) {
        case FORCE_ALL_LINK: aux << ",ALL_LINK"; break; // FIXME presently uses extended fields only so tuneKey is messed up
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        arg.mu = mu;
        arg.sig = sig;
        switch (type) {
        case FORCE_ALL_LINK:
          if (goes_forward(sig) && goes_forward(mu))
            allLinkKernel<real,1,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_forward(sig) && goes_backward(mu))
            allLinkKernel<real,1,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_backward(sig) && goes_forward(mu))
            allLinkKernel<real,0,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else
            allLinkKernel<real,0,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_ALL_LINK:
          arg.force.save();
          arg.shortP.save();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_ALL_LINK:
          arg.force.load();
          arg.shortP.load();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_ALL_LINK:
          return 2*arg.threads*(goes_forward(sig) ? 1242ll : 828ll);
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_ALL_LINK:
          return 2*arg.threads*( (goes_forward(sig) ? 4 : 2)*arg.force.Bytes() + 3*arg.link.Bytes()
                                 + arg.oprod.Bytes() + arg.qPrev.Bytes() + 2*arg.shortP.Bytes());
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, typename Arg>
    class MiddleLinkForce : public TunableVectorY {

    private:
      Arg &arg;
      const GaugeField &meta;
      int mu;
      int sig;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      MiddleLinkForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorY(2), arg(arg), meta(meta), sig(sig), mu(mu), type(type) { }
      virtual ~MiddleLinkForce() { }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << "threads=" << arg.threads << ",sig=" << sig << ",mu=" << mu <<
          ",pMu=" << arg.p_mu << ",q_muu=" << arg.q_mu << ",q_prev=" << arg.q_prev;
        switch (type) {
        case FORCE_MIDDLE_LINK: aux << ",MIDDLE_LINK"; break; // FIXME presently uses extended fields only so tuneKey is messed up
        case FORCE_LEPAGE_MIDDLE_LINK: aux << ",LEPAGE_MIDDLE_LINK"; break; // FIXME presently uses extended fields only so tuneKey is messed up
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        arg.mu = mu;
        arg.sig = sig;
        switch (type) {
        case FORCE_MIDDLE_LINK:
          if (!arg.p_mu || !arg.q_mu) errorQuda("Expect p_mu=%d and q_mu=%d to both be true", arg.p_mu, arg.q_mu);
          if (arg.q_prev) {
            if (goes_forward(sig) && goes_forward(mu))
              middleLinkKernel<real,1,1,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_forward(sig) && goes_backward(mu))
              middleLinkKernel<real,1,0,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_backward(sig) && goes_forward(mu))
              middleLinkKernel<real,0,1,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else
              middleLinkKernel<real,0,0,true,true,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          } else {
            if (goes_forward(sig) && goes_forward(mu))
              middleLinkKernel<real,1,1,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_forward(sig) && goes_backward(mu))
              middleLinkKernel<real,1,0,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else if (goes_backward(sig) && goes_forward(mu))
              middleLinkKernel<real,0,1,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
            else
              middleLinkKernel<real,0,0,true,true,false,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          }
          break;
        case FORCE_LEPAGE_MIDDLE_LINK:
          if (arg.p_mu || arg.q_mu || !arg.q_prev)
            errorQuda("Expect p_mu=%d and q_mu=%d to both be false and q_prev=%d true", arg.p_mu, arg.q_mu, arg.q_prev);
          if (goes_forward(sig) && goes_forward(mu))
            middleLinkKernel<real,1,1,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_forward(sig) && goes_backward(mu))
            middleLinkKernel<real,1,0,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_backward(sig) && goes_forward(mu))
            middleLinkKernel<real,0,1,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else
            middleLinkKernel<real,0,0,false,false,true,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        default:
            errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_MIDDLE_LINK:
          arg.pMu.save();
          arg.qMu.save();
        case FORCE_LEPAGE_MIDDLE_LINK:
          arg.newOprod.save();
          arg.p3.save();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_MIDDLE_LINK:
          arg.pMu.load();
          arg.qMu.load();
        case FORCE_LEPAGE_MIDDLE_LINK:
          arg.newOprod.load();
          arg.p3.load();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads*(2 * 198 +
                                (!arg.q_prev && goes_forward(sig) ? 198 : 0) +
                                (arg.q_prev && (arg.q_mu || goes_forward(sig) ) ? 198 : 0) +
                                ((arg.q_prev && goes_forward(sig) ) ?  198 : 0) +
                                ( goes_forward(sig) ? 216 : 0) );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads*( ( goes_forward(sig) ? 2*arg.newOprod.Bytes() : 0 ) +
                                 (arg.p_mu ? arg.pMu.Bytes() : 0) +
                                 (arg.q_mu ? arg.qMu.Bytes() : 0) +
                                 ( ( goes_forward(sig) || arg.q_mu ) ? arg.qPrev.Bytes() : 0) +
                                 arg.p3.Bytes() + 3*arg.link.Bytes() + arg.oProd.Bytes() );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };


    template <typename real, typename Arg>
    class SideLinkForce : public TunableVectorY {

    private:
      Arg &arg;
      const GaugeField &meta;
      int mu;
      int sig;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      SideLinkForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorY(2), arg(arg), meta(meta), sig(sig), mu(mu), type(type) { }
      virtual ~SideLinkForce() { }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << "threads=" << arg.threads << ",sig=" << sig << ",mu=" << mu;
        switch (type) {
        case FORCE_SIDE_LINK: aux << ",SIDE_LINK"; break; // FIXME presently uses extended fields only so tuneKey is messed up
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        arg.mu = mu;
        arg.sig = sig;
        switch (type) {
        case FORCE_SIDE_LINK:
          if (goes_forward(sig) && goes_forward(mu))
            sideLinkKernel<real,1,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_forward(sig) && goes_backward(mu))
            sideLinkKernel<real,1,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_backward(sig) && goes_forward(mu))
            sideLinkKernel<real,0,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else
            sideLinkKernel<real,0,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_SIDE_LINK:
          arg.newOprod.save();
          arg.shortP.save();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_SIDE_LINK:
          arg.newOprod.load();
          arg.shortP.load();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_SIDE_LINK:
          return 2*arg.threads*2*234;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_SIDE_LINK:
          return 2*arg.threads*( 2*arg.newOprod.Bytes() + 2*arg.shortP.Bytes() +
                                 arg.p3.Bytes() + arg.link.Bytes() + arg.qProd.Bytes() );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, typename Arg>
    class SideLinkShortForce : public TunableVectorY {

    private:
      Arg &arg;
      const GaugeField &meta;
      int mu;
      int sig;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      SideLinkShortForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorY(2), arg(arg), meta(meta), sig(sig), mu(mu), type(type) { }
      virtual ~SideLinkShortForce() { }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << "threads=" << arg.threads << ",sig=" << sig << ",mu=" << mu;
        switch (type) {
        case FORCE_SIDE_LINK_SHORT: aux << ",SIDE_LINK_SHORT"; break; // FIXME presently uses extended fields only so tuneKey is messed up
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        arg.mu = mu;
        arg.sig = sig;
        switch (type) {
        case FORCE_SIDE_LINK_SHORT:
          if (goes_forward(sig) && goes_forward(mu))
            sideLinkShortKernel<real,1,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_forward(sig) && goes_backward(mu))
            sideLinkShortKernel<real,1,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else if (goes_backward(sig) && goes_forward(mu))
            sideLinkShortKernel<real,0,1,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          else
            sideLinkShortKernel<real,0,0,Arg><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_SIDE_LINK_SHORT:
          arg.newOprod.save();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_SIDE_LINK_SHORT:
          arg.newOprod.load();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_SIDE_LINK_SHORT:
          return 2*arg.threads*36;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_SIDE_LINK_SHORT:
          return 2*arg.threads*( 2*arg.newOprod.Bytes() + arg.p3.Bytes() );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };


    template <typename real>
    struct OneLinkTermArg : BaseForceArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F force;
      const F oprod;
      const real coeff;

      OneLinkTermArg(GaugeField &force, const GaugeField &oprod, const GaugeField &link, real coeff)
        : BaseForceArg(link, 0), force(force), oprod(oprod), coeff(coeff)
      {
        if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
        if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());\
        if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());\
      }

    };

    template <typename real, typename Arg>
    __global__ void oneLinkTermKernel(Arg arg)
    {
      typedef Matrix<complex<real>,3> Link;
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

      Link w = arg.oprod(sig, e_cb, parity);
      Link force = arg.force(sig, e_cb, parity);
      force += arg.coeff * w;
      arg.force(sig, e_cb, parity) = force;
    }

    template <typename real, typename Arg>
    class OneLinkForce : public TunableVectorYZ {

    private:
      Arg &arg;
      const GaugeField &meta;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      OneLinkForce(Arg &arg, const GaugeField &meta, HisqForceType type)
        : TunableVectorYZ(2,4), arg(arg), meta(meta), type(type) { }
      virtual ~OneLinkForce() { }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << "threads=" << arg.threads;
        switch (type) {
        case FORCE_ONE_LINK: aux << ",ONE_LINK"; break; // FIXME presently uses extended fields only so tuneKey is messed up
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        oneLinkTermKernel<real,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }

      void preTune() {
        switch (type) {
        case FORCE_ONE_LINK: arg.force.save(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_ONE_LINK: arg.force.load(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_ONE_LINK: return 2*4*arg.threads*36ll;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_ONE_LINK: return 2*4*arg.threads*( arg.oprod.Bytes() + 2*arg.force.Bytes() );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template<class real>
    static void
    do_hisq_staples_force_cuda( PathCoefficients<real> act_path_coeff,
                                const QudaGaugeParam& param,
                                const cudaGaugeField &oprod,
                                const cudaGaugeField &link,
                                cudaGaugeField &Pmu,
                                cudaGaugeField &P3,
                                cudaGaugeField &P5,
                                cudaGaugeField &Pnumu,
                                cudaGaugeField &Qmu,
                                cudaGaugeField &Qnumu,
                                cudaGaugeField &newOprod)
      {
        real OneLink = act_path_coeff.one;
        real ThreeSt = act_path_coeff.three;
        real mThreeSt = -ThreeSt;
        real FiveSt  = act_path_coeff.five;
        real mFiveSt  = -FiveSt;
        real SevenSt = act_path_coeff.seven;
        real Lepage  = act_path_coeff.lepage;
        real mLepage  = -Lepage;

        OneLinkTermArg<real> arg(newOprod, oprod, link, OneLink);
        OneLinkForce<real, OneLinkTermArg<real> > oneLink(arg, newOprod, FORCE_ONE_LINK);
        oneLink.apply(0);
        checkCudaError();

        for (int sig=0; sig<8; sig++) {
          for (int mu=0; mu<8; mu++) {
            if ( (mu == sig) || (mu == opp_dir(sig))) continue;

            //3-link
            //Kernel A: middle link
            MiddleLinkArg<real> middleLinkArg( newOprod, Pmu, P3, Qmu, oprod, link, mThreeSt, 2, FORCE_MIDDLE_LINK);
            MiddleLinkForce<real, MiddleLinkArg<real> > middleLink(middleLinkArg, link, sig, mu, FORCE_MIDDLE_LINK);
            middleLink.apply(0);
            checkCudaError();

            for (int nu=0; nu < 8; nu++) {
              if (nu == sig || nu == opp_dir(sig) || nu == mu || nu == opp_dir(mu)) continue;

              //5-link: middle link
              //Kernel B
              MiddleLinkArg<real> middleLinkArg( newOprod, Pnumu, P5, Qnumu, Pmu, Qmu, link, FiveSt, 1, FORCE_MIDDLE_LINK);
              MiddleLinkForce<real, MiddleLinkArg<real> > middleLink(middleLinkArg, link, sig, nu, FORCE_MIDDLE_LINK);
              middleLink.apply(0);
              checkCudaError();

              for (int rho = 0; rho < 8; rho++) {
                if (rho == sig || rho == opp_dir(sig) || rho == mu || rho == opp_dir(mu) || rho == nu || rho == opp_dir(nu)) continue;

                //7-link: middle link and side link
                HisqForceArg<real> arg(newOprod, P5, link, Pnumu, Qnumu, SevenSt, FiveSt != 0 ? SevenSt/FiveSt : 0, 1);
                AllForce<real, HisqForceArg<real> > all(arg, link, sig, rho, FORCE_ALL_LINK);
                all.apply(0);
                checkCudaError();

              }//rho

              //5-link: side link
              SideLinkArg<real> arg(newOprod, P3, P5, Qmu, link, mFiveSt, (ThreeSt != 0 ? FiveSt/ThreeSt : 0), 1);
              SideLinkForce<real, SideLinkArg<real> > side(arg, link, sig, nu, FORCE_SIDE_LINK);
              side.apply(0);
              checkCudaError();

            } //nu

            //lepage
            if (Lepage != 0.) {
              MiddleLinkArg<real> middleLinkArg( newOprod, P5, Pmu, Qmu, link, Lepage, 2, FORCE_LEPAGE_MIDDLE_LINK);
              MiddleLinkForce<real, MiddleLinkArg<real> > middleLink(middleLinkArg, link, sig, mu, FORCE_LEPAGE_MIDDLE_LINK);
              middleLink.apply(0);
              checkCudaError();

              SideLinkArg<real> arg(newOprod, P3, P5, Qmu, link, mLepage, (ThreeSt != 0 ? Lepage/ThreeSt : 0), 2);
              SideLinkForce<real, SideLinkArg<real> > side(arg, link, sig, mu, FORCE_SIDE_LINK);
              side.apply(0);
              checkCudaError();

            } // Lepage != 0.0

            //3-link side link
            SideLinkShortArg<real> arg(newOprod, P3, link, ThreeSt, 1);
            SideLinkShortForce<real, SideLinkShortArg<real> > side(arg, P3, sig, mu, FORCE_SIDE_LINK_SHORT);
            side.apply(0);
            checkCudaError();

          }//mu
        }//sig

        return;
      } // do_hisq_staples_force_cuda

    void hisqCompleteForce(const QudaGaugeParam &param,
                           const cudaGaugeField &oprod,
                           const cudaGaugeField &link,
                           cudaGaugeField* force,
                           long long* flops)
    {
      QudaPrecision precision = checkPrecision(oprod, link, *force);
      if (precision == QUDA_DOUBLE_PRECISION) {
        if (link.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          typedef CompleteForceArg<double,QUDA_RECONSTRUCT_NO> Arg;
          Arg arg(*force, link, oprod);
          HisqForce<double,Arg> completeForce(arg, link, FORCE_COMPLETE);
          completeForce.apply(0);
          if (flops) *flops += completeForce.flops();
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else if (precision == QUDA_SINGLE_PRECISION) {
        if (link.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          typedef CompleteForceArg<float,QUDA_RECONSTRUCT_NO> Arg;
          Arg arg(*force, link, oprod);
          HisqForce<float, Arg> completeForce(arg, link, FORCE_COMPLETE);
          completeForce.apply(0);
          if (flops) *flops += completeForce.flops();
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else {
        errorQuda("Unsupported precision %d", precision);
      }
    }

    void hisqLongLinkForce(double coeff,
                           const QudaGaugeParam &param,
                           const cudaGaugeField &oldOprod,
                           const cudaGaugeField &link,
                           cudaGaugeField  *newOprod,
                           long long* flops)
    {
      QudaPrecision precision = checkPrecision(*newOprod, link, oldOprod);
      if (precision == QUDA_DOUBLE_PRECISION) {
        if (link.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          typedef LongLinkArg<double,QUDA_RECONSTRUCT_NO> Arg;
          Arg arg(*newOprod, link, oldOprod, coeff);
          HisqForce<double,Arg> longLink(arg, link, FORCE_LONG_LINK);
          longLink.apply(0);
          if (flops) (*flops) += longLink.flops();
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else if (precision == QUDA_SINGLE_PRECISION) {
        if (link.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          typedef LongLinkArg<float,QUDA_RECONSTRUCT_NO> Arg;
          Arg arg(*newOprod, link, oldOprod, coeff);
          HisqForce<float, Arg> longLink(arg, link, FORCE_LONG_LINK);
          longLink.apply(0);
          if (flops) (*flops) += longLink.flops();
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else {
        errorQuda("Unsupported precision %d", precision);
      }
    }

    void hisqStaplesForce(const double path_coeff_array[6],
                          const QudaGaugeParam &param,
                          const cudaGaugeField &oprod,
                          const cudaGaugeField &link,
                          cudaGaugeField* newOprod,
                          long long* flops)
      {
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

        if (link.Precision() ==  QUDA_DOUBLE_PRECISION) {
          PathCoefficients<double> act_path_coeff;
          act_path_coeff.one    = path_coeff_array[0];
          act_path_coeff.naik   = path_coeff_array[1];
          act_path_coeff.three  = path_coeff_array[2];
          act_path_coeff.five   = path_coeff_array[3];
          act_path_coeff.seven  = path_coeff_array[4];
          act_path_coeff.lepage = path_coeff_array[5];

          do_hisq_staples_force_cuda<double>( act_path_coeff, param, oprod, link, Pmu, P3, P5, Pnumu, Qmu, Qnumu, *newOprod);
        } else if (link.Precision() == QUDA_SINGLE_PRECISION) {
          PathCoefficients<float> act_path_coeff;
          act_path_coeff.one    = path_coeff_array[0];
          act_path_coeff.naik   = path_coeff_array[1];
          act_path_coeff.three  = path_coeff_array[2];
          act_path_coeff.five   = path_coeff_array[3];
          act_path_coeff.seven  = path_coeff_array[4];
          act_path_coeff.lepage = path_coeff_array[5];

          do_hisq_staples_force_cuda<float>( act_path_coeff, param, oprod, link, Pmu, P3, P5, Pnumu, Qmu, Qnumu, *newOprod);
        } else {
          errorQuda("Unsupported precision");
        }

	if (flops) {
	  int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
	  // Middle Link, side link, short side link, AllLink, OneLink
	  *flops += (long long)volume*(134784 + 24192 + 103680 + 864 + 397440 + 72 + (path_coeff_array[5] != 0 ? 28944 : 0));
	}

      }

  } // namespace fermion_force
} // namespace quda

#endif // GPU_HISQ_FORCE

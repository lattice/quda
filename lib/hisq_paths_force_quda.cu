#include <quda_internal.h>
#include <lattice_field.h>
#include <gauge_field.h>
#include <ks_improved_force.h>
#include <utility>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <color_spinor_field.h>
#include <index_helper.cuh>
#include <gauge_field_order.h>

#ifdef GPU_HISQ_FORCE

#define REWRITE

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

    __host__ __device__ constexpr int opp_dir(int dir) { return 7-dir; }
    __host__ __device__ constexpr int goes_forward(int dir) { return dir<=3; }
    __host__ __device__ constexpr int goes_backward(int dir) { return dir>3; }

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

    __device__ __host__ inline int CoeffSign(int pos_dir, int odd_lattice) {
      return 2*((pos_dir + odd_lattice + 1) & 1) - 1;
    }

    __device__ __host__ inline int Sign(int parity) {
      return parity ? -1 : 1;
    }

    __device__ __host__ inline int posDir(int dir){
      return (dir >= 4) ? 7-dir : dir;
    }

    template <int dir>
    inline __device__ __host__ void updateCoords(int x[], int shift, const int X[4], const int partitioned[]){
#ifdef MULTI_GPU
      if (shift == 1) {
        x[dir] = (partitioned[dir] || (x[dir] != X[dir]+1)) ? x[dir]+1 : 2;
      } else if (shift == -1) {
        x[dir] = (partitioned[dir] || (x[dir] != 2)) ? x[dir]-1 : X[dir]+1;
      }
#else
      x[dir] = (x[dir]+shift + X[dir])%X[dir];
#endif
    }

    inline __device__ __host__ void updateCoords(int x[], int dir, int shift, const int X[4], const int partitioned[]) {
      switch (dir) {
        case 0:
	  updateCoords<0>(x, shift, X, partitioned);
	  break;
        case 1:
	  updateCoords<1>(x, shift, X, partitioned);
	  break;
        case 2:
	  updateCoords<2>(x, shift, X, partitioned);
	  break;
        case 3:
	  updateCoords<3>(x, shift, X, partitioned);
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

#ifdef REWRITE

    template <typename real, QudaReconstructType reconstruct>
    struct CompleteForceArg {

      typedef typename gauge::FloatNOrder<real,18,2,11> M;
      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      typedef typename gauge_mapper<real,reconstruct>::type G;

      M force;
      const F oprod;
      const G link;

      const int threads;
      int X[4];
      int E[4];
      int border[4];

      const real coeff;

      CompleteForceArg(GaugeField &force, const GaugeField &link, const GaugeField &oprod)
        : force(force), link(link), oprod(oprod), threads(force.VolumeCB()), coeff(0.0)
      {
        if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
        if (!link.isNative())  errorQuda("Unsupported gauge order %d", link.Order());
        if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());

        for (int d=0; d<4; d++) {
          X[d] = force.X()[d]; // force field is reguar field
          E[d] = link.X()[d]; // link field is extended
          border[d] = (E[d] - X[d]) / 2;
        }
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
    struct LongLinkArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      typedef typename gauge_mapper<real,reconstruct>::type G;

      F force;
      const F oprod;
      const G link;

      int threads;
      int X[4];
      int E[4];
      int border[4];

      const real coeff;

      LongLinkArg(GaugeField &force, const GaugeField &link, const GaugeField &oprod, real coeff)
        : force(force), link(link), oprod(oprod), threads(1), coeff(coeff)
      {
        if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
        if (!link.isNative())  errorQuda("Unsupported gauge order %d", link.Order());
        if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());

        for (int d=0; d<4; d++) {
          E[d] = link.X()[d]; // link field is extended
#ifdef MULTI_GPU
          border[d] = 2;//link.R()[d];
#else
          border[d] = 0;
#endif
          X[d] = E[d] - 2*border[d];
          threads *= X[d];
        }
        threads /= 2;
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

      BaseForceArg(const GaugeField &meta, int base_offset) : threads(1),
        commDim{ comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3) }
      {
        for (int d=0; d<4; d++) {
          E[d] = meta.X()[d]; // link field is extended
#ifdef MULTI_GPU
          border[d] = 2;//meta.R()[d]; FIXME
#else
          border[d] = 0;
#endif
          X[d] = E[d] - 2*border[d];
          D[d] = comm_dim_partitioned(d) ? X[d]+(base_offset+1)*border[d] : X[d];
          base_idx[d] = comm_dim_partitioned(d) ? base_offset : border[d];
          threads *= D[d];
        }
        threads /= 2;
        oddness_change = (base_idx[0] + base_idx[1] + base_idx[2] + base_idx[3])&1;
      }
    };

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
                   const GaugeField &qPrev, real coeff, real accumu_coeff, int base_offset)
        : BaseForceArg(link, base_offset), force(force), shortP(shortP), link(link), oprod(oprod), qPrev(qPrev),
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
      updateCoords(y, mysig, (sig_positive ? 1 : -1), arg.X, arg.commDim);
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
        updateCoords(y, arg.mu, -1, arg.X, arg.commDim);
        int point_d = linkIndex(y,arg.E);
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg.X, arg.commDim);
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
        updateCoords(y, mu, 1, arg.X, arg.commDim);
        int point_d = linkIndex(y,arg.E);
        updateCoords(y, mysig, (sig_positive ? 1 : -1), arg.X, arg.commDim);
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
                    real coeff, int base_offset, HisqForceType type)
        : BaseForceArg(link, base_offset), newOprod(newOprod), pMu(pMu), p3(P3), qMu(qMu),
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
                    real coeff, int base_offset, HisqForceType type)
        : BaseForceArg(link, base_offset), newOprod(newOprod), pMu(pMu), p3(P3), qMu(qMu),
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
                    real coeff, int base_offset, HisqForceType type)
        : BaseForceArg(link, base_offset), newOprod(newOprod), pMu(P3), p3(P3), qMu(qPrev),
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
      updateCoords(y, mymu, (mu_positive ? -1 : 1), arg.X, arg.commDim);

      int point_d = linkIndex(y, arg.E);
      int ad_link_nbr_idx = mu_positive ? point_d : e_cb;

      int mysig = posDir(arg.sig);
      updateCoords(y, mysig, (sig_positive ? 1 : -1), arg.X, arg.commDim);
      int point_c = linkIndex(y, arg.E);

      for (int d=0; d<4; d++) y[d] = x[d];
      updateCoords(y, mysig, (sig_positive ? 1 : -1), arg.X, arg.commDim);
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
                  const GaugeField &qProd, const GaugeField &link, real coeff, real accumu_coeff, int base_offset)
        : BaseForceArg(link, base_offset), newOprod(newOprod), shortP(shortP), p3(P3), qProd(qProd), link(link),
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
                       real coeff, int base_offset)
        : BaseForceArg(link, base_offset), newOprod(newOprod), p3(P3), coeff(coeff)
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

      int y[4] = {x[0], x[1], x[2], x[3]};

      int mymu = posDir(arg.mu);
      updateCoords(y, mymu, (mu_positive ? -1 : 1), arg.X, arg.commDim);
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

      updateCoords(y, mymu, (mu_positive ? -1 : 1), arg.X, arg.commDim);
      int point_d = linkIndex(y,arg.E);
      real mycoeff = CoeffSign(sig_positive,parity)*arg.coeff;

      Link Oy = arg.p3(0, e_cb, parity);
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
    struct OneLinkTermArg {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F force;
      const F oprod;
      int threads;
      int X[4]; // regular grid dims
      int E[4]; // extended grid dims
      int border[4];
      const real coeff;

      OneLinkTermArg(GaugeField &force, const GaugeField &oprod, real coeff)
        : force(force), oprod(oprod), threads(1), coeff(coeff)
      {
        if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
        if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());\
        for (int d=0; d<4; d++) {
          E[d] = oprod.X()[d]; // link field is extended
#ifdef MULTI_GPU
          border[d] = 2;//P3.R()[d]; // FIXME need to make sure that R is set so we can extract X
#else
          border[d] = 0;
#endif
          X[d] = E[d] - 2*border[d];
          threads *= X[d];
        }
        threads /= 2;
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

    template<class Real, class RealA, class RealB>
      static void
      do_hisq_staples_force_cuda( PathCoefficients<Real> act_path_coeff,
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
        Real coeff;
        Real OneLink, Lepage, FiveSt, ThreeSt, SevenSt;
        Real mLepage, mFiveSt, mThreeSt;

        OneLink = act_path_coeff.one;
        ThreeSt = act_path_coeff.three; mThreeSt = -ThreeSt;
        FiveSt  = act_path_coeff.five; mFiveSt  = -FiveSt;
        SevenSt = act_path_coeff.seven;
        Lepage  = act_path_coeff.lepage; mLepage  = -Lepage;

        OneLinkTermArg<Real> arg(newOprod, oprod, OneLink);
        OneLinkForce<Real, OneLinkTermArg<Real> > oneLink(arg, newOprod, FORCE_ONE_LINK);
        oneLink.apply(0);
        checkCudaError();

        for (int sig=0; sig<8; sig++) {
          for (int mu=0; mu<8; mu++) {
            if ( (mu == sig) || (mu == opp_dir(sig))) {
              continue;
            }
            //3-link
            //Kernel A: middle link

            MiddleLinkArg<Real> middleLinkArg( newOprod, Pmu, P3, Qmu, oprod, link, mThreeSt, 1, FORCE_MIDDLE_LINK);
            MiddleLinkForce<Real, MiddleLinkArg<Real> > middleLink(middleLinkArg, link, sig, mu, FORCE_MIDDLE_LINK);
            middleLink.apply(0);
            checkCudaError();

            for(int nu=0; nu < 8; nu++){
              if (nu == sig || nu == opp_dir(sig)
                  || nu == mu || nu == opp_dir(mu)){
                continue;
              }
              //5-link: middle link
              //Kernel B
              MiddleLinkArg<Real> middleLinkArg( newOprod, Pnumu, P5, Qnumu,
                                                 Pmu, Qmu, link, FiveSt, 0, FORCE_MIDDLE_LINK);
              MiddleLinkForce<Real, MiddleLinkArg<Real> > middleLink(middleLinkArg, link, sig, nu, FORCE_MIDDLE_LINK);
              middleLink.apply(0);
              checkCudaError();

              for(int rho = 0; rho < 8; rho++){
                if (rho == sig || rho == opp_dir(sig)
                    || rho == mu || rho == opp_dir(mu)
                    || rho == nu || rho == opp_dir(nu)){
                  continue;
                }

                //7-link: middle link and side link
                if(FiveSt != 0)coeff = SevenSt/FiveSt; else coeff = 0;

                HisqForceArg<Real> arg(newOprod, P5, link, Pnumu, Qnumu, SevenSt, coeff, 0);
                AllForce<Real, HisqForceArg<Real> > all(arg, link, sig, rho, FORCE_ALL_LINK);
                all.apply(0);
                checkCudaError();
              }//rho

              //5-link: side link
              if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;

              SideLinkArg<Real> arg(newOprod, P3, P5, Qmu, link, mFiveSt, coeff, 0);
              SideLinkForce<Real, SideLinkArg<Real> > side(arg, link, sig, nu, FORCE_SIDE_LINK);
              side.apply(0);
              checkCudaError();

            } //nu

            //lepage
            if(Lepage != 0.){
              MiddleLinkArg<Real> middleLinkArg( newOprod, P5,
                                                 Pmu, Qmu, link, Lepage, 1, FORCE_LEPAGE_MIDDLE_LINK);
              MiddleLinkForce<Real, MiddleLinkArg<Real> > middleLink(middleLinkArg, link, sig, mu, FORCE_LEPAGE_MIDDLE_LINK);
              middleLink.apply(0);
              checkCudaError();

              if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;

              SideLinkArg<Real> arg(newOprod, P3, P5, Qmu, link, mLepage, coeff, 1);
              SideLinkForce<Real, SideLinkArg<Real> > side(arg, link, sig, mu, FORCE_SIDE_LINK);
              side.apply(0);
              checkCudaError();

            } // Lepage != 0.0

            //3-link side link
            SideLinkShortArg<Real> arg(newOprod, P3, link, ThreeSt, 0);
            SideLinkShortForce<Real, SideLinkShortArg<Real> > side(arg, P3, sig, mu, FORCE_SIDE_LINK_SHORT);
            side.apply(0);
            checkCudaError();

          }//mu
        }//sig

        return;
      } // do_hisq_staples_force_cuda

    void hisqCompleteForceCuda(const QudaGaugeParam &param,
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
          HisqForce<double,Arg> complete(arg, link, FORCE_COMPLETE);
          complete.apply(0);
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else if (precision == QUDA_SINGLE_PRECISION) {
        if (link.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          typedef CompleteForceArg<float,QUDA_RECONSTRUCT_NO> Arg;
          Arg arg(*force, link, oprod);
          HisqForce<float, Arg> complete(arg, link, FORCE_COMPLETE);
          complete.apply(0);
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else {
        errorQuda("Unsupported precision %d", precision);
      }
    }

    void hisqLongLinkForceCuda(double coeff,
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
          HisqForce<double,Arg> complete(arg, link, FORCE_LONG_LINK);
          complete.apply(0);
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else if (precision == QUDA_SINGLE_PRECISION) {
        if (link.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          typedef LongLinkArg<float,QUDA_RECONSTRUCT_NO> Arg;
          Arg arg(*newOprod, link, oldOprod, coeff);
          HisqForce<float, Arg> complete(arg, link, FORCE_LONG_LINK);
          complete.apply(0);
        } else {
          errorQuda("Reconstruct %d not supported", link.Reconstruct());
        }
      } else {
        errorQuda("Unsupported precision %d", precision);
      }
    }

    void hisqStaplesForceCuda(const double path_coeff_array[6],
                              const QudaGaugeParam &param,
                              const cudaGaugeField &oprod,
                              const cudaGaugeField &link,
                              cudaGaugeField* newOprod,
                              long long* flops)
      {
#ifdef MULTI_GPU
        int X[4] = {param.X[0]+4, param.X[1]+4, param.X[2]+4, param.X[3]+4};
#else
        int X[4] = {param.X[0], param.X[1], param.X[2], param.X[3]};
#endif

        // create color matrix fields with zero padding
        GaugeFieldParam gauge_param(X, param.cuda_prec, QUDA_RECONSTRUCT_NO, 0, QUDA_SCALAR_GEOMETRY);

        gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
        gauge_param.siteSubset = QUDA_FULL_SITE_SUBSET;
        gauge_param.order = QUDA_FLOAT2_GAUGE_ORDER;
        cudaGaugeField Pmu(gauge_param);
        cudaGaugeField P3(gauge_param);
        cudaGaugeField P5(gauge_param);
        cudaGaugeField Pnumu(gauge_param);
        cudaGaugeField Qmu(gauge_param);
        cudaGaugeField Qnumu(gauge_param);

        cudaEvent_t start, end;

        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        if (param.cuda_prec == QUDA_DOUBLE_PRECISION){

          PathCoefficients<double> act_path_coeff;
          act_path_coeff.one    = path_coeff_array[0];
          act_path_coeff.naik   = path_coeff_array[1];
          act_path_coeff.three  = path_coeff_array[2];
          act_path_coeff.five   = path_coeff_array[3];
          act_path_coeff.seven  = path_coeff_array[4];
          act_path_coeff.lepage = path_coeff_array[5];
          do_hisq_staples_force_cuda<double,double2,double2>( act_path_coeff,
              param,
              oprod,
              link,
              Pmu,
              P3,
              P5,
              Pnumu,
              Qmu,
              Qnumu,
              *newOprod);


        }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
          PathCoefficients<float> act_path_coeff;
          act_path_coeff.one    = path_coeff_array[0];
          act_path_coeff.naik   = path_coeff_array[1];
          act_path_coeff.three  = path_coeff_array[2];
          act_path_coeff.five   = path_coeff_array[3];
          act_path_coeff.seven  = path_coeff_array[4];
          act_path_coeff.lepage = path_coeff_array[5];

          do_hisq_staples_force_cuda<float,float2,float2>( act_path_coeff,
              param,
              oprod,
              link,
              Pmu,
              P3,
              P5,
              Pnumu,
              Qmu,
              Qnumu,
              *newOprod);
        }else{
          errorQuda("Unsupported precision");
        }

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float runtime;
        cudaEventElapsedTime(&runtime, start, end);

	if(flops){
	  int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
	  // Middle Link, side link, short side link, AllLink, OneLink
	  *flops = (134784 + 24192 + 103680 + 864 + 397440 + 72);

	  if(path_coeff_array[5] != 0.) *flops += 28944; // Lepage contribution
	  *flops *= volume;
	}

        cudaEventDestroy(start);
        cudaEventDestroy(end);
      }

#else // REWRITE

  } // namespace fermion_force

} // namespace quda

#include <read_gauge.h>
#include <hisq_force_macros.h>
#include <force_common.h>

//DEBUG : control compile
#define COMPILE_HISQ_DP_18
#define COMPILE_HISQ_DP_12
#define COMPILE_HISQ_SP_18
#define COMPILE_HISQ_SP_12

// Disable texture read for now. Need to revisit this.
#define HISQ_SITE_MATRIX_LOAD_TEX 1
#define HISQ_NEW_OPROD_LOAD_TEX 1

#ifdef USE_TEXTURE_OBJECTS
#define TEX1DFETCH(type, tex, idx) tex1Dfetch<type>((tex), idx)
#else
#define TEX1DFETCH(type, tex, idx) tex1Dfetch((tex), idx)
#endif


template<typename Tex>
static __inline__ __device__ double fetch_double(Tex t, int i)
{
  int2 v = TEX1DFETCH(int2, t, i);
  return __hiloint2double(v.y, v.x);
}

template <typename Tex>
static __inline__ __device__ double2 fetch_double2(Tex t, int i)
{
  int4 v = TEX1DFETCH(int4, t, i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

static __inline__ __device__ double2 fetch_double2_old(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}



namespace quda {
  namespace fermion_force {

    struct hisq_kernel_param_t{
      unsigned long threads;
      int X[4];
      int D[4];
      int D1h;
      int base_idx[4];
      int ghostDim[4];
      int color_matrix_stride;
      int thin_link_stride;
      int momentum_stride;

      mutable int oddness_change;

      void setStride(const QudaGaugeParam& param){
        int half_volume = param.X[0]*param.X[1]*param.X[2]*param.X[3]/2;
#ifdef MULTI_GPU
        int extended_half_volume = (param.X[0]+4)*(param.X[1]+4)*(param.X[2]+4)*(param.X[3]+4)/2;
        thin_link_stride = extended_half_volume + param.site_ga_pad;
        color_matrix_stride = extended_half_volume;
#else
        thin_link_stride  = half_volume + param.site_ga_pad;
        color_matrix_stride = half_volume;
#endif
        momentum_stride = half_volume + param.mom_ga_pad;
      }
    };


    //Double precision for site link
    texture<int4, 1> thinLink0TexDouble;
    texture<int4, 1> thinLink1TexDouble;

    //Single precision for site link
    texture<float2, 1, cudaReadModeElementType> thinLink0TexSingle;
    texture<float2, 1, cudaReadModeElementType> thinLink1TexSingle;

    texture<float4, 1, cudaReadModeElementType> thinLink0TexSingle_recon;
    texture<float4, 1, cudaReadModeElementType> thinLink1TexSingle_recon;


    texture<int4, 1> newOprod0TexDouble;
    texture<int4, 1> newOprod1TexDouble;
    texture<float2, 1, cudaReadModeElementType>  newOprod0TexSingle;
    texture<float2, 1, cudaReadModeElementType> newOprod1TexSingle;

    inline __device__ float2 operator*(float a, const float2 & b)
    {
      return make_float2(a*b.x,a*b.y);
    }

    inline __device__ double2 operator*(double a, const double2 & b)
    {
      return make_double2(a*b.x,a*b.y);
    }

    // Replication of code
    // This structure is already defined in
    // unitarize_utilities.h

    template<class T>
      struct RealTypeId;

    template<>
      struct RealTypeId<float2>
      {
        typedef float Type;
      };

    template<>
      struct RealTypeId<double2>
      {
        typedef double Type;
      };


    template<class T>
      inline __device__
      void adjointMatrix(T* mat)
      {
#define CONJ_INDEX(i,j) j*3 + i

        T tmp;
        mat[CONJ_INDEX(0,0)] = Conj(mat[0]);
        mat[CONJ_INDEX(1,1)] = Conj(mat[4]);
        mat[CONJ_INDEX(2,2)] = Conj(mat[8]);
        tmp  = Conj(mat[1]);
        mat[CONJ_INDEX(1,0)] = Conj(mat[3]);
        mat[CONJ_INDEX(0,1)] = tmp;
        tmp = Conj(mat[2]);
        mat[CONJ_INDEX(2,0)] = Conj(mat[6]);
        mat[CONJ_INDEX(0,2)] = tmp;
        tmp = Conj(mat[5]);
        mat[CONJ_INDEX(2,1)] = Conj(mat[7]);
        mat[CONJ_INDEX(1,2)] = tmp;

#undef CONJ_INDEX
        return;
      }


    template<int N, class T, class U>
      inline __device__
      void loadMatrixFromField(const T* const field_even, const T* const field_odd,
          int dir, int idx, U* const mat, int oddness, int stride)
      {
        const T* const field = (oddness)?field_odd:field_even;
        for(int i = 0; i < N ;i++){
          mat[i] = field[idx + dir*N*stride + i*stride];
        }
        return;
      }

    template<class T, class U>
      inline __device__
      void loadMatrixFromField(const T* const field_even, const T* const field_odd,
			       int dir, int idx, complex<U>* const mat, int oddness, int stride)
      {
        loadMatrixFromField<9> (field_even, field_odd, dir, idx, mat, oddness, stride);
        return;
      }

    inline __device__
      void loadMatrixFromField(const float4* const field_even, const float4* const field_odd,
			       int dir, int idx, complex<float>* const mat, int oddness, int stride)
      {
        const float4* const field = oddness?field_odd: field_even;
        float4 tmp;
        tmp = field[idx + dir*stride*3];
        mat[0] = make_float2(tmp.x, tmp.y);
        mat[1] = make_float2(tmp.z, tmp.w);
        tmp = field[idx + dir*stride*3 + stride];
        mat[2] = make_float2(tmp.x, tmp.y);
        mat[3] = make_float2(tmp.z, tmp.w);
        tmp = field[idx + dir*stride*3 + 2*stride];
        mat[4] = make_float2(tmp.x, tmp.y);
        mat[5] = make_float2(tmp.z, tmp.w);
        return;
      }

    template<class T, class U>
      inline __device__
      void loadMatrixFromField(const T* const field_even, const T* const field_odd, int idx, U* const mat, int oddness, int stride)
      {
        const T* const field = (oddness)?field_odd:field_even;
        mat[0] = field[idx];
        mat[1] = field[idx + stride];
        mat[2] = field[idx + stride*2];
        mat[3] = field[idx + stride*3];
        mat[4] = field[idx + stride*4];
        mat[5] = field[idx + stride*5];
        mat[6] = field[idx + stride*6];
        mat[7] = field[idx + stride*7];
        mat[8] = field[idx + stride*8];

        return;
      }

    template<class U>
    inline __device__
    void  addMatrixToNewOprod(const complex<double>* const mat,  int dir, int idx, U coeff,
			      double2* const field_even, double2* const field_odd, int oddness, int stride){
      double2* const field = (oddness)?field_odd: field_even;
      double2 value[9];

#if (HISQ_NEW_OPROD_LOAD_TEX == 1)
        value[0] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9);
        value[1] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + stride);
        value[2] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + 2*stride);
        value[3] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + 3*stride);
        value[4] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + 4*stride);
        value[5] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + 5*stride);
        value[6] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + 6*stride);
        value[7] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + 7*stride);
        value[8] = READ_DOUBLE2_TEXTURE( ((oddness)?newOprod1TexDouble:newOprod0TexDouble), field, idx+dir*stride*9 + 8*stride);
#else
        for(int i=0; i<9; ++i) value[i] = field[i];
#endif

        field[idx + dir*stride*9]              = value[0] + coeff*mat[0];
        field[idx + dir*stride*9 + stride]     = value[1] + coeff*mat[1];
        field[idx + dir*stride*9 + stride*2]   = value[2] + coeff*mat[2];
        field[idx + dir*stride*9 + stride*3]   = value[3] + coeff*mat[3];
        field[idx + dir*stride*9 + stride*4]   = value[4] + coeff*mat[4];
        field[idx + dir*stride*9 + stride*5]   = value[5] + coeff*mat[5];
        field[idx + dir*stride*9 + stride*6]   = value[6] + coeff*mat[6];
        field[idx + dir*stride*9 + stride*7]   = value[7] + coeff*mat[7];
        field[idx + dir*stride*9 + stride*8]   = value[8] + coeff*mat[8];

        return;
      }


    template<class U>
    inline __device__
    void  addMatrixToNewOprod(const complex<float>* const mat,  int dir, int idx, U coeff,
			      float2* const field_even, float2* const field_odd, int oddness, int stride){
      float2* const field = (oddness)?field_odd: field_even;
      float2 value[9];

#if (HISQ_NEW_OPROD_LOAD_TEX == 1)
        value[0] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9);
        value[1] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + stride);
        value[2] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + 2*stride);
        value[3] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + 3*stride);
        value[4] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + 4*stride);
        value[5] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + 5*stride);
        value[6] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + 6*stride);
        value[7] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + 7*stride);
        value[8] = tex1Dfetch( ((oddness)?newOprod1TexSingle:newOprod0TexSingle),  idx+dir*stride*9 + 8*stride);
#else
        for(int i=0; i<9; ++i) value[i] = field[i];
#endif
        field[idx + dir*stride*9]              = value[0] + coeff*mat[0];
        field[idx + dir*stride*9 + stride]     = value[1] + coeff*mat[1];
        field[idx + dir*stride*9 + stride*2]   = value[2] + coeff*mat[2];
        field[idx + dir*stride*9 + stride*3]   = value[3] + coeff*mat[3];
        field[idx + dir*stride*9 + stride*4]   = value[4] + coeff*mat[4];
        field[idx + dir*stride*9 + stride*5]   = value[5] + coeff*mat[5];
        field[idx + dir*stride*9 + stride*6]   = value[6] + coeff*mat[6];
        field[idx + dir*stride*9 + stride*7]   = value[7] + coeff*mat[7];
        field[idx + dir*stride*9 + stride*8]   = value[8] + coeff*mat[8];

        return;
      }


    // only works if Promote<T,U>::Type = T

    template<class T, class U, class V>
      inline __device__
      void addMatrixToField(const T* const mat, int dir, int idx, U coeff,
          V* const field_even, V* const field_odd, int oddness, int stride)
      {
        V* const field = (oddness)?field_odd: field_even;
        field[idx + dir*stride*9]          += coeff*mat[0];
        field[idx + dir*stride*9 + stride]     += coeff*mat[1];
        field[idx + dir*stride*9 + stride*2]   += coeff*mat[2];
        field[idx + dir*stride*9 + stride*3]   += coeff*mat[3];
        field[idx + dir*stride*9 + stride*4]   += coeff*mat[4];
        field[idx + dir*stride*9 + stride*5]   += coeff*mat[5];
        field[idx + dir*stride*9 + stride*6]   += coeff*mat[6];
        field[idx + dir*stride*9 + stride*7]   += coeff*mat[7];
        field[idx + dir*stride*9 + stride*8]   += coeff*mat[8];

        return;
      }


    template<class T, class U, class V>
      inline __device__
      void addMatrixToField(const T* const mat, int idx, U coeff, V* const field_even,
          V* const field_odd, int oddness, int stride)
      {
        V* const field = (oddness)?field_odd: field_even;
        field[idx + stride*0]   += coeff*mat[0];
        field[idx + stride*1]   += coeff*mat[1];
        field[idx + stride*2]   += coeff*mat[2];
        field[idx + stride*3]   += coeff*mat[3];
        field[idx + stride*4]   += coeff*mat[4];
        field[idx + stride*5]   += coeff*mat[5];
        field[idx + stride*6]   += coeff*mat[6];
        field[idx + stride*7]   += coeff*mat[7];
        field[idx + stride*8]   += coeff*mat[8];

        return;
      }

    template<class T, class U>
      inline __device__
      void addMatrixToField_test(const T* const mat, int idx, U coeff, T* const field_even,
          T* const field_odd, int oddness, int stride)
      {
        T* const field = (oddness)?field_odd: field_even;
        //T oldvalue=field[idx];
        field[idx ]         += coeff*mat[0];
        field[idx + stride]     += coeff*mat[1];
        field[idx + stride*2]   += coeff*mat[2];
        field[idx + stride*3]   += coeff*mat[3];
        field[idx + stride*4]   += coeff*mat[4];
        field[idx + stride*5]   += coeff*mat[5];
        field[idx + stride*6]   += coeff*mat[6];
        field[idx + stride*7]   += coeff*mat[7];
        field[idx + stride*8]   += coeff*mat[8];

        printf("value is  coeff(%f) * mat[0].x(%f)=%f\n", coeff, mat[0].x, field[idx].x);
        return;
      }

    template<class T, class U>
      inline __device__
      void storeMatrixToField(const T* const mat, int dir, int idx, U* const field_even, U* const field_odd, int oddness, int stride)
      {
        T* const field = (oddness)?field_odd: field_even;
        field[idx + dir*stride*9]          = mat[0];
        field[idx + dir*stride*9 + stride]     = mat[1];
        field[idx + dir*stride*9 + stride*2]   = mat[2];
        field[idx + dir*stride*9 + stride*3]   = mat[3];
        field[idx + dir*stride*9 + stride*4]   = mat[4];
        field[idx + dir*stride*9 + stride*5]   = mat[5];
        field[idx + dir*stride*9 + stride*6]   = mat[6];
        field[idx + dir*stride*9 + stride*7]   = mat[7];
        field[idx + dir*stride*9 + stride*8]   = mat[8];

        return;
      }


    template<class T, class U>
      inline __device__
      void storeMatrixToField(const T* const mat, int idx, U* const field_even, U* const field_odd, int oddness, int stride)
      {
        U* const field = (oddness)?field_odd: field_even;
        field[idx]          = mat[0];
        field[idx + stride]     = mat[1];
        field[idx + stride*2]   = mat[2];
        field[idx + stride*3]   = mat[3];
        field[idx + stride*4]   = mat[4];
        field[idx + stride*5]   = mat[5];
        field[idx + stride*6]   = mat[6];
        field[idx + stride*7]   = mat[7];
        field[idx + stride*8]   = mat[8];

        return;
      }


    template<class T, class U, class V>
      inline __device__
      void storeMatrixToMomentumField(const T* const mat, int dir, int idx, U coeff,
          V* const mom_even, V* const mom_odd, int oddness, int stride)
      {
        V* const mom_field = (oddness)?mom_odd:mom_even;
        T temp2;
        temp2.x = (mat[1].x - mat[3].x)*0.5*coeff;
        temp2.y = (mat[1].y + mat[3].y)*0.5*coeff;
        mom_field[idx + dir*stride*5] = temp2;

        temp2.x = (mat[2].x - mat[6].x)*0.5*coeff;
        temp2.y = (mat[2].y + mat[6].y)*0.5*coeff;
        mom_field[idx + dir*stride*5 + stride] = temp2;

        temp2.x = (mat[5].x - mat[7].x)*0.5*coeff;
        temp2.y = (mat[5].y + mat[7].y)*0.5*coeff;
        mom_field[idx + dir*stride*5 + stride*2] = temp2;

        const typename T::value_type temp = (mat[0].y + mat[4].y + mat[8].y)*0.3333333333333333333333333;
        temp2.x =  (mat[0].y-temp)*coeff;
        temp2.y =  (mat[4].y-temp)*coeff;
        mom_field[idx + dir*stride*5 + stride*3] = temp2;

        temp2.x = (mat[8].y - temp)*coeff;
        temp2.y = 0.0;
        mom_field[idx + dir*stride*5 + stride*4] = temp2;

        return;
      }

    template<class RealX>
      struct ArrayLength
      {
        static const int result=9;
      };

    template<>
      struct ArrayLength<float4>
      {
        static const int result=5;
      };

    // Flops: four matrix additions per lattice site = 72 Flops per lattice site
    template<class RealA>
      __global__ void
      do_one_link_term_kernel(const RealA* const oprodEven, const RealA* const oprodOdd,
          typename RealTypeId<RealA>::Type coeff,
          RealA* const outputEven, RealA* const outputOdd, hisq_kernel_param_t kparam)
      {
	typedef typename std::remove_reference<decltype(RealA::x)>::type real;
	typedef complex<real> Complex;

        int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= kparam.threads) return;
	int oddBit = threadIdx.y;
#ifdef MULTI_GPU
        int dx[4] = {0,0,0,0};
        int x[4];
        getCoords(x, sid, kparam.X, oddBit);
        int E[4] = {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};
        for(int dir=0; dir<4; ++dir) x[dir] += 2;
        int new_sid = linkIndexShift(x,dx,E);
#else
        int new_sid = sid;
#endif
	for(int sig=0; sig<4; ++sig){
          Complex COLOR_MAT_W[ArrayLength<RealA>::result];
          loadMatrixFromField(oprodEven, oprodOdd, sig, new_sid, COLOR_MAT_W, oddBit, kparam.color_matrix_stride);
          addMatrixToField(COLOR_MAT_W, sig, new_sid, coeff, outputEven, outputOdd, oddBit, kparam.color_matrix_stride);
	}
        return;
      }

    template<int N>
      __device__ void loadLink(const double2* const linkEven, const double2* const linkOdd, int dir, int idx, double2* const var, int oddness, int stride){
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
        HISQ_LOAD_MATRIX_18_DOUBLE_TEX((oddness)?thinLink1TexDouble:thinLink0TexDouble,  (oddness)?linkOdd:linkEven, dir, idx, var, stride);
#else
        loadMatrixFromField<N>(linkEven, linkOdd, dir, idx, var, oddness, stride);
#endif
      }

    template<>
      void loadLink<12>(const double2* const linkEven, const double2* const linkOdd, int dir, int idx, double2* const var, int oddness, int stride){
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
        HISQ_LOAD_MATRIX_12_DOUBLE_TEX((oddness)?thinLink1TexDouble:thinLink0TexDouble,  (oddness)?linkOdd:linkEven,dir, idx, var, stride);
#else
        loadMatrixFromField<6>(linkEven, linkOdd, dir, idx, var, oddness, stride);
#endif
      }

    template<int N>
      __device__ void loadLink(const float4* const linkEven, const float4* const linkOdd, int dir, int idx, float2* const var, int oddness, int stride){
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
        HISQ_LOAD_MATRIX_12_SINGLE_TEX((oddness)?thinLink1TexSingle_recon:thinLink0TexSingle_recon, dir, idx, var, stride);
#else
        loadMatrixFromField<N>(linkEven, linkOdd, dir, idx, var, oddness, stride);
#endif
      }

    template<int N>
      __device__ void loadLink(const float2* const linkEven, const float2* const linkOdd, int dir, int idx, float2* const var , int oddness, int stride){
#if (HISQ_SITE_MATRIX_LOAD_TEX == 1)
        HISQ_LOAD_MATRIX_18_SINGLE_TEX((oddness)?thinLink1TexSingle:thinLink0TexSingle, dir, idx, var, stride);
#else
        loadMatrixFromField<N>(linkEven, linkOdd, dir, idx, var, oddness, stride);
#endif
      }



#define DD_CONCAT(n,r) n ## r ## kernel

#define HISQ_KERNEL_NAME(a,b) DD_CONCAT(a,b)
    //precision: 0 is for double, 1 is for single

    //double precision, recon=18
#define PRECISION 0
#define RECON 18
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON

    //double precision, recon=12
#define PRECISION 0
#define RECON 12
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON

    //single precision, recon=18
#define PRECISION 1
#define RECON 18
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON

    //single precision, recon=12
#define PRECISION 1
#define RECON 12
#include "hisq_paths_force_core.h"
#undef PRECISION
#undef RECON





    template<class RealA, class RealB>
      class MiddleLink : public TunableLocalParity {

        private:
          const cudaGaugeField &link;
          const cudaGaugeField &oprod;
          const cudaGaugeField &Qprev;
          const int sig;
          const int mu;
          const typename RealTypeId<RealA>::Type &coeff;
          cudaGaugeField &Pmu;
          cudaGaugeField &P3;
          cudaGaugeField &Qmu;
          cudaGaugeField &newOprod;
          const hisq_kernel_param_t &kparam;
          unsigned int minThreads() const { return kparam.threads; }

        public:
          MiddleLink(const cudaGaugeField &link,
              const cudaGaugeField &oprod,
              const cudaGaugeField &Qprev,
              int sig, int mu,
              const typename RealTypeId<RealA>::Type &coeff,
              cudaGaugeField &Pmu, // write only
              cudaGaugeField &P3,  // write only
              cudaGaugeField &Qmu,
              cudaGaugeField &newOprod,
              const hisq_kernel_param_t &kparam) :
            link(link), oprod(oprod), Qprev(Qprev), sig(sig), mu(mu),
            coeff(coeff), Pmu(Pmu), P3(P3), Qmu(Qmu), newOprod(newOprod), kparam(kparam)
        {	; }
          // need alternative constructor to hack around null pointer passing
          MiddleLink(const cudaGaugeField &link,
              const cudaGaugeField &oprod,
              int sig, int mu,
              const typename RealTypeId<RealA>::Type &coeff,
              cudaGaugeField &Pmu, // write only
              cudaGaugeField &P3,  // write only
              cudaGaugeField &Qmu,
              cudaGaugeField &newOprod,
              const hisq_kernel_param_t &kparam) :
            link(link), oprod(oprod), Qprev(link), sig(sig), mu(mu),
            coeff(coeff), Pmu(Pmu), P3(P3), Qmu(Qmu), newOprod(newOprod), kparam(kparam)
        {	; }
          virtual ~MiddleLink() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << kparam.D[0] << "x";
            vol << kparam.D[1] << "x";
            vol << kparam.D[2] << "x";
            vol << kparam.D[3];
            aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
            aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }


#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
          ((typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
           (typeA*)Qprev_even, (typeA*)Qprev_odd,				\
           (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
           sig, mu, coeff,							\
           (typeA*)Pmu.Even_p(), (typeA*)Pmu.Odd_p(),			\
           (typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
           (typeA*)Qmu.Even_p(), (typeA*)Qmu.Odd_p(),			\
           (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(), kparam)


#define CALL_MIDDLE_LINK_KERNEL(sig_sign, mu_sign)			\
      if (sizeof(RealA) == sizeof(float2)) {				\
	if (recon  == QUDA_RECONSTRUCT_NO) {				\
	  do_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float2); \
	} else {							\
	  do_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float4); \
	}								\
      } else {								\
	if (recon  == QUDA_RECONSTRUCT_NO) {				\
	  do_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	} else {							\
	  do_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	}								\
      }

          void apply(const cudaStream_t &stream) {
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            QudaReconstructType recon = link.Reconstruct();
            kparam.oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
				  + kparam.base_idx[2] + kparam.base_idx[3])&1;

            const void *Qprev_even = (&Qprev == &link) ? NULL : Qprev.Even_p();
            const void *Qprev_odd = (&Qprev == &link) ? NULL : Qprev.Odd_p();

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_MIDDLE_LINK_KERNEL(1,1);
            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
              CALL_MIDDLE_LINK_KERNEL(1,0);
            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_MIDDLE_LINK_KERNEL(0,1);
            }else{
              CALL_MIDDLE_LINK_KERNEL(0,0);
            }
          }

#undef CALL_ARGUMENTS
#undef CALL_MIDDLE_LINK_KERNEL

          void preTune() {
            Pmu.backup();
            P3.backup();
            Qmu.backup();
            newOprod.backup();
          }

          void postTune() {
            Pmu.restore();
            P3.restore();
            Qmu.restore();
            newOprod.restore();
          }

          long long flops() const { return 0; }
      };


    template<class RealA, class RealB>
      class LepageMiddleLink : public TunableLocalParity {

        private:
          const cudaGaugeField &link;
          const cudaGaugeField &oprod;
          const cudaGaugeField &Qprev;
          const int sig;
          const int mu;
          const typename RealTypeId<RealA>::Type &coeff;
          cudaGaugeField &P3; // write only
          cudaGaugeField &newOprod;
          const hisq_kernel_param_t &kparam;
          unsigned int minThreads() const { return kparam.threads; }

        public:
          LepageMiddleLink(const cudaGaugeField &link,
              const cudaGaugeField &oprod,
              const cudaGaugeField &Qprev,
              int sig, int mu,
              const typename RealTypeId<RealA>::Type &coeff,
              cudaGaugeField &P3, cudaGaugeField &newOprod,
              const hisq_kernel_param_t &kparam) :
            link(link), oprod(oprod), Qprev(Qprev), sig(sig), mu(mu),
            coeff(coeff), P3(P3), newOprod(newOprod), kparam(kparam)
        {	; }
          virtual ~LepageMiddleLink() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << kparam.D[0] << "x";
            vol << kparam.D[1] << "x";
            vol << kparam.D[2] << "x";
            vol << kparam.D[3];
            aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
            aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }

#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
      ((typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
       (typeA*)Qprev.Even_p(), (typeA*)Qprev.Odd_p(),			\
       (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
       sig, mu, coeff,							\
       (typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
       (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(),		\
       kparam)

#define CALL_MIDDLE_LINK_KERNEL(sig_sign, mu_sign)			\
      if (sizeof(RealA) == sizeof(float2)) {				\
	if (recon  == QUDA_RECONSTRUCT_NO) {				\
	  do_lepage_middle_link_sp_18_kernel<float2, float2, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float2); \
	} else {							\
	  do_lepage_middle_link_sp_12_kernel<float2, float4, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float4); \
	}								\
      } else {								\
	if (recon  == QUDA_RECONSTRUCT_NO) {				\
	  do_lepage_middle_link_dp_18_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	} else {							\
	  do_lepage_middle_link_dp_12_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	}								\
      }									\

      void apply(const cudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	QudaReconstructType recon = link.Reconstruct();
	kparam.oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
				 + kparam.base_idx[2] + kparam.base_idx[3])&1;

	if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(1,1);
	}else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(1,0);
	}else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	  CALL_MIDDLE_LINK_KERNEL(0,1);
	}else{
	  CALL_MIDDLE_LINK_KERNEL(0,0);
	}

      }

#undef CALL_ARGUMENTS
#undef CALL_MIDDLE_LINK_KERNEL

      void preTune() {
	P3.backup();
	newOprod.backup();
      }

      void postTune() {
	P3.restore();
	newOprod.restore();
      }

      long long flops() const {
	if(GOES_FORWARDS(sig)) return 810ll*kparam.X[0]*kparam.X[1]*kparam.X[2]*kparam.X[3];
	return kparam.X[0]*kparam.X[1]*kparam.X[2]*kparam.X[3]*396ll;
      }
    };

    template<class RealA, class RealB>
      class SideLink : public TunableLocalParity {

        private:
          const cudaGaugeField &link;
          const cudaGaugeField &P3;
          const cudaGaugeField &oprod;
          const int sig;
          const int mu;
          const typename RealTypeId<RealA>::Type &coeff;
          const typename RealTypeId<RealA>::Type &accumu_coeff;
          cudaGaugeField &shortP;
          cudaGaugeField &newOprod;
          const hisq_kernel_param_t &kparam;

          unsigned int minThreads() const { return kparam.threads; }

        public:
          SideLink(const cudaGaugeField &link,
              const cudaGaugeField &P3,
              const cudaGaugeField &oprod,
              int sig, int mu,
              const typename RealTypeId<RealA>::Type &coeff,
              const typename RealTypeId<RealA>::Type &accumu_coeff,
              cudaGaugeField &shortP,
              cudaGaugeField &newOprod,
              const hisq_kernel_param_t &kparam) :
            link(link), P3(P3), oprod(oprod),
            sig(sig), mu(mu), coeff(coeff), accumu_coeff(accumu_coeff),
            shortP(shortP), newOprod(newOprod), kparam(kparam)
        {	; }
          virtual ~SideLink() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << kparam.D[0] << "x";
            vol << kparam.D[1] << "x";
            vol << kparam.D[2] << "x";
            vol << kparam.D[3];
            aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
            aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }

#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
          ((typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
           (typeA*)oprod.Even_p(),  (typeA*)oprod.Odd_p(),			\
           (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
           sig, mu,								\
           coeff,			\
           (typename RealTypeId<typeA>::Type) accumu_coeff,			\
           (typeA*)shortP.Even_p(), (typeA*)shortP.Odd_p(),			\
           (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(),		\
           kparam)

#define CALL_SIDE_LINK_KERNEL(sig_sign, mu_sign)			\
      if (sizeof(RealA) == sizeof(float2)) {				\
	if (recon  == QUDA_RECONSTRUCT_NO) {				\
	  do_side_link_sp_18_kernel<float2, float2, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float2); \
	} else {							\
	  do_side_link_sp_12_kernel<float2, float4, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float4); \
	}								\
      } else {								\
	if(recon  == QUDA_RECONSTRUCT_NO){				\
	  do_side_link_dp_18_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	} else {							\
	  do_side_link_dp_12_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	}								\
      }

          void apply(const cudaStream_t &stream) {
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            QudaReconstructType recon = link.Reconstruct();
            kparam.oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
                + kparam.base_idx[2] + kparam.base_idx[3])&1;

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_SIDE_LINK_KERNEL(1,1);
            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
              CALL_SIDE_LINK_KERNEL(1,0);
            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_SIDE_LINK_KERNEL(0,1);
            }else{
              CALL_SIDE_LINK_KERNEL(0,0);
            }
          }

#undef CALL_SIDE_LINK_KERNEL
#undef CALL_ARGUMENTS

          void preTune() {
            shortP.backup();
            newOprod.backup();
          }

          void postTune() {
            shortP.restore();
            newOprod.restore();
          }

          long long flops() const { return 0; }
      };


    template<class RealA, class RealB>
      class SideLinkShort : public TunableLocalParity {

        private:
          const cudaGaugeField &link;
          const cudaGaugeField &P3;
          const int sig;
          const int mu;
          const typename RealTypeId<RealA>::Type &coeff;
          cudaGaugeField &newOprod;
          const hisq_kernel_param_t &kparam;

          unsigned int minThreads() const { return kparam.threads; }

        public:
          SideLinkShort(const cudaGaugeField &link, const cudaGaugeField &P3, int sig, int mu,
              const typename RealTypeId<RealA>::Type &coeff, cudaGaugeField &newOprod,
              const hisq_kernel_param_t &kparam) :
            link(link), P3(P3), sig(sig), mu(mu), coeff(coeff), newOprod(newOprod), kparam(kparam)
        {	; }
          virtual ~SideLinkShort() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << kparam.D[0] << "x";
            vol << kparam.D[1] << "x";
            vol << kparam.D[2] << "x";
            vol << kparam.D[3];
            aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
            aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }

#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
          ((typeA*)P3.Even_p(), (typeA*)P3.Odd_p(),				\
           (typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
           sig, mu,	(typename RealTypeId<typeA>::Type) coeff,		\
           (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(), kparam)


#define CALL_SIDE_LINK_KERNEL(sig_sign, mu_sign)			\
    if (sizeof(RealA) == sizeof(float2)) {				\
      if (recon  == QUDA_RECONSTRUCT_NO) {				\
	do_side_link_short_sp_18_kernel<float2, float2, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float2); \
      }else{								\
	do_side_link_short_sp_12_kernel<float2, float4, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float4); \
      }									\
    } else {								\
      if(recon  == QUDA_RECONSTRUCT_NO){				\
	do_side_link_short_dp_18_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
      }else{								\
	do_side_link_short_dp_12_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
      }									\
    }

          void apply(const cudaStream_t &stream) {
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            QudaReconstructType recon = link.Reconstruct();
            kparam.oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
                + kparam.base_idx[2] + kparam.base_idx[3])&1;

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_SIDE_LINK_KERNEL(1,1);
            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
              CALL_SIDE_LINK_KERNEL(1,0);

            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_SIDE_LINK_KERNEL(0,1);
            }else{
              CALL_SIDE_LINK_KERNEL(0,0);
            }
          }

#undef CALL_SIDE_LINK_KERNEL
#undef CALL_ARGUMENTS


          void preTune() {
            newOprod.backup();
          }

          void postTune() {
            newOprod.restore();
          }

          long long flops() const { return 0; }
      };

    template<class RealA, class RealB>
      class AllLink : public TunableLocalParity {

        private:
          const cudaGaugeField &link;
          const cudaGaugeField &oprod;
          const cudaGaugeField &Qprev;
          const int sig;
          const int mu;
          const typename RealTypeId<RealA>::Type &coeff;
          const typename RealTypeId<RealA>::Type &accumu_coeff;
          cudaGaugeField &shortP;
          cudaGaugeField &newOprod;
          const hisq_kernel_param_t &kparam;

          unsigned int minThreads() const { return kparam.threads; }

        public:
          AllLink(const cudaGaugeField &link,
              const cudaGaugeField &oprod,
              const cudaGaugeField &Qprev,
              int sig, int mu,
              const typename RealTypeId<RealA>::Type &coeff,
              const typename RealTypeId<RealA>::Type &accumu_coeff,
              cudaGaugeField &shortP, cudaGaugeField &newOprod,
              const hisq_kernel_param_t &kparam) :
            link(link), oprod(oprod), Qprev(Qprev), sig(sig), mu(mu),
            coeff(coeff), accumu_coeff(accumu_coeff), shortP(shortP),
            newOprod(newOprod), kparam(kparam)
        { ; }
          virtual ~AllLink() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << kparam.D[0] << "x";
            vol << kparam.D[1] << "x";
            vol << kparam.D[2] << "x";
            vol << kparam.D[3];
            aux << "threads=" << kparam.threads << ",prec=" << link.Precision();
            aux << ",recon=" << link.Reconstruct() << ",sig=" << sig << ",mu=" << mu;
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }

#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid, tp.block>>>		\
          ((typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
           (typeA*)Qprev.Even_p(), (typeA*)Qprev.Odd_p(),			\
           (typeB*)link.Even_p(), (typeB*)link.Odd_p(), sig,  mu,		\
           (typename RealTypeId<typeA>::Type)coeff,				\
           (typename RealTypeId<typeA>::Type)accumu_coeff,			\
           (typeA*)shortP.Even_p(),(typeA*)shortP.Odd_p(),			\
           (typeA*)newOprod.Even_p(), (typeA*)newOprod.Odd_p(), kparam)

#define CALL_ALL_LINK_KERNEL(sig_sign, mu_sign)				\
      if (sizeof(RealA) == sizeof(float2)) {				\
	if (recon  == QUDA_RECONSTRUCT_NO) {				\
	  do_all_link_sp_18_kernel<float2, float2, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float2); \
	} else {							\
	  do_all_link_sp_12_kernel<float2, float4, sig_sign, mu_sign> CALL_ARGUMENTS(float2, float4); \
	}								\
      } else {								\
	if (recon  == QUDA_RECONSTRUCT_NO) {				\
	  do_all_link_dp_18_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	} else {							\
	  do_all_link_dp_12_kernel<double2, double2, sig_sign, mu_sign> CALL_ARGUMENTS(double2, double2); \
	}								\
      }

          void apply(const cudaStream_t &stream) {
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            QudaReconstructType recon = link.Reconstruct();
            kparam.oddness_change = (kparam.base_idx[0] + kparam.base_idx[1]
                + kparam.base_idx[2] + kparam.base_idx[3])&1;

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_ALL_LINK_KERNEL(1, 1);
            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
              CALL_ALL_LINK_KERNEL(1, 0);
            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
              CALL_ALL_LINK_KERNEL(0, 1);
            }else{
              CALL_ALL_LINK_KERNEL(0, 0);
            }

            return;
          }

#undef CALL_ARGUMENTS
#undef CALL_ALL_LINK_KERNEL

          void preTune() {
            shortP.backup();
            newOprod.backup();
          }

          void postTune() {
            shortP.restore();
            newOprod.restore();
          }

          long long flops() const {
	    if(GOES_FORWARDS(sig)) return kparam.X[0]*kparam.X[1]*kparam.X[2]*kparam.X[3]*1242ll;

	    return kparam.X[0]*kparam.X[1]*kparam.X[2]*kparam.X[3]*828ll;
	  }
      };


    template<class RealA, class RealB>
      class OneLinkTerm : public TunableLocalParity {

        private:
          const cudaGaugeField &oprod;
          const typename RealTypeId<RealA>::Type &coeff;
          cudaGaugeField &ForceMatrix;
          int X[4];
          hisq_kernel_param_t kparam;

          unsigned int minThreads() const { return X[0]*X[1]*X[2]*X[3]/2; }

        public:
          OneLinkTerm(const cudaGaugeField &oprod,
              const typename RealTypeId<RealA>::Type &coeff,
              cudaGaugeField &ForceMatrix, const QudaGaugeParam& param) :
            oprod(oprod), coeff(coeff), ForceMatrix(ForceMatrix)
        {
          for(int dir=0; dir<4; ++dir) X[dir] = param.X[dir];

          kparam.threads = X[0]*X[1]*X[2]*X[3]/2;
          for(int dir=0; dir<4; ++dir){
            kparam.X[dir] = X[dir];
          }
          kparam.setStride(param);
        }

          virtual ~OneLinkTerm() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << X[0] << "x";
            vol << X[1] << "x";
            vol << X[2] << "x";
            vol << X[3];
            int threads = X[0]*X[1]*X[2]*X[3]/2;
            aux << "threads=" << threads << ",prec=" << oprod.Precision();
            aux << ",coeff=" << coeff;
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }

          void apply(const cudaStream_t &stream) {
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

            do_one_link_term_kernel<RealA><<<tp.grid,tp.block>>>(static_cast<const RealA*>(oprod.Even_p()),
								 static_cast<const RealA*>(oprod.Odd_p()),
								 coeff,
								 static_cast<RealA*>(ForceMatrix.Even_p()),
								 static_cast<RealA*>(ForceMatrix.Odd_p()),
								 kparam);
          }

          void preTune() {
            ForceMatrix.backup();
          }

          void postTune() {
            ForceMatrix.restore();
          }

          long long flops() const {
	    return 72ll*kparam.X[0]*kparam.X[1]*kparam.X[2]*kparam.X[3];
	  }
      };


    template<class RealA, class RealB>
      class LongLinkTerm : public TunableLocalParity {

        private:
          const cudaGaugeField &link;
          const cudaGaugeField &naikOprod;
          const typename RealTypeId<RealA>::Type naik_coeff;
          cudaGaugeField &output;
          int X[4];
          const hisq_kernel_param_t &kparam;

          unsigned int minThreads() const { return X[0]*X[1]*X[2]*X[3]/2; }

        public:
          LongLinkTerm(const cudaGaugeField &link, const cudaGaugeField &naikOprod,
              const typename RealTypeId<RealA>::Type &naik_coeff,
              cudaGaugeField &output, const hisq_kernel_param_t &kparam) :
            link(link), naikOprod(naikOprod),  naik_coeff(naik_coeff), output(output),
            kparam(kparam)
        { for(int dir=0; dir<4; ++dir) X[dir] = kparam.X[dir]; }

          virtual ~LongLinkTerm() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << X[0] << "x";
            vol << X[1] << "x";
            vol << X[2] << "x";
            vol << X[3];
            int threads = X[0]*X[1]*X[2]*X[3]/2;
            aux << "threads=" << threads << ",prec=" << link.Precision();
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }

#define CALL_ARGUMENTS(typeA, typeB) <<<tp.grid,tp.block>>>		\
          ((typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
           (typeA*)naikOprod.Even_p(),  (typeA*)naikOprod.Odd_p(),		\
           naik_coeff,							\
           (typeA*)output.Even_p(), (typeA*)output.Odd_p(),			\
           kparam);

          void apply(const cudaStream_t &stream) {
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            QudaReconstructType recon = link.Reconstruct();

            if(sizeof(RealA) == sizeof(float2)){
              if(recon == QUDA_RECONSTRUCT_NO){
                do_longlink_sp_18_kernel<float2,float2> CALL_ARGUMENTS(float2, float2);
              }else{
                do_longlink_sp_12_kernel<float2,float4> CALL_ARGUMENTS(float2, float4);
              }
            }else{
              if(recon == QUDA_RECONSTRUCT_NO){
                do_longlink_dp_18_kernel<double2,double2> CALL_ARGUMENTS(double2, double2);
              }else{
                do_longlink_dp_12_kernel<double2,double2> CALL_ARGUMENTS(double2, double2);
              }
            }
          }

#undef CALL_ARGUMENTS

          void preTune() {
            output.backup();
          }

          void postTune() {
            output.restore();
          }

          long long flops() const { return 4968ll*kparam.X[0]*kparam.X[1]*kparam.X[2]*kparam.X[3]; }
      };



    template<class RealA, class RealB>
      class CompleteForce : public TunableLocalParity {

        private:
          const cudaGaugeField &link;
          const cudaGaugeField &oprod;
          cudaGaugeField &mom;
          int X[4];
          hisq_kernel_param_t kparam;

          unsigned int minThreads() const { return X[0]*X[1]*X[2]*X[3]/2; }

        public:
          CompleteForce(const cudaGaugeField &link, const cudaGaugeField &oprod,
             cudaGaugeField &mom, const QudaGaugeParam &param) :
            link(link), oprod(oprod), mom(mom)
        {

          for(int dir=0; dir<4; ++dir){
            X[dir] = param.X[dir];
            kparam.X[dir] = X[dir];
          }
          kparam.threads = X[0]*X[1]*X[2]*X[3]/2;
          kparam.setStride(param);
        }

          virtual ~CompleteForce() { ; }

          TuneKey tuneKey() const {
            std::stringstream vol, aux;
            vol << X[0] << "x";
            vol << X[1] << "x";
            vol << X[2] << "x";
            vol << X[3];
            int threads = X[0]*X[1]*X[2]*X[3]/2;
            aux << "threads=" << threads << ",prec=" << link.Precision();
            return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
          }

#define CALL_ARGUMENTS(typeA, typeB)  <<<tp.grid, tp.block>>>	\
          ((typeB*)link.Even_p(), (typeB*)link.Odd_p(),			\
           (typeA*)oprod.Even_p(), (typeA*)oprod.Odd_p(),			\
           (typeA*)mom.Even_p(), (typeA*)mom.Odd_p(),			\
           kparam);

          void apply(const cudaStream_t &stream) {
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            QudaReconstructType recon = link.Reconstruct();

            if(sizeof(RealA) == sizeof(float2)){
              if(recon == QUDA_RECONSTRUCT_NO){
                do_complete_force_sp_18_kernel<float2,float2> CALL_ARGUMENTS(float2, float2);
              }else{
                do_complete_force_sp_12_kernel<float2,float4> CALL_ARGUMENTS(float2, float4);
              }
            }else{
              if(recon == QUDA_RECONSTRUCT_NO){
                do_complete_force_dp_18_kernel<double2,double2> CALL_ARGUMENTS(double2, double2);
              }else{
                do_complete_force_dp_12_kernel<double2,double2> CALL_ARGUMENTS(double2, double2);
              }
            }
          }

#undef CALL_ARGUMENTS

          void preTune() {
            mom.backup();
          }

          void postTune() {
            mom.restore();
          }

          long long flops() const {
	    return kparam.X[0]*kparam.X[1]*kparam.X[2]*kparam.X[3]*792ll;
	  }
      };


#undef Pmu
#undef Pnumu
#undef P3
#undef P5
#undef Qmu
#undef Qnumu

    static void
      bind_tex_link(const cudaGaugeField& link, const cudaGaugeField& newOprod)
      {
        if(link.Precision() == QUDA_DOUBLE_PRECISION){
          cudaBindTexture(0, thinLink0TexDouble, link.Even_p(), link.Bytes()/2);
          cudaBindTexture(0, thinLink1TexDouble, link.Odd_p(), link.Bytes()/2);

          cudaBindTexture(0, newOprod0TexDouble, newOprod.Even_p(), newOprod.Bytes()/2);
          cudaBindTexture(0, newOprod1TexDouble, newOprod.Odd_p(), newOprod.Bytes()/2);
        }else{
          if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
            cudaBindTexture(0, thinLink0TexSingle, link.Even_p(), link.Bytes()/2);
            cudaBindTexture(0, thinLink1TexSingle, link.Odd_p(), link.Bytes()/2);
          }else{
            cudaBindTexture(0, thinLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
            cudaBindTexture(0, thinLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);
          }
          cudaBindTexture(0, newOprod0TexSingle, newOprod.Even_p(), newOprod.Bytes()/2);
          cudaBindTexture(0, newOprod1TexSingle, newOprod.Odd_p(), newOprod.Bytes()/2);

        }
      }

    static void
      unbind_tex_link(const cudaGaugeField& link, const cudaGaugeField& newOprod)
      {
        if(link.Precision() == QUDA_DOUBLE_PRECISION){
          cudaUnbindTexture(thinLink0TexDouble);
          cudaUnbindTexture(thinLink1TexDouble);
          cudaUnbindTexture(newOprod0TexDouble);
          cudaUnbindTexture(newOprod1TexDouble);
        }else{
          if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
            cudaUnbindTexture(thinLink0TexSingle);
            cudaUnbindTexture(thinLink1TexSingle);
          }else{
            cudaUnbindTexture(thinLink0TexSingle_recon);
            cudaUnbindTexture(thinLink1TexSingle_recon);
          }
          cudaUnbindTexture(newOprod0TexSingle);
          cudaUnbindTexture(newOprod1TexSingle);
        }
      }

    template<class Real, class RealA, class RealB>
      static void
      do_hisq_staples_force_cuda( PathCoefficients<Real> act_path_coeff,
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
        Real coeff;
        Real OneLink, Lepage, FiveSt, ThreeSt, SevenSt;
        Real mLepage, mFiveSt, mThreeSt;

        OneLink = act_path_coeff.one;
        ThreeSt = act_path_coeff.three; mThreeSt = -ThreeSt;
        FiveSt  = act_path_coeff.five; mFiveSt  = -FiveSt;
        SevenSt = act_path_coeff.seven;
        Lepage  = act_path_coeff.lepage; mLepage  = -Lepage;

       	OneLinkTerm<RealA, RealB> oneLink(oprod, OneLink, newOprod, param);
        oneLink.apply(0);
        checkCudaError();

        int ghostDim[4]={
          commDimPartitioned(0),
          commDimPartitioned(1),
          commDimPartitioned(2),
          commDimPartitioned(3)
        };

        hisq_kernel_param_t kparam_1g, kparam_2g;

        for(int dir=0; dir<4; ++dir){
          kparam_1g.X[dir] = param.X[dir];
          kparam_2g.X[dir] = param.X[dir];
        }

        kparam_1g.setStride(param);
        kparam_2g.setStride(param);

#ifdef MULTI_GPU
        kparam_1g.D[0] = commDimPartitioned(0)?(param.X[0]+2):(param.X[0]);
        kparam_1g.D[1] = commDimPartitioned(1)?(param.X[1]+2):(param.X[1]);
        kparam_1g.D[2] = commDimPartitioned(2)?(param.X[2]+2):(param.X[2]);
        kparam_1g.D[3] = commDimPartitioned(3)?(param.X[3]+2):(param.X[3]);
        kparam_1g.D1h =  kparam_1g.D[0]/2;
        kparam_1g.base_idx[0]=commDimPartitioned(0)?1:2;
        kparam_1g.base_idx[1]=commDimPartitioned(1)?1:2;
        kparam_1g.base_idx[2]=commDimPartitioned(2)?1:2;
        kparam_1g.base_idx[3]=commDimPartitioned(3)?1:2;
        kparam_1g.threads = kparam_1g.D[0]*kparam_1g.D[1]*kparam_1g.D[2]*kparam_1g.D[3]/2;

        kparam_2g.D[0] = commDimPartitioned(0)?(param.X[0]+4):(param.X[0]);
        kparam_2g.D[1] = commDimPartitioned(1)?(param.X[1]+4):(param.X[1]);
        kparam_2g.D[2] = commDimPartitioned(2)?(param.X[2]+4):(param.X[2]);
        kparam_2g.D[3] = commDimPartitioned(3)?(param.X[3]+4):(param.X[3]);
        kparam_2g.D1h = kparam_2g.D[0]/2;
        kparam_2g.base_idx[0]=commDimPartitioned(0)?0:2;
        kparam_2g.base_idx[1]=commDimPartitioned(1)?0:2;
        kparam_2g.base_idx[2]=commDimPartitioned(2)?0:2;
        kparam_2g.base_idx[3]=commDimPartitioned(3)?0:2;
        kparam_2g.threads = kparam_2g.D[0]*kparam_2g.D[1]*kparam_2g.D[2]*kparam_2g.D[3]/2;


        for(int i=0;i < 4; i++){
          kparam_1g.ghostDim[i] = ghostDim[i];
	  kparam_2g.ghostDim[i] = ghostDim[i];
	  kparam_1g.ghostDim[i] = ghostDim[i];
	  kparam_2g.ghostDim[i] = ghostDim[i];
        }
#else
        hisq_kernel_param_t kparam;
        kparam.D[0] = param.X[0];
        kparam.D[1] = param.X[1];
        kparam.D[2] = param.X[2];
        kparam.D[3] = param.X[3];
        kparam.D1h = param.X[0]/2;
        kparam.threads=param.X[0]*param.X[1]*param.X[2]*param.X[3]/2;
        kparam.base_idx[0]=0;
        kparam.base_idx[1]=0;
        kparam.base_idx[2]=0;
        kparam.base_idx[3]=0;
        kparam_2g.threads = kparam_1g.threads = kparam.threads;

        for(int i=0; i<4; ++i){
          kparam_2g.D[i] = kparam_1g.D[i] = kparam.D[i];
          kparam_2g.D1h  = kparam_1g.D1h  = kparam.D1h;
          kparam_2g.base_idx[i] = kparam_1g.base_idx[i] = 0;
          kparam_2g.ghostDim[i] = kparam_1g.ghostDim[i] = 0;
        }
#endif

        for(int sig=0; sig<8; sig++){
          for(int mu=0; mu<8; mu++){
            if ( (mu == sig) || (mu == OPP_DIR(sig))){
              continue;
            }
            //3-link
            //Kernel A: middle link

            MiddleLink<RealA,RealB> middleLink( link, oprod,  // read only
                sig, mu, mThreeSt,
                Pmu, P3, Qmu, // write only
                newOprod, kparam_2g);
            middleLink.apply(0);
            checkCudaError();

            for(int nu=0; nu < 8; nu++){
              if (nu == sig || nu == OPP_DIR(sig)
                  || nu == mu || nu == OPP_DIR(mu)){
                continue;
              }
              //5-link: middle link
              //Kernel B
              MiddleLink<RealA,RealB> middleLink( link, Pmu, Qmu, // read only
                                                  sig, nu, FiveSt,
                                                  Pnumu, P5, Qnumu, // write only
                                                  newOprod, kparam_1g);
              middleLink.apply(0);
              checkCudaError();

              for(int rho = 0; rho < 8; rho++){
                if (rho == sig || rho == OPP_DIR(sig)
                    || rho == mu || rho == OPP_DIR(mu)
                    || rho == nu || rho == OPP_DIR(nu)){
                  continue;
                }

                //7-link: middle link and side link
                if(FiveSt != 0)coeff = SevenSt/FiveSt; else coeff = 0;

                AllLink<RealA,RealB> allLink(link, Pnumu, Qnumu, sig, rho, SevenSt, coeff, P5, newOprod, kparam_1g);
                allLink.apply(0);
                checkCudaError();
              }//rho

              //5-link: side link
              if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;

              SideLink<RealA,RealB> sideLink(link, P5, Qmu, //read only
                  sig, nu, mFiveSt, coeff,
                  P3, // write only
                  newOprod, kparam_1g);
              sideLink.apply(0);
              checkCudaError();

            } //nu

            //lepage
            if(Lepage != 0.){
              LepageMiddleLink<RealA,RealB>
                lepageMiddleLink ( link, Pmu, Qmu, // read only
                    sig, mu, Lepage,
                    P5, // write only
                    newOprod, kparam_2g);
              lepageMiddleLink.apply(0);
              checkCudaError();

              if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;

              SideLink<RealA, RealB> sideLink(link, P5, Qmu, // read only
                  sig, mu, mLepage, coeff,
                  P3, //write only
                  newOprod, kparam_2g);
              sideLink.apply(0);
              checkCudaError();

            } // Lepage != 0.0

            //3-link side link
            SideLinkShort<RealA,RealB> sideLinkShort(link, P3, // read only
                sig, mu, ThreeSt,
                newOprod, kparam_1g);
            sideLinkShort.apply(0);
            checkCudaError();

          }//mu
        }//sig

        return;
      } // do_hisq_staples_force_cuda


    void hisqCompleteForceCuda(const QudaGaugeParam &param,
        const cudaGaugeField &oprod,
        const cudaGaugeField &link,
        cudaGaugeField* force,
	long long* flops)
    {
      bind_tex_link(link, oprod);

      if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
        CompleteForce<double2,double2> completeForce(link, oprod, *force, param);
        completeForce.apply(0);
        if (flops) *flops = completeForce.flops();
        checkCudaError();
      }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
        CompleteForce<float2,float2> completeForce(link, oprod, *force, param);
        completeForce.apply(0);
        if (flops) *flops = completeForce.flops();
        checkCudaError();
      }else{
          errorQuda("Unsupported precision");
      }

      unbind_tex_link(link, oprod);
    }


    void hisqLongLinkForceCuda(double coeff,
        const QudaGaugeParam &param,
        const cudaGaugeField &oldOprod,
        const cudaGaugeField &link,
        cudaGaugeField  *newOprod,
	long long* flops)
    {
      bind_tex_link(link, *newOprod);
      const int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
      hisq_kernel_param_t kparam;
      for(int i=0; i<4; i++){
        kparam.X[i] = param.X[i];
        kparam.ghostDim[i] = commDimPartitioned(i);
      }
      kparam.threads = volume/2;
      kparam.setStride(param);

      if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
        LongLinkTerm<double2,double2> longLink(link, oldOprod, coeff, *newOprod, kparam);
        longLink.apply(0);
	if(flops) (*flops) = longLink.flops();
        checkCudaError();
      }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
        LongLinkTerm<float2,float2> longLink(link, oldOprod, static_cast<float>(coeff), *newOprod, kparam);
        longLink.apply(0);
	if(flops) (*flops) = longLink.flops();
        checkCudaError();
      }else{
        errorQuda("Unsupported precision");
      }
      unbind_tex_link(link, *newOprod);
      return;
    }

    void
      hisqStaplesForceCuda(const double path_coeff_array[6],
          const QudaGaugeParam &param,
          const cudaGaugeField &oprod,
          const cudaGaugeField &link,
          cudaGaugeField* newOprod,
	  long long* flops)
      {

#ifdef MULTI_GPU
        int X[4] = {
          param.X[0]+4,  param.X[1]+4,  param.X[2]+4,  param.X[3]+4
        };
#else
        int X[4] = {
          param.X[0],  param.X[1],  param.X[2],  param.X[3]
        };
#endif

        // create color matrix fields with zero padding
        int pad = 0;
        GaugeFieldParam gauge_param(X, param.cuda_prec, QUDA_RECONSTRUCT_NO, pad, QUDA_SCALAR_GEOMETRY);

        gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
        gauge_param.siteSubset = QUDA_FULL_SITE_SUBSET;
        gauge_param.order = QUDA_FLOAT2_GAUGE_ORDER;
        cudaGaugeField Pmu(gauge_param);
        cudaGaugeField P3(gauge_param);
        cudaGaugeField P5(gauge_param);
        cudaGaugeField Pnumu(gauge_param);
        cudaGaugeField Qmu(gauge_param);
        cudaGaugeField Qnumu(gauge_param);

        bind_tex_link(link, *newOprod);

        cudaEvent_t start, end;

        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        if (param.cuda_prec == QUDA_DOUBLE_PRECISION){

          PathCoefficients<double> act_path_coeff;
          act_path_coeff.one    = path_coeff_array[0];
          act_path_coeff.naik   = path_coeff_array[1];
          act_path_coeff.three  = path_coeff_array[2];
          act_path_coeff.five   = path_coeff_array[3];
          act_path_coeff.seven  = path_coeff_array[4];
          act_path_coeff.lepage = path_coeff_array[5];
          do_hisq_staples_force_cuda<double,double2,double2>( act_path_coeff,
              param,
              oprod,
              link,
              Pmu,
              P3,
              P5,
              Pnumu,
              Qmu,
              Qnumu,
              *newOprod);


        }else if(param.cuda_prec == QUDA_SINGLE_PRECISION){
          PathCoefficients<float> act_path_coeff;
          act_path_coeff.one    = path_coeff_array[0];
          act_path_coeff.naik   = path_coeff_array[1];
          act_path_coeff.three  = path_coeff_array[2];
          act_path_coeff.five   = path_coeff_array[3];
          act_path_coeff.seven  = path_coeff_array[4];
          act_path_coeff.lepage = path_coeff_array[5];

          do_hisq_staples_force_cuda<float,float2,float2>( act_path_coeff,
              param,
              oprod,
              link,
              Pmu,
              P3,
              P5,
              Pnumu,
              Qmu,
              Qnumu,
              *newOprod);
        }else{
          errorQuda("Unsupported precision");
        }


        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float runtime;
        cudaEventElapsedTime(&runtime, start, end);

	if(flops){
	  int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
	  // Middle Link, side link, short side link, AllLink, OneLink
	  *flops = (134784 + 24192 + 103680 + 864 + 397440 + 72);

	  if(path_coeff_array[5] != 0.) *flops += 28944; // Lepage contribution
	  *flops *= volume;
	}

        unbind_tex_link(link, *newOprod);

        cudaEventDestroy(start);
        cudaEventDestroy(end);

        return;
      }

#endif

  } // namespace fermion_force
} // namespace quda

#endif // GPU_HISQ_FORCE

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>

namespace quda
{
  // This function gets stap = S_{mu,nu} i.e., the staple of length 3.
  //
  // |- > -|                /- > -/                /- > -
  // ^     v               ^     v                ^
  // |     |              /     /                /- < -
  //         + |     |  +         +  /     /  +         +  - > -/
  //           v     ^              v     ^                    v
  //           |- > -|             /- > -/                - < -/
  // Each dimension requires 2 matrix additions, 4 matrix-matrix multiplications
  // matrix*matrix = 9*3*6 + 9*2*2 = 198 floating-point ops
  // matrix+matrix = 18 floating-point ops
  // => Total number of floating point ops per function call
  // dims * (2*18 + 4*198) = dims*828
  template <typename Arg, typename Link>
  __host__ __device__ void computeStaple(Arg &arg, const int *x, const int *X, const int parity, const int dir, Link &staple, const int dir_ignore)
  {    
    setZero(&staple);
    for (int mu = 0; mu < 4 ; mu++) {
      // Identify directions orthogonal to the link and 
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (mu != dir && mu != dir_ignore) {
	
        int nu = dir;
        {
          int dx[4] = {0, 0, 0, 0};
          Link U1, U2, U3;

          // Get link U_{\mu}(x)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[mu]++;
          // Get link U_{\nu}(x+\mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]--;
          dx[nu]++;
          // Get link U_{\mu}(x+\nu)
          U3 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          // staple += U_{\mu}(x) * U_{\nu}(x+\mu) * U^\dag_{\mu}(x+\nu)
          staple = staple + U1 * U2 * conj(U3);

          dx[mu]--;
          dx[nu]--;
          // Get link U_{\mu}(x-\mu)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
          // Get link U_{\nu}(x-\mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          dx[nu]++;
          // Get link U_{\mu}(x-\mu+\nu)
          U3 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          // staple += U^\dag_{\mu}(x-\mu) * U_{\nu}(x-\mu) * U_{\mu}(x-\mu+\nu)
          staple = staple + conj(U1) * U2 * U3;
        }
      }
    }
  }

  // Each dimension requires 8 matrix additions and 28 matrix-matrix multiplications
  // matrix*matrix = 9*3*6 + 9*2*2 = 198 floating-point ops
  // matrix+matrix = 18 floating-point ops
  // => Total number of floating point ops per function call
  // dims * (8*18 + 28*198) = dims*5688
  template <typename Arg, typename Link>
  __host__ __device__ void computeStapleRectangle(Arg &arg, const int *x, const int *X, const int parity, const int dir, Link &staple, Link &rectangle, const int dir_ignore)
  {
    setZero(&staple);
    setZero(&rectangle);
    for (int mu = 0; mu < 4; mu++) {      
      // Identify directions orthogonal to the link.
      // Over-Improved stout is usually done for topological
      // measuremnts which will include the temporal direction.
      if (mu != dir && mu != dir_ignore) {
        int nu = dir;

        // RECTANGLE calculation
        // This is done in three parts. For some link U_nu(x) there are
        // 1x2 rectangles (R12) and two sets of 2x1 rectangles, defined as
        // 'forward' (R21f)  and 'backward' (R21b).

        // STAPLE calculation
        // This is done part way through the computation of (R21f) as the
        // First two links of the staple are already in memory.

        // Memory usage and communications.
        // There are 10 unique links to be fetched per direction. 3 of these
        // links (the ones that form the simple staple) can be recycled on
        // the fly. The two links immediately succeeding and preceding
        // U_nu(x) in the nu directon are also reused when changing from
        // +ve to -ve mu.

        {
          int dx[4] = {0, 0, 0, 0};
          Link U1, U2, U3, U4, U5, U6, U7;

          //--------//
          // +ve mu //
          //--------//

          //----------------------------------------------------------------
          // R12 = U_mu(x)*U_mu(x+mu)*U_nu(x+2mu)*U^d_mu(x+nu+mu)*U^d_mu(x+nu)
          // Get link U_mu(x)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[mu]++;
          // Get link U_mu(x+mu)
          U2 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]++;
          // Get link U_nu(x+2mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[mu]--;
          dx[nu]++;
          // Get link U_mu(x+nu+mu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[mu]--;
          // Get link U_mu(x+nu)
          U5 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          rectangle = rectangle + U1 * U2 * U3 * conj(U4) * conj(U5);
          //---------------------------------------------------------------

          // reset dx
          dx[nu]--;
          //---------------------------------------------------------------
          // R21f=U_mu(x)*U_nu(x+mu)*U_nu(x+nu+mu)*U^d_mu(x+2nu)*U^d_nu(x+nu)
          // Get link U_mu(x)
          // Same as U1 from R12

          dx[mu]++;
          // Get link U_nu(x+mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          ///////////////////////////////////////////////////////
          // Here we get the third link in the staple and compute.
          // Get U_mu(x+nu)
          // Same as U5 from R12
          staple = staple + U1 * U2 * conj(U5);
          ///////////////////////////////////////////////////////

          dx[nu]++;
          // Get link U_nu(x+nu+mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[mu]--;
          dx[nu]++;
          // Get link U_mu(x+2nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          dx[nu]--;
          // Get link U_nu(x+nu)
          U6 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          rectangle = rectangle + U1 * U2 * U3 * conj(U4) * conj(U6);
          //---------------------------------------------------------------

          // reset dx
          dx[nu]--;
          //---------------------------------------------------------------
          // R21b=U^d_nu(x-nu)*U_mu(x-nu)*U_nu(x+nu+mu)*U^d_mu(x+2nu)*U^dag_nu(x+nu)

          // Get link U_nu(x-nu)
          dx[nu]--;
          U7 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_mu(x-nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_nu(x-nu+mu)
          dx[mu]++;
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          // Get link U_nu(x+mu)
          // Same as U2 from R21f

          // Get link U_mu(x+nu)
          // Same as U5 from R12

          rectangle = rectangle + conj(U7) * U4 * U3 * U2 * conj(U5);
          //---------------------------------------------------------------

          //--------//
          // -ve mu //
          //--------//

          // reset dx
          dx[mu]--;
          dx[nu]++;
          //---------------------------------------------------------------
          // R12 = U^dag_mu(x-mu) * U^dag_mu(x-2mu) * U_nu(x-2mu) * U_mu(x-2mu+nu) * U_mu(x-mu+nu)

          dx[mu]--;
          // Get link U_mu(x-mu)
          U1 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]--;
          // Get link U_mu(x-2mu)
          U2 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          // Get link U_nu(x-2mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[nu]++;
          // Get link U_mu(x-2mu+nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          dx[mu]++;
          // Get link U_mu(x-mu+nu)
          U5 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          rectangle = rectangle + conj(U1) * conj(U2) * U3 * U4 * U5;
          //---------------------------------------------------------------

          // reset dx
          dx[mu]++;
          dx[nu]--;
          //---------------------------------------------------------------
          // R21f = U^dag_mu(x-mu) * U_nu(x-mu) * U_nu(x-mu+nu) * U_mu(x-mu+2nu) * U^dag_nu(x+nu)

          // Get link U_mu(x-mu)
          // Same as U1 from R12

          dx[mu]--;
          // Get link U_nu(x-mu)
          U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          ///////////////////////////////////////////////////////
          // Here we get the third link in the staple and compute.
          // Get U_mu(x-mu+nu)
          // Same as U5 from R12
          staple = staple + conj(U1) * U2 * U5;
          ///////////////////////////////////////////////////////

          dx[nu]++;
          // Get link U_nu(x-mu+nu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          dx[nu]++;
          // Get link U_mu(x-mu+2nu)
          U4 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_nu(x+nu)
          // Same as U6 from +ve R21f

          rectangle = rectangle + conj(U1) * U2 * U3 * U4 * conj(U6);
          //---------------------------------------------------------------

          // reset dx
          dx[nu]--;
          dx[nu]--;
          dx[mu]++;
          //---------------------------------------------------------------
          // R21b= U^dag_nu(x-nu) * U^dag_mu(x-mu-nu) * U_nu(x-mu-nu) * U_nu(x-mu) * U_mu(x-mu+nu)

          // Get link U_nu(x-nu)
          // Same as U7 from +ve R21b

          // Get link U_mu(x-mu-nu)
          dx[nu]--;
          dx[mu]--;
          U4 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          // Get link U_nu(x-nu-mu)
          U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

          // Get link U_nu(x-mu)
          // Same as U2 from R21f

          // Get link U_mu(x-mu+nu)
          // Same as U5 from R12

          rectangle = rectangle + conj(U7) * conj(U4) * U3 * U2 * U5;
          //---------------------------------------------------------------
        }
      }
    }
  }
}

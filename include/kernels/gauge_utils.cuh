#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <thread_array.h>

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
  template <typename Arg, typename Staple, typename Int>
  __host__ __device__ inline void computeStaple(const Arg &arg, const int *x, const Int *X, const int parity, const int nu, Staple &staple, const int dir_ignore)
  {
    using Link = typename get_type<Staple>::type;
    staple = Link();

    thread_array<int, 4> dx = { };
#pragma unroll
    for (int mu = 0; mu < 4 ; mu++) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (mu != nu && mu != dir_ignore) {
        {
          // Get link U_{\mu}(x)
          Link U1 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          // Get link U_{\nu}(x+\mu)
          dx[mu]++;
          Link U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);
          dx[mu]--;

          // Get link U_{\mu}(x+\nu)
          dx[nu]++;
          Link U3 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
          dx[nu]--;

          // staple += U_{\mu}(x) * U_{\nu}(x+\mu) * U^\dag_{\mu}(x+\nu)
          staple = staple + U1 * U2 * conj(U3);
        }

        {
          // Get link U_{\mu}(x-\mu)
          dx[mu]--;
          Link U1 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
          // Get link U_{\nu}(x-\mu)
          Link U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_{\mu}(x-\mu+\nu)
          dx[nu]++;
          Link U3 = arg.in(mu, linkIndexShift(x, dx, X), parity);

          // reset dx
          dx[nu]--;
          dx[mu]++;

          // staple += U^\dag_{\mu}(x-\mu) * U_{\nu}(x-\mu) * U_{\mu}(x-\mu+\nu)
          staple = staple + conj(U1) * U2 * U3;
        }
      }
    }
  }

  // This function will get the three 2x1 rectangles and the central 1x1 square
  // that define the Symanzik staple around the link.
  //
  // ----<----
  // |       |
  // V       ^  <- Forward Staple R21f 
  // |       | 
  // ----<---+---<----
  // x       |       |
  // U       ^       ^  <- Side Staple R21s
  // |       |       |
  // ---->---+--->----
  // |       |
  // V       ^  <- Backward Staple R21b
  // |       |
  // ---->----
  //
  // Each dimension requires 8 matrix additions and 28 matrix-matrix multiplications
  // matrix*matrix = 9*3*6 + 9*2*2 = 198 floating-point ops
  // matrix+matrix = 18 floating-point ops
  // => Total number of floating point ops per function call
  // dims * (8*18 + 28*198) = dims*5688
  template <typename Arg, typename Staple, typename Rectangle, typename Int>
  __host__ __device__ inline void computeStapleRectangle(const Arg &arg, const int *x, const Int *X, const int parity, const int nu,
                                                         Staple &staple, Rectangle &rectangle, const int dir_ignore)
  {
    using Link = typename get_type<Staple>::type;
    staple = Link();
    rectangle = Link();

    thread_array<int, 4> dx = { };
    for (int mu = 0; mu < 4; mu++) { // do not unroll loop to prevent register spilling
      // Identify directions orthogonal to the link.
      // Over-Improved stout is usually done for topological
      // measurements which will include the temporal direction.
      if (mu != nu && mu != dir_ignore) {
        // RECTANGLE calculation
        // This is done in three parts. For some link U_nu(x) there are
        // 1x2 rectangles (R12) and two sets of 2x1 rectangles, defined as
        // 'forward' (R21f)  and 'backward' (R21b).

        // STAPLE calculation
        // This is done part way through the computation of (R21f) as the
        // First two links of the staple are already in memory.

        // Memory usage and communications.
        // There are 12 unique links to be fetched per direction. 3 of these
        // links (the ones that form the simple staple) can be recycled on
        // the fly. The two links immediately succeeding and preceding
        // U_nu(x) in the nu direction are reloaded when switching from
        // +ve to -ve mu to reduce the stack frame.

        //--------//
        // +ve mu //
        //--------//
	{
	  // Accumulate backward staple in U1
	  dx[nu]--; //0,-1
	  // Get link U_nu(x-nu)
	  Link U1 = conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  // Get link U_mu(x-nu)
	  U1 = U1 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	  dx[mu]++; //1,-1
	  // Get link U_nu(x-nu+mu)
	  U1 = U1 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity));
	  
	
	  // Get links U_nu(x+mu) and U_mu(x+nu)
	  dx[nu]++; //1,0
	  Link U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);
	  dx[nu]++; //1,1
	  dx[mu]--; //0,1
	  Link U3 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
	
	  // Complete R12b
	  rectangle = rectangle + U1 * U2 * conj(U3);
	  
	  // Get link U_mu(x)
	  dx[nu]--; //0,0
	  U1 = arg.in(mu, linkIndexShift(x, dx, X), parity);
	  
	  //Complete Wilson staple
	  staple = staple + U1 * U2 * conj(U3);
	  
	  dx[mu]++; //1,0
	  dx[nu]++; //1,1
	  // Accumulate forward staple in U2
	  U2 = U1 * U2;
	  // Get link U_nu(x+mu+nu)
	  U2 = U2 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity));
	  dx[nu]++; //1,2
	  dx[mu]--; //0,2
	  // Get link U_mu(x+nu)
	  U2 = U2 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity)));
	  dx[nu]--; //0,1
	  // Get link U_nu(x+nu)
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  
	  // complete R21f
	  rectangle = rectangle + U2;
	  
	  dx[nu]--; //0,0
	  U2 = U1;
	  dx[mu]++; //1,0
	  // Accumulate side staple in U2
	  // Get link U_mu(x+mu)
	  U2 = U2 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	  dx[mu]++; //2,0
	  // Get link U_nu(x+2mu)
	  U2 = U2 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity));
	  dx[nu]++; //2,1
	  dx[mu]--; //1,1
	  // Get link U_mu(x+mu+nu)
	  U2 = U2 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity)));
	  
	  // Complete R21s
	  rectangle = rectangle + U2 * conj(U3);
	}
        //--------//
        // -ve mu //
        //--------//
	{
	  // reset dx
	  dx[nu]--; //1,0
	  dx[mu]--; //0,0
	  
	  // Accumulate backward staple in U1
	  dx[nu]--; //0,-1
	  // Get link U_nu(x-nu)
	  Link U1 = conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  dx[mu]--; //-1,-1
	  // Get link U_mu(x-nu-mu)
	  U1 = U1 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity)));
	  // Get link U_nu(x-nu-mu)
	  U1 = U1 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity));


	  dx[nu]++; //-1,0
	  // Get links U_nu(x+mu) and U_mu(x+nu)
	  Link U2 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);
	  dx[nu]++; //-1,1
	  Link U3 = arg.in(mu, linkIndexShift(x, dx, X), parity);
	
	  // Complete R12b
	  rectangle = rectangle + U1 * U2 * U3;
	  
	  dx[nu]--; //-1,0
	  // Get link U_mu(x-mu)
	  U1 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
	  
	  // Complete Wilson staple
	  staple = staple + conj(U1) * U2 * U3;
	  
	  dx[nu]++; //-1,1
	  // Accumulate forward staple in U2
	  U2 = conj(U1) * U2;
	  U2 = U2 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity));
	  dx[nu]++; //-1,2
	  U2 = U2 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	  dx[mu]++; //0,2
	  dx[nu]--; //0,1
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  
	  // Complete R21f
	  rectangle = rectangle + U2;
	  
	  dx[nu]--; //0,0
	  dx[mu]--; //-1,0
	  dx[mu]--; //-2,0
	  // Accumulate side staple in U2
	  U2 = conj(U1);
	  U2 = U2 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity)));
	  U2 = U2 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity));
	  dx[nu]++; //-2,1
	  U2 = U2 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  // Complete R21s
	  rectangle = rectangle + U2 * U3;
	  
	  //reset dx
	  dx[nu]--; //-2,0
	  dx[mu]++; //-1,0
	  dx[mu]++; //0,0
	}
      }
    }
  }
}

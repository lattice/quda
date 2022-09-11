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
  // Each mu requires 2 matrix additions and 4 matrix-matrix multiplications
  // Force FLOPs = 3 * (4*(6*3 + 4)*9 + 2*18) = 2484
  // Smear FLOPs = 2 * (4*(6*3 + 4)*9 + 2*18) = 1656
  // Smear version uses forward paths
  template <typename Arg, typename Link, typename Int>
  __host__ __device__ inline void computeStaple(const Arg &arg, const int *x, const Int *X, const int parity, const int nu, Link &staple, const int dir_ignore)
  {
    setZero(&staple);
    thread_array<int, 4> dx = { };
#pragma unroll
    for (int mu = 0; mu < 4 ; mu++) {
      // Identify directions orthogonal to the link.
      // Over-Improved stout is usually done for topological
      // measurements which will include the temporal direction.
      if (mu != nu && mu != dir_ignore) {	
	{
          // Get link U_{\mu}(x)
          Link U1 = static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));
	  
          // Get link U_{\nu}(x+\mu)
          dx[mu]++;
          U1 =  U1 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));
          dx[mu]--;

          // Get link U_{\mu}(x+\nu)
          dx[nu]++;
          U1 = U1 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity)));
          dx[nu]--;
	  
          // staple += U_{\mu}(x) * U_{\nu}(x+\mu) * U^\dag_{\mu}(x+\nu)
          staple = staple + U1;
	}

	{
          // Get link U_{\mu}(x-\mu)
          dx[mu]--;
          Link U1 = conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity)));
          // Get link U_{\nu}(x-\mu)
          U1 = U1 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));

          // Get link U_{\mu}(x-\mu+\nu)
          dx[nu]++;
          U1 = U1 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));
	  
          // reset dx
          dx[nu]--;
          dx[mu]++;

          // staple += U^\dag_{\mu}(x-\mu) * U_{\nu}(x-\mu) * U_{\mu}(x-\mu+\nu)
          staple = staple + U1;
        }
      }
    }
  }

  // Force version is reversed path of Smear version
  template <typename Arg, typename Link, typename Int>
  __host__ __device__ inline void computeForceStaple(Arg &arg, const int *x, const Int *X, const int parity, const int nu, Link &staple)
  {
    setZero(&staple);
    int dx[4] = {0,0,0,0};
#pragma unroll
    for (int mu = 0; mu < 4 ; mu++) {
      // Identify directions orthogonal to the link
      if (mu != nu) {
	{
	  // Positive mu
	  //-------------------------------------------------------------------
	  // Start from the end of the link
	  dx[nu]++;
	  // Get link U_{\mu}(x+nu)
	  Link U1 = static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	
	  // Get link U_{\nu}(x+\mu)
	  dx[mu]++;
	  dx[nu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  
	  // Get link U_{\mu}(x)
	  dx[mu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity)));
	  
	  // staple += U_{\mu}(x) * U_{\nu}(x+\mu) * U^\dag_{\mu}(x+\nu)
	  staple = staple + U1;
	  //-------------------------------------------------------------------
	}
	
	{
	  // Negative mu
	  //-------------------------------------------------------------------
	  dx[nu]++;
	  // Get link U_{\mu}(x+\nu-\mu)
	  dx[mu]--;
	  Link U1 = conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity)));
	  // Get link U_{\nu}(x-\mu)
	  dx[nu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  
	  // Get link U_{\mu}(x-\mu)
	  U1 = U1 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  // reset dx
	  dx[mu]++;
	  
	  // staple += U^\dag_{\mu}(x-\mu) * U_{\nu}(x-\mu) * U_{\mu}(x-\mu+\nu)
	  staple = staple + U1;
	  //-------------------------------------------------------------------
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
  // Each mu requires 8 matrix additions and 28 matrix-matrix multiplications
  // This routine will perfrom 28 matrix multiplications and 8 matrix additions
  // Force FLOPs = 3 * (28*(6*3 + 4)*9 + 8*18) = 16,740
  // Smear FLOPS = 2 * (28*(6*3 + 4)*9 + 8*18) = 11,160
  // Smear version uses forward paths
  template <typename Arg, typename Link, typename Int>
  __host__ __device__ inline void computeStapleRectangle(const Arg &arg, const int *x, const Int *X, const int parity, const int nu,
                                                         Link &staple, Link &rectangle, const int dir_ignore)
  {
    setZero(&staple);
    setZero(&rectangle);
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
	  dx[mu]++; // 0,0
	}
      }
    }
  }

  // Force version uses reverse paths
  template <typename Arg, typename Link, typename Int>
  __host__ __device__ inline void computeForceStapleRectangle(Arg &arg, const int *x, const Int *X, const int parity, const int nu,
							      Link &staple, Link &rectangle)
  {
    setZero(&staple);
    setZero(&rectangle);

    for (int mu = 0; mu < 4; mu++) { // do not unroll loop to prevent register spilling
      // Identify directions orthogonal to the link.
      if (mu != nu) {
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
	
	{
	  // Positive mu
	  //-------------------------------------------------------------------
	  int dx[4] = {0,0,0,0};
	  // Accumulate backward staple
	  //---------------------------
	  // Get link U_mu(x), preserve
	  Link U3 = static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));	  
	  
	  // Start from the end of the link
	  dx[nu]++;

	  // Get link U_mu(x+nu) preserve
	  Link U1 = static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  // Get link U_nu(x+nu)
	  dx[mu]++;
	  dx[nu]--;
	  Link U2 = static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  //Complete Wilson staple
	  staple = staple + U1 * conj(U2) * conj(U3);

	  // Accumulate the rear rectangle in U2
	  U2 = U1 * conj(U2);
	  // Get link U_nu(x-nu-mu)
	  dx[nu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity)));
	  
	  // Get links U_nu(x-mu) and U_mu(x-nu)
	  dx[mu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity)));
	  U2 = U2 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));
	
	  // Complete R12b
	  rectangle = rectangle + U2;
	  
	  // Return to end of link
	  dx[nu] += 2;

	  // Accumulate forward staple
	  //--------------------------	  
	  // Get link U_nu(x+nu)
	  U2 = static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  // Get link U_mu(x+2nu)
	  dx[nu]++;
	  U2 = U2 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));
	  
	  // Get link U_nu(x+nu+mu)
	  dx[mu]++;
	  dx[nu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity)));
	  
	  // Get link U_nu(x+mu)
	  dx[nu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  
	  // Complete R12f
	  U2 = U2 * conj(U3);
	  rectangle = rectangle + U2;

	  // Return to end of link
	  dx[mu]--;
	  dx[nu]++;

	  // Accumulate side staple
	  U2 = U1;
	  
	  // Get link U_mu(x+mu+nu)
	  dx[mu]++;
	  U2 = U2 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));

	  // Get link U_nu(x+2mu)
	  dx[mu]++;
	  dx[nu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity)));

	  // Get link U_mu(x+mu)
	  dx[mu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity)));

	  // Complete side rectangle
	  rectangle = rectangle + U2 * conj(U3);

	  // Return to end of link
	  dx[mu]--;
	  dx[nu]++;	  
	}
	

	{
	  // Negative mu
	  //-------------------------------------------------------------------
	  int dx[4] = {0,0,0,0};
	  // Accumulate backward staple
	  //---------------------------
	  // Get link U_mu(x+nu+mu)
	  dx[nu]++;
	  dx[mu]--;
	  Link U1 = static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));

	  // Get link U_nu(x-mu)
	  dx[nu]--;
	  Link U2 = static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  // Get link U_mu(x-mu)
	  Link U3 = static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  //Complete Wilson staple
	  staple = staple + conj(U1) * conj(U2) * U3;

	  // Accumulate the rear rectangle in U2	  
	  U2 = conj(U1) * conj(U2);

	  // Get links U_nu(x-mu-nu) and U_mu(x-nu-mu)
	  dx[nu]--; 
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity)));
	  U2 = U2 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));

	  // Get link U_nu(x-nu)
	  dx[mu]++; 
	  U2 = U2 * static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));
	  
	  // Complete R12b
	  rectangle = rectangle + U2;
	  
	  // Return to end of link
	  dx[nu] += 2;

	  // Accumulate forward staple
	  //--------------------------	  
	  // Get link U_nu(x+nu)
	  U2 = static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity));

	  // Get link U_mu(x+nu+mu)
	  dx[nu]++;
	  dx[mu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity)));
	  
	  // Get link U_nu(x+nu+mu)
	  dx[nu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity)));

	  // Get link U_nu(x+mu)
	  dx[nu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), 1 - parity)));
	  
	  // Complete R12f
	  U2 = U2 * U3;
	  rectangle = rectangle + U2;

	  // Return to end of link
	  dx[mu]++;
	  dx[nu]++;

	  // Accumulate side staple
	  U2 = conj(U1);
	  
	  // Get link U_mu(x-mu+nu)
	  dx[mu] -= 2;
	  U2 = U2 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity)));

	  // Get link U_nu(x-2mu) and U_mu(x-2mu)
	  dx[nu]--;
	  U2 = U2 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dx, X), parity)));
	  U2 = U2 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity));
	  
	  // Complete side rectangle
	  rectangle = rectangle + U2 * U3;
	}
      }
    }
  }

  
  // This function will get the four 1x1x1 "twisted" parallelograms that are needed
  // for the Leuscher-Weisz gauge action 
  // M. Luscher and P. Weisz, Nucl. Phys. B266, 309 (1986)
  //
  // For a link in the nu direction there are 3 orthogonal mu dimensions. There are
  // two remaining dimensions over which to compute the parallelogram (or twisted)
  // staple which yields 8 staples per nu, mu combination
  //
  //  link   loop 
  //   nu :   mu   +rho -nu -mu -rho
  //   nu :   mu   -rho -nu -mu +rho
  //   nu :   mu   +tau -nu -mu -rho
  //   nu :   mu   -tau -nu -mu +rho
  //   nu :  -mu   +rho -nu -mu -rho
  //   nu :  -mu   -rho -nu -mu +rho
  //   nu :  -mu   +tau -nu -mu -rho
  //   nu :  -mu   -tau -nu -mu +rho
  
  // Each mu requires 32 matrix multiplications and 7 matrix additions
  // Force FLOPs = 3 * (32*(6*3 + 4)*9 + 7*18) = 19,386
  // This kernel appear in force terms only
  template <typename Arg, typename Link, typename Int>
  __host__ __device__ inline void computeForceParallelogram(Arg &arg, const int *x, const Int *X, const int parity, const int nu, Link &staple)
  {
    setZero(&staple);
    int mu = 0, i = 0;
    
#pragma unroll
    for (mu = 0; mu < 4; mu++) { 
      // Identify directions orthogonal to the link.
      // Computations are done relative to the start of the U(nu) link, and we always
      // start with the U(mu) or U(-mu) link 
      if (mu != nu) {
	// Deduce the remaining dimensions
	// This is slightly inefficient because rho will be set twice in this loop and
	// tau will be set once.
	int rhotau[2] = {0,0};
	for (i = 0; i < 4; i++) 
	  if (i != nu && i != mu) rhotau[0] = i;
	
	for (i = 0; i < 4; i++) 
	  if (i != nu && i != mu && i != rhotau[0]) rhotau[1] = i;
	
	// Positive mu.
	//---------------------------------------------------------------------
	int dx[4] = {0,0,0,0};
	dx[nu]++;
	Link Umu = static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), 1 - parity));
	Link U1 = Umu;
	// Positive rho,tau
	for(i=0; i<2; i++) {

	  int dxp[4] = {0,0,0,0};
	  dxp[nu]++;
	  dxp[mu]++;
	  U1 = Umu;
	  U1 = U1 * static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxp, X), parity));
	  dxp[rhotau[i]]++;
	  dxp[nu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dxp, X), parity)));
	  dxp[mu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dxp, X), 1 - parity)));
	  dxp[rhotau[i]]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxp, X), parity)));
	  staple += U1;	  

	}

	// Negative rho,tau
	for(i=0; i<2; i++) {
	  int dxp[4] = {0,0,0,0};
	  dxp[nu]++;
	  dxp[mu]++;
	  U1 = Umu;
	  dxp[rhotau[i]]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxp, X), 1 - parity)));	    
	  dxp[nu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dxp, X), parity)));
	  dxp[mu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dxp, X), 1 - parity)));
	  U1 = U1 * static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxp, X), 1 - parity));	  
	  staple += U1;
	}
	
	// Negative mu.
	//---------------------------------------------------------------------
	dx[mu]--;
	Umu = conj(static_cast<Link>(arg.in(mu, linkIndexShift(x, dx, X), parity)));
	
	//---------------------------------------------------------------------
	// Positive rho,tau
	for(i=0; i<2; i++) {
	  int dxn[4] = {0,0,0,0};
	  dxn[nu]++;
	  dxn[mu]--;

	  U1 = Umu;
	  U1 = U1 * static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxn, X), parity));
	  dxn[rhotau[i]]++;
	  dxn[nu]--;
	  
	  U1 = U1 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dxn, X), parity)));
	  U1 = U1 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dxn, X), parity));
	  dxn[mu]++;
	  dxn[rhotau[i]]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxn, X), parity)));
	  staple += U1;	  
	}
	
	// Negative rho,tau
	for(i=0; i<2; i++) {
	  int dxn[4] = {0,0,0,0};
	  dxn[nu]++;
	  dxn[mu]--;
	  
	  U1 = Umu;
	  dxn[rhotau[i]]--;	  
	  U1 = U1 * conj(static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxn, X), 1 - parity)));	    
	  dxn[nu]--;
	  U1 = U1 * conj(static_cast<Link>(arg.in(nu, linkIndexShift(x, dxn, X), parity)));
	  U1 = U1 * static_cast<Link>(arg.in(mu, linkIndexShift(x, dxn, X), parity));
	  dxn[mu]++;
	  U1 = U1 * static_cast<Link>(arg.in(rhotau[i], linkIndexShift(x, dxn, X), 1 - parity));
	  
	  staple += U1;
	}	
      }
    }
  }
}

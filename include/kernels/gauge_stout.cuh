#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>

namespace quda {

  template <typename Float, typename GaugeOr, typename GaugeDs>
  struct GaugeSTOUTArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    int border[4];
    GaugeOr origin;
    const Float rho;
    const Float tolerance;

    GaugeDs dest;

    GaugeSTOUTArg(GaugeOr &origin, GaugeDs &dest, const GaugeField &data, const Float rho, const Float tolerance)
      : threads(1), origin(origin), dest(dest), rho(rho), tolerance(tolerance) {
      for ( int dir = 0; dir < 4; ++dir ) {
        border[dir] = data.R()[dir];
        X[dir] = data.X()[dir] - border[dir] * 2;
	threads *= X[dir];
      }
      threads /= 2;
    }
  };


  template <typename Float, typename Arg, typename Link>
  __host__ __device__ void computeStaple(Arg &arg, int idx, int parity, int dir, Link &staple) {

    // compute spacetime dimensions and parity
    int X[4];
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for(int dr=0; dr<4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2*arg.border[dr];
    }

    setZero(&staple);

    // I believe most users won't want to include time staples in smearing
    for (int mu=0; mu<3; mu++) {

      //identify directions orthogonal to the link.
      if (mu != dir) {

	int nu = dir;
	{
	  int dx[4] = {0, 0, 0, 0};
	  Link U1, U2, U3;

	  //Get link U_{\mu}(x)
	  U1 = arg.origin(mu, linkIndexShift(x,dx,X), parity);

	  dx[mu]++;
	  //Get link U_{\nu}(x+\mu)
	  U2 = arg.origin(nu, linkIndexShift(x,dx,X), 1-parity);

	  dx[mu]--;
	  dx[nu]++;
	  //Get link U_{\mu}(x+\nu)
	  U3 = arg.origin(mu, linkIndexShift(x,dx,X), 1-parity);

	  // staple += U_{\mu}(x) * U_{\nu}(x+\mu) * U^\dag_{\mu}(x+\nu)
	  staple = staple + U1 * U2 * conj(U3);

	  dx[mu]--;
	  dx[nu]--;
	  //Get link U_{\mu}(x-\mu)
	  U1 = arg.origin(mu, linkIndexShift(x,dx,X), 1-parity);
	  //Get link U_{\nu}(x-\mu)
	  U2 = arg.origin(nu, linkIndexShift(x,dx,X), 1-parity);

	  dx[nu]++;
	  //Get link U_{\mu}(x-\mu+\nu)
	  U3 = arg.origin(mu, linkIndexShift(x,dx,X), parity);

	  // staple += U^\dag_{\mu}(x-\mu) * U_{\nu}(x-\mu) * U_{\mu}(x-\mu+\nu)
	  staple = staple + conj(U1) * U2 * U3;
	}
      }
    }
  }

  template<typename Float, typename Arg>
  __global__ void computeSTOUTStep(Arg arg)
  {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    int dir = threadIdx.z + blockIdx.z*blockDim.z;
    if (idx >= arg.threads) return;
    if (dir >= 3) return;
    typedef complex<Float> Complex;
    typedef Matrix<complex<Float>,3> Link;

    int X[4];
    for (int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
    for (int dr=0; dr<4; ++dr) {
      x[dr] += arg.border[dr];
      X[dr] += 2*arg.border[dr];
    }

    int dx[4] = {0, 0, 0, 0};
    //Only spatial dimensions are smeared
    {
      Link U, UDag, Stap, Omega, OmegaDiff, ODT, Q, exp_iQ;
      Complex OmegaDiffTr;
      Complex i_2(0,0.5);

      //This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple<Float>(arg,idx,parity,dir,Stap);
      //
      // |- > -|                /- > -/                /- > -
      // ^     v               ^     v                ^
      // |     |              /     /                /- < -
      //         + |     |  +         +  /     /  +         +  - > -/
      //           v     ^              v     ^                    v
      //           |- > -|             /- > -/                - < -/

      // Get link U
      U = arg.origin(dir, linkIndexShift(x,dx,X), parity);

      //Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag

      //Get U^{\dagger}
      UDag = inverse(U);

      //Compute \Omega = \rho * S * U^{\dagger}
      Omega = (arg.rho * Stap) * UDag;

      //Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]

      OmegaDiff = conj(Omega) - Omega;

      Q = OmegaDiff;
      OmegaDiffTr = getTrace(OmegaDiff);
      OmegaDiffTr = (1.0/3.0) * OmegaDiffTr;

      //Matrix proportional to OmegaDiffTr
      setIdentity(&ODT);

      Q = Q - OmegaDiffTr * ODT;
      Q = i_2 * Q;
      //Q is now defined.

#if 0
      //Test for Tracless:
      //reuse OmegaDiffTr
      OmegaDiffTr = getTrace(Q);
      double error;
      error = OmegaDiffTr.real();
      printf("Trace test %d %d %.15e\n", idx, dir, error);

      //Test for hemiticity:
      Link Q_diff = conj(Q);
      Q_diff -= Q; //This should be the zero matrix. Test by ReTr(Q_diff^2);
      Q_diff *= Q_diff;
      //reuse OmegaDiffTr
      OmegaDiffTr = getTrace(Q_diff);
      error = OmegaDiffTr.real();
      printf("Herm test %d %d %.15e\n", idx, dir, error);
#endif

      exponentiate_iQ(Q,&exp_iQ);

#if 0
      //Test for expiQ unitarity:
      error = ErrorSU3(exp_iQ);
      printf("expiQ test %d %d %.15e\n", idx, dir, error);
#endif

      U = exp_iQ * U;
#if 0
      //Test for expiQ*U unitarity:
      error = ErrorSU3(U);
      printf("expiQ*u test %d %d %.15e\n", idx, dir, error);
#endif

      arg.dest(dir, linkIndexShift(x,dx,X), parity) = U;
    }
  }

} // namespace quda

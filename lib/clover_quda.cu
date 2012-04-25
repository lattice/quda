#include <quda_internal.h>
#include <quda_matrix.h>
#include <clover_field.h>
#include <gauge_field.h>

namespace quda {

  struct CloverParam {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    int gaugeLengthCB; // half length (include checkerboard spacetime, color, complex, direction)
    int gaugeStride; // stride used on gauge field
    int FmunuLengthCB; // half length (include checkerboard spacetime, color, complex, direction)
    int FmunuStride; // stride used on Fmunu field
  };
  
  /**
   linkIndex computes the spacetime index of the link with coordinate
   y = x + dx.

   @param x - coordinate in spacetime
   @param dx - coordinate offsets in spacetime
   @param param - CloverParam struct
  */
  __device__ inline int linkIndex(int x[], int dx[], const CloverParam &param) {
    int y[4];
    for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + param.X[i]-1) % param.X[i];
    int idx = (((y[3]*param.X[2] + y[2])*param.X[1] + y[1])*param.X[0] + y[0]) >> 1;
    return idx;
  }

  /**
     Construct the field-strength tensor field Fmunu
     First pass only supports no reconstruct for expediency 
     
     @param Fmunu - Pointer to field-strength tensor array.  Result is stored here.
     @param gauge - Pointer to gauge field.
     @param param - CloverParam struct
  */
  template <typename Cmplx>
  __global__ void computeFmunuKernel(Cmplx *Fmunu, const Cmplx *gauge, const CloverParam param) {

    // this is a full lattice index
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    if (X >= param.threads) return;
  
    // thread ordering
    // X = (((parity*X4 + x4)*X3 + x3)*X2 + x2)*X1h + x1h
    // with x1 = 2*x1h + parity  

    // For multi-gpu these need to have offets added on
    // keep X[d] as the true local problem

    // compute spacetime dimensions and parity
    int aux1 = X / (param.X[0]/2);
    int x[4];
    x[0] = X - aux1 * (param.X[0]/2); // this is chbd x
    int aux2 = aux1 / param.X[1];
    x[1] = aux1 - aux2 * param.X[1];
    int aux3 = aux2 / param.X[2];
    x[2] = aux2 - aux3 * param.X[2];
    int parity = aux3 / param.X[3];
    x[3] = aux3 - parity * param.X[3];
    x[0] += parity; // now this is the full index

    for (int mu=0; mu<4; mu++) {
      for (int nu=0; nu<mu; nu++) {
	Matrix<Cmplx,3> F;
	setZero(&F);

	{ // positive mu, nu
	  
	  // load U(x)_(+mu)
	  Matrix<Cmplx,3> U1;
	  int dx[4] = {0, 0, 0, 0};
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, mu, linkIndex(x, dx, param), 
				    param.gaugeStride, &U1);
	  
	  // load U(x+mu)_(+nu)
	  Matrix<Cmplx,3> U2;
	  dx[mu]++;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U2);
	  dx[mu]--;
	  
	  Matrix<Cmplx,3> Ftmp = U2 * U1;
	  
	  // load U(x+nu)_(+mu)
	  Matrix<Cmplx,3> U3;
	  dx[nu]++;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, mu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U3);
	  dx[nu]--;
	  
	  Ftmp = conj(U3) * Ftmp;
	  
	  // load U(x)_(+nu)
	  Matrix<Cmplx,3> U4;
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U4);
	  
	  // complete the plaquette
	  Ftmp = conj(U4) * Ftmp;
	  
	  // sum this contribution to Fmunu
	  F += Ftmp + conj(Ftmp);
	}

	{ // positive mu, negative nu
	  
	  // load U(x)_(+mu)
	  Matrix<Cmplx,3> U1;
	  int dx[4] = {0, 0, 0, 0};
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, mu, linkIndex(x, dx, param), 
				    param.gaugeStride, &U1);
	  
	  // load U(x+mu)_(-nu) = U(x+mu-nu)_(+nu)
	  Matrix<Cmplx,3> U2;
	  dx[mu]++;
	  dx[nu]--;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U2);
	  dx[nu]++;
	  dx[mu]--;
	  
	  Matrix<Cmplx,3> Ftmp = conj(U2) * U1;
	  
	  // load U(x-nu)_mu
	  Matrix<Cmplx,3> U3;
	  dx[nu]--;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, mu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U3);
	  dx[nu]++;
	  
	  Ftmp = conj(U3) * Ftmp;
	  
	  // load U(x)_(-nu) = U(x-nu)_(+nu)
	  Matrix<Cmplx,3> U4;
	  dx[nu]--;
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U4);
	  dx[nu]++;
	  
	  // complete the plaquette
	  Ftmp = U4 * Ftmp;
	  
	  // sum this contribution to Fmunu
	  F += Ftmp + conj(Ftmp);
	}


	{ // negative mu, positive nu
	  
	  // load U(x)_(-mu)
	  Matrix<Cmplx,3> U1;
	  int dx[4] = {0, 0, 0, 0};
	  dx[mu]--;
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, mu, linkIndex(x, dx, param), 
				    param.gaugeStride, &U1);
	  dx[mu]++;

	  // load U(x-mu)_(+nu)
	  Matrix<Cmplx,3> U2;
	  dx[mu]--;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U2);
	  dx[mu]++;
	  
	  Matrix<Cmplx,3> Ftmp = U2 * conj(U1);
	  
	  // load U(x+nu)_(+mu)
	  Matrix<Cmplx,3> U3;
	  dx[nu]++;
	  dx[mu]--;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, mu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U3);
	  dx[mu]++;
	  dx[nu]--;
	  
	  Ftmp = U3 * Ftmp;
	  
	  // load U(x)_(+nu)
	  Matrix<Cmplx,3> U4;
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U4);
	  
	  // complete the plaquette
	  Ftmp = conj(U4) * Ftmp;
	  
	  // sum this contribution to Fmunu
	  F += Ftmp + conj(Ftmp);
	}

	{ // negative mu, negative nu
	  
	  // load U(x)_(-mu)
	  Matrix<Cmplx,3> U1;
	  int dx[4] = {0, 0, 0, 0};
	  dx[mu]--;
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, mu, linkIndex(x, dx, param), 
				    param.gaugeStride, &U1);
	  dx[mu]++;

	  // load U(x-mu)_(-nu) = U(x-mu-nu)_(+nu)
	  Matrix<Cmplx,3> U2;
	  dx[mu]--;
	  dx[nu]--;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U2);
	  dx[nu]++;
	  dx[mu]++;
	  
	  Matrix<Cmplx,3> Ftmp = conj(U2) * conj(U1);
	  
	  // load U(x-nu)_mu
	  Matrix<Cmplx,3> U3;
	  dx[mu]--;
	  dx[nu]--;
	  loadLinkVariableFromArray(gauge+((parity+1)&1)*param.gaugeLengthCB, mu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U3);
	  dx[nu]++;
	  dx[mu]++;
	  
	  Ftmp = U3 * Ftmp;
	  
	  // load U(x)_(-nu) = U(x-nu)_(+nu)
	  Matrix<Cmplx,3> U4;
	  dx[nu]--;
	  loadLinkVariableFromArray(gauge+parity*param.gaugeLengthCB, nu, linkIndex(x,dx,param), 
				    param.gaugeStride, &U4);
	  dx[nu]++;
	  
	  // complete the plaquette
	  Ftmp = U4 * Ftmp;
	  
	  // sum this contribution to Fmunu
	  F += Ftmp + conj(Ftmp);
	}


	int munu_idx = (mu*(mu-1))/2 + nu; // lower-triangular indexing
	writeLinkVariableToArray(F, munu_idx, X/2, param.FmunuStride, Fmunu+parity*param.FmunuLengthCB);
      }
    }

  }

} // namespace quda

void computeCloverCuda(cudaCloverField &clover, const cudaGaugeField &gauge) {

#ifdef GPU_CLOVER_DIRAC
  using namespace quda;

  // first create the field-strength tensor
  int pad = 0;
  GaugeFieldParam tensor_param(gauge.X(), gauge.Precision(), QUDA_RECONSTRUCT_NO, pad, QUDA_TENSOR_GEOMETRY);
  cudaGaugeField Fmunu(tensor_param);

  // set the kernel parameters
  CloverParam param;
  param.threads = gauge.Volume();
  for (int i=0; i<4; i++) param.X[i] = gauge.X()[i];
  param.gaugeLengthCB = gauge.Length() / 2; 
  param.gaugeStride = gauge.Stride();
  param.FmunuLengthCB = Fmunu.Length() / 2;
  param.FmunuStride = Fmunu.Stride();

  dim3 blockDim(32, 1, 1);
  dim3 gridDim((param.threads + blockDim.x - 1) / blockDim.x, 1, 1);

  if (gauge.Precision() == QUDA_DOUBLE_PRECISION) { 
    computeFmunuKernel<<<gridDim,blockDim>>>((double2*)Fmunu.Gauge_p(), (double2*)gauge.Gauge_p(), param);
  } else {
    computeFmunuKernel<<<gridDim,blockDim>>>((float2*)Fmunu.Gauge_p(), (float2*)gauge.Gauge_p(), param);
  }

  // Now contract this into the clover term


#else
  errorQuda("Clover dslash has not been built");
#endif

}

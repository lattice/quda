#pragma once

#include <host_utils.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;
template <typename T> using complex = std::complex<T>;

// Color contract two ColorSpinors returning M[n_spin,n_spin].
template <typename Float>
  inline void contractColors(const Float *const spinorX,
			     const Float *const spinorY,
			     const int n_spin, complex<Float> M[])
{
  for (int s1 = 0; s1 < n_spin; s1++) {
    for (int s2 = 0; s2 < n_spin; s2++) {
      Float re = 0.0; Float im = 0.0;
      for (int c = 0; c < 3; c++) {
	re += (spinorX[6 * s1 + 2 * c + 0] * spinorY[6 * s2 + 2 * c + 0] +
               spinorX[6 * s1 + 2 * c + 1] * spinorY[6 * s2 + 2 * c + 1]);

	im += (spinorX[6 * s1 + 2 * c + 0] * spinorY[6 * s2 + 2 * c + 1] -
	       spinorX[6 * s1 + 2 * c + 1] * spinorY[6 * s2 + 2 * c + 0]);
      }
      M[n_spin * s1 + s2].real() = re;
      M[n_spin * s1 + s2].imag() = im;
    }
  }
};

// accumulate Fourier phase; multiply by exp(i theta)
template <typename Float>
inline void FourierPhase(complex<Float>* z, const Float theta, const QudaFFTSymmType fft_type)
{
  if(theta == 0.) return;
  complex<Float> w{z};
  if(fft_type == QUDA_FFT_SYMM_EVEN){
    Float costh = cos(theta);
    *z.real() = w.real()*costh;
    *z.imag() = w.imag()*costh;
  }
  else if(fft_type == QUDA_FFT_SYMM_ODD){
    Float sinth = sin(theta);
    *z.real() = -w.imag()*sinth;
    *z.imag() =  w.real()*sinth;
  }
  else if(fft_type == QUDA_FFT_SYMM_EO){
    Float costh = cos(theta);
    Float sinth = sin(theta);
    *z.real() = w.real()*costh-w.imag()*sinth;
    *z.imag() = w.imag()*costh+w.real()*sinth;
  }
};

template <typename Float>
void contractFTHost(Float *h_prop_array_flavor_1, Float *h_prop_array_flavor_2,
		    const int n_spin, const int src_colors,
		    const QudaContractType cType,
		    const int *const source_position,
		    const int n_mom, const int *const mom_modes, const QudaFFTSymmType *const fft_type
		    double *h_result
		    )
{
  const int pX[4]{}; // TODO this grid partition offset
  const int X[4]{xdim, ydim, zdim, tdim}; // local grid dims

  // The number of contraction results expected in the output
  size_t num_out_results = n_spin * n_spin;

  size_t corr_dim = 3;
  if (cType == QUDA_CONTRACT_TYPE_DR_FT_Z) corr_dim = 2;

  // The number of slices in the decay dimension on this MPI rank.
  size_t local_decay_dim_slices = X[corr_dim];

  // The number of slices in the decay dimension globally.
  size_t global_decay_dim_slices = local_decay_dim_slices * comm_dim(corr_dim);

  // Array for all decay slices and channels, is zeroed prior to kernel launch
  std::vector<complex<double> > result_global(num_out_results * global_decay_dim_slices);
  std::fill(result_global.begin(), result_global.end(), complex<double>{0.0,0.0});

  complex<Float> M[n_spin*n_spin];
  size_t spinor_sz = n_spin * 3 * 2; // in Floats
  int lx, ly, lz, lt; // local coords
  for(tl=0; tl<X[3]; ++tl)
    for(zl=0; zl<X[2]; ++zl)
      for(yl=0; yl<X[1]; ++yl)
	for(xl=0; xl<X[0]; ++xl) {
	  int sink[4]{xl+pX[0],yl+pX[1],zl+pX[2],tl+pX[3]}; // sink global coordinates
	  size_t sindx = 0; //TODO
	  size_t soff = sindx * spinor_sz;

	  for (size_t s1 = 0; s1 < n_spin; ++s1) {
	    for (size_t s2 = 0; s2 < n_spin; ++s2) {
	      for (size_t c1 = 0; c1 < src_colors; c1++) {
		// color contraction is in M
		contractColors<Float>(spinorX[s1*src_colors+c1][soff],
				      spinorY[s2*src_colors+c1][soff], ,&M);
	  
		if(cType != QUDA_CONTRACT_TYPE_STAGGERED_FT_T) {
		  // place holder for apply gamma matrices
		}

		// compute requested momenta modes
		for (int mom_idx=0; mom_idx<n_mom; ++mom_idx) {
		  complex<Float> fphase{1.,0.};
		  for(int dir=0; dir<4; ++dir) {
		    Float theta = mom_modes[4*mom_indx+dir]*(source_position[dir]-sink[dir]-offsets[dir])/N[dir];
		    FourierPhase<Float>(&fphase,theta,fft_type[4*mom_indx+dir]);
		  }

		  // mutiply by phase and accumulate reduction

		} // momenta
	      } // src color
	    } // src spin
	  } // src spin
	} // coords
};

void printContraction(int red_size, int mom[], int n_mom, int n_spin, double* result)
{
  printfQuda("contractions:");
  for(int k=0; k<n_mom; ++k) {
    printfQuda("\np = %d %d %d %d",mom[4*k+0],mom[4*k+1],mom[4*k+2],mom[4*k+3]);
    for(int c=0; c<red_size*n_spin*n_spin*2; c+= 2) {
      int indx = k*red_size*n_spin*n_spin*2 + c;
      if( c % 8 == 0 ) printfQuda("\n%3d",indx);
      printfQuda(" (%10.3e,%10.3e)",result[indx],result[indx+1]);
    }
  }
  printfQuda("\n");
}

template <typename Float>
int contractionFT_reference(Float *spinorX, Float *spinorY, int n_spin, int src_colors,
			    QudaContractType cType,
			    int red_size, const int *const mom, int n_mom, const QudaFFTSymmType *const fft_type,
			    const double const* d_result)
{
  int faults = 0;
  Float tol = (sizeof(Float) == sizeof(double) ? 1e-9 : 2e-5);

  if(cType != QUDA_CONTRACT_TYPE_STAGGERED_FT_T) {
    printfQuda("WARNING: Host test not implemented for contraction type %d\n",cType);
    printContraction(red_size,mom,n_mom,n_spin,d_result);
    return faults;
  }

  printContraction(red_size,mom,n_mom,n_spin,d_result);

  void contractFTHost(spinorX, spinorY, n_spin, src_colors, cType,
		      const int *X, const int *const source_position,
		      n_mom, mom_modes, fft_type,
		      double *h_result
		      );

  return faults;
};


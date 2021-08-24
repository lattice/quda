#pragma once

#include <host_utils.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;
template <typename T> using complex = std::complex<T>;

// Color contract two ColorSpinors at a site returning a [nSpin x nSpin] matrix.
template <typename Float, int nSpin>
  inline void contractColors(const Float *const spinorX, const Float *const spinorY, Float M[])
{
  for (int s1 = 0; s1 < nSpin; s1++) {
    for (int s2 = 0; s2 < nSpin; s2++) {
      Float re = 0.0; Float im = 0.0;
      for (int c = 0; c < 3; c++) {
	re += (spinorX[6 * s1 + 2 * c + 0] * spinorY[6 * s2 + 2 * c + 0] +
               spinorX[6 * s1 + 2 * c + 1] * spinorY[6 * s2 + 2 * c + 1]);

	im += (spinorX[6 * s1 + 2 * c + 0] * spinorY[6 * s2 + 2 * c + 1] -
	       spinorX[6 * s1 + 2 * c + 1] * spinorY[6 * s2 + 2 * c + 0]);
      }

      M[2 * (nSpin * s1 + s2) + 0] = re;
      M[2 * (nSpin * s1 + s2) + 1] = im;
    }
  }
};

// accumulate Fourier phase
template <typename Float>
inline void FourierPhase(Float z[2], const Float theta, const QudaFFTSymmType fft_type)
{
  Float w[2]{z[0],z[1]};
  if(fft_type == QUDA_FFT_SYMM_EVEN){
    Float costh = cos(theta);
    z[0] = w[0]*costh;
    z[1] = w[1]*costh;
  }
  else if(fft_type == QUDA_FFT_SYMM_ODD){
    Float sinth = sin(theta);
    z[0] = -w[1]*sinth;
    z[1] =  w[0]*sinth;
  }
  else if(fft_type == QUDA_FFT_SYMM_EO){
    Float costh = cos(theta);
    Float sinth = sin(theta);
    z[0] = w[0]*costh-w[1]*sinth;
    z[1] = w[1]*costh+w[0]*sinth;
  }
};

template <typename Float, int nSpin>
void contractFTHost(Float **h_prop_array_flavor_1, Float **h_prop_array_flavor_2, double **h_result,
		    const QudaContractType cType, const int src_colors,
		    const int *X, const int *const source_position,
		    const int n_mom, const int *const mom_modes, const QudaFFTSymmType *const fft_type)
{
  // The number of contraction results expected in the output
  size_t num_out_results = nSpin * nSpin;

  size_t corr_dim = 3;
  if (cType == QUDA_CONTRACT_TYPE_DR_FT_Z) corr_dim = 2;

  // The number of slices in the decay dimension on this MPI rank.
  size_t local_decay_dim_slices = X[corr_dim];

  // The number of slices in the decay dimension globally.
  size_t global_decay_dim_slices = local_decay_dim_slices * comm_dim(corr_dim);

  // Array for all decay slices and channels, is zeroed prior to kernel launch
  std::vector<Complex> result_global(num_out_results * global_decay_dim_slices);
  std::fill(result_global.begin(), result_global.end(), Complex{0.0,0.0});

  Float M[nSpin*nSpin*2];
  size_t soff = 0;
  for (size_t sidx=0; sidx < V; ++sidx, soff += )
    {
      for (size_t s1 = 0; s1 < nSpin; s1++)
	{
	  for (size_t s2 = 0; s2 < nSpin; s2++)
	    {
	      for (size_t c1 = 0; c1 < src_nColor; c1++)
		{
		  // color contraction
		  contractColors<Float,nSpin>(spinorX[s1*src_colors+c1][soff],
					      spinorY[s2*src_colors+c1][soff], ,&M);

		  // place holder for apply gamma matrices

		  // compute requested momenta modes
		  for (int mom_idx=0; mom_idx<n_mom; ++mom_idx)
		    {
		    }
		}
	    }
	}
    }
}


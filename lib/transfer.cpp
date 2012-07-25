#include <transfer.h>

Transfer::Transfer(cpuColorSpinorField *B, int Nvec, int *geo_bs, int spin_bs) {

}

Transfer::~Transfer() {

}

// apply the prolongator
void Transfer::P(cpuColorSpinorField &out, cpuColorSpinorField &in) {

}

// apply the restrictor
void Transfer::R(cpuColorSpinorField &out, cpuColorSpinorField &in) {

}

// Applies the grid prolongation operator (coarse to fine)
template <class FineSpinor, class CoarseSpinor>
void prolongate(FineSpinor &out, const CoarseSpinor &in, const uint *geo_map, const uint* spin_map) {

  for (uint x=0; x<out.Volume(); x++) {
    for (uint s=0; s<out.Nspin(); s++) {
      for (uint c=0; c<out.Ncolor(); c++) {
	out(x, s, c) = in(geo_map[x], spin_map[s], c);
      }
    }
  }

}

// Applies the grid restriction operator (fine to coarse)
template <class CoarseSpinor, class FineSpinor>
void restrict(CoarseSpinor &out, const FineSpinor &in, const uint* geo_map, const uint* spin_map) {

  // We need to zero all elements first, since this is a summation operation
  for (uint x=0; x<in.Volume(); x++) {
    for (uint s=0; s<in.Nspin(); s++) {
      for (uint c=0; c<in.Ncolor(); c++) {
	out(geo_map[is], spin_map[s], c) = 0.0;
      }
    }
  }

  for (uint x=0; x<in.Volume(); x++) {
    for (uint s=0; s<in.Nspin(); s++) {
      for (uint c=0; c<in.Ncolor(); c++) {
	out(geo_map[x], spin_map[s], c) += in(x, s, c);
      }
    }
  }

}

/*
  Rotates from the coarse-color basis into the fine-color basis.  This
  is the second step of applying the prolongator.
 */
template <class FineColor, class CoarseColor, class Rotator>
void rotateFineColor(FineColor &out, const CoarseColor &in, const Rotator &V) {

  for(int x=0; x<in.Volume(); x++) {

    for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

      for (int i=0; i<out.Ncolor(); i++) {
	for (int s=0; s<in.Nspin(); s++) {
	  for (int j=0; j<in.Ncolor(); j++) { 
	    // V is a ColorMatrixField with dimension [out.Nc][in.Ns*in.Nc] - the rotation has spin dependence
	    out(x, s, i) += V(x, i, s*in.Ncolor() + j) * in(x, s, j);
	  }
	}
      }
      
  }

}

/*
  Rotates from the fine-color basis into the color-color basis.
 */
template <class CoarseColor, class FineColor, class Rotator>
void rotateCoarseColor(CoarseColor &out, const FineColor &in, const Rotator &V) {

  for(int x=0; x<in.Volume(); x++) {

    for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

    for (int i=0; i<out.Ncolor(); i++) {
      for (int s=0; s<out.Nspin(); s++) {
	for (int j=0; j<in.Ncolor(); j++) {
	  out(x, s, j) += conj(V(x, i, s*in.Ncolor() + j)) * in(x, s, i);
	}
      }
    }

}

/*
  Orthogonalise the nc vectors v[] of length n
*/
template <class Complex>
void blockGramSchmidt(Complex ***v, uint nBlocks, uint Nc, uint blockSize) {

  for (uint b=0; b<nBlocks; b++) {
    for (int jc=0; jc<Nc; jc++) {
      
      for (int ic=0; ic<jc; ic++) {
	// Calculate dot product
	complex<double> dot = 0.0;
	for (int i=0; i<blockSize; i++) dot += v[b][ic][i] * v[b][jc][i];
	
	// Subtract the blocks to orthogonalise
	for (int i=0; i<blockSize; i++) v[b][jc][i] -= dot * v[b][ic][i];
      }
      
      // Normalize the block
      double nrm2 = 0.0;
      for (int i=0; i<blockSize; i++) 
	norm2 += real(v[b][jc][i])*real(v[b][jc][i]) + imag(v[b][jc][i])*imag(v[b][jc][i]);
      nrm2 = 1.0/sqrt(nrm2);
      
      for (int i=0; i<blockSize; i++) v[b][jc][i] = nrm2 * v[b][jc][i];
    }

  }

}

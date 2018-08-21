/* C. Kallidonis: File that contains kernels for shifting
 * quark propagators and gauge fields, as required by
 * qPDF and TMD contractions
 * August 2018
 */

#include <qlua_contract_shifts.cuh>

namespace quda {

  /* This function performs Forward and Backward covariant shifts of propagators in any direction,
   * with the derivative acting either on the right (on quarks) or on the left (on anti-quarks)
   */
  __global__ void CovShiftDevicePropPM1(QluaContractArg *arg,
					Vector *outShf, Propagator prop[],
					int dir, qcShiftType shiftType){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    const int Ns = QC_Ns;
    const int Nc = QC_Nc;

    typedef Matrix<complex<QC_REAL>,Nc> Link;
    typedef ColorSpinor<QC_REAL,Nc,Ns> Vector;

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    if(shiftType == qcFwdShfActR){ // Forward shift, derivative acting on quark
      const int fwdIdx = linkIndexP1(coord, arg->dim, dir);

      if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);
	const Link U = arg->U(dir, x_cb, pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i].Ghost(dir, 1, ghostIdx, nbrPty);
	  outShf[i] = U * pIn;   //-- y(x) = U_\mu(x) y(x+\mu)
	}
      }
      else{
	const Link U = arg->U(dir, x_cb, pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](fwdIdx, nbrPty);
	  outShf[i] = U * pIn;   //-- y(x) = U_\mu(x) y(x+\mu)
	}
      }
    }
    else if(shiftType == qcBwdShfActR){ // Backward shift, derivative acting on quark
      const int bwdIdx = linkIndexM1(coord, arg->dim, dir);
      const int gaugeIdx = bwdIdx;

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	const Link U = arg->U.Ghost(dir, ghostIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i].Ghost(dir, 0, ghostIdx, nbrPty);
	  outShf[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x-\mu) \psi(x-\mu)
	}
      }
      else{
	const Link U = arg->U(dir, gaugeIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](bwdIdx, nbrPty);
	  outShf[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x-\mu) \psi(x-\mu)
	}
      }
    }
    else if(shiftType == qcFwdShfActL){ // Forward shift, derivative acting on anti-quark
      const int fwdIdx = linkIndexP1(coord, arg->dim, dir);

      if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);
	const Link U = arg->U(dir, x_cb, pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i].Ghost(dir, 1, ghostIdx, nbrPty);
	  outShf[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x) y(x+\mu)
	}
      }
      else{
	const Link U = arg->U(dir, x_cb, pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](fwdIdx, nbrPty);
	  outShf[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x) y(x+\mu)
	}
      }
    }
    else if(shiftType == qcBwdShfActL){ // Backward shift, derivative acting on anti-quark
      const int bwdIdx = linkIndexM1(coord, arg->dim, dir);
      const int gaugeIdx = bwdIdx;

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	const Link U = arg->U.Ghost(dir, ghostIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i].Ghost(dir, 0, ghostIdx, nbrPty);
	  outShf[i] = U * pIn;   //-- y(x) = U_\mu(x-\mu) \psi(x-\mu)
	}
      }
      else{
	const Link U = arg->U(dir, gaugeIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](bwdIdx, nbrPty);
	  outShf[i] = U * pIn;   //-- y(x) = U_\mu(x-\mu) \psi(x-\mu)
	}
      }
    }
    else{ // All threads printing!!!
      printf("CovShiftDevicePropPM1 - ERROR: Got invalid shiftType = %d\n", shiftType);
      return;
    }

  }//- Function CovShiftDevicePropPM1

} //- namespace quda

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
  __device__ void CovShiftPropPM1_dev(QluaContractArg *arg,
				      Vector *outShf, Propagator prop[],
				      int x_cb, int pty,
				      int dir, qcCovShiftType shiftType){

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    if(shiftType == qcFwdCovShfActR){ // Forward shift, derivative acting on quark
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
    else if(shiftType == qcBwdCovShfActR){ // Backward shift, derivative acting on quark
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
    else if(shiftType == qcFwdCovShfActL){ // Forward shift, derivative acting on anti-quark
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
    else if(shiftType == qcBwdCovShfActL){ // Backward shift, derivative acting on anti-quark
      const int bwdIdx = linkIndexM1(coord, arg->dim, dir);
      const int gaugeIdx = bwdIdx;

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	const Link U = arg->U.Ghost(dir, ghostIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i].Ghost(dir, 0, ghostIdx, nbrPty);
	  outShf[i] = U * pIn;   //-- y(x) = U_\mu(x-\mu) y(x-\mu)
	}
      }
      else{
	const Link U = arg->U(dir, gaugeIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](bwdIdx, nbrPty);
	  outShf[i] = U * pIn;   //-- y(x) = U_\mu(x-\mu) y(x-\mu)
	}
      }
    }
    else{
      if(x_cb == 0) printf("CovShiftPropPM1_dev - ERROR: Got invalid shiftType = %d\n", shiftType);
      return;
    }

  }//- CovShiftDevicePropPM1
  //------------------------------------------------------------------------------------------

  __global__ void CovShiftPropPM1_kernel(QluaContractArg *arg, QluaAuxCntrArg *auxArg,
					 int shfDir, qcCovShiftType shiftType){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    Vector outShf[QUDA_PROP_NVEC];

    CovShiftPropPM1_dev(arg, outShf, arg->prop1, x_cb, pty, shfDir, shiftType);

    for(int i=0;i<QUDA_PROP_NVEC;i++)
      auxArg->auxProp1[i](x_cb, pty) = outShf[i];

  }//- CovShiftPropPM1_dev
  //------------------------------------------------------------------------------------------



  /* This function performs Forward and Backward non-covariant shifts of propagators in any direction.
   * It only supports shifting the forward propagator, hence it does not accept a propagator argument.
   */
  __device__ void NonCovShiftPropOnAxis_dev(QluaContractArg *arg, Vector *outShf,
					    int x_cb, int pty,
					    qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    if(shfSgn == qcShfSgnPlus){ // Forward shift
      const int fwdIdx = linkIndexP1(coord, arg->dim, dir);

      if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = arg->prop1[i].Ghost(dir, 1, ghostIdx, nbrPty);
	  outShf[i] = pIn;   //-- y(x) <- y(x+\mu)
	}
      }
      else{
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = arg->prop1[i](fwdIdx, nbrPty);
	  outShf[i] = pIn;   //-- y(x) <- y(x+\mu)
	}
      }
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      const int bwdIdx = linkIndexM1(coord, arg->dim, dir);

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = arg->prop1[i].Ghost(dir, 0, ghostIdx, nbrPty);
	  outShf[i] = pIn;   //-- y(x) <- y(x-\mu)
	}
      }
      else{
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = arg->prop1[i](bwdIdx, nbrPty);
	  outShf[i] = pIn;   //-- y(x) <- y(x-\mu)
	}
      }
    }
    else{
      if(x_cb == 0) printf("NonCovShiftPropOnAxis_dev - ERROR: Got invalid Shift Sign = %d\n", (int)shfSgn);
      return;
    }

  }//- NonCovShiftPropOnAxis_dev
  //------------------------------------------------------------------------------------------


  __global__ void NonCovShiftPropOnAxis_kernel(QluaContractArg *arg, QluaAuxCntrArg *auxArg,
					       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    Vector outShf[QUDA_PROP_NVEC];

    NonCovShiftPropOnAxis_dev(arg, outShf, x_cb, pty, shfDir, shfSgn);

    for(int i=0;i<QUDA_PROP_NVEC;i++)
      auxArg->auxProp1[i](x_cb, pty) = outShf[i];

  }//- CovShiftDevicePropPM1_kernel
  //------------------------------------------------------------------------------------------




} //- namespace quda

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
  __device__ void CovShiftPropPM1_dev(Vector *shfVec, QluaContractArg *arg, Propagator prop[],
				      int dir, qcCovShiftType shiftType,
				      int x_cb, int pty){

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
	  shfVec[i] = U * pIn;   //-- y(x) = U_\mu(x) y(x+\mu)
	}
      }
      else{
	const Link U = arg->U(dir, x_cb, pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](fwdIdx, nbrPty);
	  shfVec[i] = U * pIn;   //-- y(x) = U_\mu(x) y(x+\mu)
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
	  shfVec[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x-\mu) \psi(x-\mu)
	}
      }
      else{
	const Link U = arg->U(dir, gaugeIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](bwdIdx, nbrPty);
	  shfVec[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x-\mu) \psi(x-\mu)
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
	  shfVec[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x) y(x+\mu)
	}
      }
      else{
	const Link U = arg->U(dir, x_cb, pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](fwdIdx, nbrPty);
	  shfVec[i] = conj(U) * pIn;   //-- y(x) = U_\mu^\dag(x) y(x+\mu)
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
	  shfVec[i] = U * pIn;   //-- y(x) = U_\mu(x-\mu) y(x-\mu)
	}
      }
      else{
	const Link U = arg->U(dir, gaugeIdx, 1-pty);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = prop[i](bwdIdx, nbrPty);
	  shfVec[i] = U * pIn;   //-- y(x) = U_\mu(x-\mu) y(x-\mu)
	}
      }
    }
    else{
      if(x_cb == 0) printf("CovShiftPropPM1_dev - ERROR: Got invalid shiftType = %d\n", shiftType);
      return;
    }

  }//- CovShiftDevicePropPM1
  //------------------------------------------------------------------------------------------

  __global__ void CovShiftPropPM1_kernel(QluaContractArg *arg,
					 int shfDir, qcCovShiftType shiftType){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    Vector shfVec[QUDA_PROP_NVEC];

    CovShiftPropPM1_dev(shfVec, arg, arg->prop1, shfDir, shiftType, x_cb, pty);

    //- Assign the shifted propagator to the third propagator in the arguments structure
    for(int i=0;i<QUDA_PROP_NVEC;i++)
      arg->prop3[i](x_cb, pty) = shfVec[i];

  }//- CovShiftPropPM1_dev
  //------------------------------------------------------------------------------------------



  /* This function performs Forward and Backward non-covariant shifts of propagators in any direction.
   * It only supports shifting the forward propagator, hence it does not accept a propagator argument.
   */
  __device__ void NonCovShiftPropOnAxis_dev(Vector *shfVec, QluaContractArg *arg, 
					    qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn,
					    int x_cb, int pty){

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
	  shfVec[i] = pIn;   //-- y(x) <- y(x+\mu)
	}//- QUDA_PROP_NVEC
      }
      else{
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = arg->prop1[i](fwdIdx, nbrPty);
	  shfVec[i] = pIn;   //-- y(x) <- y(x+\mu)
	}
      }
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      const int bwdIdx = linkIndexM1(coord, arg->dim, dir);

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = arg->prop1[i].Ghost(dir, 0, ghostIdx, nbrPty);
	  shfVec[i] = pIn;   //-- y(x) <- y(x-\mu)
	}
      }
      else{
	for(int i=0;i<QUDA_PROP_NVEC;i++){
	  const Vector pIn = arg->prop1[i](bwdIdx, nbrPty);
	  shfVec[i] = pIn;   //-- y(x) <- y(x-\mu)
	}
      }
    }
    else{
      if(x_cb == 0) printf("NonCovShiftPropOnAxis_dev - ERROR: Got invalid Shift Sign = %d\n", (int)shfSgn);
      return;
    }

  }//- NonCovShiftPropOnAxis_dev
  //------------------------------------------------------------------------------------------


  __global__ void NonCovShiftPropOnAxis_kernel(QluaContractArg *arg,
					       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    Vector shfVec[QUDA_PROP_NVEC];

    NonCovShiftPropOnAxis_dev(shfVec, arg, shfDir, shfSgn, x_cb, pty);

    //- Assign the shifted propagator to the third propagator in the arguments structure
    for(int i=0;i<QUDA_PROP_NVEC;i++)
      arg->prop3[i](x_cb, pty) = shfVec[i];

  }//- NonCovShiftPropOnAxis_kernel
  //------------------------------------------------------------------------------------------


  //- This function performs Forward and Backward non-covariant shifts of vectors
  //- within the forward propagator in any direction.
  __device__ void NonCovShiftVectorOnAxis_dev(Vector &shfVec, QluaCntrTMDArg *TMDarg, int ivec,
					      qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn,
					      int x_cb, int pty){

    const int nbrPty = (TMDarg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, TMDarg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    if(shfSgn == qcShfSgnPlus){ // Forward shift
      const int fwdIdx = linkIndexP1(coord, TMDarg->dim, dir);

      if( TMDarg->commDim[dir] && (coord[dir] + TMDarg->nFace >= TMDarg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, TMDarg->dim, dir, TMDarg->nFace);
	shfVec = TMDarg->fwdVec.Ghost(dir, 1, ghostIdx, nbrPty); //-- y(x) <- y(x+\mu)
      }
      else{
	shfVec = TMDarg->fwdVec(fwdIdx, nbrPty); //-- y(x) <- y(x+\mu)
      }
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      const int bwdIdx = linkIndexM1(coord, TMDarg->dim, dir);

      if ( TMDarg->commDim[dir] && (coord[dir] - TMDarg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, TMDarg->dim, dir, TMDarg->nFace);
	shfVec = TMDarg->fwdVec.Ghost(dir, 0, ghostIdx, nbrPty); //-- y(x) <- y(x-\mu)
      }
      else{
	shfVec = TMDarg->fwdVec(bwdIdx, nbrPty); //-- y(x) <- y(x-\mu)
      }
    }
    else{
      if(x_cb == 0) printf("NonCovShiftVectorOnAxis_dev - ERROR: Got invalid Shift Sign = %d\n", (int)shfSgn);
      return;
    }

  }//- NonCovShiftVectorOnAxis_dev
  //------------------------------------------------------------------------------------------


  __global__ void NonCovShiftVectorOnAxis_kernel(QluaCntrTMDArg *TMDarg, int ivec,
						 qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (TMDarg->nParity == 2) ? pty : TMDarg->parity;
    if (x_cb >= TMDarg->volumeCB) return;
    if (pty >= TMDarg->nParity) return;

    Vector shfVec;

    NonCovShiftVectorOnAxis_dev(shfVec, TMDarg, ivec, shfDir, shfSgn, x_cb, pty);

    TMDarg->shfVec(x_cb, pty) = shfVec;

  }//- NonCovShiftVectorOnAxis_kernel
  //------------------------------------------------------------------------------------------


} //- namespace quda

/* C. Kallidonis: File that contains kernels for shifting
 * quark propagators and gauge fields, as required by
 * TMD contractions
 * August 2018
 */

#include <qlua_contract_shifts.cuh>

namespace quda {

  /* This function performs Forward and Backward covariant and non-covariant shifts of vectors (cudaColorSpinorFields)
   * within the forward propagator in any direction.
   * One needs to properly have the ghosts loaded in the input propagator before calling this function.
   * This is the main function that will be used in TMD contractions.
   */
  __device__ void ShiftVectorOnAxis_dev(Vector &shfVec, TMDcontractState *TMDcs, int ivec,
					qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType,
					int x_cb, int pty){

    const int nbrPty = (TMDcs->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, TMDcs->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    if(shfSgn == qcShfSgnPlus){ // Forward shift
      Vector vIn;
      const int fwdIdx = linkIndexP1(coord, TMDcs->dim, dir);

      if( TMDcs->commDim[dir] && (coord[dir] + TMDcs->nFace >= TMDcs->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, TMDcs->dim, dir, TMDcs->nFace);      
	vIn = TMDcs->fwdProp[ivec].Ghost(dir, 1, ghostIdx, nbrPty);
      }
      else vIn = TMDcs->fwdProp[ivec](fwdIdx, nbrPty);

      if(shfType == qcCovShift){
	const Link U = TMDcs->U(dir, x_cb, pty);
	shfVec = U * vIn;                             //-- y(x) <- U_\mu(x) * y(x+\mu)
      }
      else if(shfType == qcNonCovShift) shfVec = vIn; //-- y(x) <- y(x+\mu)
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      const int bwdIdx = linkIndexM1(coord, TMDcs->dim, dir);

      if ( TMDcs->commDim[dir] && (coord[dir] - TMDcs->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, TMDcs->dim, dir, TMDcs->nFace);
	const Vector vIn = TMDcs->fwdProp[ivec].Ghost(dir, 0, ghostIdx, nbrPty);

	if(shfType == qcCovShift){
	  const Link U = TMDcs->U.Ghost(dir, ghostIdx, 1-pty);
	  shfVec = conj(U) * vIn;                           //-- y(x) <- U_\mu^\dag(x-\mu) * y(x-\mu)
	}
	else if(shfType == qcNonCovShift) shfVec = vIn;     //-- y(x) <- y(x-\mu)
      }
      else{
	const Vector vIn = TMDcs->fwdProp[ivec](bwdIdx, nbrPty);

	if(shfType == qcCovShift){
	  const Link U = TMDcs->U(dir, bwdIdx, 1-pty);
	  shfVec = conj(U) * vIn;                           //-- y(x) <- U_\mu^\dag(x-\mu) * y(x-\mu)
	}
	else if(shfType == qcNonCovShift) shfVec = vIn;     //-- y(x) <- y(x-\mu)
      }
    }
    else{
      if(x_cb == 0) printf("ShiftVectorOnAxis_dev - ERROR: Got invalid Shift Sign = %d\n", (int)shfSgn);
      return;
    }

  }//- ShiftVectorOnAxis_dev
  //------------------------------------------------------------------------------------------

  __global__ void ShiftVectorOnAxis_kernel(TMDcontractState *TMDcs, int ivec,
					   qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (TMDcs->nParity == 2) ? pty : TMDcs->parity;
    if (x_cb >= TMDcs->volumeCB) return;
    if (pty >= TMDcs->nParity) return;

    Vector shfVec;

    ShiftVectorOnAxis_dev(shfVec, TMDcs, ivec, shfDir, shfSgn, shfType, x_cb, pty);

    TMDcs->auxProp[ivec](x_cb, pty) = shfVec;

  }//- ShiftVectorOnAxis_kernel
  //------------------------------------------------------------------------------------------



  /* This function performs Forward and Backward covariant and non-covariant shifts of 
   * gauge fields in any direction and dimension.
   * One needs to properly have the ghosts loaded in the input gauge field before calling this function.
   */
  __device__ void ShiftGauge_dev(Link &shfGauge, TMDcontractState *TMDcs, qcTMD_ShiftDir muSrc,
  				 qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType,
  				 int x_cb, int pty){
    
    const int nbrPty = (TMDcs->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, TMDcs->dim, pty);
    coord[4] = 0;

    int nu = (int)shfDir;     //-- Direction of the shift
    int srcDim = (int)muSrc;  //-- Lorentz index of input (source) gauge field

    if(shfSgn == qcShfSgnPlus){ // Forward shift
      const int fwdIdx = linkIndexP1(coord, TMDcs->dim, nu);

      Link Usrc;
      if( TMDcs->commDim[nu] && (coord[nu] + TMDcs->nFace >= TMDcs->dim[nu]) ){
  	const int ghostIdx = ghostFaceIndex<1>(coord, TMDcs->dim, nu, TMDcs->nFace);      
  	Usrc = TMDcs->U.Ghost(srcDim, ghostIdx, nbrPty);
      }
      else Usrc = TMDcs->U(srcDim, fwdIdx, nbrPty);

      if(shfType == qcCovShift){
  	const Link U = TMDcs->U(nu, x_cb, pty);
  	shfGauge = U * Usrc;                              //-- U(x) <- U_\nu(x) * U_src(x+\nu)
      }
      else if(shfType == qcNonCovShift) shfGauge = Usrc;  //-- U(x) <- U_src(x+\nu)
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      const int bwdIdx = linkIndexM1(coord, TMDcs->dim, nu);

      if ( TMDcs->commDim[nu] && (coord[nu] - TMDcs->nFace < 0) ) {
  	const int ghostIdx = ghostFaceIndex<0>(coord, TMDcs->dim, nu, TMDcs->nFace);
  	const Link Usrc = TMDcs->U.Ghost(srcDim, ghostIdx, nbrPty);

  	if(shfType == qcCovShift){
  	  const Link U = TMDcs->U.Ghost(nu, ghostIdx, nbrPty);
  	  shfGauge = conj(U) * Usrc;                           //-- U(x) <- U_\nu^\dag(x-\nu) * U_src(x-\nu)
  	}
  	else if(shfType == qcNonCovShift) shfGauge = Usrc;     //-- U(x) <- U_src(x-\nu)
      }
      else{
  	const Link Usrc = TMDcs->U(srcDim, bwdIdx, nbrPty);

  	if(shfType == qcCovShift){
  	  const Link U = TMDcs->U(nu, bwdIdx, nbrPty);
  	  shfGauge = conj(U) * Usrc;                           //-- U(x) <- U_\nu^\dag(x-\nu) * U_src(x-\nu)
  	}
  	else if(shfType == qcNonCovShift) shfGauge = Usrc;     //-- y(x) <- U_src(x-\nu)
      }
    }
    else{
      if(x_cb == 0) printf("ShiftGauge_dev - ERROR: Got invalid Shift Sign = %d\n", (int)shfSgn);
      return;
    }

  }//- ShiftGauge_dev
  //------------------------------------------------------------------------------------------


  __global__ void ShiftGauge_kernel(TMDcontractState *TMDcs, qcTMD_ShiftDir muDst, qcTMD_ShiftDir muSrc,
                                    qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (TMDcs->nParity == 2) ? pty : TMDcs->parity;
    if (x_cb >= TMDcs->volumeCB) return;
    if (pty >= TMDcs->nParity) return;

    Link shfGauge;

    ShiftGauge_dev(shfGauge, TMDcs, muSrc, shfDir, shfSgn, shfType, x_cb, pty);

    TMDcs->shfU(muDst, x_cb, pty) = shfGauge;

  }//- ShiftGauge_kernel


} //- namespace quda

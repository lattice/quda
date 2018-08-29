/* C. Kallidonis: File that contains kernels for shifting
 * quark propagators and gauge fields, as required by
 * qPDF and TMD contractions
 * August 2018
 */

#include <qlua_contract_shifts.cuh>

namespace quda {

  /* This function performs Forward and Backward covariant and non-covariant shifts of vectors (cudaColorSpinorFields)
   * within the forward propagator in any direction.
   * One needs to properly have the ghosts loaded in the input propagator before calling this function.
   * This is the main function that will be used in TMD contractions.
   */
  __device__ void ShiftVectorOnAxis_dev(Vector &shfVec, QluaCntrTMDArg &TMDarg, int ivec,
					qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType,
					int x_cb, int pty){

    const int nbrPty = (TMDarg.nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, TMDarg.dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    if(shfSgn == qcShfSgnPlus){ // Forward shift
      Vector vIn;
      const int fwdIdx = linkIndexP1(coord, TMDarg.dim, dir);

      if( TMDarg.commDim[dir] && (coord[dir] + TMDarg.nFace >= TMDarg.dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, TMDarg.dim, dir, TMDarg.nFace);      
	vIn = TMDarg.fwdVec.Ghost(dir, 1, ghostIdx, nbrPty);
      }
      else vIn = TMDarg.fwdVec(fwdIdx, nbrPty);

      if(shfType == qcCovShift){
	const Link U = TMDarg.U(dir, x_cb, pty);
	shfVec = U * vIn;                             //-- y(x) <- U_\mu(x) * y(x+\mu)
      }
      else if(shfType == qcNonCovShift) shfVec = vIn; //-- y(x) <- y(x+\mu)
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      const int bwdIdx = linkIndexM1(coord, TMDarg.dim, dir);

      if ( TMDarg.commDim[dir] && (coord[dir] - TMDarg.nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, TMDarg.dim, dir, TMDarg.nFace);
	const Vector vIn = TMDarg.fwdVec.Ghost(dir, 0, ghostIdx, nbrPty);

	if(shfType == qcCovShift){
	  const Link U = TMDarg.U.Ghost(dir, ghostIdx, 1-pty);
	  shfVec = conj(U) * vIn;                           //-- y(x) <- U_\mu^\dag(x-\mu) * y(x-\mu)
	}
	else if(shfType == qcNonCovShift) shfVec = vIn;     //-- y(x) <- y(x-\mu)
      }
      else{
	const Vector vIn = TMDarg.fwdVec(bwdIdx, nbrPty);

	if(shfType == qcCovShift){
	  const Link U = TMDarg.U(dir, bwdIdx, 1-pty);
	  shfVec = conj(U) * vIn;                           //-- y(x) <- U_\mu^\dag(x-\mu) * y(x-\mu)
	}
	else if(shfType == qcNonCovShift) shfVec = vIn;     //-- y(x) <- y(x-\mu)
      }
    }
    else{
      if(x_cb == 0) printf("NonCovShiftVectorOnAxis_dev - ERROR: Got invalid Shift Sign = %d\n", (int)shfSgn);
      return;
    }

  }//- NonCovShiftVectorOnAxis_dev
  //------------------------------------------------------------------------------------------


  __global__ void ShiftVectorOnAxis_kernel(QluaCntrTMDArg TMDarg, int ivec,
					   qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (TMDarg.nParity == 2) ? pty : TMDarg.parity;
    if (x_cb >= TMDarg.volumeCB) return;
    if (pty >= TMDarg.nParity) return;

    Vector shfVec;

    ShiftVectorOnAxis_dev(shfVec, TMDarg, ivec, shfDir, shfSgn, shfType, x_cb, pty);

    TMDarg.shfVec(x_cb, pty) = shfVec;

  }//- NonCovShiftVectorOnAxis_kernel
  //------------------------------------------------------------------------------------------


} //- namespace quda

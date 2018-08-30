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
      if(x_cb == 0) printf("ShiftVectorOnAxis_dev - ERROR: Got invalid Shift Sign = %d\n", (int)shfSgn);
      return;
    }

  }//- ShiftVectorOnAxis_dev
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

  }//- ShiftVectorOnAxis_kernel
  //------------------------------------------------------------------------------------------



  /* This function performs Forward and Backward covariant and non-covariant shifts of 
   * gauge fields in any direction and dimension.
   * One needs to properly have the ghosts loaded in the input gauge field before calling this function.
   */
  __device__ void ShiftGauge_dev(Link &shfGauge, QluaCntrTMDArg &TMDarg, qcTMD_ShiftDir muSrc,
				 qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType,
				 int x_cb, int pty){
    
    const int nbrPty = (TMDarg.nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, TMDarg.dim, pty);
    coord[4] = 0;

    int nu = (int)shfDir;     //-- Direction of the shift
    int srcDim = (int)muSrc;  //-- Lorentz index of input (source) gauge field

    if(shfSgn == qcShfSgnPlus){ // Forward shift
      const int fwdIdx = linkIndexP1(coord, TMDarg.dim, nu);

      Link Usrc;
      if( TMDarg.commDim[nu] && (coord[nu] + TMDarg.nFace >= TMDarg.dim[nu]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, TMDarg.dim, nu, TMDarg.nFace);      
	Usrc = TMDarg.U.Ghost(srcDim, ghostIdx, nbrPty);
      }
      else Usrc = TMDarg.U(srcDim, fwdIdx, nbrPty);

      if(shfType == qcCovShift){
	const Link U = TMDarg.U(nu, x_cb, pty);
	shfGauge = U * Usrc;                              //-- U(x) <- U_\nu(x) * U_src(x+\nu)
      }
      else if(shfType == qcNonCovShift) shfGauge = Usrc;  //-- U(x) <- U_src(x+\nu)
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      const int bwdIdx = linkIndexM1(coord, TMDarg.dim, nu);

      if ( TMDarg.commDim[nu] && (coord[nu] - TMDarg.nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, TMDarg.dim, nu, TMDarg.nFace);
	const Link Usrc = TMDarg.U.Ghost(srcDim, ghostIdx, nbrPty);

	if(shfType == qcCovShift){
	  const Link U = TMDarg.U.Ghost(nu, ghostIdx, nbrPty);
	  shfGauge = conj(U) * Usrc;                           //-- U(x) <- U_\nu^\dag(x-\nu) * U_src(x-\nu)
	}
	else if(shfType == qcNonCovShift) shfGauge = Usrc;     //-- U(x) <- U_src(x-\nu)
      }
      else{
	const Link Usrc = TMDarg.U(srcDim, bwdIdx, nbrPty);

	if(shfType == qcCovShift){
	  const Link U = TMDarg.U(nu, bwdIdx, nbrPty);
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


  __global__ void ShiftGauge_kernel(QluaCntrTMDArg TMDarg, qcTMD_ShiftDir muDst, qcTMD_ShiftDir muSrc,
                                    qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (TMDarg.nParity == 2) ? pty : TMDarg.parity;
    if (x_cb >= TMDarg.volumeCB) return;
    if (pty >= TMDarg.nParity) return;

    Link shfGauge;

    ShiftGauge_dev(shfGauge, TMDarg, muSrc, shfDir, shfSgn, shfType, x_cb, pty);

    TMDarg.shfU(muDst, x_cb, pty) = shfGauge;

  }//- ShiftGauge_kernel


} //- namespace quda

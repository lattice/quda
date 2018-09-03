/* C. Kallidonis: File that contains kernels for shifting
 * quark propagators and gauge fields, as required by
 * TMD contractions
 * August 2018
 */

#include <qlua_contract_shifts.cuh>

namespace quda {


  __global__ void ShiftCudaVec_nonCov_kernel(Arg_ShiftCudaVec_nonCov *arg,
					     qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    Vector shfVec;
    if(shfSgn == qcShfSgnPlus){ // Forward shift  y(x) <- y(x+\mu)
      const int fwdIdx = linkIndexP1(coord, arg->dim, dir);

      if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);      
	shfVec = arg->src.Ghost(dir, 1, ghostIdx, nbrPty);
      }
      else shfVec = arg->src(fwdIdx, nbrPty);
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift   y(x) <- y(x-\mu)
      const int bwdIdx = linkIndexM1(coord, arg->dim, dir);

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	shfVec = arg->src.Ghost(dir, 0, ghostIdx, nbrPty);
      }
      else shfVec = arg->src(bwdIdx, nbrPty);
    }

    arg->dst(x_cb, pty) = shfVec;
  }//- ShiftCudaVec_nonCov_kernel


  __global__ void ShiftCudaVec_Cov_kernel(Arg_ShiftCudaVec_Cov *arg,
					  qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    Vector shfVec;
    if(shfSgn == qcShfSgnPlus){ // Forward shift   y(x) <- U_\mu(x) * y(x+\mu)
      const int fwdIdx = linkIndexP1(coord, arg->dim, dir);

      Vector vIn;
      if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);      
	vIn = arg->src.Ghost(dir, 1, ghostIdx, nbrPty);
      }
      else vIn = arg->src(fwdIdx, nbrPty);

      const Link U = arg->U(dir, x_cb, pty);
      shfVec = U * vIn;
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift   y(x) <- U_\mu^\dag(x-\mu) * y(x-\mu)
      Vector vIn;
      Link U;

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	vIn = arg->src.Ghost(dir, 0, ghostIdx, nbrPty);
	U = arg->U.Ghost(dir, ghostIdx, nbrPty);
      }
      else{
        const int bwdIdx = linkIndexM1(coord, arg->dim, dir);
	vIn = arg->src(bwdIdx, nbrPty);
	U = arg->U(dir, bwdIdx, nbrPty);
      }

      shfVec = conj(U) * vIn;
    }

    arg->dst(x_cb, pty) = shfVec;
  }//-- ShiftCudaVec_Cov_kernel


  __global__ void ShiftGauge_nonCov_kernel(Arg_ShiftGauge_nonCov *arg,
					   qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    for (int mu=0;mu<4;mu++){
      Link shfU;

      if(shfSgn == qcShfSgnPlus){ // Forward shift     U(x) <- U_src(x+\nu)
        const int fwdIdx = linkIndexP1(coord, arg->dim, dir);
	
        if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
          const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);      
          shfU = arg->src.Ghost(mu, ghostIdx, nbrPty);
        }
        else shfU = arg->src(mu, fwdIdx, nbrPty);
      }
      else if(shfSgn == qcShfSgnMinus){ // Backward shift     U(x) <- U_src(x+-nu)
        if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
          const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
          shfU = arg->src.Ghost(mu, ghostIdx, nbrPty);
        }
        else{
          const int bwdIdx = linkIndexM1(coord, arg->dim, dir);
          shfU = arg->src(mu, bwdIdx, nbrPty);
        }
      }
      
      arg->dst(mu, x_cb, pty) = shfU;
    }//-- for mu

  }//-- ShiftGauge_nonCov_kernel


  __global__ void ShiftLink_Cov_kernel(Arg_ShiftLink_Cov *arg,
				       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    Link shfU;
    if(shfSgn == qcShfSgnPlus){ // Forward shift  U(x) <- U_\nu(x) * U_src(x+\nu)
      const int fwdIdx = linkIndexP1(coord, arg->dim, dir);

      Link Usrc;
      if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);      
	Usrc = arg->src.Ghost(arg->i_src, ghostIdx, nbrPty);
      }
      else Usrc = arg->src(arg->i_src, fwdIdx, nbrPty);

      const Link U = arg->U(dir, x_cb, pty);
      shfU = U * Usrc;
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift     U(x) <- U_\nu^\dag(x-\nu) * U_src(x-\nu)
      Link Usrc;
      Link U;

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	Usrc = arg->src.Ghost(arg->i_src, ghostIdx, nbrPty);
        U = arg->U.Ghost(dir, ghostIdx, nbrPty);
      }
      else{
        const int bwdIdx = linkIndexM1(coord, arg->dim, dir);
	Usrc = arg->src(arg->i_src, bwdIdx, nbrPty);
        U = arg->U(dir, bwdIdx, nbrPty);
      }

      shfU = conj(U) * Usrc;
    }

    arg->dst(arg->i_dst, x_cb, pty) = shfU;
  }//-- perform_ShiftLink_Cov


  __global__ void ShiftLink_AdjSplitCov_kernel(Arg_ShiftLink_AdjSplitCov *arg,
					       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){ 

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    const int nbrPty = (arg->nParity == 2) ? 1-pty : 0; // Parity of neighboring site
    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    Link shfU;
    if(shfSgn == qcShfSgnPlus){ // Forward shift
      const int fwdIdx = linkIndexP1(coord, arg->dim, dir);

      Link Usrc;
      if( arg->commDim[dir] && (coord[dir] + arg->nFace >= arg->dim[dir]) ){
	const int ghostIdx = ghostFaceIndex<1>(coord, arg->dim, dir, arg->nFace);      
	Usrc = arg->src.Ghost(arg->i_src, ghostIdx, nbrPty);
      }
      else Usrc = arg->src(arg->i_src, fwdIdx, nbrPty);

      const Link U = arg->U(dir, x_cb, pty);
      const Link U2= arg->U2(dir, x_cb, pty);
      shfU = U * Usrc * conj(U2);
    }
    else if(shfSgn == qcShfSgnMinus){ // Backward shift
      Link Usrc;
      Link U, U2;

      if ( arg->commDim[dir] && (coord[dir] - arg->nFace < 0) ) {
	const int ghostIdx = ghostFaceIndex<0>(coord, arg->dim, dir, arg->nFace);
	Usrc = arg->src.Ghost(arg->i_src, ghostIdx, nbrPty);
        U   = arg->U.Ghost(dir, ghostIdx, nbrPty);
        U2  = arg->U2.Ghost(dir, ghostIdx, nbrPty);
      }
      else{
        const int bwdIdx = linkIndexM1(coord, arg->dim, dir);
	Usrc = arg->src(arg->i_src, bwdIdx, nbrPty);
        U   = arg->U(dir, bwdIdx, nbrPty);
        U2  = arg->U2(dir, bwdIdx, nbrPty);
      }

      shfU = conj(U) * Usrc * U2;
    }
 
    arg->dst(arg->i_dst, x_cb, pty) = shfU;
  }
  //------------------------------------------------------------------------------------------


} //- namespace quda

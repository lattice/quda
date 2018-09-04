/* C. Kallidonis: File that contains kernels for shifting
 * quark propagators and gauge fields, as required by
 * TMD contractions
 * August 2018
 */

#include <qlua_contract_shifts.cuh>

namespace quda {

  //-- Helper functions

  inline static __device__ int coord2pty4d(const int x[]){
    return (x[0] + x[1] + x[2] + x[3]) % 2;
  }

  inline static __device__ void putLinkGFext(GaugeU &F, int mu,
					     const int coord[], int pty,
					     Link &src,
					     const int dimEx[], const int brd[]){
    int c2[4] = {0,0,0,0};
    for (int i=0;i<4;i++) c2[i] = coord[i] + brd[i];

    F(mu, linkIndex(c2, dimEx), pty) = src; //-- NOTE: CHECK
  }

  inline static __device__ Link getLinkShiftGFext(GaugeU &F, int mu,
						  const int coord[], int pty,
						  const int dx[],
						  const int dimEx[], const int brd[]){
    int newPty = coord2pty4d(dx) ? pty : 1 - pty;
    int c2[4] = {0,0,0,0};
    for (int i=0;i<4;i++) c2[i] = coord[i] + brd[i];

    return F(mu, linkIndexShift(c2, dx, dimEx), newPty);
  }

  inline static __device__ Link getLinkNbrShiftGFext(GaugeU &F, int dir,
						     const int coord[], int pty,
						     const int dx[],
						     qcTMD_ShiftSgn qcShfSgn,
						     const int dimEx[], const int brd[]){
    int dx1[4] = {0,0,0,0};
    int c2[4]  = {0,0,0,0};
    for (int i=0;i<4;i++){
      dx1[i] = dx[i];
      c2[i]  = coord[i] + brd[i];
    }
    if (qcShfSgnMinus == qcShfSgn) dx1[dir] -= 1;
    int newPty = coord2pty4d(dx1) ? pty : 1 - pty;
    
    if (qcShfSgnPlus == qcShfSgn)
      return F(dir, linkIndexShift(c2, dx, dimEx), newPty); //-- NOTE: Might replace with linkIndex(c2, dimEx)
    else
      return conj(F(dir, linkIndexShift(c2, dx1, dimEx), newPty)); //-- NOTE: Was (c2, dx, dimEx), replaced with (c2, dx1, dimEx)
  }
  
  inline static __device__ Link getLinkNbrGFext(GaugeU &F, int dir,
						const int coord[], int pty,
						qcTMD_ShiftSgn qcShfSgn,
						const int dimEx[], const int brd[]){
    int dx[] = {0,0,0,0};
    return getLinkNbrShiftGFext(F, dir, coord, pty, dx, qcShfSgn, dimEx, brd);
  }

  inline static __device__ Link getLinkNbrGF(GaugeU &F, int dir,
					     const int coord[], int pty,
					     qcTMD_ShiftSgn shfSgn,
					     const int dim[], const int commDim[], const int nFace){
    int nbrPty = 1 - pty;
    
    Link shfU;
    if (shfSgn == qcShfSgnPlus) {
      shfU = F(dir, linkIndex(coord, dim), pty);
    }
    else if(shfSgn == qcShfSgnMinus){
      if (commDim[dir] && (coord[dir] - nFace < 0)) {
	const int ghostIdx = ghostFaceIndex<0>(coord, dim, dir, nFace);
        shfU = conj(F.Ghost(dir, ghostIdx, nbrPty));
      }
      else{
        const int bwdIdx = linkIndexM1(coord, dim, dir);
        shfU = conj(F(dir, bwdIdx, nbrPty));
      }
    }
    return shfU;
  }

  inline static __device__ Vector getSiteShiftCS(Propagator &F, const int coord[], int pty, int dir, qcTMD_ShiftSgn shfSgn,
						 const int dim[], const int commDim[], const int nFace){
    int nbrPty = 1 - pty;

    Vector shfU;
    if (shfSgn == qcShfSgnPlus){
      if (commDim[dir] && (coord[dir] + nFace >= dim[dir]) ) {
	const int ghostIdx = ghostFaceIndex<1>(coord, dim, dir, nFace);
        shfU = F.Ghost(dir, 1, ghostIdx, nbrPty);
      }
      else{
	const int fwdIdx = linkIndexP1(coord, dim, dir);
        shfU = F(fwdIdx, nbrPty);
      }
    }
    else if(shfSgn == qcShfSgnMinus){
      if (commDim[dir] && (coord[dir] - nFace < 0)) {
	const int ghostIdx = ghostFaceIndex<0>(coord, dim, dir, nFace);
        shfU = F.Ghost(dir, 0, ghostIdx, nbrPty);
      }
      else{
        const int bwdIdx = linkIndexM1(coord, dim, dir);
        shfU = F(bwdIdx, nbrPty);
      }
    }
    return shfU;
  }

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  //-- Kernels for GaugeField Shifts

  //-- Main kernel to perform a non-covariant shift of an extended(!) gauge field
  __global__ void ShiftGauge_nonCov_kernel(Arg_ShiftGauge_nonCov *arg, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    int coord[4];
    getCoords(coord, x_cb, arg->dim, pty);

    int dir = (int)shfDir; //-- Direction of the shift
    int dx[4] = {0,0,0,0};
    dx[dir] = qcShfSgnPlus == shfSgn ? 1 : -1;

    for (int mu=0;mu<4;mu++){
      Link shfU = getLinkShiftGFext(arg->src, mu, coord, pty, dx, arg->dimEx, arg->brd);  //- shfU = U_\mu(x +- d)
      putLinkGFext(arg->dst, mu, coord, pty, shfU, arg->dimEx, arg->brd);                 //- arg->dst = U_\mu(x +- d)
    }
  }


  //-- Main kernel to perform a covariant shift of one dimension of an extended(!) gauge field
  __global__ void ShiftLink_Cov_kernel(Arg_ShiftLink_Cov *arg, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    int coord[4];
    getCoords(coord, x_cb, arg->dim, pty);

    int dir = (int)shfDir; //-- Direction of the shift
    int dx[4] = {0,0,0,0};
    dx[dir] = qcShfSgnPlus == shfSgn ? 1 : -1;

    Link shfU = getLinkShiftGFext(arg->src, arg->i_src, coord, pty, dx, arg->dimEx, arg->brd);  //- shfU = U_src(x +- d)
    Link nbrU = getLinkNbrGFext(arg->gf_u, dir, coord, pty, shfSgn, arg->dimEx, arg->brd); //- nbrU = U_d(x) | U_d^\dag(x - d)
    Link tmpU = nbrU * shfU;
    putLinkGFext(arg->dst, arg->i_dst, coord, pty, tmpU, arg->dimEx, arg->brd);  //- arg-dst = nbrU * shfU
  }


  //-- Main kernel to perform an adjoin split covariant shift of an extended(!) gauge field
  __global__ void ShiftLink_AdjSplitCov_kernel(Arg_ShiftLink_AdjSplitCov *arg, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    int coord[4];
    getCoords(coord, x_cb, arg->dim, pty);

    int dir = (int)shfDir; //-- Direction of the shift
    int dx[] = {0,0,0,0,0};
    dx[dir] = qcShfSgnPlus == shfSgn ? 1 : -1;
    Link shfU  = getLinkShiftGFext(arg->src, arg->i_src, coord, pty, dx, arg->dimEx, arg->brd); //- shfU = U_src(x +- d)
    Link nbrU  = getLinkNbrGFext(arg->gf_u, dir, coord, pty, shfSgn, arg->dimEx, arg->brd);  //- nbrU  = U_d(x)  | U_d^\dag(x - d)
    Link nbrU2 = getLinkNbrGFext(arg->bsh_u, dir, coord, pty, shfSgn, arg->dimEx, arg->brd); //- nbrU2 = Ub_d(x) | Ub_d^\dag(x - d)
    Link tmpU = nbrU * shfU * conj(nbrU2);
    putLinkGFext(arg->dst, arg->i_dst, coord, pty, tmpU, arg->dimEx, arg->brd);
  }

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  //-- Kernels for ColorSpinorField Shifts

  //-- Main kernel to perform a non-covariant ColorSpinorField (vector) shift
  __global__ void ShiftCudaVec_nonCov_kernel(Arg_ShiftCudaVec_nonCov *arg, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    Vector shfVec = getSiteShiftCS(arg->src, coord, pty, dir, shfSgn, arg->dim, arg->commDim, arg->nFace);
    arg->dst(x_cb, pty) = shfVec; //- dst(x) = y(x+d) | y(x-d)
  }


  //-- Main kernel to perform a covariant ColorSpinorField (vector) shift
  __global__ void ShiftCudaVec_Cov_kernel(Arg_ShiftCudaVec_Cov *arg, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    int coord[5];
    getCoords(coord, x_cb, arg->dim, pty);
    coord[4] = 0;

    int dir = (int)shfDir; //-- Direction of the shift

    Vector shfVec = getSiteShiftCS(arg->src, coord, pty, dir, shfSgn, arg->dim, arg->commDim, arg->nFace);

    Link nbrU;
    if(arg->extendedGauge)
      nbrU = getLinkNbrGFext(arg->U, dir, coord, pty, shfSgn, arg->dimEx, arg->brd);
    else
      nbrU = getLinkNbrGF(arg->U, dir, coord, pty, shfSgn, arg->dim, arg->commDim, arg->nFace);

    arg->dst(x_cb, pty) = nbrU * shfVec;  //- dst(x) = U_d(x) * y(x+d) |  U_d^\dag(x - d) * y(x-d)
  }


} //- namespace quda

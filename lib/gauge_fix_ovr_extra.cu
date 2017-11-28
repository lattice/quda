#include <comm_quda.h>
#include <thrust_helper.cuh>
#include <gauge_fix_ovr_extra.h>

namespace quda {

#if defined(GPU_GAUGE_ALG) && defined(MULTI_GPU)

  struct BorderIdArg {
    int X[4]; // grid dimensions
    int border[4];
    BorderIdArg(int X_[4], int border_[4]) {
      for ( int dir = 0; dir < 4; ++dir ) border[dir] = border_[dir];
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = X_[dir];
    }
  };

  __global__ void ComputeBorderPointsActiveFaceIndex(BorderIdArg arg, int *faceindices, int facesize, int faceid, int parity){
    int idd = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idd < facesize ) {
      int borderid = 0;
      int idx = idd;
      if ( idx >= facesize / 2 ) {
        borderid = arg.X[faceid] - 1;
        idx -= facesize / 2;
      }
      int X[4];
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
      int x[4];
      int za, xodd;
      switch ( faceid ) {
      case 0: //X FACE
        za = idx / ( X[1] / 2);
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[0] = borderid;
        xodd = (borderid + x[2] + x[3] + parity) & 1;
        x[1] = (2 * idx + xodd)  - za * X[1];
        break;
      case 1: //Y FACE
        za = idx / ( X[0] / 2);
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[1] = borderid;
        xodd = (borderid + x[2] + x[3] + parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      case 2: //Z FACE
        za = idx / ( X[0] / 2);
        x[3] = za / X[1];
        x[1] = za - x[3] * X[1];
        x[2] = borderid;
        xodd = (borderid + x[1] + x[3] + parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      case 3: //T FACE
        za = idx / ( X[0] / 2);
        x[2] = za / X[1];
        x[1] = za - x[2] * X[1];
        x[3] = borderid;
        xodd = (borderid + x[1] + x[2] + parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]);;
      faceindices[idd] = idx;
    }
  }

  /**
  * @brief Pre-calculate lattice border points used by the gauge fixing with overrelaxation in multi-GPU implementation
   */
  void PreCalculateLatticeIndices(size_t faceVolume[4], size_t faceVolumeCB[4], int X[4], int border[4], \
                                  int &threads, int *borderpoints[2]){
    BorderIdArg arg(X, border);
    int nlinksfaces = 0;
    for ( int dir = 0; dir < 4; ++dir )
      if ( comm_dim_partitioned(dir)) nlinksfaces += faceVolume[dir];
    thrust::device_ptr<int> array_faceT[2];
    thrust::device_ptr<int> array_interiorT[2];
    for ( int i = 0; i < 2; i++ ) { //even and odd ids
      cudaMalloc(&borderpoints[i], nlinksfaces * sizeof(int) );
      cudaMemset(borderpoints[i], 0, nlinksfaces * sizeof(int) );
      array_faceT[i] = thrust::device_pointer_cast(borderpoints[i]);
    }
    dim3 nthreads(128, 1, 1);
    int start = 0;
    for ( int dir = 0; dir < 4; ++dir ) {
      if ( comm_dim_partitioned(dir)) {
        dim3 blocks((faceVolume[dir] + nthreads.x - 1) / nthreads.x,1,1);
        for ( int oddbit = 0; oddbit < 2; oddbit++ )
          ComputeBorderPointsActiveFaceIndex << < blocks, nthreads >> > (arg, borderpoints[oddbit] + start, faceVolume[dir], dir, oddbit);
        start += faceVolume[dir];
      }
    }
    int size[2];
    for ( int i = 0; i < 2; i++ ) {
      //sort and remove duplicated lattice indices
      thrust::sort(array_faceT[i], array_faceT[i] + nlinksfaces);
      thrust::device_ptr<int> new_end = thrust::unique(array_faceT[i], array_faceT[i] + nlinksfaces);
      size[i] = thrust::raw_pointer_cast(new_end) - thrust::raw_pointer_cast(array_faceT[i]);
    }
    if ( size[0] == size[1] ) threads = size[0];
    else errorQuda("BORDER: Even and Odd sizes does not match, not supported!!!!, %d:%d",size[0],size[1]);
  }

#endif // GPU_GAUGE_ALG && MULTI_GPU

}


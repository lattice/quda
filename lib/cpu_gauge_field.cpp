#include <gauge_field.h>
#include <face_quda.h>

cpuGaugeField::cpuGaugeField(const GaugeFieldParam &param) : 
  GaugeField(param, QUDA_CPU_FIELD_LOCATION) {

  if (reconstruct != QUDA_RECONSTRUCT_NO)
    errorQuda("Reconstruction type %d not supported", recoconstruct);

  for (int d=0; d<nDim; d++) {
    if (create == QUDA_NULL_FIELD_CREATE) {
      gauge[d] = malloc(volume[d] * reconstruct * precision * 4);
    } else if (create == QUDA_REFERENCE_FIELD_CREATE) {
      gauge[d] = param.gauge[d];
    } else {
      errorQuda("Unsupported creation type %d", create);
    }
    ghost[d] = malloc(nFace * surface[d] * reconstruct * precision);
  }

}

cpuGaugeField::~cpuGaugeField() {

  for (int d=0; d<nDim; d++) {
    if (create == QUDA_NULL_FIELD_CREATE) free(gauge[d]);
    if (ghost[d]) free(ghost[d]);
  }

}

template <typename Float>
void packGhost(Float **cpuLink, Float **cpuGhost, int nFace) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];

  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];
  
  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];


  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };

  for(int dir =0; dir < 4; dir++)
    {
      Float* even_src = cpuLink[dir];
      Float* odd_src = cpuLink[dir] + volumeCB*gaugeSiteSize;

      Float* even_dst;
      Float* odd_dst;
     
     //switching odd and even ghost cpuLink when that dimension size is odd
     //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
        even_dst = cpuGhost[dir];
        odd_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
     }else{
	even_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
        odd_dst = cpuGhost[dir];
     }

      int even_dst_index = 0;
      int odd_dst_index = 0;

      int d;
      int a,b,c;
      for(d = X[dir]- nFace; d < X[dir]; d++){
        for(a = 0; a < A[dir]; a++){
          for(b = 0; b < B[dir]; b++){
            for(c = 0; c < C[dir]; c++){
              int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
              int oddness = (a+b+c+d)%2;
              if (oddness == 0){ //even
                for(int i=0;i < 18;i++){
                  even_dst[18*even_dst_index+i] = even_src[18*index + i];
                }
                even_dst_index++;
              }else{ //odd
                for(int i=0;i < 18;i++){
                  odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
                }
                odd_dst_index++;
              }
            }//c
          }//b
        }//a
      }//d

      assert( even_dst_index == nFace*faceVolumeCB[dir]);
      assert( odd_dst_index == nFace*faceVolumeCB[dir]);
    }

}

// This does the exchange of the gauge field ghost zone and places it
// into the ghost array.
void cpuGaugeField::exchangeGhost() {
  void *send[QUDA_MAX_DIM];

  for (int d=0; d<nDim; d++) {
    send[d] = malloc(nFace * surface[d] * reconstruct * precision);
  }

  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhost((double**)gauge, (double**)send, nFace);
  } else {
    packGhost((float**)gauge, (float**)send, nFace);
  }

  FaceBuffer faceBuf(x, nDim, reconstruct, nFace, precision);
  faceBuf.exchangeCpuLink(ghost, send);

  for (int d=0; d<nDim; d++) free(send[d]);
}

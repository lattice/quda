#include <gauge_field.h>
#include <face_quda.h>
#include <assert.h>
#include <string.h>

cpuGaugeField::cpuGaugeField(const GaugeFieldParam &param) : 
  GaugeField(param, QUDA_CPU_FIELD_LOCATION), pinned(param.pinned) {

  if (reconstruct != QUDA_RECONSTRUCT_NO && 
      reconstruct != QUDA_RECONSTRUCT_10)
    errorQuda("Reconstruction type %d not supported", reconstruct);

  if (reconstruct == QUDA_RECONSTRUCT_10 && order != QUDA_MILC_GAUGE_ORDER)
    errorQuda("10 reconstruction only supported with MILC gauge order");

  if (order == QUDA_QDP_GAUGE_ORDER) {
    gauge = (void**)malloc(nDim * sizeof(void*));

    for (int d=0; d<nDim; d++) {
      if (create == QUDA_NULL_FIELD_CREATE 
	  || create == QUDA_ZERO_FIELD_CREATE) {
	if(pinned){
	  cudaMallocHost(&gauge[d], volume * reconstruct * precision);
	}else{
	  gauge[d] = malloc(volume * reconstruct * precision);
	}
	
	if(create == QUDA_ZERO_FIELD_CREATE){
	  memset(gauge[d], 0, volume * reconstruct * precision);
	}
      } else if (create == QUDA_REFERENCE_FIELD_CREATE) {
	gauge[d] = ((void**)param.gauge)[d];
      } else {
	errorQuda("Unsupported creation type %d", create);
      }
    }
    
  } else if (order == QUDA_CPS_WILSON_GAUGE_ORDER || 
	     order == QUDA_MILC_GAUGE_ORDER) {
    if (create == QUDA_NULL_FIELD_CREATE ||
	create == QUDA_ZERO_FIELD_CREATE) {
      if(pinned){
	cudaMallocHost(&(gauge), nDim*volume*reconstruct*precision);
      }else{
	gauge = (void**)malloc(nDim * volume * reconstruct * precision);
      }
      if(create == QUDA_ZERO_FIELD_CREATE){
	memset(gauge, 0, nDim*volume * reconstruct * precision);
      }
    } else if (create == QUDA_REFERENCE_FIELD_CREATE) {
      gauge = (void**)param.gauge;
    } else {
      errorQuda("Unsupported creation type %d", create);
    }
  } else {
  errorQuda("Unsupported gauge order type %d", order);
  }
  
  // Ghost zone is always 2-dimensional
  ghost = (void**)malloc(sizeof(void*)*QUDA_MAX_DIM);
  for (int i=0; i<nDim; i++) {
    if(pinned){
      cudaMallocHost(&ghost[i], nFace * surface[i] * reconstruct * precision);
    }else{
      ghost[i] = malloc(nFace * surface[i] * reconstruct * precision);
    }
  }
  
}

cpuGaugeField::~cpuGaugeField() {

  if (create == QUDA_NULL_FIELD_CREATE  
      || create == QUDA_ZERO_FIELD_CREATE  ) {
    if (order == QUDA_QDP_GAUGE_ORDER){
      for (int d=0; d<nDim; d++) {
	if(pinned){
	  if (gauge[d]) cudaFreeHost(gauge[d]);	  
	}else{
	  if (gauge[d]) free(gauge[d]);
	}
      }
      if (gauge) free(gauge);
    }else{      
      if(pinned){
	  if (gauge) cudaFreeHost(gauge);	  
	}else{
	  if (gauge) free(gauge);
	}
    }
  }else{ // QUDA_REFERENCE_FIELD_CREATE 
    if (order == QUDA_QDP_GAUGE_ORDER){
      free(gauge);
    }
  }
  
  
  for (int i=0; i<nDim; i++) {
    if(pinned){
      if (ghost[i]) cudaFreeHost(ghost[i]);
    }else{
      if (ghost[i]) free(ghost[i]);
    }
  }
  free(ghost);
  
}

template <typename Float>
void packGhost(Float **gauge, Float **ghost, const int nFace, const int *X, 
	       const int volumeCB, const int *surfaceCB) {
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
      Float* even_src = gauge[dir];
      Float* odd_src = gauge[dir] + volumeCB*gaugeSiteSize;

      Float* even_dst;
      Float* odd_dst;
     
     //switching odd and even ghost gauge when that dimension size is odd
     //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
        even_dst = ghost[dir];
        odd_dst = ghost[dir] + nFace*surfaceCB[dir]*gaugeSiteSize;	
     }else{
	even_dst = ghost[dir] + nFace*surfaceCB[dir]*gaugeSiteSize;
        odd_dst = ghost[dir];
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

      assert( even_dst_index == nFace*surfaceCB[dir]);
      assert( odd_dst_index == nFace*surfaceCB[dir]);
    }

}

// This does the exchange of the gauge field ghost zone and places it
// into the ghost array.
// This should be optimized so it is reused if called multiple times
void cpuGaugeField::exchangeGhost() const {
  void **send = (void**)malloc(sizeof(void*)*QUDA_MAX_DIM);

  for (int d=0; d<nDim; d++) {
    send[d] = malloc(nFace * surface[d] * reconstruct * precision);
  }

  // get the links into a contiguous buffer
  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhost((double**)gauge, (double**)send, nFace, x, volumeCB, surfaceCB);
  } else {
    packGhost((float**)gauge, (float**)send, nFace, x, volumeCB, surfaceCB);
  }

  // communicate between nodes
  FaceBuffer faceBuf(x, nDim, reconstruct, nFace, precision);
  faceBuf.exchangeCpuLink(ghost, send);

  for (int i=0; i<4; i++) {
    double sum = 0.0;
    for (int j=0; j<nFace*surface[i]*reconstruct; j++) {
      sum += ((double*)(ghost[i]))[j];
    }
  }

  for (int d=0; d<nDim; d++) free(send[d]);
  free(send);
}

void cpuGaugeField::setGauge(void**_gauge)
{
  if(create != QUDA_REFERENCE_FIELD_CREATE){
    errorQuda("Setting gauge pointer is only allowed when cpu gauge"
	      "is of QUDA_REFERENCE_FIELD_CREATE type\n");
  }
  gauge= _gauge;
}

#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>
#include <sys/time.h>
#include <mpicomm.h>

using namespace std;

cudaStream_t *stream;

FaceBuffer::FaceBuffer(const int *X, const int nDim, const int Ninternal, 
		       const int nFace, const QudaPrecision precision) : 
  Ninternal(Ninternal), precision(precision), nDim(nDim), nFace(nFace)
{
  setupDims(X);

  // set these both = 0 `for no overlap of qmp and cudamemcpyasync
  // sendBackStrmIdx = 0, and sendFwdStrmIdx = 1 for overlap
  sendBackStrmIdx = 0;
  sendFwdStrmIdx = 1;
  recFwdStrmIdx = sendBackStrmIdx;
  recBackStrmIdx = sendFwdStrmIdx;


  memset(send_request1, 0, sizeof(send_request1));
  memset(send_request1, 0, sizeof(recv_request1));
  memset(send_request1, 0, sizeof(send_request2));
  memset(send_request1, 0, sizeof(recv_request1));
  for(int dir =0 ; dir < 4;dir++){
    nbytes[dir] = nFace*faceVolumeCB[dir]*Ninternal*precision;
    if (precision == QUDA_HALF_PRECISION) nbytes[dir] += nFace*faceVolumeCB[dir]*sizeof(float);
    
    cudaMallocHost((void**)&fwd_nbr_spinor_sendbuf[dir], nbytes[dir]); CUERR;
    cudaMallocHost((void**)&back_nbr_spinor_sendbuf[dir], nbytes[dir]); CUERR;
    
    if (fwd_nbr_spinor_sendbuf[dir] == NULL || back_nbr_spinor_sendbuf[dir] == NULL)
      errorQuda("dir =%d, malloc failed for fwd_nbr_spinor_sendbuf/back_nbr_spinor_sendbuf", dir); 
    
    cudaMallocHost((void**)&fwd_nbr_spinor[dir], nbytes[dir]); CUERR;
    cudaMallocHost((void**)&back_nbr_spinor[dir], nbytes[dir]); CUERR;
    
    if (fwd_nbr_spinor[dir] == NULL || back_nbr_spinor[dir] == NULL)
      errorQuda("malloc failed for fwd_nbr_spinor/back_nbr_spinor"); 

    pageable_fwd_nbr_spinor_sendbuf[dir] = malloc(nbytes[dir]);
    pageable_back_nbr_spinor_sendbuf[dir] = malloc(nbytes[dir]);
    
    if (pageable_fwd_nbr_spinor_sendbuf[dir] == NULL || pageable_back_nbr_spinor_sendbuf[dir] == NULL)
      errorQuda("malloc failed for pageable_fwd_nbr_spinor_sendbuf/pageable_back_nbr_spinor_sendbuf");
    
    pageable_fwd_nbr_spinor[dir]=malloc(nbytes[dir]);
    pageable_back_nbr_spinor[dir]=malloc(nbytes[dir]);
    
    if (pageable_fwd_nbr_spinor[dir] == NULL || pageable_back_nbr_spinor[dir] == NULL)
      errorQuda("malloc failed for pageable_fwd_nbr_spinor/pageable_back_nbr_spinor"); 

  }
  
  return;
}

FaceBuffer::FaceBuffer(const FaceBuffer &face) {
  errorQuda("FaceBuffer copy constructor not implemented");
}

// X here is a checkboarded volume
void FaceBuffer::setupDims(const int* X)
{
  Volume = 1;
  for (int d=0; d<nDim; d++) {
    this->X[d] = X[d];
    Volume *= this->X[d];    
  }
  VolumeCB = Volume/2;

  for (int i=0; i<nDim; i++) {
    faceVolume[i] = 1;
    for (int j=0; j<nDim; j++) {
      if (i==j) continue;
      faceVolume[i] *= this->X[j];
    }
    faceVolumeCB[i] = faceVolume[i]/2;
  }
}

FaceBuffer::~FaceBuffer()
{
  for(int dir =0; dir < 4; dir++){
    if(fwd_nbr_spinor_sendbuf[dir]) cudaFreeHost(fwd_nbr_spinor_sendbuf[dir]);
    if(back_nbr_spinor_sendbuf[dir]) cudaFreeHost(back_nbr_spinor_sendbuf[dir]);
    if(fwd_nbr_spinor[dir]) cudaFreeHost(fwd_nbr_spinor[dir]);
    if(back_nbr_spinor[dir]) cudaFreeHost(back_nbr_spinor[dir]);
  }
}

void FaceBuffer::exchangeFacesStart(cudaColorSpinorField &in, int parity,
				    int dagger, cudaStream_t *stream_p)
{
  in.allocateGhostBuffer();   // allocate the ghost buffer if not yet allocated

  stream = stream_p;

  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  
  for(int dir = 0; dir  < 4; dir++){
    if(!commDimPartitioned(dir)){
      continue;
    }
    // Prepost all receives
    recv_request1[dir] = comm_recv_with_tag(pageable_back_nbr_spinor[dir], nbytes[dir], back_nbr[dir], uptags[dir]);
    recv_request2[dir] = comm_recv_with_tag(pageable_fwd_nbr_spinor[dir], nbytes[dir], fwd_nbr[dir], downtags[dir]);

    // gather for backwards send
    in.packGhost(back_nbr_spinor_sendbuf[dir], dir, QUDA_BACKWARDS, 
		 (QudaParity)parity, dagger, &stream[2*dir + sendBackStrmIdx]); CUERR;  

    // gather for forwards send
    in.packGhost(fwd_nbr_spinor_sendbuf[dir], dir, QUDA_FORWARDS, 
		 (QudaParity)parity, dagger, &stream[2*dir + sendFwdStrmIdx]); CUERR;
  }
}

void FaceBuffer::exchangeFacesComms(int dir) 
{
  
  if(!commDimPartitioned(dir)){
    return;
  }

  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};


  cudaStreamSynchronize(stream[2*dir + sendBackStrmIdx]); //required the data to be there before sending out
  memcpy(pageable_back_nbr_spinor_sendbuf[dir], back_nbr_spinor_sendbuf[dir], nbytes[dir]);
  send_request2[dir] = comm_send_with_tag(pageable_back_nbr_spinor_sendbuf[dir], nbytes[dir], back_nbr[dir], downtags[dir]);
    
  cudaStreamSynchronize(stream[2*dir + sendFwdStrmIdx]); //required the data to be there before sending out
  memcpy(pageable_fwd_nbr_spinor_sendbuf[dir], fwd_nbr_spinor_sendbuf[dir], nbytes[dir]);
  send_request1[dir]= comm_send_with_tag(pageable_fwd_nbr_spinor_sendbuf[dir], nbytes[dir], fwd_nbr[dir], uptags[dir]);
  
} 


void FaceBuffer::exchangeFacesWait(cudaColorSpinorField &out, int dagger, int dir)
{
  if(!commDimPartitioned(dir)){
    return;
  }
  
  comm_wait(recv_request2[dir]);  
  comm_wait(send_request2[dir]);

  memcpy(fwd_nbr_spinor[dir], pageable_fwd_nbr_spinor[dir], nbytes[dir]);
  out.unpackGhost(fwd_nbr_spinor[dir], dir, QUDA_FORWARDS,  dagger, &stream[2*dir + recFwdStrmIdx]); CUERR;

  comm_wait(recv_request1[dir]);
  comm_wait(send_request1[dir]);

  memcpy(back_nbr_spinor[dir], pageable_back_nbr_spinor[dir], nbytes[dir]);  
  out.unpackGhost(back_nbr_spinor[dir], dir, QUDA_BACKWARDS,  dagger, &stream[2*dir + recBackStrmIdx]); CUERR;
}

void FaceBuffer::exchangeCpuSpinor(cpuColorSpinorField &spinor, int oddBit, int dagger)
{

  //for all dimensions
  int len[4] = {
    nFace*faceVolumeCB[0]*Ninternal*precision,
    nFace*faceVolumeCB[1]*Ninternal*precision,
    nFace*faceVolumeCB[2]*Ninternal*precision,
    nFace*faceVolumeCB[3]*Ninternal*precision
  };

  // allocate the ghost buffer if not yet allocated
  spinor.allocateGhostBuffer();

  for(int i=0;i < 4; i++){
    spinor.packGhost(spinor.backGhostFaceSendBuffer[i], i, QUDA_BACKWARDS, (QudaParity)oddBit, dagger);
    spinor.packGhost(spinor.fwdGhostFaceSendBuffer[i], i, QUDA_FORWARDS, (QudaParity)oddBit, dagger);
  }

  unsigned long recv_request1[4], recv_request2[4];
  unsigned long send_request1[4], send_request2[4];
  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  
  for(int i= 0;i < 4; i++){
    recv_request1[i] = comm_recv_with_tag(spinor.backGhostFaceBuffer[i], len[i], back_nbr[i], uptags[i]);
    recv_request2[i] = comm_recv_with_tag(spinor.fwdGhostFaceBuffer[i], len[i], fwd_nbr[i], downtags[i]);    
    send_request1[i]= comm_send_with_tag(spinor.fwdGhostFaceSendBuffer[i], len[i], fwd_nbr[i], uptags[i]);
    send_request2[i] = comm_send_with_tag(spinor.backGhostFaceSendBuffer[i], len[i], back_nbr[i], downtags[i]);
  }

  for(int i=0;i < 4;i++){
    comm_wait(recv_request1[i]);
    comm_wait(recv_request2[i]);
    comm_wait(send_request1[i]);
    comm_wait(send_request2[i]);
  }

}


void FaceBuffer::exchangeCpuLink(void** ghost_link, void** link_sendbuf) {
  int uptags[4] = {XUP, YUP, ZUP,TUP};
  int fwd_nbrs[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR, T_FWD_NBR};
  int back_nbrs[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR, T_BACK_NBR};

  for(int dir =0; dir < 4; dir++)
    {
      int len = 2*nFace*faceVolumeCB[dir]*Ninternal;
      unsigned long recv_request = 
	comm_recv_with_tag(ghost_link[dir], len*precision, back_nbrs[dir], uptags[dir]);
      unsigned long send_request = 
	comm_send_with_tag(link_sendbuf[dir], len*precision, fwd_nbrs[dir], uptags[dir]);
      comm_wait(recv_request);
      comm_wait(send_request);
    }
}


void reduceDouble(double &sum) {

#ifdef MPI_COMMS
  comm_allreduce(&sum);
#endif

}

void reduceDoubleArray(double *sum, const int len) {

#ifdef MPI_COMMS
  comm_allreduce_array(sum, len);
#endif

}

int commDim(int dir) { return comm_dim(dir); }

int commCoords(int dir) { return comm_coords(dir); }

int commDimPartitioned(int dir){ return comm_dim_partitioned(dir);}

void commDimPartitionedSet(int dir) { comm_dim_partitioned_set(dir);}

void commBarrier() { comm_barrier(); }

#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>
#include <sys/time.h>
#include <mpicomm.h>

using namespace std;

#ifdef GPU_STAGGERED_DIRAC
static int dir_start = 0;
#else
static int dir_start = 3;
#endif

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

    pagable_fwd_nbr_spinor_sendbuf[dir] = malloc(nbytes[dir]);
    pagable_back_nbr_spinor_sendbuf[dir] = malloc(nbytes[dir]);
    
    if (pagable_fwd_nbr_spinor_sendbuf[dir] == NULL || pagable_back_nbr_spinor_sendbuf[dir] == NULL)
      errorQuda("malloc failed for pagable_fwd_nbr_spinor_sendbuf/pagable_back_nbr_spinor_sendbuf");
    
    pagable_fwd_nbr_spinor[dir]=malloc(nbytes[dir]);
    pagable_back_nbr_spinor[dir]=malloc(nbytes[dir]);
    
    if (pagable_fwd_nbr_spinor[dir] == NULL || pagable_back_nbr_spinor[dir] == NULL)
      errorQuda("malloc failed for pagable_fwd_nbr_spinor/pagable_back_nbr_spinor"); 

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
#ifdef DSLASH_PROFILE
static cudaEvent_t pack_start[4][2], pack_stop[4][2];
float first_memcpy_time[2];
#endif


void FaceBuffer::exchangeFacesStart(cudaColorSpinorField &in, int parity,
				    int dagger, cudaStream_t *stream_p)
{
  stream = stream_p;

  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  
#ifdef DSLASH_PROFILE  
  for(int dir=0;dir < 4; dir++){
    for(int i =0;i < 2; i++){
      cudaEventCreate(&pack_start[dir][i]);
      cudaEventCreate(&pack_stop[dir][i]);
    }
  }
#endif  
  
  for(int dir = dir_start; dir  < 4; dir++){
    if(!commDimPartitioned(dir)){
      continue;
    }
    // Prepost all receives
    recv_request1[dir] = comm_recv_with_tag(pagable_back_nbr_spinor[dir], nbytes[dir], back_nbr[dir], uptags[dir]);
    recv_request2[dir] = comm_recv_with_tag(pagable_fwd_nbr_spinor[dir], nbytes[dir], fwd_nbr[dir], downtags[dir]);
#ifdef DSLASH_PROFILE    
    cudaEventRecord(pack_start[dir][0], stream[2*dir + sendBackStrmIdx]);
#endif
    // gather for backwards send
    in.packGhost(back_nbr_spinor_sendbuf[dir], dir, QUDA_BACKWARDS, 
		 (QudaParity)parity, dagger, &stream[2*dir + sendBackStrmIdx]); CUERR;  
#ifdef DSLASH_PROFILE
    cudaEventRecord(pack_stop[dir][0], stream[2*dir + sendBackStrmIdx]);    
    cudaEventRecord(pack_start[dir][1], stream[2*dir + sendFwdStrmIdx]);
#endif
    // gather for forwards send
    in.packGhost(fwd_nbr_spinor_sendbuf[dir], dir, QUDA_FORWARDS, 
		 (QudaParity)parity, dagger, &stream[2*dir + sendFwdStrmIdx]); CUERR;
#ifdef DSLASH_PROFILE
    cudaEventRecord(pack_stop[dir][1], stream[2*dir + sendFwdStrmIdx]);
#endif

  }
}


void FaceBuffer::exchangeFacesComms(int dir) 
{
  
  if(!commDimPartitioned(dir)){
    return;
  }

#ifdef DSLASH_PROFILE
  struct timeval memcpy_start[2], memcpy_stop[2];
#endif

  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};


  cudaStreamSynchronize(stream[2*dir + sendBackStrmIdx]); //required the data to be there before sending out
#ifdef DSLASH_PROFILE
  gettimeofday(&memcpy_start[0], NULL);
#endif
  memcpy(pagable_back_nbr_spinor_sendbuf[dir], back_nbr_spinor_sendbuf[dir], nbytes[dir]);
#ifdef DSLASH_PROFILE
  gettimeofday(&memcpy_stop[0], NULL);
#endif
  send_request2[dir] = comm_send_with_tag(pagable_back_nbr_spinor_sendbuf[dir], nbytes[dir], back_nbr[dir], downtags[dir]);
    
  cudaStreamSynchronize(stream[2*dir + sendFwdStrmIdx]); //required the data to be there before sending out
#ifdef DSLASH_PROFILE  
  gettimeofday(&memcpy_start[1], NULL);
#endif
  memcpy(pagable_fwd_nbr_spinor_sendbuf[dir], fwd_nbr_spinor_sendbuf[dir], nbytes[dir]);
#ifdef DSLASH_PROFILE
  gettimeofday(&memcpy_stop[1], NULL);
#endif
  send_request1[dir]= comm_send_with_tag(pagable_fwd_nbr_spinor_sendbuf[dir], nbytes[dir], fwd_nbr[dir], uptags[dir]);

#ifdef DSLASH_PROFILE
  for(int i=0;i < 2;i++){
    first_memcpy_time[i] = (memcpy_stop[i].tv_sec - memcpy_start[i].tv_sec)*1e+3
      + (memcpy_stop[i].tv_usec - memcpy_start[i].tv_usec)*1e-3;
  }

#endif
  

  
} 

void FaceBuffer::exchangeFacesWait(cudaColorSpinorField &out, int dagger, int dir)
{
  if(!commDimPartitioned(dir)){
    return;
  }

#ifdef DSLASH_PROFILE
  static cudaEvent_t unpack_start[2], unpack_stop[2];  
  for(int i =0;i < 2; i++){
    cudaEventCreate(&unpack_start[i]);
    cudaEventCreate(&unpack_stop[i]);
  }

  struct timeval memcpy_start[2], memcpy_stop[2];
  struct timeval mpi_start[2], mpi_stop[2];

  gettimeofday(&mpi_start[0], NULL);
#endif


  
  comm_wait(recv_request2[dir]);  
  comm_wait(send_request2[dir]);

#ifdef DSLASH_PROFILE
  gettimeofday(&mpi_stop[0], NULL);
  gettimeofday(&memcpy_start[0], NULL);
#endif
  memcpy(fwd_nbr_spinor[dir], pagable_fwd_nbr_spinor[dir], nbytes[dir]);
#ifdef DSLASH_PROFILE  
  gettimeofday(&memcpy_stop[0], NULL);
  cudaEventRecord(unpack_start[0], stream[2*dir+recFwdStrmIdx]);
#endif
  out.unpackGhost(fwd_nbr_spinor[dir], dir, QUDA_FORWARDS,  dagger, &stream[2*dir + recFwdStrmIdx]); CUERR;
#ifdef DSLASH_PROFILE
  cudaEventRecord(unpack_stop[0], stream[2*dir+recFwdStrmIdx]);
  gettimeofday(&mpi_start[1], NULL);
#endif
  
  comm_wait(recv_request1[dir]);
  comm_wait(send_request1[dir]);
#ifdef DSLASH_PROFILE
  gettimeofday(&mpi_stop[1], NULL);
  gettimeofday(&memcpy_start[1], NULL);
#endif
  memcpy(back_nbr_spinor[dir], pagable_back_nbr_spinor[dir], nbytes[dir]);  
#ifdef DSLASH_PROFILE
  gettimeofday(&memcpy_stop[1], NULL);
  cudaEventRecord(unpack_start[1], stream[2*dir+recBackStrmIdx]);
#endif
  out.unpackGhost(back_nbr_spinor[dir], dir, QUDA_BACKWARDS,  dagger, &stream[2*dir + recBackStrmIdx]); CUERR;
#ifdef DSLASH_PROFILE
  cudaEventRecord(unpack_stop[1], stream[2*dir+recBackStrmIdx]);
#endif

#ifdef DSLASH_PROFILE

  float pack_time[2], unpack_time[2], second_memcpy_time[2], mpi_time[2];
  float total_time;
  for(int i=0;i < 2;i++){
    cudaEventElapsedTime(&pack_time[i], pack_start[dir][i], pack_stop[dir][i]);
    cudaEventElapsedTime(&unpack_time[i], unpack_start[i], unpack_stop[i]);
    second_memcpy_time[i] = (memcpy_stop[i].tv_sec - memcpy_start[i].tv_sec)*1e+3
      + (memcpy_stop[i].tv_usec - memcpy_start[i].tv_usec)*1e-3;
    mpi_time[i] = (mpi_stop[i].tv_sec - mpi_start[i].tv_sec)*1e+3
      + (mpi_stop[i].tv_usec - mpi_start[i].tv_usec)*1e-3;
  }
  total_time = pack_time[0] + pack_time[1] + first_memcpy_time[0] + first_memcpy_time[1]
		+ mpi_time[0] + mpi_time[1] + second_memcpy_time[0] + second_memcpy_time[1]
		+ unpack_time[0] + unpack_time[1];
  printfQuda("dir=%d, pack_time=%.2f, 1st_memcpy_time=%f, mpi_time=%.2f ms, 2nd memcpy_time=%.2f ms,  unpack_time=%.2f ms, total=%.2f ms\n", dir,
	     pack_time[0] + pack_time[1], 
	     first_memcpy_time[0] + first_memcpy_time[1],	     
	     mpi_time[0] + mpi_time[1],
	     second_memcpy_time[0] + second_memcpy_time[1],
	     unpack_time[0] + unpack_time[1], 
	     total_time);

  for(int i=0;i < 2;i++){
    //cudaEventDestroy(unpack_start[i]);
    //cudaEventDestroy(unpack_stop[i]);
    //cudaEventDestroy(pack_start[dir][i]);
    //cudaEventDestroy(pack_stop[dir][i]);
  }

#endif

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

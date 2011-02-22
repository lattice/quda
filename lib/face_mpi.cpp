#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>

#include <mpi_comms.h>

using namespace std;

cudaStream_t *stream;

#define COMM_ERROR(a) \
  printf(a);	      \
  comm_exit(1);

FaceBuffer::FaceBuffer(const int *X, const int nDim, const int Ninternal, const int nFaces, 
		       const QudaPrecision precision) : nDim(nDim), Ninternal(Ninternal), 
							nFace(nFace), precision(precision)
{
  setupDims(X);

  // set these both = 0 `for no overlap of qmp and cudamemcpyasync
  // sendBackStrmIdx = 0, and sendFwdStrmIdx = 1 for overlap
  sendBackStrmIdx = 0;
  sendFwdStrmIdx = 1;
  recFwdStrmIdx = sendBackStrmIdx;
  recBackStrmIdx = sendFwdStrmIdx;

  nbytes = nFace*faceVolumeCB[3]*Ninternal*precision;
  nbytes_norm = nFace*faceVolumeCB[3]*sizeof(float);
   
  cudaMallocHost((void**)&fwd_nbr_spinor_sendbuf, nbytes); CUERR;
  cudaMallocHost((void**)&back_nbr_spinor_sendbuf, nbytes); CUERR;
  if (fwd_nbr_spinor_sendbuf == NULL || back_nbr_spinor_sendbuf == NULL)
    COMM_ERROR("ERROR: malloc failed for fwd_nbr_spinor_sendbuf/back_nbr_spinor_sendbuf\n"); 
  
  cudaMallocHost((void**)&fwd_nbr_spinor, nbytes); CUERR;
  cudaMallocHost((void**)&back_nbr_spinor, nbytes); CUERR;
  if (fwd_nbr_spinor == NULL || back_nbr_spinor == NULL)
    COMM_ERROR("ERROR: malloc failed for fwd_nbr_spinor/back_nbr_spinor\n"); 
  
  pagable_fwd_nbr_spinor_sendbuf = malloc(nbytes);
  pagable_back_nbr_spinor_sendbuf = malloc(nbytes);
  if (pagable_fwd_nbr_spinor_sendbuf == NULL || pagable_back_nbr_spinor_sendbuf == NULL)
    COMM_ERROR("ERROR: malloc failed for pagable_fwd_nbr_spinor_sendbuf/pagable_back_nbr_spinor_sendbuf\n");
  
  pagable_fwd_nbr_spinor=malloc(nbytes);
  pagable_back_nbr_spinor=malloc(nbytes);
  if (pagable_fwd_nbr_spinor == NULL || pagable_back_nbr_spinor == NULL)
    COMM_ERROR("ERROR: malloc failed for pagable_fwd_nbr_spinor/pagable_back_nbr_spinor\n"); 
  
  if (cudaSpinor.Precision() == QUDA_HALF_PRECISION){
    cudaMallocHost(&f_norm_sendbuf, nbytes_norm);CUERR;
    cudaMallocHost(&b_norm_sendbuf, nbytes_norm);CUERR;
    if (f_norm_sendbuf == NULL || b_norm_sendbuf == NULL)
      COMM_ERROR("ERROR: malloc failed for b_norm_sendbuf/f_norm_sendbuf\n");
    
    cudaMallocHost(&f_norm, nbytes_norm);CUERR;
    cudaMallocHost(&b_norm, nbytes_norm);CUERR;
    if (f_norm== NULL || b_norm== NULL)
      COMM_ERROR("ERROR: malloc failed for b_norm/f_norm\n");
    
    pagable_f_norm_sendbuf=malloc(nbytes_norm);
    pagable_b_norm_sendbuf=malloc(nbytes_norm);
    if (pagable_f_norm_sendbuf == NULL || pagable_b_norm_sendbuf == NULL)
      COMM_ERROR("ERROR: malloc failed for pagable_b_norm_sendbuf/pagable_f_norm_sendbuf\n");

    pagable_f_norm=malloc(nbytes_norm);
    pagable_b_norm=malloc(nbytes_norm);
    if (pagable_f_norm== NULL || pagable_b_norm== NULL)
      COMM_ERROR("ERROR: malloc failed for pagable_b_norm/pagable_f_norm\n");
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
  for (int d=0; d< 4; d++) {
    this->X[d] = X[d];
    if (d==0) this->X[d] *= 2;
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
  if(fwd_nbr_spinor_sendbuf) cudaFreeHost(fwd_nbr_spinor_sendbuf);
  if(back_nbr_spinor_sendbuf) cudaFreeHost(back_nbr_spinor_sendbuf);
  if (f_norm_sendbuf) cudaFreeHost(f_norm_sendbuf);
  if (b_norm_sendbuf) cudaFreeHost(b_norm_sendbuf);
  if(fwd_nbr_spinor) cudaFreeHost(fwd_nbr_spinor);
  if(back_nbr_spinor) cudaFreeHost(back_nbr_spinor);
  if (f_norm) cudaFreeHost(f_norm);
  if (b_norm) cudaFreeHost(b_norm);
}

void FaceBuffer::exchangeStart(cudaColorSpinorField &in, int parity, 
			       int dagger, cudaStream_t* stream_p)
{
  stream = stream_p;
  
  // Prepost all receives
  recv_request1 = comm_recv(pagable_back_nbr_spinor, nbytes, BACK_NBR);
  recv_request2 = comm_recv(pagable_fwd_nbr_spinor, nbytes, FWD_NBR);
  if (precision == QUDA_HALF_PRECISION) {
    recv_request3 = comm_recv(pagable_b_norm, nbytes_norm, BACK_NBR);
    recv_request4 = comm_recv(pagable_f_norm, nbytes_norm, FWD_NBR);
  }

  // gather for backwards send
  in.packGhost(back_nbr_spinor_sendbuf, b_norm_sendbuf, 3, QUDA_BACKWARDS, 
	       (QudaParity)parity, dagger, &stream[sendBackStrmIdx]); CUERR;  

  // gather for forwards send
  in.packGhost(fwd_nbr_spinor_sendbuf, f_norm_sendbuf, 3, QUDA_FORWARDS, 
	       (QudaParity)parity, dagger, &stream[sendFwdStrmIdx]); CUERR;
}

void FaceBuffer::exchangeFacesComms() {
  cudaStreamSynchronize(stream[sendBackStrmIdx]); //required the data to be there before sending out

  memcpy(pagable_back_nbr_spinor_sendbuf, back_nbr_spinor_sendbuf, nbytes);
  send_request2 = comm_send(pagable_back_nbr_spinor_sendbuf, nbytes, BACK_NBR);
  if (precision == QUDA_HALF_PRECISION){
    memcpy(pagable_b_norm_sendbuf, b_norm_sendbuf, nbytes_norm);
    send_request4 = comm_send(pagable_b_norm_sendbuf, nbytes_norm, BACK_NBR);
  }

  cudaStreamSynchronize(stream[sendFwdStrmIdx]); //required the data to be there before sending out

  memcpy(pagable_fwd_nbr_spinor_sendbuf, fwd_nbr_spinor_sendbuf, nbytes);
  send_request1= comm_send(pagable_fwd_nbr_spinor_sendbuf, nbytes, FWD_NBR);
  if (precision == QUDA_HALF_PRECISION){
    memcpy(pagable_f_norm_sendbuf, f_norm_sendbuf, nbytes_norm);
    send_request3 = comm_send(pagable_f_norm_sendbuf, nbytes_norm, FWD_NBR);
  }
  
} 


void FaceBuffer::exchangeFacesWait(cudaColorSpinorField &out, int dagger)
{
  comm_wait(recv_request2);  
  comm_wait(send_request2);
  memcpy(fwd_nbr_spinor, pagable_fwd_nbr_spinor, nbytes);
  if (precision == QUDA_HALF_PRECISION){
    comm_wait(recv_request4);
    comm_wait(send_request4);
    memcpy(f_norm, pagable_f_norm, nbytes_norm);
  }

  out.unpackGhost(fwd_nbr_spinor, f_norm, 3, QUDA_FORWARDS,  dagger, &stream[recFwdStrmIdx]); CUERR;

  comm_wait(recv_request1);
  comm_wait(send_request1);
  memcpy(back_nbr_spinor, pagable_back_nbr_spinor, nbytes);

  if (precision == QUDA_HALF_PRECISION){
    comm_wait(recv_request3);
    comm_wait(send_request3);
    memcpy(b_norm, pagable_b_norm, nbytes_norm);
  }

  out.unpackGhost(back_nbr_spinor, b_norm, 3, QUDA_BACKWARDS,  dagger, &stream[recBeckStrmIdx]); CUERR;
  cudaStreamSynchronize(*mystream);
}

// This is just an initial hack for CPU comms
void FaceBuffer::exchangeCpuSpinor(void *spinorField, void **cpu_fwd_nbr_spinor, void **cpu_back_nbr_spinor, int oddBit)
{

  //for all dimensions
  int len[4] = {
    nFace*faceVolumeCB[0]*Ninternal*precision,
    nFace*faceVolumeCB[1]*Ninternal*precision,
    nFace*faceVolumeCB[2]*Ninternal*precision,
    nFace*faceVolumeCB[3]*Ninternal*precision
  };

  void* cpu_fwd_nbr_spinor_sendbuf[4];
  void* cpu_back_nbr_spinor_sendbuf[4];
  for(int i=0;i < 4;i++){
    cpu_fwd_nbr_spinor_sendbuf[i] = (void*)malloc(len[i]);
    cpu_back_nbr_spinor_sendbuf[i] = (void*)malloc(len[i]);
  }

  for( int i=0;i < Vh;i++){
    //compute full index
    int boundaryCrossings = i/(X[0]/2) + i/(X[0]*X[1]/2) + i/(X[0]*X[1]*X[2]/2);
    int Y = 2*i + (boundaryCrossings + oddBit) % 2;
    int x[4];
    x[3] = Y/(X[2]*X[1]*X[0]);
    x[2] = (Y/(X[1]*X[0])) % X[2];
    x[1] = (Y/X[0]) % X[1];
    x[0] = Y % X[0];

    int ghost_face_idx ;

    //X dimension
    if (x[0] < nFace){
      ghost_face_idx = (x[0]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] +x[1])>>1;
      memcpy(&cpu_back_nbr_spinor_sendbuf[0][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }
    if (x[0] >=X[0]-nFace){
      ghost_face_idx = ((x[0]-X[0]+nFace)*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] +x[1])>>1;
      memcpy(&cpu_fwd_nbr_spinor_sendbuf[0][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }

    //Y dimension
    if (x[1] < nFace){
      ghost_face_idx = (x[1]*X[3]*X[2]*X[0] + x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1;
      memcpy(&cpu_back_nbr_spinor_sendbuf[1][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }
    if (x[1] >= X[1]-nFace){
      ghost_face_idx = ((x[1]-X[1]+nFace)*X[3]*X[2]*X[0]+ x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1;
      memcpy(&cpu_fwd_nbr_spinor_sendbuf[1][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }

    //Z dimension
    if (x[2] < nFace){
      ghost_face_idx = (x[2]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
      memcpy(&cpu_back_nbr_spinor_sendbuf[2][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }
    if (x[2] >= X[2] - nFace){
      ghost_face_idx = ((x[2]-X[2]+nFace)*X[3]*X[1]*X[0] + x[3]*X[1]*X[0] + x[1]*X[0] + x[0])>>1;
      memcpy(&cpu_fwd_nbr_spinor_sendbuf[2][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }

    //T dimension
    if (x[3] < nFace){
      ghost_face_idx = (x[3]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
      memcpy(&cpu_back_nbr_spinor_sendbuf[3][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }
    if (x[3] >= X[3] - nFace){
      ghost_face_idx = ((x[3]-X[3]+nFace)*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
      memcpy(&cpu_fwd_nbr_spinor_sendbuf[3][Ninternal*ghost_face_idx], &spinorField[Ninternal*i], Ninternal*precision);
    }

  }//i

  unsigned long recv_request1[4], recv_request2[4];
  unsigned long send_request1[4], send_request2[4];
  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  
  for(int i=0;i < 4; i++){
    recv_request1[i] = comm_recv_with_tag(cpu_back_nbr_spinor[i], len[i], back_nbr[i], uptags[i]);
    recv_request2[i] = comm_recv_with_tag(cpu_fwd_nbr_spinor[i], len[i], fwd_nbr[i], downtags[i]);
    send_request1[i]= comm_send_with_tag(cpu_fwd_nbr_spinor_sendbuf[i], len[i], fwd_nbr[i], uptags[i]);
    send_request2[i] = comm_send_with_tag(cpu_back_nbr_spinor_sendbuf[i], len[i], back_nbr[i], downtags[i]);
  }

  for(int i=0;i < 4;i++){
    comm_wait(recv_request1[i]);
    comm_wait(recv_request2[i]);
    comm_wait(send_request1[i]);
    comm_wait(send_request2[i]);
  }

  for(int i=0;i < 4;i++){
    free(cpu_fwd_nbr_spinor_sendbuf[i]);
    free(cpu_back_nbr_spinor_sendbuf[i]);
  }
}

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

bool globalReduce = true;

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
				    int dagger, int dir, cudaStream_t *stream_p)
{
  if(!commDimPartitioned(dir)){
    return ;
  }

  in.allocateGhostBuffer();   // allocate the ghost buffer if not yet allocated
  
  stream = stream_p;
  
  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  
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

void reduceMaxDouble(double &max) {

#ifdef MPI_COMMS
  comm_allreduce_max(&max);
#endif

}
void reduceDouble(double &sum) {

#ifdef MPI_COMMS
  if (globalReduce) comm_allreduce(&sum);
#endif

}

void reduceDoubleArray(double *sum, const int len) {

#ifdef MPI_COMMS
  if (globalReduce) comm_allreduce_array(sum, len);
#endif

}

int commDim(int dir) { return comm_dim(dir); }

int commCoords(int dir) { return comm_coords(dir); }

int commDimPartitioned(int dir){ return comm_dim_partitioned(dir);}

void commDimPartitionedSet(int dir) { comm_dim_partitioned_set(dir);}

void commBarrier() { comm_barrier(); }



/**************************************************************
 * Staple exchange routine
 * used in fat link computation
 ***************************************************************/
#ifdef GPU_FATLINK

#define gaugeSiteSize 18

static void* fwd_nbr_staple_cpu = NULL;
static void* back_nbr_staple_cpu = NULL;
static void* fwd_nbr_staple_sendbuf_cpu = NULL;
static void* back_nbr_staple_sendbuf_cpu = NULL;

static void* fwd_nbr_staple = NULL;
static void* back_nbr_staple = NULL;
static void* fwd_nbr_staple_sendbuf = NULL;
static void* back_nbr_staple_sendbuf = NULL;

static int dims[4];
static int X1,X2,X3,X4;
static int V;
static int Vh;
static int Vs;
static int Vs_x, Vs_y, Vs_z, Vs_t;
static int Vsh_x, Vsh_y, Vsh_z, Vsh_t;

#include <gauge_quda.h>

static void
setup_dims(int* X)
{
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    dims[d] = X[d];
  }
  Vh = V/2;
  
  Vs = X[0]*X[1]*X[2];
  Vsh_t = Vs/2;

  X1=X[0];
  X2=X[1];
  X3=X[2];
  X4=X[3];


  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];

  Vsh_x = Vs_x/2;
  Vsh_y = Vs_y/2;
  Vsh_z = Vs_z/2;
  Vsh_t = Vs_t/2;


}

void 
exchange_llfat_init(FullStaple* cudaStaple)
{
  static int initialized = 0;
  if (initialized){
    return;
  }
  initialized = 1;
  
  QudaPrecision prec = cudaStaple->precision;

  cudaMallocHost((void**)&fwd_nbr_staple, Vs*gaugeSiteSize*prec);
  cudaMallocHost((void**)&back_nbr_staple, Vs*gaugeSiteSize*prec);
  cudaMallocHost((void**)&fwd_nbr_staple_sendbuf, Vs*gaugeSiteSize*prec);
  cudaMallocHost((void**)&back_nbr_staple_sendbuf, Vs*gaugeSiteSize*prec);

  CUERR;

  fwd_nbr_staple_cpu = malloc(Vs*gaugeSiteSize*prec);
  back_nbr_staple_cpu = malloc(Vs*gaugeSiteSize*prec);
  if (fwd_nbr_staple_cpu == NULL||back_nbr_staple_cpu == NULL){
    printf("ERROR: malloc failed for fwd_nbr_staple/back_nbr_staple\n");
    comm_exit(1);
  }
  
  fwd_nbr_staple_sendbuf_cpu = malloc(Vs*gaugeSiteSize*prec);
  back_nbr_staple_sendbuf_cpu = malloc(Vs*gaugeSiteSize*prec);
  if (fwd_nbr_staple_sendbuf_cpu == NULL || back_nbr_staple_sendbuf_cpu == NULL){
    printf("ERROR: malloc failed for fwd_nbr_staple_sendbuf/back_nbr_staple_sendbuf\n");
    comm_exit(1);
  }
  
  return;
}



template<typename Float>
void
exchange_sitelink(Float** sitelink, Float* ghost_sitelink, Float* sitelink_fwd_sendbuf, Float* sitelink_back_sendbuf)
{

  int i;
  int len = Vsh_t*gaugeSiteSize*sizeof(Float);
  for(i=0;i < 4;i++){
    Float* even_sitelink_back_src = sitelink[i];
    Float* odd_sitelink_back_src = sitelink[i] + Vh*gaugeSiteSize;
    Float* sitelink_back_dst = sitelink_back_sendbuf + 2*i*Vsh_t*gaugeSiteSize;

    if(dims[3] % 2 == 0){    
      memcpy(sitelink_back_dst, even_sitelink_back_src, len);
      memcpy(sitelink_back_dst + Vsh_t*gaugeSiteSize, odd_sitelink_back_src, len);
    }else{
      //switching odd and even ghost sitelink
      memcpy(sitelink_back_dst, odd_sitelink_back_src, len);
      memcpy(sitelink_back_dst + Vsh_t*gaugeSiteSize, even_sitelink_back_src, len);
    }
  }

  for(i=0;i < 4;i++){
    Float* even_sitelink_fwd_src = sitelink[i] + (Vh - Vsh_t)*gaugeSiteSize;
    Float* odd_sitelink_fwd_src = sitelink[i] + Vh*gaugeSiteSize + (Vh - Vsh_t)*gaugeSiteSize;
    Float* sitelink_fwd_dst = sitelink_fwd_sendbuf + 2*i*Vsh_t*gaugeSiteSize;
    if(dims[3] % 2 == 0){    
      memcpy(sitelink_fwd_dst, even_sitelink_fwd_src, len);
      memcpy(sitelink_fwd_dst + Vsh_t*gaugeSiteSize, odd_sitelink_fwd_src, len);
    }else{
      //switching odd and even ghost sitelink
      memcpy(sitelink_fwd_dst, odd_sitelink_fwd_src, len);
      memcpy(sitelink_fwd_dst + Vsh_t*gaugeSiteSize, even_sitelink_fwd_src, len);
    }
    
  }
  
  Float* ghost_sitelink_back = ghost_sitelink;
  Float* ghost_sitelink_fwd = ghost_sitelink + 8*Vsh_t*gaugeSiteSize;

  unsigned long recv_request1 = comm_recv(ghost_sitelink_back, 8*len, BACK_NBR);
  unsigned long recv_request2 = comm_recv(ghost_sitelink_fwd, 8*len, FWD_NBR);
  unsigned long send_request1 = comm_send(sitelink_fwd_sendbuf, 8*len, FWD_NBR);
  unsigned long send_request2 = comm_send(sitelink_back_sendbuf, 8*len, BACK_NBR);
  comm_wait(recv_request1);
  comm_wait(recv_request2);
  comm_wait(send_request1);
  comm_wait(send_request2);
}

//this function is used for link fattening computation
void exchange_cpu_sitelink(int* X,
			   void** sitelink, void* ghost_sitelink,
			   QudaPrecision gPrecision)
{
  

  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    dims[d] = X[d];
  }
  Vh = V/2;

  Vs = X[0]*X[1]*X[2];
  Vsh_t = Vs/2;

  void*  sitelink_fwd_sendbuf = malloc(4*Vs*gaugeSiteSize*gPrecision);
  void*  sitelink_back_sendbuf = malloc(4*Vs*gaugeSiteSize*gPrecision);
  if (sitelink_fwd_sendbuf == NULL|| sitelink_back_sendbuf == NULL){
    printf("ERROR: malloc failed for sitelink_sendbuf/site_link_back_sendbuf\n");
    exit(1);
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    exchange_sitelink((double**)sitelink, (double*)ghost_sitelink, 
		      (double*)sitelink_fwd_sendbuf, (double*)sitelink_back_sendbuf);
  }else{ //single
    exchange_sitelink((float**)sitelink, (float*)ghost_sitelink, 
		      (float*)sitelink_fwd_sendbuf, (float*)sitelink_back_sendbuf);
  }

  free(sitelink_fwd_sendbuf);
  free(sitelink_back_sendbuf);

}




template<typename Float>
void
exchange_staple(Float* staple, Float* ghost_staple, Float* staple_fwd_sendbuf, Float* staple_back_sendbuf)
{
  
  int len = Vsh_t*gaugeSiteSize*sizeof(Float);

  Float* even_staple_back_src = staple;
  Float* odd_staple_back_src = staple + Vh*gaugeSiteSize;
  Float* staple_back_dst = staple_back_sendbuf;
  
  if(dims[3] % 2 == 0){    
    memcpy(staple_back_dst, even_staple_back_src, len);
    memcpy(staple_back_dst + Vsh_t*gaugeSiteSize, odd_staple_back_src, len);
  }else{
    //switching odd and even ghost staple
    memcpy(staple_back_dst, odd_staple_back_src, len);
    memcpy(staple_back_dst + Vsh_t*gaugeSiteSize, even_staple_back_src, len);
  }
  
  
  Float* even_staple_fwd_src = staple + (Vh - Vsh_t)*gaugeSiteSize;
  Float* odd_staple_fwd_src = staple + Vh*gaugeSiteSize + (Vh - Vsh_t)*gaugeSiteSize;
  Float* staple_fwd_dst = staple_fwd_sendbuf;
  if(dims[3] % 2 == 0){    
    memcpy(staple_fwd_dst, even_staple_fwd_src, len);
    memcpy(staple_fwd_dst + Vsh_t*gaugeSiteSize, odd_staple_fwd_src, len);
  }else{
    //switching odd and even ghost staple
    memcpy(staple_fwd_dst, odd_staple_fwd_src, len);
    memcpy(staple_fwd_dst + Vsh_t*gaugeSiteSize, even_staple_fwd_src, len);
  }
  
  
  Float* ghost_staple_back = ghost_staple;
  Float* ghost_staple_fwd = ghost_staple + 2*Vsh_t*gaugeSiteSize;

  unsigned long recv_request1 = comm_recv(ghost_staple_back, 2*len, BACK_NBR);
  unsigned long recv_request2 = comm_recv(ghost_staple_fwd, 2*len, FWD_NBR);
  unsigned long send_request1 = comm_send(staple_fwd_sendbuf, 2*len, FWD_NBR);
  unsigned long send_request2 = comm_send(staple_back_sendbuf, 2*len, BACK_NBR);
  comm_wait(recv_request1);
  comm_wait(recv_request2);
  comm_wait(send_request1);
  comm_wait(send_request2);
}
//this function is used for link fattening computation
void exchange_cpu_staple(int* X,
			 void* staple, void* ghost_staple,
			 QudaPrecision gPrecision)
{
  
  
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    dims[d] = X[d];
  }
  Vh = V/2;
  
  Vs = X[0]*X[1]*X[2];
  Vsh_t = Vs/2;
  
  void*  staple_fwd_sendbuf = malloc(Vs*gaugeSiteSize*gPrecision);
  void*  staple_back_sendbuf = malloc(Vs*gaugeSiteSize*gPrecision);
  if (staple_fwd_sendbuf == NULL|| staple_back_sendbuf == NULL){
    printf("ERROR: malloc failed for staple_sendbuf/site_link_back_sendbuf\n");
    exit(1);
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    exchange_staple((double*)staple, (double*)ghost_staple, 
		    (double*)staple_fwd_sendbuf, (double*)staple_back_sendbuf);
  }else{ //single
    exchange_staple((float*)staple, (float*)ghost_staple, 
		    (float*)staple_fwd_sendbuf, (float*)staple_back_sendbuf);
  }
  
  free(staple_fwd_sendbuf);
  free(staple_back_sendbuf);
}



void
exchange_gpu_staple(int* X, void* _cudaStaple, cudaStream_t * stream)
{
  setup_dims(X);
  
  FullStaple* cudaStaple = (FullStaple*) _cudaStaple;
  exchange_llfat_init(cudaStaple);
  
  int len = Vs*gaugeSiteSize*cudaStaple->precision;
  int normlen = Vs*sizeof(float);
  
  packGhostStaple(cudaStaple, fwd_nbr_staple_sendbuf, back_nbr_staple_sendbuf, NULL, NULL, stream);
  cudaStreamSynchronize(*stream);
  

  unsigned long recv_request1 = comm_recv(back_nbr_staple_cpu, len, BACK_NBR);
  unsigned long recv_request2 = comm_recv(fwd_nbr_staple_cpu, len, FWD_NBR);
  
  memcpy(fwd_nbr_staple_sendbuf_cpu, fwd_nbr_staple_sendbuf, len);
  memcpy(back_nbr_staple_sendbuf_cpu, back_nbr_staple_sendbuf, len);

  unsigned long send_request1= comm_send(fwd_nbr_staple_sendbuf_cpu, len, FWD_NBR);
  unsigned long send_request2 = comm_send(back_nbr_staple_sendbuf_cpu, len, BACK_NBR);

  unsigned long recv_request3 = 0;
  unsigned long recv_request4 = 0;
  unsigned long send_request3 = 0;
  unsigned long send_request4 = 0;
  
  if (cudaStaple->precision == QUDA_HALF_PRECISION){
    //FIXME: half precision not suppported yet
    /*
      recv_request3 = comm_recv(b_norm, normlen, BACK_NBR);
      recv_request4 = comm_recv(f_norm, normlen, FWD_NBR);
      send_request3 = comm_send(f_norm_sendbuf, normlen, FWD_NBR);
      send_request4 = comm_send(b_norm_sendbuf, normlen, BACK_NBR);
    */
  }
  
  
  
  comm_wait(recv_request1);
  comm_wait(recv_request2);  
  comm_wait(send_request1);
  comm_wait(send_request2);
  
  if (cudaStaple->precision == QUDA_HALF_PRECISION){
    comm_wait(recv_request3);
    comm_wait(recv_request4);
    comm_wait(send_request3);
    comm_wait(send_request4);
  }
  
  memcpy(fwd_nbr_staple, fwd_nbr_staple_cpu, len);
  memcpy(back_nbr_staple, back_nbr_staple_cpu, len);

  unpackGhostStaple(cudaStaple, fwd_nbr_staple, back_nbr_staple, NULL, NULL, stream);
  cudaStreamSynchronize(*stream);
}
void
exchange_gpu_staple_start(int* X, void* _cudaStaple, cudaStream_t * stream)
{
  setup_dims(X);
  
  FullStaple* cudaStaple = (FullStaple*) _cudaStaple;
  exchange_llfat_init(cudaStaple);
  
  packGhostStaple(cudaStaple, fwd_nbr_staple_sendbuf, back_nbr_staple_sendbuf, NULL, NULL, stream);
}

void
exchange_gpu_staple_wait(int* X, void* _cudaStaple, cudaStream_t * stream)
{
  FullStaple* cudaStaple = (FullStaple*) _cudaStaple;  
  int len = Vs*gaugeSiteSize*cudaStaple->precision;
  int normlen = Vs*sizeof(float);
  
  cudaStreamSynchronize(*stream);  

  unsigned long recv_request1 = comm_recv(back_nbr_staple_cpu, len, BACK_NBR);
  unsigned long recv_request2 = comm_recv(fwd_nbr_staple_cpu, len, FWD_NBR);
  
  memcpy(fwd_nbr_staple_sendbuf_cpu, fwd_nbr_staple_sendbuf, len);
  memcpy(back_nbr_staple_sendbuf_cpu, back_nbr_staple_sendbuf, len);

  unsigned long send_request1= comm_send(fwd_nbr_staple_sendbuf_cpu, len, FWD_NBR);
  unsigned long send_request2 = comm_send(back_nbr_staple_sendbuf_cpu, len, BACK_NBR);

  unsigned long recv_request3 = 0;
  unsigned long recv_request4 = 0;
  unsigned long send_request3 = 0;
  unsigned long send_request4 = 0;
  
  if (cudaStaple->precision == QUDA_HALF_PRECISION){
    //FIXME: half precisiono not supported yet
    /*
    recv_request3 = comm_recv(b_norm, normlen, BACK_NBR);
    recv_request4 = comm_recv(f_norm, normlen, FWD_NBR);
    send_request3 = comm_send(f_norm_sendbuf, normlen, FWD_NBR);
    send_request4 = comm_send(b_norm_sendbuf, normlen, BACK_NBR);
    */
  }
  
  
  
  comm_wait(recv_request1);
  comm_wait(recv_request2);  
  comm_wait(send_request1);
  comm_wait(send_request2);
  
  if (cudaStaple->precision == QUDA_HALF_PRECISION){
    comm_wait(recv_request3);
    comm_wait(recv_request4);
    comm_wait(send_request3);
    comm_wait(send_request4);
  }
  
  memcpy(fwd_nbr_staple, fwd_nbr_staple_cpu, len);
  memcpy(back_nbr_staple, back_nbr_staple_cpu, len);

  unpackGhostStaple(cudaStaple, fwd_nbr_staple, back_nbr_staple, NULL, NULL, stream);
  cudaStreamSynchronize(*stream);
}


static void
exchange_llfat_cleanup(void)
{
  if(fwd_nbr_staple_cpu){
    free(fwd_nbr_staple_cpu); fwd_nbr_staple_cpu =NULL;
  }      
  if(back_nbr_staple_cpu){
    free(back_nbr_staple_cpu);back_nbr_staple_cpu = NULL;
  }
  if(fwd_nbr_staple_sendbuf_cpu){
    free(fwd_nbr_staple_sendbuf_cpu); fwd_nbr_staple_sendbuf_cpu = NULL;
  }
  if(back_nbr_staple_sendbuf_cpu){
    free(back_nbr_staple_sendbuf_cpu); back_nbr_staple_sendbuf_cpu = NULL;
  }    
  if(fwd_nbr_staple){
    cudaFreeHost(fwd_nbr_staple); fwd_nbr_staple = NULL;
  }
  if(back_nbr_staple){
    cudaFreeHost(back_nbr_staple); back_nbr_staple = NULL;
  }
  if(fwd_nbr_staple_sendbuf){
    cudaFreeHost(fwd_nbr_staple_sendbuf); fwd_nbr_staple_sendbuf = NULL;
  }
  if(back_nbr_staple_sendbuf){
    cudaFreeHost(back_nbr_staple_sendbuf); back_nbr_staple_sendbuf = NULL;
  }

}

#endif

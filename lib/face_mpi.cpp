#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>
#include <sys/time.h>
#include <mpicomm.h>
#include <cuda.h>

using namespace std;

#if (CUDA_VERSION >=4000)
#define GPU_DIRECT
#endif


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

#ifdef GPU_DIRECT
    pageable_fwd_nbr_spinor_sendbuf[dir] = fwd_nbr_spinor_sendbuf[dir];
    pageable_back_nbr_spinor_sendbuf[dir] = back_nbr_spinor_sendbuf[dir];
    pageable_fwd_nbr_spinor[dir] = fwd_nbr_spinor[dir];
    pageable_back_nbr_spinor[dir] = back_nbr_spinor[dir];
#else
    pageable_fwd_nbr_spinor_sendbuf[dir] = malloc(nbytes[dir]);
    pageable_back_nbr_spinor_sendbuf[dir] = malloc(nbytes[dir]);
    
    if (pageable_fwd_nbr_spinor_sendbuf[dir] == NULL || pageable_back_nbr_spinor_sendbuf[dir] == NULL)
      errorQuda("malloc failed for pageable_fwd_nbr_spinor_sendbuf/pageable_back_nbr_spinor_sendbuf");
    
    pageable_fwd_nbr_spinor[dir]=malloc(nbytes[dir]);
    pageable_back_nbr_spinor[dir]=malloc(nbytes[dir]);
    
    if (pageable_fwd_nbr_spinor[dir] == NULL || pageable_back_nbr_spinor[dir] == NULL)
      errorQuda("malloc failed for pageable_fwd_nbr_spinor/pageable_back_nbr_spinor"); 
#endif
    
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
    if(fwd_nbr_spinor_sendbuf[dir]) {
      cudaFreeHost(fwd_nbr_spinor_sendbuf[dir]);
      fwd_nbr_spinor_sendbuf[dir] = NULL;
    }
    if(back_nbr_spinor_sendbuf[dir]) {
      cudaFreeHost(back_nbr_spinor_sendbuf[dir]);
      back_nbr_spinor_sendbuf[dir] = NULL;
    }
    if(fwd_nbr_spinor[dir]) {
      cudaFreeHost(fwd_nbr_spinor[dir]);
      fwd_nbr_spinor[dir] = NULL;
    }
    if(back_nbr_spinor[dir]) {
      cudaFreeHost(back_nbr_spinor[dir]);
      back_nbr_spinor[dir] = NULL;
    }    

#ifdef GPU_DIRECT
    pageable_fwd_nbr_spinor_sendbuf[dir] = NULL;
    pageable_back_nbr_spinor_sendbuf[dir]=NULL;
    pageable_fwd_nbr_spinor[dir]=NULL;
    pageable_back_nbr_spinor[dir]=NULL;
#else
    if(pageable_fwd_nbr_spinor_sendbuf[dir]){
      free(pageable_fwd_nbr_spinor_sendbuf[dir]);
      pageable_fwd_nbr_spinor_sendbuf[dir] = NULL;
    }

    if(pageable_back_nbr_spinor_sendbuf[dir]){
      free(pageable_back_nbr_spinor_sendbuf[dir]);
      pageable_back_nbr_spinor_sendbuf[dir]=NULL;
    }
    
    if(pageable_fwd_nbr_spinor[dir]){
      free(pageable_fwd_nbr_spinor[dir]);
      pageable_fwd_nbr_spinor[dir]=NULL;
    }
    
    if(pageable_back_nbr_spinor[dir]){
      free(pageable_back_nbr_spinor[dir]);
      pageable_back_nbr_spinor[dir]=NULL;
    }
#endif

    
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
#ifndef GPU_DIRECT
  memcpy(pageable_back_nbr_spinor_sendbuf[dir], back_nbr_spinor_sendbuf[dir], nbytes[dir]);
#endif
  send_request2[dir] = comm_send_with_tag(pageable_back_nbr_spinor_sendbuf[dir], nbytes[dir], back_nbr[dir], downtags[dir]);
    
  cudaStreamSynchronize(stream[2*dir + sendFwdStrmIdx]); //required the data to be there before sending out
#ifndef GPU_DIRECT
  memcpy(pageable_fwd_nbr_spinor_sendbuf[dir], fwd_nbr_spinor_sendbuf[dir], nbytes[dir]);
#endif
  send_request1[dir]= comm_send_with_tag(pageable_fwd_nbr_spinor_sendbuf[dir], nbytes[dir], fwd_nbr[dir], uptags[dir]);
  
} 


void FaceBuffer::exchangeFacesWait(cudaColorSpinorField &out, int dagger, int dir)
{
  if(!commDimPartitioned(dir)){
    return;
  }
  
  comm_wait(recv_request2[dir]);  
  comm_wait(send_request2[dir]);
#ifndef GPU_DIRECT
  memcpy(fwd_nbr_spinor[dir], pageable_fwd_nbr_spinor[dir], nbytes[dir]);
#endif
  out.unpackGhost(fwd_nbr_spinor[dir], dir, QUDA_FORWARDS,  dagger, &stream[2*dir + recFwdStrmIdx]); CUERR;

  comm_wait(recv_request1[dir]);
  comm_wait(send_request1[dir]);

#ifndef GPU_DIRECT
  memcpy(back_nbr_spinor[dir], pageable_back_nbr_spinor[dir], nbytes[dir]);  
#endif
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

#if (CUDA_VERSION < 4000)
static void* fwd_nbr_staple_cpu[4];
static void* back_nbr_staple_cpu[4];
static void* fwd_nbr_staple_sendbuf_cpu[4];
static void* back_nbr_staple_sendbuf_cpu[4];
#endif

static void* fwd_nbr_staple_gpu[4];
static void* back_nbr_staple_gpu[4];

static void* fwd_nbr_staple[4];
static void* back_nbr_staple[4];
static void* fwd_nbr_staple_sendbuf[4];
static void* back_nbr_staple_sendbuf[4];

static int dims[4];
static int X1,X2,X3,X4;
static int V;
static int Vh;
static int Vs[4], Vsh[4];
static int Vs_x, Vs_y, Vs_z, Vs_t;
static int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
static unsigned long llfat_send_request1[4];
static unsigned long llfat_recv_request1[4];
static unsigned long llfat_send_request2[4];
static unsigned long llfat_recv_request2[4];

#include <gauge_quda.h>
extern void setup_dims_in_gauge(int *XX);

static void
setup_dims(int* X)
{
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    dims[d] = X[d];
  }
  Vh = V/2;
  
  X1=X[0];
  X2=X[1];
  X3=X[2];
  X4=X[3];


  Vs[0] = Vs_x = X[1]*X[2]*X[3];
  Vs[1] = Vs_y = X[0]*X[2]*X[3];
  Vs[2] = Vs_z = X[0]*X[1]*X[3];
  Vs[3] = Vs_t = X[0]*X[1]*X[2];

  Vsh[0] = Vsh_x = Vs_x/2;
  Vsh[1] = Vsh_y = Vs_y/2;
  Vsh[2] = Vsh_z = Vs_z/2;
  Vsh[3] = Vsh_t = Vs_t/2;

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

  for(int i=0;i < 4; i++){
    cudaMalloc((void**)&fwd_nbr_staple_gpu[i], Vs[i]*gaugeSiteSize*prec);
    cudaMalloc((void**)&back_nbr_staple_gpu[i], Vs[i]*gaugeSiteSize*prec);

    cudaMallocHost((void**)&fwd_nbr_staple[i], Vs[i]*gaugeSiteSize*prec);
    cudaMallocHost((void**)&back_nbr_staple[i], Vs[i]*gaugeSiteSize*prec);

    cudaMallocHost((void**)&fwd_nbr_staple_sendbuf[i], Vs[i]*gaugeSiteSize*prec);
    cudaMallocHost((void**)&back_nbr_staple_sendbuf[i], Vs[i]*gaugeSiteSize*prec);
  }

  
  CUERR;

#if (CUDA_VERSION < 4000)
  for(int i=0;i < 4; i++){
    fwd_nbr_staple_cpu[i] = malloc(Vs[i]*gaugeSiteSize*prec);
    back_nbr_staple_cpu[i] = malloc(Vs[i]*gaugeSiteSize*prec);
    if (fwd_nbr_staple_cpu[i] == NULL||back_nbr_staple_cpu[i] == NULL){
      printf("ERROR: malloc failed for fwd_nbr_staple_cpu/back_nbr_staple_cpu\n");
      comm_exit(1);
    }

  }

  for(int i=0;i < 4; i++){
    fwd_nbr_staple_sendbuf_cpu[i] = malloc(Vs[i]*gaugeSiteSize*prec);
    back_nbr_staple_sendbuf_cpu[i] = malloc(Vs[i]*gaugeSiteSize*prec);
    if (fwd_nbr_staple_sendbuf_cpu[i] == NULL || back_nbr_staple_sendbuf_cpu[i] == NULL){
      printf("ERROR: malloc failed for fwd_nbr_staple_sendbuf/back_nbr_staple_sendbuf\n");
      comm_exit(1);
    }
  }
#endif
  
  return;
}

template<typename Float>
void
exchange_sitelink_diag(int* X, Float** sitelink,  Float** ghost_sitelink_diag, int optflag)
{
  /*
    nu |          |
       |__________|
           mu 

  * There are total 12 different combinations for (nu,mu)
  * since nu/mu = X,Y,Z,T and nu != mu
  * For each combination, we need to communicate with the corresponding
  * neighbor and get the diag ghost data
  * The neighbor we need to get data from is dx[nu]=-1, dx[mu]= +1
  * and we need to send our data to neighbor with dx[nu]=+1, dx[mu]=-1
  */
  
  for(int nu = XUP; nu <=TUP; nu++){
    for(int mu = XUP; mu <= TUP; mu++){
      if(nu == mu){
	continue;
      }
      if(optflag && (!commDimPartitioned(mu) || !commDimPartitioned(nu))){
	continue;
      }

      int dir1, dir2; //other two dimensions
      for(dir1=0; dir1 < 4; dir1 ++){
	if(dir1 != nu && dir1 != mu){
	  break;
	}
      }
      for(dir2=0; dir2 < 4; dir2 ++){
	if(dir2 != nu && dir2 != mu && dir2 != dir1){
	  break;
	}
      }  
      
      if(dir1 == 4 || dir2 == 4){
	errorQuda("Invalid dir1/dir2");
      }
      int len = X[dir1]*X[dir2]*gaugeSiteSize*sizeof(Float);
      void* sendbuf = malloc(len);
      if(sendbuf == NULL){
	errorQuda("Malloc failed for diag sendbuf\n");
      }
      
      pack_gauge_diag(sendbuf, X, (void**)sitelink, nu, mu, dir1, dir2, (QudaPrecision)sizeof(Float));
  
      //post recv
      int dx[4]={0,0,0,0};
      dx[nu]=-1;
      dx[mu]=+1;
      int src_rank = comm_get_neighbor_rank(dx[0], dx[1], dx[2], dx[3]);
      unsigned long recv_request = comm_recv_from_rank(ghost_sitelink_diag[nu*4+mu], len, src_rank);
      //do send
      dx[nu]=+1;
      dx[mu]=-1;
      int dst_rank = comm_get_neighbor_rank(dx[0], dx[1], dx[2], dx[3]);
      unsigned long send_request = comm_send_to_rank(sendbuf, len, dst_rank);
      
      comm_wait(recv_request);
      comm_wait(send_request);
            
      free(sendbuf);
    }//mu
  }//nu
}


template<typename Float>
void
exchange_sitelink(int*X, Float** sitelink, Float** ghost_sitelink, Float** ghost_sitelink_diag, 
		  Float** sitelink_fwd_sendbuf, Float** sitelink_back_sendbuf, int optflag)
{


#if 0
  int i;
  int len = Vsh_t*gaugeSiteSize*sizeof(Float);
  for(i=0;i < 4;i++){
    Float* even_sitelink_back_src = sitelink[i];
    Float* odd_sitelink_back_src = sitelink[i] + Vh*gaugeSiteSize;
    Float* sitelink_back_dst = sitelink_back_sendbuf[3] + 2*i*Vsh_t*gaugeSiteSize;

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
    Float* sitelink_fwd_dst = sitelink_fwd_sendbuf[3] + 2*i*Vsh_t*gaugeSiteSize;
    if(dims[3] % 2 == 0){    
      memcpy(sitelink_fwd_dst, even_sitelink_fwd_src, len);
      memcpy(sitelink_fwd_dst + Vsh_t*gaugeSiteSize, odd_sitelink_fwd_src, len);
    }else{
      //switching odd and even ghost sitelink
      memcpy(sitelink_fwd_dst, odd_sitelink_fwd_src, len);
      memcpy(sitelink_fwd_dst + Vsh_t*gaugeSiteSize, even_sitelink_fwd_src, len);
    }
    
  }
#else
  int nFace =1;
  for(int dir=0; dir < 4; dir++){
    if(optflag && !commDimPartitioned(dir)) continue;
    pack_ghost_all_links((void**)sitelink, (void**)sitelink_back_sendbuf, (void**)sitelink_fwd_sendbuf, dir, nFace, (QudaPrecision)(sizeof(Float)));
  }
#endif


  int fwd_neighbors[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR, T_FWD_NBR};
  int back_neighbors[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR, T_BACK_NBR};
  int up_tags[4] = {XUP, YUP, ZUP, TUP};
  int down_tags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};

  for(int dir  =0; dir < 4; dir++){
    if(optflag && !commDimPartitioned(dir)) continue;
    int len = Vsh[dir]*gaugeSiteSize*sizeof(Float);
    Float* ghost_sitelink_back = ghost_sitelink[dir];
    Float* ghost_sitelink_fwd = ghost_sitelink[dir] + 8*Vsh[dir]*gaugeSiteSize;
    
    unsigned long recv_request1 = comm_recv_with_tag(ghost_sitelink_back, 8*len, back_neighbors[dir], up_tags[dir]);
    unsigned long recv_request2 = comm_recv_with_tag(ghost_sitelink_fwd, 8*len, fwd_neighbors[dir], down_tags[dir]);
    unsigned long send_request1 = comm_send_with_tag(sitelink_fwd_sendbuf[dir], 8*len, fwd_neighbors[dir], up_tags[dir]);
    unsigned long send_request2 = comm_send_with_tag(sitelink_back_sendbuf[dir], 8*len, back_neighbors[dir], down_tags[dir]);
    comm_wait(recv_request1);
    comm_wait(recv_request2);
    comm_wait(send_request1);
    comm_wait(send_request2);
  }

  exchange_sitelink_diag(X, sitelink, ghost_sitelink_diag, optflag);
}

//this function is used for link fattening computation
//@optflag: if this flag is set, we only communicate in directions that are partitioned
//          if not set, then we communicate in all directions regradless of partitions
void exchange_cpu_sitelink(int* X,
			   void** sitelink, void** ghost_sitelink,
			   void** ghost_sitelink_diag,
			   QudaPrecision gPrecision, int optflag)
{  
  setup_dims(X);
  set_dim(X);
  void*  sitelink_fwd_sendbuf[4];
  void*  sitelink_back_sendbuf[4];
  
  for(int i=0;i < 4;i++){
    sitelink_fwd_sendbuf[i] = malloc(4*Vs[i]*gaugeSiteSize*gPrecision);
    sitelink_back_sendbuf[i] = malloc(4*Vs[i]*gaugeSiteSize*gPrecision);
    if (sitelink_fwd_sendbuf[i] == NULL|| sitelink_back_sendbuf[i] == NULL){
      errorQuda("ERROR: malloc failed for sitelink_sendbuf/site_link_back_sendbuf\n");
    }  
    memset(sitelink_fwd_sendbuf[i], 0, 4*Vs[i]*gaugeSiteSize*gPrecision);
    memset(sitelink_back_sendbuf[i], 0, 4*Vs[i]*gaugeSiteSize*gPrecision);
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    exchange_sitelink(X, (double**)sitelink, (double**)(ghost_sitelink), (double**)ghost_sitelink_diag, 
		      (double**)sitelink_fwd_sendbuf, (double**)sitelink_back_sendbuf, optflag);
  }else{ //single
    exchange_sitelink(X, (float**)sitelink, (float**)(ghost_sitelink), (float**)ghost_sitelink_diag, 
		      (float**)sitelink_fwd_sendbuf, (float**)sitelink_back_sendbuf, optflag);
  }
  
  for(int i=0;i < 4;i++){
    free(sitelink_fwd_sendbuf[i]);
    free(sitelink_back_sendbuf[i]);
  }
}




template<typename Float>
void
do_exchange_cpu_staple(Float* staple, Float** ghost_staple, Float** staple_fwd_sendbuf, Float** staple_back_sendbuf)
{


#if 0  
  int len = Vsh_t*gaugeSiteSize*sizeof(Float);
  Float* even_staple_back_src = staple;
  Float* odd_staple_back_src = staple + Vh*gaugeSiteSize;
  Float* staple_back_dst = staple_back_sendbuf[3];
  
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
  Float* staple_fwd_dst = staple_fwd_sendbuf[3];
  if(dims[3] % 2 == 0){    
    memcpy(staple_fwd_dst, even_staple_fwd_src, len);
    memcpy(staple_fwd_dst + Vsh_t*gaugeSiteSize, odd_staple_fwd_src, len);
  }else{
    //switching odd and even ghost staple
    memcpy(staple_fwd_dst, odd_staple_fwd_src, len);
    memcpy(staple_fwd_dst + Vsh_t*gaugeSiteSize, even_staple_fwd_src, len);
  }
#else
  int nFace =1;
  pack_ghost_all_staples_cpu(staple, (void**)staple_back_sendbuf, 
			     (void**)staple_fwd_sendbuf,  nFace, (QudaPrecision)(sizeof(Float)));

#endif  
  
  int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};
  int len[4] = {
    Vsh_x*gaugeSiteSize*sizeof(Float),
    Vsh_y*gaugeSiteSize*sizeof(Float),
    Vsh_z*gaugeSiteSize*sizeof(Float),
    Vsh_t*gaugeSiteSize*sizeof(Float)
  };
  
  int fwd_neighbors[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR, T_FWD_NBR};
  int back_neighbors[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR, T_BACK_NBR};
  int up_tags[4] = {XUP, YUP, ZUP, TUP};
  int down_tags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  
  for(int dir=0;dir < 4; dir++){
    Float* ghost_staple_back = ghost_staple[dir];
    Float* ghost_staple_fwd = ghost_staple[dir] + 2*Vsh[dir]*gaugeSiteSize;
    
    unsigned long recv_request1 = comm_recv_with_tag(ghost_staple_back, 2*len[dir], back_neighbors[dir], up_tags[dir]);
    unsigned long recv_request2 = comm_recv_with_tag(ghost_staple_fwd, 2*len[dir], fwd_neighbors[dir], down_tags[dir]);
    unsigned long send_request1 = comm_send_with_tag(staple_fwd_sendbuf[dir], 2*len[dir], fwd_neighbors[dir], up_tags[dir]);
    unsigned long send_request2 = comm_send_with_tag(staple_back_sendbuf[dir], 2*len[dir], back_neighbors[dir], down_tags[dir]);
    
    comm_wait(recv_request1);
    comm_wait(recv_request2);
    comm_wait(send_request1);
    comm_wait(send_request2);
  }
}
//this function is used for link fattening computation
void exchange_cpu_staple(int* X,
			 void* staple, void** ghost_staple,
			 QudaPrecision gPrecision)
{
  
  setup_dims(X);

  int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
  void*  staple_fwd_sendbuf[4];
  void*  staple_back_sendbuf[4];

  for(int i=0;i < 4; i++){
    staple_fwd_sendbuf[i] = malloc(Vs[i]*gaugeSiteSize*gPrecision);
    staple_back_sendbuf[i] = malloc(Vs[i]*gaugeSiteSize*gPrecision);
    if (staple_fwd_sendbuf[i] == NULL|| staple_back_sendbuf[i] == NULL){
      printf("ERROR: malloc failed for staple_sendbuf/site_link_back_sendbuf\n");
      exit(1);
    }
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    do_exchange_cpu_staple((double*)staple, (double**)ghost_staple, 
			   (double**)staple_fwd_sendbuf, (double**)staple_back_sendbuf);
  }else{ //single
    do_exchange_cpu_staple((float*)staple, (float**)ghost_staple, 
			   (float**)staple_fwd_sendbuf, (float**)staple_back_sendbuf);
  }
  
  for(int i=0;i < 4;i++){
    free(staple_fwd_sendbuf[i]);
    free(staple_back_sendbuf[i]);
  }
}

//@whichway indicates send direction
void
exchange_gpu_staple_start(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream)
{
  setup_dims(X);
  
  FullStaple* cudaStaple = (FullStaple*) _cudaStaple;
  exchange_llfat_init(cudaStaple);
  
  packGhostStaple(cudaStaple, dir, whichway, fwd_nbr_staple_gpu, back_nbr_staple_gpu,
		  fwd_nbr_staple_sendbuf, back_nbr_staple_sendbuf, stream);
}


//@whichway indicates send direction
//we use recv_whichway to indicate recv direction
void
exchange_gpu_staple_comms(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream)
{
  FullStaple* cudaStaple = (FullStaple*) _cudaStaple;  

  int fwd_neighbors[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR, T_FWD_NBR};
  int back_neighbors[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR, T_BACK_NBR};
  int up_tags[4] = {XUP, YUP, ZUP, TUP};
  int down_tags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};

  cudaStreamSynchronize(*stream);  

  int recv_whichway;
  if(whichway == QUDA_BACKWARDS){
    recv_whichway = QUDA_FORWARDS;
  }else{
    recv_whichway = QUDA_BACKWARDS;
  }
  

  int i = dir;
  int len = Vs[i]*gaugeSiteSize*cudaStaple->precision;
  int normlen = Vs[i]*sizeof(float);
  
  if(recv_whichway == QUDA_BACKWARDS){   
#if (CUDA_VERSION >= 4000)
    llfat_recv_request1[i] = comm_recv_with_tag(back_nbr_staple[i], len, back_neighbors[i], up_tags[i]);
    llfat_send_request1[i] = comm_send_with_tag(fwd_nbr_staple_sendbuf[i], len, fwd_neighbors[i],  up_tags[i]);
#else
    llfat_recv_request1[i] = comm_recv_with_tag(back_nbr_staple_cpu[i], len, back_neighbors[i], up_tags[i]);
    memcpy(fwd_nbr_staple_sendbuf_cpu[i], fwd_nbr_staple_sendbuf[i], len);
    llfat_send_request1[i] = comm_send_with_tag(fwd_nbr_staple_sendbuf_cpu[i], len, fwd_neighbors[i],  up_tags[i]);
#endif
  } else { // QUDA_FORWARDS
#if (CUDA_VERSION >= 4000)
    llfat_recv_request2[i] = comm_recv_with_tag(fwd_nbr_staple[i], len, fwd_neighbors[i], down_tags[i]);
    llfat_send_request2[i] = comm_send_with_tag(back_nbr_staple_sendbuf[i], len, back_neighbors[i] ,down_tags[i]);
#else
    llfat_recv_request2[i] = comm_recv_with_tag(fwd_nbr_staple_cpu[i], len, fwd_neighbors[i], down_tags[i]);
    memcpy(back_nbr_staple_sendbuf_cpu[i], back_nbr_staple_sendbuf[i], len);
    llfat_send_request2[i] = comm_send_with_tag(back_nbr_staple_sendbuf_cpu[i], len, back_neighbors[i] ,down_tags[i]);
#endif
  }
}


//@whichway indicates send direction
//we use recv_whichway to indicate recv direction
void
exchange_gpu_staple_wait(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream)
{
  FullStaple* cudaStaple = (FullStaple*) _cudaStaple;  

  int recv_whichway;
  if(whichway == QUDA_BACKWARDS){
    recv_whichway = QUDA_FORWARDS;
  }else{
    recv_whichway = QUDA_BACKWARDS;
  }
  

  int i = dir;
  int len = Vs[i]*gaugeSiteSize*cudaStaple->precision;
  int normlen = Vs[i]*sizeof(float);
  
  if(recv_whichway == QUDA_BACKWARDS){   
    comm_wait(llfat_recv_request1[i]);
    comm_wait(llfat_send_request1[i]);

#if (CUDA_VERSION >= 4000)
    unpackGhostStaple(cudaStaple, i, QUDA_BACKWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#else   
    memcpy(back_nbr_staple[i], back_nbr_staple_cpu[i], len);
    unpackGhostStaple(cudaStaple, i, QUDA_BACKWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#endif

  } else { // QUDA_FORWARDS
    comm_wait(llfat_recv_request2[i]);  
    comm_wait(llfat_send_request2[i]);

#if (CUDA_VERSION >= 4000)
    unpackGhostStaple(cudaStaple, i, QUDA_FORWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#else        
    memcpy(fwd_nbr_staple[i], fwd_nbr_staple_cpu[i], len);
    unpackGhostStaple(cudaStaple, i, QUDA_FORWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#endif

  }
}


void
exchange_llfat_cleanup(void)
{
  
  for(int i=0;i < 4; i++){
    if(fwd_nbr_staple_gpu[i]){
      cudaFree(fwd_nbr_staple_gpu[i]); fwd_nbr_staple_gpu[i] =NULL;
    }      
    if(back_nbr_staple_gpu[i]){
      cudaFree(back_nbr_staple_gpu[i]);back_nbr_staple_gpu[i] = NULL;
    }

  }

#if (CUDA_VERSION < 4000)
  for(int i=0;i < 4; i++){
    if(fwd_nbr_staple_cpu[i]){
      free(fwd_nbr_staple_cpu[i]); fwd_nbr_staple_cpu[i] =NULL;
    }      
    if(back_nbr_staple_cpu[i]){
      free(back_nbr_staple_cpu[i]);back_nbr_staple_cpu[i] = NULL;
    }
  }
  for(int i=0;i < 4; i++){
    if(fwd_nbr_staple_sendbuf_cpu[i]){
      free(fwd_nbr_staple_sendbuf_cpu[i]); fwd_nbr_staple_sendbuf_cpu[i] = NULL;
    }
    if(back_nbr_staple_sendbuf_cpu[i]){
      free(back_nbr_staple_sendbuf_cpu[i]); back_nbr_staple_sendbuf_cpu[i] = NULL;
    }    
  }
#endif

  for(int i=0;i < 4; i++){
    if(fwd_nbr_staple[i]){
      cudaFreeHost(fwd_nbr_staple[i]); fwd_nbr_staple[i] = NULL;
    }
    if(back_nbr_staple[i]){
      cudaFreeHost(back_nbr_staple[i]); back_nbr_staple[i] = NULL;
    }
  }
  
  for(int i=0;i < 4; i++){
    if(fwd_nbr_staple_sendbuf[i]){
      cudaFreeHost(fwd_nbr_staple_sendbuf[i]); fwd_nbr_staple_sendbuf[i] = NULL;
    }
    if(back_nbr_staple_sendbuf[i]){
      cudaFreeHost(back_nbr_staple_sendbuf[i]); back_nbr_staple_sendbuf[i] = NULL;
    }
  }

}

#endif

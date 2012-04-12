#include <quda_internal.h>
#include <face_quda.h>
#include <comm_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <cuda.h>

#include <fat_force_quda.h>

using namespace std;

#ifdef DSLASH_PROFILING
  void printDslashProfile();
#define CUDA_EVENT_RECORD(a,b) cudaEventRecord(a,b)
#else
#define CUDA_EVENT_RECORD(a,b)
#define DSLASH_TIME_PROFILE()
#endif

cudaStream_t *stream;

bool globalReduce = true;

FaceBuffer::FaceBuffer(const int *X, const int nDim, const int Ninternal, 
		       const int nFace, const QudaPrecision precision, const int Ls) : 
  Ninternal(Ninternal), precision(precision), nDim(nDim), nFace(nFace)
{
//temporal hack for DW operator  
//BEGIN NEW
  int Y[nDim];
  Y[0] = X[0];
  Y[1] = X[1];
  Y[2] = X[2];
  Y[3] = X[3];
  if(nDim == 5) Y[nDim-1] = Ls;
  setupDims(Y);
//END NEW
  
  //setupDims(X);

  // set these both = 0 `for no overlap of qmp and cudamemcpyasync
  // sendBackStrmIdx = 0, and sendFwdStrmIdx = 1 for overlap
  sendBackStrmIdx = 0;
  sendFwdStrmIdx = 1;
  recFwdStrmIdx = sendBackStrmIdx;
  recBackStrmIdx = sendFwdStrmIdx;

  
  for(int i=0;i < QUDA_MAX_DIM; i++){
    recv_request1[i] = malloc(sizeof(MPI_Request));
    recv_request2[i] = malloc(sizeof(MPI_Request));
    send_request1[i] = malloc(sizeof(MPI_Request));    
    send_request2[i] = malloc(sizeof(MPI_Request));    
    if( recv_request1[i] == 0 || recv_request2[i] == 0
	|| send_request1[i] == 0 || send_request2[i] == 0){
      errorQuda("ERROR: malloc failed for recv/send request handles\n");
    }
  }

  for(int dir =0 ; dir < 4;dir++){
    nbytes[dir] = nFace*faceVolumeCB[dir]*Ninternal*precision;
    if (precision == QUDA_HALF_PRECISION) nbytes[dir] += nFace*faceVolumeCB[dir]*sizeof(float);
    
    cudaMallocHost((void**)&fwd_nbr_spinor_sendbuf[dir], nbytes[dir]); 
    cudaMallocHost((void**)&back_nbr_spinor_sendbuf[dir], nbytes[dir]);
    
    if (fwd_nbr_spinor_sendbuf[dir] == NULL || back_nbr_spinor_sendbuf[dir] == NULL)
      errorQuda("dir =%d, malloc failed for fwd_nbr_spinor_sendbuf/back_nbr_spinor_sendbuf", dir); 
    
    cudaMallocHost((void**)&fwd_nbr_spinor[dir], nbytes[dir]); 
    cudaMallocHost((void**)&back_nbr_spinor[dir], nbytes[dir]); 
    
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
  
  for(int i=0;i < QUDA_MAX_DIM; i++){
    free((void*)recv_request1[i]);
    free((void*)recv_request2[i]);
    free((void*)send_request1[i]);
    free((void*)send_request2[i]);
  }

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

void FaceBuffer::pack(cudaColorSpinorField &in, int parity, int dagger, int dim, cudaStream_t *stream_p)
{
  if(!commDimPartitioned(dim)) return;

  in.allocateGhostBuffer();   // allocate the ghost buffer if not yet allocated  
  stream = stream_p;

  in.packGhost(dim, (QudaParity)parity, dagger, &stream[Nstream-1]);
}

void FaceBuffer::gather(cudaColorSpinorField &in, int dagger, int dir)
{
  int dim = dir/2;
  if(!commDimPartitioned(dim)) return;

  if (dir%2==0){ // backwards send
    in.sendGhost(back_nbr_spinor_sendbuf[dim], dim, QUDA_BACKWARDS, dagger, &stream[2*dim + sendBackStrmIdx]);
  } else { // forwards send
    in.sendGhost(fwd_nbr_spinor_sendbuf[dim], dim, QUDA_FORWARDS, dagger, &stream[2*dim + sendFwdStrmIdx]); 
  }
}

void FaceBuffer::commsStart(int dir) { 
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return;

  int back_nbr[4] = {X_BACK_NBR,Y_BACK_NBR,Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR,Y_FWD_NBR,Z_FWD_NBR,T_FWD_NBR};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  int uptags[4] = {XUP, YUP, ZUP, TUP};

  if (dir %2 == 0) {
    // Prepost all receives

    comm_recv_with_tag(pageable_fwd_nbr_spinor[dim], nbytes[dim], fwd_nbr[dim], downtags[dim], recv_request1[dim]);
#ifndef GPU_DIRECT
    memcpy(pageable_back_nbr_spinor_sendbuf[dim], 
	   back_nbr_spinor_sendbuf[dim], nbytes[dim]);
#endif
    comm_send_with_tag(pageable_back_nbr_spinor_sendbuf[dim], nbytes[dim], back_nbr[dim], downtags[dim], send_request1[dim]);
  } else {

    comm_recv_with_tag(pageable_back_nbr_spinor[dim], nbytes[dim], back_nbr[dim], uptags[dim], recv_request2[dim]);
#ifndef GPU_DIRECT
    memcpy(pageable_fwd_nbr_spinor_sendbuf[dim], 
	   fwd_nbr_spinor_sendbuf[dim], nbytes[dim]);
#endif
    comm_send_with_tag(pageable_fwd_nbr_spinor_sendbuf[dim], nbytes[dim], fwd_nbr[dim], uptags[dim], send_request2[dim]);
  }
}


int FaceBuffer::commsQuery(int dir) {
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return 0;

  if(dir%2==0) {
    if (comm_query(recv_request1[dim]) && 
	comm_query(send_request1[dim])) {
#ifndef GPU_DIRECT
      memcpy(fwd_nbr_spinor[dim], pageable_fwd_nbr_spinor[dim], nbytes[dim]);
#endif
      return 1;
    }
  } else {
    if (comm_query(recv_request2[dim]) &&
	comm_query(send_request2[dim])) {
#ifndef GPU_DIRECT
      memcpy(back_nbr_spinor[dim], pageable_back_nbr_spinor[dim], nbytes[dim]);
#endif
      return 1;
    }
  }

  return 0;
}

void FaceBuffer::scatter(cudaColorSpinorField &out, int dagger, int dir) {
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return;

  if (dir%2 == 0) {
    out.unpackGhost(fwd_nbr_spinor[dim], dim, QUDA_FORWARDS,  dagger, &stream[2*dim + recFwdStrmIdx]); 
  } else {
    out.unpackGhost(back_nbr_spinor[dim], dim, QUDA_BACKWARDS,  dagger, &stream[2*dim + recBackStrmIdx]);
  }
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

  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  
  for(int i= 0;i < 4; i++){
    comm_recv_with_tag(spinor.backGhostFaceBuffer[i], len[i], back_nbr[i], uptags[i], recv_request1[i]);
    comm_recv_with_tag(spinor.fwdGhostFaceBuffer[i], len[i], fwd_nbr[i], downtags[i], recv_request2[i]);    
    comm_send_with_tag(spinor.fwdGhostFaceSendBuffer[i], len[i], fwd_nbr[i], uptags[i], send_request1[i]);
    comm_send_with_tag(spinor.backGhostFaceSendBuffer[i], len[i], back_nbr[i], downtags[i], send_request2[i]);
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
      MPI_Request recv_request; 
	comm_recv_with_tag(ghost_link[dir], len*precision, back_nbrs[dir], uptags[dir], &recv_request);
      MPI_Request send_request;
	comm_send_with_tag(link_sendbuf[dir], len*precision, fwd_nbrs[dir], uptags[dir], &send_request);
      comm_wait(&recv_request);
      comm_wait(&send_request);
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
#if defined(GPU_FATLINK)||defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE)

#define gaugeSiteSize 18

#ifdef GPU_DIRECT
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
static MPI_Request llfat_send_request1[4];
static MPI_Request llfat_recv_request1[4];
static MPI_Request llfat_send_request2[4];
static MPI_Request llfat_recv_request2[4];

#include "gauge_field.h"
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
exchange_llfat_init(QudaPrecision prec)
{
  static int initialized = 0;
  if (initialized){
    return;
  }
  initialized = 1;
  

  for(int i=0;i < 4; i++){
    if(cudaMalloc((void**)&fwd_nbr_staple_gpu[i], Vs[i]*gaugeSiteSize*prec) != cudaSuccess){
      errorQuda("cudaMalloc() failed for fwd_nbr_staple_gpu\n");
    }
    if(cudaMalloc((void**)&back_nbr_staple_gpu[i], Vs[i]*gaugeSiteSize*prec) != cudaSuccess){
      errorQuda("cudaMalloc() failed for back_nbr_staple_gpu\n");
    }

    cudaMallocHost((void**)&fwd_nbr_staple[i], Vs[i]*gaugeSiteSize*prec);
    cudaMallocHost((void**)&back_nbr_staple[i], Vs[i]*gaugeSiteSize*prec);

    cudaMallocHost((void**)&fwd_nbr_staple_sendbuf[i], Vs[i]*gaugeSiteSize*prec);
    cudaMallocHost((void**)&back_nbr_staple_sendbuf[i], Vs[i]*gaugeSiteSize*prec);
  }

  

#ifdef GPU_DIRECT
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
      MPI_Request recv_request;
      comm_recv_from_rank(ghost_sitelink_diag[nu*4+mu], len, src_rank, &recv_request);
      //do send
      dx[nu]=+1;
      dx[mu]=-1;
      int dst_rank = comm_get_neighbor_rank(dx[0], dx[1], dx[2], dx[3]);
      MPI_Request send_request;
      comm_send_to_rank(sendbuf, len, dst_rank, &send_request);
      
      comm_wait(&recv_request);
      comm_wait(&send_request);
            
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
    pack_ghost_all_links((void**)sitelink, (void**)sitelink_back_sendbuf, (void**)sitelink_fwd_sendbuf, dir, nFace, (QudaPrecision)(sizeof(Float)), X);
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
    
    MPI_Request recv_request1;
    MPI_Request recv_request2;
    comm_recv_with_tag(ghost_sitelink_back, 8*len, back_neighbors[dir], up_tags[dir], &recv_request1);
    comm_recv_with_tag(ghost_sitelink_fwd, 8*len, fwd_neighbors[dir], down_tags[dir], &recv_request2);
    MPI_Request send_request1; 
    MPI_Request send_request2;
    comm_send_with_tag(sitelink_fwd_sendbuf[dir], 8*len, fwd_neighbors[dir], up_tags[dir], &send_request1);
    comm_send_with_tag(sitelink_back_sendbuf[dir], 8*len, back_neighbors[dir], down_tags[dir], &send_request2);
    comm_wait(&recv_request1);
    comm_wait(&recv_request2);
    comm_wait(&send_request1);
    comm_wait(&send_request2);
  }

  exchange_sitelink_diag(X, sitelink, ghost_sitelink_diag, optflag);
}

//this function is used for link fattening computation
//@optflag: if this flag is set, we only communicate in directions that are partitioned
//          if not set, then we communicate in all directions regradless of partitions
void exchange_cpu_sitelink(int* X,
			   void** sitelink, void** ghost_sitelink,
			   void** ghost_sitelink_diag,
			   QudaPrecision gPrecision, QudaGaugeParam* param, int optflag)
{  
  setup_dims(X);
  static void*  sitelink_fwd_sendbuf[4];
  static void*  sitelink_back_sendbuf[4];
  static int allocated = 0;

  if(!allocated){
    for(int i=0;i < 4;i++){
      sitelink_fwd_sendbuf[i] = malloc(4*Vs[i]*gaugeSiteSize*gPrecision);
      sitelink_back_sendbuf[i] = malloc(4*Vs[i]*gaugeSiteSize*gPrecision);
      if (sitelink_fwd_sendbuf[i] == NULL|| sitelink_back_sendbuf[i] == NULL){
	errorQuda("ERROR: malloc failed for sitelink_sendbuf/site_link_back_sendbuf\n");
      }  
      memset(sitelink_fwd_sendbuf[i], 0, 4*Vs[i]*gaugeSiteSize*gPrecision);
      memset(sitelink_back_sendbuf[i], 0, 4*Vs[i]*gaugeSiteSize*gPrecision);
    }
    allocated = 1;
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    exchange_sitelink(X, (double**)sitelink, (double**)(ghost_sitelink), (double**)ghost_sitelink_diag, 
		      (double**)sitelink_fwd_sendbuf, (double**)sitelink_back_sendbuf, optflag);
  }else{ //single
    exchange_sitelink(X, (float**)sitelink, (float**)(ghost_sitelink), (float**)ghost_sitelink_diag, 
		      (float**)sitelink_fwd_sendbuf, (float**)sitelink_back_sendbuf, optflag);
  }
  
  if(!(param->preserve_gauge & QUDA_FAT_PRESERVE_COMM_MEM)){
    for(int i=0;i < 4;i++){
      free(sitelink_fwd_sendbuf[i]);
      free(sitelink_back_sendbuf[i]);
    }
    allocated = 0;
  }
}




/* This function exchange the sitelink and store them in the correspoinding portion of 
 * the extended sitelink memory region
 * @sitelink: this is stored according to dimension size (X1+4) (X2+4) (X3+4) (X4+4)
 */

void exchange_cpu_sitelink_ex(int* X, void** sitelink, QudaGaugeFieldOrder cpu_order,
			      QudaPrecision gPrecision, int optflag)
{
  int X1,X2,X3,X4;
  int E1,E2,E3,E4;  
  X1 = X[0]; X2 = X[1]; X3 = X[2]; X4 = X[3]; 
  E1 = X[0]+4; E2 = X[1]+4; E3 = X[2]+4; E4 = X[3]+4; 
  int E3E2E1=E3*E2*E1;
  int E2E1=E2*E1;
  int E4E3E2=E4*E3*E2;
  int E3E2=E3*E2;
  int E4E3E1=E4*E3*E1;
  int E3E1=E3*E1;
  int E4E2E1=E4*E2*E1;
  int Vh_ex = E4*E3*E2*E1/2;
  
  //...............x.........y.....z......t
  int starta[] = {2,      2,       2,     0};
  int enda[]   = {X4+2,   X4+2,    X4+2,  X3+4};
  
  int startb[] = {2,      2,       0,     0};
  int endb[]   = {X3+2,   X3+2,    X2+4,  X2+4};
  
  int startc[] = {2,      0,       0,     0};
  int endc[]   = {X2+2,   X1+4,    X1+4,  X1+4};
  
  int f_main[4][4] = {
    {E3E2E1,    E2E1, E1,     1},
    {E3E2E1,    E2E1,    1,  E1},
    {E3E2E1,  E1,    1,    E2E1},
    {E2E1,  E1,    1,   E3E2E1}
  };  
  
  int f_bound[4][4]={
    {E3E2, E2, 1, E4E3E2},
    {E3E1, E1, 1, E4E3E1}, 
    {E2E1, E1, 1, E4E2E1},
    {E2E1, E1, 1, E3E2E1}
  };
  
  int nslices = 2;
  int slice_3d[] = { E4E3E2, E4E3E1, E4E2E1, E3E2E1};  
  int len[4];
  for(int i=0;i < 4;i++){
    len[i] = slice_3d[i] * nslices* 4*gaugeSiteSize*gPrecision; //2 slices, 4 directions' links
  }
  void* ghost_sitelink_fwd_sendbuf[4];
  void* ghost_sitelink_back_sendbuf[4];
  void* ghost_sitelink_fwd[4];
  void* ghost_sitelink_back[4];  
  for(int i=0;i < 4;i++){    
    if(!commDimPartitioned(i)) continue;
    ghost_sitelink_fwd_sendbuf[i] = malloc(len[i]);
    ghost_sitelink_back_sendbuf[i] = malloc(len[i]);
    if(ghost_sitelink_fwd_sendbuf[i] == NULL || ghost_sitelink_back_sendbuf[i] == NULL){
      errorQuda("Error: malloc for ghost sitelink send buffer failed\n");
    } 
    
    ghost_sitelink_fwd[i] = malloc(len[i]);
    ghost_sitelink_back[i] = malloc(len[i]);
    if(ghost_sitelink_fwd[i] == NULL || ghost_sitelink_back[i] == NULL){
      errorQuda("Error: malloc for ghost sitelink failed\n");
    } 
    
  }

  int back_nbr[4] = {X_BACK_NBR, Y_BACK_NBR, Z_BACK_NBR,T_BACK_NBR};
  int fwd_nbr[4] = {X_FWD_NBR, Y_FWD_NBR, Z_FWD_NBR,T_FWD_NBR};
  int uptags[4] = {XUP, YUP, ZUP, TUP};
  int downtags[4] = {XDOWN, YDOWN, ZDOWN, TDOWN};
  MPI_Request recv_request1[4], recv_request2[4];
  MPI_Request send_request1[4], send_request2[4];
  
  int gaugebytes = gaugeSiteSize*gPrecision;
  int a, b, c,d;
  for(int dir =0;dir < 4;dir++){
    if( (!commDimPartitioned(dir)) && optflag) continue;
    if(commDimPartitioned(dir)){
      //fill the sendbuf here
      //back
      for(d=2; d < 4; d++)
	for(a=starta[dir];a < enda[dir]; a++)
	  for(b=startb[dir]; b < endb[dir]; b++)

	    if(f_main[dir][2] != 1 || f_bound[dir][2] !=1){
	      for (c=startc[dir]; c < endc[dir]; c++){
		int oddness = (a+b+c+d)%2;
		int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-2)*f_bound[dir][3])>> 1;	      
		
		int src_oddness = oddness;
		int dst_oddness = oddness;
		if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		  dst_oddness = 1-oddness;
		}

#define MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_buf, dst_idx, sitelink, src_idx, num, dir) \
		if(src_oddness){					\
		  src_idx += Vh_ex;					\
		}							\
		if(dst_oddness){					\
		  dst_idx += nslices*slice_3d[dir]/2;			\
		}							\
		if(cpu_order == QUDA_QDP_GAUGE_ORDER){			\
		  for(int linkdir=0; linkdir < 4; linkdir++){		\
		    char* src = (char*) sitelink[linkdir] + (src_idx)*gaugebytes; \
		    char* dst = ((char*)ghost_buf[dir])+ linkdir*nslices*slice_3d[dir]*gaugebytes + (dst_idx)*gaugebytes; \
		    memcpy(dst, src, gaugebytes*(num));			\
		  }							\
		}else{	/*QUDA_MILC_GAUGE_ORDER*/			\
		  char* src = ((char*)sitelink)+ 4*(src_idx)*gaugebytes; \
		  char* dst = ((char*)ghost_buf[dir]) + 4*(dst_idx)*gaugebytes; \
		  memcpy(dst, src, 4*gaugebytes*(num));		\
		}							\

		MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_back_sendbuf, dst_idx, sitelink, src_idx, 1, dir);		

	      }//c
	    }else{
	      for(int loop=0; loop < 2; loop++){
		c=startc[dir]+loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-2)*f_bound[dir][3])>> 1;	      
		  
		  int src_oddness = oddness;
		  int dst_oddness = oddness;
		  if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		    dst_oddness = 1-oddness;
		  }
		  MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_back_sendbuf, dst_idx, sitelink, src_idx, (endc[dir] -c +1)/2, dir);	

		}//if c
	      }//for loop
	    }//if
      
      
      //fwd
      for(d=X[dir]; d < X[dir]+2; d++)
	for(a=starta[dir];a < enda[dir]; a++)
	  for(b=startb[dir]; b < endb[dir]; b++)
	    
	    if(f_main[dir][2] != 1 || f_bound[dir][2] !=1){
	    for (c=startc[dir]; c < endc[dir]; c++){
	      int oddness = (a+b+c+d)%2;
	      int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
	      int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir])*f_bound[dir][3])>> 1;

	      int src_oddness = oddness;
	      int dst_oddness = oddness;
	      if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		dst_oddness = 1-oddness;
	      }

	      MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_fwd_sendbuf, dst_idx, sitelink, src_idx, 1,dir);
	    }//c
	    }else{
	      for(int loop=0; loop < 2; loop++){
		c=startc[dir]+loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir])*f_bound[dir][3])>> 1;
		  
		  int src_oddness = oddness;
		  int dst_oddness = oddness;
		  if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		    dst_oddness = 1-oddness;
		  }
		  
		  MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_fwd_sendbuf, dst_idx, sitelink, src_idx, (endc[dir] -c +1)/2,dir);	


		}
	      }//for loop
	    }//if
      comm_recv_with_tag(ghost_sitelink_back[dir], len[dir], back_nbr[dir], uptags[dir], &recv_request1[dir]);
      comm_recv_with_tag(ghost_sitelink_fwd[dir], len[dir], fwd_nbr[dir], downtags[dir], &recv_request2[dir]);
      comm_send_with_tag(ghost_sitelink_fwd_sendbuf[dir], len[dir], fwd_nbr[dir], uptags[dir], &send_request1[dir]);
      comm_send_with_tag(ghost_sitelink_back_sendbuf[dir], len[dir], back_nbr[dir], downtags[dir], &send_request2[dir]);    
      
      //need the messages to be here before we can send the next messages
      comm_wait(&recv_request1[dir]);
      comm_wait(&recv_request2[dir]);
      comm_wait(&send_request1[dir]);
      comm_wait(&send_request2[dir]);
    }//if

    //use the messages to fill the sitelink data
    //back
    if (dir < 3 ){
      for(d=0; d < 2; d++)
	for(a=starta[dir];a < enda[dir]; a++)
	  for(b=startb[dir]; b < endb[dir]; b++)
	    if(f_main[dir][2] != 1 || f_bound[dir][2] !=1){
	      for (c=startc[dir]; c < endc[dir]; c++){
		int oddness = (a+b+c+d)%2;
		int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		int src_idx;
		if(commDimPartitioned(dir)){
		  src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + d*f_bound[dir][3])>> 1;
		}else{
		  src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d+X[dir])*f_main[dir][3])>> 1;
		}
		/*
		if(oddness){
		  if(commDimPartitioned(dir)){
		    src_idx += nslices*slice_3d[dir]/2;
		  }else{
		    src_idx += Vh_ex;
		  }
		  dst_idx += Vh_ex;
		}
		*/
#define MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_buf, src_idx, num, dir) \
		if(oddness){						\
		  if(commDimPartitioned(dir)){				\
		    src_idx += nslices*slice_3d[dir]/2;			\
		  }else{						\
		    src_idx += Vh_ex;					\
		  }							\
		  dst_idx += Vh_ex;					\
		}							\
		if(cpu_order == QUDA_QDP_GAUGE_ORDER){			\
		  for(int linkdir=0; linkdir < 4; linkdir++){		\
		    char* src;						\
		    if(commDimPartitioned(dir)){			\
		      src = ((char*)ghost_buf[dir])+ linkdir*nslices*slice_3d[dir]*gaugebytes + (src_idx)*gaugebytes; \
		    }else{						\
		      src = ((char*)sitelink[linkdir])+ (src_idx)*gaugebytes; \
		    }							\
		    char* dst = (char*) sitelink[linkdir] + (dst_idx)*gaugebytes; \
		    memcpy(dst, src, gaugebytes*(num));			\
		  }							\
		}else{/*QUDA_MILC_GAUGE_FIELD*/				\
		  char* src;						\
		  if(commDimPartitioned(dir)){				\
		    src=((char*)ghost_buf[dir]) + 4*(src_idx)*gaugebytes; \
		  }else{						\
		    src = ((char*)sitelink)+ 4*(src_idx)*gaugebytes;	\
		  }							\
		  char* dst = ((char*)sitelink) + 4*(dst_idx)*gaugebytes; \
		  memcpy(dst, src, 4*gaugebytes*(num));			\
		}

		MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_back, src_idx, 1, dir);
		
	      }//c    
	    }else{
	      //optimized copy
	      //first half:   startc[dir] -> end[dir] with step=2

	      for(int loop =0;loop <2;loop++){
		int c=startc[dir]+loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int src_idx;
		  if(commDimPartitioned(dir)){
		    src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + d*f_bound[dir][3])>> 1;
		  }else{
		    src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d+X[dir])*f_main[dir][3])>> 1;
		  }

		  MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_back, src_idx, (endc[dir]-c+1)/2, dir);

		}//if c  		
	      }//for loop	      

	    }//if
    }else{
      //when dir == 3 (T direction), the data layout format in sitelink and the message is the same, we can do large copys
      
#define MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID_T(sitelink, ghost_buf, dst_face, src_face, dir) \
	/*even*/							\
	int even_dst_idx = (dst_face*E3E2E1)/2;				\
	int even_src_idx;						\
	if(commDimPartitioned(dir)){					\
	  even_src_idx = 0;						\
	}else{								\
	  even_src_idx = (src_face*E3E2E1)/2;				\
	}								\
 	/*odd*/								\
	int odd_dst_idx = even_dst_idx+Vh_ex;				\
	int odd_src_idx;						\
	if(commDimPartitioned(dir)){					\
	  odd_src_idx = nslices*slice_3d[dir]/2;			\
	}else{								\
	  odd_src_idx = even_src_idx+Vh_ex;				\
	}								\
	if(cpu_order == QUDA_QDP_GAUGE_ORDER){								\
	  for(int linkdir=0; linkdir < 4; linkdir ++){			\
	    char* dst = (char*)sitelink[linkdir];			\
	    char* src;							\
	    if(commDimPartitioned(dir)){				\
	      src = ((char*)ghost_buf[dir]) + linkdir*nslices*slice_3d[dir]*gaugebytes; \
	    }else{							\
	      src = (char*)sitelink[linkdir];				\
	    }								\
	    memcpy(dst + even_dst_idx * gaugebytes, src + even_src_idx*gaugebytes, nslices*slice_3d[dir]*gaugebytes/2); \
	    memcpy(dst + odd_dst_idx * gaugebytes, src + odd_src_idx*gaugebytes, nslices*slice_3d[dir]*gaugebytes/2); \
	  }								\
	}else{/*QUDA_MILC_GAUGE_ORDER*/					\
	  char* dst = (char*)sitelink;					\
	  char* src;							\
	  if(commDimPartitioned(dir)){					\
	    src = (char*)ghost_buf[dir];				\
	  }else{							\
	    src = (char*)sitelink;					\
	  }								\
	  memcpy(dst+4*even_dst_idx*gaugebytes, src+4*even_src_idx*gaugebytes, 4*nslices*slice_3d[dir]*gaugebytes/2); \
	  memcpy(dst+4*odd_dst_idx*gaugebytes, src+4*odd_src_idx*gaugebytes, 4*nslices*slice_3d[dir]*gaugebytes/2); \
	}      

      MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID_T(sitelink, ghost_sitelink_back, 0, X4, dir)
    }//if
    
    //fwd
    if( dir < 3 ){
      for(d=X[dir]+2; d < X[dir]+4; d++)
	for(a=starta[dir];a < enda[dir]; a++)
	  for(b=startb[dir]; b < endb[dir]; b++)

	    if(f_main[dir][2] != 1 || f_bound[dir][2] != 1){
	      for (c=startc[dir]; c < endc[dir]; c++){
		int oddness = (a+b+c+d)%2;
		int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		int src_idx;
		if(commDimPartitioned(dir)){
		  src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir]-2)*f_bound[dir][3])>> 1;
		}else{
		  src_idx =  ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d-X[dir])*f_main[dir][3])>> 1;
		}

		MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_fwd, src_idx, 1, dir);

	      }//c
	    }else{
	      for(int loop =0; loop < 2; loop++){
		//for (c=startc[dir]; c < endc[dir]; c++){
		c=startc[dir] + loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int src_idx;
		  if(commDimPartitioned(dir)){
		    src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir]-2)*f_bound[dir][3])>> 1;
		  }else{
		    src_idx =  ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d-X[dir])*f_main[dir][3])>> 1;
		  }

		  MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_fwd, src_idx, (endc[dir]-c+1)/2, dir);
		}//if
		
	      }//for loop
	      
	    }//if 
      

    }else{
      //when dir == 3 (T direction), the data layout format in sitelink and the message is the same, we can do large copys
      MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID_T(sitelink, ghost_sitelink_fwd, (X4+2), 2, dir)

    }//if    

  }//dir for loop
    
  
  for(int dir=0;dir < 4;dir++){
    if(!commDimPartitioned(dir)) continue;
    free(ghost_sitelink_fwd_sendbuf[dir]);
    free(ghost_sitelink_back_sendbuf[dir]);    
    free(ghost_sitelink_fwd[dir]);
    free(ghost_sitelink_back[dir]);    
  }
  
  
}



template<typename Float>
void
do_exchange_cpu_staple(Float* staple, Float** ghost_staple, Float** staple_fwd_sendbuf, Float** staple_back_sendbuf, int* X)
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
			     (void**)staple_fwd_sendbuf,  nFace, (QudaPrecision)(sizeof(Float)), X);

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
    
    MPI_Request recv_request1;  
    MPI_Request recv_request2;  
    MPI_Request send_request1; 
    MPI_Request send_request2; 
    comm_recv_with_tag(ghost_staple_back, 2*len[dir], back_neighbors[dir], up_tags[dir], &recv_request1);
    comm_recv_with_tag(ghost_staple_fwd, 2*len[dir], fwd_neighbors[dir], down_tags[dir], &recv_request2);
    comm_send_with_tag(staple_fwd_sendbuf[dir], 2*len[dir], fwd_neighbors[dir], up_tags[dir], &send_request1);
    comm_send_with_tag(staple_back_sendbuf[dir], 2*len[dir], back_neighbors[dir], down_tags[dir], &send_request2);
    
    comm_wait(&recv_request1);
    comm_wait(&recv_request2);
    comm_wait(&send_request1);
    comm_wait(&send_request2);
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
			   (double**)staple_fwd_sendbuf, (double**)staple_back_sendbuf, X);
  }else{ //single
    do_exchange_cpu_staple((float*)staple, (float**)ghost_staple, 
			   (float**)staple_fwd_sendbuf, (float**)staple_back_sendbuf, X);
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
  
  cudaGaugeField* cudaStaple = (cudaGaugeField*) _cudaStaple;
  exchange_llfat_init(cudaStaple->Precision());
  

  void* even = cudaStaple->Even_p();
  void* odd = cudaStaple->Odd_p();
  int volume = cudaStaple->VolumeCB();
  QudaPrecision prec = cudaStaple->Precision();
  int stride = cudaStaple->Stride();
  
  packGhostStaple(X, even, odd, volume, prec, stride, 
		  dir, whichway, fwd_nbr_staple_gpu, back_nbr_staple_gpu,
		  fwd_nbr_staple_sendbuf, back_nbr_staple_sendbuf, stream);
}


//@whichway indicates send direction
//we use recv_whichway to indicate recv direction
void
exchange_gpu_staple_comms(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream)
{
  cudaGaugeField* cudaStaple = (cudaGaugeField*) _cudaStaple;  
  QudaPrecision prec = cudaStaple->Precision();
  
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
  int len = Vs[i]*gaugeSiteSize*prec;
  int normlen = Vs[i]*sizeof(float);
  
  if(recv_whichway == QUDA_BACKWARDS){   
#ifdef GPU_DIRECT
    comm_recv_with_tag(back_nbr_staple[i], len, back_neighbors[i], up_tags[i], &llfat_recv_request1[i]);
    comm_send_with_tag(fwd_nbr_staple_sendbuf[i], len, fwd_neighbors[i],  up_tags[i], &llfat_send_request1[i]);
#else
    comm_recv_with_tag(back_nbr_staple_cpu[i], len, back_neighbors[i], up_tags[i], &llfat_recv_request1[i]);
    memcpy(fwd_nbr_staple_sendbuf_cpu[i], fwd_nbr_staple_sendbuf[i], len);
    comm_send_with_tag(fwd_nbr_staple_sendbuf_cpu[i], len, fwd_neighbors[i],  up_tags[i], &llfat_send_request1[i]);
#endif
  } else { // QUDA_FORWARDS
#ifdef GPU_DIRECT
    comm_recv_with_tag(fwd_nbr_staple[i], len, fwd_neighbors[i], down_tags[i], &llfat_recv_request2[i]);
    comm_send_with_tag(back_nbr_staple_sendbuf[i], len, back_neighbors[i] ,down_tags[i], &llfat_send_request2[i]);
#else
    comm_recv_with_tag(fwd_nbr_staple_cpu[i], len, fwd_neighbors[i], down_tags[i], &llfat_recv_request2[i]);
    memcpy(back_nbr_staple_sendbuf_cpu[i], back_nbr_staple_sendbuf[i], len);
    comm_send_with_tag(back_nbr_staple_sendbuf_cpu[i], len, back_neighbors[i] ,down_tags[i], &llfat_send_request2[i]);
#endif
  }
}


//@whichway indicates send direction
//we use recv_whichway to indicate recv direction
void
exchange_gpu_staple_wait(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream)
{
  cudaGaugeField* cudaStaple = (cudaGaugeField*) _cudaStaple;  

  void* even = cudaStaple->Even_p();
  void* odd = cudaStaple->Odd_p();
  int volume = cudaStaple->VolumeCB();
  QudaPrecision prec = cudaStaple->Precision();
  int stride = cudaStaple->Stride();

  int recv_whichway;
  if(whichway == QUDA_BACKWARDS){
    recv_whichway = QUDA_FORWARDS;
  }else{
    recv_whichway = QUDA_BACKWARDS;
  }
  

  int i = dir;
  int len = Vs[i]*gaugeSiteSize*prec;
  int normlen = Vs[i]*sizeof(float);
  
  if(recv_whichway == QUDA_BACKWARDS){   
    comm_wait(&llfat_recv_request1[i]);
    comm_wait(&llfat_send_request1[i]);

#ifdef GPU_DIRECT
    unpackGhostStaple(X, even, odd, volume, prec, stride, 
		      i, QUDA_BACKWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#else   
    memcpy(back_nbr_staple[i], back_nbr_staple_cpu[i], len);
    unpackGhostStaple(X, even, odd, volume, prec, stride, 
		      i, QUDA_BACKWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#endif

  } else { // QUDA_FORWARDS
    comm_wait(&llfat_recv_request2[i]);  
    comm_wait(&llfat_send_request2[i]);

#ifdef GPU_DIRECT
    unpackGhostStaple(X, even, odd, volume, prec, stride, 
		      i, QUDA_FORWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#else        
    memcpy(fwd_nbr_staple[i], fwd_nbr_staple_cpu[i], len);
    unpackGhostStaple(X, even, odd, volume, prec, stride,
		      i, QUDA_FORWARDS, fwd_nbr_staple, back_nbr_staple, stream);
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

#ifdef GPU_DIRECT
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

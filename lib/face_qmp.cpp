#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

/*
  Multi-GPU TODOs
  - test qmp code
  - implement OpenMP version?
  - split face kernels
  - separate block sizes for body and face
  - single coalesced D->H copy - first pass implemented, enable with GATHER_COALESCE 
    (could be done better as a kernel - add to blas and autotune)
  - minimize pointer arithmetic in core code (need extra constant to replace SPINOR_HOP)
 */

using namespace std;

cudaStream_t *stream;

// enabling this coalseces all per face transactions into a single buffer before the PCIe transfer
//#define GATHER_COALESCE

// Easy to switch between overlapping communication or not
#ifdef OVERLAP_COMMS
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpyAsync(dst, src, size, type, stream)
#else
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpy(dst, src, size, type)
#endif

FaceBuffer::FaceBuffer(const int *XX, const int nDim, const int Ninternal, 
		       const int nFace, const QudaPrecision precision) :
  my_fwd_face(0), my_back_face(0), from_back_face(0), from_fwd_face(0), 
  Ninternal(Ninternal), precision(precision), nDim(nDim), nFace(nFace)
{
  setupDims(XX);

  // set these both = 0 `for no overlap of qmp and cudamemcpyasync
  // sendBackStrmIdx = 0, and sendFwdStrmIdx = 1 for overlap
  sendBackStrmIdx = 0;
  sendFwdStrmIdx = 1;
  recFwdStrmIdx = sendBackStrmIdx;
  recBackStrmIdx = sendFwdStrmIdx;
  
  // Buffers hold half spinors
  nbytes = nFace*faceVolumeCB[3]*Ninternal*precision;
  
  // add extra space for the norms for half precision
  if (precision == QUDA_HALF_PRECISION) nbytes += nFace*faceVolumeCB[3]*sizeof(float);

  unsigned int flag = cudaHostAllocDefault;
  cudaHostAlloc(&(my_fwd_face), nbytes, flag);
  if( !my_fwd_face ) errorQuda("Unable to allocate my_fwd_face with size %lu", nbytes);
  
  cudaHostAlloc(&(my_back_face), nbytes, flag);
  if( !my_back_face ) errorQuda("Unable to allocate my_back_face with size %lu", nbytes);
  
#ifdef GATHER_COALESCE
  cudaMalloc(&(gather_fwd_face), nbytes);
  cudaMalloc(&(gather_back_face), nbytes);
#endif

#ifdef QMP_COMMS
  cudaHostAlloc(&(from_fwd_face), nbytes, flag);
  if( !from_fwd_face ) errorQuda("Unable to allocate from_fwd_face with size %lu", nbytes);
  
  cudaHostAlloc(&(from_back_face), nbytes, flag);
  if( !from_back_face ) errorQuda("Unable to allocate from_back_face with size %lu", nbytes);   
#else
  from_fwd_face = my_back_face;
  from_back_face = my_fwd_face;
#endif  


#ifdef QMP_COMMS
  mm_send_fwd = QMP_declare_msgmem(my_fwd_face, nbytes);
  if( mm_send_fwd == NULL ) errorQuda("Unable to allocate send fwd message mem");
  
  mm_send_back = QMP_declare_msgmem(my_back_face, nbytes);
  if( mm_send_back == NULL ) errorQuda("Unable to allocate send back message mem");
  
  mm_from_fwd = QMP_declare_msgmem(from_fwd_face, nbytes);
  if( mm_from_fwd == NULL ) errorQuda("Unable to allocate recv from fwd message mem");
  
  mm_from_back = QMP_declare_msgmem(from_back_face, nbytes);
  if( mm_from_back == NULL ) errorQuda("Unable to allocate recv from back message mem");
  
  mh_send_fwd = QMP_declare_send_relative(mm_send_fwd, 3, +1, 0);
  if( mh_send_fwd == NULL ) errorQuda("Unable to allocate forward send");
  
  mh_send_back = QMP_declare_send_relative(mm_send_back, 3, -1, 0);
  if( mh_send_back == NULL ) errorQuda("Unable to allocate backward send");
  
  mh_from_fwd = QMP_declare_receive_relative(mm_from_fwd, 3, +1, 0);
  if( mh_from_fwd == NULL ) errorQuda("Unable to allocate forward recv");
  
  mh_from_back = QMP_declare_receive_relative(mm_from_back, 3, -1, 0);
  if( mh_from_back == NULL ) errorQuda("Unable to allocate backward recv");
#endif

}

FaceBuffer::FaceBuffer(const FaceBuffer &face) {
  errorQuda("FaceBuffer copy constructor not implemented");
}

// FIXME: The input X here is already checkboarded so need to undo this
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
  
#ifdef QMP_COMMS
  QMP_free_msghandle(mh_send_fwd);
  QMP_free_msghandle(mh_send_back);
  QMP_free_msghandle(mh_from_fwd);
  QMP_free_msghandle(mh_from_back);
  QMP_free_msgmem(mm_send_fwd);
  QMP_free_msgmem(mm_send_back);
  QMP_free_msgmem(mm_from_fwd);
  QMP_free_msgmem(mm_from_back);
  cudaFreeHost(from_fwd_face); // these are aliasing pointers for non-qmp case
  cudaFreeHost(from_back_face);// these are aliasing pointers for non-qmp case
#endif
  cudaFreeHost(my_fwd_face);
  cudaFreeHost(my_back_face);

#ifdef GATHER_COALESCE
  cudaFree(gather_fwd_face);
  cudaFree(gather_back_face);
#endif

  my_fwd_face=NULL;
  my_back_face=NULL;
  from_fwd_face=NULL;
  from_back_face=NULL;
}

void FaceBuffer::exchangeFacesStart(cudaColorSpinorField &in, int parity,
				    int dagger, cudaStream_t *stream_p)
{
  stream = stream_p;

#ifdef QMP_COMMS
  // Prepost all receives
  QMP_start(mh_from_fwd);
  QMP_start(mh_from_back);
#endif

#ifdef GATHER_COALESCE
  void *back_face = gather_back_face;
  void *fwd_face = gather_fwd_face;
#else
  void *back_face = my_back_face;
  void *fwd_face = my_fwd_face;
#endif

  // gather for backwards send
  in.packGhost(back_face, 0, 3, QUDA_BACKWARDS, (QudaParity)parity, dagger, &stream[sendBackStrmIdx]);

  // gather for forwards send
  in.packGhost(fwd_face, 0, 3, QUDA_FORWARDS, (QudaParity)parity, dagger, &stream[sendFwdStrmIdx]);
 
#ifdef GATHER_COALESCE  
  // Copy to host if we are coalescing into single face messages to reduce latency
  CUDAMEMCPY((void *)my_back_face, (void *)gather_back_face,  nbytes, cudaMemcpyDeviceToHost, stream[sendBackStrmIdx]); 
  CUDAMEMCPY((void *)my_fwd_face, (void *)gather_fwd_face,  nbytes, cudaMemcpyDeviceToHost, stream[sendFwdStrmIdx]); 
#endif
}

void FaceBuffer::exchangeFacesComms() {

#ifdef OVERLAP_COMMS
  // Need to wait for copy to finish before sending to neighbour
  cudaStreamSynchronize(stream[sendBackStrmIdx]);
#endif

#ifdef QMP_COMMS
  // Begin backward send
  QMP_start(mh_send_back);
#endif

#ifdef OVERLAP_COMMS
  // Need to wait for copy to finish before sending to neighbour
  cudaStreamSynchronize(stream[sendFwdStrmIdx]);
#endif

#ifdef QMP_COMMS
  // Begin forward send
  QMP_start(mh_send_fwd);
#endif

} 

// Finish backwards send and forwards receive
#ifdef QMP_COMMS				
#define QMP_finish_from_fwd					\
  QMP_wait(mh_send_back);					\
  QMP_wait(mh_from_fwd);					\

// Finish forwards send and backwards receive
#define QMP_finish_from_back					\
  QMP_wait(mh_send_fwd);					\
  QMP_wait(mh_from_back);					\

#else
#define QMP_finish_from_fwd					

#define QMP_finish_from_back					

#endif

void FaceBuffer::exchangeFacesWait(cudaColorSpinorField &out, int dagger)
{
  // replaced this memcopy with aliasing pointers - useful benchmarking
#ifndef QMP_COMMS
  // NO QMP -- do copies
  //CUDAMEMCPY(from_fwd_face, my_back_face, nbytes, cudaMemcpyHostToHost, stream[sendBackStrmIdx]); // 174 without these
  //CUDAMEMCPY(from_back_face, my_fwd_face, nbytes, cudaMemcpyHostToHost, stream[sendFwdStrmIdx]);
#endif // QMP_COMMS

  // Scatter faces.
  QMP_finish_from_fwd;
  
  out.unpackGhost(from_fwd_face, 0, 3, QUDA_FORWARDS, dagger, &stream[recFwdStrmIdx]);

  QMP_finish_from_back;
  
  out.unpackGhost(from_back_face, 0, 3, QUDA_BACKWARDS, dagger, &stream[recBackStrmIdx]);
}

// This is just an initial hack for CPU comms
void FaceBuffer::exchangeCpuSpinor(char *spinor, char **fwd, char **back, int oddBit)
{

  //for all dimensions
  int len[4] = {
    nFace*faceVolumeCB[0]*Ninternal*precision,
    nFace*faceVolumeCB[1]*Ninternal*precision,
    nFace*faceVolumeCB[2]*Ninternal*precision,
    nFace*faceVolumeCB[3]*Ninternal*precision
  };

  char* fwd_sendbuf[4];
  char* back_sendbuf[4];
#ifdef QMP_COMMS
  for(int i=0;i < 4;i++){
    fwd_sendbuf[i] = (char*)malloc(len[i]);
    back_sendbuf[i] = (char*)malloc(len[i]);
  }
#else
  for(int i=0;i < 4;i++){
    fwd_sendbuf[i] = fwd[i];
    back_sendbuf[i] = back[i];
  }  
#endif

  int bytes = Ninternal*precision;

  for(int i=0; i<VolumeCB; i++){
    //compute full index
    int boundaryCrossings = i/(X[0]/2) + i/(X[0]*X[1]/2) + i/(X[0]*X[1]*X[2]/2);
    int Y = 2*i + (boundaryCrossings + oddBit) % 2;
    int x[4];
    x[3] = Y/(X[2]*X[1]*X[0]);
    x[2] = (Y/(X[1]*X[0])) % X[2];
    x[1] = (Y/X[0]) % X[1];
    x[0] = Y % X[0];

    int ghost_face_idx ;

    int offset = Ninternal*i*precision;

    //X dimension
    if (x[0] < nFace){
      ghost_face_idx = (x[0]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] +x[1])>>1 * precision;
      memcpy(&back_sendbuf[0][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }
    if (x[0] >=X[0]-nFace){
      ghost_face_idx = ((x[0]-X[0]+nFace)*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] +x[1])>>1 * precision;
      memcpy(&fwd_sendbuf[0][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }

    //Y dimension
    if (x[1] < nFace){
      ghost_face_idx = (x[1]*X[3]*X[2]*X[0] + x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1 * precision;
      memcpy(&back_sendbuf[1][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }
    if (x[1] >= X[1]-nFace){
      ghost_face_idx = ((x[1]-X[1]+nFace)*X[3]*X[2]*X[0]+ x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1 * precision;
      memcpy(&fwd_sendbuf[1][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }

    //Z dimension
    if (x[2] < nFace){
      ghost_face_idx = (x[2]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0]+x[1]*X[0]+x[0])>>1 * precision;
      memcpy(&back_sendbuf[2][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }
    if (x[2] >= X[2] - nFace){
      ghost_face_idx = ((x[2]-X[2]+nFace)*X[3]*X[1]*X[0] + x[3]*X[1]*X[0] + x[1]*X[0] + x[0])>>1 * precision;
      memcpy(&fwd_sendbuf[2][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }

    //T dimension
    if (x[3] < nFace){
      ghost_face_idx = (x[3]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1 * precision;
      memcpy(&back_sendbuf[3][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }
    if (x[3] >= X[3] - nFace){
      ghost_face_idx = ((x[3]-X[3]+nFace)*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1 * precision;
      memcpy(&fwd_sendbuf[3][Ninternal*ghost_face_idx], &spinor[offset], bytes);
    }

  }//i

#ifdef QMP_COMMS

  QMP_msgmem_t mm_send_fwd[4];
  QMP_msgmem_t mm_send_back[4];
  QMP_msgmem_t mm_from_fwd[4];
  QMP_msgmem_t mm_from_back[4];
  QMP_msghandle_t mh_send_fwd[4];
  QMP_msghandle_t mh_send_back[4];
  QMP_msghandle_t mh_from_fwd[4];
  QMP_msghandle_t mh_from_back[4];

  for (int i=0; i<4; i++) {
    mm_send_fwd[i] = QMP_declare_msgmem(fwd_sendbuf[i], len[i]);
    if( mm_send_fwd[i] == NULL ) errorQuda("Unable to allocate send fwd message mem");
    
    mm_send_back[i] = QMP_declare_msgmem(back_sendbuf[i], len[i]);
    if( mm_send_back == NULL ) errorQuda("Unable to allocate send back message mem");
    
    mm_from_fwd[i] = QMP_declare_msgmem(fwd[i], len[i]);
    if( mm_from_fwd[i] == NULL ) errorQuda("Unable to allocate recv from fwd message mem");
    
    mm_from_back[i] = QMP_declare_msgmem(back[i], len[i]);
    if( mm_from_back[i] == NULL ) errorQuda("Unable to allocate recv from back message mem");
    
    mh_send_fwd[i] = QMP_declare_send_relative(mm_send_fwd[i], i, +1, 0);
    if( mh_send_fwd[i] == NULL ) errorQuda("Unable to allocate forward send");
    
    mh_send_back[i] = QMP_declare_send_relative(mm_send_back[i], i, -1, 0);
    if( mh_send_back[i] == NULL ) errorQuda("Unable to allocate backward send");
    
    mh_from_fwd[i] = QMP_declare_receive_relative(mm_from_fwd[i], i, +1, 0);
    if( mh_from_fwd[i] == NULL ) errorQuda("Unable to allocate forward recv");
    
    mh_from_back[i] = QMP_declare_receive_relative(mm_from_back[i], i, -1, 0);
    if( mh_from_back[i] == NULL ) errorQuda("Unable to allocate backward recv");
  }

  for (int i=0; i<4; i++) {
    QMP_start(mh_from_back[i]);
    QMP_start(mh_from_fwd[i]);
    QMP_start(mh_send_fwd[i]);
    QMP_start(mh_send_back[i]);
  }

  for (int i=0; i<4; i++) {
    QMP_wait(mh_send_fwd[i]);
    QMP_wait(mh_send_back[i]);
    QMP_wait(mh_from_back[i]);
    QMP_wait(mh_from_fwd[i]);
  }

  for (int i=0; i<4; i++) {
    QMP_free_msghandle(mh_send_fwd[i]);
    QMP_free_msghandle(mh_from_back[i]);
    QMP_free_msghandle(mh_from_fwd[i]);
    QMP_free_msghandle(mh_send_back[i]);
    QMP_free_msgmem(mm_send_fwd[i]);
    QMP_free_msgmem(mm_from_back[i]);
    QMP_free_msgmem(mm_from_fwd[i]);
    QMP_free_msgmem(mm_send_back[i]);
  }

  for(int i=0;i < 4;i++){
    free(fwd_sendbuf[i]);
    free(back_sendbuf[i]);
  }
#endif
}

void FaceBuffer::exchangeCpuLink(void** ghost_link, void** link_sendbuf, int nFace) {

#ifdef QMP_COMMS

  QMP_msgmem_t mm_send_fwd[4];
  QMP_msgmem_t mm_from_back[4];
  QMP_msghandle_t mh_send_fwd[4];
  QMP_msghandle_t mh_from_back[4];

  for (int i=0; i<4; i++) {
    int len = nFace*faceVolumeCB[i]*Ninternal*precision;
    mm_send_fwd[i] = QMP_declare_msgmem(link_sendbuf[i], 2*len);
    if( mm_send_fwd[i] == NULL ) errorQuda("Unable to allocate send fwd message mem");
    
    mm_from_back[i] = QMP_declare_msgmem(ghost_link[i], 2*len);
    if( mm_from_back[i] == NULL ) errorQuda("Unable to allocate recv from back message mem");
    
    mh_send_fwd[i] = QMP_declare_send_relative(mm_send_fwd[i], i, +1, 0);
    if( mh_send_fwd[i] == NULL ) errorQuda("Unable to allocate forward send");
    
    mh_from_back[i] = QMP_declare_receive_relative(mm_from_back[i], i, -1, 0);
    if( mh_from_back[i] == NULL ) errorQuda("Unable to allocate backward recv");
  }

  for (int i=0; i<4; i++) {
    QMP_start(mh_send_fwd[i]);
    QMP_start(mh_from_back[i]);
  }

  for (int i=0; i<4; i++) {
    QMP_wait(mh_send_fwd[i]);
    QMP_wait(mh_from_back[i]);
  }

  for (int i=0; i<4; i++) {
    QMP_free_msghandle(mh_send_fwd[i]);
    QMP_free_msghandle(mh_from_back[i]);
    QMP_free_msgmem(mm_send_fwd[i]);
    QMP_free_msgmem(mm_from_back[i]);
  }

#else

  for(int dir =0; dir < 4; dir++) {
    int len = nFace*faceVolumeCB[dir]*gaugeSiteSize*precision;
    memcpy(ghost_link[dir], link_sendbuf[i], 2*len);
  }

#endif

}



void transferGaugeFaces(void *gauge, void *gauge_face, QudaPrecision precision,
			int Nvec, ReconstructType reconstruct, int V, int Vs)
{
  int nblocks, ndim=4;
  size_t blocksize;//, nbytes;
  ptrdiff_t offset, stride;
  void *g;

  nblocks = ndim*reconstruct/Nvec;
  blocksize = Vs*Nvec*precision;
  offset = (V-Vs)*Nvec*precision;
  stride = (V+Vs)*Nvec*precision; // assume that pad = Vs

#ifdef QMP_COMMS

  QMP_msgmem_t mm_gauge_send_fwd;
  QMP_msgmem_t mm_gauge_from_back;
  QMP_msghandle_t mh_gauge_send_fwd;
  QMP_msghandle_t mh_gauge_from_back;

  g = (void *) ((char *) gauge + offset);
  mm_gauge_send_fwd = QMP_declare_strided_msgmem(g, blocksize, nblocks, stride);
  if (!mm_gauge_send_fwd) {
    errorQuda("Unable to allocate gauge message mem");
  }

  mm_gauge_from_back = QMP_declare_strided_msgmem(gauge_face, blocksize, nblocks, stride);
  if (!mm_gauge_from_back) { 
    errorQuda("Unable to allocate gauge face message mem");
  }

  mh_gauge_send_fwd = QMP_declare_send_relative(mm_gauge_send_fwd, 3, +1, 0);
  if (!mh_gauge_send_fwd) {
    errorQuda("Unable to allocate gauge message handle");
  }
  mh_gauge_from_back = QMP_declare_receive_relative(mm_gauge_from_back, 3, -1, 0);
  if (!mh_gauge_from_back) {
    errorQuda("Unable to allocate gauge face message handle");
  }

  QMP_start(mh_gauge_send_fwd);
  QMP_start(mh_gauge_from_back);
  
  QMP_wait(mh_gauge_send_fwd);
  QMP_wait(mh_gauge_from_back);

  QMP_free_msghandle(mh_gauge_send_fwd);
  QMP_free_msghandle(mh_gauge_from_back);
  QMP_free_msgmem(mm_gauge_send_fwd);
  QMP_free_msgmem(mm_gauge_from_back);

#else 

  void *gf;

  for (int i=0; i<nblocks; i++) {
    g = (void *) ((char *) gauge + offset + i*stride);
    gf = (void *) ((char *) gauge_face + i*stride);
    cudaMemcpy(gf, g, blocksize, cudaMemcpyHostToHost);
  }

#endif // QMP_COMMS
}

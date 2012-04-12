#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>
#include <gauge_field.h>
#include <sys/time.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

/*
  Multi-GPU TODOs
  - test qmp code
  - implement OpenMP version?
  - split face kernels
  - separate block sizes for body and face
  - minimize pointer arithmetic in core code (need extra constant to replace SPINOR_HOP)
 */

using namespace std;

cudaStream_t *stream;

bool globalReduce = true;

// Easy to switch between overlapping communication or not
#ifdef OVERLAP_COMMS
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpyAsync(dst, src, size, type, stream)
#else
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpy(dst, src, size, type)
#endif

FaceBuffer::FaceBuffer(const int *X, const int nDim, const int Ninternal, 
		       const int nFace, const QudaPrecision precision, const int Ls) :
  Ninternal(Ninternal), precision(precision), nDim(nDim), nFace(nFace)
{
  if (nDim > QUDA_MAX_DIM) errorQuda("nDim = %d is greater than the maximum of %d\n", nDim, QUDA_MAX_DIM);
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

  // set these both = 0 separate streams for forwards and backwards comms
  // sendBackStrmIdx = 0, and sendFwdStrmIdx = 1 for overlap
  sendBackStrmIdx = 0;
  sendFwdStrmIdx = 1;
  recFwdStrmIdx = sendBackStrmIdx;
  recBackStrmIdx = sendFwdStrmIdx;
  
  unsigned int flag = cudaHostAllocDefault;

  //printf("nDim = %d\n", nDim);

  // Buffers hold half spinors
  for (int i=0; i<nDim; i++) {
    nbytes[i] = nFace*faceVolumeCB[i]*Ninternal*precision;

    // add extra space for the norms for half precision
    if (precision == QUDA_HALF_PRECISION) nbytes[i] += nFace*faceVolumeCB[i]*sizeof(float);
    //printf("bytes = %d, nFace = %d, faceVolume = %d, Ndof = %d, prec =  %d\n", 
    //	   nbytes[i], nFace, faceVolumeCB[i], Ninternal, precision);

    cudaHostAlloc(&(my_fwd_face[i]), nbytes[i], flag);
    if( !my_fwd_face[i] ) errorQuda("Unable to allocate my_fwd_face with size %lu", nbytes[i]);
  
    //printf("%d\n", nbytes[i]);

    cudaHostAlloc(&(my_back_face[i]), nbytes[i], flag);
    if( !my_back_face[i] ) errorQuda("Unable to allocate my_back_face with size %lu", nbytes[i]);
  }

  for (int i=0; i<nDim; i++) {
#ifdef QMP_COMMS
    cudaHostAlloc(&(from_fwd_face[i]), nbytes[i], flag);
    if( !from_fwd_face[i] ) errorQuda("Unable to allocate from_fwd_face with size %lu", nbytes[i]);
    
    cudaHostAlloc(&(from_back_face[i]), nbytes[i], flag);
    if( !from_back_face[i] ) errorQuda("Unable to allocate from_back_face with size %lu", nbytes[i]);

// if no GPUDirect so need separate IB and GPU host buffers
#ifndef GPU_DIRECT
    ib_my_fwd_face[i] = malloc(nbytes[i]);
    if (!ib_my_fwd_face[i]) errorQuda("Unable to allocate ib_my_fwd_face with size %lu", nbytes[i]);

    ib_my_back_face[i] = malloc(nbytes[i]);
    if (!ib_my_back_face[i]) errorQuda("Unable to allocate ib_my_back_face with size %lu", nbytes[i]);

    ib_from_fwd_face[i] = malloc(nbytes[i]);
    if (!ib_from_fwd_face[i]) errorQuda("Unable to allocate ib_from_fwd_face with size %lu", nbytes[i]);

    ib_from_back_face[i] = malloc(nbytes[i]);
    if (!ib_from_back_face[i]) errorQuda("Unable to allocate ib_from_back_face with size %lu", nbytes[i]);
#else // else just alias the pointer
    ib_my_fwd_face[i] = my_fwd_face[i];
    ib_my_back_face[i] = my_back_face[i];
    ib_from_fwd_face[i] = from_fwd_face[i];
    ib_from_back_face[i] = from_back_face[i];
#endif

#else
    from_fwd_face[i] = my_back_face[i];
    from_back_face[i] = my_fwd_face[i];
#endif  
  }

#ifdef QMP_COMMS
  for (int i=0; i<nDim; i++) {

    mm_send_fwd[i] = QMP_declare_msgmem(ib_my_fwd_face[i], nbytes[i]);
    if( mm_send_fwd[i] == NULL ) errorQuda("Unable to allocate send fwd message mem");
    
    mm_send_back[i] = QMP_declare_msgmem(ib_my_back_face[i], nbytes[i]);
    if( mm_send_back[i] == NULL ) errorQuda("Unable to allocate send back message mem");
    
    mm_from_fwd[i] = QMP_declare_msgmem(ib_from_fwd_face[i], nbytes[i]);
    if( mm_from_fwd[i] == NULL ) errorQuda("Unable to allocate recv from fwd message mem");
    
    mm_from_back[i] = QMP_declare_msgmem(ib_from_back_face[i], nbytes[i]);
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
#endif

}

FaceBuffer::FaceBuffer(const FaceBuffer &face) {
  errorQuda("FaceBuffer copy constructor not implemented");
}

void FaceBuffer::setupDims(const int* X)
{
  Volume = 1;
  for (int d=0; d< nDim; d++) {
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
  
  //printf("Ndim = %d\n", nDim);
  for (int i=0; i<nDim; i++) {
#ifdef QMP_COMMS

#ifndef GPU_DIRECT
    free(ib_my_fwd_face[i]);
    free(ib_my_back_face[i]);
    free(ib_from_fwd_face[i]);
    free(ib_from_back_face[i]);
#endif

    QMP_free_msghandle(mh_send_fwd[i]);
    QMP_free_msghandle(mh_send_back[i]);
    QMP_free_msghandle(mh_from_fwd[i]);
    QMP_free_msghandle(mh_from_back[i]);
    QMP_free_msgmem(mm_send_fwd[i]);
    QMP_free_msgmem(mm_send_back[i]);
    QMP_free_msgmem(mm_from_fwd[i]);
    QMP_free_msgmem(mm_from_back[i]);
    cudaFreeHost(from_fwd_face[i]); // these are aliasing pointers for non-qmp case
    cudaFreeHost(from_back_face[i]);// these are aliasing pointers for non-qmp case
#endif
    cudaFreeHost(my_fwd_face[i]);
    cudaFreeHost(my_back_face[i]);
  }

  for (int i=0; i<nDim; i++) {
    my_fwd_face[i]=NULL;
    my_back_face[i]=NULL;
    from_fwd_face[i]=NULL;
    from_back_face[i]=NULL;
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

  if (dir%2==0) {
    // backwards copy to host
    in.sendGhost(my_back_face[dim], dim, QUDA_BACKWARDS, dagger, &stream[2*dim+sendBackStrmIdx]);  
  } else {
    // forwards copy to host
    in.sendGhost(my_fwd_face[dim], dim, QUDA_FORWARDS, dagger, &stream[2*dim+sendFwdStrmIdx]);
  }
}

void FaceBuffer::commsStart(int dir) {
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return;

  if (dir%2 == 0) { // sending backwards

#ifdef QMP_COMMS  // Begin backward send
    // Prepost receive
    QMP_start(mh_from_fwd[dim]);
#ifndef GPU_DIRECT
    memcpy(ib_my_back_face[dim], my_back_face[dim], nbytes[dim]);
#endif
    QMP_start(mh_send_back[dim]);
#endif

  } else { //sending forwards
    
#ifdef QMP_COMMS
  // Prepost receive
    QMP_start(mh_from_back[dim]);
    // Begin forward send
#ifndef GPU_DIRECT
    memcpy(ib_my_fwd_face[dim], my_fwd_face[dim], nbytes[dim]);
#endif
    QMP_start(mh_send_fwd[dim]);
#endif
  }

} 

int FaceBuffer::commsQuery(int dir) {

#ifdef QMP_COMMS

  int dim = dir/2;
  if(!commDimPartitioned(dim)) return 0;

  if (dir%2==0) {// receive from forwards
    if (QMP_is_complete(mh_send_back[dim]) == QMP_TRUE &&
	QMP_is_complete(mh_from_fwd[dim]) == QMP_TRUE) {
#ifndef GPU_DIRECT
      memcpy(from_fwd_face[dim], ib_from_fwd_face[dim], nbytes[dim]);		
#endif
      return 1;
    }
  } else { // receive from backwards
    if (QMP_is_complete(mh_send_fwd[dim]) == QMP_TRUE && 
	QMP_is_complete(mh_from_back[dim]) == QMP_TRUE) {
#ifndef GPU_DIRECT
      memcpy(from_back_face[dim], ib_from_back_face[dim], nbytes[dim]);		
#endif
      return 1;
    }
  }
  return 0;

#else // no communications so just return true

  return 1;

#endif
}

void FaceBuffer::scatter(cudaColorSpinorField &out, int dagger, int dir)
{
  int dim = dir/2;
  if(!commDimPartitioned(dim)) return;

  if (dir%2==0) {// receive from forwards
    out.unpackGhost(from_fwd_face[dim], dim, QUDA_FORWARDS, dagger, &stream[2*dim+recFwdStrmIdx]); // 0, 2, 4, 6
  } else { // receive from backwards
    out.unpackGhost(from_back_face[dim], dim, QUDA_BACKWARDS, dagger, &stream[2*dim+recBackStrmIdx]); // 1, 3, 5, 7
  }
}

// This is just an initial hack for CPU comms - should be creating the message handlers at instantiation
void FaceBuffer::exchangeCpuSpinor(cpuColorSpinorField &spinor, int oddBit, int dagger)
{

  // allocate the ghost buffer if not yet allocated
  spinor.allocateGhostBuffer();

  for(int i=0;i < 4; i++){
    spinor.packGhost(spinor.backGhostFaceSendBuffer[i], i, QUDA_BACKWARDS, (QudaParity)oddBit, dagger);
    spinor.packGhost(spinor.fwdGhostFaceSendBuffer[i], i, QUDA_FORWARDS, (QudaParity)oddBit, dagger);
  }

#ifdef QMP_COMMS

  QMP_msgmem_t mm_send_fwd[4];
  QMP_msgmem_t mm_from_back[4];
  QMP_msgmem_t mm_from_fwd[4];
  QMP_msgmem_t mm_send_back[4];
  QMP_msghandle_t mh_send_fwd[4];
  QMP_msghandle_t mh_from_back[4];
  QMP_msghandle_t mh_from_fwd[4];
  QMP_msghandle_t mh_send_back[4];

  for (int i=0; i<4; i++) {
    mm_send_fwd[i] = QMP_declare_msgmem(spinor.fwdGhostFaceSendBuffer[i], nbytes[i]);
    if( mm_send_fwd[i] == NULL ) errorQuda("Unable to allocate send fwd message mem");
    
    mm_send_back[i] = QMP_declare_msgmem(spinor.backGhostFaceSendBuffer[i], nbytes[i]);
    if( mm_send_back == NULL ) errorQuda("Unable to allocate send back message mem");
    
    mm_from_fwd[i] = QMP_declare_msgmem(spinor.fwdGhostFaceBuffer[i], nbytes[i]);
    if( mm_from_fwd[i] == NULL ) errorQuda("Unable to allocate recv from fwd message mem");
    
    mm_from_back[i] = QMP_declare_msgmem(spinor.backGhostFaceBuffer[i], nbytes[i]);
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
    QMP_free_msghandle(mh_send_back[i]);
    QMP_free_msghandle(mh_from_fwd[i]);
    QMP_free_msghandle(mh_from_back[i]);
    QMP_free_msgmem(mm_send_fwd[i]);
    QMP_free_msgmem(mm_send_back[i]);
    QMP_free_msgmem(mm_from_back[i]);
    QMP_free_msgmem(mm_from_fwd[i]);
  }

#else

  for (int i=0; i<4; i++) {
    //printf("%d COPY length = %d\n", i, nbytes[i]/precision);
    memcpy(spinor.fwdGhostFaceBuffer[i], spinor.backGhostFaceSendBuffer[i], nbytes[i]);
    memcpy(spinor.backGhostFaceBuffer[i], spinor.fwdGhostFaceSendBuffer[i], nbytes[i]);
  }

#endif
}

void FaceBuffer::exchangeCpuLink(void** ghost_link, void** link_sendbuf) {

#ifdef QMP_COMMS

  QMP_msgmem_t mm_send_fwd[4];
  QMP_msgmem_t mm_from_back[4];
  QMP_msghandle_t mh_send_fwd[4];
  QMP_msghandle_t mh_from_back[4];

  for (int i=0; i<4; i++) {
    int len = 2*nFace*faceVolumeCB[i]*Ninternal;
    mm_send_fwd[i] = QMP_declare_msgmem(link_sendbuf[i], len*precision);
    if( mm_send_fwd[i] == NULL ) errorQuda("Unable to allocate send fwd message mem");
    
    mm_from_back[i] = QMP_declare_msgmem(ghost_link[i], len*precision);
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
    int len = 2*nFace*faceVolumeCB[dir]*Ninternal; // factor 2 since we have both parities
    //printf("%d COPY length = %d\n", dir, len);
    memcpy(ghost_link[dir], link_sendbuf[dir], len*precision); 
  }

#endif

}



void transferGaugeFaces(void *gauge, void *gauge_face, QudaPrecision precision,
			int Nvec, QudaReconstructType reconstruct, int V, int Vs)
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

void reduceMaxDouble(double &max) {

#ifdef QMP_COMMS
  QMP_max_double(&max);
#endif

}

void reduceDouble(double &sum) {

#ifdef QMP_COMMS
  if (globalReduce) QMP_sum_double(&sum);
#endif

}

void reduceDoubleArray(double *sum, const int len) {

#ifdef QMP_COMMS
  if (globalReduce) QMP_sum_double_array(sum,len);
#endif

}

#ifdef QMP_COMMS
static int manual_set_partition[4] ={0, 0, 0, 0};
int commDim(int dir) { return QMP_get_logical_dimensions()[dir]; }
int commCoords(int dir) { return QMP_get_logical_coordinates()[dir]; }
int commDimPartitioned(int dir){ return (manual_set_partition[dir] || ((commDim(dir) > 1)));}
void commDimPartitionedSet(int dir){ manual_set_partition[dir] = 1; }
void commBarrier() { QMP_barrier(); }
#else
int commDim(int dir) { return 1; }
int commCoords(int dir) { return 0; }
int commDimPartitioned(int dir){ return 0; }
void commDimPartitionedSet(int dir){ ; }
void commBarrier() { ; }
#endif

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

#define QMP_CHECK(a)							\
  {QMP_status_t status;							\
  if ((status = a) != QMP_SUCCESS)					\
    errorQuda("QMP returned with error %s", QMP_error_string(status) );	\
  }

using namespace quda;

extern cudaStream_t *stream;

bool globalReduce = true;

// Easy to switch between overlapping communication or not
#ifdef OVERLAP_COMMS
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpyAsync(dst, src, size, type, stream)
#else
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpy(dst, src, size, type)
#endif

FaceBuffer::FaceBuffer(const int *X, const int nDim, const int Ninternal, 
		       const int nFace, const QudaPrecision precision, const int Ls) :
  Ninternal(Ninternal), precision(precision), nDim(nDim), nDimComms(nDim), nFace(nFace)
{
  if (nDim > QUDA_MAX_DIM) errorQuda("nDim = %d is greater than the maximum of %d\n", nDim, QUDA_MAX_DIM);
//BEGIN NEW
  int Y[nDim];
  Y[0] = X[0];
  Y[1] = X[1];
  Y[2] = X[2];
  Y[3] = X[3];
  if(nDim == 5) {
    Y[nDim-1] = Ls;
    nDimComms = 4;
  }
  setupDims(Y);

  // set these both = 0 separate streams for forwards and backwards comms
  // sendBackStrmIdx = 0, and sendFwdStrmIdx = 1 for overlap
  sendBackStrmIdx = 0;
  sendFwdStrmIdx = 1;
  recFwdStrmIdx = sendBackStrmIdx;
  recBackStrmIdx = sendFwdStrmIdx;

  // Buffers hold half spinors
  for (int i=0; i<nDimComms; i++) {
    nbytes[i] = nFace*faceVolumeCB[i]*Ninternal*precision;
    // add extra space for the norms for half precision
    if (precision == QUDA_HALF_PRECISION) nbytes[i] += nFace*faceVolumeCB[i]*sizeof(float);

    my_fwd_face[i] = allocatePinned(nbytes[i]);
    my_back_face[i] = allocatePinned(nbytes[i]);

#ifdef QMP_COMMS

    from_fwd_face[i] = allocatePinned(nbytes[i]);
    from_back_face[i] = allocatePinned(nbytes[i]);


#ifdef GPU_DIRECT //  just alias the pointer
    ib_my_fwd_face[i] = my_fwd_face[i];
    ib_my_back_face[i] = my_back_face[i];
    ib_from_fwd_face[i] = from_fwd_face[i];
    ib_from_back_face[i] = from_back_face[i];
#else // if no GPUDirect so need separate IB and GPU host buffers
    ib_my_fwd_face[i] = safe_malloc(nbytes[i]);
    ib_my_back_face[i] = safe_malloc(nbytes[i]);
    ib_from_fwd_face[i] = safe_malloc(nbytes[i]);
    ib_from_back_face[i] = safe_malloc(nbytes[i]);
#endif

#else
    from_fwd_face[i] = my_back_face[i];
    from_back_face[i] = my_fwd_face[i];
#endif  
  }

#ifdef QMP_COMMS
  for (int i=0; i<nDimComms; i++) {

    mm_send_fwd[i] = QMP_declare_msgmem(ib_my_fwd_face[i], nbytes[i]);
    if( mm_send_fwd[i] == NULL ) errorQuda("Unable to allocate send fwd message mem");
    
    mm_send_back[i] = QMP_declare_msgmem(ib_my_back_face[i], nbytes[i]);
    if( mm_send_back[i] == NULL ) errorQuda("Unable to allocate send back message mem");
    
    mm_from_fwd[i] = QMP_declare_msgmem(ib_from_fwd_face[i], nbytes[i]);
    if( mm_from_fwd[i] == NULL ) errorQuda("Unable to allocate recv from fwd message mem");
    
    mm_from_back[i] = QMP_declare_msgmem(ib_from_back_face[i], nbytes[i]);
    if( mm_from_back[i] == NULL ) errorQuda("Unable to allocate recv from back message mem");

    send_handle_fwd[i] = comm_declare_send_relative(mm_send_fwd[i], i, +1);
    send_handle_back[i] = comm_declare_send_relative(mm_send_back[i], i, -1);
    recv_handle_fwd[i] = comm_declare_receive_relative(mm_from_fwd[i], i, +1);
    recv_handle_back[i] = comm_declare_receive_relative(mm_from_back[i], i, -1);
  }
#endif

  checkCudaError();
}

FaceBuffer::~FaceBuffer()
{  
  for (int i=0; i<nDimComms; i++) {
#ifdef QMP_COMMS

#ifndef GPU_DIRECT
    host_free(ib_my_fwd_face[i]);
    host_free(ib_my_back_face[i]);
    host_free(ib_from_fwd_face[i]);
    host_free(ib_from_back_face[i]);
#endif

    ib_my_fwd_face[i] = NULL;
    ib_my_back_face[i] = NULL;
    ib_from_fwd_face[i] = NULL;
    ib_from_back_face[i] = NULL;

    comm_free(send_handle_fwd[i]);
    comm_free(send_handle_back[i]);
    comm_free(recv_handle_fwd[i]);
    comm_free(recv_handle_back[i]);

    QMP_free_msgmem(mm_send_fwd[i]);
    QMP_free_msgmem(mm_send_back[i]);
    QMP_free_msgmem(mm_from_fwd[i]);
    QMP_free_msgmem(mm_from_back[i]);

    freePinned(from_fwd_face[i]);
    freePinned(from_back_face[i]);

#endif // QMP_COMMS

    freePinned(my_fwd_face[i]);
    freePinned(my_back_face[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    my_fwd_face[i]=NULL;
    my_back_face[i]=NULL;
    from_fwd_face[i]=NULL;
    from_back_face[i]=NULL;

    recv_handle_fwd[i] = NULL;
    recv_handle_back[i] = NULL;
    send_handle_fwd[i] = NULL;
    send_handle_back[i] = NULL;
  }

  checkCudaError();
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
    QMP_CHECK(QMP_start(mh_from_back[i]));
    QMP_CHECK(QMP_start(mh_from_fwd[i]));
    QMP_CHECK(QMP_start(mh_send_fwd[i]));
    QMP_CHECK(QMP_start(mh_send_back[i]));
  }

  for (int i=0; i<4; i++) {
    QMP_CHECK(QMP_wait(mh_send_fwd[i]));
    QMP_CHECK(QMP_wait(mh_send_back[i]));
    QMP_CHECK(QMP_wait(mh_from_back[i]));
    QMP_CHECK(QMP_wait(mh_from_fwd[i]));
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
    QMP_CHECK(QMP_start(mh_send_fwd[i]));
    QMP_CHECK(QMP_start(mh_from_back[i]));
  }

  for (int i=0; i<4; i++) {
    QMP_CHECK(QMP_wait(mh_send_fwd[i]));
    QMP_CHECK(QMP_wait(mh_from_back[i]));
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

#ifdef QMP_COMMS
static int manual_set_partition[4] ={0, 0, 0, 0};
int commDim(int dir) { return QMP_get_logical_dimensions()[dir]; }
int commCoords(int dir) { return QMP_get_logical_coordinates()[dir]; }
int commDimPartitioned(int dir){ return (manual_set_partition[dir] || ((commDim(dir) > 1)));}
void commDimPartitionedSet(int dir){ manual_set_partition[dir] = 1; }
#endif

#include <quda_internal.h>
#include <face_quda.h>
#include <dslash_quda.h>

#ifndef GPU_DIRECT
#include <string.h>    
#endif

using namespace quda;

cudaStream_t *stream;

bool globalReduce = true;


FaceBuffer::FaceBuffer(const int *X, const int nDim, const int Ninternal, 
		       const int nFace, const QudaPrecision precision, const int Ls) :
  Ninternal(Ninternal), precision(precision), nDim(nDim), nDimComms(nDim), nFace(nFace)
{
  setupDims(X, Ls);

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

    my_back_face[i] = allocatePinned(2*nbytes[i]);
    my_fwd_face[i] = (char*)my_back_face[i] + nbytes[i];
    from_back_face[i] = allocatePinned(2*nbytes[i]);
    from_fwd_face[i] = (char*)from_back_face[i] + nbytes[i];

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

  }

  for (int i=0; i<nDimComms; i++) {
    mh_send_fwd[i] = comm_declare_send_relative(ib_my_fwd_face[i], i, 1, nbytes[i]);
    mh_send_back[i] = comm_declare_send_relative(ib_my_back_face[i], i, -1, nbytes[i]);
    mh_recv_fwd[i] = comm_declare_receive_relative(ib_from_fwd_face[i], i, +1, nbytes[i]);
    mh_recv_back[i] = comm_declare_receive_relative(ib_from_back_face[i], i, -1, nbytes[i]);
  }

  checkCudaError();
}


FaceBuffer::FaceBuffer(const FaceBuffer &face) {
  errorQuda("FaceBuffer copy constructor not implemented");
}


FaceBuffer::~FaceBuffer()
{  
  for (int i=0; i<nDimComms; i++) {

#ifndef GPU_DIRECT
    host_free(ib_my_fwd_face[i]);
    host_free(ib_my_back_face[i]);
    host_free(ib_from_fwd_face[i]);
    host_free(ib_from_back_face[i]);
#endif

    comm_free(mh_send_fwd[i]);
    comm_free(mh_send_back[i]);
    comm_free(mh_recv_fwd[i]);
    comm_free(mh_recv_back[i]);

    freePinned(from_back_face[i]);
    freePinned(my_back_face[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    ib_my_fwd_face[i] = NULL;
    ib_my_back_face[i] = NULL;
    ib_from_fwd_face[i] = NULL;
    ib_from_back_face[i] = NULL;

    my_fwd_face[i] = NULL;
    my_back_face[i] = NULL;
    from_fwd_face[i] = NULL;
    from_back_face[i] = NULL;

    mh_recv_fwd[i] = NULL;
    mh_recv_back[i] = NULL;
    mh_send_fwd[i] = NULL;
    mh_send_back[i] = NULL;
  }

  checkCudaError();
}


// X here is a checkboarded volume
void FaceBuffer::setupDims(const int* X, int Ls)
{
  if (nDim > QUDA_MAX_DIM) errorQuda("nDim = %d is greater than the maximum of %d\n", nDim, QUDA_MAX_DIM);
  for (int d=0; d<4; d++) this->X[d] = X[d];
  if(nDim == 5) {
    this->X[nDim-1] = Ls;
    nDimComms = 4;
  }

  Volume = 1;
  for (int d=0; d<nDim; d++) Volume *= this->X[d];    
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


// cache of inactive allocations
std::multimap<size_t, void *> FaceBuffer::pinnedCache;

// sizes of active allocations
std::map<void *, size_t> FaceBuffer::pinnedSize;


void *FaceBuffer::allocatePinned(size_t nbytes)
{
  std::multimap<size_t, void *>::iterator it;
  void *ptr = 0;

  if (pinnedCache.empty()) {
    ptr = pinned_malloc(nbytes);
  } else {
    it = pinnedCache.lower_bound(nbytes);
    if (it != pinnedCache.end()) { // sufficiently large allocation found
      nbytes = it->first;
      ptr = it->second;
      pinnedCache.erase(it);
    } else { // sacrifice the smallest cached allocation
      it = pinnedCache.begin();
      ptr = it->second;
      pinnedCache.erase(it);
      host_free(ptr);
      ptr = pinned_malloc(nbytes);
    }
  }
  pinnedSize[ptr] = nbytes;
  return ptr;
}


void FaceBuffer::freePinned(void *ptr)
{
  if (!pinnedSize.count(ptr)) {
    errorQuda("Attempt to free invalid pointer");
  }
  pinnedCache.insert(std::make_pair(pinnedSize[ptr], ptr));
  pinnedSize.erase(ptr);
}


void FaceBuffer::flushPinnedCache()
{
  std::multimap<size_t, void *>::iterator it;
  for (it = pinnedCache.begin(); it != pinnedCache.end(); it++) {
    void *ptr = it->second;
    host_free(ptr);
  }
  pinnedCache.clear();
}


void FaceBuffer::pack(cudaColorSpinorField &in, int parity, int dagger, cudaStream_t *stream_p)
{
  in.allocateGhostBuffer();   // allocate the ghost buffer if not yet allocated  
  stream = stream_p;
  in.packGhost((QudaParity)parity, dagger, &stream[Nstream-1]);
}

void FaceBuffer::pack(cudaColorSpinorField &in, int parity, int dagger, double a, double b, cudaStream_t *stream_p)
{
  in.allocateGhostBuffer();   // allocate the ghost buffer if not yet allocated  
  stream = stream_p;
  in.packTwistedGhost((QudaParity)parity, dagger, a, b, &stream[Nstream-1]);
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


// experimenting with callbacks for GPU -> MPI interaction.
// much slower though because callbacks are done on a background thread
//#define QUDA_CALLBACK

#ifdef QUDA_CALLBACK

struct commCallback_t {
  MsgHandle *mh_recv;
  MsgHandle *mh_send;
  void *ib_buffer;
  void *face_buffer;
  size_t bytes;
};

static commCallback_t commCB[2*QUDA_MAX_DIM];

void CUDART_CB commCallback(cudaStream_t stream, cudaError_t status, void *data) {
  const unsigned long long dir = (unsigned long long)data;

  comm_start(commCB[dir].mh_recv);
#ifndef GPU_DIRECT
  memcpy(commCB[dir].ib_buffer, commCB[dir].face_buffer, commCB[dir].bytes);
#endif
  comm_start(commCB[dir].mh_send);

}

void FaceBuffer::commsStart(int dir) {
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return;

  if (dir%2 == 0) { // sending backwards
    commCB[dir].mh_recv = mh_recv_fwd[dim]; 
    commCB[dir].mh_send = mh_send_back[dim];
    commCB[dir].ib_buffer = ib_my_back_face[dim];
    commCB[dir].face_buffer = my_back_face[dim];
    commCB[dir].bytes = nbytes[dim];
  } else { //sending forwards
    commCB[dir].mh_recv = mh_recv_back[dim]; 
    commCB[dir].mh_send = mh_send_fwd[dim];
    commCB[dir].ib_buffer = ib_my_fwd_face[dim];
    commCB[dir].face_buffer = my_fwd_face[dim];
    commCB[dir].bytes = nbytes[dim];
  }

  cudaStreamAddCallback(stream[dir], commCallback, (void*)dir, 0);
} 

#else // !defined(QUDA_CALLBACK)

void FaceBuffer::commsStart(int dir) {
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return;

  if (dir%2 == 0) { // sending backwards
    // Prepost receive
    comm_start(mh_recv_fwd[dim]);
#ifndef GPU_DIRECT
    memcpy(ib_my_back_face[dim], my_back_face[dim], nbytes[dim]);
#endif
    comm_start(mh_send_back[dim]);
  } else { //sending forwards
    // Prepost receive
    comm_start(mh_recv_back[dim]);
    // Begin forward send
#ifndef GPU_DIRECT
    memcpy(ib_my_fwd_face[dim], my_fwd_face[dim], nbytes[dim]);
#endif
    comm_start(mh_send_fwd[dim]);
  }
}

#endif // QUDA_CALLBACK


int FaceBuffer::commsQuery(int dir)
{
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return 0;

  if(dir%2==0) {
    if (comm_query(mh_recv_fwd[dim]) && comm_query(mh_send_back[dim])) {
#ifndef GPU_DIRECT
      memcpy(from_fwd_face[dim], ib_from_fwd_face[dim], nbytes[dim]);		
#endif
      return 1;
    }
  } else {
    if (comm_query(mh_recv_back[dim]) && comm_query(mh_send_fwd[dim])) {
#ifndef GPU_DIRECT
      memcpy(from_back_face[dim], ib_from_back_face[dim], nbytes[dim]);		
#endif
      return 1;
    }
  }

  return 0;
}


void FaceBuffer::scatter(cudaColorSpinorField &out, int dagger, int dir)
{
  int dim = dir/2;
  if(!commDimPartitioned(dim)) return;

  // both scattering occurances now go through the same stream
  if (dir%2==0) {// receive from forwards
    out.unpackGhost(from_fwd_face[dim], dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+recFwdStrmIdx*/]); // 0, 2, 4, 6
  } else { // receive from backwards
    out.unpackGhost(from_back_face[dim], dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+recBackStrmIdx*/]); // 1, 3, 5, 7
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

  MsgHandle *mh_send_fwd[4];
  MsgHandle *mh_from_back[4];
  MsgHandle *mh_from_fwd[4];
  MsgHandle *mh_send_back[4];

  for (int i=0; i<nDimComms; i++) {
    mh_send_fwd[i] = comm_declare_send_relative(spinor.fwdGhostFaceSendBuffer[i], i, +1, nbytes[i]);
    mh_send_back[i] = comm_declare_send_relative(spinor.backGhostFaceSendBuffer[i], i, -1, nbytes[i]);
    mh_from_fwd[i] = comm_declare_receive_relative(spinor.fwdGhostFaceBuffer[i], i, +1, nbytes[i]);
    mh_from_back[i] = comm_declare_receive_relative(spinor.backGhostFaceBuffer[i], i, -1, nbytes[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_start(mh_from_back[i]);
    comm_start(mh_from_fwd[i]);
    comm_start(mh_send_fwd[i]);
    comm_start(mh_send_back[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_wait(mh_send_fwd[i]);
    comm_wait(mh_send_back[i]);
    comm_wait(mh_from_back[i]);
    comm_wait(mh_from_fwd[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_free(mh_send_fwd[i]);
    comm_free(mh_send_back[i]);
    comm_free(mh_from_back[i]);
    comm_free(mh_from_fwd[i]);
  }
}


void FaceBuffer::exchangeLink(void** ghost_link, void** link_sendbuf, QudaFieldLocation location)
{
  MsgHandle *mh_from_back[4];
  MsgHandle *mh_send_fwd[4];

  void *send[4];
  void *receive[4];
  if (location == QUDA_CPU_FIELD_LOCATION) {
    for (int i=0; i<nDimComms; i++) {
      send[i] = link_sendbuf[i];
      receive[i] = ghost_link[i];
    }
  } else { // FIXME for CUDA field copy back to the CPU
    for (int i=0; i<nDimComms; i++) {
      int len = 2*nFace*faceVolumeCB[i]*Ninternal;
      send[i] = allocatePinned(len);
      receive[i] = allocatePinned(len);
      cudaMemcpy(send[i], link_sendbuf[i], len, cudaMemcpyDeviceToHost);
    }
  }

  for (int i=0; i<nDimComms; i++) {
    int len = 2*nFace*faceVolumeCB[i]*Ninternal;
    mh_send_fwd[i] = comm_declare_send_relative(send[i], i, +1, len*precision);
    mh_from_back[i] = comm_declare_receive_relative(receive[i], i, -1, len*precision);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_start(mh_send_fwd[i]);
    comm_start(mh_from_back[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_wait(mh_send_fwd[i]);
    comm_wait(mh_from_back[i]);
  }

  if (location == QUDA_CUDA_FIELD_LOCATION) {
    for (int i=0; i<nDimComms; i++) {
      int len = 2*nFace*faceVolumeCB[i]*Ninternal;
      cudaMemcpy(ghost_link[i], receive[i], len, cudaMemcpyHostToDevice);
      freePinned(send[i]);
      freePinned(receive[i]);
    }
  }

  for (int i=0; i<nDimComms; i++) {
    comm_free(mh_send_fwd[i]);
    comm_free(mh_from_back[i]);
  }
}


void reduceMaxDouble(double &max) { comm_allreduce_max(&max); }

void reduceDouble(double &sum) { if (globalReduce) comm_allreduce(&sum); }

void reduceDoubleArray(double *sum, const int len) 
{ if (globalReduce) comm_allreduce_array(sum, len); }

int commDim(int dir) { return comm_dim(dir); }

int commCoords(int dir) { return comm_coord(dir); }

int commDimPartitioned(int dir){ return comm_dim_partitioned(dir);}

void commDimPartitionedSet(int dir) { comm_dim_partitioned_set(dir);}

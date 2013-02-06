#include <quda_internal.h>
#include <face_quda.h>

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
    comm_send_fwd[i] = comm_declare_send_relative(ib_my_fwd_face[i], i, 1, nbytes[i]);
    comm_send_back[i] = comm_declare_send_relative(ib_my_back_face[i], i, -1, nbytes[i]);
    comm_recv_fwd[i] = comm_declare_receive_relative(ib_from_fwd_face[i], i, +1, nbytes[i]);
    comm_recv_back[i] = comm_declare_receive_relative(ib_from_back_face[i], i, -1, nbytes[i]);
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

    comm_free(comm_send_fwd[i]);
    comm_free(comm_send_back[i]);
    comm_free(comm_recv_fwd[i]);
    comm_free(comm_recv_back[i]);

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

    comm_recv_fwd[i] = NULL;
    comm_recv_back[i] = NULL;
    comm_send_fwd[i] = NULL;
    comm_send_back[i] = NULL;
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

    // Prepost receive
    comm_start(comm_recv_fwd[dim]);
#ifndef GPU_DIRECT
    memcpy(ib_my_back_face[dim], my_back_face[dim], nbytes[dim]);
#endif
    comm_start(comm_send_back[dim]);

  } else { //sending forwards
    
  // Prepost receive
    comm_start(comm_recv_back[dim]);
    // Begin forward send
#ifndef GPU_DIRECT
    memcpy(ib_my_fwd_face[dim], my_fwd_face[dim], nbytes[dim]);
#endif
    comm_start(comm_send_fwd[dim]);
  }

} 

int FaceBuffer::commsQuery(int dir) {
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return 0;

  if(dir%2==0) {
    if (comm_query(comm_recv_fwd[dim]) && comm_query(comm_send_back[dim])) {
#ifndef GPU_DIRECT
      memcpy(from_fwd_face[dim], ib_from_fwd_face[dim], nbytes[dim]);		
#endif
      return 1;
    }
  } else {
    if (comm_query(comm_recv_back[dim]) && comm_query(comm_send_fwd[dim])) {
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

  void *comm_send_fwd[4];
  void *comm_from_back[4];
  void *comm_from_fwd[4];
  void *comm_send_back[4];

  for (int i=0; i<nDimComms; i++) {
    comm_send_fwd[i] = comm_declare_send_relative(spinor.fwdGhostFaceSendBuffer[i], i, +1, nbytes[i]);
    comm_send_back[i] = comm_declare_send_relative(spinor.backGhostFaceSendBuffer[i], i, -1, nbytes[i]);
    comm_from_fwd[i] = comm_declare_receive_relative(spinor.fwdGhostFaceBuffer[i], i, +1, nbytes[i]);
    comm_from_back[i] = comm_declare_receive_relative(spinor.backGhostFaceBuffer[i], i, -1, nbytes[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_start(comm_from_back[i]);
    comm_start(comm_from_fwd[i]);
    comm_start(comm_send_fwd[i]);
    comm_start(comm_send_back[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_wait(comm_send_fwd[i]);
    comm_wait(comm_send_back[i]);
    comm_wait(comm_from_back[i]);
    comm_wait(comm_from_fwd[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_free(comm_send_fwd[i]);
    comm_free(comm_send_back[i]);
    comm_free(comm_from_back[i]);
    comm_free(comm_from_fwd[i]);
  }

}

void FaceBuffer::exchangeCpuLink(void** ghost_link, void** link_sendbuf) {

  void *comm_from_back[4];
  void *comm_send_fwd[4];

  for (int i=0; i<nDimComms; i++) {
    int len = 2*nFace*faceVolumeCB[i]*Ninternal;
    comm_send_fwd[i] = comm_declare_send_relative(link_sendbuf[i], i, +1, len*precision);
    comm_from_back[i] = comm_declare_receive_relative(ghost_link[i], i, -1, len*precision);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_start(comm_send_fwd[i]);
    comm_start(comm_from_back[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_wait(comm_send_fwd[i]);
    comm_wait(comm_from_back[i]);
  }

  for (int i=0; i<nDimComms; i++) {
    comm_free(comm_send_fwd[i]);
    comm_free(comm_from_back[i]);
  }

}


void reduceMaxDouble(double &max) { comm_allreduce_max(&max); }

void reduceDouble(double &sum) { if (globalReduce) comm_allreduce(&sum); }

void reduceDoubleArray(double *sum, const int len) 
{ if (globalReduce) comm_allreduce_array(sum, len); }

int commDim(int dir) { return comm_dim(dir); }

int commCoords(int dir) { return comm_coords(dir); }

int commDimPartitioned(int dir){ return comm_dim_partitioned(dir);}

void commDimPartitionedSet(int dir) { comm_dim_partitioned_set(dir);}



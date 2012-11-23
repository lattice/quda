#include <quda_internal.h>
#include <face_quda.h>

using namespace quda;

cudaStream_t *stream;

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
    comm_start(recv_handle_fwd[dim]);
#ifndef GPU_DIRECT
    memcpy(ib_my_back_face[dim], my_back_face[dim], nbytes[dim]);
#endif
    comm_start(send_handle_back[dim]);

  } else { //sending forwards
    
  // Prepost receive
    comm_start(recv_handle_back[dim]);
    // Begin forward send
#ifndef GPU_DIRECT
    memcpy(ib_my_fwd_face[dim], my_fwd_face[dim], nbytes[dim]);
#endif
    comm_start(send_handle_fwd[dim]);
  }

} 

int FaceBuffer::commsQuery(int dir) {
  int dim = dir / 2;
  if(!commDimPartitioned(dim)) return 0;

  if(dir%2==0) {
    if (comm_query(recv_handle_fwd[dim]) && comm_query(send_handle_back[dim])) {
#ifndef GPU_DIRECT
      memcpy(from_fwd_face[dim], ib_from_fwd_face[dim], nbytes[dim]);		
#endif
      return 1;
    }
  } else {
    if (comm_query(recv_handle_back[dim]) && comm_query(send_handle_fwd[dim])) {
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

  if (dir%2==0) {// receive from forwards
    out.unpackGhost(from_fwd_face[dim], dim, QUDA_FORWARDS, dagger, &stream[2*dim+recFwdStrmIdx]); // 0, 2, 4, 6
  } else { // receive from backwards
    out.unpackGhost(from_back_face[dim], dim, QUDA_BACKWARDS, dagger, &stream[2*dim+recBackStrmIdx]); // 1, 3, 5, 7
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

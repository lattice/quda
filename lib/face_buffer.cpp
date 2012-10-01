#include <quda_internal.h>
#include <face_quda.h>

using namespace quda;

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

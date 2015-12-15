#ifndef _GAUGE_FIX_OVR_EXTRA_H
#define _GAUGE_FIX_OVR_EXTRA_H


#ifdef MULTI_GPU
#include <map>
#include <quda_internal.h>


namespace quda {
  
void PreCalculateLatticeIndices(size_t faceVolume_[4], size_t faceVolumeCB_[4], int X[4], int border[4], \
  int &threads, int *borderpoints[2]);

}
#endif
#endif 

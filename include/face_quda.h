#ifndef _FACE_QUDA_H
#define _FACE_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif


  FaceBuffer allocateFaceBuffer(int Vs, int V, int stride, Precision precision);
  void freeFaceBuffer(FaceBuffer bufs);

  void gatherFromSpinor(FaceBuffer face, ParitySpinor in,  int dagger);
  void exchangeFaces(FaceBuffer bufs);
  void scatterToPads(ParitySpinor out, FaceBuffer face, int dagger);

#ifdef __cplusplus
}
#endif

#endif // _FACE_QUDA_H

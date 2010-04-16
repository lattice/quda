#ifndef _FACE_QUDA_H
#define _FACE_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  FaceBuffer allocateFaceBuffer(int Vs, int V, int stride, Precision precision);
  void freeFaceBuffer(FaceBuffer bufs);

  void gatherFromSpinor(FaceBuffer face, ParitySpinor in,  int dagger);

  void exchangeFacesStart(FaceBuffer face, ParitySpinor in, int dagger, cudaStream_t *stream);
  void exchangeFacesComms(FaceBuffer face);
  void exchangeFacesWait(FaceBuffer face, ParitySpinor out, int dagger);

  void scatterToPads(ParitySpinor out, FaceBuffer face, int dagger);

  void transferGaugeFaces(void *gauge, void *gauge_face, Precision precision,
			  int veclength, ReconstructType reconstruct, int V, int Vs);
 
#ifdef __cplusplus
}
#endif

#endif // _FACE_QUDA_H

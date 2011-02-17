#ifndef _FACE_QUDA_H
#define _FACE_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>

class FaceBuffer {

 private:  
  // set these both = 0 `for no overlap of qmp and cudamemcpyasync
  // sendBackIdx = 0, and sendFwdIdx = 1 for overlap
  int sendBackStrmIdx; // = 0;
  int sendFwdStrmIdx; // = 1;
  int recFwdStrmIdx; // = sendBackIdx;
  int recBackStrmIdx; // = sendFwdIdx;

  // Device memory buffer for coalescing the gathered messages
  void *gather_fwd_face;
  void *gather_back_face;

  void *my_fwd_face;
  void *my_back_face;
  void *from_back_face;
  void *from_fwd_face;
  int Vs; 
  int V;
  QudaPrecision precision;
  size_t nbytes;
#ifdef QMP_COMMS
  QMP_msgmem_t mm_send_fwd;
  QMP_msgmem_t mm_from_fwd;
  QMP_msgmem_t mm_send_back;
  QMP_msgmem_t mm_from_back;
  
  QMP_msghandle_t mh_send_fwd;
  QMP_msghandle_t mh_from_fwd;
  QMP_msghandle_t mh_send_back;
  QMP_msghandle_t mh_from_back;
#endif

  void gatherFromSpinor(void *in, void *inNorm, int stride, int dagger);
  void scatterToEndZone(cudaColorSpinorField &out, int dagger);

 public:
  FaceBuffer(int Vs, int V, QudaPrecision precision);
  FaceBuffer(const FaceBuffer &);
  virtual ~FaceBuffer();

  void exchangeFacesStart(cudaColorSpinorField &in, int dagger, cudaStream_t *stream);
  void exchangeFacesComms();
  void exchangeFacesWait(cudaColorSpinorField &out, int dagger);


};

void transferGaugeFaces(void *gauge, void *gauge_face, QudaPrecision precision,
			int veclength, ReconstructType reconstruct, int V, int Vs);

#endif // _FACE_QUDA_H

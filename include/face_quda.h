#ifndef _FACE_QUDA_H
#define _FACE_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>

#ifndef MPI_COMMS

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
  int Ninternal; // number of internal degrees of freedom (12 for spin projected Wilson, 6 for staggered)
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

 public:
  FaceBuffer(const int *X, const int nDim, const int Ninternal, QudaPrecision precision);
  FaceBuffer(const FaceBuffer &);
  virtual ~FaceBuffer();

  void exchangeFacesStart(cudaColorSpinorField &in, int parity,
			  int dagger, cudaStream_t *stream);
  void exchangeFacesComms();
  void exchangeFacesWait(cudaColorSpinorField &out, int dagger);
};

void transferGaugeFaces(void *gauge, void *gauge_face, QudaPrecision precision,
			int veclength, ReconstructType reconstruct, int V, int Vs);

#else // MPI comms

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7

class FaceBuffer {

 private:
  // set these both = 0 `for no overlap of qmp and cudamemcpyasync
  // sendBackIdx = 0, and sendFwdIdx = 1 for overlap
  int sendBackStrmIdx; // = 0;
  int sendFwdStrmIdx; // = 1;
  int recFwdStrmIdx; // = sendBackIdx;
  int recBackStrmIdx; // = sendFwdIdx;

  int Ninternal; // number of internal degrees of freedom (12 for spin projected Wilson, 6 for staggered)
  QudaPrecision precision;
  size_t nbytes;
  size_t nbytes_norm;

  int Volume;
  int VolumeCB;
  int faceVolume[QUDA_MAX_DIM];
  int faceVolumeCB[QUDA_MAX_DIM];
  int X[QUDA_MAX_DIM];
  int nDim;

  void* fwd_nbr_spinor_sendbuf = NULL;
  void* back_nbr_spinor_sendbuf = NULL;
  void* f_norm_sendbuf = NULL;
  void* b_norm_sendbuf = NULL;
  
  void* fwd_nbr_spinor = NULL;
  void* back_nbr_spinor = NULL;
  void* f_norm = NULL;
  void* b_norm = NULL;

  void* pagable_fwd_nbr_spinor_sendbuf = NULL;
  void* pagable_back_nbr_spinor_sendbuf = NULL;
  void* pagable_f_norm_sendbuf = NULL;
  void* pagable_b_norm_sendbuf = NULL;
  
  void* pagable_fwd_nbr_spinor = NULL;
  void* pagable_back_nbr_spinor = NULL;
  void* pagable_f_norm = NULL;
  void* pagable_b_norm = NULL;

  unsigned long recv_request1, recv_request2, recv_request3, recv_request4;
  unsigned long send_request1, send_request2, send_request3, send_request4;

  void setupDims(const int *X);
  
 public:
  FaceBuffer(const int *X, const int nDim, const int Ninternal, QudaPrecision precision)
  FaceBuffer(const FaceBuffer &);
  virtual ~FaceBuffer();

  void exchangeFacesStart(cudaColorSpinorField &in, int parity,
			  int dagger, cudaStream_t *stream);
  void exchangeFacesComms();
  void exchangeFacesWait(cudaColorSpinorField &out, int dagger);
};

#endif // MPI_COMMS

#endif // _FACE_QUDA_H

#ifndef _FACE_QUDA_H
#define _FACE_QUDA_H

#include <map>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <comm_quda.h>

namespace quda {
  class FaceBuffer {
    
  private:  
    
    // We cache pinned memory allocations so that Dirac objects can be created and
    // destroyed at will with minimal overhead.
    static std::multimap<size_t, void *> pinnedCache;
    
    // For convenience, we keep track of the sizes of active allocations (i.e., those not in the cache).
    static std::map<void *, size_t> pinnedSize;
    
    // set these both = 0 `for no overlap of qmp and cudamemcpyasync
    // sendBackIdx = 0, and sendFwdIdx = 1 for overlap
    int sendBackStrmIdx; // = 0;
    int sendFwdStrmIdx; // = 1;
    int recFwdStrmIdx; // = sendBackIdx;
    int recBackStrmIdx; // = sendFwdIdx;
    
    // CUDA pinned memory
    void *my_face;
    void *my_fwd_face[QUDA_MAX_DIM];
    void *my_back_face[QUDA_MAX_DIM];
    void *from_face;
    void *from_back_face[QUDA_MAX_DIM];
    void *from_fwd_face[QUDA_MAX_DIM];
    
    // IB pinned memory
    void* ib_my_fwd_face[QUDA_MAX_DIM];
    void* ib_my_back_face[QUDA_MAX_DIM];  
    void* ib_from_back_face[QUDA_MAX_DIM];
    void* ib_from_fwd_face[QUDA_MAX_DIM];
    
    // Message handles
    MsgHandle* mh_recv_fwd[QUDA_MAX_DIM];
    MsgHandle* mh_recv_back[QUDA_MAX_DIM];
    MsgHandle* mh_send_fwd[QUDA_MAX_DIM];
    MsgHandle* mh_send_back[QUDA_MAX_DIM];
   
    int Ninternal; // number of internal degrees of freedom (12 for spin projected Wilson, 6 for staggered)
    QudaPrecision precision;
    
    int Volume;
    int VolumeCB;
    int faceVolume[QUDA_MAX_DIM];
    int faceVolumeCB[QUDA_MAX_DIM];
    int X[QUDA_MAX_DIM];
    int nDim; // the actual number of space-time communications
    int nDimComms; // the number of dimensions in which we communicate
    int nFace;
    size_t nbytes[QUDA_MAX_DIM];
    
    void setupDims(const int *X, int Ls);
    
    void *allocatePinned(size_t nbytes);
    void freePinned(void *ptr);
    
  public:
    FaceBuffer(const int *X, const int nDim, const int Ninternal,
	       const int nFace, const QudaPrecision precision, const int Ls = 1);
    FaceBuffer(const FaceBuffer &);
    virtual ~FaceBuffer();

    /**
       Pack the cudaColorSpinorField's ghost zone into contiguous buffers
       @param in The cudaColorSpinorField whose ghost zone we are extracting
       @param dim The dimension in which we pack
       @param dir Whether we pack data to be sent forward or backwards or in both directions
       @param parity The parity of this field
       @param dagger Whether the operator for which we are applying is the Hermitian conjugate or not
       @param stream The stream to which the packing kernel will be issued
       @param zeroCopyPack Whether we are packing to a device buffer or packing straight to host memory
       @param a Twisted mass parameter (default=0)
       @param b Twisted mass parameter (Default=0)
     */
    void pack(quda::cudaColorSpinorField &in, quda::FullClover &clov, quda::FullClover &clovInv, 
	      int dim, int dir, int parity, int dagger, cudaStream_t *stream,
	      bool zeroCopyPack=false, double a=0);

    void pack(quda::cudaColorSpinorField &in, quda::FullClover &clov, quda::FullClover &clovInv, int dir, int parity, int dagger, 
              cudaStream_t *stream, bool zeroCopyPack=false, double a=0);

    void pack(quda::cudaColorSpinorField &in, quda::FullClover &clov, quda::FullClover &clovInv, int parity, int dagger, 
	      cudaStream_t *stream, bool zeroCopyPack=false, double a=0);

    void pack(quda::cudaColorSpinorField &in, int dim, int dir, int parity, int dagger, 
	      cudaStream_t *stream, bool zeroCopyPack=false, double a=0, double b=0);

    void pack(quda::cudaColorSpinorField &in, int dir, int parity, int dagger, 
              cudaStream_t *stream, bool zeroCopyPack=false, double a=0, double b=0);

    void pack(quda::cudaColorSpinorField &in, int parity, int dagger, 
	      cudaStream_t *stream, bool zeroCopyPack=false, double a=0, double b=0);


    void gather(quda::cudaColorSpinorField &in, int dagger, int dir, int streamIdx);

    void gather(quda::cudaColorSpinorField &in, int dagger, int dir);



    void sendStart(int dir);
    void recvStart(int dir);
    void commsStart(int dir);
    int  commsQuery(int dir);
    void scatter(quda::cudaColorSpinorField &out, int dagger, int dir);
  
    void scatter(quda::cudaColorSpinorField &out, int dagger, int dir, int streamIdx); 

 
    void exchangeCpuSpinor(quda::cpuColorSpinorField &in, int parity, int dagger);
    
    void exchangeLink(void** ghost_link, void** link_sendbuf, QudaFieldLocation location);
    
    static void flushPinnedCache();
  };
}
  
void reduceMaxDouble(double &);
void reduceDouble(double &);
void reduceDoubleArray(double *, const int len);
int commDim(int);
int commCoords(int);
int commDimPartitioned(int dir);
void commDimPartitionedSet(int dir);

#ifdef __cplusplus
  extern "C" {
#endif

    // implemented in face_gauge.cpp

    void exchange_cpu_sitelink(int* X,void** sitelink, void** ghost_sitelink,
			       void** ghost_sitelink_diag, 
			       QudaPrecision gPrecision, QudaGaugeParam* param, int optflag); 
    void exchange_cpu_sitelink_ex(int* X, int *R, void** sitelink, QudaGaugeFieldOrder cpu_order,
				  QudaPrecision gPrecision, int optflag, int geometry);
    void exchange_gpu_staple_start(int* X, void* _cudaStaple, int dir, int whichway,  cudaStream_t * stream);
    void exchange_gpu_staple_comms(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream);
    void exchange_gpu_staple_wait(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream);
    void exchange_gpu_staple(int* X, void* _cudaStaple, cudaStream_t * stream);
    void exchange_gpu_staple(int* X, void* _cudaStaple, cudaStream_t * stream);
    void exchange_cpu_staple(int* X, void* staple, void** ghost_staple,
			     QudaPrecision gPrecision);
    void exchange_llfat_init(QudaPrecision prec);
    void exchange_llfat_cleanup(void);

    extern bool globalReduce;

#ifdef __cplusplus
  }
#endif

#endif // _FACE_QUDA_H

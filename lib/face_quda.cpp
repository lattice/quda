#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

/*
  Multi-GPU TODOs
  - test qmp code
  - implement OpenMP version?
  - split face kernels
  - separate block sizes for body and face
  - single coalesced D->H copy - first pass implemented, enable with GATHER_COALESCE 
    (could be done better as a kernel - add to blas and autotune)
  - minimize pointer arithmetic in core code (need extra constant to replace SPINOR_HOP)
 */

using namespace std;

cudaStream_t *stream;

//#define GATHER_COALESCE

// Easy to switch between overlapping communication or not
#ifdef OVERLAP_COMMS
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpyAsync(dst, src, size, type, stream)
#else
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpy(dst, src, size, type)
#endif

FaceBuffer::FaceBuffer(int Vs, int V, QudaPrecision precision) :
  my_fwd_face(0), my_back_face(0), from_back_face(0), from_fwd_face(0), 
  Vs(Vs), V(V), precision(precision)
{

  // set these both = 0 `for no overlap of qmp and cudamemcpyasync
  // sendBackStrmIdx = 0, and sendFwdStrmIdx = 1 for overlap
  sendBackStrmIdx = 0;
  sendFwdStrmIdx = 1;
  recFwdStrmIdx = sendBackStrmIdx;
  recBackStrmIdx = sendFwdStrmIdx;
  
  // Buffers hold half spinors
  nbytes = Vs*12*precision;
  
  // add extra space for the norms for half precision
  if (precision == QUDA_HALF_PRECISION) nbytes += Vs*sizeof(float);
  
  unsigned int flag = cudaHostAllocDefault;
  cudaHostAlloc(&(my_fwd_face), nbytes, flag);
  if( !my_fwd_face ) errorQuda("Unable to allocate my_fwd_face with size %lu", nbytes);
  
  cudaHostAlloc(&(my_back_face), nbytes, flag);
  if( !my_back_face ) errorQuda("Unable to allocate my_back_face with size %lu", nbytes);
  
#ifdef GATHER_COALESCE
  cudaMalloc(&(gather_fwd_face), nbytes);
  cudaMalloc(&(gather_back_face), nbytes);
#endif

#ifdef QMP_COMMS
  cudaHostAlloc(&(from_fwd_face), nbytes, flag);
  if( !from_fwd_face ) errorQuda("Unable to allocate from_fwd_face with size %lu", nbytes);
  
  cudaHostAlloc(&(from_back_face), nbytes, flag);
  if( !from_back_face ) errorQuda("Unable to allocate from_back_face with size %lu", nbytes);   
#else
  from_fwd_face = my_back_face;
  from_back_face = my_fwd_face;
#endif  


#ifdef QMP_COMMS
  mm_send_fwd = QMP_declare_msgmem(my_fwd_face, nbytes);
  if( mm_send_fwd == NULL ) errorQuda("Unable to allocate send fwd message mem");
  
  mm_send_back = QMP_declare_msgmem(my_back_face, nbytes);
  if( mm_send_back == NULL ) errorQuda("Unable to allocate send back message mem");
  
  mm_from_fwd = QMP_declare_msgmem(from_fwd_face, nbytes);
  if( mm_from_fwd == NULL ) errorQuda("Unable to allocate recv from fwd message mem");
  
  mm_from_back = QMP_declare_msgmem(from_back_face, nbytes);
  if( mm_from_back == NULL ) errorQuda("Unable to allocate recv from back message mem");
  
  mh_send_fwd = QMP_declare_send_relative(mm_send_fwd, 3, +1, 0);
  if( mh_send_fwd == NULL ) errorQuda("Unable to allocate forward send");
  
  mh_send_back = QMP_declare_send_relative(mm_send_back, 3, -1, 0);
  if( mh_send_back == NULL ) errorQuda("Unable to allocate backward send");
  
  mh_from_fwd = QMP_declare_receive_relative(mm_from_fwd, 3, +1, 0);
  if( mh_from_fwd == NULL ) errorQuda("Unable to allocate forward recv");
  
  mh_from_back = QMP_declare_receive_relative(mm_from_back, 3, -1, 0);
  if( mh_from_back == NULL ) errorQuda("Unable to allocate backward recv");
#endif

}

FaceBuffer::FaceBuffer(const FaceBuffer &face) {
  errorQuda("FaceBuffer copy constructor not implemented");
}

FaceBuffer::~FaceBuffer()
{
  
#ifdef QMP_COMMS
  QMP_free_msghandle(mh_send_fwd);
  QMP_free_msghandle(mh_send_back);
  QMP_free_msghandle(mh_from_fwd);
  QMP_free_msghandle(mh_from_back);
  QMP_free_msgmem(mm_send_fwd);
  QMP_free_msgmem(mm_send_back);
  QMP_free_msgmem(mm_from_fwd);
  QMP_free_msgmem(mm_from_back);
  cudaFreeHost(from_fwd_face); // these are aliasing pointers for non-qmp case
  cudaFreeHost(from_back_face);// these are aliasing pointers for non-qmp case
#endif
  cudaFreeHost(my_fwd_face);
  cudaFreeHost(my_back_face);

#ifdef GATHER_COALESCE
  cudaFree(gather_fwd_face);
  cudaFree(gather_back_face);
#endif

  my_fwd_face=NULL;
  my_back_face=NULL;
  from_fwd_face=NULL;
  from_back_face=NULL;
}

//templating here is overkill
//template <typename Float>
void gather12Float(char* dest, char* spinor, int vecLen, int Vs, int V, int stride, 
		   bool upper, bool tIsZero, int strmIdx, char *buffer, QudaPrecision precision)
{
  int Npad = 12/vecLen;  // Number of Pad's in one half spinor... generalize based on floatN, precision, nspin, ncolor etc.
  int lower_spin_offset=vecLen*Npad*stride;	
  int t0_offset=0; // T=0 is the first VS block
  int Nt_minus1_offset = vecLen*(V - Vs); // N_t -1 = V-Vs & vecLen is from FloatN.
 
  int offset = 0;
  if (upper) offset = tIsZero ? t0_offset : Nt_minus1_offset;
  else offset = lower_spin_offset + (tIsZero ? t0_offset : Nt_minus1_offset);

#ifndef GATHER_COALESCE
  // QUDA Memcpy NPad's worth. 
  //  -- Dest will point to the right beginning PAD. 
  //  -- Each Pad has size vecLen*Vs Floats. 
  //  --  There is vecLen*Stride Floats from the start of one PAD to the start of the next
  for(int i=0; i < Npad; i++) {
    CUDAMEMCPY((void *)(dest + precision*vecLen*i*Vs), (void *)(spinor + precision*(offset + i*vecLen*stride)),
	       vecLen*Vs*precision, cudaMemcpyDeviceToHost, stream[strmIdx]);
  }
#else
  // Coalesce into a single gather
  // Much better to do this as a kernel to allow for some overlapping?
  for(int i=0; i < Npad; i++) {
    CUDAMEMCPY((void *)(buffer + precision*vecLen*i*Vs), (void *)(spinor + precision*(offset + i*vecLen*stride)),
	       vecLen*Vs*precision, cudaMemcpyDeviceToDevice, stream[strmIdx]);
  }
#endif

}

// like the above but for the norms required for QUDA_HALF_PRECISION
void gatherNorm(float* dest, float* norm, int Vs, int V, int stride, bool tIsZero, 
		int strmIdx, float *buffer)
{
  int t0_offset=0; // T=0 is the first VS block
  int Nt_minus1_offset=V - Vs; // N_t -1 = V-Vs.
  int offset = tIsZero ? t0_offset : Nt_minus1_offset;
  QudaPrecision precision = QUDA_HALF_PRECISION;

#ifndef GATHER_COALESCE
  CUDAMEMCPY((void *)(dest + 12*Vs*precision), (void *)(norm + offset),
	     Vs*sizeof(float), cudaMemcpyDeviceToHost, stream[strmIdx]); 
#else
  CUDAMEMCPY((void *)(buffer + 12*Vs*precision, (void *)(norm + offset),
		      Vs*sizeof(float), cudaMemcpyDeviceToDevice, stream[strmIdx]); 
#endif

}

void FaceBuffer::gatherFromSpinor(void *in, void *inNorm, int stride, int dagger)
{
  
  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward

  int vecLength = (precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;

  // !dagger: send lower components backwards, send upper components forwards
  // dagger: send upper components backwards, send lower components forwards
  bool upperBack = dagger ? true : false; // Fwd is !Back
    
  // gather for backwards send, tIsZero=true
  gather12Float((char*)(my_back_face), (char*)in, vecLength, Vs, V, stride, upperBack, true, 
		sendBackStrmIdx, (char*)gather_back_face, precision);

  if (precision == QUDA_HALF_PRECISION)
    gatherNorm((float*)((short*)my_back_face), (float*)inNorm, Vs, V, 
	       stride, true, sendBackStrmIdx, (float*)((short *)gather_back_face));
    
  // gather for forwards send, tIsZero=false
  gather12Float((char*)(my_fwd_face), (char*)in, vecLength, Vs, V, stride, !upperBack, false, 
		sendFwdStrmIdx, (char*)gather_fwd_face, precision);

  if (precision == QUDA_HALF_PRECISION)
    gatherNorm((float*)((short*)my_fwd_face), (float*)inNorm, Vs, V, 
	       stride, false, sendFwdStrmIdx, (float*)((short *)gather_fwd_face));
 
}

void FaceBuffer::exchangeFacesStart(void *in, void *inNorm, int stride, int dagger, 
				    cudaStream_t *stream_p)
{
  stream = stream_p;

#ifdef QMP_COMMS
  // Prepost all receives
  QMP_start(mh_from_fwd);
  QMP_start(mh_from_back);
#endif

  // Gather into face...
  gatherFromSpinor(in, inNorm, stride, dagger);

#ifdef GATHER_COALESCE  
  // Copy to host if we are coalescing into single face messages to reduce latency
  CUDAMEMCPY((void *)my_back_face, (void *)gather_back_face,  nbytes, cudaMemcpyDeviceToHost, stream[sendBackStrmIdx]); 
  CUDAMEMCPY((void *)my_fwd_face, (void *)gather_fwd_face,  nbytes, cudaMemcpyDeviceToHost, stream[sendFwdStrmIdx]); 
#endif
}

void FaceBuffer::exchangeFacesComms() {

#ifdef OVERLAP_COMMS
  // Need to wait for copy to finish before sending to neighbour
  cudaStreamSynchronize(stream[sendBackStrmIdx]);
#endif

#ifdef QMP_COMMS
  // Begin backward send
  QMP_start(mh_send_back);
#endif

#ifdef OVERLAP_COMMS
  // Need to wait for copy to finish before sending to neighbour
  cudaStreamSynchronize(stream[sendFwdStrmIdx]);
#endif

#ifdef QMP_COMMS
  // Begin forward send
  QMP_start(mh_send_fwd);
#endif

} 

// QUDA Memcpy 3 Pad's worth. 
//  -- Dest will point to the right beginning PAD. 
//  -- Each Pad has size vecLen * Vs Floats. 
//  --  There is 4Stride Floats from the
//           start of one PAD to the start of the next

void scatter12Float(char* spinor, char* buf, int vecLen, int Vs, int V, int stride, 
		    bool upper, int strmIdx, QudaPrecision precision)
{
  int Npad = 12/vecLen;
  int spinor_end = 2*Npad*vecLen*stride;
  int face_size = Npad*vecLen*Vs;
  int offset = spinor_end + (upper ? 0 : face_size);

  CUDAMEMCPY((void *)(spinor + precision*offset), (void *)(buf), face_size*precision, 
	     cudaMemcpyHostToDevice, stream[strmIdx]);
  
}

// half precision norm version of the above
template <typename Float>
void scatterNorm(Float* norm, Float* buf, int Vs, int V, int stride, bool upper, int strmIdx=0)
{
  int norm_end = stride;
  int face_size = Vs;
  // upper goes in the 1st norm zone, lower in the 2nd norm zone
  int offset = norm_end + (upper ? 0 : face_size); 

  CUDAMEMCPY((void *)(norm + offset), (void *)(buf), Vs*sizeof(Float), 
	     cudaMemcpyHostToDevice, stream[strmIdx]);  
}


// Finish backwards send and forwards receive
#ifdef QMP_COMMS				
#define QMP_finish_from_fwd					\
  QMP_wait(mh_send_back);					\
  QMP_wait(mh_from_fwd);					\

// Finish forwards send and backwards receive
#define QMP_finish_from_back					\
  QMP_wait(mh_send_fwd);					\
  QMP_wait(mh_from_back);					\

#else
#define QMP_finish_from_fwd					

#define QMP_finish_from_back					

#endif


void FaceBuffer::scatterToEndZone(void *out, void *outNorm, int stride, int dagger)
{
 
  int vecLength = (precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;

  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward
  // !dagger: receive lower components forwards, receive upper components backwards
  // dagger: receive upper components forwards, receive lower components backwards
  bool upperBack = dagger? false : true;

  QMP_finish_from_fwd;
  
  scatter12Float((char*)out, (char*)from_fwd_face, vecLength,
		 Vs, V, stride, !upperBack, recFwdStrmIdx, precision); // LOWER
  
  if (precision == QUDA_HALF_PRECISION)
    scatterNorm((float*)outNorm, (float*)((short*)from_fwd_face+12*Vs), 
		Vs, V, stride, !upperBack, recFwdStrmIdx);
  
  QMP_finish_from_back;
  
  scatter12Float((char*)out, (char*)from_back_face, vecLength,
		 Vs, V, stride, upperBack, recBackStrmIdx, precision);  // Upper
  
  if (precision == QUDA_HALF_PRECISION)
    scatterNorm((float*)outNorm, (float*)((short*)from_back_face+12*Vs),
		Vs, V, stride, upperBack, recBackStrmIdx); // Lower

}

void FaceBuffer::exchangeFacesWait(void *out, void *outNorm, int stride, int dagger)
{

  // replaced this memcopy with aliasing pointers - useful benchmarking
#ifndef QMP_COMMS
  // NO QMP -- do copies
  //CUDAMEMCPY(from_fwd_face, my_back_face, nbytes, cudaMemcpyHostToHost, stream[sendBackStrmIdx]); // 174 without these
  //CUDAMEMCPY(from_back_face, my_fwd_face, nbytes, cudaMemcpyHostToHost, stream[sendFwdStrmIdx]);
#endif // QMP_COMMS

  // Scatter faces.
  scatterToEndZone(out, outNorm, stride, dagger);
}

void transferGaugeFaces(void *gauge, void *gauge_face, QudaPrecision precision,
			 int veclength, ReconstructType reconstruct, int V, int Vs)
{
  int nblocks, ndim=4;
  size_t blocksize;//, nbytes;
  ptrdiff_t offset, stride;
  void *g;

  nblocks = ndim*reconstruct/veclength;
  blocksize = Vs*veclength*precision;
  offset = (V-Vs)*veclength*precision;
  stride = (V+Vs)*veclength*precision; // assume that pad = Vs
  // stride = V*veclength*precision;
  // nbytes = Vs*ndim*linksize*precision; /* for contiguous face buffer */

#ifdef QMP_COMMS

  QMP_msgmem_t mm_gauge_send_fwd;
  QMP_msgmem_t mm_gauge_from_back;
  QMP_msghandle_t mh_gauge_send_fwd;
  QMP_msghandle_t mh_gauge_from_back;

  g = (void *) ((char *) gauge + offset);
  mm_gauge_send_fwd = QMP_declare_strided_msgmem(g, blocksize, nblocks, stride);
  if (!mm_gauge_send_fwd) {
    errorQuda("Unable to allocate gauge message mem");
  }
  // mm_gauge_from_back = QMP_declare_msgmem(gauge_face, nbytes); /* for contiguous face buffer */
  mm_gauge_from_back = QMP_declare_strided_msgmem(gauge_face, blocksize, nblocks, stride);
  if (!mm_gauge_from_back) { 
    errorQuda("Unable to allocate gauge face message mem");
  }

  mh_gauge_send_fwd = QMP_declare_send_relative(mm_gauge_send_fwd, 3, +1, 0);
  if (!mh_gauge_send_fwd) {
    errorQuda("Unable to allocate gauge message handle");
  }
  mh_gauge_from_back = QMP_declare_receive_relative(mm_gauge_from_back, 3, -1, 0);
  if (!mh_gauge_from_back) {
    errorQuda("Unable to allocate gauge face message handle");
  }

  QMP_start(mh_gauge_send_fwd);
  QMP_start(mh_gauge_from_back);
  
  QMP_wait(mh_gauge_send_fwd);
  QMP_wait(mh_gauge_from_back);

  QMP_free_msghandle(mh_gauge_send_fwd);
  QMP_free_msghandle(mh_gauge_from_back);
  QMP_free_msgmem(mm_gauge_send_fwd);
  QMP_free_msgmem(mm_gauge_from_back);

#else 

  void *gf;

  for (int i=0; i<nblocks; i++) {
    g = (void *) ((char *) gauge + offset + i*stride);
    gf = (void *) ((char *) gauge_face + i*stride);
    // gf = (void *) ((char *) gauge_face + i*blocksize); /* for contiguous face buffer */

    // I don't think stream has been set here so can't use async copy
    /*#ifdef OVERLAP_COMMS
    cudaMemcpyAsync(gf, g, blocksize, cudaMemcpyHostToHost, *stream);
    #else*/
    cudaMemcpy(gf, g, blocksize, cudaMemcpyHostToHost);
    //#endif
  }

#endif // QMP_COMMS
}

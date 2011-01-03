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
  - implement as OpenMP?
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
  if( !my_fwd_face ) errorQuda("Unable to allocate my_fwd_face");
  
  cudaHostAlloc(&(my_back_face), nbytes, flag);
  if( !my_back_face ) errorQuda("Unable to allocate my_back_face");
  
#ifdef GATHER_COALESCE
  cudaMalloc(&(gather_fwd_face), nbytes);
  cudaMalloc(&(gather_back_face), nbytes);
#endif

#ifdef QMP_COMMS
  cudaHostAlloc(&(from_fwd_face), nbytes, flag);
  if( !from_fwd_face ) errorQuda("Unable to allocate from_fwd_face");
  
  cudaHostAlloc(&(from_back_face), nbytes, flag);
  if( !from_back_face ) errorQuda("Unable to allocate from_back_face");   
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

  /*  for (int i=0; i<2; i++)  {
    cudaEventCreate(start+i);
    cudaEventCreate(gather+i);
    cudaEventCreate(qmp+i);
    cudaEventCreate(stop+i);
    }*/

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
#else
  cudaFreeHost(my_fwd_face);
  cudaFreeHost(my_back_face);
  cudaFreeHost(from_fwd_face);
  cudaFreeHost(from_back_face);
#endif

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
template <int vecLen, typename Float>
void gather12Float(Float* dest, Float* spinor, int Vs, int V, int stride, bool upper, bool tIsZero,
		   int strmIdx, Float *buffer)
{
  int Npad = 12/vecLen;  // Number of Pad's in one half spinor... generalize based on floatN, precision, nspin, ncolor etc.
  int lower_spin_offset=vecLen*Npad*stride;	
  int t0_offset=0; // T=0 is the first VS block
  int Nt_minus1_offset = vecLen*(V - Vs); // N_t -1 = V-Vs & vecLen is from FloatN.
 
#ifndef GATHER_COALESCE
  // QUDA Memcpy NPad's worth. 
  //  -- Dest will point to the right beginning PAD. 
  //  -- Each Pad has size vecLen*Vs Floats. 
  //  --  There is vecLen*Stride Floats from the start of one PAD to the start of the next
  for(int i=0; i < Npad; i++) {
    if( upper ) { 
      if( tIsZero ) { 
        CUDAMEMCPY((void *)(dest + vecLen*i*Vs), (void *)(spinor + t0_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToHost, stream[strmIdx]);
      } else {
        CUDAMEMCPY((void *)(dest + vecLen*i*Vs), (void *)(spinor + Nt_minus1_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToHost, stream[strmIdx]);      
      }
    } else {
      if( tIsZero ) { 
        CUDAMEMCPY((void *)(dest + vecLen*i*Vs), (void *)(spinor + lower_spin_offset + t0_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToHost, stream[strmIdx]);      
      } else {
        CUDAMEMCPY((void *)(dest + vecLen*i*Vs), (void *)(spinor + lower_spin_offset + Nt_minus1_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToHost, stream[strmIdx]);      
      }
    }
  }
#else
  // Coalesce into a single gather
  // Much better to do this as a kernel to allow for some overlapping
  for(int i=0; i < Npad; i++) {
    if( upper ) { 
      if( tIsZero ) { 
        CUDAMEMCPY((void *)(buffer + vecLen*i*Vs), (void *)(spinor + t0_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToDevice, stream[strmIdx]);
      } else {
        CUDAMEMCPY((void *)(buffer + vecLen*i*Vs), (void *)(spinor + Nt_minus1_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToDevice, stream[strmIdx]);      
      }
    } else {
      if( tIsZero ) { 
        CUDAMEMCPY((void *)(buffer + vecLen*i*Vs), (void *)(spinor + lower_spin_offset + t0_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToDevice, stream[strmIdx]);      
      } else {
        CUDAMEMCPY((void *)(buffer + vecLen*i*Vs), (void *)(spinor + lower_spin_offset + Nt_minus1_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float), cudaMemcpyDeviceToDevice, stream[strmIdx]);      
      }
    }
  }
#endif

}

// like the above but for the norms required for QUDA_HALF_PRECISION
template <typename Float>
  void gatherNorm(Float* dest, Float* norm, int Vs, int V, int stride, bool tIsZero, 
		  int strmIdx, Float *buffer)
{
  int t0_offset=0; // T=0 is the first VS block
  int Nt_minus1_offset=V - Vs; // N_t -1 = V-Vs.
 
#ifndef GATHER_COALESCE
  if( tIsZero ) { 
    CUDAMEMCPY((void *)dest, (void *)(norm + t0_offset),
	       Vs*sizeof(Float), cudaMemcpyDeviceToHost, stream[strmIdx]); 
  } else {
    CUDAMEMCPY((void *)dest, (void *)(norm + Nt_minus1_offset),
	       Vs*sizeof(Float), cudaMemcpyDeviceToHost, stream[strmIdx]);      
  }
#else
  if( tIsZero ) { 
    CUDAMEMCPY((void *)buffer, (void *)(norm + t0_offset),
	       Vs*sizeof(Float), cudaMemcpyDeviceToDevice, stream[strmIdx]); 
  } else {
    CUDAMEMCPY((void *)buffer, (void *)(norm + Nt_minus1_offset),
	       Vs*sizeof(Float), cudaMemcpyDeviceToDevice, stream[strmIdx]);
  }
#endif

}

void FaceBuffer::gatherFromSpinor(void *in, void *inNorm, int stride, int dagger)
{
  
  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward
  if (!dagger) { 

    // Not HC: send lower components back, receive them from forward
    // lower components = buffers 4,5,6

    if (precision == QUDA_DOUBLE_PRECISION) {
      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<2>((double *)(my_back_face), (double *)in, 
		       Vs, V, stride, false, true, sendBackStrmIdx, (double *)gather_back_face);
    
      // Not Hermitian conjugate: send upper spinors forward/recv from back
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<2>((double *)(my_fwd_face), (double *)in,
		       Vs, V, stride, true, false, sendFwdStrmIdx, (double *)gather_fwd_face);
    
    } else if (precision == QUDA_SINGLE_PRECISION) {
      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<4>((float *)(my_back_face), (float *)in, 
		       Vs, V, stride, false, true, sendBackStrmIdx, (float *)gather_back_face);
    
      // Not Hermitian conjugate: send upper spinors forward/recv from back
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<4>((float *)(my_fwd_face), (float *)in,
		       Vs, V, stride, true, false, sendFwdStrmIdx, (float *)gather_fwd_face);
    
    } else {       
      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<4>((short *)(my_back_face), (short *)in, 
		       Vs, V, stride, false, true, sendBackStrmIdx, (short *)gather_back_face);
    
      gatherNorm((float*)((short*)my_back_face+12*Vs), 
		 (float*)inNorm, Vs, V, stride, true, sendBackStrmIdx, (float*)((short *)gather_back_face+12*Vs));

      // Not Hermitian conjugate: send upper spinors forward/recv from back
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<4>((short *)(my_fwd_face), (short *)in,
		       Vs, V, stride, true, false, sendFwdStrmIdx, (short *)gather_fwd_face);

      gatherNorm((float*)((short*)my_fwd_face+12*Vs), 
		 (float*)inNorm, Vs, V, stride, false, sendFwdStrmIdx, (float*)((short *)gather_fwd_face+12*Vs));
    }
 
  } else { 

    if (precision == QUDA_DOUBLE_PRECISION) {
      // HC: Send upper components back, receive them from front
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<2>((double *)(my_back_face), (double *)in,
		       Vs, V, stride, true, true, sendBackStrmIdx, (double *)gather_back_face);
    
      // HC: send lower components fwd, receive them from back
      // Lower Spins => upper = false, t=Nt-1      => tIsZero = true
      gather12Float<2>((double *)(my_fwd_face), (double *)in, 
		       Vs, V, stride, false, false, sendFwdStrmIdx, (double *)gather_fwd_face);    
    } else if (precision == QUDA_SINGLE_PRECISION) {
      // HC: Send upper components back, receive them from front
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<4>((float *)(my_back_face), (float *)in,
		       Vs, V, stride, true, true, sendBackStrmIdx, (float *)gather_back_face);
    
      // Lower Spins => upper = false, t=Nt-1      => tIsZero = true
      gather12Float<4>((float *)(my_fwd_face), (float *)in, 
		       Vs, V, stride, false, false, sendFwdStrmIdx, (float *)gather_fwd_face);
    
    } else {       
      // HC: Send upper components back, receive them from front
      //UpperSpins => upper = true, t=0 => tIsZero = true
      gather12Float<4>((short *)(my_back_face), (short *)in,
		       Vs, V, stride, true, true, sendBackStrmIdx, (short *)gather_back_face);

      gatherNorm((float*)((short*)my_back_face+12*Vs), 
		 (float*)inNorm, Vs, V, stride, true, sendBackStrmIdx, (float *)((short *)gather_back_face+12*Vs));

      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<4>((short *)(my_fwd_face), (short *)in, 
		       Vs, V, stride, false, false, sendFwdStrmIdx, (short *)gather_fwd_face);
    
      gatherNorm((float*)((short*)my_fwd_face+12*Vs), 
		 (float*)inNorm, Vs, V, stride, false, sendFwdStrmIdx, (float *)((short *)gather_fwd_face+12*Vs));

    }

  }
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

  // Create events for timing the gathering
  //cudaEventRecord(start[sendBackStrmIdx], stream[sendBackStrmIdx]);
  //cudaEventRecord(start[sendFwdStrmIdx], stream[sendFwdStrmIdx]);

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
  // record completion of gather
  //cudaEventRecord(gather[sendBackStrmIdx], stream[sendBackStrmIdx]);
#endif

#ifdef QMP_COMMS
  // Begin backward send
  QMP_start(mh_send_back);
#endif

#ifdef OVERLAP_COMMS
  // Need to wait for copy to finish before sending to neighbour
  cudaStreamSynchronize(stream[sendFwdStrmIdx]);
  // record completion of gather
  //cudaEventRecord(gather[sendFwdStrmIdx], stream[sendFwdStrmIdx]);
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

template <int vecLen, typename Float>
  void scatter12Float(Float* spinor, Float* buf, int Vs, int V, int stride, 
		      bool upper, int strmIdx=0)
{
  int Npad = 12/vecLen;
  int spinor_end = 2*Npad*vecLen*stride;
  int face_size = Npad*vecLen*Vs;
  
  if( upper ) { 
    CUDAMEMCPY((void *)(spinor + spinor_end), (void *)(buf), face_size*sizeof(Float), 
	       cudaMemcpyHostToDevice, stream[strmIdx]);
  } else {
    CUDAMEMCPY((void *)(spinor + spinor_end + face_size), (void *)(buf), face_size*sizeof(Float), 
	       cudaMemcpyHostToDevice, stream[strmIdx]);
  }
  
}

// half precision norm version of the above
template <typename Float>
  void scatterNorm(Float* norm, Float* buf, int Vs, int V, int stride, bool upper, int strmIdx=0)
{
  int norm_end = stride;
  int face_size = Vs;

  if (upper) { // upper goes in the first norm zone
    CUDAMEMCPY((void *)(norm + norm_end), (void *)(buf), Vs*sizeof(Float), 
	       cudaMemcpyHostToDevice, stream[strmIdx]);  
  } else { // lower goes in the second norm zone
    CUDAMEMCPY((void *)(norm + norm_end + face_size), (void *)(buf), Vs*sizeof(Float), 
	       cudaMemcpyHostToDevice, stream[strmIdx]);
  }

}


// Finish backwards send and forwards receive
#ifdef QMP_COMMS				
#define QMP_finish_from_fwd					\
  QMP_wait(mh_send_back);					\
  QMP_wait(mh_from_fwd);					\
  //cudaEventRecord(qmp[recFwdStrmIdx], stream[recFwdStrmIdx]);

// Finish forwards send and backwards receive
#define QMP_finish_from_back					\
  QMP_wait(mh_send_fwd);					\
  QMP_wait(mh_from_back);					\
  //cudaEventRecord(qmp[recBackStrmIdx], stream[recBackStrmIdx]);

#else
#define QMP_finish_from_fwd					\
  //cudaEventRecord(qmp[recFwdStrmIdx], stream[recFwdStrmIdx]);
#define QMP_finish_from_back					\
  //cudaEventRecord(qmp[recBackStrmIdx], stream[recBackStrmIdx]);

#endif


void FaceBuffer::scatterToEndZone(void *out, void *outNorm, int stride, int dagger)
{
 
  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward
  if (!dagger) { 

    if (precision == QUDA_DOUBLE_PRECISION) {
      // Not HC: send lower components back, receive them from forward
      // lower components = buffers 4,5,6
      QMP_finish_from_fwd;

      scatter12Float<2>((double *)out, (double *)from_fwd_face, 
			Vs, V, stride, false, recFwdStrmIdx); // LOWER
      
      QMP_finish_from_back;

      // Not H: Send upper components forward, receive them from back
      scatter12Float<2>((double *)out, (double *)from_back_face,
			Vs, V, stride, true, recBackStrmIdx);  // Upper
    } else if (precision == QUDA_SINGLE_PRECISION) {
      QMP_finish_from_fwd;

      // Not HC: send lower components back, receive them from forward
      // lower components = buffers 4,5,6
      scatter12Float<4>((float *)out, (float *)from_fwd_face, 
			Vs, V, stride, false, recFwdStrmIdx); // LOWER
      
      QMP_finish_from_back;

      // Not H: Send upper components forward, receive them from back
      scatter12Float<4>((float *)out, (float *)from_back_face,
			Vs, V, stride, true, recBackStrmIdx); // Upper
    } else {
      QMP_finish_from_fwd;

      // Not HC: send lower components back, receive them from forward
      // lower components = buffers 4,5,6
      scatter12Float<4>((short *)out, (short *)from_fwd_face, 
			Vs, V, stride, false, recFwdStrmIdx); // LOWER
      
      scatterNorm((float*)outNorm, (float*)((short*)from_fwd_face+12*Vs), 
		  Vs, V, stride, false, recFwdStrmIdx);

      QMP_finish_from_back;

      // Not H: Send upper components forward, receive them from back
      scatter12Float<4>((short *)out, (short *)from_back_face,
			Vs, V, stride, true, recBackStrmIdx); // Upper

      scatterNorm((float*)outNorm, (float*)((short*)from_back_face+12*Vs),
		  Vs, V, stride, true, recBackStrmIdx); // Lower

    }
    
  } else { 
    if (precision == QUDA_DOUBLE_PRECISION) {
      QMP_finish_from_fwd;

      // HC: send lower components fwd, receive them from back
      // upper components = buffers 1, 2,3, go forward (into my_fwd face)
      scatter12Float<2>((double *)out, (double *)from_fwd_face,
			Vs, V, stride, true, recFwdStrmIdx);       

      QMP_finish_from_back;

      // lower components = buffers 4,5,6
      scatter12Float<2>((double *)out, (double *)from_back_face,
			Vs, V, stride, false, recBackStrmIdx);       // Lower
      
    } else if (precision == QUDA_SINGLE_PRECISION) {
      QMP_finish_from_fwd;

      // HC: send lower components fwd, receive them from back
      // upper components = buffers 1, 2,3, go forward (into my_fwd face)
      scatter12Float<4>((float *)out, (float *)from_fwd_face,
			Vs, V, stride, true, recFwdStrmIdx );

      QMP_finish_from_back;

       // lower components = buffers 4,5,6
      scatter12Float<4>((float *)out, (float *)from_back_face,
			Vs, V, stride, false, recBackStrmIdx);       // Lower
      
   } else {
      QMP_finish_from_fwd;

      // HC: send lower components fwd, receive them from back
      // upper components = buffers 1, 2,3, go forward (into my_fwd face)
      scatter12Float<4>((short *)out, (short *)from_fwd_face,
			Vs, V, stride, true, recFwdStrmIdx );
 
      scatterNorm((float*)outNorm, (float*)((short*)from_fwd_face+12*Vs), 
		  Vs, V, stride, true, recFwdStrmIdx);

      QMP_finish_from_back;

      // lower components = buffers 4,5,6
      scatter12Float<4>((short *)out, (short *)from_back_face,
			Vs, V, stride, false, recBackStrmIdx);       // Lower
      
      scatterNorm((float*)outNorm, (float*)((short*)from_back_face+12*Vs),
		  Vs, V, stride, false, recBackStrmIdx);
      
    }

  }
}

void FaceBuffer::exchangeFacesWait(void *out, void *outNorm, int stride, int dagger)
{

  // removed this memcopy with aliasing pointers - useful benchmarking
#ifndef QMP_COMMS
  // NO QMP -- do copies
  //CUDAMEMCPY(from_fwd_face, my_back_face, nbytes, cudaMemcpyHostToHost, stream[sendBackStrmIdx]); // 174 without these
  //CUDAMEMCPY(from_back_face, my_fwd_face, nbytes, cudaMemcpyHostToHost, stream[sendFwdStrmIdx]);
#endif // QMP_COMMS

  // Scatter faces.
  scatterToEndZone(out, outNorm, stride, dagger);

  // record completion of scattering
  //cudaEventRecord(stop[recFwdStrmIdx], stream[recFwdStrmIdx]);
  //cudaEventRecord(stop[recBackStrmIdx], stream[recBackStrmIdx]);
}

void transferGaugeFaces(void *gauge, void *gauge_face, QudaPrecision precision,
			 int veclength, ReconstructType reconstruct, int V, int Vs)
{
  int nblocks, ndim=4;
  size_t linksize, blocksize;//, nbytes;
  ptrdiff_t offset, stride;
  void *g, *gf;

  switch (reconstruct) {
  case QUDA_RECONSTRUCT_NO:
    linksize = 18;
    break;
  case QUDA_RECONSTRUCT_12:
    linksize = 12;
    break;
  case QUDA_RECONSTRUCT_8:
    linksize = 8;
    break;
  default:
    errorQuda("Invalid reconstruct type");
  }
  nblocks = ndim*linksize/veclength;
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

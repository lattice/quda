#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

using namespace std;

FaceBuffer::FaceBuffer(int Vs, int V, int stride, QudaPrecision precision) :
  Vs(Vs), V(V), stride(stride), precision(precision)
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
  
  cudaMallocHost(&(my_fwd_face), nbytes);
  if( !my_fwd_face ) errorQuda("Unable to allocate my_fwd_face");
  
  cudaMallocHost(&(my_back_face), nbytes);
  if( !my_back_face ) errorQuda("Unable to allocate my_back_face");
  
  cudaMallocHost(&(from_fwd_face), nbytes);
  if( !from_fwd_face ) errorQuda("Unable to allocate from_fwd_face");
  
  cudaMallocHost(&(from_back_face), nbytes);
  if( !from_back_face ) errorQuda("Unable to allocate from_back_face");   
  
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

  my_fwd_face=NULL;
  my_back_face=NULL;
  from_fwd_face=NULL;
  from_back_face=NULL;
}

void FaceBuffer::exchangeFacesStart(cudaColorSpinorField &in, int dagger, 
				    cudaStream_t *stream_p)
{
  stream = stream_p;

#ifdef QMP_COMMS
  // Prepost all receives
  QMP_start(mh_from_fwd);
  QMP_start(mh_from_back);
#endif

  // Gather into face...
  gatherFromSpinor(in, dagger);
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

void FaceBuffer::exchangeFacesWait(cudaColorSpinorField &out, int dagger)
{

#ifndef QMP_COMMS

// NO QMP -- do copies
#ifdef OVERLAP_COMMS
  cudaMemcpyAsync(from_fwd_face, my_back_face, nbytes, 
		  cudaMemcpyHostToHost, stream[sendBackStrmIdx]);

  cudaMemcpyAsync(from_back_face, my_fwd_face, nbytes, 
		  cudaMemcpyHostToHost, stream[sendFwdStrmIdx]);
#else
  cudaMemcpy(from_fwd_face, my_back_face, nbytes, 
	     cudaMemcpyHostToHost);

  cudaMemcpy(from_back_face, my_fwd_face, nbytes, 
	     cudaMemcpyHostToHost);
#endif // OVERLAP_COMMS

#endif // QMP_COMMS

  // Scatter faces.
  scatterToPads(out, dagger);
}

//templating here is overkill
template <int vecLen, typename Float>
void gather12Float(Float* dest, Float* spinor, int Vs, int V, int stride, bool upper, bool tIsZero,
		   int strmIdx)
{
  int Npad = 12/vecLen;  // Number of Pad's in one half spinor... generalize based on floatN, precision, nspin, ncolor etc.
  int lower_spin_offset=vecLen*Npad*stride;	
  int t_zero_offset=0; // T=0 is the first VS block
  int Nt_minus_one_offset=vecLen*(V - Vs); // N_t -1 = V-Vs & vecLen is from FloatN.

 
  // QUDA Memcpy NPad's worth. 
  //  -- Dest will point to the right beginning PAD. 
  //  -- Each Pad has size vecLen*Vs Floats. 
  //  --  There is vecLen*Stride Floats from the
  //           start of one PAD to the start of the next
#ifdef OVERLAP_COMMS
  for(int i=0; i < Npad; i++) {
    if( upper ) { 
      if( tIsZero ) { 
        cudaMemcpyAsync((void *)(dest + vecLen*i*Vs), 
			(void *)(spinor + t_zero_offset + i*vecLen*stride),
			vecLen*Vs*sizeof(Float),
			cudaMemcpyDeviceToHost, stream[strmIdx]);
      } else {
        cudaMemcpyAsync((void *)(dest + vecLen*i*Vs), 
			(void *)(spinor + Nt_minus_one_offset + i*vecLen*stride),
			vecLen*Vs*sizeof(Float),
			cudaMemcpyDeviceToHost, stream[strmIdx]);      
      }
    } else {
      if( tIsZero ) { 
        cudaMemcpyAsync((void *)(dest + vecLen*i*Vs), 
			(void *)(spinor + lower_spin_offset + t_zero_offset + i*vecLen*stride),
			vecLen*Vs*sizeof(Float),
			cudaMemcpyDeviceToHost, stream[strmIdx]);      
      } else {
        cudaMemcpyAsync((void *)(dest + vecLen*i*Vs), 
			(void *)(spinor + lower_spin_offset + Nt_minus_one_offset + i*vecLen*stride),
			vecLen*Vs*sizeof(Float),
			cudaMemcpyDeviceToHost, stream[strmIdx]);      
      }
    }
  }
#else
  for(int i=0; i < Npad; i++) {
    if( upper ) { 
      if( tIsZero ) { 
        cudaMemcpy((void *)(dest + vecLen*i*Vs), 
                   (void *)(spinor + t_zero_offset + i*vecLen*stride),
	           vecLen*Vs*sizeof(Float),
		   cudaMemcpyDeviceToHost);
      } else {
        cudaMemcpy((void *)(dest + vecLen*i*Vs), 
                   (void *)(spinor + Nt_minus_one_offset + i*vecLen*stride),
	           vecLen*Vs*sizeof(Float),
		   cudaMemcpyDeviceToHost);
      }
    } else {
      if( tIsZero ) { 
        cudaMemcpy((void *)(dest + vecLen*i*Vs), 
		   (void *)(spinor + lower_spin_offset + t_zero_offset + i*vecLen*stride),
		   vecLen*Vs*sizeof(Float),
		   cudaMemcpyDeviceToHost);
      } else {
        cudaMemcpy((void *)(dest + vecLen*i*Vs), 
                   (void *)(spinor + lower_spin_offset + Nt_minus_one_offset + i*vecLen*stride),
	           vecLen*Vs*sizeof(Float),
		   cudaMemcpyDeviceToHost);
      }
    }
  }
#endif

}

// like the above but for the norms required for QUDA_HALF_PRECISION
template <typename Float>
  void gatherNorm(Float* dest, Float* norm, int Vs, int V, int stride, bool tIsZero, int strmIdx)
{
  int t_zero_offset=0; // T=0 is the first VS block
  int Nt_minus_one_offset=V - Vs; // N_t -1 = V-Vs.
 
#ifdef OVERLAP_COMMS
  if( tIsZero ) { 
    cudaMemcpyAsync((void *)dest, 
		    (void *)(norm + t_zero_offset),
		    Vs*sizeof(Float),
		    cudaMemcpyDeviceToHost, stream[strmIdx]); 
  } else {
    cudaMemcpyAsync((void *)dest, 
		    (void *)(norm + Nt_minus_one_offset),
		    Vs*sizeof(Float),
		    cudaMemcpyDeviceToHost, stream[strmIdx]);      
  }
#else
  if( tIsZero ) { 
    cudaMemcpy((void *)dest, 
	       (void *)(norm + t_zero_offset),
	       Vs*sizeof(Float),
	       cudaMemcpyDeviceToHost);      
  } else {
    cudaMemcpy((void *)dest, 
	       (void *)(norm + Nt_minus_one_offset),
	       Vs*sizeof(Float),
	       cudaMemcpyDeviceToHost);      
  }
#endif

}

void FaceBuffer::gatherFromSpinor(cudaColorSpinorField &in, int dagger)
{
  
  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward
  if (!dagger) { 

    // Not HC: send lower components back, receive them from forward
    // lower components = buffers 4,5,6

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<2>((double *)(my_back_face), (double *)in.v, 
		       Vs, V, stride, false, true, sendBackStrmIdx);
    
      // Not Hermitian conjugate: send upper spinors forward/recv from back
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<2>((double *)(my_fwd_face), (double *)in.v,
		       Vs, V, stride, true, false, sendFwdStrmIdx);
    
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<4>((float *)(my_back_face), (float *)in.v, 
		       Vs, V, stride, false, true, sendBackStrmIdx);
    
      // Not Hermitian conjugate: send upper spinors forward/recv from back
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<4>((float *)(my_fwd_face), (float *)in.v,
		       Vs, V, stride, true, false, sendFwdStrmIdx);
    
    } else {       
      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<4>((short *)(my_back_face), (short *)in.v, 
		       Vs, V, stride, false, true, sendBackStrmIdx);
    
      gatherNorm((float*)((short*)my_back_face+12*Vs), 
		 (float*)in.norm, Vs, V, stride, true, sendBackStrmIdx);

      // Not Hermitian conjugate: send upper spinors forward/recv from back
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<4>((short *)(my_fwd_face), (short *)in.v,
		       Vs, V, stride, true, false, sendFwdStrmIdx);

      gatherNorm((float*)((short*)my_fwd_face+12*Vs), 
		 (float*)in.norm, Vs, V, stride, false, sendFwdStrmIdx);
    }
 
  } else { 

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      // HC: Send upper components back, receive them from front
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<2>((double *)(my_back_face), (double *)in.v,
		       Vs, V, stride, true, true, sendBackStrmIdx);
    
      // HC: send lower components fwd, receive them from back
      // Lower Spins => upper = false, t=Nt-1      => tIsZero = true
      gather12Float<2>((double *)(my_fwd_face), (double *)in.v, 
		       Vs, V, stride, false, false, sendFwdStrmIdx);    
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      // HC: Send upper components back, receive them from front
      // upper spins => upper = true,t=Nt-1 => tIsZero=false
      gather12Float<4>((float *)(my_back_face), (float *)in.v,
		       Vs, V, stride, true, true, sendBackStrmIdx);
    
      // Lower Spins => upper = false, t=Nt-1      => tIsZero = true
      gather12Float<4>((float *)(my_fwd_face), (float *)in.v, 
		       Vs, V, stride, false, false, sendFwdStrmIdx);
    
    } else {       
      // HC: Send upper components back, receive them from front
      //UpperSpins => upper = true, t=0 => tIsZero = true
      gather12Float<4>((short *)(my_back_face), (short *)in.v,
		       Vs, V, stride, true, true, sendBackStrmIdx);

      gatherNorm((float*)((short*)my_back_face+12*Vs), 
		 (float*)in.norm, Vs, V, stride, true, sendBackStrmIdx);

      // lower_spins => upper = false, t=0, so tIsZero=true 
      gather12Float<4>((short *)(my_fwd_face), (short *)in.v, 
		       Vs, V, stride, false, false, sendFwdStrmIdx);
    
      gatherNorm((float*)((short*)my_fwd_face+12*Vs), 
		 (float*)in.norm, Vs, V, stride, false, sendFwdStrmIdx);

    }

  }
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
  
#ifdef OVERLAP_COMMS
  if( upper ) { 
    cudaMemcpyAsync((void *)(spinor + spinor_end), (void *)(buf), face_size*sizeof(Float), 
		    cudaMemcpyHostToDevice, stream[strmIdx]);
  } else {
    cudaMemcpyAsync((void *)(spinor + spinor_end + face_size), (void *)(buf), face_size*sizeof(Float), 
		    cudaMemcpyHostToDevice, stream[strmIdx]);
#else
  if( upper ) { 
    cudaMemcpy((void *)(spinor + spinor_end), (void *)(buf), face_size*sizeof(Float), cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy((void *)(spinor + spinor_end + face_size), (void *)(buf), face_size*sizeof(Float), cudaMemcpyHostToDevice);

#endif
  }
  
}

// half precision norm version of the above
template <typename Float>
  void scatterNorm(Float* norm, Float* buf, int Vs, int V, int stride, bool upper, int strmIdx=0)
{
  int norm_end = stride;
  int face_size = Vs;

#ifdef OVERLAP_COMMS
  if (upper) { // upper goes in the first norm zone
    cudaMemcpyAsync((void *)(norm + norm_end), (void *)(buf), Vs*sizeof(Float), 
		    cudaMemcpyHostToDevice, stream[strmIdx]);  
  } else { // lower goes in the second norm zone
    cudaMemcpyAsync((void *)(norm + norm_end + face_size), (void *)(buf), Vs*sizeof(Float), 
		    cudaMemcpyHostToDevice, stream[strmIdx]);
  }
#else
  if (upper) { // upper goes in the first norm zone
    cudaMemcpy((void *)(norm + norm_end), (void *)(buf), Vs*sizeof(Float), cudaMemcpyHostToDevice);
  } else { // lower goes in the second norm zone
    cudaMemcpy((void *)(norm + norm_end + face_size), (void *)(buf), Vs*sizeof(Float), cudaMemcpyHostToDevice);
  }
#endif
}


// Finish backwards send and forwards receive
#ifdef QMP_COMMS				
#define QMP_finish_from_fwd			\
  QMP_wait(mh_send_back);			\
  QMP_wait(mh_from_fwd);			

// Finish forwards send and backwards receive
#define QMP_finish_from_back			\
  QMP_wait(mh_send_fwd);			\
  QMP_wait(mh_from_back);			
#else
#define QMP_finish_from_fwd
#define QMP_finish_from_back
#endif


 void FaceBuffer::scatterToPads(cudaColorSpinorField &out, int dagger)
{
 
  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward
  if (!dagger) { 

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      // Not HC: send lower components back, receive them from forward
      // lower components = buffers 4,5,6
      QMP_finish_from_fwd;

      scatter12Float<2>((double *)out.v, (double *)from_fwd_face, 
			Vs, V, stride, false, recFwdStrmIdx); // LOWER
      
      QMP_finish_from_back;

      // Not H: Send upper components forward, receive them from back
      scatter12Float<2>((double *)out.v, (double *)from_back_face,
			Vs, V, stride, true, recBackStrmIdx);  // Upper
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      QMP_finish_from_fwd;

      // Not HC: send lower components back, receive them from forward
      // lower components = buffers 4,5,6
      scatter12Float<4>((float *)out.v, (float *)from_fwd_face, 
			Vs, V, stride, false, recFwdStrmIdx); // LOWER
      
      QMP_finish_from_back;

      // Not H: Send upper components forward, receive them from back
      scatter12Float<4>((float *)out.v, (float *)from_back_face,
			Vs, V, stride, true, recBackStrmIdx); // Upper
    } else {
      QMP_finish_from_fwd;

      // Not HC: send lower components back, receive them from forward
      // lower components = buffers 4,5,6
      scatter12Float<4>((short *)out.v, (short *)from_fwd_face, 
			Vs, V, stride, false, recFwdStrmIdx); // LOWER
      
      scatterNorm((float*)out.norm, (float*)((short*)from_fwd_face+12*Vs), 
		  Vs, V, stride, false, recFwdStrmIdx);

      QMP_finish_from_back;

      // Not H: Send upper components forward, receive them from back
      scatter12Float<4>((short *)out.v, (short *)from_back_face,
			Vs, V, stride, true, recBackStrmIdx); // Upper

      scatterNorm((float*)out.norm, (float*)((short*)from_back_face+12*Vs),
		  Vs, V, stride, true, recBackStrmIdx); // Lower

    }
    
  } else { 
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      QMP_finish_from_fwd;

      // HC: send lower components fwd, receive them from back
      // upper components = buffers 1, 2,3, go forward (into my_fwd face)
      scatter12Float<2>((double *)out.v, (double *)from_fwd_face,
			Vs, V, stride, true, recFwdStrmIdx);       

      QMP_finish_from_back;

      // lower components = buffers 4,5,6
      scatter12Float<2>((double *)out.v, (double *)from_back_face,
			Vs, V, stride, false, recBackStrmIdx);       // Lower
      
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      QMP_finish_from_fwd;

      // HC: send lower components fwd, receive them from back
      // upper components = buffers 1, 2,3, go forward (into my_fwd face)
      scatter12Float<4>((float *)out.v, (float *)from_fwd_face,
			Vs, V, stride, true, recFwdStrmIdx );

      QMP_finish_from_back;

       // lower components = buffers 4,5,6
      scatter12Float<4>((float *)out.v, (float *)from_back_face,
			Vs, V, stride, false, recBackStrmIdx);       // Lower
      
   } else {
      QMP_finish_from_fwd;

      // HC: send lower components fwd, receive them from back
      // upper components = buffers 1, 2,3, go forward (into my_fwd face)
      scatter12Float<4>((short *)out.v, (short *)from_fwd_face,
			Vs, V, stride, true, recFwdStrmIdx );
 
      scatterNorm((float*)out.norm, (float*)((short*)from_fwd_face+12*Vs), 
		  Vs, V, stride, true, recFwdStrmIdx);

      QMP_finish_from_back;

      // lower components = buffers 4,5,6
      scatter12Float<4>((short *)out.v, (short *)from_back_face,
			Vs, V, stride, false, recBackStrmIdx);       // Lower
      
      scatterNorm((float*)out.norm, (float*)((short*)from_back_face+12*Vs),
		  Vs, V, stride, false, recBackStrmIdx);

    }

  }
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

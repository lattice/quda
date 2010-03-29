#include <quda_internal.h>
#include <face_quda.h>
#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <string.h>

#define QMP_COMMS
#ifdef QMP_COMMS
#include <qmp.h>
QMP_msgmem_t mm_send_fwd;
QMP_msgmem_t mm_from_fwd;
QMP_msgmem_t mm_send_back;
QMP_msgmem_t mm_from_back;
QMP_msghandle_t mh_send_fwd;
QMP_msghandle_t mh_send_back;
QMP_msghandle_t mh_from_fwd;
QMP_msghandle_t mh_from_back;
#endif

using namespace std;

FaceBuffer allocateFaceBuffer(int Vs, int V, int stride, Precision precision)
{
   FaceBuffer ret;
   ret.Vs = Vs;
   ret.V = V;
   ret.stride=stride;
   ret.precision=precision;
  
   size_t size=0;
   switch(precision) { 
   case QUDA_HALF_PRECISION:
     size=sizeof(short);
     break;
   case QUDA_SINGLE_PRECISION:
     size=sizeof(float);
     break;
   case QUDA_DOUBLE_PRECISION:
     size=sizeof(double);
     break;
   default:
     errorQuda("Error unknown precision");
     break;
   };

   // Buffers hold half spinors
   ret.nbytes = ret.Vs*12*size;

#ifndef __DEVICE_EMULATION__
   cudaMallocHost(&(ret.my_fwd_face), ret.nbytes);
#else
    ret.my_fwd_face = malloc(ret.nbytes);
#endif

    if( !ret.my_fwd_face ) { 
      errorQuda("Unable to allocate my_fwd_face");
    }

#ifndef __DEVICE_EMULATION__
    cudaMallocHost(&(ret.my_back_face), ret.nbytes);
#else
    ret.my_back_face = malloc(ret.nbytes);
#endif

    if( !ret.my_back_face ) { 
      errorQuda("Unable to allocate my_back_face");
    }

#ifndef __DEVICE_EMULATION__
    cudaMallocHost(&(ret.from_fwd_face), ret.nbytes);
#else
    ret.from_fwd_face = malloc(ret.nbytes);
#endif

    if( !ret.from_fwd_face ) { 
      errorQuda("Unable to allocate from_fwd_face");
    }


#ifndef __DEVICE_EMULATION__
    cudaMallocHost(&(ret.from_back_face), ret.nbytes);
#else
    ret.from_back_face = malloc(ret.nbytes);
#endif


    if( !ret.from_back_face ) { 
      errorQuda("Unable to allocate from_back_face");
    }


#ifdef QMP_COMMS
    mm_send_fwd = QMP_declare_msgmem(ret.my_fwd_face, ret.nbytes);
    if( mm_send_fwd == NULL ) { 
      errorQuda("Unable to allocate send fwd message mem");
    }
    mm_send_back = QMP_declare_msgmem(ret.my_back_face, ret.nbytes);
    if( mm_send_back == NULL ) { 
      errorQuda("Unable to allocate send back message mem");
    }


    mm_from_fwd = QMP_declare_msgmem(ret.from_fwd_face, ret.nbytes);
    if( mm_from_fwd == NULL ) { 
      errorQuda("Unable to allocate recv from fwd message mem");
    }

    mm_from_back = QMP_declare_msgmem(ret.from_back_face, ret.nbytes);
    if( mm_from_back == NULL ) { 
      errorQuda("Unable to allocate recv from back message mem");
    }

    mh_send_fwd = QMP_declare_send_relative(mm_send_fwd,
					    3,
					    +1, 
					    0);
    if( mh_send_fwd == NULL ) {
      errorQuda("Unable to allocate forward send");
    }

    mh_send_back = QMP_declare_send_relative(mm_send_back, 
					     3,
					     -1,
					     0);
    if( mh_send_back == NULL ) {
      errorQuda("Unable to allocate backward send");
    }
    
    
    mh_from_fwd = QMP_declare_receive_relative(mm_from_fwd,
					    3,
					    +1,
					    0);
    if( mh_from_fwd == NULL ) {
      errorQuda("Unable to allocate forward recv");
    }
    
    mh_from_back = QMP_declare_receive_relative(mm_from_back, 
					     3,
					     -1,
					     0);
    if( mh_from_back == NULL ) {
      errorQuda("Unable to allocate backward recv");
    }

#endif

    return ret;
}


void freeFaceBuffer(FaceBuffer f)
{

#ifdef QMP_COMMS
  QMP_free_msghandle(mh_send_fwd);
  QMP_free_msghandle(mh_send_back);
  QMP_free_msghandle(mh_from_fwd);
  QMP_free_msghandle(mh_from_back);
  QMP_free_msgmem(mm_send_fwd);
  QMP_free_msgmem(mm_send_back);
  QMP_free_msgmem(mm_from_fwd);
  QMP_free_msgmem(mm_send_back);
#else

#ifndef __DEVICE_EMULATION__
  cudaFreeHost(f.my_fwd_face);
  cudaFreeHost(f.my_back_face);
  cudaFreeHost(f.from_fwd_face);
  cudaFreeHost(f.from_back_face);
#else 
  free(f.my_fwd_face);
  free(f.my_back_face);
  free(f.from_fwd_face);
  free(f.from_back_face);
#endif
#endif
  f.my_fwd_face=NULL;
  f.my_back_face=NULL;
  f.from_fwd_face=NULL;
  f.from_back_face=NULL;

}

// This would need to change for going truly parallel..
// Right now, its just a question of copying faces. My front face
// is what I send forward so it will be my 'from-back' face
// and my back face is what I send backward -- will be my from front face
void exchangeFaces(FaceBuffer bufs)
{

#if 1


  QMP_start(mh_from_fwd);
  QMP_start(mh_from_back);

  QMP_start(mh_send_back);
  QMP_start(mh_send_fwd);

  QMP_wait(mh_send_back);
  QMP_wait(mh_send_fwd);

  QMP_wait(mh_from_fwd);
  QMP_wait(mh_from_back);

  
#else 

#ifndef __DEVICE_EMULATION__
  cudaMemcpy(bufs.from_fwd_face, bufs.my_back_face, bufs.nbytes, 
	     cudaMemcpyHostToHost);

  cudaMemcpy(bufs.from_back_face, bufs.my_fwd_face, bufs.nbytes, 
	     cudaMemcpyHostToHost);

#else
  memcpy(bufs.from_fwd_face, bufs.my_back_face, bufs.nbytes);
  memcpy(bufs.from_back_face, bufs.my_fwd_face, bufs.nbytes);
#endif
#endif
   
}


void gather12Float(float* dest, float* spinor, int Vs, int stride)
{

  // QUDA Memcpy 3 Pad's worth. 
  //  -- Dest will point to the right beginning PAD. 
  //  -- Each Pad has size 4Vs floars. 
  //  --  There is 4Stride floats from the
  //           start of one PAD to the start of the next
  for(int i=0; i < 3; i++) {
    
    cudaMemcpy((void *)(dest+4*i*Vs), (void *)(spinor + 4*i*stride), 4*Vs*sizeof(float), cudaMemcpyDeviceToHost );

  }
}

  // QUDA Memcpy 3 Pad's worth. 
  //  -- Dest will point to the right beginning PAD. 
  //  -- Each Pad has size 4Vs floars. 
  //  --  There is 4Stride floats from the
  //           start of one PAD to the start of the next

void scatter12Float(float* spinor, float* buf, int Vs, int V, int stride, bool upper)
{
  int veclen=4;
  int Npad = 12/veclen;

  for(int i=0; i < Npad; i++) {
    if( upper ) { 
      cudaMemcpy((void *)(spinor+veclen*(V+i*stride)),(void *)(buf+veclen*i*Vs), veclen*Vs*sizeof(float), cudaMemcpyHostToDevice );
    }
    else {
      cudaMemcpy((void *)(spinor+veclen*(V+(i+Npad)*stride) ), (void *)(buf+veclen*i*Vs), veclen*Vs*sizeof(float), cudaMemcpyHostToDevice);  
    }
  }
}

void gatherFromSpinor(FaceBuffer face, ParitySpinor in, int dagger)
{
  int upper_spin_offset=0;  // Upper Spin is from 0 to 12stride
  int lower_spin_offset=12*(face.stride); // Lower spin is from 12 stride to 24 stride

  int t_zero_offset=0; // T=0 is the first VS block
  int Nt_minus_one_offset=4*(face.V - face.Vs); // N_t -1 = V-Vs & 4 is from float4.

  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward
  if (!dagger) { 

    // Not HC: send lower components back, receive them from forward
    // lower components = buffers 4,5,6


      gather12Float((float *)(face.my_back_face), 
		    (float *)in.spinor + lower_spin_offset + t_zero_offset, // lower components
		    face.Vs,
		    face.stride);

    // Not Hermitian conjugate: send upper spinors forward/recv from back

      gather12Float((float *)(face.my_fwd_face),
		    (float *)in.spinor + upper_spin_offset + Nt_minus_one_offset,
		    face.Vs,               // Size of face 
		    face.stride);        // Stride

 

  }
  else { 

    // HC: send lower components fwd, receive them from back
      // lower components = buffers 4,5,6
      gather12Float((float *)face.my_fwd_face, 
		    (float *)in.spinor + lower_spin_offset+Nt_minus_one_offset, // lower components, Nt-1
		    face.Vs,
		    face.stride);

      // HC: Send upper components back, receive them from front
      gather12Float((float *)face.my_back_face,
		    (float *)in.spinor + upper_spin_offset+ t_zero_offset,  // upper 3 components,t = 0
		    face.Vs,               // Size of face 
		    face.stride);        // Stride

 

  }
}

void scatterToPads(ParitySpinor out, FaceBuffer face, int dagger)
{
 

  // I need to gather the faces with opposite Checkerboard
  // Depending on whether I do dagger or not I want top 2 components
  // from forward, and bottom 2 components from backward
  if (!dagger) { 

    // Not HC: send lower components back, receive them from forward
      // lower components = buffers 4,5,6
      scatter12Float((float *)out.spinor, // lower components
		     (float *)face.from_fwd_face, 
		     face.Vs,
                     face.V,
		     face.stride, 
	             false); // LOWER

      // Not H: Send upper fomponents forward, receiv them from back
      scatter12Float((float *)out.spinor,
		     (float *)face.from_back_face,
		     face.Vs,               // Size of face
                     face.V, 
		     face.stride, // Stride
                     true);        // Upper

 

  }
  else { 
    // HC: send lower components fwd, receive them from back
      // lower components = buffers 4,5,6
    scatter12Float((float *)out.spinor, // lower components, Nt-1
		   (float *)face.from_back_face, 
		   face.Vs,
                   face.V,
		   face.stride,
                   false);       // Lower

      // upper components = buffers 1, 2,3, go forward (into my_fwd face)
      scatter12Float((float *)out.spinor,  // upper 3 components,t = 0
		     (float *)face.from_fwd_face,
		    
		    face.Vs,               // Size of face
                    face.V, 
		    face.stride,
	            true );       

 

  }
}

void blankSpinorPads(ParitySpinor out)
{
   int i;
   for(int i=0; i < 6; i++) { 
	cudaMemset( (void *)( (float *)(out.spinor)+4*out.volume + 4*i*out.stride ),0x0,4*out.pad*sizeof(float));
   }
}


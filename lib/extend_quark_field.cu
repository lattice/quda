#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>

#include <color_spinor_field.h>

#ifdef GPU_STAGGERED_DIRAC
#if (__COMPUTE_CAPACILITY__ >= 200)
#define DIRECT_ACCESS_SPINOR
#else
#define DIRECT_ACCESS_FAT_LINK
#endif // __COMPUTE_CAPABILITY >= 200
#endif // GPU_STAGGERED_DIRAC

#include <quda_internal.h>
#include <blas_quda.h>
#include <face_quda.h>



namespace { // anonymous namespace

  int Nstream = 9; // Do I really want 9 streams
  
  cudaEvent_t packEnd[Nstream];
  cudaEvent_t gatherStart[Nstream];
  cudaEvent_t gatherEnd[Nstream];
  cudaEvent_t scatterStart[Nstream];
  cudaEvent_t scatterEnd[Nstream];

  FaceBuffer *face;
  cudaColorSpinorField *inField;
  cudaColorSpinorField *outField;
} // anonymous namespace


static void setFace(const FaceBuffer& Face){
  face = (FaceBuffer*)&Face;
  return;
}

void createCopyEvents()
{
  for(int i=0; i<Nstream; ++i){
    cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherStart[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
    cudaEventCreate(&scatterStart[i], cudaEventDisableTiming);
    cudaEventCreate(&scatterEnd[i], cudaEventDisableTiming);
  }
  checkCudaError();
  return;
}


void destroyCopyEvents()
{
  for(int i=0; i<Nstream; ++i){
    cudaEventDestroy(packEnd[i]);
    cudaEventDestroy(gatherStart[i]);
    cudaEventDestroy(gatherEnd[i]);
    cudaEventDestroy(scatterStart[i]);
    cudaEventDestroy(scatterEnd[i]);
  }
  checkCudaError();
  return;
}


void initCopyCommsPattern()
{
  for(int i=0; i<Nstream-1; ++i){
    gatherCompleted[i] = 0;
    commsCompleted[i] = 0;
  }
  gatherCompleted[Nstream-1] = 1;
  commsCompleted[Nstream-1] = 1;

  return;
}



// Need to define domainParam
void copyBorder(const int parity, const int volume, const int *faceVolumeCB)
{

  domainParam.parity = parity;
  
  // Pack the faces (single parity)
  for(int i=3; i>=0; --i){
   if(domainParam.commDim[i]){
     // pack a single parity
     face->pack(*inField, parity, 0, i, streams); // pack in streams[Nstream-1]
     cudaEvenRecord(packEnd[2*i], streams[Nstream-1]); 
   }
  }

  for(int i=3; i>=0; --i){
    if(domainParam.commDim[i]){
      for(int dir=1; dir>=0; dir--){
        // different streams for send forwards and backwards
        // if dir = 0, pack at front and send backwards
        // if dir = 1, pack at back and send forwards
        cudaStreamWaitEvent(streams[2*i+dir], packEnd[2*i], 0);
        // copy from device to host
        face->gather(*inSpinor, 0, 2*i+dir);
	       // record the end of the copy from device to host
        cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]);
      }
    }
  } // end loop over directions


  int completeSum = 0; 
  while (completeSum < commDimTotal){
    for(int i=3; i>=0; --i){
      if(domainParam.commDim[i]){
        for(int dir=1; dir>=0; --dir){
									
	  // Query if gather has completed. If so, start communication.
          if(!gatherCompleted[2*i+dir] && gatherCompleted[previousDir[2*i+dir]]){
	    if(cudaSuccess == cudaEventQuery(gatherEnd[2*i+dir])){
	      gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      face->commsStart(2*i+dir); // start sending
            }
          }
         
          // Query if comms has finished 
	  if(!commsCompleted[2*i+dir] && commsCompleted[previousDir[2*i+dir]] &&
	    gatherCompleted[2*i+dir]) {
	    if(face->commsQuery[2*i+dir]){
              commsCompleted[2*i+dir] = 1;
              completeSum++;
	      face->scatter(*inSpinor, dagger, 2*i+dir); // copy from host to device	
	    }
          }
	}
	cudaEventRecord(scatterEnd[2*i], streams[2*i]);
	cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0);

        // unpack on the device. Why pack on the same stream?
        face->unpack(*outField, parity, dagger, i, streams);
      }
    } // loop over i
  } // repeat while(completeSum < commDimTotal)

  return;
}







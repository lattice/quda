#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <domain_decomposition.h>
#include <color_spinor_field.h>
#include <resize_quda.h>

// streams is defined in interface_quda.cpp
#define DIRECT_ACCESS_PACK
namespace quda {

  namespace resize {
#include <texture.h> 
#include <spinor_types.h>
  } 
  using namespace resize;

  // code for extending and cropping cudaColorSpinorField
  template<typename FloatN, int N, typename Output, typename Input>
    __global__ void cropKernel(Output Y, Input X, unsigned int length, 
        DecompParam params, int parity)
    {
      // length is the size of the smaller domain
      int little_cb_index = blockIdx.x*blockDim.x + threadIdx.x;
      int gridSize = gridDim.x*blockDim.x;


      // Need to change this
      while(little_cb_index < length){ // length is for one parity, and does not
        // include ghost zone.
        // cb_index = ( (x4*X3X2X1h + x3*X2X1h + x2*X1h + x1h) );
        int x1h = little_cb_index % params.X1h;
        int x2 =  (little_cb_index/params.X1h) % params.X2;
        int x3 =  (little_cb_index/params.X2X1h) % params.X3;
        int x4 = little_cb_index/params.X3X2X1h;
        int x1odd = (x2+x3+x4) & parity;
        int x1 = 2*x1h + x1odd;

        // coordinates on the large lattice
        int y1 = x1 + params.B1;
        int y2 = x2 + params.B2;
        int y3 = x3 + params.B3;
        int y4 = x4 + params.B4; 

        int large_cb_index = (y4*params.Y3Y2Y1 +  y3*params.Y2Y1 + y2*params.Y1 + y1) >> 1;

        FloatN x[N]; 
        X.load(x, large_cb_index);
        Y.save(x, little_cb_index);

        little_cb_index += gridSize; 
      }
      return;
    }


  template<typename FloatN, int N, typename Output, typename Input>
    __global__ void copyInteriorKernel(Output Y, Input X, unsigned int length, 
        DecompParam params, int parity)
    {
      // length is the size of the smaller domain
      int little_cb_index = blockIdx.x*blockDim.x + threadIdx.x;
      int gridSize = gridDim.x*blockDim.x;

      while(little_cb_index < length){
        FloatN x[N];
        X.load(x, little_cb_index);

        // cb_index = ( (x4*X3X2X1h + x3*X2X1h + x2*X1h + x1h) );
        int x1h = little_cb_index % params.X1h;
        int x2 =  (little_cb_index/params.X1h) % params.X2;
        int x3 =  (little_cb_index/params.X2X1h) % params.X3;
        int x4 = little_cb_index/params.X3X2X1h;
        int x1odd = (x2+x3+x4) & parity;
        int x1 = 2*x1h + x1odd;

        // coordinates on the large lattice
        int y1 = x1 + params.B1;
        int y2 = x2 + params.B2;
        int y3 = x3 + params.B3;
        int y4 = x4 + params.B4;

        int large_cb_index = (y4*params.Y3Y2Y1 +  y3*params.Y2Y1 + y2*params.Y1 + y1) >> 1;

        Y.save(x, large_cb_index); 
        little_cb_index += gridSize;
      }
      return;
    }

  __device__ void getCoordinates(int* const x1_p, int* const x2_p,
      int* const x3_p, int* const x4_p,
      int cb_index, const DecompParam& params, int parity, int Dir)
  {

    int xh, xodd;
    switch(Dir){
      case 0:
        // cb_idx = (x1*X4X3X2 + x4*X3X2 + x3*X2 + x2)/2
        xh = cb_index % params.X2h;
        *x3_p = (cb_index/params.X2h) % params.X3;
        *x1_p = cb_index/params.X4X3X2h;
        *x4_p = (cb_index/params.X3X2h) % params.X4;
        xodd = (*x1_p + *x3_p + *x2_p + parity) & 1;
        *x2_p = 2*xh + xodd;
        break;

      case 1:
        // cb_index = (x2*X4X3X1 + x4*X3X1 + x3*X1 + x1)/2
        xh = cb_index % params.X1h;
        *x3_p = (cb_index/params.X1h) % params.X3;
        *x2_p = cb_index/params.X4X3X1h;
        *x4_p = (cb_index/(params.X3X1h)) % params.X4;
        xodd = (*x2_p + *x3_p + *x4_p + parity) & 1;
        *x1_p = 2*xh + xodd;
        break;

      case 2:
        // cb_index = (x3*X4X2X1 + x4*X2X1 + x2*X1 + x1)/2
        xh = cb_index % params.X1h;
        *x2_p = (cb_index/params.X1h) % params.X2;
        *x3_p = cb_index/params.X4X2X1h;
        *x4_p = (cb_index/params.X2X1h) % params.X4;
        xodd = (*x2_p + *x3_p + *x4_p + parity) & 1;
        *x1_p = 2*xh + xodd;
        break; 

      case 3:
        // cb_index = (x4*X3X2X1 + x3*X2X1 + x2*X1 + x1)/2
        // Note that this is the canonical ordering in the interior region.
        xh = cb_index % params.X1h;
        *x2_p = (cb_index/params.X1h) % params.X2;
        *x4_p = (cb_index/params.X3X2X1h);
        *x3_p = (cb_index/params.X2X1h) % params.X3;
        xodd = (*x2_p + *x3_p + *x4_p + parity) & 1;
        *x1_p = 2*xh + xodd;
        break;

      default:
        break;
    } // switch(Dir)

    return;
  }



  __device__ void getDomainCoordsFromGhostCoords(int* const y1_p, 
      int* const y2_p,
      int* const y3_p,
      int* const y4_p,
      int x1,
      int x2,
      int x3, 
      int x4,
      const DecompParam& params,
      int Dir)
  {

    *y1_p = x1;
    *y2_p = x2;
    *y3_p = x3;
    *y4_p = x4; 

    switch(Dir){
      case 0:
        if(x1 >= params.B1) *y1_p += params.X1;
        if(params.B2 > 0) *y2_p += params.B2;
        if(params.B3 > 0) *y3_p += params.B3;
        if(params.B4 > 0) *y4_p += params.B4;
        break;

      case 1:
        if(x2 >= params.B2) *y2_p += params.X2;
        if(params.B1 > 0) *y1_p += params.B1; 
        if(params.B3 > 0) *y3_p += params.B3;
        if(params.B4 > 0) *y4_p += params.B4;
        break;

      case 2:
        if(x3 >= params.B3) *y3_p += params.X3;
        if(params.B1 > 0) *y1_p += params.B1; 
        if(params.B2 > 0) *y2_p += params.B2;
        if(params.B4 > 0) *y4_p += params.B4;
        break;

      case 3: 
        if(x4 >= params.B4) *y4_p += params.X4;
        if(params.B1 > 0) *y1_p += params.B1; 
        if(params.B2 > 0) *y2_p += params.B2;
        if(params.B3 > 0) *y3_p += params.B3;
        break;
    }
    return;
  }


  // Need to generalise SpinorIndex
  template<typename FloatN, int N, typename Output, typename Input>
    __global__ void copyExteriorKernel(Output Y, Input X, unsigned int length, DecompParam params, int parity, int Dir)
    {
      int cb_index = blockIdx.x*blockDim.x + threadIdx.x;
      int gridSize = gridDim.x*blockDim.x; 

      int x1, x2, x3, x4;
      int y1, y2, y3, y4;

      while(cb_index < length){
        getCoordinates(&x1, &x2, &x3, &x4, cb_index, params, parity, Dir);

        getDomainCoordsFromGhostCoords(&y1, &y2, &y3, &y4,
            x1, x2, x3, x4, params, Dir); 

        int large_cb_index = (y4*params.Y3Y2Y1 + y3*params.Y2Y1 + y2*params.Y1 + y1) >> 1;

        FloatN x[N];
        X.load(x, cb_index);
        Y.save(x, large_cb_index);

        cb_index += gridSize;
      }
      return;
    }


  template <typename FloatN, int N, typename Output, typename Input>
    class ExtendCuda {

      private:
        Input &X;
        Output &Y;
        const int length;
        DecompParam params;
        const int parity;
        const int dir; // if copying from border

        int sharedBytesPerThread() const { return 0; }


      public:
        ExtendCuda(Output &Y, Input &X, int length, const DecompParam& params, int parity) : X(X), Y(Y), length(length), params(params), parity(parity), dir(-1) {}

        ExtendCuda(Output &Y, Input &X, int length, const DecompParam& params, int parity, const int dir) :
          X(X), Y(Y), length(length), params(params), parity(parity), dir(dir) {}
        virtual ~ExtendCuda(){}

        void apply(const cudaStream_t &stream){

          int parity = 0;
          const unsigned int blockX = 128;
          const unsigned int gridX = (length + (blockX-1))/blockX;

          dim3 blockDim(blockX,1,1); // warp size on GK110
          dim3 gridDim(gridX,1,1); // random choice - change this

          if(dir<0){
            copyInteriorKernel<FloatN, N><<<gridDim, blockDim, 0, stream>>>(Y, X, length, params, parity); 
          }else if(dir>=0 && dir<4){
            copyExteriorKernel<FloatN, N><<<gridDim, blockDim, 0, stream>>>(Y, X, length, params, parity, dir); 
          }else{
            errorQuda("dir %d is unrecognized");
          }
        }
    };

  template <typename FloatN, int N, typename Output, typename Input>
    class CropCuda  {

      private:
        Input &X;
        Output &Y;
        const int length;
        DecompParam params;
        int parity; // parity of the destination field


        int sharedBytesPerThread() const { return 0; }
      public:
        CropCuda(Output &Y, Input &X, int length, const DecompParam& params, int parity) : X(X), Y(Y), length(length), params(params), parity(parity) {}
        virtual ~CropCuda(){}

        void apply(const cudaStream_t &stream){
          // Need to set gridDim and blockDim
          const unsigned int blockX = 128;
          const unsigned int gridX = (length + (blockX-1))/blockX;
          dim3 blockDim(blockX,1,1); // Warp size on the GK110
          dim3 gridDim(gridX,1,1);
          cropKernel<FloatN, N><<<gridDim, blockDim, 0, stream>>>(Y, X, length, params, parity); 
        }
    };


  struct CommParam {
    struct threads; // the desired number of active threads
    int parity; // even or odd
    int commDim[QUDA_MAX_DIM]; 
    // Whether to do comms or not
    // a given dimension
  };

#ifdef MULTI_GPU
  static  int gatherCompleted[Nstream]; // transfer of ghost data from device to host
  static int previousDir[Nstream];
  static int commsCompleted[Nstream];
  static int extendCompleted[Nstream];

  static cudaEvent_t packEnd[Nstream];
  //  static cudaEvent_t gatherStart[Nstream];
  static cudaEvent_t gatherEnd[Nstream];
  //  static cudaEvent_t scatterStart[Nstream];
  static cudaEvent_t scatterEnd[Nstream];
#endif


  void createExtendEvents()
  {
    // copied from createDslashEvents
#ifndef DSLASH_PROFILING
    // add cudaEventDisableTiming for lower sync overhead 
    printfQuda("Calling createExtendEvents()\n");
    for(int i=0; i<Nstream; ++i){
      cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
      cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
      cudaEventCreate(&scatterEnd[i], cudaEventDisableTiming);
    }
#else
    printfQuda("Calling createExtendEvents()\n");
    for(int i=0; i<Nstream; ++i){
      cudaEventCreate(&packEnd[i]);
      cudaEventCreate(&gatherEnd[i]);
      cudaEventCreate(&scatterEnd[i]);
    }
#endif
    return;
  }

  void destroyExtendEvents(){
    printfQuda("Calling destroyExtendEvents()\n");
    for(int i=0; i<Nstream; ++i){
      cudaEventDestroy(packEnd[i]);
      cudaEventDestroy(gatherEnd[i]);
      cudaEventDestroy(scatterEnd[i]);
    }
    return;
  }

#ifdef MULTI_GPU
  static void initCommsPattern(int* commDimTotal, const int* const domain_overlap) 
  {

    for(int i=0; i<Nstream-1; i++){
      gatherCompleted[i] = 0;
      commsCompleted[i] = 0;
      extendCompleted[i] = 0;
    }
    gatherCompleted[Nstream-1] = 1; // nothing required there
    commsCompleted[Nstream-1] = 1; 

    // We need to know which was the previous direction in which 
    // communication was issued, since we only query a given event /
    // comms call after the previous one has successfully 
    // completed
    for(int i=3; i>=0; --i){
      if(domain_overlap[i]){
        int prev = Nstream-1;
        for(int j=3; j>i; --j) if(domain_overlap[j]) prev = 2*j;
        previousDir[2*i + 1] = prev;
        previousDir[2*i + 0] = 2*i + 1;
      }
    }

    // this tells us how many events / comms occurances there are in total.
    // Used for exiting the while loop.
    *commDimTotal = 0;
    for (int i=3; i>=0; --i) *commDimTotal += (domain_overlap[i]>0) ? 1 : 0;
    *commDimTotal *= 4; // 2 from pipe length, 2 from direction
    return;
  }
#endif

  template<class DataType, class DstSpinorType, class SrcSpinorType>
    static void extendCuda__(cudaColorSpinorField& dst, cudaColorSpinorField& src, const DecompParam& params, const int parity, const int* const domain_overlap, FaceBuffer* face)
    {

      printfQuda("Inside extendCuda__\n");
      size_t free, total;
      cudaMemGetInfo(&free, &total);
      printfQuda("Free memory : %d\n", free);
      printfQuda("Total memory: %d\n", total);
      fflush(stdout);       

#ifdef MULTI_GPU
      for(int i=3; i >= 0; i--){
        if(domain_overlap[i]){
          // Initiate pack from source spinor on the device
          printfQuda("extendCuda__ : packing direction %d\n", i);
          fflush(stdout);
          face->pack(src, parity, 0, i, streams); // pack in stream[Nstream-1]

          cudaError_t packSync = cudaDeviceSynchronize();
          if(packSync != cudaSuccess){
            printfQuda("face->pack failed\n");
          }
          cudaEventRecord(packEnd[2*i], streams[Nstream-1]);
          printfQuda("extendCuda__ : packing in direction %d complete\n", i);
          fflush(stdout);
          cudaError_t packQuery = cudaEventQuery(packEnd[2*i]);
          switch(packQuery){
            case cudaErrorInvalidValue:
              printfQuda("pack cudaErrorInvalidValue!\n");
              break;
            case cudaErrorInitializationError:
              printfQuda("pack cudaErrorInitializationError!\n");
              break;
            case cudaErrorInvalidResourceHandle:
              printfQuda("pack cudaErrorInvalidResourceHandle!\n");
              break;
            case cudaErrorLaunchFailure:
              printfQuda("pack cudaErrorLaunchFailure!\n");
              break;
            default:
              break;
          }
        }
      }

      for(int i=3; i >= 0; i--){
        if(domain_overlap[i]){
          for(int dir=1; dir >= 0; dir--){
            cudaStreamWaitEvent(streams[2*i+dir], packEnd[2*i], 0);

            printfQuda("extendCuda__ : gathering direction %d \n", i);
            fflush(stdout);

            // Initiate transfer of packed ghost data from device to host
            face->gather(src, 0, 2*i+dir); // what does dagger do, and should I be concerned?

            cudaDeviceSynchronize();

            printfQuda("extendCuda__ : gather in direction %d complete\n", i);
            fflush(stdout);

            cudaDeviceSynchronize();
            // Record the end of the gathering 
            cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]);

            cudaError_t gatherQuery = cudaEventQuery(gatherEnd[2*i+dir]);

            switch(gatherQuery){
              case cudaErrorInvalidValue:
                printfQuda("cudaErrorInvalidValue!\n");
                break;
              case cudaErrorInitializationError:
                printfQuda("cudaErrorInitializationError!\n");
                break;
              case cudaErrorInvalidResourceHandle:
                printfQuda("cudaErrorInvalidResourceHandle!\n");
                break;
              case cudaErrorLaunchFailure:
                printfQuda("cudaErrorLaunchFailure!\n");
                break;
              default:
                break;
            }

          } // dir = 0,1
        } // if domain_overlap[i]
      } // i = 0,1,2,3

#endif  // MULTI_GPU  

      cudaDeviceSynchronize();

      SrcSpinorType src_spinor(src);  
      DstSpinorType dst_spinor(dst);

      ExtendCuda<DataType, 3, DstSpinorType, SrcSpinorType> 
        extend(dst_spinor, src_spinor, src.Volume(), params, parity);

      printfQuda("Calling interior extend.apply\n");
      fflush(stdout);
      extend.apply(streams[Nstream-1]); // copy the interior region. 

      printfQuda("Call to interior extend.apply complete\n");
      fflush(stdout);
#ifdef MULTI_GPU
      int commDimTotal=0;
      initCommsPattern(&commDimTotal,domain_overlap);

      int completeSum = 0;
      printfQuda("commDimTotal = %d\n", commDimTotal);
      printfQuda("domain_overlap = %d %d %d %d\n", domain_overlap[0], domain_overlap[1],
          domain_overlap[2], domain_overlap[3]);

      int attempts = 0;
      while(completeSum < commDimTotal) {
        attempts++;
        for(volatile int i=3; i >= 0; i--){
          if(domain_overlap[i]){
            for(int dir=1; dir >= 0; dir--){
              // Query if gather (transfer of ghost data to host) has completed
              if(!gatherCompleted[2*i+dir] && gatherCompleted[previousDir[2*i+dir]]){
                printfQuda("Checking to see if gather[%d] completed\n",2*i+dir);


                cudaError_t gatherQuery = cudaEventQuery(gatherEnd[2*i+dir]);
                /*
                   switch(gatherQuery){
                   case cudaErrorInvalidValue:
                   printfQuda("cudaErrorInvalidValue!\n");
                   break;
                   case cudaErrorInitializationError:
                   printfQuda("cudaErrorInitializationError!\n");
                   break;
                   case cudaErrorInvalidResourceHandle:
                   printfQuda("cudaErrorInvalidResourceHandle!\n");
                   break;
                   case cudaErrorLaunchFailure:
                   printfQuda("cudaErrorLaunchFailure!\n");
                   break;
                   default:
                   break;
                   }
                 */
                if(cudaSuccess == cudaEventQuery(gatherEnd[2*i+dir])){
                  printfQuda("Gather Completed!\n");
                  gatherCompleted[2*i+dir] = 1;
                  completeSum++;
                  printfQuda("extendCuda__ : calling face->commsStart(%d)\n",2*i+dir);
                  fflush(stdout);
                  face->commsStart(2*i+dir); // start communication
                }
              } // if not gather completed


              // Query if comms has finished 
              if(!commsCompleted[2*i+dir] && commsCompleted[previousDir[2*i+dir]] && 
                  gatherCompleted[2*i+dir]){

                printfQuda("calling face->commsQuery\n");

                if(face->commsQuery(2*i+dir)){     
                  commsCompleted[2*i+dir] = 1;
                  completeSum++;
                  // copy data back to the ghost zones on the device
                  printfQuda("extendCuda__ : calling face->scatter in direction %d\n", 2*i+dir);
                  fflush(stdout);
                  face->scatter(src, 0, 2*i+dir);
                }
              } // if comms not completed
            } // dir = 0,1

            if((attempts % 500) == 0) printfQuda("Attempts = %d\n", attempts);

            if(!extendCompleted[2*i] && commsCompleted[2*i] && commsCompleted[2*i+1])
            {
              cudaEventRecord(scatterEnd[2*i], streams[2*i]);
              // now copy the data from the ghost zone on the smaller field 
              // to the ghost zone in the larger field
              // Call the constructor Spinor(void* spinor, float* norm, int stride)
              DstSpinorType dst_spinor(dst);

              // Wait for the scatter to finish 
              cudaStreamWaitEvent(streams[2*i], scatterEnd[2*i], 0);
              SrcSpinorType src_spinor(src.Ghost(i), static_cast<float*>(src.GhostNorm(i)), src.GhostFace()[i]); 

              printfQuda("About to call exterior extend.apply\n");
              fflush(stdout);
              ExtendCuda<DataType, 3, DstSpinorType, SrcSpinorType> 
                extend(dst_spinor, src_spinor, src.GhostFace()[i], params, parity, i);
              // Need to be careful here to ensure that face->scatter has completed
              // I believe that streams used for extend.apply now match the scatter streams
              extend.apply(streams[2*i]);
              extendCompleted[2*i] = 1;

              printfQuda("Call to extend.apply complete\n");
              fflush(stdout);
            }
          }
        } // i = 0,1,2,3
      } // completeSum < commDimTotal


#endif
      cudaDeviceSynchronize(); // as a safety measure

      printfQuda("At the end of extendCuda__\n");
      cudaMemGetInfo(&free, &total);
      printfQuda("total memory = %d\n", total);
      printfQuda("free memory = %d\n", free);
      fflush(stdout); 
      return;
    }



#ifdef EXTEND_CORE
  BORK_COMPILATION;
#endif
#define EXTEND_CORE(DataType, DST_PRECISION, SRC_PRECISION) \
  if ((src.Precision() == SRC_PRECISION) && (dst.Precision() == DST_PRECISION)) { \
    typedef typename SpinorType<1, DST_PRECISION, SRC_PRECISION>::InType SrcSpinorType; \
    typedef typename SpinorType<1, DST_PRECISION, SRC_PRECISION>::OutType DstSpinorType; \
    SrcSpinorType src_spinor(src);                                                       \
    DstSpinorType dst_spinor(dst);                                                       \
    extendCuda__<DataType,DstSpinorType,SrcSpinorType>(dst, src, params, parity, domain_overlap, face);        \
  }

  // Constructor
  Extender::Extender(const cudaColorSpinorField& field) : face(NULL)
  {
    int X[4];
    X[0] = field.X(0)*2;
    X[1] = field.X(1);
    X[2] = field.X(2);
    X[3] = field.X(3);

    // allocate the FaceBuffer
    // delete is called in Extender's destructor
    face = new FaceBuffer(X, 4, 2*field.Nspin()*field.Ncolor(), field.Nface(), field.Precision());
  }

  void Extender::operator()(cudaColorSpinorField &dst, cudaColorSpinorField &src, const DecompParam& params, const int * const domain_overlap) 
  {

    printfQuda("Calling Extender::operator()\n");
    fflush(stdout);

    if (&src == &dst) return; // aliasing fields


    if (src.Nspin() != 1 && src.Nspin() != 4) errorQuda("nSpin(%d) not supported");

    //if (src.Length() >= dst.Length()) errorQuda("src length should be less than destination length");
    if (src.Length() > dst.Length()) errorQuda("src length should be less than destination length");

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET || src.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    {
      errorQuda("QUDA_FULL_SITE_SUBSET is not yet supported\n");
      // This probably won't work. Need to think about it.
      // extendCuda(dst.Even(), src.Even(), params, domain_overlap);
      //extendCuda(dst.Odd(), src.Odd(), params, domain_overlap);   
      return;
    }

    const int parity = 0; // Need to change this
    // I should really use a function to do this
    {
      printfQuda("About to call EXTEND_CORE\n");
      fflush(stdout);

      EXTEND_CORE(double2, QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION)
        else EXTEND_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION)
          else EXTEND_CORE(float2, QUDA_HALF_PRECISION, QUDA_HALF_PRECISION)
            else EXTEND_CORE(float2, QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION)
              else EXTEND_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION)
                else EXTEND_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION)
                  else EXTEND_CORE(float2, QUDA_HALF_PRECISION, QUDA_SINGLE_PRECISION)
                    else EXTEND_CORE(double2, QUDA_DOUBLE_PRECISION, QUDA_HALF_PRECISION)
                      else EXTEND_CORE(double2, QUDA_HALF_PRECISION, QUDA_DOUBLE_PRECISION)
    }
    printfQuda("Call to EXTEND_CORE complete\n");
    fflush(stdout);
    return;
  }

#undef EXTEND_CORE




#ifdef CROP_CORE
  BORK_COMPILATION;
#endif
#define CROP_CORE(DataType, DST_PRECISION, SRC_PRECISION) \
  if ((src.Precision() == SRC_PRECISION) && (dst.Precision() == DST_PRECISION)) { \
    typedef typename SpinorType<1, DST_PRECISION, SRC_PRECISION>::InType SrcSpinorType; \
    typedef typename SpinorType<1, DST_PRECISION, SRC_PRECISION>::OutType DstSpinorType; \
    SrcSpinorType src_spinor(src);                                                       \
    DstSpinorType dst_spinor(dst);                                                       \
    CropCuda<DataType, 3, DstSpinorType, SrcSpinorType >                                       \
    crop(dst_spinor, src_spinor, src.Volume(), params, parity);                  \
    crop.apply(streams[Nstream-1]);                                                  \
  } 


  void cropCuda(cudaColorSpinorField &dst, const cudaColorSpinorField &src, const DecompParam& params) {
    if (&src == &dst) return; // aliasing fields

    if (src.Nspin() != 1 && src.Nspin() != 4) errorQuda("nSpin(%d) not supported\n");

    if (src.Length() >= dst.Length()) errorQuda("src length should be less than destination length\n");

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET || src.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    {
      errorQuda("QUDA_FULL_SITE_SUBSET is not yet supported");
      //    cropCuda(dst.Even(), src.Even(), params);
      //    cropCuda(dst.Odd(), src.Odd(), params);   
      return;
    }

    const int parity = 0; // Need to change this

    {
      // vim indentation doesn't do this right
      CROP_CORE(double2, QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION)
        else
          CROP_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION)
            else
              CROP_CORE(float2, QUDA_HALF_PRECISION, QUDA_HALF_PRECISION)
                else
                  CROP_CORE(float2, QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION)
                    else
                      CROP_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION)
                        else
                          CROP_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION)
                            else
                              CROP_CORE(float2, QUDA_HALF_PRECISION, QUDA_SINGLE_PRECISION)
                                else
                                  CROP_CORE(double2, QUDA_DOUBLE_PRECISION, QUDA_HALF_PRECISION)
                                    else
                                      CROP_CORE(double2, QUDA_HALF_PRECISION, QUDA_DOUBLE_PRECISION);
    }
    return;
  } // cropCuda
#undef CROP_CORE


} // namespace quda

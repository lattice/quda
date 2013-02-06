#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <domain_decomposition.h>
#include <color_spinor_field.h>

#include <spinor_types.h>

// streams is defined in interface_quda.cpp

namespace quda {

  FaceBuffer* face; // need to set this. I can u
  // I need to use setFace to set this. 
  // Factor out of dslash_quda.cu
#include <texture.h>  

  // code for extending and cropping cudaColorSpinorField
  template<typename FloatN, int N, typename Output, typename Input>
    __global__ void cropKernel(Output Y, Input X, unsigned int length, 
        DecompParams params, int parity)
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
        DecompParams params, int parity)
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
      int cb_index, const DecompParams& params, int parity, int Dir)
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
      const DecompParams& params,
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
    __global__ void copyExteriorKernel(Output Y, Input X, unsigned int length, DecompParams params, int parity, int Dir)
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

      }
      return;
    }


  template <typename FloatN, int N, typename Output, typename Input>
    class ExtendCuda {

      private:
        Input &X;
        Output &Y;
        const int length;
        DecompParams params;
        const int parity;
        const int dir; // if copying from border

        int sharedBytesPerThread() const { return 0; }


      public:
        ExtendCuda(Output &Y, Input &X, int length, const DecompParams& params, int parity) : X(X), Y(Y), length(length), params(params), parity(parity), dir(-1) {}

        ExtendCuda(Output &Y, Input &X, int length, const DecompParams& params, int parity, const int dir) :
          X(X), Y(Y), length(length), params(params), parity(parity), dir(dir) {}
        virtual ~ExtendCuda();

        void apply(const cudaStream_t &stream){

          int parity = 0;

          dim3 blockDim(32,1,1); // warp size on GK110
          dim3 gridDix(128,1,1); // random choice - change this

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
        DecompParams params;
        int parity; // parity of the destination field


        int sharedBytesPerThread() const { return 0; }
      public:
        CropCuda(Output &Y, Input &X, int length, const DecompParams& params, int parity) : X(X), Y(Y), length(length), params(params), parity(parity) {;}
        virtual ~CropCuda();

        void apply(const cudaStream_t &stream){
          // Need to set gridDim and blockDim
          dim3 blockDim(32,1,1); // Warp size on the GK110
          dim3 gridDim(128,1,1);
          cropKernel<FloatN, N><<<gridDim, blockDim, 0, stream>>>(Y, X, length, params, parity); 
        }
    };


  struct CommParam {
    struct threads; // the desired number of active threads
    int parity; // even or odd
    int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
    // a given dimension
  };

  static CommParam commParam;

  int gatherCompleted[Nstream]; // transfer of ghost data from device to host
  int previousDir[Nstream];
  int commsCompleted[Nstream];
  int extendCompleted[Nstream];
  int commDimTotal;


  static cudaEvent_t packEnd[Nstream];
  static cudaEvent_t gatherStart[Nstream];
  static cudaEvent_t gatherEnd[Nstream];
  static cudaEvent_t scatterStart[Nstream];
  static cudaEvent_t scatterEnd[Nstream];



  static void initCommsPattern() {

    for(int i=0; i<Nstream-1; i++){
      gatherCompleted[i] = 0;
      commsCompleted[i] = 0;
    }
    gatherCompleted[Nstream-1] = 1; // nothing required there
    commsCompleted[Nstream-1] = 1; 

    // We need to know which was the previous direction in which 
    // communication was issued, since we only query a given event /
    // comms call after the previous one has successfully 
    // completed
    for(int i=3; i>=0; --i){
      if(commParam.commDim[i]){
        int prev = Nstream-1;
        for(int j=3; j>i; --j) if(commParam.commDim[j]) prev = 2*j;
        previousDir[2*i + 1] = prev;
        previousDir[2*i + 0] = 2*i + 1;
      }
    }

    // this tells us how many events / comms occurances there are in total.
    // Used for exiting the while loop.
    commDimTotal = 0;
    for (int i=3; i>=0; --i) commDimTotal += commParam.commDim[i];
    commDimTotal *= 4; // 2 from pipe length, 2 from direction
    return;
  }


  template<class DataType, class DstSpinorType, class SrcSpinorType>
    static void extendCuda__(cudaColorSpinorField& dst, cudaColorSpinorField& src, const DecompParams& params, int parity)
    {
      commParam.parity = parity;
#ifdef MULTI_GPU
      for(int i=3; i >= 0; i--){
        if(!commParam.commDim[i]) continue;

        // Initialise pack from source spinor on the device
        face->pack(src, parity, 0, i, streams); // pack in stream[Nstream-1]
        // Record the end of the packing 
        cudaEventRecord(packEnd[2*i], streams[Nstream-1]);
      }

      for(int i=3; i >= 0; i--){
        if(!commParam.commDim[i]) continue;

        for(int dir=1; dir >= 0; dir--){
          cudaStreamWaitEvent(streams[2*i+dir], packEnd[2*i+dir], 0);

          // Initialise transfer of packed ghost data from device to host
          face->gather(src, 0, 2*i+dir); // what does dagger do, and should I be concerned

          // Record the end of the gathering 
          cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]);
        } // dir = 0,1
      } // i = 0,1,2,3

#endif  // MULTI_GPU  

      SrcSpinorType src_spinor(src);  
      DstSpinorType dst_spinor(dst);

      ExtendCuda<DataType, 3, DstSpinorType, SrcSpinorType> 
        extend(dst_spinor, src_spinor, src.Volume(), params, parity);
      extend.apply(streams[Nstream-1]); // copy the interior region. 

      initCommsPattern();

      int completeSum = 0;
      while(completeSum < commDimTotal) {
        for(int i=3; i >= 0; i--){
          if(!commParam.commDim[i]) continue;

          for(int dir=1; dir >= 0; dir--){

            // Query if gather (transfer of ghost data to host) has completed
            if(!gatherCompleted[2*i+dir] && gatherCompleted[previousDir[2*i+dir]]){

              if(cudaSuccess == cudaEventQuery(gatherEnd[2*i+dir])){
                gatherCompleted[2*i+dir] = 1;
                completeSum++;
                face->commsStart(2*i+dir); // start communication
              }
            } // if not gather completed


            // Query if comms has finished 
            if(!commsCompleted[2*i+dir] && commsCompleted[previousDir[2*i+dir]] && 
                gatherCompleted[2*i+dir]){

              if(face->commsQuery(2*i+dir)){     
                commsCompleted[2*i+dir] = 1;
                completeSum++;

                // copy data back to the ghost zones on the device
                face->scatter(src, 0, 2*i+dir);
              }
            } // if comms not completed
          } // dir = 0,1

          // now copy the data from the ghost zone on the smaller field 
          // to the ghost zone in the larger field
          // Call the constructor Spinor(void* spinor, float* norm, int stride)
          SrcSpinorType src_spinor(src.Ghost(i), static_cast<float*>(src.GhostNorm(i)), src.GhostFace()[i]); 
          DstSpinorType dst_spinor(dst);

          ExtendCuda<DataType, 3, DstSpinorType, SrcSpinorType> 
            extend(dst_spinor, src_spinor, src.GhostFace()[i], params, parity, i);

          // Need to be careful here to ensure that face->scatter has completed
          // I believe that streams used for extend.apply now match the scatter streams
          extend.apply(streams[2*i]);


        } // i = 0,1,2,3
      } // completeSum < commDimTotal

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
    extendCuda__<DataType,DstSpinorType,SrcSpinorType>(dst, src, params, parity);        \
  }


  void extendCuda(cudaColorSpinorField &dst, cudaColorSpinorField &src, const DecompParams& params) {
    if (&src == &dst) return; // aliasing fields

    if (src.Nspin() != 1 && src.Nspin() != 4) errorQuda("nSpin(%d) not supported\n");

    if (src.Length() >= dst.Length()) errorQuda("src length should be less than destination length\n");

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET || src.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    {
      extendCuda(dst.Even(), src.Even(), params);
      extendCuda(dst.Odd(), src.Odd(), params);   
      return;
    }

    const int parity = 0; // Need to change this

    // I should really use a function to do this
    EXTEND_CORE(double2, QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION)
    else
      EXTEND_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION)
    else 
      EXTEND_CORE(float2, QUDA_HALF_PRECISION, QUDA_HALF_PRECISION)
    else
      EXTEND_CORE(float2, QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION)
    else
      EXTEND_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION)
    else
      EXTEND_CORE(float2, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION)
    else
      EXTEND_CORE(float2, QUDA_HALF_PRECISION, QUDA_SINGLE_PRECISION)
    else
      EXTEND_CORE(double2, QUDA_DOUBLE_PRECISION, QUDA_HALF_PRECISION)
    else
      EXTEND_CORE(double2, QUDA_HALF_PRECISION, QUDA_DOUBLE_PRECISION)

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


  void cropCuda(cudaColorSpinorField &dst, const cudaColorSpinorField &src, const DecompParams& params) {
    if (&src == &dst) return; // aliasing fields

    if (src.Nspin() != 1 && src.Nspin() != 4) errorQuda("nSpin(%d) not supported\n");

    if (src.Length() >= dst.Length()) errorQuda("src length should be less than destination length\n");

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET || src.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    {
      cropCuda(dst.Even(), src.Even(), params);
      cropCuda(dst.Odd(), src.Odd(), params);   
      return;
    }

    const int parity = 0; // Need to change this

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

                    return;
  } // cropCuda
#undef CROP_CORE



} // namespace

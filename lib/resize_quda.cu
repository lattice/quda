#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <domain_decomposition.h>
#include <color_spinor_field.h>

namespace quda {

/*
  cudaStream_t* getBlasStream();

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

  template<int Dir> 
    __device__ void getCoordinates(int* const x1_p, int* const x2_p,
        int* const x3_p, int* const x4_p,
        int cb_index, const DecompParams& params, int parity)
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



  template<int Dir>
    __device__ void getDomainCoordsFromGhostCoords(int* const y1_p, 
        int* const y2_p,
        int* const y3_p,
        int* const y4_p,
        int x1,
        int x2,
        int x3, 
        int x4,
        const DecompParams& params)
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
  template<int Dir, typename FloatN>
    __global__ void copyExteriorKernel(int parity, DecompParams params)
    {
      int cb_index = blockIdx.x*blockDim.x + threadIdx.x;
      int gridSize = gridDim.x*blockDim.x; 

      int x1, x2, x3, x4;
      int y1, y2, y3, y4;

      getCoordinates<Dir>(&x1, &x2, &x3, &x4, cb_index, params, parity);

      getDomainCoordsFromGhostCoords<Dir>(&y1, &y2, &y3, &y4,
          x1, x2, x3, x4, params); 

      int large_cb_index = (y4*params.Y3Y2Y1 + y3*params.Y2Y1 + y2*params.Y1 + y1) >> 1;

      // Need to put a copy routine in there

      return;
    }

  /*
     template<int Dir, typename FloatN, int N, typename Output, typename Input>
     __global__ void copyExteriorKernel(Output Y, Input X, DecompParams params, int parity)
     {

     int cb_index = blockIdx.x*blockDim.x + threadIdx.x;
     int gridSize = gridDim.x*blockDim.x; 
     }
   */

  template <typename FloatN, int N, typename Output, typename Input>
    class ExtendCuda {

      private:
        Input &X;
        Output &Y;
        const int length;
        DecompParams params;
        const int parity;

        int sharedBytesPerThread() const { return 0; }
        /*
           virtual bool advanceSharedBytes(TuneParam &param) const
           {
           TuneParam next(param);
           advanceBlockDim(next); // to get next blockDim
           int nthreads = next.block.x * next.block.y * next.block.z;
           param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ?
           sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
           return false;
           }
         */
      public:
        ExtendCuda(Output &Y, Input &X, int length, const DecompParams& params, int parity) : X(X), Y(Y), length(length), params(params), parity(parity) {;}
        virtual ~ExtendCuda();

        void apply(const cudaStream_t &stream){

          int parity = 0;

          dim3 blockDim(32,1,1); // warp size on GK110
          dim3 gridDix(128,1,1); // random choice - change this
          copyInteriorKernel<FloatN, N><<<gridDim, blockDim, 0, stream>>>(Y, X, length, params, parity); 
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
        /*
           virtual bool advanceSharedBytes(TuneParam &param) const
           {
           TuneParam next(param);
           advanceBlockDim(next); // to get next blockDim
           int nthreads = next.block.x * next.block.y * next.block.z;
           param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ?
           sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
           return false;
           }
         */
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





  void extendCuda(cudaColorSpinorField &dst, const cudaColorSpinorField &src, const DecompParams& params) {
    if (&src == &dst) return; // aliasing fields

    if (src.Nspin() != 1 && src.Nspin() != 4) errorQuda("nSpin(%d) not supported\n");

    if (src.Length() >= dst.Length()) errorQuda("src length should be less than destination length\n");

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET || src.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    {
      extendCuda(dst.Even(), src.Even(), params);
      extendCuda(dst.Odd(), src.Odd(), params);   
      return;
    }

    const int parity = 1; // Need to change this


    if(dst.Precision() == src.Precision()) {
      if(dst.Precision() == QUDA_DOUBLE_PRECISION){
        SpinorTexture<double2, double2, double2, 3, 0> src_tex(src);
        Spinor<double2, double2, double2, 3> dst_spinor(dst);
        ExtendCuda<double2, 3, Spinor<double2, double2, double2, 3>,
          SpinorTexture<double2, double2, double2, 3, 0> >
            extend(dst_spinor, src_tex, src.Volume(), params, parity);
        extend.apply(*getBlasStream());

      }else if(dst.Precision() == QUDA_SINGLE_PRECISION){
        SpinorTexture<float2, float2, float2, 3, 0> src_tex(src);
        Spinor<float2, float2, float2, 3> dst_spinor(dst);
        ExtendCuda<float2, 3, Spinor<float2, float2, float2, 3>,
          SpinorTexture<float2, float2, float2, 3, 0> >
            extend(dst_spinor, src_tex, src.Volume(), params, parity);
        extend.apply(*getBlasStream());

      }else if(dst.Precision() == QUDA_HALF_PRECISION){ 

        errorQuda("half precision not yet supported\n");
      }
    }
    return;
  }


  void cropCuda(cudaColorSpinorField &dst, const cudaColorSpinorField &src, const DecompParams& params) {
    if (&src == &dst) return; // aliasing fields

    if (src.Nspin() != 1 && src.Nspin() != 4) errorQuda("nSpin(%d) not supported\n");

    if (src.Length() <= dst.Length()) errorQuda("src length should be less greater destination length\n");

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET || src.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    {
      cropCuda(dst.Even(), src.Even(), params);
      cropCuda(dst.Odd(), src.Odd(), params);   
      return;
    }

    const int parity = 1;

    if(dst.Precision() == src.Precision()) {
      if(dst.Precision() == QUDA_DOUBLE_PRECISION){
        SpinorTexture<double2, double2, double2, 3, 0> src_tex(src);
        Spinor<double2, double2, double2, 3> dst_spinor(dst);
        CropCuda<double2, 3, Spinor<double2, double2, double2, 3>,
          SpinorTexture<double2, double2, double2, 3, 0> >
            crop(dst_spinor, src_tex, src.Volume(), params, parity);
        crop.apply(*getBlasStream());
      }else if(dst.Precision() == QUDA_SINGLE_PRECISION){
        SpinorTexture<float2, float2, float2, 3, 0> src_tex(src);
        Spinor<float2, float2, float2, 3> dst_spinor(dst);
        CropCuda<float2, 3, Spinor<float2, float2, float2, 3>,
          SpinorTexture<float2, float2, float2, 3, 0> >
            crop(dst_spinor, src_tex, dst.Volume(), params, parity);
        crop.apply(*getBlasStream());
      }else if(dst.Precision() == QUDA_HALF_PRECISION){ 
        errorQuda("half precision not yet supported\n");
      }
    }
    return;
  }
*/

} // namespace

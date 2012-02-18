#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <typeinfo>
#include <quda.h>
#include <gauge_field.h>
#include <quda_internal.h>
#include <face_quda.h>
#include <misc_helpers.h>
#include <assert.h>
#include <cuda.h>

#ifdef MPI_COMMS
#include <face_quda.h>
#endif

static double anisotropy_;
extern float fat_link_max_;
static int X_[4];
static QudaTboundary t_boundary_;



#include <pack_gauge.h>
#include <hisq_force_utils.h>

// The following routines are needed to test the hisq fermion force code
// Actually, some of these routines are deprecated, or will be soon, and 
// ought to be removed.
namespace hisq{
  namespace fermion_force{


    static ParityMatrix
      allocateParityMatrix(const int X[4], QudaPrecision precision, bool compressed)
      {
        if(precision == QUDA_DOUBLE_PRECISION && compressed) {
          printf("Error: double precision not supported for compressed matrix\n");
          exit(0); 
        }

        ParityMatrix ret;

        ret.precision = precision;
        ret.X[0] = X[0]/2;
        ret.volume = X[0]/2;
        for(int d=1; d<4; d++){
          ret.X[d] = X[d];
          ret.volume *= X[d]; 
        } 
        ret.Nc = 3;
        if(compressed){
          // store the matrices in a compressed form => 6 complex numbers per matrix
          // = 12 real number => store as 3 float4s
          ret.length = ret.volume*ret.Nc*4; // => number of real numbers  
        }else{
          // store as float2 or double2
          ret.length = ret.volume*ret.Nc*ret.Nc*2; // => number of real numbers   
        }

        if(precision == QUDA_DOUBLE_PRECISION){
          if(compressed){
            printf("Error allocating compressed matrix in double precision\n");
            exit(0);  
          }
          ret.bytes = ret.length*sizeof(double);
        }
        else { // QUDA_SINGLE_PRECISION
          ret.bytes = ret.length*sizeof(float);
        }

        if(cudaMalloc((void**)&ret.data, ret.bytes) == cudaErrorMemoryAllocation){
          printf("Error  allocating matrix\n");
          exit(0);
        }

        cudaMemset(ret.data, 0, ret.bytes);

        return ret;
      }

    FullMatrix
      createMatQuda(const int X[4], QudaPrecision precision)
      {
        FullMatrix ret;
        ret.even = allocateParityMatrix(X, precision, false); // compressed = false
        ret.odd  = allocateParityMatrix(X, precision, false); // compressed = false
        return ret;
      }


    FullCompMatrix
      createCompMatQuda(const int X[4], QudaPrecision precision)
      {
        FullCompMatrix ret;
        ret.even = allocateParityMatrix(X, precision, true); // compressed = true
        ret.odd  = allocateParityMatrix(X, precision, true); // compressed = true
        return ret;
      }


    static void
      freeParityMatQuda(ParityMatrix parity_mat)
      {
        cudaFree(parity_mat.data);
        parity_mat.data = NULL;
      }


    void
      freeMatQuda(FullMatrix mat)
      {
        freeParityMatQuda(mat.even);
        freeParityMatQuda(mat.odd);
      }


    void
      freeCompMatQuda(FullCompMatrix mat)
      {
        freeParityMatQuda(mat.even);
        freeParityMatQuda(mat.odd);
      }



    static void pack18Oprod(float2 *res, float *g, int dir, int Vh){
      float2 *r = res + dir*9*Vh;
      for(int j=0; j<9; j++){
        r[j*Vh].x = g[j*2+0];
        r[j*Vh].y = g[j*2+1];	
      }
      // r[dir][j][i] // where i is the position index, j labels segments of the su3 matrix, dir is direction
    }


    static void packOprodField(float2 *res, float *oprod, int oddBit, int Vh){
      // gaugeSiteSize = 18
      int gss=18;
      int total_volume = Vh*2;
      for(int dir=0; dir<4; dir++){
        float *op = oprod + dir*total_volume*gss + oddBit*Vh*gss; // prod[dir][0/1][x]
        for (int i=0; i<Vh; i++){
          pack18Oprod(res+i, op+i*gss, dir, Vh);
        }	
      }
    }


    static void packOprodFieldDir(float2 *res, float *oprod, int dir, int oddBit, int Vh){
      int gss=18;
      int total_volume = Vh*2;
      float *op = oprod + dir*total_volume*gss + oddBit*Vh*gss;
      for(int i=0; i<Vh; i++){
        pack18Oprod(res+i, op+i*gss, 0, Vh);
      }
    }


    static ParityOprod
      allocateParityOprod(int *X, QudaPrecision precision)
      {
        if(precision == QUDA_DOUBLE_PRECISION) {
          errorQuda("Error: double precision not supported for compressed matrix\n");
        }

        ParityOprod ret;

        ret.precision = precision;
        ret.X[0] = X[0]/2;
        ret.volume = X[0]/2;
        for(int d=1; d<4; d++){
          ret.X[d] = X[d];
          ret.volume *= X[d];
        }
        ret.Nc = 3;
        ret.length = ret.volume*ret.Nc*ret.Nc*2; // => number of real numbers       

        if(precision == QUDA_DOUBLE_PRECISION){
          ret.bytes = ret.length*sizeof(double);
        }
        else { // QUDA_SINGLE_PRECISION
          ret.bytes = ret.length*sizeof(float);
        }

        // loop over the eight directions 
        for(int dir=0; dir<4; dir++){
          if(cudaMalloc((void**)&(ret.data[dir]), ret.bytes) == cudaErrorMemoryAllocation){
            printf("Error allocating ParityOprod\n");
            exit(0);
          }
          cudaMemset(ret.data[dir], 0, ret.bytes);
        }
        return ret;
      }



    FullOprod
      createOprodQuda(int *X, QudaPrecision precision)
      {
        FullOprod ret;
        ret.even = allocateParityOprod(X, precision);
        ret.odd  = allocateParityOprod(X, precision);
        return ret;
      }




    static void
      copyOprodFromCPUArrayQuda(FullOprod cudaOprod, void *cpuOprod,
          size_t bytes_per_dir, int Vh)
      {
        // Use pinned memory 
        float2 *packedEven, *packedOdd;
        cudaMallocHost(&packedEven, bytes_per_dir); // now
        cudaMallocHost(&packedOdd, bytes_per_dir);
        for(int dir=0; dir<4; dir++){
          packOprodFieldDir(packedEven, (float*)cpuOprod, dir, 0, Vh);
          packOprodFieldDir(packedOdd,  (float*)cpuOprod, dir, 1, Vh);

          cudaMemset(cudaOprod.even.data[dir], 0, bytes_per_dir);
          cudaMemset(cudaOprod.odd.data[dir],  0, bytes_per_dir);
          checkCudaError();

          cudaMemcpy(cudaOprod.even.data[dir], packedEven, bytes_per_dir, cudaMemcpyHostToDevice);
          cudaMemcpy(cudaOprod.odd.data[dir], packedOdd, bytes_per_dir, cudaMemcpyHostToDevice);
          checkCudaError();
        }
        cudaFreeHost(packedEven);
        cudaFreeHost(packedOdd);
      }



    void copyOprodToGPU(FullOprod cudaOprod,
        void  *oprod,
        int half_volume){
      int bytes_per_dir = half_volume*18*sizeof(float);
      copyOprodFromCPUArrayQuda(cudaOprod, oprod, bytes_per_dir, half_volume);
    }


    static void unpack18Oprod(float *h_oprod, float2 *d_oprod, int dir, int Vh) {
      float2 *dg = d_oprod + dir*9*Vh;
      for (int j=0; j<9; j++) {
        h_oprod[j*2+0] = dg[j*Vh].x; 
        h_oprod[j*2+1] = dg[j*Vh].y;
      }
    }

    static void unpackOprodField(float* res, float2* cudaOprod, int oddBit, int Vh){
      int gss=18; 
      int total_volume = Vh*2;
      for(int dir=0; dir<4; dir++){
        float* res_ptr = res + dir*total_volume*gss + oddBit*Vh*gss;
        for(int i=0; i<Vh; i++){
          unpack18Oprod(res_ptr + i*gss, cudaOprod+i, dir, Vh);
        }
      }
    }


    static void 
      fetchOprodFromGPUArraysQuda(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod, size_t bytes, int Vh)
      {
        float2 *packedEven, *packedOdd;
        cudaMallocHost(&packedEven,bytes);
        cudaMallocHost(&packedOdd, bytes);


        cudaMemcpy(packedEven, cudaOprodEven, bytes, cudaMemcpyDeviceToHost);
        checkCudaError();
        cudaMemcpy(packedOdd, cudaOprodOdd, bytes, cudaMemcpyDeviceToHost);
        checkCudaError();

        unpackOprodField((float*)cpuOprod, packedEven, 0, Vh);
        unpackOprodField((float*)cpuOprod, packedOdd,  1, Vh);

        cudaFreeHost(packedEven);
        cudaFreeHost(packedOdd);
      }


    void fetchOprodFromGPU(void *cudaOprodEven,
        void *cudaOprodOdd,
        void *oprod,
        int half_vol){
        int bytes = 4*half_vol*18*sizeof(float);
        fetchOprodFromGPUArraysQuda(cudaOprodEven, cudaOprodOdd, oprod, bytes, half_vol);
    }


    static void 
      loadOprodFromCPUArrayQuda(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod,
          size_t bytes, int Vh)
      {
        // Use pinned memory 
	float2 *packedEven, *packedOdd;
        checkCudaError();

        cudaMallocHost(&packedEven, bytes);
        cudaMallocHost(&packedOdd, bytes);
        checkCudaError();


        packOprodField(packedEven, (float*)cpuOprod, 0, Vh);
        packOprodField(packedOdd,  (float*)cpuOprod, 1, Vh);
        checkCudaError();


        cudaMemset(cudaOprodEven, 0, bytes);
        cudaMemset(cudaOprodOdd, 0, bytes);
        checkCudaError();

        cudaMemcpy(cudaOprodEven, packedEven, bytes, cudaMemcpyHostToDevice);
        checkCudaError();
        cudaMemcpy(cudaOprodOdd, packedOdd, bytes, cudaMemcpyHostToDevice);
        checkCudaError();

        cudaFreeHost(packedEven);
        cudaFreeHost(packedOdd);
      }



    void loadOprodToGPU(void *cudaOprodEven, 
        void *cudaOprodOdd,
        void  *oprod,
        int vol){
        checkCudaError();
        int bytes = 4*vol*18*sizeof(float);

				std::cout << "vol = " << vol << std::endl;
        std::cout << "bytes = " << bytes << std::endl;
        checkCudaError();
        loadOprodFromCPUArrayQuda(cudaOprodEven, cudaOprodOdd, oprod, bytes, vol);
    }



    void allocateOprodFields(void **cudaOprodEven, void **cudaOprodOdd, int vol){
      int bytes = 4*vol*18*sizeof(float);

      if (cudaMalloc((void **)cudaOprodEven, bytes) == cudaErrorMemoryAllocation) {
        errorQuda("Error allocating even outer product field");
      }

      cudaMemset((*cudaOprodEven), 0, bytes);
      checkCudaError();

      if (cudaMalloc((void **)cudaOprodOdd, bytes) == cudaErrorMemoryAllocation) {
        errorQuda("Error allocating odd outer product field");
      }

      cudaMemset((*cudaOprodOdd), 0, bytes);

      checkCudaError();
    }

  } // end namespace fermion_force
} // end namespace hisq

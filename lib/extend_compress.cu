#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>

namespace quda {
  QudaTune getBlasTuning();
  QudaVerbosity getBlasVerbosity();
  cudaStream_t* getBlasStream();

  namespace quda {
#include <texture.h>
   
    static struct {
      int x[QUDA_MAX_DIM];
      int stride;
    } blasConstants;

    template <typename FloatN, int N, typename Input>
      __global__ void copyKernel(Output Y, Input X, int length){
        unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
	unsigned int gridSize = gridDim.x*blockDim.x;

	while(i<length){
          FloatN x[N];
	  X.load(x, i);
	  Y.save(x, i);
	  i += gridSize;
	}
      }

     // Cut Kernel -> takes a larger kernel and cuts off a border region
     __global__ void cutKernel(Output Y, Input X, int length, int dim[4]){

       unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
       unsigned int gridSize = gridDim.x*blockDim.x;

       while(i<length){

       }
     }


  }

} // namespace copy

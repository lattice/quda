/* C. Kallidonis: GPU kernel to perform di-quark contractions.
 * This function takes as input two forward propagators, performs
 * the relevant contractions the stores the result in another 
 * propagator object.
 * November 2017
 */

#include <transfer.h>
#include <quda_internal.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <mpi.h>
#include <interface_qlua_internal.h>

namespace quda {
  
  struct ContractQQArg {
    
    typedef typename colorspinor_mapper<QUDA_REAL,QUDA_Ns,QUDA_Nc>::type Propagator;        

    Propagator pIn1[QUDA_PROP_NVEC];  // Input propagator 1
    Propagator pIn2[QUDA_PROP_NVEC];  // Input propagator 2
    Propagator pOut[QUDA_PROP_NVEC];  // Output propagator 
    
    const int parity;                 // only use this for single parity fields
    const int nParity;                // number of parities we're working on
    const int nFace;                  // hard code to 1 for now
    const int dim[5];                 // full lattice dimensions
    const int commDim[4];             // whether a given dimension is partitioned or not
    const int volumeCB;               // checkerboarded volume
    const int nVec;                   // number of vectors in the propagator (usually 12)

    qudaAPI_ContractId cntrID;        // contract index

  ContractQQArg(ColorSpinorField **propOut, ColorSpinorField **propIn1, ColorSpinorField **propIn2, int parity, contractParam cParam)
  :   parity(parity), nParity(propIn1[0]->SiteSubset()), nFace(1),
      dim{ (3-nParity) * propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3), 1 },      
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(propIn1[0]->VolumeCB()), nVec(cParam.nVec), cntrID(cParam.cntrID)
    {
      
      for(int ivec=0;ivec<nVec;ivec++){
	pIn1[ivec].init(*propIn1[ivec]);
	pIn2[ivec].init(*propIn2[ivec]);
	pOut[ivec].init(*propOut[ivec]);
      }

    }    
  }; //-- Structure definition

  
  //-- Main calculation kernel
#define ContractQQ_macro(cntrIdx, A, B, C, D)				\
  __device__ __host__ inline void performContractQQ ## cntrIdx(ColorSpinor<QUDA_REAL,QUDA_Nc,QUDA_Ns> *out, \
							       ContractQQArg *arg, \
							       int x_cb, \
							       int pty) \
  {									\
									\
    int eps[3][3] = { { 0, 1, 2},					\
		      { 1, 2, 0},					\
		      { 2, 0, 1} };					\
									\
    const int Ns = QUDA_Ns;						\
    const int Nc = QUDA_Nc;						\
									\
    typedef ColorSpinor<QUDA_REAL,Nc,Ns> Vector;			\
    Vector In11,In12;							\
    Vector In21,In22;							\
    									\
    const int their_spinor_parity = (arg->nParity == 2) ? 1-pty : 0;	\
  									\
    complex<QUDA_REAL> accum;						\
    for(int p_a = 0; p_a < QUDA_Nc; p_a++){				\
      int i1 = eps[p_a][0];						\
      int j1 = eps[p_a][1];						\
      int k1 = eps[p_a][2];						\
      for (int p_b = 0; p_b < QUDA_Nc; p_b++){				\
	int i2 = eps[p_b][0];						\
    	int j2 = eps[p_b][1];						\
	int k2 = eps[p_b][2];						\
    	for (int a = 0; a < QUDA_Ns; a++){				\
    	  for (int b = 0; b < QUDA_Ns; b++){				\
	    accum = 0.0;						\
	    for (int c = 0; c < QUDA_Ns; c++){				\
	      In11 = arg->pIn1[(B)+Ns*i2](x_cb, their_spinor_parity);	\
	      In12 = arg->pIn1[(B)+Ns*j2](x_cb, their_spinor_parity);	\
	      In21 = arg->pIn2[(D)+Ns*i2](x_cb, their_spinor_parity);	\
	      In22 = arg->pIn2[(D)+Ns*j2](x_cb, their_spinor_parity);	\
									\
  	      accum += In11.data[i1+Nc*(A)] * In22.data[j1+Nc*(C)];	\
	      accum -= In12.data[i1+Nc*(A)] * In21.data[j1+Nc*(C)];	\
	      accum -= In11.data[j1+Nc*(A)] * In22.data[i1+Nc*(C)];	\
	      accum += In12.data[j1+Nc*(A)] * In21.data[i1+Nc*(C)];	\
	    }								\
	    out[b+Ns*k1].assign(accum, Nc, k2, a);			\
	  }								\
	}								\
      }									\
    }									\
  } //-- function closes
  
  ContractQQ_macro(12, c,c,a,b);
  ContractQQ_macro(13, c,a,c,b);
  ContractQQ_macro(14, c,a,b,c);
  ContractQQ_macro(23, a,c,c,b);
  ContractQQ_macro(24, a,c,b,c);
  ContractQQ_macro(34, a,b,c,c);
#undef ContractQQ_macro
  

  __device__ __host__ inline void computeContractQQ(ContractQQArg *arg, int x_cb, int pty){
    
    typedef ColorSpinor<QUDA_REAL,QUDA_Nc,QUDA_Ns> Vector;
    Vector out[QUDA_PROP_NVEC];
    
    switch(arg->cntrID){
    case cntr12:
      performContractQQ12(out, arg, x_cb, pty);
      break;
    case cntr13:
      performContractQQ13(out, arg, x_cb, pty);
      break;
    case cntr14:
      performContractQQ14(out, arg, x_cb, pty);
      break;
    case cntr23:
      performContractQQ23(out, arg, x_cb, pty);
      break;
    case cntr24:
      performContractQQ24(out, arg, x_cb, pty);
      break;
    case cntr34:
      performContractQQ34(out, arg, x_cb, pty);
      break;
    case cntr_INVALID: // Added it just to avoid the compilation warning, check has already been made
      break;
    }    

    for(int ivec=0;ivec<arg->nVec;ivec++){
      arg->pOut[ivec](x_cb,pty) = out[ivec];
    }
    
  }
  
  
  //-- CPU kernel for performing the contractions
  void ContractQQ_CPU(ContractQQArg arg){    
    for (int parity= 0; parity < arg.nParity; parity++){
      parity = (arg.nParity == 2) ? parity : arg.parity;
      
      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++){
	computeContractQQ(&arg,x_cb,parity);
      }
    }
  }

  //-- GPU kernel for performing the contractions
  __global__ void ContractQQ_GPU(ContractQQArg *arg_dev){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (x_cb >= arg_dev->volumeCB) return;
    if (parity >= arg_dev->nParity) return;
    parity = (arg_dev->nParity == 2) ? parity : arg_dev->parity;

    computeContractQQ(arg_dev, x_cb, parity);
  }    
  

  //-- Class definition
  class ContractQQ : public TunableVectorY {

  protected:
    ContractQQArg &arg;
    ContractQQArg *arg_dev;
    const ColorSpinorField &meta;
    
    long long flops() const{
      return QUDA_Nc*QUDA_Nc*QUDA_Ns*QUDA_Ns*QUDA_Ns*4*(3+2)*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const{
      return meta.Bytes() + QUDA_Nc*QUDA_Ns*2*meta.Bytes();
    }
    
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }
    unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock / arg.nParity; }
    
  public:
  ContractQQ(ContractQQArg &arg, ContractQQArg *arg_dev, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), arg_dev(arg_dev), meta(meta)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
    }
    virtual ~ContractQQ() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        double t1 = MPI_Wtime();
	ContractQQ_CPU(arg);
	double t2 = MPI_Wtime();
	printfQuda("TIMING - ContractQQ_CPU: Done in %.6f sec\n", t2-t1);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        double t1 = MPI_Wtime();
	ContractQQ_GPU <<<tp.grid,tp.block,tp.shared_bytes,stream>>> (arg_dev);
        double t2 = MPI_Wtime();
        printfQuda("TIMING - ContractQQ_GPU: Done in %.6f sec\n", t2-t1);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  }; //-- Class definition


  
  //-- Top level function, called within interface-qlua
  void cudaContractQQ(ColorSpinorField **propOut, ColorSpinorField **propIn1, ColorSpinorField **propIn2, int parity, int Nc, int Ns, contractParam cParam){
    
    if(Nc != QUDA_Nc) errorQuda("cudaContractQQ: Supports only Ncolor = %d. Got Nc = %d\n", QUDA_Nc, Nc);
    if(Ns != QUDA_Ns) errorQuda("cudaContractQQ: Supports only Nspin = %d.  Got Ns = %d\n", QUDA_Ns, Ns);
    if(cParam.nVec != QUDA_PROP_NVEC) errorQuda("cudaContractQQ: Supports only nVec = %d.  Got nVec = %d\n", QUDA_PROP_NVEC, cParam.nVec);
    
    ContractQQArg arg(propOut, propIn1, propIn2, parity, cParam);
    
    ContractQQArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(ContractQQArg) );
    cudaMemcpy(arg_dev, &arg, sizeof(ContractQQArg), cudaMemcpyHostToDevice);
    
    ContractQQ contract(arg, arg_dev, *propIn1[0]);
    contract.apply(0);

  }

  
} //-- namespace quda

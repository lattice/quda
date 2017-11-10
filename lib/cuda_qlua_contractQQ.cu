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
#include <interface_qlua.h>

namespace quda {

  struct ContractQQArg {

    ColorSpinorField &propOut;        // output propagator
    const ColorSpinorField &propIn1;  // input propagator 1
    const ColorSpinorField &propIn2;  // input propagator 2

    double **propElemOut;             // Elements of output propagator
    const double **propElemIn1;       // Elements of input propagator 1
    const double **propElemIn2;       // Elements of input propagator 2

    const int parity;                 // only use this for single parity fields
    const int nParity;                // number of parities we're working on
    const int nFace;                  // hard code to 1 for now
    const int dim[5];                 // full lattice dimensions
    const int commDim[4];             // whether a given dimension is partitioned or not
    const int volumeCB;               // checkerboarded volume
    const int nVec;                   // number of vectors in the propagator (usually 12)

    qudaAPI_ContractId cntrID;        // contract index
    
  ContractQQArg(ColorSpinorField &propOut, const ColorSpinorField &propIn1, const ColorSpinorField &propIn2, int parity, contractParam cParam)
  :   propOut(propOut), propIn1(propIn1), propIn2(propIn2),
      parity(parity), nParity((&propIn1)[0].SiteSubset()), nFace(1),
      dim{ (3-nParity) * (&propIn1)[0].X(0), (&propIn1)[0].X(1), (&propIn1)[0].X(2), (&propIn1)[0].X(3), 1 },      
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB((&propIn1)[0].VolumeCB()), nVec(cParam.nVec), cntrID(cParam.cntrID)
    {
      for(int ivec=0;ivec<nVec;ivec++){
	propElemIn1[ivec] = (double*)(&propIn1)[ivec].V();
	propElemIn2[ivec] = (double*)(&propIn2)[ivec].V();
      }
    }
  }; //-- Structure definition

  
  //-- Main calculation kernel
#define ContractQQ_macro(cntrIdx, A, B, C, D)				\
  __device__ __host__ inline void performContractQQ ## cntrIdx(ContractQQArg &arg, \
							       int crd,	\
							       int parity) \
  {									\
    int eps[3][3] = { { 0, 1, 2},					\
		      { 1, 2, 0},					\
		      { 2, 0, 1} };					\
    									\
    int lV = arg.volumeCB;						\
    									\
    for(int p_a = 0; p_a < QUDA_Nc; p_a++){				\
      int i1 = eps[p_a][0];						\
      int j1 = eps[p_a][1];						\
      int k1 = eps[p_a][2];						\
      for (int p_b = 0; p_b < QUDA_Nc; p_b++){				\
	int i2 = eps[p_b][0];						\
    	int j2 = eps[p_b][1];						\
	int k2 = eps[p_b][2];						\
    	int a, b, c;							\
    	for (a = 0; a < QUDA_Ns; a++){					\
    	  for (b = 0; b < QUDA_Ns; b++){				\
	    QLUA_COMPLEX accum = {.re = 0.0, .im = 0.0};		\
	    for (c = 0; c < QUDA_Ns; c++){				\
	      accum = CADD(accum,					\
			   CMUL(PROP_ELEM(arg.propElemIn1, lV, crd, i1,(A),i2,(B)), \
				PROP_ELEM(arg.propElemIn2, lV, crd, j1,(C),j2,(D)))); \
	      								\
  	      accum = CSUB(accum,					\
	      		   CMUL(PROP_ELEM(arg.propElemIn1, lV, crd, j1,(A),i2,(B)), \
	      			PROP_ELEM(arg.propElemIn2, lV, crd, j1,(C),i2,(D)))); \
	      								\
    	      accum = CSUB(accum,					\
	      		   CMUL(PROP_ELEM(arg.propElemIn1, lV, crd, j1,(A),i2,(B)), \
	      			PROP_ELEM(arg.propElemIn2, lV, crd, i1,(C),j2,(D)))); \
	      								\
    	      accum = CADD(accum,					\
	      		   CMUL(PROP_ELEM(arg.propElemIn1, lV, crd, j1,(A),j2,(B)), \
	      			PROP_ELEM(arg.propElemIn2, lV, crd, i1,(C),i2,(D)))); \
	    }								\
	    arg.propElemOut[k1+QUDA_Nc*b][0+2*crd+2*lV*k2+2*lV*QUDA_Nc*a] = accum.re; \
	    arg.propElemOut[k1+QUDA_Nc*b][1+2*crd+2*lV*k2+2*lV*QUDA_Nc*a] = accum.im; \
	  }								\
	}								\
      }									\
    }									\
    									\
  } //-- function closes

  ContractQQ_macro(12, c,c,a,b);
  ContractQQ_macro(13, c,a,c,b);
  ContractQQ_macro(14, c,a,b,c);
  ContractQQ_macro(23, a,c,c,b);
  ContractQQ_macro(24, a,c,b,c);
  ContractQQ_macro(34, a,b,c,c);

#undef ContractQQ_macro
  
  
  //-- CPU kernel for performing the contractions
  void ContractQQ_CPU(ContractQQArg arg){
    
    switch(arg.cntrID){
    case cntr12:
      for (int parity= 0; parity < arg.nParity; parity++){
	parity = (arg.nParity == 2) ? parity : arg.parity;
	
	for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++)
	  performContractQQ12(arg,x_cb,parity);
      }
      break;
    case cntr13:
      for (int parity= 0; parity < arg.nParity; parity++){
	parity = (arg.nParity == 2) ? parity : arg.parity;
	
	for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++)
	  performContractQQ13(arg,x_cb,parity);
      }
      break;
    case cntr14:
      for (int parity= 0; parity < arg.nParity; parity++){
	parity = (arg.nParity == 2) ? parity : arg.parity;
	
	for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++)
	  performContractQQ14(arg,x_cb,parity);
      }
      break;
    case cntr23:
      for (int parity= 0; parity < arg.nParity; parity++){
	parity = (arg.nParity == 2) ? parity : arg.parity;
	
	for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++)
	  performContractQQ23(arg,x_cb,parity);
      }
      break;
    case cntr24:
      for (int parity= 0; parity < arg.nParity; parity++){
	parity = (arg.nParity == 2) ? parity : arg.parity;
	
	for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++)
	  performContractQQ24(arg,x_cb,parity);
      }
      break;
    case cntr34:
      for (int parity= 0; parity < arg.nParity; parity++){
	parity = (arg.nParity == 2) ? parity : arg.parity;
	
	for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++)
	  performContractQQ34(arg,x_cb,parity);
      }
      break;
    case cntr_INVALID: // Added it just to avoid the compilation warning, check has already been made
      break;
    }    
  }//-- CPU kernel

  //-- GPU kernel for performing the contractions
  __global__ void ContractQQ_GPU(ContractQQArg arg){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;
    parity = (arg.nParity == 2) ? parity : arg.parity;
    
    switch(arg.cntrID){
    case cntr12:
      performContractQQ12(arg, x_cb, parity);
      break;
    case cntr13:
      performContractQQ13(arg, x_cb, parity);
      break;
    case cntr14:
      performContractQQ14(arg, x_cb, parity);
      break;
    case cntr23:
      performContractQQ23(arg, x_cb, parity);
      break;
    case cntr24:
      performContractQQ24(arg, x_cb, parity);
      break;
    case cntr34:
      performContractQQ34(arg, x_cb, parity);
      break;
    case cntr_INVALID: // Added it just to avoid the compilation warning, check has already been made
      break;
    }    
  }//-- GPU kernel
    
  
  //-- Class definition
  class ContractQQ : public TunableVectorY {

  protected:
    ContractQQArg &arg;
    const ColorSpinorField &meta;

    //-- FIXME: Fix the numbers
    long long flops() const{
      return (2*3*QUDA_Ns*QUDA_Nc*(8*QUDA_Nc-2) + 2*3*QUDA_Nc*QUDA_Ns )*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const{
      return 0; //arg.out.Bytes() + (2*3+1)*arg.in.Bytes() + arg.nParity*2*3*arg.U.Bytes()*meta.VolumeCB();
    }
    
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }
    unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock / arg.nParity; }
    
  public:
    ContractQQ(ContractQQArg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
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
	ContractQQ_GPU <<<tp.grid,tp.block,tp.shared_bytes,stream>>> (arg);
        double t2 = MPI_Wtime();
        printfQuda("TIMING - ContractQQ_GPU: Done in %.6f sec\n", t2-t1);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  }; //-- Class definition


  //-- Top level function, called within interface-qlua
  void cudaContractQQ(ColorSpinorField &propOut, ColorSpinorField &propIn1, ColorSpinorField &propIn2, int Nc, int Ns, contractParam cParam){
    
    if(Nc != QUDA_Nc) errorQuda("cudaContractQQ: Supports only Ncolor = %d. Got Nc = %d\n", QUDA_Nc, Nc);
    if(Ns != QUDA_Ns) errorQuda("cudaContractQQ: Supports only Nspin = %d. Got Ns = %d\n", QUDA_Ns, Ns);
    
    int parity = 0;

    ContractQQArg arg(propOut, propIn1, propIn2, parity, cParam);
    ContractQQ contract(arg, (&propIn1)[0]);
    contract.apply(0);
    
  } //-- cudaContractQQ

  
} //-- namespace quda

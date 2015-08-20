namespace quda {
#ifdef GPU_CONTRACT

#include "contract_core.h"
#include "contract_core_plus.h"
#include "contract_core_minus.h"

#ifndef	_TWIST_QUDA_CONTRACT
#error	"Contraction core undefined"
#endif

#ifndef	_TWIST_QUDA_CONTRACT_PLUS
#error	"Contraction core (plus) undefined"
#endif

#ifndef	_TWIST_QUDA_CONTRACT_MINUS
#error	"Contraction core (minus) undefined"
#endif

#define checkSpinor(a, b)						\
  {									\
    if	(a.Precision() != b.Precision())				\
      errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision()); \
    if	(a.Length() != b.Length())					\
      errorQuda("lengths do not match: %d %d", a.Length(), b.Length());	\
    if	(a.Stride() != b.Stride())					\
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());	\
  }

  /**
     Class for the contract kernels, Float2 is the typename of the spinor components (double2, float4...)
     whereas rFloat is the typename of the precision (double, float...)
  */

  template <typename Float2, typename rFloat>
  class ContractCuda : public Tunable {

  private:
    const cudaColorSpinorField x;		// Spinor to be contracted
    const cudaColorSpinorField y;		// Spinor to be contracted
    const QudaParity parity;		// Parity of the field, actual kernels act on parity spinors
    const QudaContractType contract_type;	// Type of contraction, to be detailed later

    /**
       The result of the contraction is stored in a double2 or float2 array, whose size is Volume x 16. This array
       must be reserved in the GPU device BEFORE calling the contract functions, and must have the appropiate size.
       What is stored is the contraction per point and per gamma index. Since each spinor have a gamma index that
       runs from 1 to 4, the result is a 4x4 matrix--> x_\mu y_\nu = result_{\mu\nu}, so 16 components are stored
       per point. The order is straightforward:

       (  1   2   3   4  )
       (  5   6   7   8  )
       (  9  10  11  12  )
       ( 13  14  15  16  )

       With these 16 components one can reconstruct any gamma structure G for the contraction x_\mu G_@s[\mu\nu} y_\nu.

       The ordering of the points at the output is such that FFT can be performed just by plugin the output array in
       the functions of the cuFFT library.

    */

    void *result;				// The output array with the result of the contraction

    const int nTSlice;			// Time-slice in case of time-dilution

    char aux[16][TuneKey::aux_n];			// For tuning purposes

    unsigned int sharedBytesPerThread() const { return 16*sizeof(rFloat); }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return x.X(0) * x.X(1) * x.X(2) * x.X(3); }

    char *saveOut, *saveOutNorm;

    void fillAux(QudaContractType contract_type, const char *contract_str) { strcpy(aux[contract_type], contract_str); }

  public:
    ContractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaParity parity, const QudaContractType contract_type) :
      x(x), y(y), result(result), parity(parity), contract_type(contract_type), nTSlice(-1) {
      fillAux(QUDA_CONTRACT, "type=plain");
      fillAux(QUDA_CONTRACT_PLUS, "type=plain-plus");
      fillAux(QUDA_CONTRACT_MINUS, "type=plain-minus");
      fillAux(QUDA_CONTRACT_GAMMA5, "type=gamma5");
      fillAux(QUDA_CONTRACT_GAMMA5_PLUS, "type=gamma5-plus");
      fillAux(QUDA_CONTRACT_GAMMA5_MINUS, "type=gamma5-minus");
      fillAux(QUDA_CONTRACT_TSLICE, "type=tslice");
      fillAux(QUDA_CONTRACT_TSLICE_PLUS, "type=tslice-plus");
      fillAux(QUDA_CONTRACT_TSLICE_MINUS, "type=tslice-minus");

      bindSpinorTex<Float2>(&x, &y);
    }

    ContractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaParity parity, const QudaContractType contract_type, const int tSlice) :
      x(x), y(y), result(result), parity(parity), contract_type(contract_type), nTSlice(tSlice) {
      fillAux(QUDA_CONTRACT, "type=plain");
      fillAux(QUDA_CONTRACT_PLUS, "type=plain-plus");
      fillAux(QUDA_CONTRACT_MINUS, "type=plain-minus");
      fillAux(QUDA_CONTRACT_GAMMA5, "type=gamma5");
      fillAux(QUDA_CONTRACT_GAMMA5_PLUS, "type=gamma5-plus");
      fillAux(QUDA_CONTRACT_GAMMA5_MINUS, "type=gamma5-minus");
      fillAux(QUDA_CONTRACT_TSLICE, "type=tslice");
      fillAux(QUDA_CONTRACT_TSLICE_PLUS, "type=tslice-plus");
      fillAux(QUDA_CONTRACT_TSLICE_MINUS, "type=tslice-minus");

      bindSpinorTex<Float2>(&x, &y);
    }

    virtual ~ContractCuda() { unbindSpinorTex<Float2>(&x, &y); } // if (tSlice != NULL) { cudaFreeHost(tSlice); } }

    QudaContractType ContractType() const { return contract_type; }

    TuneKey tuneKey() const
    {
      return TuneKey(x.VolString(), typeid(*this).name(), aux[contract_type]);
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch	(contract_type)
	{
	default:
	case	QUDA_CONTRACT_GAMMA5:		// Calculates the volume contraction (x^+ g5)_\mu y_\nu and stores it in result
	  contractGamma5Kernel     <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
	  break;
	
	case	QUDA_CONTRACT_GAMMA5_PLUS:	// Calculates the volume contraction (x^+ g5)_\mu y_\nu and adds it to result
	  contractGamma5PlusKernel <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
	  break;

	case	QUDA_CONTRACT_GAMMA5_MINUS:	// Calculates the volume contraction (x^+ g5)_\mu y_\nu and substracts it from result
	  contractGamma5MinusKernel<<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
	  break;

	case	QUDA_CONTRACT:			// Calculates the volume contraction x^+_\mu y_\nu and stores it in result
	  contractKernel    	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
	  break;                                                  
	                                                                                
	case	QUDA_CONTRACT_PLUS:		// Calculates the volume contraction x^+_\mu y_\nu and adds it to result
	  contractPlusKernel	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
	  break;                                                  
                                                                                        
	case	QUDA_CONTRACT_MINUS:		// Calculates the volume contraction x^+_\mu y_\nu and substracts it from result
	  contractMinusKernel	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
	  break;

	case	QUDA_CONTRACT_TSLICE:		// Calculates the time-slice contraction x^+_\mu y_\nu and stores it in result
	  contractTsliceKernel   	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), nTSlice, parity, dslashParam);
	  break;                                                  
	                                                                                
	case	QUDA_CONTRACT_TSLICE_PLUS:	// Calculates the time-slice contraction x^+_\mu y_\nu and adds it to result
	  contractTslicePlusKernel <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), nTSlice, parity, dslashParam);
	  break;                                                  
                                                                                        
	case	QUDA_CONTRACT_TSLICE_MINUS:	// Calculates the time-slice contraction x^+_\mu y_\nu and substracts it from result
	  contractTsliceMinusKernel<<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), nTSlice, parity, dslashParam);
	  break;
	}
    }

    void preTune()	{}

    void postTune()	{}

    std::string paramString(const TuneParam &param) const
    {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 120ll * x.VolumeCB(); }
    long long bytes() const { return x.Bytes() + x.NormBytes() + y.Bytes() + y.NormBytes(); }
  };
#endif

  /**
     Contracts the x and y spinors (x is daggered) and stores the result in the array result.
     One must specify the contract type (time-sliced or volumed contract, and whether we should
     include a gamma5 in the middle), as well as the time-slice (see overloaded version of the
     same function) in case we don't want a volume contraction. The function works only with
     parity spinors, and the parity must be specified.
  */

  void	contractCuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const QudaParity parity, TimeProfile &profile)
  {
#ifdef GPU_CONTRACT
    if	((contract_type == QUDA_CONTRACT_TSLICE) || (contract_type == QUDA_CONTRACT_TSLICE_PLUS) || (contract_type == QUDA_CONTRACT_TSLICE_MINUS)) {
      errorQuda("No time-slice specified for contraction\n");
      return;
    }

    profile.TPSTART(QUDA_PROFILE_TOTAL);
    profile.TPSTART(QUDA_PROFILE_INIT);

    dslashParam.threads = x.Volume();

    Tunable *contract = 0;

    if		(x.Precision() == QUDA_DOUBLE_PRECISION)
      {
#if (__COMPUTE_CAPABILITY__ >= 130)
	contract = new ContractCuda<double2,double2>(x, y, result, parity, contract_type);
#else
	errorQuda("Double precision not supported on this GPU");
#endif
      } else if	(x.Precision() == QUDA_SINGLE_PRECISION) {
      contract = new ContractCuda<float4,float2>(x, y, result, parity, contract_type);
    } else if	(x.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported for gamma5 kernel yet");
    }
    profile.TPSTOP(QUDA_PROFILE_INIT);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    contract->apply(streams[Nstream-1]);
    cudaStreamSynchronize(streams[Nstream-1]);
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);
    checkCudaError();

    delete contract;

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
#else
    errorQuda("Contraction code has not been built");
#endif
  }

  /**
     Contracts the x and y spinors (x is daggered) and stores the result in the array result.
     One must specify the contract type (time-sliced or volumed contract, and whether we should
     include a gamma5 in the middle), as well as the time-slice in case we don't want a volume
     contraction. The function works only with parity spinors, and the parity must be specified.
  */

  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type,
		    const int nTSlice, const QudaParity parity, TimeProfile &profile)
  {
#ifdef GPU_CONTRACT
    if	((contract_type != QUDA_CONTRACT_TSLICE) || (contract_type != QUDA_CONTRACT_TSLICE_PLUS) || (contract_type != QUDA_CONTRACT_TSLICE_MINUS)) {
      errorQuda("No time-slice input allowed for volume contractions\n");
      return;
    }

    profile.TPSTART(QUDA_PROFILE_TOTAL);
    profile.TPSTART(QUDA_PROFILE_INIT);

    dslashParam.threads = x.X(0)*x.X(1)*x.X(2);

    Tunable *contract = 0;

    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      contract = new ContractCuda<double2,double2>(x, y, result, parity, contract_type, nTSlice);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      contract = new ContractCuda<float4,float2>(x, y, result, parity, contract_type, nTSlice);
    } else if (x.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported for gamma5 kernel yet");
    }
    profile.TPSTOP(QUDA_PROFILE_INIT);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    contract->apply(streams[Nstream-1]);
    cudaStreamSynchronize(streams[Nstream-1]);
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);
    checkCudaError();
    delete contract;

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);
#else
    errorQuda("Contraction code has not been built");
#endif
  }
  
} // namespace quda


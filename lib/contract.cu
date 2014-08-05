//
// double2 contractCuda(float2 *x, float2 *y, float2 *result) {}
//

namespace quda
{
	#include <gamma5.h>		// g5 kernel

	template <typename sFloat>
	class Gamma5Cuda : public Tunable {

	private:
		cudaColorSpinorField *out;
		const cudaColorSpinorField *in;

		unsigned int sharedBytesPerThread() const { return 0; }
		unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
		bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
		unsigned int minThreads() const { return dslashConstants.VolumeCB(); }

		char *saveOut, *saveOutNorm;
		char auxStr[8];

	public:
		Gamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in) :
		out(out), in(in) { bindSpinorTex<sFloat>(in, out); strcpy(aux,"gamma5");}

		virtual ~Gamma5Cuda() { unbindSpinorTex<sFloat>(in, out); }

		TuneKey tuneKey() const
		{
			return TuneKey(in->VolString(), typeid(*this).name(), auxStr);
		}

		void apply(const cudaStream_t &stream)
		{
			TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
			gamma5Kernel<<<tp.grid, tp.block, tp.shared_bytes>>> ((sFloat*)out->V(), (float*)out->Norm(), (sFloat*)in->V(), (float*)in->Norm(), dslashParam, in->Stride());
		}

		void preTune()
		{
			saveOut = new char[out->Bytes()];
			cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);

			if (typeid(sFloat) == typeid(short4))
			{
				saveOutNorm = new char[out->NormBytes()];
				cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
			}
		}

		void postTune()
		{
			cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
			delete[] saveOut;

			if (typeid(sFloat) == typeid(short4))
			{
				cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
				delete[] saveOutNorm;
			}
		}

		std::string paramString(const TuneParam &param) const
		{
			std::stringstream ps;
			ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
			ps << "shared=" << param.shared_bytes;
			return ps.str();
		}

		long long flops() const { return 12ll * dslashConstants.VolumeCB(); }
		long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
	};

	void	gamma5Cuda	(cudaColorSpinorField *out, const cudaColorSpinorField *in)
	{
		dslashParam.threads = in->Volume();

		Tunable *gamma5 = 0;

		if		(in->Precision() == QUDA_DOUBLE_PRECISION)
		{
			#if (__COMPUTE_CAPABILITY__ >= 130)
				gamma5 = new Gamma5Cuda<double2>(out, in);
			#else
				errorQuda("Double precision not supported on this GPU");
			#endif
		} else if	(in->Precision() == QUDA_SINGLE_PRECISION) {
			gamma5 = new Gamma5Cuda<float4>(out, in);
		} else if	(in->Precision() == QUDA_HALF_PRECISION) {
			errorQuda("Half precision not supported for gamma5 kernel yet");
//			gamma5 = new Gamma5Cuda<short4>(out, in);
		}

		gamma5->apply(streams[Nstream-1]);
		checkCudaError();

		delete gamma5;
	}

/*
	void	gamma5Cuda	(cudaColorSpinorField *out, const cudaColorSpinorField *in, const dim3 &block)
	{
		dslashParam.threads	 = in->Volume();

		if	(in->Precision() == QUDA_DOUBLE_PRECISION)
		{

		#if (__COMPUTE_CAPABILITY__ >= 130)
			dim3 blockDim (block.x, block.y, block.z);
			dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);

			bindSpinorTex<double2>(in); //for multi-gpu usage
			gamma5Kernel<<<gridDim, blockDim, 0>>> ((double2*)out->V(), (float*)out->Norm(), dslashParam, in->Stride());
		#else
			errorQuda("Double precision not supported on this GPU");
		#endif
		}
		else if	(in->Precision() == QUDA_SINGLE_PRECISION)
		{
			dim3 blockDim (block.x, block.y, block.z);
			dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);

			bindSpinorTex<float4>(in); //for multi-gpu usage
			gamma5Kernel<<<gridDim, blockDim, 0>>> ((float4*)out->V(), (float*)out->Norm(), dslashParam, in->Stride());
		}

		cudaDeviceSynchronize	();
		checkCudaError		();
	}
*/

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

	#define checkSpinor(a, b)									\
	{												\
		if	(a.Precision() != b.Precision())						\
			errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision());	\
		if	(a.Length() != b.Length())							\
			errorQuda("lengths do not match: %d %d", a.Length(), b.Length());		\
		if	(a.Stride() != b.Stride())							\
			errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());		\
	}

	template <typename Float2, typename rFloat>
	class ContractCuda : public Tunable {

	private:
		const cudaColorSpinorField x;
		const cudaColorSpinorField y;
		const int parity;		//QudaParity parity;
		const QudaContractType contract_type;
		void *result;

//		int *tSlice;
		const int nTSlice;

		char aux[16][256];

		unsigned int sharedBytesPerThread() const { return 16*sizeof(rFloat); }
		unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
		bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
		unsigned int minThreads() const { return dslashConstants.VolumeCB(); }

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

//		void SetTimeSlice	(const int nTslices, int *tSlices)
/*		void SetTimeSlice	(const int nTslices)
		{
			if	((contract_type != QUDA_CONTRACT_TSLICE) && (contract_type != QUDA_CONTRACT_TSLICE_PLUS) && (contract_type != QUDA_CONTRACT_TSLICE_MINUS))
			{
				errorQuda("Can't set time slice for contraction in a full volume contraction\n");
				return;
			}
/*
			if	(cudaMallocHost(&tSlice, nTslices*sizeof(int)) != cudaSuccess)
			{
				errorQuda("Can't allocate memory to store contraction time-slices (...really?)\n");
				return;
			}
*/
//			nTSlice	= nTslices;
/*
			if	(cudaMemCpy((void*) tSlice, (void*) tSlices, nTslice*sizeof(int), cudaMemcpyHostToHost) != cudaSuccess)
			{
				errorQuda("Can't copy time-slices\n");
				return;
			}
*/
//			return;
//		}	

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
				case	QUDA_CONTRACT_GAMMA5:
				contractGamma5Kernel     <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
				break;
	
				case	QUDA_CONTRACT_GAMMA5_PLUS:
				contractGamma5PlusKernel <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
				break;

				case	QUDA_CONTRACT_GAMMA5_MINUS:
				contractGamma5MinusKernel<<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
				break;

				case	QUDA_CONTRACT:
				contractKernel    	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
				break;                                                  
	                                                                                
				case	QUDA_CONTRACT_PLUS:                             
				contractPlusKernel	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
				break;                                                  
                                                                                        
				case	QUDA_CONTRACT_MINUS:                            
				contractMinusKernel	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), parity, dslashParam);
				break;

				case	QUDA_CONTRACT_TSLICE:
/*				for	(int i=0; i<nTslice; i++) {
					#ifdef	MULTI_GPU			//In MultiGPU we perform the time-slice contraction only if the corresponding rank is storing that time-slice
						if	(comm_dim(3) > 1)
						{
							tO	= comm_coord(3)*x.X(3);
							tF	= t_O + x.X(3);

							if	((tSlice[i] < t_O) || (tSlice[i] > t_F))
								continue;
						}
					#endif*/
					contractTsliceKernel   	 <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), nTSlice, parity, dslashParam);
//				}
				break;                                                  
	                                                                                
				case	QUDA_CONTRACT_TSLICE_PLUS:                             
/*				for	(int i=0; i<nTslice; i++) {
					#ifdef	MULTI_GPU			//In MultiGPU we perform the time-slice contraction only if the corresponding rank is storing that time-slice
						if	(comm_dim(3) > 1)
						{
							tO	= comm_coord(3)*x.X(3);
							tF	= t_O + x.X(3);

							if	((tSlice[i] < t_O) || (tSlice[i] > t_F))
								continue;
						}
					#endif*/
					contractTslicePlusKernel <<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), nTSlice, parity, dslashParam);
//				}
				break;                                                  
                                                                                        
				case	QUDA_CONTRACT_TSLICE_MINUS:                            
		/*		for	(int i=0; i<nTslice; i++) {
					#ifdef	MULTI_GPU			//In MultiGPU we perform the time-slice contraction only if the corresponding rank is storing that time-slice
						if	(comm_dim(3) > 1)
						{
							tO	= comm_coord(3)*x.X(3);
							tF	= t_O + x.X(3);

							if	((tSlice[i] < t_O) || (tSlice[i] > t_F))
								continue;
						}
					#endif*/
					contractTsliceMinusKernel<<<tp.grid, tp.block, tp.shared_bytes>>>((rFloat*)result, (Float2*)x.V(), (Float2*)y.V(), x.Stride(), nTSlice, parity, dslashParam);
//				}
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

		long long flops() const { return 120ll * dslashConstants.VolumeCB(); }
		long long bytes() const { return x.Bytes() + x.NormBytes() + y.Bytes() + y.NormBytes(); }
	};

	void	contractCuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const QudaParity parity)
	{
		if	((contract_type == QUDA_CONTRACT_TSLICE) || (contract_type == QUDA_CONTRACT_TSLICE_PLUS) || (contract_type == QUDA_CONTRACT_TSLICE_MINUS)) {
			errorQuda("No time-slice specified for contraction\n");
			return;
		}

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
//			gamma5 = new Gamma5Cuda<short4>(out, in);
		}

		contract->apply(streams[Nstream-1]);
		checkCudaError();

		delete contract;
	}

	void	contractCuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const int nTSlice, const QudaParity parity) //int *tSlices, const int parity)
	{
		if	((contract_type != QUDA_CONTRACT_TSLICE) || (contract_type != QUDA_CONTRACT_TSLICE_PLUS) || (contract_type != QUDA_CONTRACT_TSLICE_MINUS)) {
			errorQuda("No time-slice input allowed for volume contractions\n");
			return;
		}

		dslashParam.threads = x.X(0)*x.X(1)*x.X(2);

		Tunable *contract = 0;

		if		(x.Precision() == QUDA_DOUBLE_PRECISION)
		{
			#if (__COMPUTE_CAPABILITY__ >= 130)
				contract = new ContractCuda<double2,double2>(x, y, result, parity, contract_type, nTSlice);
			#else
				errorQuda("Double precision not supported on this GPU");
			#endif
		} else if	(x.Precision() == QUDA_SINGLE_PRECISION) {
			contract = new ContractCuda<float4,float2>(x, y, result, parity, contract_type, nTSlice);
		} else if	(x.Precision() == QUDA_HALF_PRECISION) {
			errorQuda("Half precision not supported for gamma5 kernel yet");
//			gamma5 = new Gamma5Cuda<short4>(out, in);
		}

		contract->apply(streams[Nstream-1]);
		checkCudaError();

		delete contract;
	}

/*
	void	contractGamma5Cuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Parity)
	{
		dim3 gridDim((x.Volume() - 1) / blockDim.x + 1, 1, 1);	//CHANGE FOR MULTI_GPU

		checkSpinor(x,y);

		if	(x.Precision() == QUDA_HALF_PRECISION)
			errorQuda("Error: Half precision not supported");

		if	(x.Precision() == QUDA_DOUBLE_PRECISION)
		{
			bindSpinorTex<double2>(&x, &y);

			switch	(sign)
			{
				default:
				case	0:
				contractGamma5Kernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractGamma5PlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractGamma5MinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}
		}
		else if	(x.Precision() == QUDA_SINGLE_PRECISION)
		{
			bindSpinorTex<float4>(&x, &y);

			switch	(sign)
			{
				default:
				case	0:
				contractGamma5Kernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractGamma5PlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractGamma5MinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}
		}

		cudaDeviceSynchronize	();
		checkCudaError		();

		return;
	}
*/
/*
	void	contractTsliceCuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Tslice, const int Parity)
	{
	        const int tVolume = x.X(0)*x.X(1)*x.X(2);//!half volume (check it!), i.e., x.Volume() / x.X(3);
		dim3 gridDim((tVolume - 1) / blockDim.x + 1, 1, 1);	//CHANGE FOR MULTI_GPU

		checkSpinor(x,y);

		if	(x.Precision() == QUDA_HALF_PRECISION)
			errorQuda("Error: Half precision not supported");

		if	(x.Precision() == QUDA_DOUBLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(double);

				cudaBindTexture(0, spinorTexDouble, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexDouble,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractTsliceKernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
		
				case	1:
				contractTslicePlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;

				case	2:
				contractTsliceMinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexDouble);
				cudaUnbindTexture(interTexDouble);
			#endif
		}

		if	(x.Precision() == QUDA_SINGLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(float);

				cudaBindTexture(0, spinorTexSingle, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexSingle,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractTsliceKernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
		
				case	1:
				contractTslicePlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;

				case	2:
				contractTsliceMinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexSingle);
				cudaUnbindTexture(interTexSingle);
			#endif
		}
		cudaDeviceSynchronize	();
		checkCudaError		();

		return;
	}
/*
	void	contractCuda		(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Parity)
	{
		dim3 gridDim((x.Volume() - 1) / blockDim.x + 1, 1, 1);	//CHANGE FOR MULTI_GPU

		checkSpinor(x,y);

		if	(x.Precision() == QUDA_HALF_PRECISION)
			errorQuda("Error: Half precision not supported");

		if	(x.Precision() == QUDA_DOUBLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(double);

				cudaBindTexture(0, spinorTexDouble, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexDouble,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractKernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractPlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractMinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexDouble);
				cudaUnbindTexture(interTexDouble);
			#endif
		}

		if	(x.Precision() == QUDA_SINGLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(float);

				cudaBindTexture(0, spinorTexSingle, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexSingle,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractKernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractPlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractMinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexSingle);
				cudaUnbindTexture(interTexSingle);
			#endif
		}

		cudaDeviceSynchronize	();
		checkCudaError		();

		return;
	}
*/
}


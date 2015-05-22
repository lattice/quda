#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

#include <device_functions.h>

#include <cufft.h>
#include <CUFFT_Plans.h>



namespace quda {



//Comment if you don't want to use textures for Delta(x) and g(x)
#define GAUGEFIXING_SITE_MATRIX_LOAD_TEX

//UNCOMMENT THIS IF YOU WAN'T TO USE LESS MEMORY
#define GAUGEFIXING_DONT_USE_GX
//Without using the precalculation of g(x), 
//we loose some performance, because Delta(x) is written in normal lattice coordinates need for the FFTs
//and the gauge array in even/odd format




#ifdef GAUGEFIXING_DONT_USE_GX
#warning Don't use precalculated g(x)
#else
#warning Using precalculated g(x)
#endif


#ifndef FL_UNITARIZE_PI
#define FL_UNITARIZE_PI 3.14159265358979323846
#endif


texture<float2, 1, cudaReadModeElementType> GXTexSingle;
texture<int4, 1, cudaReadModeElementType> GXTexDouble;
//Delta is only stored using 12 real number parameters, 
//	(0,0), (0,1), (0,2), (1,1), (1,2) and (2,2)
//	(0,0), (1,1) and (0,1) don't have real part, however we need a complex for the FFTs
texture<float2, 1, cudaReadModeElementType> DELTATexSingle;
texture<int4, 1, cudaReadModeElementType> DELTATexDouble;


template <class T> 
inline __device__ T TEXTURE_GX(int id){
	return 0.0;
}
template <>
inline __device__ typename ComplexTypeId<float>::Type TEXTURE_GX<typename ComplexTypeId<float>::Type>(int id){
	return tex1Dfetch(GXTexSingle, id);
}
template <>
inline __device__ typename ComplexTypeId<double>::Type TEXTURE_GX<typename ComplexTypeId<double>::Type>(int id){
    int4 u = tex1Dfetch(GXTexDouble, id);
    return  makeComplex(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}
template <class T> 
inline __device__ T TEXTURE_DELTA(int id){
	return 0.0;
}
template <>
inline __device__ typename ComplexTypeId<float>::Type TEXTURE_DELTA<typename ComplexTypeId<float>::Type>(int id){
	return tex1Dfetch(DELTATexSingle, id);
}
template <>
inline __device__ typename ComplexTypeId<double>::Type TEXTURE_DELTA<typename ComplexTypeId<double>::Type>(int id){
    int4 u = tex1Dfetch(DELTATexDouble, id);
    return  makeComplex(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}

static void BindTex(typename ComplexTypeId<float>::Type *delta, typename ComplexTypeId<float>::Type *gx, size_t bytes){
	#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
		#ifndef GAUGEFIXING_DONT_USE_GX
		cudaBindTexture(0, GXTexSingle, gx, bytes);
		#endif
	cudaBindTexture(0, DELTATexSingle, delta, bytes);
	#endif
}
static void BindTex(typename ComplexTypeId<double>::Type *delta, typename ComplexTypeId<double>::Type *gx, size_t bytes){
	#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
		#ifndef GAUGEFIXING_DONT_USE_GX
		cudaBindTexture(0, GXTexDouble, gx, bytes);
		#endif
	cudaBindTexture(0, DELTATexDouble, delta, bytes);
	#endif
}

static void UnBindTex(typename ComplexTypeId<float>::Type *delta, typename ComplexTypeId<float>::Type *gx){
	#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
		#ifndef GAUGEFIXING_DONT_USE_GX
		cudaUnbindTexture(GXTexSingle);
		#endif
		cudaUnbindTexture(DELTATexSingle);
	#endif
}
static void UnBindTex(typename ComplexTypeId<double>::Type *delta, typename ComplexTypeId<double>::Type *gx){
	#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
		#ifndef GAUGEFIXING_DONT_USE_GX
		cudaUnbindTexture(GXTexDouble);
		#endif
	cudaUnbindTexture(DELTATexDouble);
	#endif
}


template<class T>
__device__ __host__ inline Matrix<T,3> getSubTraceUnit(const Matrix<T,3>& a){
	T tr = (a(0,0) + a(1,1) + a(2,2)) / 3.0;
	Matrix<T,3> res;
	res(0,0) = a(0,0)- tr; res(0,1) = a(0,1); res(0,2) = a(0,2);
	res(1,0) = a(1,0); res(1,1) = a(1,1)-tr; res(1,2) = a(1,2);
	res(2,0) = a(2,0); res(2,1) = a(2,1); res(2,2) = a(2,2)-tr;
	return res;
}

template<class T>
__device__ __host__ inline void SubTraceUnit(Matrix<T,3>& a){
	T tr = (a(0,0) + a(1,1) + a(2,2)) / 3.0;
	a(0,0)-= tr; a(1,1) -= tr; a(2,2) -= tr;
}

template<class T>
__device__ __host__ inline double getRealTraceUVdagger(const Matrix<T,3>& a, const Matrix<T,3>& b){
	double sum = (double)(a(0,0).x * b(0,0).x  + a(0,0).y * b(0,0).y);
	sum += (double)(a(0,1).x * b(0,1).x  + a(0,1).y * b(0,1).y);
	sum += (double)(a(0,2).x * b(0,2).x  + a(0,2).y * b(0,2).y);
	sum += (double)(a(1,0).x * b(1,0).x  + a(1,0).y * b(1,0).y);
	sum += (double)(a(1,1).x * b(1,1).x  + a(1,1).y * b(1,1).y);
	sum += (double)(a(1,2).x * b(1,2).x  + a(1,2).y * b(1,2).y);
	sum += (double)(a(2,0).x * b(2,0).x  + a(2,0).y * b(2,0).y);
	sum += (double)(a(2,1).x * b(2,1).x  + a(2,1).y * b(2,1).y);
	sum += (double)(a(2,2).x * b(2,2).x  + a(2,2).y * b(2,2).y);
	return sum;
}


static  __inline__ __device__ double atomicAdd(double *addr, double val){
	double old=*addr, assumed;
	do {
		assumed = old;
		old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
				__double_as_longlong(assumed),
				__double_as_longlong(val+assumed)));
	} while( __double_as_longlong(assumed)!=__double_as_longlong(old) );

	return old;
}

static  __inline__ __device__ double2 atomicAdd(double2 *addr, double2 val){
	double2 old=*addr;
	old.x = atomicAdd((double*)addr, val.x);
	old.y = atomicAdd((double*)addr+1, val.y);
	return old;
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200
		//CUDA 6.5 NOT DETECTING ATOMICADD FOR FLOAT TYPE!!!!!!!
static __inline__ __device__ float atomicAdd(float *address, float val)
{
	return __fAtomicAdd(address, val);
}
#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 200 */



template <typename T>
struct Summ {
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b){
		return a + b;
	}
};
template <>
struct Summ<double2>{
	__host__ __device__ __forceinline__ double2 operator()(const double2 &a, const double2 &b){
		return make_double2(a.x+b.x, a.y+b.y);
	}
};




static __device__ __host__ inline int linkIndex3(int x[], int dx[], const int X[4]) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
	int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
	return idx;
}
static __device__ __host__ inline int linkIndex(int x[], const int X[4]) {
	int idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
	return idx;
}
static __device__ __host__ inline int linkIndexM1(int x[], const int X[4], const int mu) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = x[i];
	y[mu] = (y[mu] -1 + X[mu]) % X[mu];
	int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
	return idx;
}

static __device__ __host__ inline int linkIndexP1(int x[], const int X[4], const int mu) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = x[i];
	y[mu] = (y[mu] +1 + X[mu]) % X[mu];
	int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
	return idx;
}


static __device__ __host__ inline void getCoords3(int x[4], int cb_index, const int X[4], int parity) {
	/*x[3] = cb_index/(X[2]*X[1]*X[0]/2);
	x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
	x[1] = (cb_index/(X[0]/2)) % X[1];
	x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);*/
	int za = (cb_index / (X[0]/2));
	int zb =  (za / X[1]);
	x[1] = za - zb * X[1];
	x[3] = (zb / X[2]);
	x[2] = zb - x[3] * X[2];
	int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
	x[0] = (2 * cb_index + x1odd)  - za * X[0];
	return;
}


static __device__ __host__ inline int getCoords(int cb_index, const int X[4], int parity) {
	int za = (cb_index / (X[0]/2));
	int zb =  (za / X[1]);
	int x1 = za - zb * X[1];
	int x3 = (zb / X[2]);
	int x2 = zb - x3 * X[2];
	int x1odd = (x1 + x2 + x3 + parity) & 1;
	return 2 * cb_index + x1odd;
}
















template <typename Cmplx>
struct GaugeFixFFTRotateArg {
	int threads; // number of active threads required
	int X[4]; // grid dimensions
	Cmplx *tmp0;
	Cmplx *tmp1;
	GaugeFixFFTRotateArg(const cudaGaugeField &data){
		for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
		threads = X[0]*X[1]*X[2]*X[3];
		tmp0 = 0;
		tmp1 = 0;
	}
};



template <int direction, typename Cmplx> 
__global__ void fft_rotate_kernel_2D2D(GaugeFixFFTRotateArg<Cmplx> arg){//Cmplx *data_in, Cmplx *data_out){ 
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= arg.threads) return;
	if(direction == 0){
		int x3 = id/(arg.X[0] * arg.X[1] * arg.X[2]);
		int x2 = (id/(arg.X[0] * arg.X[1])) % arg.X[2];
		int x1 = (id/arg.X[0]) % arg.X[1];
		int x0 = id % arg.X[0];

		int id  =  x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0];
		int id_out =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2];   
		arg.tmp1[id_out] = arg.tmp0[id];    
		//data_out[id_out] = data_in[id];
	}
	if(direction==1){ 

		int x1 = id/(arg.X[2] * arg.X[3] * arg.X[0]);
		int x0 = (id/(arg.X[2] * arg.X[3])) % arg.X[0];
		int x3 = (id/arg.X[2]) % arg.X[3];
		int x2 = id % arg.X[2];



		int id  =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2]; 
		int id_out =  x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0];    
		arg.tmp1[id_out] = arg.tmp0[id];    
		//data_out[id_out] = data_in[id];
	}
}






template<typename Cmplx>
class GaugeFixFFTRotate : Tunable {
	GaugeFixFFTRotateArg<Cmplx> arg;
	unsigned int direction;
	mutable char aux_string[128]; // used as a label in the autotuner
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFixFFTRotate(GaugeFixFFTRotateArg<Cmplx> &arg)
: arg(arg) {
		direction = 0;
	}
	~GaugeFixFFTRotate () {}
	void setDirection(unsigned int dir, Cmplx *data_in, Cmplx *data_out){
		direction = dir;
		arg.tmp0 = data_in;
		arg.tmp1 = data_out;
	}

	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		if(direction == 0)
			fft_rotate_kernel_2D2D<0, Cmplx><<<tp.grid, tp.block, 0, stream>>>(arg);
		else if(direction == 1) 
			fft_rotate_kernel_2D2D<1, Cmplx><<<tp.grid, tp.block, 0, stream>>>(arg);
		else 
			errorQuda("Error in GaugeFixFFTRotate option.\n");
	}

	TuneKey tuneKey() const {
		std::stringstream vol;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Cmplx) / 2);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){}
	void postTune(){}
	long long flops() const { return 0; }
	long long bytes() const { return 2LL * sizeof(Cmplx) * arg.threads; } 

}; 










template <typename Cmplx, typename Gauge>
struct GaugeFixQualityArg {
	int threads; // number of active threads required
	int X[4]; // grid dimensions
	Gauge dataOr;
	Cmplx *delta;
	double2 *quality;
	double2 *quality_h;

	GaugeFixQualityArg(const Gauge &dataOr, const cudaGaugeField &data, Cmplx *delta)
	: dataOr(dataOr), delta(delta) {
	//: dataOr(dataOr), delta(delta), quality_h(static_cast<double2*>(pinned_malloc(sizeof(double2)))) {

		for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];

		threads = X[0]*X[1]*X[2]*X[3];
		//cudaHostGetDevicePointer(&quality, quality_h, 0);
	    quality = (double2*)device_malloc(sizeof(double2));
	    quality_h = (double2*)safe_malloc(sizeof(double2));
	}
	double getAction(){return quality_h[0].x;}
	double getTheta(){return quality_h[0].y;}
};



template<int blockSize, unsigned int Elems, typename Float, typename Gauge, int gauge_dir>
__global__ void computeFix_quality(GaugeFixQualityArg<typename ComplexTypeId<Float>::Type, Gauge> argQ){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	typedef cub::BlockReduce<double2, blockSize> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	double2 data = make_double2(0.0,0.0);
	if(idx < argQ.threads) {
		typedef typename ComplexTypeId<Float>::Type Cmplx;
		int parity = 0;
		if(idx >= argQ.threads/2) {
			parity = 1;
			idx -= argQ.threads/2;
		}
		//int X[4]; 
		//for(int dr=0; dr<4; ++dr) X[dr] = argQ.X[dr];
		int x[4];
		getCoords3(x, idx, argQ.X, parity);   
		Matrix<Cmplx,3> delta;
		setZero(&delta);
		//idx = linkIndex(x,X);
		for (int mu = 0; mu < gauge_dir; mu++) { 
			Matrix<Cmplx,3> U; 
			argQ.dataOr.load((Float*)(U.data),idx, mu, parity);
			delta -= U;
		}
		//18*gauge_dir
		data.x = -delta(0,0).x - delta(1,1).x - delta(2,2).x ;
		//2
		for (int mu = 0; mu < gauge_dir; mu++) {
			Matrix<Cmplx,3> U; 
			argQ.dataOr.load((Float*)(U.data),linkIndexM1(x,argQ.X,mu), mu, 1 - parity);
			delta += U;
		}
		//18*gauge_dir
		delta -= conj(delta);
		//18
		//SAVE DELTA!!!!!
		SubTraceUnit(delta);
		idx = getCoords(idx, argQ.X, parity);
		//Saving Delta
		argQ.delta[idx] = delta(0,0);
		argQ.delta[idx + argQ.threads] = delta(0,1);
		argQ.delta[idx + 2 * argQ.threads] = delta(0,2);
		argQ.delta[idx + 3 * argQ.threads] = delta(1,1);
		argQ.delta[idx + 4 * argQ.threads] = delta(1,2);
		argQ.delta[idx + 5 * argQ.threads] = delta(2,2);
		//12
		data.y = getRealTraceUVdagger(delta, delta);
		//35
		//T=36*gauge_dir+65
	}
	double2 aggregate = BlockReduce(temp_storage).Reduce(data, Summ<double2>());
	if (threadIdx.x == 0) atomicAdd(argQ.quality, aggregate);
}



template<unsigned int Elems, typename Float, typename Gauge, int gauge_dir>
class GaugeFixQuality : Tunable {
	GaugeFixQualityArg<typename ComplexTypeId<Float>::Type, Gauge> argQ;
	mutable char aux_string[128]; // used as a label in the autotuner
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return argQ.threads; }

public:
	GaugeFixQuality(GaugeFixQualityArg<typename ComplexTypeId<Float>::Type, Gauge> &argQ)
: argQ(argQ) {}
	~GaugeFixQuality () { host_free(argQ.quality_h);device_free(argQ.quality);}

	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		//argQ.quality_h[0] = make_double2(0.0,0.0);
      	cudaMemset(argQ.quality, 0, sizeof(double2));
		LAUNCH_KERNEL(computeFix_quality, tp, stream, argQ, Elems, Float, Gauge, gauge_dir);
      	cudaMemcpy(argQ.quality_h, argQ.quality, sizeof(double2), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		argQ.quality_h[0].x  /= (double)(3*gauge_dir*argQ.threads);
		argQ.quality_h[0].y  /= (double)(3*argQ.threads);
	}

	TuneKey tuneKey() const {
		std::stringstream vol;
		vol << argQ.X[0] << "x";
		vol << argQ.X[1] << "x";
		vol << argQ.X[2] << "x";
		vol << argQ.X[3];
		sprintf(aux_string,"threads=%d,prec=%d,gaugedir=%d",argQ.threads, sizeof(Float),gauge_dir);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){}
	void postTune(){}
	long long flops() const { return (36LL*gauge_dir+65LL)*argQ.threads; }// Only correct if there is no link reconstruction, no cub reduction accounted also
	long long bytes() const { return (2LL*gauge_dir + 2LL) * Elems * argQ.threads * sizeof(Float); } //Not accounting the reduction!!!

}; 



template <typename Float>
struct GaugeFixArg {
	int threads; // number of active threads required
	int X[4]; // grid dimensions
  	cudaGaugeField &data;
	Float *invpsq;
	typename ComplexTypeId<Float>::Type *delta;
	typename ComplexTypeId<Float>::Type *gx;

	GaugeFixArg( cudaGaugeField &data, const unsigned int Elems) : data(data){
		for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
		threads = X[0]*X[1]*X[2]*X[3];
	    invpsq = (Float*)device_malloc(sizeof(Float) * threads);
	    delta = (typename ComplexTypeId<Float>::Type *)device_malloc(sizeof(typename ComplexTypeId<Float>::Type) * threads * 6);
		#ifdef GAUGEFIXING_DONT_USE_GX
	    gx = (typename ComplexTypeId<Float>::Type *)device_malloc(sizeof(typename ComplexTypeId<Float>::Type) * threads);
		#else
	    gx = (typename ComplexTypeId<Float>::Type *)device_malloc(sizeof(typename ComplexTypeId<Float>::Type) * threads * Elems);
		#endif
		BindTex(delta, gx, sizeof(typename ComplexTypeId<Float>::Type) * threads * Elems);
	}
	void free(){
		UnBindTex(delta, gx);
		device_free(invpsq);
		device_free(delta);
		device_free(gx);
	}
};




template <typename Float> 
__global__ void kernel_gauge_set_invpsq(GaugeFixArg<Float> arg){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= arg.threads) return;
	int x1 = id/(arg.X[2] * arg.X[3] * arg.X[0]);
	int x0 = (id/(arg.X[2] * arg.X[3])) % arg.X[0];
	int x3 = (id/arg.X[2]) % arg.X[3];
	int x2 = id % arg.X[2];
	//id  =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2]; 
	Float sx = sin( (Float)x0 * FL_UNITARIZE_PI / (Float)arg.X[0]);
	Float sy = sin( (Float)x1 * FL_UNITARIZE_PI / (Float)arg.X[1]);
	Float sz = sin( (Float)x2 * FL_UNITARIZE_PI / (Float)arg.X[2]);
	Float st = sin( (Float)x3 * FL_UNITARIZE_PI / (Float)arg.X[3]);
	Float sinsq = sx * sx + sy * sy + sz * sz + st * st;
	Float prcfact = 0.0;
	//The FFT normalization is done here
	if ( sinsq > 0.00001 )   prcfact = 4.0 / (sinsq * (Float)arg.threads);   
	arg.invpsq[id] = prcfact;
}


template<typename Float>
class GaugeFixSETINVPSP : Tunable {
	GaugeFixArg<Float> arg;
	mutable char aux_string[128]; // used as a label in the autotuner
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFixSETINVPSP(GaugeFixArg<Float> &arg) : arg(arg) {    }
	~GaugeFixSETINVPSP () { }

	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_gauge_set_invpsq<Float><<<tp.grid, tp.block, 0, stream>>>(arg);
	}

	TuneKey tuneKey() const {
		std::stringstream vol;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
		return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){}
	void postTune(){}
	long long flops() const { return 21 * arg.threads; }
	long long bytes() const { return sizeof(Float) * arg.threads; } 

}; 

template<typename Float>
__global__ void kernel_gauge_mult_norm_2D(GaugeFixArg<Float> arg){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < arg.threads) arg.gx[id] = arg.gx[id] * arg.invpsq[id]; 
}


template<typename Float>
class GaugeFixINVPSP : Tunable {
	GaugeFixArg<Float> arg;
	mutable char aux_string[128]; // used as a label in the autotuner
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFixINVPSP(GaugeFixArg<Float> &arg)
: arg(arg){ 
    cudaFuncSetCacheConfig( kernel_gauge_mult_norm_2D<Float>,   cudaFuncCachePreferL1);   }
	~GaugeFixINVPSP () {}

	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_gauge_mult_norm_2D<Float><<<tp.grid, tp.block, 0, stream>>>(arg);
	}

	TuneKey tuneKey() const {
		std::stringstream vol;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
		return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){
		//since delta contents are irrelevant at this point, we can swap gx with delta
		typename ComplexTypeId<Float>::Type *tmp = arg.gx;
		arg.gx = arg.delta;
		arg.delta = tmp;
	}
	void postTune(){
		arg.gx = arg.delta;
	}
	long long flops() const { return 2LL * arg.threads; }
	long long bytes() const { return 5LL * sizeof(Float) * arg.threads; } 

}; 







template<typename Cmplx>
__device__ __host__ inline typename RealTypeId<Cmplx>::Type Abs2(const Cmplx & a){
	return a.x * a.x + a.y * a.y;
}



template <typename Float> 
__host__ __device__ inline void reunit_link( Matrix<typename ComplexTypeId<Float>::Type,3> &U ){

	typedef typename ComplexTypeId<Float>::Type Cmplx;

	Cmplx t2 = makeComplex((Float)0.0, (Float)0.0);
	Float t1 = 0.0;
	//first normalize first row
	//sum of squares of row
#pragma unroll
	for(int c = 0; c < 3; c++)    t1 += Abs2(U(0, c));
	t1 = (Float)1.0 / sqrt(t1);
	//14
	//used to normalize row
#pragma unroll
	for(int c = 0; c < 3; c++)    U(0,c) *= t1;
	//6      
#pragma unroll
	for(int c = 0; c < 3; c++)    t2 += Conj(U(0,c)) * U(1,c);
	//24
#pragma unroll
	for(int c = 0; c < 3; c++)    U(1,c) -= t2 * U(0,c);
	//24
	//normalize second row
	//sum of squares of row
	t1 = 0.0;
#pragma unroll
	for(int c = 0; c < 3; c++)    t1 += Abs2(U(1,c));
	t1 = (Float)1.0 / sqrt(t1);
	//14
	//used to normalize row
#pragma unroll
	for(int c = 0; c < 3; c++)    U(1, c) *= t1;
	//6      
	//Reconstruct lat row
	U(2,0) = Conj(U(0,1) * U(1,2) - U(0,2) * U(1,1));
	U(2,1) = Conj(U(0,2) * U(1,0) - U(0,0) * U(1,2));
	U(2,2) = Conj(U(0,0) * U(1,1) - U(0,1) * U(1,0));
	//42
	//T=130
}








#ifdef GAUGEFIXING_DONT_USE_GX
static __device__ __host__ inline int linkNormalIndexP1(int x[], const int X[4], const int mu) {
	int y[4];
	for (int i=0; i<4; i++) y[i] = x[i];
	y[mu] = (y[mu] + 1 + X[mu]) % X[mu];
	int idx = ((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0];
	return idx;
}

template <typename Float, typename Gauge> 
__global__ void kernel_gauge_fix_U_EO_NEW( GaugeFixArg<Float> arg, Gauge dataOr, Float half_alpha){
	int idd = threadIdx.x + blockIdx.x*blockDim.x;

	if(idd >= arg.threads) return;

	int parity = 0;
	int id = idd;
	if(idd >= arg.threads / 2){
		parity = 1;
		id -= arg.threads / 2;
	}
	typedef typename ComplexTypeId<Float>::Type Cmplx;

	int x[4];
	getCoords3(x, id, arg.X, parity);
	int idx = ((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0];
	Matrix<Cmplx,3> de;
	//Read Delta
	#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
	de(0,0) = TEXTURE_DELTA<Cmplx>(idx);
	de(0,1) = TEXTURE_DELTA<Cmplx>(idx + arg.threads);
	de(0,2) = TEXTURE_DELTA<Cmplx>(idx + 2 * arg.threads);
	de(1,1) = TEXTURE_DELTA<Cmplx>(idx + 3 * arg.threads);
	de(1,2) = TEXTURE_DELTA<Cmplx>(idx + 4 * arg.threads);
	de(2,2) = TEXTURE_DELTA<Cmplx>(idx + 5 * arg.threads);
	#else
	de(0,0) = arg.delta[idx];
	de(0,1) = arg.delta[idx + arg.threads];
	de(0,2) = arg.delta[idx + 2 * arg.threads];
	de(1,1) = arg.delta[idx + 3 * arg.threads];
	de(1,2) = arg.delta[idx + 4 * arg.threads];
	de(2,2) = arg.delta[idx + 5 * arg.threads];
	#endif
	de(1,0) = makeComplex(-de(0,1).x, de(0,1).y);
	de(2,0) = makeComplex(-de(0,2).x, de(0,2).y);
	de(2,1) = makeComplex(-de(1,2).x, de(1,2).y);
	Matrix<Cmplx,3> g;
	setIdentity(&g);
	g += de * half_alpha;
	//36
	reunit_link<Float>( g );
	//130


	for(int mu = 0; mu < 4; mu++){
		Matrix<Cmplx,3> U;
		Matrix<Cmplx,3> g0;
		dataOr.load((Float*)(U.data),id, mu, parity);
		U = g * U;
		//198
		idx = linkNormalIndexP1(x,arg.X,mu);
		//Read Delta
		#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
		de(0,0) = TEXTURE_DELTA<Cmplx>(idx);
		de(0,1) = TEXTURE_DELTA<Cmplx>(idx + arg.threads);
		de(0,2) = TEXTURE_DELTA<Cmplx>(idx + 2 * arg.threads);
		de(1,1) = TEXTURE_DELTA<Cmplx>(idx + 3 * arg.threads);
		de(1,2) = TEXTURE_DELTA<Cmplx>(idx + 4 * arg.threads);
		de(2,2) = TEXTURE_DELTA<Cmplx>(idx + 5 * arg.threads);
		#else
		de(0,0) = arg.delta[idx];
		de(0,1) = arg.delta[idx + arg.threads];
		de(0,2) = arg.delta[idx + 2 * arg.threads];
		de(1,1) = arg.delta[idx + 3 * arg.threads];
		de(1,2) = arg.delta[idx + 4 * arg.threads];
		de(2,2) = arg.delta[idx + 5 * arg.threads];
		#endif
		de(1,0) = makeComplex(-de(0,1).x, de(0,1).y);
		de(2,0) = makeComplex(-de(0,2).x, de(0,2).y);
		de(2,1) = makeComplex(-de(1,2).x, de(1,2).y);

		setIdentity(&g0);
		g0 += de * half_alpha;
		//36
		reunit_link<Float>( g0 );
		//130
	
		U = U * conj(g0);
		//198
		dataOr.save((Float*)(U.data),id, mu, parity);
	}
}


template<typename Float, typename Gauge>
class GaugeFixNEW : Tunable {
	GaugeFixArg<Float> arg;
	Float half_alpha;
	Gauge dataOr;
	mutable char aux_string[128]; // used as a label in the autotuner
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFixNEW(Gauge &dataOr, GaugeFixArg<Float> &arg, Float alpha)
: dataOr(dataOr), arg(arg) { 
	half_alpha = alpha * 0.5;
    cudaFuncSetCacheConfig( kernel_gauge_fix_U_EO_NEW<Float, Gauge>,   cudaFuncCachePreferL1);
   }
	~GaugeFixNEW () { }

	void setAlpha(Float alpha){ half_alpha = alpha * 0.5; }


	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_gauge_fix_U_EO_NEW<Float, Gauge><<<tp.grid, tp.block, 0, stream>>>(arg, dataOr, half_alpha);
	}

	TuneKey tuneKey() const {
		std::stringstream vol;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
		return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	//need this
	void preTune() { arg.data.backup(); }
	void postTune() { arg.data.restore(); }
	long long flops() const {  
		return 2414LL * arg.threads;
		//Not accounting here the reconstruction of the gauge if 12 or 8!!!!!! 
	}
  	long long bytes() const { return ( dataOr.Bytes()*4LL + 5*12LL * sizeof(Float)) * arg.threads;}  

}; 



#else
template <unsigned int Elems, typename Float> 
__global__ void kernel_gauge_GX(GaugeFixArg<Float> arg, Float half_alpha){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= arg.threads) return;

	typedef typename ComplexTypeId<Float>::Type Cmplx;

	Matrix<Cmplx,3> de;
	//Read Delta
	#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
	de(0,0) = TEXTURE_DELTA<Cmplx>(id);
	de(0,1) = TEXTURE_DELTA<Cmplx>(id + arg.threads);
	de(0,2) = TEXTURE_DELTA<Cmplx>(id + 2 * arg.threads);
	de(1,1) = TEXTURE_DELTA<Cmplx>(id + 3 * arg.threads);
	de(1,2) = TEXTURE_DELTA<Cmplx>(id + 4 * arg.threads);
	de(2,2) = TEXTURE_DELTA<Cmplx>(id + 5 * arg.threads);
	#else
	de(0,0) = arg.delta[id];
	de(0,1) = arg.delta[id + arg.threads];
	de(0,2) = arg.delta[id + 2 * arg.threads];
	de(1,1) = arg.delta[id + 3 * arg.threads];
	de(1,2) = arg.delta[id + 4 * arg.threads];
	de(2,2) = arg.delta[id + 5 * arg.threads];
	#endif
	de(1,0) = makeComplex(-de(0,1).x, de(0,1).y);
	de(2,0) = makeComplex(-de(0,2).x, de(0,2).y);
	de(2,1) = makeComplex(-de(1,2).x, de(1,2).y);


	Matrix<Cmplx,3> g;
	setIdentity(&g);
	g += de * half_alpha;
	//36
	reunit_link<Float>( g );
	//130
	//gx is represented in even/odd order
	//normal lattice index to even/odd index
	int x3 = id/(arg.X[0] * arg.X[1] * arg.X[2]);
	int x2 = (id/(arg.X[0] * arg.X[1])) % arg.X[2];
	int x1 = (id/arg.X[0]) % arg.X[1];
	int x0 = id % arg.X[0];
	id  =  (x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0]) >> 1;
	id += ((x0 + x1 + x2 + x3) & 1 ) * arg.threads / 2;

	for(int i = 0; i < Elems; i++) arg.gx[id + i * arg.threads] = g.data[i];  
	//T=166 for Elems 9
	//T=208 for Elems 6
}




template<unsigned int Elems, typename Float>
class GaugeFix_GX : Tunable {
	GaugeFixArg<Float> arg;
	Float half_alpha;
	mutable char aux_string[128]; // used as a label in the autotuner
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFix_GX(GaugeFixArg<Float> &arg, Float alpha)
: arg(arg) {  half_alpha = alpha * 0.5;
    cudaFuncSetCacheConfig( kernel_gauge_GX<Elems, Float>,   cudaFuncCachePreferL1);  }
	~GaugeFix_GX () { }

	void setAlpha(Float alpha){ half_alpha = alpha * 0.5; }


	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_gauge_GX<Elems, Float><<<tp.grid, tp.block, 0, stream>>>(arg, half_alpha);
	}

	TuneKey tuneKey() const {
		std::stringstream vol;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
		return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){}
	void postTune(){}
	long long flops() const { 
		if(Elems==6)  return 208LL * arg.threads;
		else          return 166LL * arg.threads;
	}
  long long bytes() const { return 4LL * Elems * sizeof(Float) * arg.threads; }

}; 


template <unsigned int Elems, typename Float, typename Gauge> 
__global__ void kernel_gauge_fix_U_EO( GaugeFixArg<Float> arg, Gauge dataOr){
	int idd = threadIdx.x + blockIdx.x*blockDim.x;

	if(idd >= arg.threads) return;

	int parity = 0;
	int id = idd;
	if(idd >= arg.threads / 2){
		parity = 1;
		id -= arg.threads / 2;
	}
	typedef typename ComplexTypeId<Float>::Type Cmplx;

	Matrix<Cmplx,3> g;
	//for(int i = 0; i < Elems; i++) g.data[i] = arg.gx[idd + i * arg.threads]; 
	for(int i = 0; i < Elems; i++){
		#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
		g.data[i] = TEXTURE_GX<Cmplx>(idd + i * arg.threads);
		#else
		g.data[i] = arg.gx[idd + i * arg.threads];
		#endif
	}
	if(Elems==6){    
		g(2,0) = Conj(g(0,1) * g(1,2) - g(0,2) * g(1,1));
		g(2,1) = Conj(g(0,2) * g(1,0) - g(0,0) * g(1,2));
		g(2,2) = Conj(g(0,0) * g(1,1) - g(0,1) * g(1,0));
		//42
	}
	int x[4];
	getCoords3(x, id, arg.X, parity); 
	for(int mu = 0; mu < 4; mu++){
		Matrix<Cmplx,3> U;
		Matrix<Cmplx,3> g0;
		dataOr.load((Float*)(U.data),id, mu, parity);
		U = g * U;
		//198
		int idm1 = linkIndexP1(x,arg.X,mu);
		idm1 += (1 - parity) * arg.threads / 2;
		//for(int i = 0; i < Elems; i++) g0.data[i] = arg.gx[idm1 + i * arg.threads];
		for(int i = 0; i < Elems; i++){
			#ifdef GAUGEFIXING_SITE_MATRIX_LOAD_TEX
			g0.data[i] = TEXTURE_GX<Cmplx>(idm1 + i * arg.threads);
			#else
			g0.data[i] = arg.gx[idm1 + i * arg.threads];
			#endif
		}
		if(Elems==6){    
			g0(2,0) = Conj(g0(0,1) * g0(1,2) - g0(0,2) * g0(1,1));
			g0(2,1) = Conj(g0(0,2) * g0(1,0) - g0(0,0) * g0(1,2));
			g0(2,2) = Conj(g0(0,0) * g0(1,1) - g0(0,1) * g0(1,0));
			//42
		}
		U = U * conj(g0);
		//198
		dataOr.save((Float*)(U.data),id, mu, parity);
	}
	//T=42+4*(198*2+42) Elems=6
	//T=4*(198*2) Elems=9
	//Not accounting here the reconstruction of the gauge if 12 or 8!!!!!!
}


template<unsigned int Elems, typename Float, typename Gauge>
class GaugeFix : Tunable {
	GaugeFixArg<Float> arg;
	Gauge dataOr;
	mutable char aux_string[128]; // used as a label in the autotuner
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFix(Gauge &dataOr, GaugeFixArg<Float> &arg)
: dataOr(dataOr), arg(arg) { 
    cudaFuncSetCacheConfig( kernel_gauge_fix_U_EO<Elems, Float, Gauge>,   cudaFuncCachePreferL1);
   }
	~GaugeFix () { }


	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_gauge_fix_U_EO<Elems, Float, Gauge><<<tp.grid, tp.block, 0, stream>>>(arg, dataOr);
	}

	TuneKey tuneKey() const {
		std::stringstream vol;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
		return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	//need this
	void preTune() { arg.data.backup(); }
	void postTune() { arg.data.restore(); }
	long long flops() const { 
		if(Elems==6) return 1794LL * arg.threads;
		else         return 1536LL * arg.threads;
		//Not accounting here the reconstruction of the gauge if 12 or 8!!!!!! 
	}
  	long long bytes() const { return 26LL * Elems * sizeof(Float) * arg.threads;}  

}; 
#endif 
//GAUGEFIXING_DONT_USE_GX
















































template<unsigned int Elems, typename Float, typename Gauge, int gauge_dir>
void gaugefixingFFT( Gauge dataOr,  cudaGaugeField& data, \
		const unsigned int Nsteps, const unsigned int verbose_interval, \
		const Float alpha0, const unsigned int autotune, const double tolerance, \
		const unsigned int stopWtheta) {

	TimeProfile profileGaugeFix("GaugeFixCuda");

	profileGaugeFix.Start(QUDA_PROFILE_COMPUTE);

	Float alpha = alpha0;
	std::cout << "\tAlpha parameter of the Steepest Descent Method: " << alpha << std::endl;
	if(autotune) std::cout << "\tAuto tune active: yes" << std::endl;
	else         std::cout << "\tAuto tune active: no" << std::endl;
	std::cout << "\tStop criterium: " << tolerance << std::endl;
	if(stopWtheta) std::cout << "\tStop criterium method: theta" << std::endl;
	else           std::cout << "\tStop criterium method: Delta" << std::endl;
	std::cout << "\tMaximum number of iterations: " << Nsteps << std::endl;
	std::cout << "\tPrint convergence results at every " << verbose_interval << " steps" << std::endl;


	typedef typename ComplexTypeId<Float>::Type Cmplx;

	unsigned int delta_pad = data.X()[0] * data.X()[1] * data.X()[2] * data.X()[3];
	int4 size = make_int4( data.X()[0], data.X()[1], data.X()[2], data.X()[3] );
	cufftHandle plan_xy;
	cufftHandle plan_zt;

	GaugeFixArg<Float> arg(data, Elems);
	SetPlanFFT2DMany( plan_zt, size, 0, arg.delta); //for space and time ZT
	SetPlanFFT2DMany( plan_xy, size, 1, arg.delta);//with space only XY


	GaugeFixFFTRotateArg<Cmplx> arg_rotate(data);
	GaugeFixFFTRotate<Cmplx> GFRotate(arg_rotate);

	GaugeFixSETINVPSP<Float> setinvpsp(arg);
	setinvpsp.apply(0);
	GaugeFixINVPSP<Float> invpsp(arg);


	#ifdef GAUGEFIXING_DONT_USE_GX
	//without using GX, gx will be created only for plane rotation but with less size
	GaugeFixNEW<Float, Gauge> gfixNew(dataOr, arg, alpha);
	#else
	//using GX
	GaugeFix_GX<Elems, Float> calcGX(arg, alpha);
	GaugeFix<Elems, Float, Gauge> gfix(dataOr, arg);
	#endif

	GaugeFixQualityArg<Cmplx, Gauge> argQ(dataOr, data, arg.delta);
	GaugeFixQuality<Elems, Float, Gauge, gauge_dir> gfixquality(argQ);

	gfixquality.apply(0);
	double action0 = argQ.getAction();
	printf("Step: %d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());

	double diff = 0.0;
	int iter = 0;
	for(iter = 0; iter < Nsteps; iter++){
		for(int k = 0; k < 6; k++){  
			//------------------------------------------------------------------------
			// Set a pointer do the element k in lattice volume
			// each element is stored with stride lattice volume
			// it uses gx as temporary array!!!!!!
			//------------------------------------------------------------------------
			Cmplx *_array = arg.delta + k * delta_pad;
			//////  2D FFT + 2D FFT
			//------------------------------------------------------------------------
			// Perform FFT on xy plane
			//------------------------------------------------------------------------
			ApplyFFT(plan_xy, _array, arg.gx, CUFFT_FORWARD);
			//------------------------------------------------------------------------
			// Rotate hypercube, xyzt -> ztxy
			//------------------------------------------------------------------------
			GFRotate.setDirection(0, arg.gx, _array);
			GFRotate.apply(0);
			//------------------------------------------------------------------------
			// Perform FFT on zt plane
			//------------------------------------------------------------------------      
			ApplyFFT(plan_zt, _array, arg.gx, CUFFT_FORWARD);
			//------------------------------------------------------------------------
			// Normalize FFT and apply pmax^2/p^2
			//------------------------------------------------------------------------
			invpsp.apply(0);
			//------------------------------------------------------------------------
			// Perform IFFT on zt plane
			//------------------------------------------------------------------------  
			ApplyFFT(plan_zt, arg.gx, _array, CUFFT_INVERSE);
			//------------------------------------------------------------------------
			// Rotate hypercube, ztxy -> xyzt
			//------------------------------------------------------------------------
			GFRotate.setDirection(1, _array, arg.gx);
			GFRotate.apply(0);  
			//------------------------------------------------------------------------
			// Perform IFFT on xy plane
			//------------------------------------------------------------------------    
			ApplyFFT(plan_xy, arg.gx, _array, CUFFT_INVERSE);
		}
		#ifdef GAUGEFIXING_DONT_USE_GX
		//------------------------------------------------------------------------
		// Apply gauge fix to current gauge field
		//------------------------------------------------------------------------
		gfixNew.apply(0);
		#else
		//------------------------------------------------------------------------
		// Calculate g(x)
		//------------------------------------------------------------------------
		calcGX.apply(0);
		//------------------------------------------------------------------------
		// Apply gauge fix to current gauge field
		//------------------------------------------------------------------------
		gfix.apply(0);
		#endif
		//------------------------------------------------------------------------
		// Measure gauge quality and recalculate new Delta(x)
		//------------------------------------------------------------------------
		gfixquality.apply(0);
		double action = argQ.getAction();
		diff = abs(action0 - action);
		if((iter % verbose_interval) == (verbose_interval - 1))
			printf("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter+1, argQ.getAction(), argQ.getTheta(), diff);
		if ( autotune && ((action - action0) < -1e-14) ) {
			if(alpha > 0.01){
				alpha = 0.95 * alpha;
				#ifdef GAUGEFIXING_DONT_USE_GX
				gfixNew.setAlpha(alpha);
				#else
				calcGX.setAlpha(alpha);
				#endif 
				printf(">>>>>>>>>>>>>> Warning: changing alpha down -> %.4e\n", alpha );
			}
		} 
		//------------------------------------------------------------------------
		// Check gauge fix quality criterium
		//------------------------------------------------------------------------
		if(stopWtheta) {   if( argQ.getTheta() < tolerance ) break; }
		else { if( diff < tolerance ) break; }

		action0 = action;
	}
	if((iter % verbose_interval) != 0)
		printf("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter, argQ.getAction(), argQ.getTheta(), diff);

	arg.free();
	CUFFT_SAFE_CALL(cufftDestroy(plan_zt));
	CUFFT_SAFE_CALL(cufftDestroy(plan_xy)); 
	checkCudaError();

	cudaDeviceSynchronize();
	profileGaugeFix.Stop(QUDA_PROFILE_COMPUTE);
	double secs = profileGaugeFix.Last(QUDA_PROFILE_COMPUTE);
	double fftflop = 5.0 * (log2((double)( data.X()[0] * data.X()[1]) ) + log2( (double)(data.X()[2] * data.X()[3] )));
	fftflop *= (double)( data.X()[0] * data.X()[1] * data.X()[2] * data.X()[3] );
	double gflops = setinvpsp.flops() + gfixquality.flops();
	double gbytes = setinvpsp.bytes() + gfixquality.bytes();
	double flop = invpsp.flops() * Elems;
	double byte = invpsp.bytes() * Elems;
	flop += (GFRotate.flops() + fftflop) * Elems * 2;
	byte += GFRotate.bytes() * Elems * 4; //includes FFT reads, assuming 1 read and 1 write per site
	#ifdef GAUGEFIXING_DONT_USE_GX
	flop += gfixNew.flops();
	byte += gfixNew.bytes();
	#else
	flop += calcGX.flops();
	byte += calcGX.bytes();
	flop += gfix.flops();
	byte += gfix.bytes();
	#endif
	flop += gfixquality.flops();
	byte += gfixquality.bytes();
	gflops += flop * iter;
	gbytes += byte * iter;
	gflops = (gflops*1e-9)/(secs);
	gbytes = gbytes/(secs*1e9);


  	printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);


}

template<unsigned int Elems, typename Float, typename Gauge>
void gaugefixingFFT( Gauge dataOr,  cudaGaugeField& data, const unsigned int gauge_dir, \
		const unsigned int Nsteps, const unsigned int verbose_interval, const Float alpha, const unsigned int autotune, \
		const double tolerance, const unsigned int stopWtheta) {
	if( gauge_dir !=3 ){
		printf("Starting Landau gauge fixing with FFTs...\n");
		gaugefixingFFT<Elems, Float, Gauge, 4>(dataOr, data, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
	} 
	else {        
		printf("Starting Coulomb gauge fixing with FFTs...\n");
		gaugefixingFFT<Elems, Float, Gauge, 3>(dataOr, data, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
	}
}



template<typename Float>
void gaugefixingFFT( cudaGaugeField& data, const unsigned int gauge_dir, \
		const unsigned int Nsteps, const unsigned int verbose_interval, const Float alpha, const unsigned int autotune, \
		const double tolerance, const unsigned int stopWtheta) {

	// Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
	// Need to fix this!!
	//9 and 6 means the number of complex elements used to store g(x) and Delta(x)
	if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
		if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
			printf("QUDA_RECONSTRUCT_NO\n");
			gaugefixingFFT<9, Float>(FloatNOrder<Float, 18, 2, 18>(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
		} else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
			printf("QUDA_RECONSTRUCT_12\n");
			gaugefixingFFT<6, Float>(FloatNOrder<Float, 18, 2, 12>(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);

		} else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
			printf("QUDA_RECONSTRUCT_8\n");
			gaugefixingFFT<6, Float>(FloatNOrder<Float, 18, 2,  8>(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);

		} else {
			errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
		}
	} else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
		if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
			printf("QUDA_RECONSTRUCT_NO\n");
			gaugefixingFFT<9, Float>(FloatNOrder<Float, 18, 4, 18>(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
		} else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
			printf("QUDA_RECONSTRUCT_12\n");
			gaugefixingFFT<6, Float>(FloatNOrder<Float, 18, 4, 12>(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
		} else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
			printf("QUDA_RECONSTRUCT_8\n");
			gaugefixingFFT<6, Float>(FloatNOrder<Float, 18, 4,  8>(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
		} else {
			errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
		}
	} else {
		errorQuda("Invalid Gauge Order\n");
	}
}

void gaugefixingFFT( cudaGaugeField& data, const unsigned int gauge_dir, \
		const unsigned int Nsteps, const unsigned int verbose_interval, const double alpha, const unsigned int autotune, \
		const double tolerance, const unsigned int stopWtheta) {

#ifdef MULTI_GPU
	if(comm_size() > 1)
	errorQuda("Gauge Fixing with FFTs in multi-GPU support NOT implemented yet!\n");
#endif
	if(data.Precision() == QUDA_HALF_PRECISION) {
		errorQuda("Half precision not supported\n");
	}
	if (data.Precision() == QUDA_SINGLE_PRECISION) {
		gaugefixingFFT<float> (data, gauge_dir, Nsteps, verbose_interval, (float)alpha, autotune, tolerance, stopWtheta);
	} else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
		gaugefixingFFT<double>(data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
	} else {
		errorQuda("Precision %d not supported", data.Precision());
	}
}



}

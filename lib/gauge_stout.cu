#include <quda_internal.h>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>

#define  DOUBLE_TOL	1e-15
#define  SINGLE_TOL	2e-6

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template <typename Float, typename GaugeOr, typename GaugeDs>
  struct GaugeSTOUTArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4]; 
#endif
    GaugeOr origin;
    const Float rho;
    const Float tolerance;
    
    GaugeDs dest;

    GaugeSTOUTArg(GaugeOr &origin, GaugeDs &dest, const GaugeField &data, const Float rho, const Float tolerance) 
      : origin(origin), dest(dest), rho(rho), tolerance(tolerance) {
#ifdef MULTI_GPU
      for ( int dir = 0; dir < 4; ++dir ) {
        border[dir] = data.R()[dir];
        X[dir] = data.X()[dir] - border[dir] * 2;
      } 
#else
        for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
	threads = X[0]*X[1]*X[2]*X[3];
    }
  };


  template <typename Float, typename GaugeOr, typename GaugeDs, typename Float2>
  __host__ __device__ void computeStaple(GaugeSTOUTArg<Float,GaugeOr,GaugeDs>& arg, int idx, int parity, int dir, Matrix<Float2,3> &staple) {

    typedef typename ComplexTypeId<Float>::Type Cmplx;
      // compute spacetime dimensions and parity

    int X[4]; 
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords(x, idx, X, parity);
#ifdef MULTI_GPU
    for(int dr=0; dr<4; ++dr) {
         x[dr] += arg.border[dr];
         X[dr] += 2*arg.border[dr];
    }
#endif

    setZero(&staple);

    for (int mu=0; mu<3; mu++) {  // I believe most users won't want to include time staples in smearing
      if (mu == dir) {
        continue;
      }

      int nu = dir;

      {
        int dx[4] = {0, 0, 0, 0};
        Matrix<Cmplx,3> U1;
        arg.origin.load((Float*)(U1.data),linkIndexShift(x,dx,X), mu, parity); 

        Matrix<Cmplx,3> U2;
        dx[mu]++;
        arg.origin.load((Float*)(U2.data),linkIndexShift(x,dx,X), nu, 1-parity); 

        Matrix<Cmplx,3> U3;
        dx[mu]--;
        dx[nu]++;
        arg.origin.load((Float*)(U3.data),linkIndexShift(x,dx,X), mu, 1-parity); 
   
        Matrix<Cmplx,3> tmpS;

        tmpS	= U1 * U2;
	tmpS	= tmpS * conj(U3);

	staple = staple + tmpS;

        dx[mu]--;
        dx[nu]--;
        arg.origin.load((Float*)(U1.data),linkIndexShift(x,dx,X), mu, 1-parity); 
        arg.origin.load((Float*)(U2.data),linkIndexShift(x,dx,X), nu, 1-parity); 

        dx[nu]++;
        arg.origin.load((Float*)(U3.data),linkIndexShift(x,dx,X), mu, parity); 

        tmpS	= conj(U1);
	tmpS	= tmpS * U2;
	tmpS	= tmpS * U3;

	staple = staple + tmpS;
      }
    }
  }

  template<typename Float, typename GaugeOr, typename GaugeDs>
    __global__ void computeSTOUTStep(GaugeSTOUTArg<Float,GaugeOr,GaugeDs> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= arg.threads) return;
      typedef typename ComplexTypeId<Float>::Type Cmplx;

      int parity = 0;
      if(idx >= arg.threads/2) {
        parity = 1;
        idx -= arg.threads/2;
      }

      int X[4]; 
      for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, idx, X, parity);
#ifdef MULTI_GPU
      for(int dr=0; dr<4; ++dr) {
           x[dr] += arg.border[dr];
           X[dr] += 2*arg.border[dr];
      }
#endif

      int dx[4] = {0, 0, 0, 0};
      for (int dir=0; dir < 3; dir++) {				//Only spatial dimensions are smeared
        Matrix<Cmplx,3> U, UDag, Stap, Omega, OmegaDag, OmegaDiff, ODT, Q,
	  exp_iQ, tmp1, tmp2;
	Cmplx OmegaDiffTr;
	Cmplx i_2 = makeComplex<Cmplx>(0,0.5);

	//This function gets stap = S_{mu,nu} i.e., the staple of length 3,
        computeStaple<Float,GaugeOr,GaugeDs,Cmplx>(arg,idx,parity,dir,Stap);
	//
	// |- > -|
	// ^     v
	// |     |
	//          +  |     |
	//             v     ^
	//             |- > -|

	// Get link U
        arg.origin.load((Float*)(U.data),linkIndexShift(x,dx,X),dir,parity);

	//Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag

	//Get U^{\dagger}
	computeMatrixInverse(U,&UDag);
	
	//Compute \Omega = \rho * S * U^{\dagger}
	tmp1 = arg.rho * Stap;
	Omega = tmp1 * UDag;

	//Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu} 
	//                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]

	OmegaDag = conj(Omega);
	OmegaDiff = OmegaDag - Omega;

	Q = OmegaDiff;
	OmegaDiffTr = getTrace(OmegaDiff);
	OmegaDiffTr =  1.0/3.0 * OmegaDiffTr;

	//Matrix proportional to OmegaDiffTr
	setIdentity(&ODT);
	tmp1 = OmegaDiffTr * ODT;

	Q = Q - tmp1;
	Q = i_2 * Q;
	//Q is now defined.

	exponentiate_iQ(Q,&exp_iQ);
	U = exp_iQ * U;

	//No need to project back down to SU(3)
        //polarSu3<Cmplx,Float>(&U, arg.tolerance);
        arg.dest.save((Float*)(U.data),linkIndexShift(x,dx,X), dir, parity); 
    }
  }

  template<typename Float, typename GaugeOr, typename GaugeDs>
    class GaugeSTOUT : Tunable {
      GaugeSTOUTArg<Float,GaugeOr,GaugeDs> arg;
      const QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune shared memory
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      GaugeSTOUT(GaugeSTOUTArg<Float,GaugeOr, GaugeDs> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}
      virtual ~GaugeSTOUT () {}

      void apply(const cudaStream_t &stream){
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          computeSTOUTStep<<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
        } else {
          errorQuda("CPU not supported yet\n");
          //computeSTOUTStepCPU(arg);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.X[0] << "x";
        vol << arg.X[1] << "x";
        vol << arg.X[2] << "x";
        vol << arg.X[3];
        aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
        return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
      }


      std::string paramString(const TuneParam &param) const {
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      void preTune(){}
      void postTune(){}
      long long flops() const { return (1)*6*arg.threads; }
      long long bytes() const { return (1)*6*arg.threads*sizeof(Float); } // Only correct if there is no link reconstruction

    }; // GaugeSTOUT

  template<typename Float,typename GaugeOr, typename GaugeDs>
    void STOUTStep(GaugeOr origin, GaugeDs dest, const GaugeField& dataOr, Float rho, QudaFieldLocation location) {
      if (dataOr.Precision() == QUDA_DOUBLE_PRECISION) {
        GaugeSTOUTArg<Float,GaugeOr,GaugeDs> arg(origin, dest, dataOr, rho, DOUBLE_TOL);
        GaugeSTOUT<Float,GaugeOr,GaugeDs> gaugeSTOUT(arg, location);
        gaugeSTOUT.apply(0);
      } else {
        GaugeSTOUTArg<Float,GaugeOr,GaugeDs> arg(origin, dest, dataOr, rho, SINGLE_TOL);
        GaugeSTOUT<Float,GaugeOr,GaugeDs> gaugeSTOUT(arg, location);
        gaugeSTOUT.apply(0);
      }
      cudaDeviceSynchronize();
    }

  template<typename Float>
    void STOUTStep(GaugeField &dataDs, const GaugeField& dataOr, Float rho, QudaFieldLocation location) {

    if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GDs;

      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
      }
    } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GDs;
      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
      }
    } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GDs;
      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, location);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
    } else {
      errorQuda("Reconstruction type %d of destination gauge field not supported", dataDs.Reconstruct());
    }

  }

#endif

  void STOUTStep(GaugeField &dataDs, const GaugeField& dataOr, double rho, QudaFieldLocation location) {

#ifdef GPU_GAUGE_TOOLS

    if(dataOr.Precision() != dataDs.Precision()) {
      errorQuda("Oriign and destination fields must have the same precision\n");
    }

    if(dataDs.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported\n");
    }

    if (!dataOr.isNative())
      errorQuda("Order %d with %d reconstruct not supported", dataOr.Order(), dataOr.Reconstruct());

    if (!dataDs.isNative())
      errorQuda("Order %d with %d reconstruct not supported", dataDs.Order(), dataDs.Reconstruct());

    if (dataDs.Precision() == QUDA_SINGLE_PRECISION){
      STOUTStep<float>(dataDs, dataOr, (float) rho, location);
    } else if(dataDs.Precision() == QUDA_DOUBLE_PRECISION) {
      STOUTStep<double>(dataDs, dataOr, rho, location);
    } else {
      errorQuda("Precision %d not supported", dataDs.Precision());
    }
    return;
#else
  errorQuda("Gauge tools are not build");
#endif
  }

}

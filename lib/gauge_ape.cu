#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

#define  DOUBLE_TOL	1e-15
#define  SINGLE_TOL	2e-6

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template <typename Float, typename GaugeOr, typename GaugeDs>
  struct GaugeAPEArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4]; 
#endif
    GaugeOr origin;
    const Float alpha;
    const Float tolerance;
    
    GaugeDs dest;

    GaugeAPEArg(GaugeOr &origin, GaugeDs &dest, const GaugeField &data, const Float alpha, const Float tolerance) 
      : origin(origin), dest(dest), alpha(alpha), tolerance(tolerance) {
#ifdef MULTI_GPU
        for(int dir=0; dir<4; ++dir){
          border[dir] = 2;
        }
        for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
        for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
	threads = X[0]*X[1]*X[2]*X[3];
    }
  };


  __device__ __host__ inline int linkIndex2(int x[], int dx[], const int X[4]) {
    int y[4];
    for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
    return idx;
  }


  __device__ __host__ inline void getCoords2(int x[4], int cb_index, const int X[4], int parity) 
  {
    x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    x[1] = (cb_index/(X[0]/2)) % X[1];
    x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    return;
  }

  template <typename Float2, typename Float>
  __host__ __device__ int checkUnitary(Matrix<Float2,3> in, Matrix<Float2,3> *inv, const Float tol)
  {
    computeMatrixInverse(in, inv);

    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
      {
        if (fabs(in(i,j).x - (*inv)(j,i).x) > tol)
          return 1;
        if (fabs(in(i,j).y + (*inv)(j,i).y) > tol)
          return 1;
      }
    return 0;
  }

  template <typename Float2>
  __host__ __device__ int checkUnitaryPrint(Matrix<Float2,3> in, Matrix<Float2,3> *inv)
  {
    computeMatrixInverse(in, inv);
    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
      {
        printf("TESTR: %+.3le %+.3le %+.3le\n", in(i,j).x, (*inv)(j,i).x, fabs(in(i,j).x - (*inv)(j,i).x));
	printf("TESTI: %+.3le %+.3le %+.3le\n", in(i,j).y, (*inv)(j,i).y, fabs(in(i,j).y + (*inv)(j,i).y));
        cudaDeviceSynchronize();
        if (fabs(in(i,j).x - (*inv)(j,i).x) > 1e-14)
          return 1;
        if (fabs(in(i,j).y + (*inv)(j,i).y) > 1e-14)
          return 1;
      }
    return 0;  
  }

  template <typename Float2,typename Float>
  __host__ __device__ void polarSu3(Matrix<Float2,3> *in, Float tol)
  {
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    Matrix<Cmplx,3> inv, out;

    out = *in;
    computeMatrixInverse(out, &inv);

    do
    {
      out = out + conj(inv);
      out = out*0.5;
    } while(checkUnitary(out, &inv, tol));
/*
    printf("Convergence after %d iterations\n", N);
    cudaDeviceSynchronize();
    printf("%+.3lf %+.3lfi    %+.3lf %+.3lfi    %+.3lf %+.3lfi\n", out(0,0).x, out(0,0).y, out(0,1).x, out(0,1).y, out(0,2).x, out(0,2).y);
    printf("%+.3lf %+.3lfi    %+.3lf %+.3lfi    %+.3lf %+.3lfi\n", out(1,0).x, out(1,0).y, out(1,1).x, out(1,1).y, out(1,2).x, out(1,2).y);
    printf("%+.3lf %+.3lfi    %+.3lf %+.3lfi    %+.3lf %+.3lfi\n", out(2,0).x, out(2,0).y, out(2,1).x, out(2,1).y, out(2,2).x, out(2,2).y);
    printf("\n\n");
    printf("%+.3lf %+.3lfi    %+.3lf %+.3lfi    %+.3lf %+.3lfi\n", inv(0,0).x, inv(0,0).y, inv(0,1).x, inv(0,1).y, inv(0,2).x, inv(0,2).y);
    printf("%+.3lf %+.3lfi    %+.3lf %+.3lfi    %+.3lf %+.3lfi\n", inv(1,0).x, inv(1,0).y, inv(1,1).x, inv(1,1).y, inv(1,2).x, inv(1,2).y);
    printf("%+.3lf %+.3lfi    %+.3lf %+.3lfi    %+.3lf %+.3lfi\n", inv(2,0).x, inv(2,0).y, inv(2,1).x, inv(2,1).y, inv(2,2).x, inv(2,2).y);
    printf("\n\n\n\n");
    cudaDeviceSynchronize();
*/
    Cmplx  det = getDeterminant(out);
    double mod = det.x*det.x + det.y*det.y;
    mod = pow(mod, (1./6.));
    double angle = atan2(det.y, det.x);
    angle /= -3.;
    
    Cmplx cTemp;

    cTemp.x = cos(angle)/mod;
    cTemp.y = sin(angle)/mod;

//    out = out*cTemp;
    *in = out*cTemp;
/*    if (checkUnitary(out, &inv))
    {
    	cTemp = getDeterminant(out);
	printf ("DetX: %+.3lf  %+.3lfi, %.3lf %.3lf\nDetN: %+.3lf  %+.3lfi", det.x, det.y, mod, angle, cTemp.x, cTemp.y);
        cudaDeviceSynchronize();
	checkUnitaryPrint(out, &inv);
	setIdentity(in);
        *in = *in * 0.5;
    }
    else
    {
      cTemp = getDeterminant(out);
//      printf("Det: %+.3lf %+.3lf\n", cTemp.x, cTemp.y);
      cudaDeviceSynchronize();

      if (fabs(cTemp.x - 1.0) > 1e-8)
	setIdentity(in);
      else if (fabs(cTemp.y) > 1e-8)
      {
	setIdentity(in);
        printf("DadadaUnitary failed\n");
        *in = *in * 0.1;
      }
      else
        *in = out;
    }*/
  }


  template <typename Float, typename GaugeOr, typename GaugeDs, typename Float2>
  __host__ __device__ void computeStaple(GaugeAPEArg<Float,GaugeOr,GaugeDs>& arg, int idx, int parity, int dir, Matrix<Float2,3> &staple) {

    typedef typename ComplexTypeId<Float>::Type Cmplx;
      // compute spacetime dimensions and parity

    int X[4]; 
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords2(x, idx, X, parity);
#ifdef MULTI_GPU
    for(int dr=0; dr<4; ++dr) {
         x[dr] += arg.border[dr];
         X[dr] += 2*arg.border[dr];
    }
#endif

    setZero(&staple);

    for (int mu=0; mu<4; mu++) {
      if (mu == dir) {
        continue;
      }

      int nu = dir;

      {
        int dx[4] = {0, 0, 0, 0};
        Matrix<Cmplx,3> U1;
        arg.origin.load((Float*)(U1.data),linkIndex2(x,dx,X), mu, parity); 

        Matrix<Cmplx,3> U2;
        dx[mu]++;
        arg.origin.load((Float*)(U2.data),linkIndex2(x,dx,X), nu, 1-parity); 

        Matrix<Cmplx,3> U3;
        dx[mu]--;
        dx[nu]++;
        arg.origin.load((Float*)(U3.data),linkIndex2(x,dx,X), mu, 1-parity); 
   
        Matrix<Cmplx,3> tmpS;

        tmpS	= U1 * U2;
	tmpS	= tmpS * conj(U3);

	staple = staple + tmpS;

        dx[mu]--;
        dx[nu]--;
        arg.origin.load((Float*)(U1.data),linkIndex2(x,dx,X), mu, 1-parity); 
        arg.origin.load((Float*)(U2.data),linkIndex2(x,dx,X), nu, 1-parity); 

        dx[nu]++;
        arg.origin.load((Float*)(U3.data),linkIndex2(x,dx,X), mu, parity); 

        tmpS	= conj(U1);
	tmpS	= tmpS * U2;
	tmpS	= tmpS * U3;

	staple = staple + tmpS;
      }
    }
  }

  template<typename Float, typename GaugeOr, typename GaugeDs>
    __global__ void computeAPEStep(GaugeAPEArg<Float,GaugeOr,GaugeDs> arg){
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
      getCoords2(x, idx, X, parity);
#ifdef MULTI_GPU
      for(int dr=0; dr<4; ++dr) {
           x[dr] += arg.border[dr];
           X[dr] += 2*arg.border[dr];
      }
#endif

      int dx[4] = {0, 0, 0, 0};
      for (int dir=0; dir < 3; dir++) {				//Only spatial dimensions are smeared
        Matrix<Cmplx,3> U, S;

        computeStaple<Float,GaugeOr,GaugeDs,Cmplx>(arg,idx,parity,dir,S);

        arg.origin.load((Float*)(U.data),linkIndex2(x,dx,X), dir, parity);

	U  = U * (1. - arg.alpha);
	S  = S * (arg.alpha/6.);

	U  = U + S;

        polarSu3<Cmplx,Float>(&U, arg.tolerance);
        arg.dest.save((Float*)(U.data),linkIndex2(x,dx,X), dir, parity); 
    }
  }

  template<typename Float, typename GaugeOr, typename GaugeDs>
    class GaugeAPE : Tunable {
      GaugeAPEArg<Float,GaugeOr,GaugeDs> arg;
      const QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune shared memory
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      GaugeAPE(GaugeAPEArg<Float,GaugeOr, GaugeDs> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}
      virtual ~GaugeAPE () {}

      void apply(const cudaStream_t &stream){
        if(location == QUDA_CUDA_FIELD_LOCATION){
#if (__COMPUTE_CAPABILITY__ >= 200)
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          computeAPEStep<<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
	  errorQuda("GaugeAPE not supported on pre-Fermi architecture");
#endif
        }else{
          errorQuda("CPU not supported yet\n");
          //computeAPEStepCPU(arg);
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

    }; // GaugeAPE

  template<typename Float,typename GaugeOr, typename GaugeDs>
    void APEStep(GaugeOr origin, GaugeDs dest, const GaugeField& dataOr, Float alpha, QudaFieldLocation location) {
      if (dataOr.Precision() == QUDA_DOUBLE_PRECISION) {
        GaugeAPEArg<Float,GaugeOr,GaugeDs> arg(origin, dest, dataOr, alpha, DOUBLE_TOL);
        GaugeAPE<Float,GaugeOr,GaugeDs> gaugeAPE(arg, location);
        gaugeAPE.apply(0);
      } else {
        GaugeAPEArg<Float,GaugeOr,GaugeDs> arg(origin, dest, dataOr, alpha, SINGLE_TOL);
        GaugeAPE<Float,GaugeOr,GaugeDs> gaugeAPE(arg, location);
        gaugeAPE.apply(0);
      }
      cudaDeviceSynchronize();
    }

  template<typename Float>
    void APEStep(GaugeField &dataDs, const GaugeField& dataOr, Float alpha, QudaFieldLocation location) {

      // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
      // Need to fix this!!

      if(dataDs.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
        if(dataOr.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
          if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO) {
              APEStep(FloatNOrder<Float, 18, 2, 18>(dataOr), FloatNOrder<Float, 18, 2, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 2, 12>(dataOr), FloatNOrder<Float, 18, 2, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 2,  8>(dataOr), FloatNOrder<Float, 18, 2, 18>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 2, 18>(dataOr), FloatNOrder<Float, 18, 2, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 2, 12>(dataOr), FloatNOrder<Float, 18, 2, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 2,  8>(dataOr), FloatNOrder<Float, 18, 2, 12>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 2, 18>(dataOr), FloatNOrder<Float, 18, 2,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 2, 12>(dataOr), FloatNOrder<Float, 18, 2,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 2,  8>(dataOr), FloatNOrder<Float, 18, 2,  8>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else {
            errorQuda("Reconstruction type %d of destination gauge field not supported", dataDs.Reconstruct());
          }
        } else if(dataOr.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
          if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO) {
              APEStep(FloatNOrder<Float, 18, 4, 18>(dataOr), FloatNOrder<Float, 18, 2, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 4, 12>(dataOr), FloatNOrder<Float, 18, 2, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 4,  8>(dataOr), FloatNOrder<Float, 18, 2, 18>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 4, 18>(dataOr), FloatNOrder<Float, 18, 2, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 4, 12>(dataOr), FloatNOrder<Float, 18, 2, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 4,  8>(dataOr), FloatNOrder<Float, 18, 2, 12>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 4, 18>(dataOr), FloatNOrder<Float, 18, 2,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 4, 12>(dataOr), FloatNOrder<Float, 18, 2,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 4,  8>(dataOr), FloatNOrder<Float, 18, 2,  8>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else {
            errorQuda("Reconstruction type %d of destination gauge field not supported", dataDs.Reconstruct());
          }
        } else {
	  errorQuda("Invalid Gauge Order origin field\n");
        }
      } else if(dataDs.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
        if(dataOr.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
          if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO) {
              APEStep(FloatNOrder<Float, 18, 2, 18>(dataOr), FloatNOrder<Float, 18, 4, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 2, 12>(dataOr), FloatNOrder<Float, 18, 4, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 2,  8>(dataOr), FloatNOrder<Float, 18, 4, 18>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 2, 18>(dataOr), FloatNOrder<Float, 18, 4, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 2, 12>(dataOr), FloatNOrder<Float, 18, 4, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 2,  8>(dataOr), FloatNOrder<Float, 18, 4, 12>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 2, 18>(dataOr), FloatNOrder<Float, 18, 4,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 2, 12>(dataOr), FloatNOrder<Float, 18, 4,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 2,  8>(dataOr), FloatNOrder<Float, 18, 4,  8>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else {
            errorQuda("Reconstruction type %d of destination gauge field not supported", dataDs.Reconstruct());
          }
        } else if(dataOr.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
          if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO) {
              APEStep(FloatNOrder<Float, 18, 4, 18>(dataOr), FloatNOrder<Float, 18, 4, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 4, 12>(dataOr), FloatNOrder<Float, 18, 4, 18>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 4,  8>(dataOr), FloatNOrder<Float, 18, 4, 18>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 4, 18>(dataOr), FloatNOrder<Float, 18, 4, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 4, 12>(dataOr), FloatNOrder<Float, 18, 4, 12>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 4,  8>(dataOr), FloatNOrder<Float, 18, 4, 12>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
            if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
              APEStep(FloatNOrder<Float, 18, 4, 18>(dataOr), FloatNOrder<Float, 18, 4,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
              APEStep(FloatNOrder<Float, 18, 4, 12>(dataOr), FloatNOrder<Float, 18, 4,  8>(dataDs), dataOr, alpha, location);
            }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
              APEStep(FloatNOrder<Float, 18, 4,  8>(dataOr), FloatNOrder<Float, 18, 4,  8>(dataDs), dataOr, alpha, location);
            }else{
              errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
          } else {
            errorQuda("Reconstruction type %d of destination gauge field not supported", dataDs.Reconstruct());
          }
        } else {
	  errorQuda("Invalid Gauge Order origin field\n");
        }
      } else {
        errorQuda("Invalid Gauge Order destination field\n");
      }
  }
#endif

  void APEStep(GaugeField &dataDs, const GaugeField& dataOr, double alpha, QudaFieldLocation location) {

#ifdef GPU_GAUGE_TOOLS

    if(dataOr.Precision() != dataDs.Precision()) {
      errorQuda("Oriign and destination fields must have the same precision\n");
    }

    if(dataDs.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported\n");
    }

    if (dataDs.Precision() == QUDA_SINGLE_PRECISION){
      APEStep<float>(dataDs, dataOr, (float) alpha, location);
    } else if(dataDs.Precision() == QUDA_DOUBLE_PRECISION) {
      APEStep<double>(dataDs, dataOr, alpha, location);
    } else {
      errorQuda("Precision %d not supported", dataDs.Precision());
    }
    return;
#else
  errorQuda("Gauge tools are not build");
#endif
  }


}

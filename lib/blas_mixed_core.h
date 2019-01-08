namespace mixed {

  /**
     Driver for generic blas routine with four loads and two store.
  */
  template <template <typename Float, typename FloatN> class Functor,
    int writeX=0, int writeY=0, int writeZ=0, int writeW=0, int writeV=0>
    void blasCuda(const double2 &a, const double2 &b, const double2 &c,
		  ColorSpinorField &x, ColorSpinorField &y,
		  ColorSpinorField &z, ColorSpinorField &w,
                  ColorSpinorField &v) {

    checkPrecision(x, z, w);
    checkPrecision(y, v);

    if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

      if (!x.isNative()) {
        warningQuda("Device blas on non-native fields is not supported\n");
        return;
      }

      if (x.Precision() == QUDA_SINGLE_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 12;
	  blas::blasCuda<double2,float4,double2,M,Functor,writeX,writeY,writeZ,writeW,writeV>(a,b,c,x,y,z,w,v,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<double2,float2,double2,M,Functor,writeX,writeY,writeZ,writeW,writeV>(a,b,c,x,y,z,w,v,x.Volume());
	}
      } else if (x.Precision() == QUDA_HALF_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 12;
	  blas::blasCuda<double2,short4,double2,M,Functor,writeX,writeY,writeZ,writeW,writeV>(a,b,c,x,y,z,w,v,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<double2,short2,double2,M,Functor,writeX,writeY,writeZ,writeW,writeV>(a,b,c,x,y,z,w,v,x.Volume());
	}
      } else if (x.Precision() == QUDA_HALF_PRECISION && y.Precision() == QUDA_SINGLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 6;
	  blas::blasCuda<float4,short4,float4,M,Functor,writeX,writeY,writeZ,writeW,writeV>(a,b,c,x,y,z,w,v,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<float2,short2,float2,M,Functor,writeX,writeY,writeZ,writeW,writeV>(a,b,c,x,y,z,w,v,x.Volume());
	}
      } else if (x.Precision() == QUDA_QUARTER_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 6;
	  blas::blasCuda<double2,char4,double2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<double2,char2,double2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	}
      } else if (x.Precision() == QUDA_QUARTER_PRECISION && y.Precision() == QUDA_SINGLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 6;
	  blas::blasCuda<float4,char4,float4,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<float2,char2,float2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	}
      } else if (x.Precision() == QUDA_QUARTER_PRECISION && y.Precision() == QUDA_HALF_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 6;
	  blas::blasCuda<float4,char4,short4,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<float2,char2,short2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	}
      } else {
	errorQuda("Not implemented for this precision combination %d %d", x.Precision(), y.Precision());
      }
    } else { // fields on the cpu
      using namespace quda::colorspinor;
      if (x.Precision() == QUDA_SINGLE_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
	Functor<double2, double2> f(a, b, c);
	genericBlas<float, double, writeX, writeY, writeZ, writeW, writeV>(x, y, z, w, v, f);
      } else {
	errorQuda("Not implemented");
      }
    }
  }

}

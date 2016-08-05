namespace mixed {

  /**
     Driver for generic blas routine with four loads and two store.
  */
  template <template <typename Float, typename FloatN> class Functor,
    int writeX, int writeY, int writeZ, int writeW>
    void blasCuda(const double2 &a, const double2 &b, const double2 &c,
		  ColorSpinorField &x, ColorSpinorField &y,
		  ColorSpinorField &z, ColorSpinorField &w) {

    if (Location(x, y, z, w) == QUDA_CUDA_FIELD_LOCATION) {
      if (x.Precision() == QUDA_SINGLE_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 12;
	  blas::blasCuda<double2,float4,double2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<double2,float2,double2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	}
      } else if (x.Precision() == QUDA_HALF_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 12;
	  blas::blasCuda<double2,short4,double2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<double2,short2,double2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	}
      } else if (x.Precision() == QUDA_HALF_PRECISION && y.Precision() == QUDA_SINGLE_PRECISION) {
	if (x.Nspin() == 4) {
	  const int M = 6;
	  blas::blasCuda<float4,short4,float4,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	} else if (x.Nspin() == 1) {
	  const int M = 3;
	  blas::blasCuda<float2,short2,float2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
	}
      } else {
	errorQuda("Not implemented for this precision combination");
      }
    } else { // fields on the cpu
      using namespace quda::colorspinor;
      if (x.Precision() == QUDA_SINGLE_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
	Functor<double2, double2> f(a, b, c);
	genericBlas<float, double, writeX, writeY, writeZ, writeW>(x, y, z, w, f);
      } else {
	errorQuda("Not implemented");
      }
    }
  }

}

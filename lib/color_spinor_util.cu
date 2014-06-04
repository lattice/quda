#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

namespace quda {

  using namespace colorspinor;

  /**
     Random number insertion over all field elements
  */
  template <class T>
  void random(T &t) {
    for (int x=0; x<t.Volume(); x++) {
      for (int s=0; s<t.Nspin(); s++) {
	for (int c=0; c<t.Ncolor(); c++) {
	  t(x,s,c).real(comm_drand());
	  t(x,s,c).imag(comm_drand());
	}
      }
    }
  }

  /**
     Create a point source at spacetime point x, spin s and colour c
  */
  template <class T>
  void point(T &t, int x, int s, int c) { t(x, s, c) = 1.0; }

  /**
     Set all space-time real elements at spin s and color c of the
     field equal to k
  */
  template <class T>
  void constant(T &t, int k, int s, int c) {
    for (int x=0; x<t.Volume(); x++) {
      // set all color-spin components to zero
      for (int s2=0; s2<t.Nspin(); s2++) {
	for (int c2=0; c2<t.Ncolor(); c2++) {
	  t(x,s2,c2) = 0.0;
	}
      }
      t(x,s,c) = k; // now set the one we want
    }
  }

  /**
     Insert a sinusoidal wave sin ( n * (x[d] / X[d]) * pi ) in dimension d
   */
  template <class P>
  void sin(P &p, int d, int n) {
    for (int t=0; t<p.X(3); t++) {
      for (int z=0; z<p.X(2); z++) {
	for (int y=0; y<p.X(1); y++) {
	  for (int x=0; x<p.X(0); x++) {
	    double mode;
	    switch (d) {
	    case 0:
	      mode = n * (double)x / p.X(0);
	      break;
	    case 1:
	      mode = n * (double)y / p.X(1);
	      break;
	    case 2:
	      mode = n * (double)z / p.X(2);
	      break;
	    case 3:
	      mode = n * (double)t / p.X(3);
	      break;
	    }
	    double k = sin (M_PI * mode);

	    int offset = ((t*p.X(2)+z)*p.X(1) + y)*p.X(0) + x;
	    int offset_h = offset / 2;
	    int parity = offset % 2;
	    int linear_index = parity * p.Volume()/2 + offset_h;

	    for (int s=0; s<p.Nspin(); s++) 
	      for (int c=0; c<p.Ncolor(); c++) 
		p(linear_index, s, c) = k;
	  }
	}
      }
    }
  }

  template <typename Float, QudaFieldOrder order>
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {
    typedef typename accessor<Float,order>::type F;
    F A(a);
    if (sourceType == QUDA_RANDOM_SOURCE) random(A);
    else if (sourceType == QUDA_POINT_SOURCE) point(A, x, s, c);
    else if (sourceType == QUDA_CONSTANT_SOURCE) constant(A, x, s, c);
    else if (sourceType == QUDA_SINUSOIDAL_SOURCE) sin(A, x, s);
    else errorQuda("Unsupported source type %d", sourceType);
  }

  // print out the vector at volume point x
  template <typename Float>
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericSource<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(a,sourceType, x, s, c);
    } else {
      errorQuda("Unsupported field order %d\n", a.FieldOrder());
    }

  }

  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      genericSource<double>(a,sourceType, x, s, c);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      genericSource<float>(a,sourceType, x, s, c);      
    } else {
      errorQuda("Precision not supported");
    }

  }


  template <class U, class V>
  int compareSpinor(const U &u, const V &v, const int tol) {
    int fail_check = 16*tol;
    int *fail = new int[fail_check];
    for (int f=0; f<fail_check; f++) fail[f] = 0;

    int N = 2*u.Nspin()*u.Ncolor();
    int *iter = new int[N];
    for (int i=0; i<N; i++) iter[i] = 0;

    for (int x=0; x<u.Volume(); x++) {
      //int test[u.Nspin()*u.Ncolor()*2];
      
      //printf("x = %d (", x);
      for (int s=0; s<u.Nspin(); s++) {
	for (int c=0; c<u.Ncolor(); c++) {
	  for (int z=0; z<2; z++) {
	    int j = (s*u.Ncolor() + c)*2+z;
	    //test[j] = 0;

	    double diff = z==0 ? fabs(u(x,s,c,z).real() - v(x,s,c,z).real()) :
	      fabs(u(x,s,c).imag() - v(x,s,c).imag());

	    for (int f=0; f<fail_check; f++) {
	      if (diff > pow(10.0,-(f+1)/(double)tol)) {
		fail[f]++;
	      }
	    }

	    if (diff > 1e-3) {
	      iter[j]++;
	      //printf("%d %d %e %e\n", x, j, u(x,s,c,z), v(x,s,c,z));
	      //test[j] = 1;
	    }
	    //printf("%d ", test[j]);

	  }
	}
      }
      //      printf(")\n");
    }

    for (int i=0; i<N; i++) printfQuda("%d fails = %d\n", i, iter[i]);
    
    int accuracy_level =0;
    for (int f=0; f<fail_check; f++) {
      if (fail[f] == 0) accuracy_level = f+1;
    }

    for (int f=0; f<fail_check; f++) {
      printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)/(double)tol), 
		 fail[f], u.Volume()*N, fail[f] / (double)(u.Volume()*N));
    }
  
    delete []iter;
    delete []fail;
  
    return accuracy_level;
  }

  template <typename oFloat, typename iFloat, QudaFieldOrder order>
  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol) {
    int ret = 0;
    typedef typename accessor<oFloat,order>::type FA;
    typedef typename accessor<iFloat,order>::type FB;
    FA A(a);
    FB B(b);
    ret = compareSpinor(A, B, tol);
    return ret;
  }


  template <typename oFloat, typename iFloat>
  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol) {
    int ret = 0;
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER &&
	a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericCompare<oFloat,iFloat,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(a, b, tol);
    } else {
      errorQuda("Unsupported field order %d\n", a.FieldOrder());
    }
    return ret;
  }


  template <typename oFloat>
  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol) {
    int ret = 0;
    if (b.Precision() == QUDA_DOUBLE_PRECISION) {
      ret = genericCompare<oFloat,double>(a, b, tol);
    } else if (b.Precision() == QUDA_SINGLE_PRECISION) {
      ret = genericCompare<oFloat,float>(a, b, tol);
    } else {
      errorQuda("Precision not supported");
    }
    return ret;
  }


  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol) {
    int ret = 0;
    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      ret = genericCompare<double>(a, b, tol);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      ret = genericCompare<float>(a, b, tol);
    } else {
      errorQuda("Precision not supported");
    }
    return ret;
  }


  template <class Order>
  void print_vector(const Order &o, unsigned int x) {

    for (int s=0; s<o.Nspin(); s++) {
      std::cout << "x = " << x << ", s = " << s << ", { ";
      for (int c=0; c<o.Ncolor(); c++) {
	std::cout << o(x, s, c) ;
	std::cout << ((c<o.Ncolor()-1) ? " , "  : " " ) ;
      }
      std::cout << "}" << std::endl;
    }

  }

  // print out the vector at volume point x
  template <typename Float, QudaFieldOrder order>
  void genericPrintVector(cpuColorSpinorField &a, unsigned int x) {
    typedef typename accessor<Float,order>::type F;
    F A(a);
    print_vector(A, x);
  }

  // print out the vector at volume point x
  template <typename Float>
  void genericPrintVector(cpuColorSpinorField &a, unsigned int x) {
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericPrintVector<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(a,x);
    } else {
      errorQuda("Unsupported field order %d\n", a.FieldOrder());
    }

  }

  // print out the vector at volume point x
  void genericPrintVector(cpuColorSpinorField &a, unsigned int x) {
  
    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      genericPrintVector<double>(a,x);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      genericPrintVector<float>(a,x);
    } else {
      errorQuda("Precision %d not implemented", a.Precision()); 
    }
    
  }



} // namespace quda

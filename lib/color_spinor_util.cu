#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>

#include <random>

namespace quda {

  using namespace colorspinor;

  const unsigned int rn_seed = 1984;
  std::ranlux24      rn_gen(rn_seed);

  /**
     Random number insertion over all field elements
  */
  template <class T>
  void random(T &t) {
    for (int parity=0; parity<t.Nparity(); parity++) {
      for (int x_cb=0; x_cb<t.VolumeCB(); x_cb++) {
	for (int s=0; s<t.Nspin(); s++) {
	  for (int c=0; c<t.Ncolor(); c++) {
	    t(parity,x_cb,s,c).real(comm_drand());
	    t(parity,x_cb,s,c).imag(comm_drand());
	  }
	}
      }
    }
  }

  /**
     Create a point source at spacetime point x, spin s and colour c
  */
  template <class T>
  void point(T &t, int x, int s, int c) { t(x%2, x/2, s, c) = 1.0; }

  /**
     Set all space-time real elements at spin s and color c of the
     field equal to k
  */
  template <class T>
  void constant(T &t, int k, int s, int c) {
    for (int parity=0; parity<t.Nparity(); parity++) {
      for (int x_cb=0; x_cb<t.VolumeCB(); x_cb++) {
	// set all color-spin components to zero
	for (int s2=0; s2<t.Nspin(); s2++) {
	  for (int c2=0; c2<t.Ncolor(); c2++) {
	    t(parity,x_cb,s2,c2) = 0.0;
	  }
	}
	t(parity,x_cb,s,c) = k; // now set the one we want
      }
    }
  }

  /**
     Insert a sinusoidal wave sin ( n * (x[d] / X[d]) * pi ) in dimension d
   */
  template <class P>
  void sin(P &p, int d, int n, int offset) {
    int coord[4];
    int X[4] = { p.X(0), p.X(1), p.X(2), p.X(3)};
    X[0] *= (p.Nparity() == 1) ? 2 : 1; // need full lattice dims

    for (int parity=0; parity<p.Nparity(); parity++) {
      for (int x_cb=0; x_cb<p.VolumeCB(); x_cb++) {
	getCoords(coord, x_cb, X, parity);

	double mode = n * (double)coord[d] / X[d];
	double k = (double)offset + sin (M_PI * mode);

	for (int s=0; s<p.Nspin(); s++)
	  for (int c=0; c<p.Ncolor(); c++)
	    p(parity, x_cb, s, c) = k;
      }
    }
  }

  // print out the vector at volume point x
  template <typename Float, int nSpin, int nColor, QudaFieldOrder order>
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {
    FieldOrderCB<Float,nSpin,nColor,1,order> A(a);
    if (sourceType == QUDA_RANDOM_SOURCE) random(A);
    else if (sourceType == QUDA_POINT_SOURCE) point(A, x, s, c);
    else if (sourceType == QUDA_CONSTANT_SOURCE) constant(A, x, s, c);
    else if (sourceType == QUDA_SINUSOIDAL_SOURCE) sin(A, x, s, c);
    else errorQuda("Unsupported source type %d", sourceType);
  }

  template <typename Float, int nSpin, QudaFieldOrder order>
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {
    if (a.Ncolor() == 2) {
      genericSource<Float,nSpin,2,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 3) {
      genericSource<Float,nSpin,3,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 4) {
      genericSource<Float,nSpin,4,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 8) {
      genericSource<Float,nSpin,8,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 12) {
      genericSource<Float,nSpin,12,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 16) {
      genericSource<Float,nSpin,16,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 20) {
      genericSource<Float,nSpin,20,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 24) {
      genericSource<Float,nSpin,24,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 32) {
      genericSource<Float,nSpin,32,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 48) {
      genericSource<Float,nSpin,48,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 96) {
      genericSource<Float,nSpin,96,order>(a,sourceType, x, s, c);
    } else {
      errorQuda("Unsupported nColor=%d\n", a.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder order>
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {
    if (a.Nspin() == 1) {
      genericSource<Float,1,order>(a,sourceType, x, s, c);
    } else if (a.Nspin() == 2) {
      genericSource<Float,2,order>(a,sourceType, x, s, c);
    } else if (a.Nspin() == 4) {
      genericSource<Float,4,order>(a,sourceType, x, s, c);
    } else {
      errorQuda("Unsupported nSpin=%d\n", a.Nspin());
    }
  }

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

  template<typename Float, int fineSpin>
  void generate2DU1Vector(cpuColorSpinorField &a, std::ranlux24 &gen) {
     if(a.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) errorQuda("\nIncorrect feild order (%d) or null vector index (%d).\n", a.FieldOrder());
     quda::colorspinor::FieldOrderCB<Float,fineSpin,3,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> aOrder(static_cast<ColorSpinorField&>(a));

     //blas::zero(a);
     warningQuda("\nProcessing 2d null vector.");

     std::normal_distribution<> dist(0.0, 1.0);

     const int c = 0;

     for (int parity = 0; parity < 2; parity++) {
        for(int x_cb = 0; x_cb < a.VolumeCB(); x_cb++) {

           int i = parity*a.VolumeCB() + x_cb;
           int xx[4] = {0};
           a.LatticeIndex(xx, i);

           if( xx[2] == 0 && xx[3] == 0 )
           {
             std::complex<Float> rnd_val = std::complex<Float>(static_cast<Float>(dist(gen)), static_cast<Float>(dist(gen)));
             aOrder(parity, x_cb, 0, c) = rnd_val; 
           }
       }
     }
  }

  void generic2DSource(cpuColorSpinorField &a) {

    if(a.Nspin() == 4) errorQuda("\nUnsupported spin\n");

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      if     ( a.Nspin() == 1 ) generate2DU1Vector<double, 1>(a, rn_gen);
      else if( a.Nspin() == 2 ) generate2DU1Vector<double, 2>(a, rn_gen);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
  
      if     ( a.Nspin() == 1 ) generate2DU1Vector<float, 1>(a, rn_gen);
      else if( a.Nspin() == 2 ) generate2DU1Vector<float, 2>(a, rn_gen);
      
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

    for (int parity=0; parity<v.Nparity(); parity++) {
      for (int x_cb=0; x_cb<u.VolumeCB(); x_cb++) {

	for (int s=0; s<u.Nspin(); s++) {
	  for (int c=0; c<u.Ncolor(); c++) {
	    for (int z=0; z<2; z++) {
	      int j = (s*u.Ncolor() + c)*2+z;

	      double diff = z==0 ? fabs(u(parity,x_cb,s,c,z).real() - v(parity,x_cb,s,c,z).real()) :
		fabs(u(parity,x_cb,s,c).imag() - v(parity,x_cb,s,c).imag());

	      for (int f=0; f<fail_check; f++) {
		if (diff > pow(10.0,-(f+1)/(double)tol)) {
		  fail[f]++;
		}
	      }

	      if (diff > 1e-3) iter[j]++;
	    }
	  }
	}
      }
    }

    // reduce over all processes
    for (int i=0; i<N; i++) comm_allreduce_int(&iter[i]);
    for (int f=0; f<fail_check; f++) comm_allreduce_int(&fail[f]);

    for (int i=0; i<N; i++) printfQuda("%d fails = %d\n", i, iter[i]);

    int accuracy_level =0;
    for (int f=0; f<fail_check; f++) {
      if (fail[f] == 0) accuracy_level = f+1;
    }

    size_t total = u.Nparity()*u.VolumeCB()*N*comm_size();
    for (int f=0; f<fail_check; f++) {
      printfQuda("%e Failures: %d / %lu  = %e\n", pow(10.0,-(f+1)/(double)tol),
		 fail[f], total, fail[f] / (double)total);
    }

    delete []iter;
    delete []fail;

    return accuracy_level;
  }

  template <typename oFloat, typename iFloat, QudaFieldOrder order>
  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol) {
    int ret = 0;
    if (a.Ncolor() == 3) {
      const int Nc = 3;
      if (a.Nspin() == 4) {
	const int Ns = 4;
	FieldOrderCB<oFloat,Ns,Nc,1,order> A(a);
	FieldOrderCB<iFloat,Ns,Nc,1,order> B(b);
	ret = compareSpinor(A, B, tol);
      } else if (a.Nspin() == 1) {
	const int Ns = 1;
	FieldOrderCB<oFloat,Ns,Nc,1,order> A(a);
	FieldOrderCB<iFloat,Ns,Nc,1,order> B(b);
	ret = compareSpinor(A, B, tol);
      }
    } else {
      errorQuda("Number of colors %d not supported", a.Ncolor());
    }
    return ret;
  }


  template <typename oFloat, typename iFloat>
  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol) {
    int ret = 0;
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER &&
	a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      ret = genericCompare<oFloat,iFloat,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(a, b, tol);
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

    int x_cb = x / o.Nparity();
    int parity = x%o.Nparity();

    for (int s=0; s<o.Nspin(); s++) {
      std::cout << "x = " << x << ", s = " << s << ", { ";
      for (int c=0; c<o.Ncolor(); c++) {
	std::cout << o(parity, x_cb, s, c) ;
	std::cout << ((c<o.Ncolor()-1) ? " , "  : " " ) ;
      }
      std::cout << "}" << std::endl;
    }

  }

  // print out the vector at volume point x
  template <typename Float, QudaFieldOrder order>
  void genericPrintVector(cpuColorSpinorField &a, unsigned int x) {
    if (a.Ncolor() == 3 && a.Nspin() == 4)  {
      FieldOrderCB<Float,4,3,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 2 && a.Nspin() == 2) {
      FieldOrderCB<Float,2,2,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 24 && a.Nspin() == 2) {
      FieldOrderCB<Float,2,24,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 6 && a.Nspin() == 4) {
      FieldOrderCB<Float,4,6,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 72 && a.Nspin() == 4) {
      FieldOrderCB<Float,4,72,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 576 && a.Nspin() == 2) {
      FieldOrderCB<Float,2,576,1,order> A(a);
      print_vector(A, x);
    }    
    else {
      errorQuda("Not supported Ncolor = %d, Nspin = %d", a.Ncolor(), a.Nspin());	 
    }
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

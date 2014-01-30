#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

namespace quda {

  using namespace colorspinor;

  // Random number insertion over all field elements
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

  // Create a point source at spacetime point x, spin s and colour c
  template <class T>
  void point(T &t, int x, int s, int c) { t(x, s, c, 0) = 1.0; }

  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      FieldOrder<double> *A = createOrder<double>(a);
      if (sourceType == QUDA_RANDOM_SOURCE) random(*A);
      else if (sourceType == QUDA_POINT_SOURCE) point(*A, x, s, c);
      else errorQuda("Unsupported source type %d", sourceType);
      delete A;
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      FieldOrder<float> *A = createOrder<float>(a);
      if (sourceType == QUDA_RANDOM_SOURCE) random(*A);
      else if (sourceType == QUDA_POINT_SOURCE) point(*A, x, s, c);
      else errorQuda("Unsupported source type %d", sourceType);
      delete A;
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

  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol) {
    int ret = 0;
    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      FieldOrder<double> *A = createOrder<double>(a);
      if (b.Precision() == QUDA_DOUBLE_PRECISION) {
	FieldOrder<double> *B = createOrder<double>(b);
	ret = compareSpinor(*A, *B, tol);
	delete B;
      } else {
	FieldOrder<float> *B = createOrder<float>(b);
	ret = compareSpinor(*A, *B, tol);
	delete B;
      }
      delete A;
    } else {
      FieldOrder<float> *A = createOrder<float>(a);
      if (b.Precision() == QUDA_DOUBLE_PRECISION) {
	FieldOrder<double> *B = createOrder<double>(b);
	ret = compareSpinor(*A, *B, tol);
	delete B;
      } else {
	FieldOrder<float> *B = createOrder<float>(b);
	ret = compareSpinor(*A, *B, tol);
	delete B;
      }
      delete A;
    }
    return ret;
  }


  template <class Order>
  void print_vector(const Order &o, unsigned int x) {

    for (int s=0; s<o.Nspin(); s++) {
      std::cout << "x = " << x << ", s = " << s << ", { ";
      for (int c=0; c<o.Ncolor(); c++) {
	std::cout << " ( " << o(x, s, c, 0) << " , " ;
	if (c<o.Ncolor()-1) std::cout << o(x, s, c, 1) << " ) ," ;
	else std::cout << o(x, s, c, 1) << " ) " ;
      }
      std::cout << " } " << std::endl;
    }

  }

  // print out the vector at volume point x
  void genericPrintVector(cpuColorSpinorField &a, unsigned int x) {
  
    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      FieldOrder<double> *A = createOrder<double>(a);
      print_vector(*A, x);
      delete A;
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      FieldOrder<float> *A = createOrder<float>(a);
      print_vector(*A, x);
      delete A;
    } else {
      errorQuda("Precision %d not implemented", a.Precision()); 
    }
    
  }



} // namespace quda

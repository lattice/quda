#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>

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

  // print out the vector at volume point x
  template <typename Float, int nSpin, int nColor, QudaFieldOrder order>
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {
    FieldOrder<Float,nSpin,nColor,1,order> A(a);
    if (sourceType == QUDA_RANDOM_SOURCE) random(A);
    else if (sourceType == QUDA_POINT_SOURCE) point(A, x, s, c);
    else if (sourceType == QUDA_CONSTANT_SOURCE) constant(A, x, s, c);
    else if (sourceType == QUDA_SINUSOIDAL_SOURCE) sin(A, x, s);
    else errorQuda("Unsupported source type %d", sourceType);
  }

  template <typename Float, int nSpin, QudaFieldOrder order>
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c) {
    if (a.Ncolor() == 2) {
      genericSource<Float,nSpin,2,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 3) {
      genericSource<Float,nSpin,3,order>(a,sourceType, x, s, c);
    } else if (a.Ncolor() == 24) {
      genericSource<Float,nSpin,24,order>(a,sourceType, x, s, c);
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
    if (a.Ncolor() == 3) {
      const int Nc = 3;
      if (a.Nspin() == 4) {
	const int Ns = 4;
	FieldOrder<oFloat,Ns,Nc,1,order> A(a);
	FieldOrder<iFloat,Ns,Nc,1,order> B(b);
	ret = compareSpinor(A, B, tol);
      } else if (a.Nspin() == 1) {
	const int Ns = 1;
	FieldOrder<oFloat,Ns,Nc,1,order> A(a);
	FieldOrder<iFloat,Ns,Nc,1,order> B(b);
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
    if (a.Ncolor() == 3 && a.Nspin() == 4)  {
      FieldOrder<Float,4,3,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 2 && a.Nspin() == 2) {
      FieldOrder<Float,2,2,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 24 && a.Nspin() == 2) {
      FieldOrder<Float,2,24,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 72 && a.Nspin() == 4) {
      FieldOrder<Float,2,24,1,order> A(a);
      print_vector(A, x);
    }
    else if (a.Ncolor() == 576 && a.Nspin() == 2) {
      FieldOrder<Float,2,576,1,order> A(a);
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

  struct PackGhostArg {

    int X[QUDA_MAX_DIM];
    const int nDim;
    const int nFace;
    const int parity;
    const int dagger;
    const QudaDWFPCType pc_type;

    PackGhostArg(const ColorSpinorField &a, int parity, int dagger)
      : nDim(a.Ndim()),
	nFace(a.Nspin() == 1 ? 3 : 1),
	parity(parity), dagger(dagger),
	pc_type(a.DWFPCtype())
    {
      for (int d=0; d<nDim; d++) X[d] = a.X(d);
      X[0] *= 2; // set to full lattice size
      X[4] = (nDim == 5) ? a.X(4) : 1; // set fifth dimension correctly
    }

  };

  template <typename Float, int Ns, int Nc>
  __host__ void packGhost(void **ghost, const ColorSpinorField &a, PackGhostArg &arg, int cb_idx) {
    int spinor_size = 2*Ns*Nc*sizeof(Float);

    const int *X = arg.X;
    int x[5] = { };
    if (arg.nDim == 5)  getCoords5(x, cb_idx, X, arg.parity, arg.pc_type);
    else getCoords(x, cb_idx, X, arg.parity);

    const void *v = a.V();

    if (x[0] < arg.nFace){
      int ghost_face_idx = (x[0]*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] + x[1])>>1;
      memcpy( ((char*)ghost[0]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

    if (x[0] >= X[0] - arg.nFace){
      int ghost_face_idx = ((x[0]-X[0]+arg.nFace)*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1]) + x[2]*X[1] + x[1])>>1;
      memcpy( ((char*)ghost[1]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

    if (x[1] < arg.nFace){
      int ghost_face_idx = (x[1]*X[4]*X[3]*X[2]*X[0] + x[4]*X[3]*X[2]*X[0] + x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1;
      memcpy( ((char*)ghost[2]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

    if (x[1] >= X[1] - arg.nFace){
      int ghost_face_idx = ((x[1]-X[1]+arg.nFace)*X[4]*X[3]*X[2]*X[0] +x[4]*X[3]*X[2]*X[0]+ x[3]*X[2]*X[0] + x[2]*X[0] + x[0])>>1;
      memcpy( ((char*)ghost[3]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

    if (x[2] < arg.nFace){
      int ghost_face_idx = (x[2]*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
      memcpy( ((char*)ghost[4]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

    if (x[2] >= X[2] - arg.nFace){
      int ghost_face_idx = ((x[2]-X[2]+arg.nFace)*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0] + x[1]*X[0] + x[0])>>1;
      memcpy( ((char*)ghost[5]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

    if (x[3] < arg.nFace){
      int ghost_face_idx = (x[3]*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
      memcpy( ((char*)ghost[6]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

    if (x[3] >= X[3] - arg.nFace){
      int ghost_face_idx = ((x[3]-X[3]+arg.nFace)*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0] + x[0])>>1;
      memcpy( ((char*)ghost[7]) + ghost_face_idx*spinor_size, ((char*)v)+cb_idx*spinor_size, spinor_size);
    }

  }

  template <typename Float, int Ns, int Nc>
  void GenericPackGhost(void **ghost, const ColorSpinorField &a, PackGhostArg &arg) {
    for (int i=0; i<a.VolumeCB(); i++) packGhost<Float,Ns,Nc>(ghost, a, arg, i);
  }


  template <typename Float, int Ns>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    PackGhostArg arg(a, parity, dagger);

    if (a.Ncolor() == 3) {
      GenericPackGhost<Float,Ns,3>(ghost, a, arg);
    } else {
      errorQuda("Unsupported nColor = %d", a.Ncolor());
    }

  }

  template <typename Float>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.Nspin() == 4) {
      genericPackGhost<Float,4>(ghost, a, parity, dagger);
    } else if (a.Nspin() == 2) {
      genericPackGhost<Float,2>(ghost, a, parity, dagger);
    } else if (a.Nspin() == 1) {
      genericPackGhost<Float,1>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported nSpin = %d", a.Nspin());
    }

  }

  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.SiteSubset() == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in packGhost for cpu");
    }

    if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      errorQuda("Field order %d not supported", a.FieldOrder());
    }

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      genericPackGhost<double>(ghost, a, parity, dagger);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      genericPackGhost<float>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported precision %d", a.Precision());
    }

  }


} // namespace quda

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <complex>

#include <quda.h>
#include <gauge_field.h>


//namespace quda{
#define RETURN_IF_ERR if(err) return;

extern int gauge_order;
extern int Vh;
extern int Vh_ex;

  static int OPP_DIR(int dir){ return 7-dir; }
  static bool GOES_FORWARDS(int dir){ return (dir<=3); }
  static bool GOES_BACKWARDS(int dir){ return (dir>3); }

  template<int N>
	struct Sign{
	  static const int result = 1;
	};

   template<>
	struct Sign<1>
	{
	  static const int result = -1;
	};


  template<class T, class U>
  struct Promote
  {
    typedef T Type;
  };

  template<>
  struct Promote<int,float>
  {
    typedef float Type;
  };

  template<>
  struct Promote<float,int>
  {
    typedef float Type;
  };

  template<>
  struct Promote<int, double>
  {
    typedef double Type;
  };

  template<>
  struct Promote<double, int>
  {
    typedef double Type;
  };

  template<>
  struct Promote<float, double>
  {
    typedef double Type;
  };

  template<>
  struct Promote<double, float>
  {
    typedef double Type;
  };

  template<>
  struct Promote< int, std::complex<float> >
  {
    typedef std::complex<float> Type;
  };

  template<>
  struct Promote< std::complex<float>, int >
  {
    typedef std::complex<float> Type;
  };

  template<> 
  struct Promote< float, std::complex<float> >
  {
    typedef std::complex<float> Type;
  };

  template<>
  struct Promote< int, std::complex<double> >
  {
    typedef std::complex<double> Type;
  };

  template<>
  struct Promote< std::complex<double>, int > 
  {
    typedef std::complex<double> Type;
  };

  template<> 
  struct Promote< float, std::complex<double> > 
  { 
    typedef std::complex<double> Type;
  };

  template<>
  struct Promote< std::complex<double>, float >
  {
    typedef std::complex<double> Type;
  };

  template<> 
  struct Promote< double, std::complex<double> >
  {
    typedef std::complex<double> Type;
  };

  template<>
  struct Promote< std::complex<double>, double >
  {
    typedef std::complex<double> Type;
  };

  template<int N, class T>
  class Matrix 
  {
    private:
      T data[N][N];
	
    public: 
      Matrix(); // default constructor
      Matrix(const Matrix<N,T>& mat); // copy constructor	
      Matrix & operator += (const Matrix<N,T>& mat);
      Matrix & operator -= (const Matrix<N,T>& mat);
      const T& operator()(int i, int j) const;
      T& operator()(int i, int j);
  };
   
  template<int N, class T>
  Matrix<N,T>::Matrix()
  {
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	data[i][j] = static_cast<T>(0);
      }
    }
  }

  template<int N, class T>
  Matrix<N,T>::Matrix(const Matrix<N,T>& mat)
  {
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	data[i][j] = mat.data[i][j];
      }
    }
  }

  template<int N, class T>
  T& Matrix<N,T>::operator()(int i, int j)
  {
    return data[i][j];
  }

  template<int N, class T>
  const T& Matrix<N,T>::operator()(int i, int j) const 
  {
    return data[i][j];
  }	

  template<int N, class T>
  Matrix<N,T> & Matrix<N,T>::operator+=(const Matrix<N,T>& mat)
  {
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	data[i][j] += mat.data[i][j];
      }
    }
    return *this;
  }

  template<int N, class T>
  Matrix<N,T> & Matrix<N,T>::operator-=(const Matrix<N,T>& mat)
  {
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	data[i][j] -= mat.data[i][j];
      }
    }
    return *this;
  }

  template<int N, class T> 
  Matrix<N,T> operator+(const Matrix<N,T>& a, const Matrix<N,T>& b)
  {
    Matrix<N,T> result(a);
    result += b;
    return result;
  }

  template<int N, class T>
  Matrix<N,T> operator-(const Matrix<N,T>& a, const Matrix<N,T>& b)
  {
    Matrix<N,T> result(a);
    result -= b;
    return result;
  }

  template<int N, class T>
  Matrix<N,T> operator*(const Matrix<N,T>& a, const Matrix<N,T>& b)
  {
    Matrix<N,T> result;
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
        result(i,j) = static_cast<T>(0);
	for(int k=0; k<N; ++k){
	  result(i,j) += a(i,k)*b(k,j);
	}
      }
    }
    return result;
  }

  template<int N, class T>
  Matrix<N,std::complex<T> > conj(const Matrix<N,std::complex<T> >& mat)
  {
    Matrix<N,std::complex<T> > result;
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	result(i,j) = std::conj(mat(j,i));
      }
    }
    return result;
  }

  template<int N, class T> 
  Matrix<N,T> transpose(const Matrix<N,std::complex<T> >& mat)
  {
    Matrix<N,T> result;
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	result(i,j) = mat(j,i);
      }
    }
    return result;
  }

  template<int N, class T, class U>
  Matrix<N, typename Promote<T,U>::Type> operator*(const Matrix<N,T>& mat, const U& scalar)
  {
    typedef typename Promote<T,U>::Type return_type;
    Matrix<N,return_type> result;

    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	result(i,j) = scalar*mat(i,j);
      }	
    }	 
    return result;
  }

  template<int N, class T, class U>
  Matrix<N, typename Promote<T,U>::Type> operator*(const U& scalar, const Matrix<N,T>& mat)
  {
    return mat*scalar;	
  }

  template<int N, class T>
  struct Identity
  {
    Matrix<N,T> operator()() const {
      Matrix<N,T> id;
      for(int i=0; i<N; ++i){
	id(i,i) = static_cast<T>(1);
      }
      return id;
    } // operator()
  };

  template<int N, class T>
  struct Zero
  {
    // the default constructor zeros all matrix elements
    Matrix<N,T> operator()() const {
      return Matrix<N,T>();
    }
  };


  template<int N, class T>
    std::ostream & operator << (std::ostream & os, const Matrix<N,T> & m)
  {
    for(int i=0; i<N; ++i){
      for(int j=0; j<N; ++j){
	os << m(i,j) << " ";
      }
      if(i<N-1) os << std::endl;
    }
    return os;
  }



  template<class Real>
  class LoadStore{
    private: 
     const int volume;
    const int half_volume;
    public:
    LoadStore(int vol) : volume(vol), half_volume(vol/2) {}

    void loadMatrixFromField(const Real* const field, int oddBit, int half_lattice_index, Matrix<3, std::complex<Real> >* const mat) const;
     
    void loadMatrixFromField(const Real* const field, int oddBit, int dir, int half_lattice_index, Matrix<3, std::complex<Real> >* const mat) const;

      void storeMatrixToField(const Matrix<3, std::complex<Real> >& mat, int oddBit, int half_lattice_index, Real* const field) const;
      
      void addMatrixToField(const Matrix<3, std::complex<Real> >& mat, int oddBit, int half_lattice_index, Real coeff, Real* const) const;

      void addMatrixToField(const Matrix<3, std::complex<Real> >& mat, int oddBit, int dir, int half_lattice_index, Real coeff, Real* const) const;
      
     void storeMatrixToMomentumField(const Matrix<3, std::complex<Real> >& mat, int oddBit, int dir, int half_lattice_index, Real coeff, Real* const) const;
    Real getData(const Real* const field, int idx, int dir, int oddBit, int offset, int hfv) const;
    void addData(Real* const field, int idx, int dir, int oddBit, int offset, Real, int hfv) const;
    int half_idx_conversion_ex2normal(int half_lattice_index, const int* dim, int oddBit) const ;
    int half_idx_conversion_normal2ex(int half_lattice_index, const int* dim, int oddBit) const ;
 };

template<class Real>
int LoadStore<Real>::half_idx_conversion_ex2normal(int half_lattice_index_ex, const int* dim, int oddBit) const
{
  int X1=dim[0];
  int X2=dim[1];
  int X3=dim[2];
  //int X4=dim[3];
  int X1h=X1/2;
  
  int E1=dim[0]+4;
  int E2=dim[1]+4;
  int E3=dim[2]+4;
  //int E4=dim[3]+4;
  int E1h=E1/2;

  int sid = half_lattice_index_ex;

  int za = sid/E1h;
  int x1h = sid - za*E1h;
  int zb = za/E2;
  int x2 = za - zb*E2;
  int x4 = zb/E3;
  int x3 = zb - x4*E3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2*x1h + x1odd;

  int idx = ((x4-2)*X3*X2*X1 + (x3-2)*X2*X1+(x2-2)*X1+(x1-2))/2;
  return idx;
}

template<class Real>
int LoadStore<Real>::half_idx_conversion_normal2ex(int half_lattice_index, const int* dim, int oddBit) const
{
  int X1=dim[0];
  int X2=dim[1];
  int X3=dim[2];
  //int X4=dim[3];
  int X1h=X1/2;

  int E1=dim[0]+4;
  int E2=dim[1]+4;
  int E3=dim[2]+4;
  //int E4=dim[3]+4;
  //int E1h=E1/2;

  int sid = half_lattice_index;

  int za = sid/X1h;
  int x1h = sid - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2*x1h + x1odd;

  int idx = ((x4+2)*E3*E2*E1 + (x3+2)*E2*E1+(x2+2)*E1+(x1+2))/2;    
  
  return idx;
}

template<class Real>
Real LoadStore<Real>::getData(const Real* const field, int idx, int dir, int oddBit, int offset, int hfv) const
{
  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    return  field[(4*hfv*oddBit +4*idx + dir)*18+offset];
  }else{ //QDP format
    return  ((Real**)field)[dir][(hfv*oddBit+idx)*18 +offset];
  }
}
template<class Real>
void LoadStore<Real>::addData(Real* const field, int idx, int dir, int oddBit, int offset, Real v, int hfv) const
{
  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    field[(4*hfv*oddBit +4*idx + dir)*18+offset] += v;
  }else{ //QDP format
    ((Real**)field)[dir][(hfv*oddBit+idx)*18 +offset] += v;
  }
}



template<class Real>
  void LoadStore<Real>::loadMatrixFromField(const Real* const field, 
					    int oddBit,
			   		    int half_lattice_index, 
			   		    Matrix<3, std::complex<Real> >* const mat
					    ) const
{ 
#ifdef MULTI_GPU
  int hfv = Vh_ex;
#else
  int hfv = Vh;
#endif

   int offset = 0;
   for(int i=0; i<3; ++i){
     for(int j=0; j<3; ++j){
       (*mat)(i,j) = (*(field + (oddBit*hfv + half_lattice_index)*18 + offset++));
       (*mat)(i,j) += std::complex<Real>(0, *(field + (oddBit*hfv + half_lattice_index)*18 + offset++));
     }
   }
    return;
  }

  template<class Real>
  void LoadStore<Real>::loadMatrixFromField(const Real* const field, 
					    int oddBit,
					    int dir,
			   		    int half_lattice_index, 
			   		    Matrix<3, std::complex<Real> >* const mat
					    ) const
  {
#ifdef MULTI_GPU
    int hfv = Vh_ex;
#else
    int hfv = Vh;
#endif

    //const Real* const local_field = field + ((oddBit*half_volume + half_lattice_index)*4 + dir)*18;
    int offset = 0;
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
	(*mat)(i,j) = (getData(field, half_lattice_index, dir, oddBit, offset++, hfv));
	(*mat)(i,j) += std::complex<Real>(0, getData(field, half_lattice_index, dir, oddBit, offset++, hfv));
      }
    }
    return;
  }

  template<class Real> 
   void LoadStore<Real>::storeMatrixToField(const Matrix<3, std::complex<Real> >& mat,
					    int oddBit, 
					    int half_lattice_index, 
					    Real* const field) const
  {
#ifdef MULTI_GPU
    int hfv = Vh_ex;
#else
    int hfv = Vh;
#endif

    int offset = 0;
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        *(field + (oddBit*hfv + half_lattice_index)*18 + offset++) = (mat)(i,j).real();
        *(field + (oddBit*hfv + half_lattice_index)*18 + offset++) = (mat)(i,j).imag();
      }
    }
    return;
  }

  template<class Real> 
   void LoadStore<Real>::addMatrixToField(const Matrix<3, std::complex<Real> >& mat,
					    int oddBit, 
					    int half_lattice_index, 
					    Real coeff,
					    Real* const field) const
  {
#ifdef MULTI_GPU
    int hfv = Vh_ex;
#else
    int hfv = Vh;
#endif
    Real* const local_field = field + (oddBit*hfv + half_lattice_index)*18;

    
    int offset = 0;
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        local_field[offset++] += coeff*mat(i,j).real();
        local_field[offset++] += coeff*mat(i,j).imag();
      }
    }
    return;
  }


 template<class Real> 
   void LoadStore<Real>::addMatrixToField(const Matrix<3, std::complex<Real> >& mat,
					    int oddBit, 
					    int dir,
					    int half_lattice_index, 
					    Real coeff,
					    Real* const field) const
  {
    
#ifdef MULTI_GPU
    int hfv = Vh_ex;
#else
    int hfv = Vh;
#endif

    //Real* const local_field = field + ((oddBit*half_volume + half_lattice_index)*4 + dir)*18;
    int offset = 0;
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        //local_field[offset++] += coeff*mat(i,j).real();
	addData(field, half_lattice_index, dir, oddBit, offset++, coeff*mat(i,j).real(), hfv);

        //local_field[offset++] += coeff*mat(i,j).imag();
	addData(field, half_lattice_index, dir, oddBit, offset++, coeff*mat(i,j).imag(), hfv);
      }
    }
    return;
  }


  template<class Real> 
    void LoadStore<Real>::storeMatrixToMomentumField(const Matrix<3, std::complex<Real> >& mat,
						     int oddBit, 
						     int dir, 
						     int half_lattice_index, 
						     Real coeff,
						     Real* const field) const
  {
     Real* const mom_field = field + ((oddBit*half_volume + half_lattice_index)*4 + dir)*10;
     mom_field[0] = (mat(0,1).real() - mat(1,0).real())*0.5*coeff;
     mom_field[1] = (mat(0,1).imag() + mat(1,0).imag())*0.5*coeff;
     
     mom_field[2] = (mat(0,2).real() - mat(2,0).real())*0.5*coeff;
     mom_field[3] = (mat(0,2).imag() + mat(2,0).imag())*0.5*coeff;

     mom_field[4] = (mat(1,2).real() - mat(2,1).real())*0.5*coeff;
     mom_field[5] = (mat(1,2).imag() + mat(2,1).imag())*0.5*coeff;

     const Real temp = (mat(0,0).imag() + mat(1,1).imag() + mat(2,2).imag())*0.3333333333333333333;
     mom_field[6]    = (mat(0,0).imag() - temp)*coeff;
     mom_field[7]    = (mat(1,1).imag() - temp)*coeff;
     mom_field[8]    = (mat(2,2).imag() - temp)*coeff;
     mom_field[9]    = 0.0;
    return;
  }


  
  template<int oddBit>
  struct Locator
  {
     private: 
	int local_dim[4];
	int volume;
        int half_index;
        int full_index;
        int full_coord[4];
        void getCoordsFromHalfIndex(int half_index, int coord[4]);
        void getCoordsFromFullIndex(int full_index, int coord[4]);
        void cache(int half_lattice_index); // caches the half-lattice index, full-lattice index, 
					    // and full-lattice coordinates
   

     public:
	Locator(const int dim[4]); 
	int getFullFromHalfIndex(int half_lattice_index);
    int getNeighborFromFullIndex(int full_lattice_index, int dir, int* err=NULL);
 };

  template<int oddBit>
  Locator<oddBit>::Locator(const int dim[4]) : half_index(-1), full_index(-1){
    volume = 1;
    for(int dir=0; dir<4; ++dir){
      local_dim[dir] = dim[dir];
      volume *= local_dim[dir];
    }
  }

  // Store the half_lattice index, works out and stores the full lattice index 
  // and the coordinates.
  template<int oddBit>
  void Locator<oddBit>::getCoordsFromHalfIndex(int half_lattice_index, int coord[4])
  {
#ifdef MULTI_GPU
    int E1 = local_dim[0]+4;
    int E2 = local_dim[1]+4;
    int E3 = local_dim[2]+4;
    //int E4 = local_dim[3]+4;
    int E1h = E1/2;
    
    int z1    = half_lattice_index/E1h;
    int x1h   = half_lattice_index - z1*E1h;
    int z2    = z1/E2;
    coord[1]      = z1 - z2*E2;
    coord[3]      = z2/E3;
    coord[2]      = z2 - coord[3]*E3;
    int x1odd = (coord[1] + coord[2] + coord[3] + oddBit) & 1;
    coord[0]  = 2*x1h + x1odd;
#else
    int half_dim_0 = local_dim[0]/2;
    int z1    = half_lattice_index/half_dim_0;
    int x1h   = half_lattice_index - z1*half_dim_0;
    int z2    = z1/local_dim[1];
    coord[1]      = z1 - z2*local_dim[1];
    coord[3]      = z2/local_dim[2];
    coord[2]      = z2 - coord[3]*local_dim[2];
    int x1odd = (coord[1] + coord[2] + coord[3] + oddBit) & 1;
    coord[0]  = 2*x1h + x1odd;
#endif

  }

  template<int oddBit>
  void Locator<oddBit>::getCoordsFromFullIndex(int full_lattice_index, int coord[4])
  {
#ifdef MULTI_GPU
    int D1=local_dim[0]+4;
    int D2=local_dim[1]+4;
    int D3=local_dim[2]+4;
    //int D4=local_dim[3]+4;
    //int D1h=D1/2;
#else
    int D1=local_dim[0];
    int D2=local_dim[1];
    int D3=local_dim[2];
    //int D4=local_dim[3];
    //int D1h=D1/2;
#endif    

    int z1        = full_lattice_index/D1;
    coord[0]      = full_lattice_index - z1*D1;
    int z2        = z1/D2;
    coord[1]      = z1 - z2*D2;
    coord[3]      = z2/D3;
    coord[2]      = z2 - coord[3]*D3;
  }




  
  template<int oddBit>
  void Locator<oddBit>::cache(int half_lattice_index) 
  {
    half_index = half_lattice_index;
    getCoordsFromHalfIndex(half_lattice_index, full_coord);
    int x1odd  = (full_coord[1] + full_coord[2] + full_coord[3] + oddBit) & 1;
    full_index = 2*half_lattice_index + x1odd;
    return;
  }

  template<int oddBit>
  int Locator<oddBit>::getFullFromHalfIndex(int half_lattice_index) 
  {
    if(half_index != half_lattice_index) cache(half_lattice_index);
    return full_index;
  }

  // From full index return the neighbouring full index
  template<int oddBit> 
  int Locator<oddBit>::getNeighborFromFullIndex(int full_lattice_index, int dir, int* err)
  {
    if(err) *err = 0;
    
     int coord[4];
     int neighbor_index;
     getCoordsFromFullIndex(full_lattice_index, coord);
#ifdef MULTI_GPU
     int E1 = local_dim[0] + 4;
     int E2 = local_dim[1] + 4;
     int E3 = local_dim[2] + 4;
     int E4 = local_dim[3] + 4;
     switch(dir){
     case 0:  //+X
       neighbor_index =  full_lattice_index + 1;
       if(err && (coord[0] == E1-1) ) *err = 1;
       break; 
     case 1:  //+Y
       neighbor_index =  full_lattice_index + E1;
       if(err && (coord[1] == E2-1) ) *err = 1;
       break;
     case 2:  //+Z
       neighbor_index =  full_lattice_index + E2*E1;
       if(err && (coord[2] == E3-1) ) *err = 1;
       break;
     case 3:  //+T
       neighbor_index = full_lattice_index + E3*E2*E1;
       if(err && (coord[3] == E4-1) ) *err = 1;
       break;
     case 7:  //-X
       neighbor_index = full_lattice_index - 1;
       if(err && (coord[0] == 0) ) *err = 1;
       break;
     case 6:  //-Y
       neighbor_index = full_lattice_index - E1;
       if(err && (coord[1] == 0) ) *err = 1;
       break;
     case 5:  //-Z
       neighbor_index = full_lattice_index - E2*E1;
       if(err && (coord[2] == 0) ) *err = 1;
       break;
     case 4:  //-T
       neighbor_index = full_lattice_index - E3*E2*E1;
       if(err && (coord[3] == 0) ) *err = 1;
       break;
     default:
       errorQuda("Neighbor index could not be determined\n");
       exit(1);
       break;
     } // switch(dir)
     
#else
     switch(dir){
	case 0:
	  neighbor_index = (coord[0] == local_dim[0]-1) ? full_lattice_index + 1 - local_dim[0] : full_lattice_index + 1;
	  break;
	case 1:
	  neighbor_index = (coord[1] == local_dim[1]-1) ? full_lattice_index + local_dim[0]*(1 - local_dim[1]) : full_lattice_index + local_dim[0];
	  break;
	case 2:
	  neighbor_index = (coord[2] == local_dim[2]-1) ? full_lattice_index + local_dim[0]*local_dim[1]*(1 - local_dim[2]) : full_lattice_index + local_dim[0]*local_dim[1];
	  break;
        case 3:
	  neighbor_index = (coord[3] == local_dim[3]-1) ? full_lattice_index + local_dim[0]*local_dim[1]*local_dim[2]*(1-local_dim[3]) : full_lattice_index + local_dim[0]*local_dim[1]*local_dim[2];	  
	  break;
	case 7:
	  neighbor_index = (coord[0] == 0) ? full_lattice_index - 1 + local_dim[0] : full_lattice_index - 1;
	  break;
	case 6:
	  neighbor_index = (coord[1] == 0) ? full_lattice_index  - local_dim[0]*(1 - local_dim[1]) : full_lattice_index - local_dim[0];
	  break;
	case 5:
	  neighbor_index = (coord[2] == 0) ? full_lattice_index - local_dim[0]*local_dim[1]*(1 - local_dim[2]) : full_lattice_index - local_dim[0]*local_dim[1];
	  break;
	case 4:
	  neighbor_index = (coord[3] == 0) ? full_lattice_index - local_dim[0]*local_dim[1]*local_dim[2]*(1 - local_dim[3]) : full_lattice_index - local_dim[0]*local_dim[1]*local_dim[2];
	  break;
	default: 
	  errorQuda("Neighbor index could not be determined\n");
	  exit(1);
	  break;
     } // switch(dir)
     if(err) *err = 0;
#endif
     return neighbor_index;
  }

// Can't typedef a template 
template<class Real>
struct ColorMatrix
{
  typedef Matrix<3, std::complex<Real> > Type;
};

  template<class Real, int oddBit> 
  void computeOneLinkSite(const int dim[4], 
			  int half_lattice_index, 		
			   const Real* const oprod,
		           int sig, Real coeff,	
			   const LoadStore<Real>& ls,
			   Real* const output)
   {
     if( GOES_FORWARDS(sig) ){
       typename ColorMatrix<Real>::Type colorMatW;
#ifdef MULTI_GPU
       int idx = ls.half_idx_conversion_normal2ex(half_lattice_index, dim, oddBit);
#else
       int idx = half_lattice_index;
#endif
       ls.loadMatrixFromField(oprod, oddBit, sig, idx, &colorMatW);
       ls.addMatrixToField(colorMatW, oddBit, sig, idx, coeff, output);
     } 
     return;
   }

  template<class Real>
    void computeOneLinkField(const int dim[4], 
			     const Real* const oprod,
			     int sig, Real coeff,
			     Real* const output)
   {
     int volume = 1;
     for(int dir=0; dir<4; ++dir) volume *= dim[dir];
     const int half_volume = volume/2;
     LoadStore<Real> ls(volume);
     for(int site=0; site<half_volume; ++site){
       computeOneLinkSite<Real,0>(dim, site, 
			   oprod, 
			   sig, coeff, ls,
			   output);
			 
     }
     // Loop over odd lattice sites
     for(int site=0; site<half_volume; ++site){
       computeOneLinkSite<Real,1>(dim, site, 
			   oprod, 
			   sig, coeff, ls,
			   output);
     }
     return;
   }



			  
			   




  // middleLinkKernel compiles for now, but lots of debugging to be done 
  template<class Real, int oddBit>
  void computeMiddleLinkSite(int half_lattice_index, // half_lattice_index to better match the GPU code.
			     const int dim[4],
			     const Real* const oprod,
			     const Real* const Qprev,
			     const Real* const link,
                             int sig, int mu,
			     Real coeff,
			     const LoadStore<Real>& ls, // pass a function object to read from and write to matrix fields
	                     Real* const Pmu,
			     Real* const P3,
			     Real* const Qmu, 
			     Real* const newOprod
		           )
  {
    const bool mu_positive  = (GOES_FORWARDS(mu)) ? true : false;
    const bool sig_positive = (GOES_FORWARDS(sig)) ? true : false;


    Locator<oddBit> locator(dim);
    int point_b, point_c, point_d;
    int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
    int X = locator.getFullFromHalfIndex(half_lattice_index);

    int err;
    int new_mem_idx = locator.getNeighborFromFullIndex(X,OPP_DIR(mu), &err); RETURN_IF_ERR;
    point_d = new_mem_idx >> 1;
    // getNeighborFromFullIndex will work on any site on the lattice, odd or even
    new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx,sig, &err); RETURN_IF_ERR;
    point_c = new_mem_idx >> 1;

    new_mem_idx = locator.getNeighborFromFullIndex(X,sig); RETURN_IF_ERR;
    point_b = new_mem_idx >> 1; 

    ad_link_nbr_idx = (mu_positive) ? point_d : half_lattice_index;
    bc_link_nbr_idx = (mu_positive) ? point_c : point_b;
    ab_link_nbr_idx = (sig_positive) ? half_lattice_index : point_b;

    typename ColorMatrix<Real>::Type ab_link, bc_link, ad_link;
    typename ColorMatrix<Real>::Type colorMatW, colorMatX, colorMatY, colorMatZ;


   


    if(sig_positive){
      ls.loadMatrixFromField(link, oddBit, sig, ab_link_nbr_idx, &ab_link);
    }else{
      ls.loadMatrixFromField(link, 1-oddBit, OPP_DIR(sig), ab_link_nbr_idx, &ab_link);
    }

    if(mu_positive){
      ls.loadMatrixFromField(link, oddBit, mu, bc_link_nbr_idx, &bc_link);
    }else{
      ls.loadMatrixFromField(link, 1-oddBit, OPP_DIR(mu), bc_link_nbr_idx, &bc_link);
    }

    if(Qprev == NULL){
      if(sig_positive){
	ls.loadMatrixFromField(oprod, 1-oddBit, sig, point_d, &colorMatY);
      }else{
	ls.loadMatrixFromField(oprod, oddBit, OPP_DIR(sig), point_c, &colorMatY);
	colorMatY = conj(colorMatY);
      }
    }else{ // Qprev != NULL
      ls.loadMatrixFromField(oprod, oddBit, point_c, &colorMatY);
    } 

    colorMatW = (!mu_positive) ? bc_link*colorMatY : conj(bc_link)*colorMatY;
    if(Pmu) ls.storeMatrixToField(colorMatW, 1-oddBit, point_b, Pmu);

    colorMatY = (sig_positive) ? ab_link*colorMatW : conj(ab_link)*colorMatW;
    ls.storeMatrixToField(colorMatY, oddBit, half_lattice_index, P3);
    
    if(mu_positive){
      ls.loadMatrixFromField(link, 1-oddBit, mu, ad_link_nbr_idx, &ad_link);
    }else{
      ls.loadMatrixFromField(link, oddBit, OPP_DIR(mu), ad_link_nbr_idx, &ad_link);
      ad_link = conj(ad_link);
    }


   if(Qprev == NULL){
     if(sig_positive) colorMatY = colorMatW*ad_link;
     if(Qmu) ls.storeMatrixToField(ad_link, oddBit, half_lattice_index, Qmu);
   }else{ // Qprev != NULL
     if(Qmu  || sig_positive){
       ls.loadMatrixFromField(Qprev, 1-oddBit, point_d, &colorMatY);
       colorMatX= colorMatY*ad_link;
     }
     if(Qmu) ls.storeMatrixToField(colorMatX, oddBit, half_lattice_index, Qmu);
     if(sig_positive) colorMatY = colorMatW*colorMatX;	
   }

   if(sig_positive) ls.addMatrixToField(colorMatY, oddBit, sig, half_lattice_index, coeff, newOprod); 

    return;
  } // computeMiddleLinkSite


  template<class Real>
  void computeMiddleLinkField(const int dim[4],
			     const Real* const oprod,
			     const Real* const Qprev,
			     const Real* const link,
                             int sig, int mu,
			     Real coeff,
	                     Real* const Pmu,
			     Real* const P3,
			     Real* const Qmu, 
			     Real* const newOprod
		           )
  {

    int volume = 1;
    for(int dir=0; dir<4; ++dir) volume *= dim[dir];
#ifdef MULTI_GPU
    const int loop_count = Vh_ex;
#else
    const int loop_count = volume/2;
#endif
   // loop over the lattice volume	
   // To keep the code as close to the GPU code as possible, we'll 
   // loop over the even sites first and then the odd sites
   LoadStore<Real> ls(volume);
   for(int site=0; site<loop_count; ++site){
     computeMiddleLinkSite<Real, 0>(site, dim,
				      oprod, Qprev, link,
				      sig, mu, coeff,
				      ls, 
				      Pmu, P3, Qmu, newOprod);
   }
   // Loop over odd lattice sites
   for(int site=0; site<loop_count; ++site){
     computeMiddleLinkSite<Real,1>(site, dim,
				   oprod, Qprev, link,
				   sig, mu, coeff,
				   ls, 
				   Pmu, P3, Qmu, newOprod);
   }
   return;
  }



 
  template<class Real, int oddBit>
  void computeSideLinkSite(int half_lattice_index, // half_lattice_index to better match the GPU code.
			   const int dim[4],
			   const Real* const P3,
			   const Real* const Qprod, // why?
			   const Real* const link,
                           int sig, int mu,
			   Real coeff, Real accumu_coeff,
			   const LoadStore<Real>& ls, // pass a function object to read from and write to matrix fields
			   Real* const shortP,
			   Real* const newOprod
		          )
  {

    const bool mu_positive  = (GOES_FORWARDS(mu)) ? true : false;
    const bool sig_positive = (GOES_FORWARDS(sig)) ? true : false;

    Locator<oddBit> locator(dim);
    int point_d;
    int ad_link_nbr_idx;
    int X = locator.getFullFromHalfIndex(half_lattice_index);

    int err;
    int new_mem_idx = locator.getNeighborFromFullIndex(X,OPP_DIR(mu), &err); RETURN_IF_ERR;
    point_d = new_mem_idx >> 1;
    ad_link_nbr_idx = (mu_positive) ? point_d : half_lattice_index;


    typename ColorMatrix<Real>::Type ad_link;
    typename ColorMatrix<Real>::Type colorMatW, colorMatX, colorMatY;

    ls.loadMatrixFromField(P3, oddBit, half_lattice_index, &colorMatY);	

    if(shortP){
      if(mu_positive){
	ad_link_nbr_idx = point_d;
        ls.loadMatrixFromField(link, 1-oddBit, mu, ad_link_nbr_idx, &ad_link);
      }else{
	ad_link_nbr_idx = half_lattice_index;
        ls.loadMatrixFromField(link, oddBit, OPP_DIR(mu), ad_link_nbr_idx, &ad_link);
      }
      colorMatW = (mu_positive) ? ad_link*colorMatY : conj(ad_link)*colorMatY;
      ls.addMatrixToField(colorMatW, 1-oddBit, point_d, accumu_coeff, shortP); 
    } // if(shortP)
 

    Real mycoeff = ( (sig_positive && oddBit) || (!sig_positive && !oddBit) ) ? coeff : -coeff;

    if(Qprod){
      ls.loadMatrixFromField(Qprod, 1-oddBit, point_d, &colorMatX);
      if(mu_positive){
	colorMatW = colorMatY*colorMatX;
	if(!oddBit){ mycoeff = -mycoeff; }
	ls.addMatrixToField(colorMatW, 1-oddBit, mu, point_d, mycoeff, newOprod);
      }else{
	colorMatW = conj(colorMatX)*conj(colorMatY);
	if(oddBit){ mycoeff = -mycoeff; }
	ls.addMatrixToField(colorMatW, oddBit, OPP_DIR(mu), half_lattice_index, mycoeff, newOprod);
      }
    }

    if(!Qprod){
      if(mu_positive){
	if(!oddBit){ mycoeff = -mycoeff; }
	ls.addMatrixToField(colorMatY, 1-oddBit, mu, point_d, mycoeff, newOprod);
      }else{
	if(oddBit){ mycoeff = -mycoeff; }
	colorMatW = conj(colorMatY);
	ls.addMatrixToField(colorMatW, oddBit, OPP_DIR(mu), half_lattice_index, mycoeff, newOprod);
      }
    } // if !(Qprod)
 
    return;
  } // computeSideLinkSite


  // Maybe change to computeSideLinkField
  template<class Real>
  void computeSideLinkField(const int dim[4],
			    const Real* const P3,
			    const Real* const Qprod, // why?
			    const Real* const link,
                            int sig, int mu,
			    Real coeff, Real accumu_coeff,
			    Real* const shortP,
			    Real* const newOprod
		          )
  {
    // Need some way of setting half_volume
    int volume = 1;
    for(int dir=0; dir<4; ++dir) volume *= dim[dir];
#ifdef MULTI_GPU
    const int loop_count = Vh_ex;
#else
    const int loop_count = volume/2;   
#endif
    LoadStore<Real> ls(volume);

    for(int site=0; site<loop_count; ++site){
      computeSideLinkSite<Real,0>(site, dim,
			  	  P3, Qprod, link, 
			  	  sig, mu, 
			  	  coeff, accumu_coeff, 
			  	  ls, shortP, newOprod);
    }

    for(int site=0; site<loop_count; ++site){
      computeSideLinkSite<Real,1>(site, dim,
			  	  P3, Qprod, link, 
			  	  sig, mu, 
			  	  coeff, accumu_coeff, 
			  	  ls, shortP, newOprod);
    }

    return;
  }





  template<class Real, int oddBit>
  void computeAllLinkSite(int half_lattice_index, // half_lattice_index to better match the GPU code.
			  const int dim[4],
			  const Real* const oprod,
			  const Real* const Qprev,
			  const Real* const link,
                          int sig, int mu,
			  Real coeff, Real accumu_coeff,
			  const LoadStore<Real>& ls, // pass a function object to read from and write to matrix fields
			  Real* const shortP,
			  Real* const newOprod)
   {

     const bool mu_positive  = (GOES_FORWARDS(mu)) ? true : false;
     const bool sig_positive = (GOES_FORWARDS(sig)) ? true : false;

     typename ColorMatrix<Real>::Type ab_link, bc_link, ad_link;
     typename ColorMatrix<Real>::Type colorMatW, colorMatX, colorMatY, colorMatZ; 


     int ab_link_nbr_idx, point_b, point_c, point_d;

     Locator<oddBit> locator(dim);
     int X = locator.getFullFromHalfIndex(half_lattice_index);

     int err;
     int new_mem_idx = locator.getNeighborFromFullIndex(X,OPP_DIR(mu), &err); RETURN_IF_ERR;
     point_d = new_mem_idx >> 1;

     new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx,sig, &err); RETURN_IF_ERR;
     point_c = new_mem_idx >> 1;

     new_mem_idx = locator.getNeighborFromFullIndex(X,sig, &err);  RETURN_IF_ERR;
     point_b = new_mem_idx >> 1; 
     ab_link_nbr_idx = (sig_positive) ? half_lattice_index : point_b;



     Real mycoeff = ( (sig_positive && oddBit) || (!sig_positive && !oddBit) ) ? coeff : -coeff;

     if(mu_positive){ 
       ls.loadMatrixFromField(Qprev, 1-oddBit, point_d, &colorMatX);
       ls.loadMatrixFromField(link, 1-oddBit, mu, point_d, &ad_link);
	// compute point_c
	ls.loadMatrixFromField(oprod, oddBit, point_c, &colorMatY);
	ls.loadMatrixFromField(link, oddBit, mu, point_c, &bc_link);
	colorMatZ = conj(bc_link)*colorMatY; // okay
	
	if(sig_positive)
 	{	
	  colorMatY = colorMatX*ad_link;
	  colorMatW = colorMatZ*colorMatY;
	  ls.addMatrixToField(colorMatW, oddBit, sig, half_lattice_index, Sign<oddBit>::result*mycoeff, newOprod);
	}

	if(sig_positive){
	  ls.loadMatrixFromField(link, oddBit, sig, ab_link_nbr_idx, &ab_link);
	}else{
	  ls.loadMatrixFromField(link, 1-oddBit, OPP_DIR(sig), ab_link_nbr_idx, &ab_link);
	}
	colorMatY = (sig_positive) ? ab_link*colorMatZ : conj(ab_link)*colorMatZ; // okay
	colorMatW = colorMatY*colorMatX;
	ls.addMatrixToField(colorMatW, 1-oddBit, mu, point_d, -Sign<oddBit>::result*mycoeff, newOprod);
	colorMatW = ad_link*colorMatY;
	ls.addMatrixToField(colorMatW, 1-oddBit, point_d, accumu_coeff, shortP); // Warning! Need to check this!
     }else{ //negative mu
	mu = OPP_DIR(mu);
	ls.loadMatrixFromField(Qprev, 1-oddBit, point_d, &colorMatX);
	ls.loadMatrixFromField(link, oddBit, mu, half_lattice_index, &ad_link);
        ls.loadMatrixFromField(oprod, oddBit, point_c, &colorMatY);
	ls.loadMatrixFromField(link, 1-oddBit, mu, point_b, &bc_link);
	
	if(sig_positive) colorMatW = colorMatX*conj(ad_link);
 	colorMatZ = bc_link*colorMatY;
	if(sig_positive){ 
	  colorMatY = colorMatZ*colorMatW;
	  ls.addMatrixToField(colorMatY, oddBit, sig, half_lattice_index, Sign<oddBit>::result*mycoeff, newOprod);
	}

	if(sig_positive){
	  ls.loadMatrixFromField(link, oddBit, sig, ab_link_nbr_idx, &ab_link);
	}else{
	  ls.loadMatrixFromField(link, 1-oddBit, OPP_DIR(sig), ab_link_nbr_idx, &ab_link);
	}

	colorMatY = (sig_positive) ? ab_link*colorMatZ : conj(ab_link)*colorMatZ; // 611
	colorMatW = conj(colorMatX)*conj(colorMatY);
	
	ls.addMatrixToField(colorMatW, oddBit, mu, half_lattice_index, Sign<oddBit>::result*mycoeff, newOprod);
	colorMatW = conj(ad_link)*colorMatY; 
	ls.addMatrixToField(colorMatW, 1-oddBit, point_d, accumu_coeff, shortP);
     } // end mu	
     return;
   } // allLinkKernel



  template<class Real>
  void computeAllLinkField(const int dim[4],
			  const Real* const oprod,
			  const Real* const Qprev,
			  const Real* const link,
                          int sig, int mu,
			  Real coeff, Real accumu_coeff,
			  Real* const shortP,
			  Real* const newOprod)
  {
    int volume = 1;
    for(int dir=0; dir<4; ++dir) volume *= dim[dir];
#ifdef MULTI_GPU
    const int loop_count = Vh_ex;
#else
    const int loop_count = volume/2;
#endif

    LoadStore<Real> ls(volume);
    for(int site=0; site<loop_count; ++site){

      computeAllLinkSite<Real,0>(site, dim,
				  oprod, Qprev, link,
				  sig, mu, 
				  coeff, accumu_coeff,
				  ls,
				  shortP, newOprod);
    }
    
    for(int site=0; site<loop_count; ++site){
       computeAllLinkSite<Real, 1>(site, dim,
				   oprod, Qprev, link,
				   sig, mu, 
				   coeff, accumu_coeff,
				   ls,
				   shortP, newOprod);
    }

    return;
  }

#define Pmu   tempmat[0]
#define P3    tempmat[1]
#define P5    tempmat[2]
#define Pnumu tempmat[3]
#define Qmu   tempmat[4]
#define Qnumu tempmat[5]

  template<class Real>
  struct PathCoefficients
  {
    Real one;
    Real three; 
    Real five;
    Real seven; 
    Real naik;
    Real lepage;
  };


  template<class Real>
  void doHisqStaplesForceCPU(const int dim[4],
			     PathCoefficients<double> staple_coeff,
			     Real* oprod,
			     Real* link,
			     Real** tempmat,
			     Real* newOprod)
  {
    Real OneLink, ThreeSt, FiveSt, SevenSt, Lepage, coeff;
    
    OneLink = staple_coeff.one; 
    ThreeSt = staple_coeff.three;
    FiveSt  = staple_coeff.five;
    SevenSt = staple_coeff.seven;
    Lepage  = staple_coeff.lepage;

    for(int sig=0; sig<4; ++sig){
      computeOneLinkField(dim,
			  oprod,
			  sig, OneLink,
			  newOprod);
    }


    // sig labels the net displacement of the staple
    for(int sig=0; sig<8; ++sig){
      for(int mu=0; mu<8; ++mu){
	if( mu == sig || mu == OPP_DIR(sig) ) continue; 

	computeMiddleLinkField<Real>(dim,
			       oprod, NULL, link,
			       sig, mu, -ThreeSt,
			       Pmu, P3, Qmu,
			       newOprod); 

	for(int nu=0; nu<8; ++nu){
	  if(   nu==mu  || nu==OPP_DIR(mu) 
	     || nu==sig || nu==OPP_DIR(sig) ) continue;  
	
	  computeMiddleLinkField<Real>(dim,
			         Pmu, Qmu, link,
				 sig, nu, staple_coeff.five,
				 Pnumu, P5, Qnumu,
				 newOprod);	


          for(int rho=0; rho<8; ++rho){
	    if(   rho == sig || rho == OPP_DIR(sig)
	       || rho == mu  || rho == OPP_DIR(mu)
	       || rho == nu  || rho == OPP_DIR(nu) )
	    { 
	      continue;
            } 

            if(FiveSt != 0)coeff = SevenSt/FiveSt; else coeff = 0;
	    computeAllLinkField<Real>(dim, 
				Pnumu, Qnumu, link,
				sig, rho, staple_coeff.seven, coeff, 
				P5, newOprod);		

	  } // rho 

	  // 5-staple: side link
	  if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;
	  computeSideLinkField<Real>(dim, 
			       P5, Qmu, link, 
			       sig, nu, -FiveSt, coeff, 
			       P3, newOprod);


	} // nu 


	// lepage
	if(staple_coeff.lepage != 0.){
	  computeMiddleLinkField<Real>(dim,
				 Pmu, Qmu, link, 
				 sig, mu, Lepage,
				 NULL, P5, NULL,
				 newOprod);

	  if(ThreeSt != 0)coeff = Lepage/ThreeSt; else coeff = 0;
	  computeSideLinkField<Real>(dim, 
			       P5, Qmu, link,
			       sig, mu, -Lepage, coeff, 
			       P3, newOprod);
        } // lepage != 0

	  
	computeSideLinkField<Real>(dim, 
			     P3, NULL, link, 
			     sig, mu, ThreeSt, 0., 
			     NULL, newOprod);
      } // mu	
    }  // sig
    
    // Need also to compute the one-link contribution 
  return;
  }

#undef Pmu
#undef P3
#undef P5
#undef Pnumu
#undef Qmu
#undef Qnumu


  void hisqStaplesForceCPU(const double* path_coeff,
			   const QudaGaugeParam &param,
			   cpuGaugeField  &oprod,
			   cpuGaugeField  &link,
			   cpuGaugeField* newOprod)
  {
    int volume = 1;
    for(int dir=0; dir<4; ++dir) volume *= param.X[dir];

#ifdef MULTI_GPU
    int len = Vh_ex*2;
#else
    int len = volume;
#endif    
    // allocate memory for temporary fields
    void* tempmat[6]; 
    if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
      for(int i=0; i<6; ++i) tempmat[i] = malloc(len*18*sizeof(double));
    }else{
      for(int i=0; i<6; ++i) tempmat[i] = malloc(len*18*sizeof(float));
    }

    PathCoefficients<double> act_path_coeff;
    act_path_coeff.one    = path_coeff[0];
    act_path_coeff.naik   = path_coeff[1];
    act_path_coeff.three  = path_coeff[2];
    act_path_coeff.five   = path_coeff[3];
    act_path_coeff.seven  = path_coeff[4];
    act_path_coeff.lepage = path_coeff[5];

    if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
      doHisqStaplesForceCPU<double>(param.X,
				    act_path_coeff, 
				    (double*)oprod.Gauge_p(),
				    (double*)link.Gauge_p(),
				    (double**)tempmat,
				    (double*)newOprod->Gauge_p() 
				    );

    }else if(param.cpu_prec == QUDA_SINGLE_PRECISION){
      doHisqStaplesForceCPU<float>(param.X,
				   act_path_coeff, 
				   (float*)oprod.Gauge_p(),
				   (float*)link.Gauge_p(),
				   (float**)tempmat, 
				   (float*)newOprod->Gauge_p()
				   );
    }else{
      errorQuda("Unsupported precision");
    }

    for(int i=0; i<6; ++i){
      free(tempmat[i]);
    }
    return;
  }



  template<class Real, int oddBit> 
   void computeLongLinkSite(int half_lattice_index, 	
			   const int dim[4],
			   const Real* const oprod,
			   const Real* const link,
		           int sig, Real coeff,	
			   const LoadStore<Real>& ls,
			   Real* const output)
   {
     if( GOES_FORWARDS(sig) ){

       Locator<oddBit> locator(dim);

       typename ColorMatrix<Real>::Type ab_link, bc_link, de_link, ef_link;
       typename ColorMatrix<Real>::Type colorMatU, colorMatV, colorMatW, colorMatX, colorMatY, colorMatZ;

       int point_a, point_b, point_c, point_d, point_e;	
#ifdef MULTI_GPU
       int idx = ls.half_idx_conversion_normal2ex(half_lattice_index, dim, oddBit);
#else
       int idx = half_lattice_index;
#endif

       int X = locator.getFullFromHalfIndex(idx);
       point_c = idx;

       int new_mem_idx = locator.getNeighborFromFullIndex(X,sig);
       point_d = new_mem_idx >> 1;

       new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx, sig);
       point_e = new_mem_idx >> 1;

       new_mem_idx = locator.getNeighborFromFullIndex(X, OPP_DIR(sig));
       point_b = new_mem_idx >> 1;

       new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx, OPP_DIR(sig));
       point_a = new_mem_idx >> 1;

       ls.loadMatrixFromField(link, oddBit, sig, point_a, &ab_link);
       ls.loadMatrixFromField(link, 1-oddBit, sig, point_b, &bc_link);
       ls.loadMatrixFromField(link, 1-oddBit, sig, point_d, &de_link);
       ls.loadMatrixFromField(link, oddBit, sig, point_e, &ef_link);

       ls.loadMatrixFromField(oprod, oddBit, sig, point_c, &colorMatZ);
       ls.loadMatrixFromField(oprod, 1-oddBit, sig, point_b, &colorMatY);
       ls.loadMatrixFromField(oprod, oddBit, sig, point_a, &colorMatX);

       colorMatV = de_link*ef_link*colorMatZ  
		  - de_link*colorMatY*bc_link
		  + colorMatX*ab_link*bc_link;

       ls.addMatrixToField(colorMatV, oddBit, sig, point_c, coeff, output);
     } 
     return;
   }

  template<class Real>
    void computeLongLinkField(const int dim[4], 
			     const Real* const oprod,
			     const Real* const link,
			     int sig, Real coeff,
			     Real* const output)
   {
     int volume = 1;
     for(int dir=0; dir<4; ++dir) volume *= dim[dir];
     const int half_volume = volume/2;
     
     LoadStore<Real> ls(volume);
     for(int site=0; site<half_volume; ++site){
       computeLongLinkSite<Real,0>(site, 
			   dim,
			   oprod,
		           link, 
			   sig, coeff, ls,
			   output);
			 
     }
     // Loop over odd lattice sites
     for(int site=0; site<half_volume; ++site){
	computeLongLinkSite<Real,1>(site, 
			   dim,
			   oprod,
			   link, 
			   sig, coeff, ls,
			   output);
     }
     return;
   }


  void hisqLongLinkForceCPU(double coeff, 
			    const QudaGaugeParam &param,
			    cpuGaugeField &oprod, 
			    cpuGaugeField &link,
			    cpuGaugeField *newOprod)
  {
    for(int sig=0; sig<4; ++sig){
      if(param.cpu_prec == QUDA_SINGLE_PRECISION){
	computeLongLinkField<float>(param.X,	
	                     (float*)oprod.Gauge_p(),
			     (float*)link.Gauge_p(),
			      sig, coeff,
			     (float*)newOprod->Gauge_p());  
      }else if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
	computeLongLinkField<double>(param.X,	
	                     (double*)oprod.Gauge_p(),
			     (double*)link.Gauge_p(),
			      sig, coeff,
			     (double*)newOprod->Gauge_p());  
      }else {
	errorQuda("Unrecognised precision\n");
      }
    } // sig
    return;
  }


template<class Real, int oddBit>
void completeForceSite(int half_lattice_index,
		       const int dim[4],
		       const Real* const oprod,
		       const Real* const link,
		       int sig,
		       const LoadStore<Real>& ls,
		       Real* const mom)
{

  typename ColorMatrix<Real>::Type colorMatX, colorMatY, linkW;

#ifdef MULTI_GPU
  int half_lattice_index_ex = ls.half_idx_conversion_normal2ex(half_lattice_index, dim, oddBit);
  int idx = half_lattice_index_ex;  
#else
  int idx = half_lattice_index;
#endif  
  ls.loadMatrixFromField(link, oddBit, sig, idx, &linkW);
  ls.loadMatrixFromField(oprod, oddBit, sig, idx, &colorMatX);

  const Real coeff = (oddBit) ? -1 : 1;
  colorMatY = linkW*colorMatX;

  ls.storeMatrixToMomentumField(colorMatY, oddBit, sig, half_lattice_index, coeff, mom);
  return;
}

template <class Real>
void completeForceField(const int dim[4],
			const Real* const oprod,
			const Real* const link,
			int sig,
			Real* const mom)
{
  int volume = dim[0]*dim[1]*dim[2]*dim[3];
  const int half_volume = volume/2;
  LoadStore<Real> ls(volume);


  for(int site=0; site<half_volume; ++site){
    completeForceSite<Real,0>(site,
			      dim,
			      oprod, link,
			      sig,
			      ls,
			      mom);

  }
  for(int site=0; site<half_volume; ++site){
    completeForceSite<Real,1>(site,
			      dim,
			      oprod, link,
			      sig,
			      ls,
			      mom);
  }
  return;
}


  void hisqCompleteForceCPU(const QudaGaugeParam &param, 
			    cpuGaugeField &oprod,
			    cpuGaugeField &link,
			    cpuGaugeField* mom)
  {
    for(int sig=0; sig<4; ++sig){
       if(param.cpu_prec == QUDA_SINGLE_PRECISION){
	 completeForceField<float>(param.X, 
				   (float*)oprod.Gauge_p(), 
			           (float*)link.Gauge_p(), 
				   sig, 
				   (float*)mom->Gauge_p());
       }else if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
	  completeForceField<double>(param.X, 
				   (double*)oprod.Gauge_p(), 
			           (double*)link.Gauge_p(), 
				   sig, 
				   (double*)mom->Gauge_p());
       }else{
         errorQuda("Unrecognised precision\n");
       }
    } // loop over sig
    return;
  }
  

//} // namespace quda





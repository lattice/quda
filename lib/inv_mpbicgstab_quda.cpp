#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <face_quda.h>

#include <iostream>

namespace quda {

  MPBiCGstab::MPBiCGstab(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat)
  {
  }

  MPBiCGstab::~MPBiCGstab() {
  }


  void MPBiCGstab::computeMatrixPowers(std::vector<cudaColorSpinorField>& pr, cudaColorSpinorField& p, cudaColorSpinorField& r, int nsteps){
    cudaColorSpinorField temp(p);
    pr[0] = p;
    for(int i=1; i<=(2*nsteps); ++i){
      mat(pr[i], pr[i-1], temp);
    }

    pr[(2*nsteps)+1] = r;
  //  for(int i=(2*nsteps+2); i<(4*nsteps+2); ++i){
    for(int i=(2*nsteps+2); i<(4*nsteps+1); ++i){
      mat(pr[i], pr[i-1], temp);
    }
  }

#ifdef SSTEP
  static void print(const double d[], int n){
    for(int i=0; i<n; ++i){
      std::cout << d[i] << " ";
    }
    std::cout << std::endl;
  }

  static void print(const Complex d[], int n){
    for(int i=0; i<n; ++i){
      std::cout <<  "(" << real(d[i]) << "," << imag(d[i]) << ") ";
    }
    std::cout << std::endl;
  }


  template<typename T>
    static void zero(T d[], int N){
      for(int i=0; i<N; ++i) d[i] = static_cast<T>(0);
    }


  static void computeGramMatrix(Complex** G, std::vector<cudaColorSpinorField>& v){

    const int dim = v.size();

    for(int i=0; i<dim; ++i){
      for(int j=0; j<dim; ++j){
        G[i][j] = cDotProductCuda(v[i],v[j]);
      }
    }
    return;
  }

  static void computeGramVector(Complex* g, cudaColorSpinorField& r0, std::vector<cudaColorSpinorField>& pr){

    const int dim = pr.size();

    for(int i=0; i<dim; ++i){
      g[i] = cDotProductCuda(r0,pr[i]);
    }
  }

/*
  // Here, B is an (s+1)x(s+1) matrix with 1s on the subdiagonal
  template<class T>
    static void getBColumn(T *col, int col_index, int nsteps){
      zero(col,nsteps+1);
      col[col_index] = static_cast<T>(1);
    }

  template<typename T>
    static void init_c_vector(T *c, int index, int nsteps){
      zero(c,2*nsteps+2);
      getBColumn(c+(nsteps+1),index,nsteps);
    }

  template<typename T>
    static void init_a_vector(T *a, int index, int nsteps){
      zero(a,2*nsteps+2);
      getBColumn(a,index,nsteps);
    }

  template<typename T>
    static void init_e_vector(T *e, int nsteps){
      zero(e,2*nsteps+2);
      e[2*nsteps+2] = static_cast<T>(1);
    }
*/

  template<typename T>
    static T zip(T a[], T b[], int dim){
      T result = 0.0;
      for(int i=0; i<dim; ++i){
        result += a[i]*b[i];
      }
      return result;
    }

  static Complex computeUdaggerMV(Complex* u, Complex** M, Complex* v, int dim)
  {
    Complex result(0,0);

    for(int i=0; i<dim; ++i){
      for(int j=0; j<dim; ++j){
        result += conj(u[i])*v[j]*M[i][j];
      }
    }
    return result;
  } 
#endif

  void MPBiCGstab::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {
#ifndef SSTEP
    errorQuda("S-step solvers not built\n");
#else
    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    cudaColorSpinorField temp(b, csParam);

    cudaColorSpinorField r(b);



    mat(r, x, temp);  // r = Ax
    double r2 = xmyNormCuda(b,r); // r = b - Ax



    cudaColorSpinorField r0(r);
    cudaColorSpinorField p(r);
    cudaColorSpinorField Ap(r);


    const int s = 3;

    // Vector of matrix powers
    std::vector<cudaColorSpinorField> PR(4*s+2,cudaColorSpinorField(b,csParam));


    Complex r0r;
    Complex alpha;
    Complex omega;
    Complex beta;

    Complex** G = new Complex*[4*s+2];
    for(int i=0; i<(4*s+2); ++i){
      G[i] = new Complex[4*s+2];   
    }
    Complex* g = new Complex[4*s+2];

    Complex** a = new Complex*[2*s+1];
    Complex** c = new Complex*[2*s+1];
    Complex** a_new = new Complex*[2*s+1];
    Complex** c_new = new Complex*[2*s+1];

    for(int i=0; i<(2*s+1); ++i){
      a[i] = new Complex[4*s+2];
      c[i] = new Complex[4*s+2];
      a_new[i] = new Complex[4*s+2];
      c_new[i] = new Complex[4*s+2];
    }


    Complex* e = new Complex[4*s+2];




    double stop = stopping(param.tol, b2, param.residual_type);
    int it=0;
    int m=0;
    while(!convergence(r2, 0.0, stop, 0.0 ) && it<param.maxiter){

      computeMatrixPowers(PR, p, r, s); 
      computeGramVector(g, r0, PR); 
      computeGramMatrix(G, PR);

      // initialize coefficient vectors
      for(int i=0; i<(2*s+1); ++i){
        zero(a[i],(4*s+2));
        zero(c[i],(4*s+2));
        a[i][i] = static_cast<Complex>(1);
        c[i][i + (2*s+1)] = static_cast<Complex>(1);
      }


      zero(e,(4*s+2));
      int j=0;
      while(!convergence(r2,0.0,stop,0.0) && j<s){
        PrintStats("MPBiCGstab", it, r2, b2, 0.0);

        alpha = zip(g,c[0],4*s+2)/zip(g,a[1],4*s+2);

        Complex omega_num = computeUdaggerMV(c[0],G,c[1],(4*s+2)) 
          - alpha*computeUdaggerMV(c[0],G,a[2],(4*s+2))
          - conj(alpha)*computeUdaggerMV(a[1],G,c[1],(4*s+2))
          + conj(alpha)*alpha*computeUdaggerMV(a[1],G,a[2],(4*s+2));

        Complex omega_denom = computeUdaggerMV(c[1],G,c[1],(4*s+2))
          - alpha*computeUdaggerMV(c[1],G,a[2],(4*s+2))
          - conj(alpha)*computeUdaggerMV(a[2],G,c[1],(4*s+2))
          + conj(alpha)*alpha*computeUdaggerMV(a[2],G,a[2],(4*s+2));

        omega = omega_num/omega_denom;
        // Update candidate solution
        for(int i=0; i<(4*s+2); ++i){
          e[i] += alpha*a[0][i] + omega*c[0][i] - alpha*omega*a[1][i];
        }

        // Update residual
        for(int k=0; k<=(2*(s - j - 1)); ++k){
          for(int i=0; i<(4*s+2); ++i){
            c_new[k][i] = c[k][i] - alpha*a[k+1][i] - omega*c[k+1][i] + alpha*omega*a[k+2][i];
          }
        }

        // update search direction
        beta = (zip(g,c_new[0],(4*s+2))/zip(g,c[0],(4*s+2)))*(alpha/omega);

        for(int k=0; k<=(2*(s - j - 1)); ++k){
          for(int i=0; i<(4*s+2); ++i){
            a_new[k][i] = c_new[k][i] + beta*a[k][i] - beta*omega*a[k+1][i];
          }

          for(int i=0; i<(4*s+2); ++i){
            a[k][i] = a_new[k][i];
            c[k][i] = c_new[k][i];
          }
        }
        zeroCuda(r);
        for(int i=0; i<(4*s+2); ++i){
          caxpyCuda(c[0][i], PR[i], r);
        }
        r2 = norm2(r);
        j++;
        it++;
      } // j

      zeroCuda(p);
      for(int i=0; i<(4*s+2); ++i){
        caxpyCuda(a[0][i], PR[i], p);
        caxpyCuda(e[i], PR[i], x);
      }

      m++;
    } 

    if(it >= param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual
    mat(r, x, temp);
    param.true_res = sqrt(xmyNormCuda(b, r)/b2);

    PrintSummary("MPBiCGstab", it, r2, b2);



    for(int i=0; i<(4*s+2); ++i){
      delete[] G[i];
    }
    delete[] G;


    delete[] g;

    // Is 2*s + 3 really correct?
    for(int i=0; i<(2*s+1); ++i){
      delete[] a[i];
      delete[] a_new[i];
      delete[] c[i];
      delete[] c_new[i];
    }
    delete[] a;
    delete[] a_new;
    delete[] c;
    delete[] c_new;
    delete[] e;
#endif
    return;
  }

} // namespace quda

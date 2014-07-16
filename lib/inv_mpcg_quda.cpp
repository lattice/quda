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


  template<typename T>
  static void applyT(T d_out[], const T d_in[], const T gamma[], const T rho[], int N)
  { 
    if(N <= 0) return;
    for(int i=0; i<N; ++i){
      d_out[i] = d_in[i]/gamma[i];
    }

    for(int i=0; i<N-1; ++i){
      d_out[i] += (d_in[i+1]*(1-rho[i+1])/(gamma[i+1]*rho[i+1]));
    }

    for(int i=1; i<N; ++i){
      d_out[i] -= d_in[i-1]/(rho[i-1]*gamma[i-1]);
    }

    return;
  }

  template<typename T>
  static void applyB(T d_out[], const T d_in[], int N)
  {
    d_out[0] = static_cast<T>(0);
    for(int i=1; i<N; ++i) d_out[i] = d_in[i-1]; 
    return;
  }


  template<typename T>
  static void zero(T d[], int N){
    for(int i=0; i<N; ++i) d[i] = static_cast<T>(0);
  }

  template<typename T>
  static void applyThirdTerm(T d_out[], const T d_in[], int k, int s, const T gamma[], const T rho[], const T gamma_prev[], const T rho_prev[])
  {
    // s is the number of steps
    // The input and output vectors are of dimension 2*s + 1
    const int dim = 2*s + 1;    

    zero(d_out, dim);

    applyT(d_out, d_in, gamma_prev, rho_prev, s); // compute the upper half of the vector
    applyB(d_out, d_in + s, s+1); // update the lower half

    for(int i=s; i<(2*s+1); ++i) d_out[i] = -d_out[i];
    
    // This has to come after applyB
    d_out[s] -= d_in[s-1]/(rho_prev[s-1]*gamma_prev[s-1]);

    // Finally scale everything
    for(int i=0; i<dim; ++i) d_out[i] *= -rho[k]*gamma[k];
    
    return;
  }

  template<typename T>
  static void computeCoeffs(T d_out[], const T d_p1[], const T d_p2[], int j, int s, const T gamma[], const T rho[], const T gamma_prev[], const T rho_prev[])
  {
    applyThirdTerm(d_out, d_p1, j-1, s, gamma, rho, gamma_prev, rho_prev);

    for(int i=0; i<(2*s+1); ++i){
      d_out[i] += rho[j-1]*d_p1[i] + (1 - rho[j-1])*d_p2[i];
    }
    return;
  }
                  



  MPCG::MPCG(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat)
  {

  }

  MPCG::~MPCG() {

  }

  void MPCG::computeMatrixPowers(cudaColorSpinorField out[], cudaColorSpinorField &in, int nvec)
  {
    cudaColorSpinorField temp(in);
    out[0] = in;
    for(int i=1; i<nvec; ++i){
      mat(out[i], out[i-1], temp);
    }
    return;
  }

  void MPCG::computeMatrixPowers(std::vector<cudaColorSpinorField>& out, std::vector<cudaColorSpinorField>& in, int nsteps)
  {
    cudaColorSpinorField temp(in[0]);

    for(int i=0; i<=nsteps; ++i) out[i] = in[i];

    for(int i=(nsteps+1); i<=(2*nsteps); ++i){
      mat(out[i], out[i-1], temp);
    }
  }



  void print(double d[], int n){
    for(int i=0; i<n; ++i){
      std::cout << d[i] << " ";
    }
    std::cout << std::endl;
  }

  void MPCG::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }


    cudaColorSpinorField temp(b); // temporary field
  
    // Use ColorSpinorParam to create zerod fields
    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField x_prev(x,csParam);

    const int s = 2;

    // create the residual array and the matrix powers array
    std::vector<cudaColorSpinorField> R(s+1,cudaColorSpinorField(b,csParam));
    std::vector<cudaColorSpinorField> R_new(s+1,cudaColorSpinorField(b,csParam));
    std::vector<cudaColorSpinorField> V(2*s+1,cudaColorSpinorField(b,csParam));

    // Set up the first residual
    mat(R[s], x, temp);
    double r2 = xmyNormCuda(b,R[s]);


    double stop = stopping(param.tol, b2, param.residual_type);


    int it = 0;
    double* d    = new double[2*s+1];
    double* d_p1 = new double[2*s+1];
    double* d_p2 = new double[2*s+1];

    // Matrix powers kernel
    // The first s vectors hold the previous residuals 
    // v[s] holds current residua
    // v[s+1] holds A r
    // v[s+2] holds A^(2)r
    // v[2*s] holds A^(s)r
    //cudaColorSpinorField* v = new cudaColorSpinorField[2*s+1];
    cudaColorSpinorField* v;
    cudaColorSpinorField w(b);

    double rAr = 0.0;
    
    double gamma_old = 0.0;
    double mu_old = 0.0;
    double rho_old = 0.0;
    double rho[s];
    double mu[s];
    double gamma[s];   
    double rho_prev[s];
    double gamma_prev[s];
     

    cudaColorSpinorField x_new(x,csParam);
    cudaColorSpinorField r_old(x,csParam);
 
    int k = 0;
    while(!convergence(r2,0.0,stop,0.0) && it < param.maxiter){
      // compute the matrix powers kernel - need to set r[s] above
      computeMatrixPowers(V, R, s); 


      zero(d_p2, 2*s+1);


      r_old = R[s-1];
      R[0] = R[s];

      for(int j=0; j<s; ++j){ 
        
        if(j==0){
          zero(d, 2*s+1); d[s+1] = 1.0;
          w = V[s+1];
          r2 = norm2(R[j]); // v[s] = r[s*k]
          rAr = reDotProductCuda(w,R[j]);
          mu[j] = r2;
          gamma[j] = r2/rAr;
          
          if(it==0){
            rho[j] = 1.0;
          }else{
            rho[j] = 1.0/(1.0 - (gamma[j]/gamma_old)*(mu[j]/mu_old)*(1.0/rho_old));  
          }

          R[j+1] = r_old;
          axCuda((1.0 - rho[j]), R[j+1]);
          axpyCuda(rho[j], R[j], R[j+1]);
          axpyCuda(-rho[j]*gamma[j], w, R[j+1]);

          x_new = x_prev;
          axCuda((1.0 - rho[j]), x_new);
          axpyCuda(rho[j], x, x_new);
          axpyCuda(rho[j]*gamma[j], R[j], x_new);
          

          // copy d to d_p1
          for(int i=0; i<(2*s+1); ++i) d_p1[i] = d[i];

        }

        zeroCuda(w); 
        if(j==1){
          if(k > 0) d_p2[s-1] = 1.0; 
          computeCoeffs(d, d_p1, d_p2, j, s, gamma, rho, gamma_prev, rho_prev);
    
          print(d, 2*s+1);  
 
          for(int i=0; i<(2*s+1); ++i){
            if(d[i] != 0.) axpyCuda(d[i], V[i], w);
          }
  
          r2 = norm2(R[j]); // v[s] = r[s*k];
          rAr = reDotProductCuda(w,R[j]);
          mu[j] = r2;
          gamma[j] = r2/rAr;
    
          rho[j] = 1.0/(1.0 - (gamma[j]/gamma[j-1])*(mu[j]/mu[j-1])*(1.0/rho[j-1]));

          R[j+1] = R[j-1];
          axCuda((1.0 - rho[j]),R[j+1]);
          axpyCuda(rho[j], R[j], R[j+1]);
          axpyCuda(-gamma[j]*rho[j], w, R[j+1]);

          x_new = x_prev;
          axCuda((1.0 - rho[j]), x_new);
          axpyCuda(rho[j], x, x_new);
          axpyCuda(gamma[j]*rho[j], R[j], x_new);

          for(int i=0; i<(2*s+1); ++i){
            d_p2[i] = d_p1[i];
            d_p1[i] = d[i];
          }
        } // j == 1

        if(j>=2){
          computeCoeffs(d, d_p1, d_p2, j, s, gamma, rho, gamma_prev, rho_prev);

          for(int i=0; i<(2*s+1); ++i){
            if(d[i] != 0.) axpyCuda(d[i], V[i], w);
          }

          r2 = norm2(R[j]);
          rAr = reDotProductCuda(w,R[j]);
          mu[j] = r2;
          gamma[j] = r2/rAr;

          rho[j] = 1.0/(1.0 - (gamma[j]/gamma[j-1])*(mu[j]/mu[j-1])*(1.0/rho[j-1]));

          R[j+1] = R[j-1];
          axCuda((1.0 - rho[j]),R[j+1]);
          axpyCuda(rho[j], R[j], R[j+1]);
          axpyCuda(-gamma[j]*rho[j], w, R[j+1]);

          x_new = x_prev;
          axCuda((1.0 - rho[j]), x_new);
          axpyCuda(rho[j], x, x_new);
          axpyCuda(gamma[j]*rho[j], R[j], x_new);

          for(int i=0; i<(2*s+1); ++i){
            d_p2[i] = d_p1[i];
            d_p1[i] = d[i];
          }
        } // j >= 2     
  

        for(int i=0; i<s; ++i){
          rho_prev[i] = rho[i];
          gamma_prev[i] = gamma[i];
        } 
        printfQuda("s = %d\n",s);
        PrintStats("MPCG", it, r2, b2, 0.0);
        if(it == 3) return;

        x = x_new;
        x_prev = x;
 
        it++; 
      } // loop over j

      k++;
      gamma_old = gamma[s-1];
      mu_old = mu[s-1];
      rho_old = rho[s-1];
    }

  

    PrintSummary("MPCG", it, r2, b2);
 
    delete[] d;
    delete[] d_p1;
    delete[] d_p2;

    return;
  }

} // namespace quda

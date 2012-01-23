#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <gauge_field.h>

#include "quda_matrix.h"
#include "svd_quda.h"
//#include "unitarize_utilities.h"




#define FORCE_UNITARIZE_PI  3.141592653589793
#define FORCE_UNITARIZE_PI23 FORCE_UNITARIZE_PI*2.0/3.0
#define FORCE_UNITARIZE_EPS 1e-5
#define HISQ_FORCE_FILTER 5e-5

// constants - File scope only
//__device__ __constant__ double FORCE_UNITARIZE_EPS;
//__device__ __constant__ double HISQ_FORCE_FILTER;



namespace hisq{
  namespace fermion_force{

    void set_unitarize_force_constants(double unitarize_eps, double hisq_force_filter)
    {
      cudaMemcpyToSymbol("FORCE_UNITARIZE_EPS", &unitarize_eps, sizeof(double));
      cudaMemcpyToSymbol("HISQ_FORCE_FILTER", &hisq_force_filter, sizeof(double));
      return;
    }



    template<class Real>
     class DerivativeCoefficients{
       private:
         Real b[6]; 
         __device__ __host__       
         Real computeC00(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC01(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC02(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC11(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC12(const Real &, const Real &, const Real &);
         __device__ __host__
         Real computeC22(const Real &, const Real &, const Real &);

       public:
         __device__ __host__ void set(const Real & u, const Real & v, const Real & w);
         __device__ __host__
         Real getB00() const { return b[0]; }
         __device__ __host__
         Real getB01() const { return b[1]; }
         __device__ __host__
         Real getB02() const { return b[2]; }
         __device__ __host__
         Real getB11() const { return b[3]; }
         __device__ __host__
         Real getB12() const { return b[4]; }
         __device__ __host__
         Real getB22() const { return b[5]; }
     };

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC00(const Real & u, const Real & v, const Real & w){
        Real result =   -pow(w,3)*pow(u,6)
          + 3*v*pow(w,3)*pow(u,4)
          + 3*pow(v,4)*w*pow(u,4)
          -   pow(v,6)*pow(u,3)
          - 4*pow(w,4)*pow(u,3)
          - 12*pow(v,3)*pow(w,2)*pow(u,3)
          + 16*pow(v,2)*pow(w,3)*pow(u,2)
          + 3*pow(v,5)*w*pow(u,2)
          - 8*v*pow(w,4)*u
          - 3*pow(v,4)*pow(w,2)*u
          + pow(w,5)
          + pow(v,3)*pow(w,3);

        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC01(const Real & u, const Real & v, const Real & w){
        Real result =  - pow(w,2)*pow(u,7)
          - pow(v,2)*w*pow(u,6)
          + pow(v,4)*pow(u,5)   // This was corrected!
          + 6*v*pow(w,2)*pow(u,5)
          - 5*pow(w,3)*pow(u,4)    // This was corrected!
          - pow(v,3)*w*pow(u,4)
          - 2*pow(v,5)*pow(u,3)
          - 6*pow(v,2)*pow(w,2)*pow(u,3)
          + 10*v*pow(w,3)*pow(u,2)
          + 6*pow(v,4)*w*pow(u,2)
          - 3*pow(w,4)*u
          - 6*pow(v,3)*pow(w,2)*u
          + 2*pow(v,2)*pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC02(const Real & u, const Real & v, const Real & w){
        Real result =   pow(w,2)*pow(u,5)
          + pow(v,2)*w*pow(u,4)
          - pow(v,4)*pow(u,3)
          - 4*v*pow(w,2)*pow(u,3)
          + 4*pow(w,3)*pow(u,2)
          + 3*pow(v,3)*w*pow(u,2)
          - 3*pow(v,2)*pow(w,2)*u
          + v*pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC11(const Real & u, const Real & v, const Real & w){
        Real result = - w*pow(u,8)
          - pow(v,2)*pow(u,7)
          + 7*v*w*pow(u,6)
          + 4*pow(v,3)*pow(u,5)
          - 5*pow(w,2)*pow(u,5)
          - 16*pow(v,2)*w*pow(u,4)
          - 4*pow(v,4)*pow(u,3)
          + 16*v*pow(w,2)*pow(u,3)
          - 3*pow(w,3)*pow(u,2)
          + 12*pow(v,3)*w*pow(u,2)
          - 12*pow(v,2)*pow(w,2)*u
          + 3*v*pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC12(const Real & u, const Real & v, const Real & w){
        Real result =  w*pow(u,6)
          + pow(v,2)*pow(u,5) // Fixed this!
          - 5*v*w*pow(u,4)  // Fixed this!
          - 2*pow(v,3)*pow(u,3)
          + 4*pow(w,2)*pow(u,3)
          + 6*pow(v,2)*w*pow(u,2)
          - 6*v*pow(w,2)*u
          + pow(w,3);
        return result;
      }

    template<class Real>
      __device__ __host__
      Real DerivativeCoefficients<Real>::computeC22(const Real & u, const Real & v, const Real & w){
        Real result = - w*pow(u,4)
          - pow(v,2)*pow(u,3)
          + 3*v*w*pow(u,2)
          - 3*pow(w,2)*u;
        return result;
      }

    template <class Real>
      __device__ __host__
      void  DerivativeCoefficients<Real>::set(const Real & u, const Real & v, const Real & w){
        const Real & denominator = 2.0*pow(w*(u*v-w),3); 
        b[0] = computeC00(u,v,w)/denominator;
        b[1] = computeC01(u,v,w)/denominator;
        b[2] = computeC02(u,v,w)/denominator;
        b[3] = computeC11(u,v,w)/denominator;
        b[4] = computeC12(u,v,w)/denominator;
        b[5] = computeC22(u,v,w)/denominator;
        return;
      }


    template<class Cmplx>
      __device__ __host__
      void accumBothDerivatives(Matrix<Cmplx,3>* result, const Matrix<Cmplx,3> & left, const Matrix<Cmplx,3> & right, const Matrix<Cmplx,3> & outer_prod)
      {
        const typename RealTypeId<Cmplx>::Type temp = 2.0*getTrace(left*outer_prod).x;
        for(int k=0; k<3; ++k){
          for(int l=0; l<3; ++l){
            // Need to write it this way to get it to work 
            // on the CPU. Not sure why.
            result->operator()(k,l).x += temp*right(k,l).x;
            result->operator()(k,l).y += temp*right(k,l).y;
          }
        }
        return;
      }


    template<class Cmplx>
      __device__ __host__
      void accumDerivatives(Matrix<Cmplx,3>* result, const Matrix<Cmplx,3> & left, const Matrix<Cmplx,3> & right, const Matrix<Cmplx,3> & outer_prod)
      {
        Cmplx temp = getTrace(left*outer_prod);
        for(int k=0; k<3; ++k){
          for(int l=0; l<3; ++l){
            result->operator()(k,l) = temp*right(k,l);
          }
        }
        return;
      }


    // Another real hack - fix this!
    template<class T>
      __device__ __host__
    T getAbsMin(const T* const array, int size){
      T min = fabs(array[0]);
      for(int i=1; i<size; ++i){
        T abs_val = fabs(array[i]);
        if((abs_val) < min){ min = abs_val; }   
      }
      return min;
    }

    // What a hack! Yuck!
    template<class Cmplx> 
      //__device__  __host__ 
      __host__ 
      void reciprocalRoot(Matrix<Cmplx,3>* res, DerivativeCoefficients<typename RealTypeId<Cmplx>::Type>* deriv_coeffs, 
								typename RealTypeId<Cmplx>::Type f[3], Matrix<Cmplx,3> & q){

        Matrix<Cmplx,3> qsq, tempq;
        qsq = q*q;
        tempq = qsq*q;

        typename RealTypeId<Cmplx>::Type c[3];
        typename RealTypeId<Cmplx>::Type g[3];
        c[0] = getTrace(q).x;
        c[1] = getTrace(qsq).x/2.0;
        c[2] = getTrace(tempq).x/3.0;

        g[0] = g[1] = g[2] = c[0]/3.;
        typename RealTypeId<Cmplx>::Type r,s,theta;
        s = c[1]/3. - c[0]*c[0]/18;
        r = c[2]/2. - (c[0]/3.)*(c[1] - c[0]*c[0]/9.);

        typename RealTypeId<Cmplx>::Type cosTheta = r/sqrt(s*s*s);
        if(fabs(s) < FORCE_UNITARIZE_EPS){
          cosTheta = 1.;
          s = 0.0; 
        }
        if(fabs(cosTheta)>1.0){ r>0 ? theta=0.0 : theta=FORCE_UNITARIZE_PI/3.0; }
        else{ theta = acos(cosTheta)/3.0; }

        s = 2.0*sqrt(s);
        for(int i=0; i<3; ++i){
          g[i] += s*cos(theta + (i-1)*FORCE_UNITARIZE_PI23);
        }

        for(int i=0; i<3; ++i){
          c[i] = g[i];
        }
        

        Matrix<Cmplx,3> tmp2;
        // eigenvalues have been computed 
        // compute the eigenvalues using the singular value decomposition
        computeSVD<Cmplx>(q,tempq,tmp2,g);

        float sum1, sum2;
        sum1 = 0.; sum2 = 0.;
        printf("Eigenvalues\n");
        for(int i=0; i<3; ++i){
          printf("%2.10f, %2.10f\n",c[i], g[i]);
          sum1 += c[i]; sum2 += g[i];
        }
        printf("difference of the sums = %2.20f\n",fabs(sum1-sum2));
        if(fabs(sum1-sum2)>0.001){
          printf("Discrepancy\n");
        }

        printf("\n\n");

        // New code!
        // Augment the eigenvalues
        // Also need to change q
        typename RealTypeId<Cmplx>::Type delta = getAbsMin(g,3);
        if(delta < HISQ_FORCE_FILTER){
          for(int i=0; i<3; ++i){ 
            g[i]     += HISQ_FORCE_FILTER; 
            q(i,i).x += HISQ_FORCE_FILTER;
          }
          qsq = q*q; // recalculate Q^2
        }
        

        // At this point we have finished with the c's 
        // use these to store sqrt(g)
        for(int i=0; i<3; ++i) c[i] = sqrt(g[i]);

        // done with the g's, use these to store u, v, w
        g[0] = c[0]+c[1]+c[2];
        g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
        g[2] = c[0]*c[1]*c[2];
        
        // set the derivative coefficients!
        deriv_coeffs->set(g[0], g[1], g[2]);

        const typename RealTypeId<Cmplx>::Type & denominator  = g[2]*(g[0]*g[1]-g[2]); 
        c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1]))/denominator;
        c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1])/denominator;
        c[2] =  g[0]/denominator;

        tempq = c[1]*q + c[2]*qsq;
        // Add a real scalar
        tempq(0,0).x += c[0];
        tempq(1,1).x += c[0];
        tempq(2,2).x += c[0];

        f[0] = c[0];
        f[1] = c[1];
        f[2] = c[2];


        *res = tempq;
        return;
      }



      // "v" denotes a "fattened" link variable
      template<class Cmplx>
      //__device__ __host__
      __host__
        void getUnitarizeForceSite(Matrix<Cmplx,3>* result, const Matrix<Cmplx,3> & v, const Matrix<Cmplx,3> & outer_prod)
        {
          typename RealTypeId<Cmplx>::Type f[3]; 
          typename RealTypeId<Cmplx>::Type b[6];

          Matrix<Cmplx,3> v_dagger = conj(v);  // okay!
          Matrix<Cmplx,3> q   = v_dagger*v;    // okay!

          Matrix<Cmplx,3> rsqrt_q;

          DerivativeCoefficients<typename RealTypeId<Cmplx>::Type> deriv_coeffs;

          reciprocalRoot<Cmplx>(&rsqrt_q, &deriv_coeffs, f, q);

          // Pure hack here
          b[0] = deriv_coeffs.getB00();
          b[1] = deriv_coeffs.getB01();
          b[2] = deriv_coeffs.getB02();
          b[3] = deriv_coeffs.getB11();
          b[4] = deriv_coeffs.getB12();
          b[5] = deriv_coeffs.getB22();


          Matrix<Cmplx,3> & local_result = *result;

          local_result = rsqrt_q*outer_prod;

          // We are now finished with rsqrt_q
          Matrix<Cmplx,3> qv_dagger  = q*v_dagger;
          Matrix<Cmplx,3> vv_dagger  = v*v_dagger; 
          Matrix<Cmplx,3> vqv_dagger = v*qv_dagger;
          Matrix<Cmplx,3> temp = f[1]*vv_dagger + f[2]*vqv_dagger;


          temp = f[1]*v_dagger + f[2]*qv_dagger;
          Matrix<Cmplx,3> conj_outer_prod = conj(outer_prod);


          temp = f[1]*v + f[2]*v*q;
          local_result = local_result + outer_prod*temp*v_dagger + f[2]*q*outer_prod*vv_dagger;

          local_result = local_result + v_dagger*conj_outer_prod*conj(temp) + f[2]*qv_dagger*conj_outer_prod*v_dagger;


          // now done with vv_dagger, I think
          Matrix<Cmplx,3> qsqv_dagger = q*qv_dagger;
          Matrix<Cmplx,3> pv_dagger   = b[0]*v_dagger + b[1]*qv_dagger + b[2]*qsqv_dagger;
          accumBothDerivatives(&local_result, v, pv_dagger, outer_prod);

          Matrix<Cmplx,3> rv_dagger = b[1]*v_dagger + b[3]*qv_dagger + b[4]*qsqv_dagger;
          Matrix<Cmplx,3> vq = v*q;
          accumBothDerivatives(&local_result, vq, rv_dagger, outer_prod);

          Matrix<Cmplx,3> sv_dagger = b[2]*v_dagger + b[4]*qv_dagger + b[5]*qsqv_dagger;
          Matrix<Cmplx,3> vqsq = vq*q;
          accumBothDerivatives(&local_result, vqsq, sv_dagger, outer_prod);
          return;
        } // get unit force term



/*
      // I don't need to swap between odd and even half lattices, do I
        template<class Cmplx>
          __global__ void getUnitarizeForceField(Cmplx* force_even, Cmplx* force_odd,
                                                 Cmplx* link_even, Cmplx* link_odd,
                                                 Cmplx* old_force_even, Cmplx* old_force_odd)
          {
            int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
       
            Cmplx* force;
            Cmplx* link;
            Cmplx* old_force;

            force = force_even;
            link = link_even;
            old_force = old_force_even;
            if(mem_idx >= Vh){
              mem_idx = mem_idx - Vh;
              force = force_odd;
              link = link_odd;
              old_force = old_force_odd;
            }

            Matrix<Cmplx,3> v, result, oprod;

           
            for(int dir=0; dir<4; ++dir){
              loadLinkVariableFromArray(&oprod, old_force, dir, mem_idx, Vh);
              loadLinkVariableFromArray(&v, link, dir, mem_idx, Vh);

              getUnitarizeForceSite<Cmplx>(&result, v, oprod); 

              writeLinkVariableToArray(force, result, dir, mem_idx, Vh); 
            }
            return;
          }

*/
        // template this! 
        void copyArrayToLink(Matrix<float2,3>* link, float* array){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              (*link)(i,j).x = array[(i*3+j)*2];
              (*link)(i,j).y = array[(i*3+j)*2 + 1];
            }
          }
          return;
        }
        
        // and this!
        void copyLinkToArray(float* array, const Matrix<float2,3>& link){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              array[(i*3+j)*2] = link(i,j).x;
              array[(i*3+j)*2 + 1] = link(i,j).y;
            }
          }
          return;
        }

        // and this!
        void printLink(Matrix<float2,3>& link){
          printf("(%lf, %lf)\t", link(0,0).x, link(0,0).y);
          printf("(%lf, %lf)\t", link(0,1).x, link(0,1).y);
          printf("(%lf, %lf)\n", link(0,2).x, link(0,2).y);
          printf("(%lf, %lf)\t", link(1,0).x, link(1,0).y);
          printf("(%lf, %lf)\t", link(1,1).x, link(1,1).y);
          printf("(%lf, %lf)\n", link(1,2).x, link(1,2).y);
          printf("(%lf, %lf)\t", link(2,0).x, link(2,0).y);
          printf("(%lf, %lf)\t", link(2,1).x, link(2,1).y);
          printf("(%lf, %lf)\n", link(2,2).x, link(2,2).y);
          printf("\n");
        }


        void unitarize_force_cpu(cpuGaugeField& cpuOldForce, cpuGaugeField& cpuGauge, cpuGaugeField &cpuNewForce)
        {
          printf("In unitarize_force_cpu\n");
          printf("Volume = %d\n",cpuOldForce.Volume());


          printf("FILTER = %lf\n",HISQ_FORCE_FILTER);
          Matrix<float2,3> old_force, new_force, v;
          for(int i=0; i<cpuGauge.Volume(); ++i){
           for(int dir=0; dir<4; ++dir){
             copyArrayToLink(&old_force, ((float*)(cpuOldForce.Gauge_p()) + (i*4 + dir)*18)); 
             copyArrayToLink(&v, ((float*)(cpuGauge.Gauge_p()) + (i*4 + dir)*18)); 
     
             if(i==0 && dir==0){
               printf("Printing link variable\n");
               printLink(v);
             } 

             printf("Calling getUnitarizeForceSite\n"); 
             getUnitarizeForceSite<float2>(&new_force, v, old_force);
            
             printLink(new_force);

             copyLinkToArray(((float*)(cpuNewForce.Gauge_p()) + (i*4 + dir)*18), new_force); 

           } // dir
          } // i
          return;
        }


        void unitarize_force_cuda(cudaGaugeField& cudaOldForce, cudaGaugeField& cudaGauge,  cudaGaugeField& cudaNewForce)
        {

          dim3 gridDim(cudaGauge.Volume()/BLOCK_DIM,1,1);
          dim3 blockDim(BLOCK_DIM,1,1);

          printf("Vh = %d\n",Vh);
          printf("BLOCK_DIM = %d\n",BLOCK_DIM);
          /*
          getUnitarizeForceField<<<gridDim,blockDim>>>((float2*)cudaNewForce.Even_p(), (float2*)cudaNewForce.Odd_p(), 
                                                       (float2*)cudaGauge.Even_p(), (float2*)cudaGauge.Odd_p(),
                                                      (float2*)cudaOldForce.Even_p(), (float2*)cudaOldForce.Odd_p());
     */
          return;
        }


  } // namespace fermion_force
} // namespace hisq



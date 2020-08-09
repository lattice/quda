#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>
#include <chebyshev_coeff.h> 

namespace quda {

DiracOverlapWilson *OvW;

  DiracOverlapWilson::DiracOverlapWilson(const DiracParam &param) : DiracWilson(param),prec0(1e-12),
       hw_evec(param.hw_evec),hw_eval(param.hw_eval),coef(param.coef),hw_size(param.hw_size) {}

  DiracOverlapWilson::DiracOverlapWilson(const DiracOverlapWilson &dirac) : DiracWilson(dirac) ,prec0(1e-12) {}

  DiracOverlapWilson::~DiracOverlapWilson() { }

  DiracOverlapWilson& DiracOverlapWilson::operator=(const DiracOverlapWilson &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
    }
    return *this;
  }
  
  void DiracOverlapWilson::Kernel(ColorSpinorField &out, const ColorSpinorField &in) const 
  {
       ApplyHWilson(out, in, *gauge,-kappa,in,QUDA_INVALID_PARITY, false, commDim,profile);
  }


  void DiracOverlapWilson::KernelSq_scaled(ColorSpinorField &out, ColorSpinorField &in,double cut) const 
  {
       double sc1=2/((1+8*kappa)*(1+8*kappa)*(1-cut));
       double sc2=(1+cut)/(1-cut);
  
       ColorSpinorField *tmp3=0, *tmp4=0;
       bool reset1 = newTmp(&tmp3, in);
       bool reset2 = newTmp(&tmp4, in);
                     
       Kernel(*tmp3, in);
       Kernel(*tmp4, *tmp3);
       blas::axpbyz(sc1,*tmp4,-sc2,in,out);
       
      deleteTmp(&tmp3, reset1);  
      deleteTmp(&tmp4, reset2);    
  
  }
  
  void DiracOverlapWilson::general_dov(ColorSpinorField &out, const ColorSpinorField &in, 
         double k0, double k1,double k2,double prec, const QudaParity parity) const
  {
       // only switch on comms needed for directions with a derivative
       int is=-(int)log(prec/0.2);
       if(is<0)is=0;
       if(is>coef.size()-1)is=coef.size()-1;

       bool reset1 = newTmp(&tmp1, in);
       checkFullSpinor(*tmp1, in);
       bool reset2 = newTmp(&tmp2, in);
       
       ColorSpinorField &high=*tmp1;
       ColorSpinorField &src=*tmp2;
       
       src=in;
       blas::zero(out);
       blas::zero(high);


       std::vector<Complex> inner(hw_size[is]);
       std::vector<ColorSpinorField *> src_vec;
       src_vec.push_back(&src);
       std::vector<ColorSpinorField *> out_vec;
       out_vec.push_back(&out);
       std::vector<ColorSpinorField *> hw_tmp=hw_evec; 
       blas::cDotProduct(inner.data(),hw_tmp,src_vec);
       for(int i=0;i<hw_size[is];i++)
             inner[i]=-inner[i];
       blas::caxpy(inner.data(),hw_tmp,src_vec);
       std::vector<Complex> sign(hw_size[is]);
       for(int i=0;i<hw_size[is];i++)
             sign[i]=(hw_eval[i]>0)?-inner[i]:inner[i];
       blas::caxpy(sign.data(),hw_tmp,out_vec);
       
/*
       for(int i=0;i<hw_size[is];i++)
       {
              Complex inner=blas::cDotProduct(*hw_evec[i],src);
              double n0=blas::norm2(src),n1=blas::norm2(*hw_evec[i]);
              blas::caxpy(-inner,*hw_evec[i],src);
              double sign=(hw_eval[i]>0)?1:-1;
              blas::caxpy(sign*inner,*hw_evec[i],out);
       }
*/
       
       double cut=pow(hw_eval[hw_size[is]-1]/(1+8*kappa),2);
       
       ColorSpinorField *pbn2=0,*pbn1=0,*pbn0=0,*ptmp;
       bool resetr2 = newTmp(&pbn2, in);
       bool resetr1 = newTmp(&pbn1, in);
       bool resetr0 = newTmp(&pbn0, in);
//       blas::zero(*pbn2);
//       blas::zero(*pbn1);
//       blas::zero(*pbn0);
       
       for(int i=coef[is].size()-1;i>=1;i--){
              if(i<coef[is].size()-1) KernelSq_scaled(high,*pbn1,cut);
              blas::axpbyz(2,high,-1,*pbn0,*pbn2); // use high as a tmp;
              blas::axpy(coef[is][i],src,*pbn2);
              ptmp=pbn0;pbn0=pbn1;pbn1=pbn2;pbn2=ptmp;
              // swap pointers;
       }

       KernelSq_scaled(high,*pbn1,cut);
       blas::axpbyz(1,high,-1,*pbn0,*pbn2);
       blas::axpy(coef[is][0],src,*pbn2);
       Kernel(high, *pbn2);
       
       blas::axpy(1.0/(1+8*kappa),high,out);

      deleteTmp(&pbn0, resetr0);  
      deleteTmp(&pbn1, resetr1);  
      deleteTmp(&pbn2, resetr2);  

       ApplyOverlapLinop(out,in,k0,k1,k2);
       
      deleteTmp(&tmp1, reset1);  
      deleteTmp(&tmp2, reset2);  
  
  }
  
  void DiracOverlapWilson::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			   double prec, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    double rho=4-0.5/kappa;
    general_dov(out,in,rho,rho,0.0,prec,parity);
    flops += 0ll*in.Volume();
  }
  

  void DiracOverlapWilson::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    general_dov(out,in,1+k*rho,k*rho,0.0,prec0,parity);
    flops += 0ll*in.Volume();
  }

  void DiracOverlapWilson::M(ColorSpinorField &out, const ColorSpinorField &in, double mass,double prec) const
  {
    checkFullSpinor(out, in);

    double eff_rho=rho-mass*0.5;
    general_dov(out,in,mass+eff_rho,eff_rho,0.0,prec,QUDA_INVALID_PARITY);

    flops += 0ll * in.Volume();
  }  

  void DiracOverlapWilson::MdagM(ColorSpinorField &out, const ColorSpinorField &in,double mass,int chirality,double prec) const
  {
    checkFullSpinor(out, in);
    double fac=(1-pow(mass*0.5/rho,2))/(2+0.5*pow(mass/rho,2));
    if(chirality==1)
    { 
       general_dov(out,in,1.0,fac,fac,prec,QUDA_INVALID_PARITY);
    }
    if(chirality==-1)
    {
       general_dov(out,in,1.0,-fac,fac,prec,QUDA_INVALID_PARITY);
    }
    if(chirality==0)
    {
      bool reset = newTmp(&tmp1, in);
      checkFullSpinor(*tmp1, in);

    //OVERLAP_TODO
      
    
      deleteTmp(&tmp1, reset);  
    }
  }

  void DiracOverlapWilson::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    //OVERLAP_TODO
    M(*tmp1, in);
    Mdag(out, *tmp1);    
    
    deleteTmp(&tmp1, reset);
  }

  void DiracOverlapWilson::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			    ColorSpinorField &x, ColorSpinorField &b, 
			    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracOverlapWilson::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				const QudaSolutionType solType) const
  {
    // do nothing
  }

} // namespace quda

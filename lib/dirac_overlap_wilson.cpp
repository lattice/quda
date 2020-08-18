#include <dirac_quda.h>
#include <blas_quda.h>
#include <eigensolve_quda.h>
#include <chebyshev_coeff.h>
#include <iostream>

namespace quda {

#define _chey_size_ 30
#define _timer_

#define TSTART() Start(__func__, __FILE__, __LINE__)
#define TSTOP() Stop(__func__, __FILE__, __LINE__)
#define TRESET() Reset(__func__, __FILE__, __LINE__)

  class DiracHwilson : public DiracMatrix {
       double cut;
public:
       DiracHwilson(const DiracOverlapWilson &d,double _cut) : DiracMatrix(d), cut(_cut*_cut) {}

       int getStencilSteps() const
       {
            return dirac->getStencilSteps(); 
       }

       bool hermitian() const {return true;}
       
       void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
       {
            ((DiracOverlapWilson*)dirac)->KernelSq_scaled(out,in,cut);
       }
       
       void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
            ColorSpinorField &tmp) const
       {
            ((DiracOverlapWilson*)dirac)->KernelSq_scaled(out,in,cut);
       }

       void operator()(ColorSpinorField &out, const ColorSpinorField &in,
            ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
       {
            ((DiracOverlapWilson*)dirac)->KernelSq_scaled(out,in,cut);
       }
  };

void setEigParam(QudaEigParam &eig_param,QudaInvertParam *inv_param,int eig_n_ev,int eig_n_kr,double eig_tol,
       double eig_amin,double eig_amax,int eig_poly_deg)
{
  eig_param.eig_type = QUDA_EIG_TR_LANCZOS;
  eig_param.spectrum = QUDA_SPECTRUM_LR_EIG;

  // The solver will exit when n_conv extremal eigenpairs have converged
  eig_param.n_conv = eig_n_ev;

  // Inverters will deflate only this number of vectors.
  eig_param.n_ev_deflate = eig_n_ev;

  eig_param.block_size = 1;
  eig_param.n_ev = eig_n_ev;
  eig_param.n_kr = eig_n_kr;
  eig_param.tol = eig_tol;
  eig_param.batched_rotate = 0;
  eig_param.require_convergence = QUDA_BOOLEAN_TRUE;
  eig_param.check_interval = 10;
  eig_param.max_restarts = 1000;
  eig_param.cuda_prec_ritz = QUDA_DOUBLE_PRECISION;

  eig_param.use_norm_op = QUDA_BOOLEAN_TRUE;
  eig_param.use_dagger = QUDA_BOOLEAN_FALSE;
  eig_param.compute_svd = QUDA_BOOLEAN_FALSE;

  eig_param.use_poly_acc = QUDA_BOOLEAN_TRUE ;
  eig_param.poly_deg = eig_poly_deg;
  eig_param.a_min = eig_amin;
  eig_param.a_max = eig_amax;

  eig_param.arpack_check = QUDA_BOOLEAN_FALSE;
  strcpy(eig_param.arpack_logfile, "arpack_logfile.log");
  strcpy(eig_param.QUDA_logfile, "QUDA_logfile.log");

  strcpy(eig_param.vec_infile, "");
  strcpy(eig_param.vec_outfile, "");
  eig_param.save_prec = QUDA_DOUBLE_PRECISION;
  eig_param.io_parity_inflate = QUDA_BOOLEAN_FALSE;
}

//!< Profiler for EigensolveHW
static TimeProfile profileEigensolve("EigensolveHW");

  DiracOverlapWilson::DiracOverlapWilson(const DiracParam &param) : 
  	DiracWilson(param),prec0(1e-12)
  {
        if (gauge == nullptr) errorQuda("Gauge field must be loaded");
        if (param.inv_param==0) errorQuda("inv_param=0. Would not be the overlap_wilson action");
        if (param.inv_param->dslash_type != QUDA_OVERLAP_WILSON_DSLASH) 
             errorQuda("The action type is not Overlap");

        int size=param.inv_param->eigen_size;
        double deflation_cut=0.07;
        if(size>0)
        {
	    hw_evec.resize(0);
            ColorSpinorParam cpuParam(nullptr,*(param.inv_param),gauge->X(),false,param.inv_param->input_location);
	    ColorSpinorParam cudaParam(cpuParam, *(param.inv_param));
            cudaParam.create = QUDA_ZERO_FIELD_CREATE;
            for (size_t i = 0; i < size; i++) hw_evec.push_back(ColorSpinorField::Create(cudaParam)); 

            hw_eval.resize(size);
 
            //set eig_param;
            QudaEigParam eig_param = newQudaEigParam();
	    double eig_amin=pow(param.inv_param->eigen_cut,2);
	    double eig_amax=pow(1+2*kappa,2);
            setEigParam(eig_param,param.inv_param,size,param.inv_param->krylov_space,1e-13,
                  eig_amin,eig_amax,100);

            DiracHwilson hw(*this,eig_amin/eig_amax);
	    std::vector<Complex> evals(size);
            EigenSolver *eig_solve = EigenSolver::create(&eig_param, hw, profileEigensolve);
            (*eig_solve)(hw_evec, evals);
	    delete eig_solve;

            bool reset1 = newTmp(&tmp1, *hw_evec[0]);
            checkFullSpinor(*tmp1, *hw_evec[0]);
            
            for (int i = 0; i < hw_evec.size(); i++) 
            {
                 Kernel(*tmp1,*hw_evec[i]);
                 hw_eval[i]=blas::reDotProduct(*tmp1,*hw_evec[i]);
		 double dtmp=blas::axpyNorm(-hw_eval[i],*hw_evec[i],*tmp1);
		 printfQuda("%05d: val=%20.10e, residual=%20.10e\n",i,hw_eval[i],sqrt(dtmp));
            }
            deleteTmp(&tmp1, reset1); 
            
            deflation_cut=hw_eval[size-1];
        }

        calc_coef(deflation_cut);
        build_hw=true;
  }

  DiracOverlapWilson::DiracOverlapWilson(const DiracParam &param, std::vector<ColorSpinorField*> &evecs, std::vector<double> &evals,
        std::vector<std::vector<double> > &coefs, std::vector<int> &sizes):
        DiracWilson(param),prec0(1e-12),
        hw_evec(evecs),hw_eval(evals),
        coef(coefs),hw_size(sizes) {build_hw=false;}

  DiracOverlapWilson::~DiracOverlapWilson() {
        if(build_hw==true)
            for (size_t i = 0; i < hw_evec.size(); i++) delete hw_evec[i];
  }
  
  void DiracOverlapWilson::calc_coef(double cut)
  {
       coef.resize(_chey_size_);
       hw_size.resize(_chey_size_);

       double cut_sq=pow(cut/(1+8*kappa),2);
       double low=0.0,high=0.0;
       if(hw_evec.size()>0)
       {
            bool reset1 = newTmp(&tmp1, *hw_evec[0]);
            checkFullSpinor(*tmp1, *hw_evec[0]);
            bool reset2 = newTmp(&tmp2, *hw_evec[0]);
       	    Timer t_eps;
       	    t_eps.TSTART();
       	    eps_l(*tmp2,*tmp1,hw_evec.size());
       	    t_eps.TSTOP();
       	    low=t_eps.time/hw_evec.size();
       	    t_eps.TRESET();
       	    t_eps.TSTART();
       	    for(int i=0;i<10;i++)
       	    {
       	           KernelSq_scaled(*tmp2,*tmp1,0.1);
       	           *tmp1=*tmp2;
       	    }
       	    t_eps.TSTOP();
       	    high=t_eps.time/10;
      	    deleteTmp(&tmp1, reset1);
            deleteTmp(&tmp2, reset2);
       }
       
       printfQuda("low_op=%10.5f s, high_op=%10.5f s\n",low,high);
       
       for(int i=0;i<_chey_size_;i++)
       {
            double prec=0.2*exp(-1.0*i);
            chebyshev_coef _coef;
            _coef.set_HwSize(hw_evec.size(),cut_sq,prec,high,low);
            hw_size[i]=_coef.get_HwSize();
            double cut_sq_tmp=(hw_evec.size()>0)?pow(hw_eval[hw_size[i]-1]/(1+8*kappa),2):cut_sq;
            _coef.run(cut_sq_tmp,prec);
	    printfQuda("prec=%13.3e, cut=%13.3e, size=%6d, order=%6d\n",prec,cut_sq_tmp,hw_size[i],_coef.size());
	    coef[i].resize(_coef.size());
            coef[i]=_coef;
       }
  }

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


  void DiracOverlapWilson::KernelSq_scaled(ColorSpinorField &out, const ColorSpinorField &in,double cut) const 
  {
       double sc1=2/((1+8*kappa)*(1+8*kappa)*(1-cut));
       double sc2=(1+cut)/(1-cut);
  
       ColorSpinorField *tmp3=0, *tmp4=0;
       bool reset1 = newTmp(&tmp3, in);
       bool reset2 = newTmp(&tmp4, in);
                     
       Kernel(*tmp3, in);
       Kernel(*tmp4, *tmp3);
       *tmp3=in;
       blas::axpbyz(sc1,*tmp4,-sc2,*tmp3,out);
       
      deleteTmp(&tmp3, reset1);  
      deleteTmp(&tmp4, reset2);    
  
  }
  
  void DiracOverlapWilson::eps_l(ColorSpinorField &out, ColorSpinorField &in, int size) const
  {
       if(size>hw_evec.size())  errorQuda("The size used in eps_l is too large, %d > %d",size,hw_evec.size());
       
       std::vector<Complex> inner(size);
       std::vector<ColorSpinorField *> src_vec;
       src_vec.push_back(&in);
       std::vector<ColorSpinorField *> out_vec;
       out_vec.push_back(&out);
       std::vector<ColorSpinorField *> hw_tmp(size); 
       for(int i=0;i<size;i++) hw_tmp[i]=hw_evec[i];

       blas::cDotProduct(inner.data(),hw_tmp,src_vec);

       std::vector<Complex> sign(size);
       for(int i=0;i<size;i++)
             sign[i]=(hw_eval[i]>0)?inner[i]:-inner[i]; 
       blas::caxpy(sign.data(),hw_tmp,out_vec);

       for(int i=0;i<size;i++)
             inner[i]=-inner[i];
       blas::caxpy(inner.data(),hw_tmp,src_vec);
  
/*
       for(int i=0;i<size;i++)
       {
              Complex inner=blas::cDotProduct(*hw_evec[i],src);
              double n0=blas::norm2(src),n1=blas::norm2(*hw_evec[i]);
              blas::caxpy(-inner,*hw_evec[i],src);
              double sign=(hw_eval[i]>0)?1:-1;
              blas::caxpy(sign*inner,*hw_evec[i],out);
       }
*/
       
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

#ifdef _timer_
       double time_l,time_h;
       Timer t_eps;
       t_eps.TSTART();
#endif
       eps_l(out,src,hw_size[is]);
#ifdef _timer_       
       t_eps.TSTOP();
       time_l=t_eps.time;
       t_eps.TRESET();
       t_eps.TSTART();
#endif
       
       double cut=pow(hw_eval[hw_size[is]-1]/(1+8*kappa),2);
       
       ColorSpinorField *pbn2=0,*pbn1=0,*pbn0=0,*ptmp;
       bool resetr2 = newTmp(&pbn2, in);
       bool resetr1 = newTmp(&pbn1, in);
       bool resetr0 = newTmp(&pbn0, in);
       
       for(int i=coef[is].size()-1;i>=1;i--){
//              if(i<coef[is].size()-1) KernelSq_scaled(high,*pbn1,cut);
              blas::axpbyz(2,high,-1,*pbn0,*pbn2); // use high as a tmp;
              blas::axpy(coef[is][i],src,*pbn2);
              // swap pointers;
              ptmp=pbn0;pbn0=pbn1;pbn1=pbn2;pbn2=ptmp;
              KernelSq_scaled(high,*pbn1,cut);
       }

//       KernelSq_scaled(high,*pbn1,cut);
       blas::axpbyz(1,high,-1,*pbn0,*pbn2);
       blas::axpy(coef[is][0],src,*pbn2);
       Kernel(high, *pbn2);
       
       blas::axpy(1.0/(1+8*kappa),high,out);
       
#ifdef _timer_
       t_eps.TSTOP();
       time_h=t_eps.time;
       printfQuda("low=%10.5f s, high=%10.5f s\n",time_l,time_h);
#endif       

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

  
  DiracOverlapWilsonPC::DiracOverlapWilsonPC(const DiracParam &param)
    : DiracOverlapWilson(param)
  {

  }

  DiracOverlapWilsonPC::DiracOverlapWilsonPC(const DiracOverlapWilsonPC &dirac) 
    : DiracOverlapWilson(dirac)
  {

  }

  DiracOverlapWilsonPC::~DiracOverlapWilsonPC()
  {

  }

  DiracOverlapWilsonPC& DiracOverlapWilsonPC::operator=(const DiracOverlapWilsonPC &dirac)
  {
    if (&dirac != this) {
      DiracOverlapWilson::operator=(dirac);
    }
    return *this;
  }

  void DiracOverlapWilsonPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;

    bool reset = newTmp(&tmp1, in);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("MatPCType %d not valid for DiracOverlapWilsonPC", matpcType);
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracOverlapWilsonPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracOverlapWilsonPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			      ColorSpinorField &x, ColorSpinorField &b, 
			      const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {
      // we desire solution to full system
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
	// src = b_e + k D_eo b_o
	DslashXpay(x.Odd(), b.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
	src = &(x.Odd());
	sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
	// src = b_o + k D_oe b_e
	DslashXpay(x.Even(), b.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
	src = &(x.Even());
	sol = &(x.Odd());
      } else {
	errorQuda("MatPCType %d not valid for DiracOverlapWilsonPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }

  }

  void DiracOverlapWilsonPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				  const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    // create full solution

    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // x_o = b_o + k D_oe x_e
      DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // x_e = b_e + k D_eo x_o
      DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    } else {
      errorQuda("MatPCType %d not valid for DiracOverlapWilsonPC", matpcType);
    }
  }

} // namespace quda

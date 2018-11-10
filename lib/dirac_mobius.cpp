#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracMobius::DiracMobius(const DiracParam &param)
    : DiracDomainWall(param), zMobius(false) {
    memcpy(b_5, param.b_5, sizeof(Complex)*param.Ls);
    memcpy(c_5, param.c_5, sizeof(Complex)*param.Ls);

    // check if doing zMobius
    for (int i=0; i<Ls; i++) {
      if (b_5[i].imag() != 0.0 || c_5[i].imag() != 0.0 ||
          (i<Ls-1 && (b_5[i] != b_5[i+1] || c_5[i] != c_5[i+1]) ) ) {
        zMobius = true;
      }
    }

    if (getVerbosity() > QUDA_VERBOSE) {
      if (zMobius) printfQuda("%s: Detected variable or complex cofficients: using zMobius\n", __func__);
      else printfQuda("%s: Detected fixed real cofficients: using regular Mobius\n", __func__);
    }
  }

  DiracMobius::DiracMobius(const DiracMobius &dirac)
    : DiracDomainWall(dirac), zMobius(dirac.zMobius) {
    memcpy(b_5, dirac.b_5, sizeof(Complex)*Ls);
    memcpy(c_5, dirac.c_5, sizeof(Complex)*Ls);
  }

  DiracMobius::~DiracMobius() { }

  DiracMobius& DiracMobius::operator=(const DiracMobius &dirac)
  {
    if (&dirac != this) {
      DiracDomainWall::operator=(dirac);
      memcpy(b_5, dirac.b_5, sizeof(Complex)*Ls);
      memcpy(c_5, dirac.c_5, sizeof(Complex)*Ls);
      zMobius = dirac.zMobius;
    }

    return *this;
  }

// Modification for the 4D preconditioned Mobius domain wall operator
  void DiracMobius::Dslash4(ColorSpinorField &out, const ColorSpinorField &in,
			    const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 0, profile);

    flops += 1320LL*(long long)in.Volume();
  }
   
  void DiracMobius::Dslash4pre(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
#if 1
    ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS_PRE);
#else
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b_5, c_5, m5, commDim, 1, profile);
#endif

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 72LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  // Unlike DWF-4d, the Mobius variant here applies the full M5 operator and not just D5
  void DiracMobius::Dslash5(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
#if 1
    ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS);
#else
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b_5, c_5, m5, commDim, 2, profile);
#endif

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 48LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  // Modification for the 4D preconditioned Mobius domain wall operator
  void DiracMobius::Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in,
				const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, b5_, c5_, m5, commDim, 0, profile);

    flops += (1320LL+48LL)*(long long)in.Volume();
  }

  void DiracMobius::Dslash4preXpay(ColorSpinorField &out, const ColorSpinorField &in,
				   const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

#if 1
    ApplyDslash5(out, in, x, mass, m5, b_5, c_5, k, dagger, DSLASH5_MOBIUS_PRE);
#else
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, b_5, c_5, m5, commDim, 1, profile);
#endif

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (72LL+48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  // The xpay operator bakes in a factor of kappa_b^2
  void DiracMobius::Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in,
				const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

#if 1
    ApplyDslash5(out, in, x, mass, m5, b_5, c_5, k, dagger, DSLASH5_MOBIUS);
#else
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, b_5, c_5, m5, commDim, 2, profile);
#endif

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (96LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracMobius::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    ColorSpinorField *tmp=0; // this hack allows for tmp2 to be full or parity field
    if (tmp2) {
      if (tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = &(tmp2->Even());
      else tmp = tmp2;
    }
    bool reset = newTmp(&tmp, in.Even());

    // FIXME broken for variable coefficients
    double kappa_b = 0.5 / (b_5[0].real()*(4.0+m5)+1.0);

    // cannot use Xpay variants since it will scale incorrectly for this operator

    Dslash4pre(out.Odd(), in.Even(), QUDA_EVEN_PARITY);
    Dslash4(*tmp, out.Odd(), QUDA_ODD_PARITY);
    Dslash5(out.Odd(), in.Odd(), QUDA_ODD_PARITY);
    blas::axpy(-kappa_b, *tmp, out.Odd());

    Dslash4pre(out.Even(), in.Odd(), QUDA_ODD_PARITY);
    Dslash4(*tmp, out.Even(), QUDA_EVEN_PARITY);
    Dslash5(out.Even(), in.Even(), QUDA_EVEN_PARITY);
    blas::axpy(-kappa_b, *tmp, out.Even());

    deleteTmp(&tmp, reset);
  }

  void DiracMobius::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracMobius::prepare(ColorSpinorField* &src, ColorSpinorField* &sol, ColorSpinorField &x, ColorSpinorField &b,
			    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracMobius::reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // do nothing
  }


  DiracMobiusPC::DiracMobiusPC(const DiracParam &param) : DiracMobius(param) {  }

  DiracMobiusPC::DiracMobiusPC(const DiracMobiusPC &dirac) : DiracMobius(dirac) {  }

  DiracMobiusPC::~DiracMobiusPC() { }

  DiracMobiusPC& DiracMobiusPC::operator=(const DiracMobiusPC &dirac)
  {
    if (&dirac != this) {
      DiracMobius::operator=(dirac);
    }

    return *this;
  }

  void DiracMobiusPC::Dslash5inv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

#if 1
    ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, zMobius ? M5_INV_ZMOBIUS : M5_INV_MOBIUS);

#else
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b_5, c_5, m5, commDim, 3, profile);
#endif
    
    if (0) {
      //M5 = 1 + 0.5*kappa_b/kappa_c * D5
      using namespace blas;
      cudaColorSpinorField A(out);
      Dslash5(A, out, parity);
      printfQuda("Dslash5Xpay = %e M5inv = %e in = %e\n", norm2(A), norm2(out), norm2(in));
      exit(0);
    }

    long long Ls = in.X(4);
    flops += 144LL*(long long)in.Volume()*Ls + 3LL*Ls*(Ls-1LL);
  }

  // The xpay operator bakes in a factor of kappa_b^2
  void DiracMobiusPC::Dslash5invXpay(ColorSpinorField &out, const ColorSpinorField &in,
                                     const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

#if 1
    ApplyDslash5(out, in, x, mass, m5, b_5, c_5, k, dagger, zMobius ? M5_INV_ZMOBIUS : M5_INV_MOBIUS);
#else
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, b_5, c_5, m5, commDim, 3, profile);
#endif

    long long Ls = in.X(4);
    flops +=  (144LL*Ls + 48LL)*(long long)in.Volume() + 3LL*Ls*(Ls-1LL);
  }

  // Apply the even-odd preconditioned mobius DWF operator
  //Actually, Dslash5 will return M5 operation and M5 = 1 + 0.5*kappa_b/kappa_c * D5
  void DiracMobiusPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset1 = newTmp(&tmp1, in);

    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    //QUDA_MATPC_EVEN_EVEN_ASYMMETRIC : M5 - kappa_b^2 * D4_{eo}D4pre_{oe}D5inv_{ee}D4_{eo}D4pre_{oe}
    //QUDA_MATPC_ODD_ODD_ASYMMETRIC : M5 - kappa_b^2 * D4_{oe}D4pre_{eo}D5inv_{oo}D4_{oe}D4pre_{eo}
    if (symmetric && !dagger) {
      Dslash4pre(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash5invXpay(out, *tmp1, parity[1], in, -1.0);
    } else if (symmetric && dagger) {
      Dslash5inv(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash4pre(*tmp1, out, parity[0]);
      Dslash5inv(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash4preXpay(out, *tmp1, parity[1], in, -1.0);
    } else if (!symmetric && !dagger) {
      Dslash4pre(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash5Xpay(out, in, parity[1], *tmp1, -1.0);
    } else if (!symmetric && dagger) {
      Dslash4(*tmp1, in, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4(out, *tmp1, parity[1]);
      Dslash4pre(*tmp1, out, parity[1]);
      Dslash5Xpay(out, in, parity[1], *tmp1, -1.0);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracMobiusPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracMobiusPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
      ColorSpinorField &x, ColorSpinorField &b, 
      const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else { // we desire solution to full system
      // prepare function in MDWF is not tested yet.
      bool reset = newTmp(&tmp1, b.Even());

      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
	// src = D5^-1 (b_e + k D4_eo * D4pre * D5^-1 b_o)
	src = &(x.Odd());
	Dslash5inv(*tmp1, b.Odd(), QUDA_ODD_PARITY);
        Dslash4pre(*src, *tmp1, QUDA_ODD_PARITY);
        Dslash4Xpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), 1.0);
	Dslash5inv(*src, *tmp1, QUDA_EVEN_PARITY);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = b_o + k D4_oe * D4pre * D5inv b_e
        src = &(x.Even());
        Dslash5inv(*tmp1, b.Even(), QUDA_EVEN_PARITY);
        Dslash4pre(*src, *tmp1, QUDA_EVEN_PARITY);
        Dslash4Xpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), 1.0);
	Dslash5inv(*src, *tmp1, QUDA_ODD_PARITY);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo * D4pre * D5inv b_o
        src = &(x.Odd());
        Dslash5inv(*src, b.Odd(), QUDA_ODD_PARITY);
        Dslash4pre(*tmp1, *src, QUDA_ODD_PARITY);
        Dslash4Xpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), 1.0);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe * D4pre * D5inv b_e
        src = &(x.Even());
        Dslash5inv(*src, b.Even(), QUDA_EVEN_PARITY);
        Dslash4pre(*tmp1, *src, QUDA_EVEN_PARITY);
        Dslash4Xpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), 1.0);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracMobiusPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want

      deleteTmp(&tmp1, reset);
    }
  }

  void DiracMobiusPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
      const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    bool reset1 = newTmp(&tmp1, x.Even());

    // create full solution
    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN ||
	matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // psi_o = M5^-1 (b_o + k_b D4_oe D4pre x_e)
      Dslash4pre(x.Odd(), x.Even(), QUDA_EVEN_PARITY);
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_ODD_PARITY, b.Odd(), 1.0);
      Dslash5inv(x.Odd(), *tmp1, QUDA_ODD_PARITY);
    } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	       matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // psi_e = M5^-1 (b_e + k_b D4_eo D4pre x_o)
      Dslash4pre(x.Even(), x.Odd(), QUDA_ODD_PARITY);
      Dslash4Xpay(*tmp1, x.Even(), QUDA_EVEN_PARITY, b.Even(), 1.0);
      Dslash5inv(x.Even(), *tmp1, QUDA_EVEN_PARITY);
    } else {
      errorQuda("MatPCType %d not valid for DiracMobiusPC", matpcType);
    }

    deleteTmp(&tmp1, reset1);
  }

// All the partial kernel implementations
 
  void DiracMobius::Dslash4Partial(ColorSpinorField &out, const ColorSpinorField &in,
			    const QudaParity parity, int sp_idx_length, int R_[4], int_fastdiv Xs_[4],
          bool expanding_, std::array<int,4> Rz_) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
     
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }   
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 0, profile, sp_idx_length, R_, Xs_, expanding_, Rz_);

    if(expanding_){
      long long vol = (Xs_[0]+2*(R_[0]-Rz_[0]))*(Xs_[1]+2*(R_[1]-Rz_[1]))*(Xs_[2]+2*(R_[2]-Rz_[2]))*(Xs_[3]+2*(R_[3]-Rz_[3]))/2;
			flops += 1320LL*vol*(long long)in.X(4);
    }else{
      flops += 1320LL*(long long)sp_idx_length*(long long)in.X(4);
    }
  } 
  
  void DiracMobius::Dslash4prePartial(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
    int sp_idx_length, int R_[4], int_fastdiv Xs_[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
     
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 1, profile, sp_idx_length, R_, Xs_);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*sp_idx_length;
    long long wall = 2*sp_idx_length;
    flops += 72LL*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
  }

  void DiracMobiusPC::Dslash5invPartial(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
    int sp_idx_length, int R_[4], int_fastdiv Xs_[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
     
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }   
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 3, profile, sp_idx_length, R_, Xs_);

    long long Ls = in.X(4);
    flops += 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
  }

  void DiracMobius::Dslash4preXpayPartial(ColorSpinorField &out, const ColorSpinorField &in,
				   const QudaParity parity, const ColorSpinorField &x, const double &k,
           int sp_idx_length, int R_[4], int_fastdiv Xs_[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }   
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, b5_, c5_, m5, commDim, 1, profile, sp_idx_length, R_, Xs_);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*sp_idx_length;
    long long wall = 2*sp_idx_length;
    flops += (72LL+48LL)*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
  }
 
  void DiracMobiusPC::dslash4_dslash5inv_dslash4pre_partial(ColorSpinorField &out, const ColorSpinorField &in,
			    const QudaParity parity, int sp_idx_length, int R_[4], int_fastdiv Xs_[4],
          bool expanding_, std::array<int,4> Rz_) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }      
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 4, profile, sp_idx_length, R_, Xs_, expanding_, Rz_);
    
    long long Ls = in.X(4);
		long long bulk = (Ls-2)*sp_idx_length;
    long long wall = 2*sp_idx_length;

    if(expanding_){
      long long vol = (Xs_[0]+R_[0]-Rz_[0])*(Xs_[1]+R_[1]-Rz_[1])*(Xs_[2]+R_[2]-Rz_[2])*(Xs_[3]+R_[3]-Rz_[3])/2;
//      flops += 1320LL*vol*(long long)in.X(4) + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
      flops += 1320LL*vol*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) + 72LL*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
//			printfQuda("ddd flops: %lld, this: %lld,\n", flops, 1320LL*vol*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) );
    }else{
      flops += 1320LL*(long long)sp_idx_length*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) + 72LL*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
    }
  }
	// The "new" fused kernel 
  void DiracMobiusPC::dslash4_dslash5inv_dslash4pre(ColorSpinorField &out, const ColorSpinorField &in,
			    const QudaParity parity, int shift[4], int halo_shift[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
    apply_fused_dslash(out, in, *gauge, out, in, mass, m5, b_5, c_5, dagger, parity, shift, halo_shift, dslash4_dslash5pre_dslash5inv);
    // mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		//    &static_cast<const cudaColorSpinorField&>(in),
		//    parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 4, profile, sp_idx_length, R_, Xs_, expanding_, Rz_);
    
//     long long Ls = in.X(4);
// 		long long bulk = (Ls-2)*sp_idx_length;
//     long long wall = 2*sp_idx_length;
// 
//     if(expanding_){
//       long long vol = (Xs_[0]+R_[0]-Rz_[0])*(Xs_[1]+R_[1]-Rz_[1])*(Xs_[2]+R_[2]-Rz_[2])*(Xs_[3]+R_[3]-Rz_[3])/2;
// //      flops += 1320LL*vol*(long long)in.X(4) + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
//       flops += 1320LL*vol*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) + 72LL*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
// //			printfQuda("ddd flops: %lld, this: %lld,\n", flops, 1320LL*vol*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) );
//     }else{
//       flops += 1320LL*(long long)sp_idx_length*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) + 72LL*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
//     }
  }

	void DiracMobiusPC::dslash4_dagger_dslash4pre_dagger_dslash5inv_dagger_partial(ColorSpinorField &out, const ColorSpinorField &in,
			    const QudaParity parity, int sp_idx_length, int R_[4], int_fastdiv Xs_[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }   
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 6, profile, sp_idx_length, R_, Xs_);
    
    long long Ls = in.X(4);
		long long bulk = (Ls-2)*sp_idx_length;
    long long wall = 2*sp_idx_length;
    flops += 1320LL*(long long)sp_idx_length*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) + 72LL*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
  }

	void DiracMobiusPC::dslash4_dslash5inv_xpay_dslash5inv_dagger_partial(ColorSpinorField &out, const ColorSpinorField &in,
			    const QudaParity parity, const ColorSpinorField &x, const double &k, int sp_idx_length, int R_[4], int_fastdiv Xs_[4],
          bool expanding_, std::array<int,4> Rz_) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
     
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
			 mass, k, b5_, c5_, m5, commDim, 5, profile, sp_idx_length, R_, Xs_, expanding_, Rz_);
    
    long long Ls = in.X(4);
		long long bulk = (Ls-2)*sp_idx_length;
    long long wall = 2*sp_idx_length;

    if(expanding_){
      long long vol = (Xs_[0]+R_[0]-Rz_[0])*(Xs_[1]+R_[1]-Rz_[1])*(Xs_[2]+R_[2]-Rz_[2])*(Xs_[3]+R_[3]-Rz_[3])/2;
//      flops += 1320LL*vol*(long long)in.X(4) + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
      flops += 1368LL*vol*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
//			printfQuda("ddd flops: %lld, this: %lld,\n", flops, 1320LL*vol*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL) );
    }else{
      flops += 1368LL*(long long)sp_idx_length*Ls + 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
    }
  }
  
  void DiracMobiusPC::dslash4_dagger_dslash4pre_dagger_xpay_partial(ColorSpinorField &out, const ColorSpinorField &in,
			    const QudaParity parity, const ColorSpinorField &x, const double &k, int sp_idx_length, int R_[4], int_fastdiv Xs_[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
			 mass, k, b5_, c5_, m5, commDim, 7, profile, sp_idx_length, R_, Xs_);
    
    long long Ls = in.X(4);
		long long bulk = (Ls-2)*sp_idx_length;
    long long wall = 2*sp_idx_length;
    
    flops += 1368LL*(long long)sp_idx_length*Ls + 72LL*(long long)sp_idx_length*Ls + 96LL*bulk + 120LL*wall;
  } 
   
	void DiracMobiusPC::dslash5inv_sm_partial(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
    int sp_idx_length, int R_[4], int_fastdiv Xs_[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 8, profile, sp_idx_length, R_, Xs_);

    long long Ls = in.X(4);
    flops += 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
  }
	
  void DiracMobiusPC::dslash5inv_sm_tc_partial(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
    const double scale, int sp_idx_length, int R_[4], int_fastdiv Xs_[4]) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    double b5_[QUDA_MAX_DWF_LS], c5_[QUDA_MAX_DWF_LS];
    for (int i=0; i<Ls; i++) { b5_[i] = b_5[i].real(); c5_[i] = c_5[i].real(); }
   
#if 1
    apply_dslash5_tensor_core(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, scale, M5_INV_MOBIUS);
#else
    mdwf_dslash_cuda_partial(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, b5_, c5_, m5, commDim, 9, profile, sp_idx_length, R_, Xs_);
#endif
    
    long long Ls = in.X(4);
    flops += 144LL*(long long)sp_idx_length*Ls*Ls + 3LL*Ls*(Ls-1LL);
  }

} // namespace quda

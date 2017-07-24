#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  namespace domainwall {
#include <dslash_init.cuh>
  }

  DiracDomainWall::DiracDomainWall(const DiracParam &param) : 
    DiracWilson(param, 5), m5(param.m5), kappa5(0.5/(5.0 + m5)) 
  { 
    domainwall::initConstants(*param.gauge, profile);
  }

  DiracDomainWall::DiracDomainWall(const DiracDomainWall &dirac) : 
    DiracWilson(dirac), m5(dirac.m5), kappa5(0.5/(5.0 + m5)) 
  { 
    domainwall::initConstants(*dirac.gauge, profile);
  }

  DiracDomainWall::~DiracDomainWall() { }

  DiracDomainWall& DiracDomainWall::operator=(const DiracDomainWall &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
      m5 = dirac.m5;
      kappa5 = dirac.kappa5;
    }
    return *this;
  }

  void DiracDomainWall::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION) {
      domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, 
			   &static_cast<const cudaColorSpinorField&>(in), 
			   parity, dagger, 0, mass, 0, commDim, profile);   
    } else {
      errorQuda("Not implemented");
    }

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 1320LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracDomainWall::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
				   const QudaParity parity, const ColorSpinorField &x,
				   const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (checkLocation(out, in, x) == QUDA_CUDA_FIELD_LOCATION) {
      domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge, 
			   &static_cast<const cudaColorSpinorField&>(in), 
			   parity, dagger, 
			   &static_cast<const cudaColorSpinorField&>(x), 
			   mass, k, commDim, profile);   
    } else {
      errorQuda("Not implemented");
    }

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (1320LL+48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracDomainWall::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), -kappa5);
    DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), -kappa5);
  }

  void DiracDomainWall::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracDomainWall::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				ColorSpinorField &x, ColorSpinorField &b, 
				const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracDomainWall::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				    const QudaSolutionType solType) const
  {
    // do nothing
  }

  DiracDomainWallPC::DiracDomainWallPC(const DiracParam &param)
    : DiracDomainWall(param)
  {

  }

  DiracDomainWallPC::DiracDomainWallPC(const DiracDomainWallPC &dirac) 
    : DiracDomainWall(dirac)
  {

  }

  DiracDomainWallPC::~DiracDomainWallPC()
  {

  }

  DiracDomainWallPC& DiracDomainWallPC::operator=(const DiracDomainWallPC &dirac)
  {
    if (&dirac != this) {
      DiracDomainWall::operator=(dirac);
    }

    return *this;
  }

  // Apply the even-odd preconditioned clover-improved Dirac operator
  void DiracDomainWallPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    double kappa2 = -kappa5*kappa5;

    bool reset = newTmp(&tmp1, in);

    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracDomainWallPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    //M(out, in);
    //Mdag(out, out);
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracDomainWallPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
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
        DslashXpay(x.Odd(), b.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa5);
        src = &(x.Odd());
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = b_o + k D_oe b_e
        DslashXpay(x.Even(), b.Even(), QUDA_ODD_PARITY, b.Odd(), kappa5);
        src = &(x.Even());
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }

  }

  void DiracDomainWallPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				      const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    // create full solution

    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // x_o = b_o + k D_oe x_e
      DslashXpay(x.Odd(), x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa5);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // x_e = b_e + k D_eo x_o
      DslashXpay(x.Even(), x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa5);
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWallPC", matpcType);
    }
  }


} // namespace quda

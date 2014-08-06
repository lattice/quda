#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  //!NEW
  DiracDomainWall::DiracDomainWall(const DiracParam &param) : 
    DiracWilson(param, 5), m5(param.m5), kappa5(0.5/(5.0 + m5)) { }

  DiracDomainWall::DiracDomainWall(const DiracDomainWall &dirac) : 
    DiracWilson(dirac), m5(dirac.m5), kappa5(0.5/(5.0 + m5)) { }

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

  //!NEW : added setFace(),   domainWallDslashCuda() got an extra argument  
  void DiracDomainWall::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			       const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    initSpinorConstants(in, profile);
    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, 0, commDim, profile);   

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 1320LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

// Modification for the 4D preconditioned domain wall operator
  void DiracDomainWall4DPC::Dslash4(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
                            const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    initSpinorConstants(in, profile);

    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, 0, commDim, 0, profile);   

    flops += 1320LL*(long long)in.Volume();
  }

  void DiracDomainWall4DPC::Dslash5(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
                            const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    initSpinorConstants(in, profile);

    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, 0, commDim, 1, profile);   

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 96LL*bulk + 120LL*wall;
  }

  void DiracDomainWall4DPC::Dslash5inv(cudaColorSpinorField &out, const cudaColorSpinorField &in, \
                            const QudaParity parity, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    initSpinorConstants(in, profile);

    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, k, commDim, 2, profile);   

    long long Ls = in.X(4);
    flops +=  144LL*(long long)in.Volume()*Ls + 3LL*Ls*(Ls-1LL);
  }


  //!NEW : added setFace(), domainWallDslashCuda() got an extra argument 
  void DiracDomainWall::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				   const QudaParity parity, const cudaColorSpinorField &x,
				   const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    initSpinorConstants(in, profile);
    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&out, gauge, &in, parity, dagger, &x, mass, k, commDim, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (1320LL+48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  // Modification for the 4D preconditioned domain wall operator
  void DiracDomainWall4DPC::Dslash4Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				   const QudaParity parity, const cudaColorSpinorField &x,
				   const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    initSpinorConstants(in, profile);
    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&out, gauge, &in, parity, dagger, &x, mass, k, commDim, 0, profile);

    flops += (1320LL+48LL)*(long long)in.Volume();
  }
  
  void DiracDomainWall4DPC::Dslash5Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				   const QudaParity parity, const cudaColorSpinorField &x,
				   const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    initSpinorConstants(in, profile);
    setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&out, gauge, &in, parity, dagger, &x, mass, k, commDim, 1, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (48LL)*(long long)in.Volume() + 96LL*bulk + 72LL*wall;
  }

  void DiracDomainWall::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), -kappa5);
    DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), -kappa5);
  }

  void DiracDomainWall::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracDomainWall::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
				cudaColorSpinorField &x, cudaColorSpinorField &b, 
				const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracDomainWall::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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
  void DiracDomainWallPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
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

  void DiracDomainWallPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    M(out, in);
    Mdag(out, out);
  }

  void DiracDomainWallPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
				  cudaColorSpinorField &x, cudaColorSpinorField &b, 
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

  void DiracDomainWallPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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

// Modification for the 4D preconditioned domain wall operator
  DiracDomainWall4DPC::DiracDomainWall4DPC(const DiracParam &param)
    : DiracDomainWallPC(param)
  {

  }

  DiracDomainWall4DPC::DiracDomainWall4DPC(const DiracDomainWall4DPC &dirac) 
    : DiracDomainWallPC(dirac)
  {

  }

  DiracDomainWall4DPC::~DiracDomainWall4DPC()
  {

  }

  DiracDomainWall4DPC& DiracDomainWall4DPC::operator=(const DiracDomainWall4DPC &dirac)
  {
    if (&dirac != this) {
      DiracDomainWallPC::operator=(dirac);
    }

    return *this;
  }

  // Apply the 4D even-odd preconditioned domain-wall Dirac operator
  void DiracDomainWall4DPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    double kappa2 = -kappa5*kappa5;

    bool reset1 = newTmp(&tmp1, in);

    //QUDA_MATPC_EVEN_EVEN : 1 - k D5 - k^2 D4_eo D5inv D4_oe
    //QUDA_MATPC_ODD_ODD : 1 - k D5 - k^2 D4_oe D5inv D4_eo
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash4(*tmp1, in, QUDA_EVEN_PARITY);
      Dslash5inv(out, *tmp1, QUDA_ODD_PARITY, kappa5);
      Dslash4Xpay(*tmp1, out, QUDA_ODD_PARITY, in, kappa2); 
      Dslash5Xpay(out, in, QUDA_EVEN_PARITY,*tmp1, -kappa5); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash4(*tmp1, in, QUDA_ODD_PARITY);
      Dslash5inv(out, *tmp1, QUDA_EVEN_PARITY, kappa5);
      Dslash4Xpay(*tmp1, out, QUDA_EVEN_PARITY, in, kappa2); 
      Dslash5Xpay(out, in, QUDA_ODD_PARITY, *tmp1, -kappa5); 
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracDomainWall4DPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
#ifdef MULTI_GPU
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
#else
    M(out, in);
    Mdag(out, out);
#endif
  }

  void DiracDomainWall4DPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
      cudaColorSpinorField &x, cudaColorSpinorField &b, 
      const QudaSolutionType solType) const
  {
    bool reset = newTmp(&tmp1, b);
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {  
      // we desire solution to full system
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = b_e + k D4_eo*D5inv b_o
        Dslash5inv(*tmp1, b.Odd(), QUDA_ODD_PARITY, kappa5);
        Dslash4Xpay(x.Odd(), *tmp1, QUDA_ODD_PARITY, b.Even(), kappa5);
        src = &(x.Odd());
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = b_o + k D4_oe*D5inv b_e
        Dslash5inv(*tmp1, b.Even(), QUDA_EVEN_PARITY, kappa5);
        Dslash4Xpay(x.Even(), *tmp1, QUDA_EVEN_PARITY, b.Odd(), kappa5);
        src = &(x.Even());
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }
    deleteTmp(&tmp1, reset);
  }

  void DiracDomainWall4DPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
      const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    bool reset1 = newTmp(&tmp1, x);

    // create full solution

    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // x_o = D5inv b_o - k D5inv D4_oe x_e
      Dslash4Xpay(*tmp1, x.Even(), QUDA_EVEN_PARITY, b.Odd(), -kappa5);
      Dslash5inv(x.Odd(), *tmp1, QUDA_EVEN_PARITY, kappa5);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // x_e = D5inv b_e - k D5inv D4_eo x_o
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_ODD_PARITY, b.Even(), -kappa5);
      Dslash5inv(x.Even(), *tmp1, QUDA_ODD_PARITY, kappa5);
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
    }
    deleteTmp(&tmp1, reset1);
  }

} // namespace quda

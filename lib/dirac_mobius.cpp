#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  namespace mobius {
#include <dslash_init.cuh>
  }

// Modification for the 4D preconditioned Mobius domain wall operator
  DiracMobiusDomainWallPC::DiracMobiusDomainWallPC(const DiracParam &param)
    : DiracDomainWallPC(param) { 
    memcpy(b_5, param.b_5, sizeof(double)*param.Ls);
    memcpy(c_5, param.c_5, sizeof(double)*param.Ls);
    mobius::initConstants(*param.gauge, profile);
  }

  DiracMobiusDomainWallPC::DiracMobiusDomainWallPC(const DiracMobiusDomainWallPC &dirac) 
    : DiracDomainWallPC(dirac) {
    memcpy(b_5, dirac.b_5, Ls);
    memcpy(c_5, dirac.c_5, Ls);
    mobius::initConstants(dirac.gauge, profile);
  }

  DiracMobiusDomainWallPC::~DiracMobiusDomainWallPC()
  { }

  DiracMobiusDomainWallPC& DiracMobiusDomainWallPC::operator=(const DiracMobiusDomainWallPC &dirac)
  {
    if (&dirac != this) {
      DiracDomainWallPC::operator=(dirac);
    }

    return *this;
  }

// Modification for the 4D preconditioned Mobius domain wall operator
  void DiracMobiusDomainWallPC::Dslash4(cudaColorSpinorField &out, const cudaColorSpinorField &in,
					const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    MDWFDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, 0, commDim, 0, profile);   

    flops += 1320LL*(long long)in.Volume();
  }
  
  void DiracMobiusDomainWallPC::Dslash4pre(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    MDWFDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, 0, commDim, 1, profile);   

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 72LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracMobiusDomainWallPC::Dslash5(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    
    MDWFDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, 0, commDim, 2, profile);   

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 48LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracMobiusDomainWallPC::Dslash5inv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    MDWFDslashCuda(&out, gauge, &in, parity, dagger, 0, mass, k, commDim, 3, profile);   

    long long Ls = in.X(4);
    flops += 144LL*(long long)in.Volume()*Ls + 3LL*Ls*(Ls-1LL);
  }

  // Modification for the 4D preconditioned Mobius domain wall operator
  void DiracMobiusDomainWallPC::Dslash4Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
					    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

     mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    MDWFDslashCuda(&out, gauge, &in, parity, dagger, &x, mass, k, commDim, 0, profile);
    
    flops += (1320LL+48LL)*(long long)in.Volume();
  }
  
  void DiracMobiusDomainWallPC::Dslash5Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				   const QudaParity parity, const cudaColorSpinorField &x, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    MDWFDslashCuda(&out, gauge, &in, parity, dagger, &x, mass, k, commDim, 2, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (96LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  // Apply the even-odd preconditioned mobius DWF operator
  void DiracMobiusDomainWallPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    if(dagger == QUDA_DAG_NO)
    {
      if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

      bool reset1 = newTmp(&tmp1, in);

      //QUDA_MATPC_EVEN_EVEN_ASYMMETRIC : M5 - kappa_b^2 * D4_{eo}D4pre_{oe}D5inv_{ee}D4_{eo}D4pre_{oe}
      //QUDA_MATPC_ODD_ODD_ASYMMETRIC : M5 - kappa_b^2 * D4_{oe}D4pre_{eo}D5inv_{oo}D4_{oe}D4pre_{eo}
      //Actually, Dslash5 will return M5 operation and M5 = 1 + 0.5*kappa_b/kappa_c * D5
      if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        Dslash4pre(*tmp1, in, QUDA_ODD_PARITY);
        Dslash4(out, *tmp1, QUDA_EVEN_PARITY);
        Dslash5inv(*tmp1, out, QUDA_ODD_PARITY, kappa5); //kappa5 is dummy value
        Dslash4pre(out, *tmp1, QUDA_EVEN_PARITY);
        Dslash4(*tmp1, out, QUDA_ODD_PARITY);
        Dslash5Xpay(out, in, QUDA_EVEN_PARITY, *tmp1, 1.0);
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        Dslash4pre(*tmp1, in, QUDA_EVEN_PARITY);
        Dslash4(out, *tmp1, QUDA_ODD_PARITY);
        Dslash5inv(*tmp1, out, QUDA_EVEN_PARITY, kappa5); //kappa5 is dummy value
        Dslash4pre(out, *tmp1, QUDA_ODD_PARITY);
        Dslash4(*tmp1, out, QUDA_EVEN_PARITY);
        Dslash5Xpay(out, in, QUDA_ODD_PARITY, *tmp1, 1.0);
      } else {
        errorQuda("MatPCType %d not valid for DiracMobiusDomainWallPC", matpcType);
      }

      deleteTmp(&tmp1, reset1);
    }
    else
    {
      if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

      bool reset1 = newTmp(&tmp1, in);

      //QUDA_MATPC_EVEN_EVEN : M5 - kappa_b^2 * D4_{eo}D4pre_{oe}D5inv_{ee}D4_{eo}D4pre_{oe}
      //QUDA_MATPC_ODD_ODD : M5 - kappa_b^2 * D4_{oe}D4pre_{eo}D5inv_{oo}D4_{oe}D4pre_{eo}
      //Actually, Dslash5 will return M5 operation and M5 = 1 + 0.5*kappa_b/kappa_c * D5
      if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        Dslash4(*tmp1, in, QUDA_EVEN_PARITY);
        Dslash4pre(out, *tmp1, QUDA_ODD_PARITY);
        Dslash5inv(*tmp1, out, QUDA_EVEN_PARITY, kappa5);
        Dslash4(out, *tmp1, QUDA_ODD_PARITY);
        Dslash4pre(*tmp1, out, QUDA_EVEN_PARITY);
        //Dslash5Xpay(out, in, QUDA_EVEN_PARITY, *tmp1, 1.0);
        Dslash5Xpay(out, in, QUDA_ODD_PARITY, *tmp1, 1.0);
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        Dslash4(*tmp1, in, QUDA_ODD_PARITY);
        Dslash4pre(out, *tmp1, QUDA_EVEN_PARITY);
        Dslash5inv(*tmp1, out, QUDA_ODD_PARITY, kappa5);
        Dslash4(out, *tmp1, QUDA_EVEN_PARITY);
        Dslash4pre(*tmp1, out, QUDA_ODD_PARITY);
        //Dslash5Xpay(out, in, QUDA_ODD_PARITY, *tmp1, 1.0);
        Dslash5Xpay(out, in, QUDA_EVEN_PARITY, *tmp1, 1.0);
      } else {
        errorQuda("MatPCType %d not valid for DiracMobiusDomainWallPC", matpcType);
      }
      deleteTmp(&tmp1, reset1);
    }
  }

  void DiracMobiusDomainWallPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracMobiusDomainWallPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
      cudaColorSpinorField &x, cudaColorSpinorField &b, 
      const QudaSolutionType solType) const
  {
    // prepare function in MDWF is not tested yet.
    bool reset = newTmp(&tmp1, b);
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {  
      // we desire solution to full system
      if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo*D5inv b_o
        Dslash5inv(x.Odd(), b.Odd(), QUDA_ODD_PARITY, kappa5);//kappa5 is dummy
        Dslash4pre(*tmp1, x.Odd(), QUDA_ODD_PARITY);
        Dslash4Xpay(x.Odd(), *tmp1, QUDA_ODD_PARITY, b.Even(), 1.0);
        src = &(x.Odd());
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe*D5inv b_e
        Dslash5inv(x.Even(), b.Even(), QUDA_EVEN_PARITY, kappa5);//kappa5 is dummy
        Dslash4pre(*tmp1, x.Even(), QUDA_EVEN_PARITY);
        Dslash4Xpay(x.Even(), *tmp1, QUDA_EVEN_PARITY, b.Odd(), 1.0);
        src = &(x.Even());
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracMobiusDomainWallPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }
    deleteTmp(&tmp1, reset);
  }

  void DiracMobiusDomainWallPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
      const QudaSolutionType solType) const
  {
    // reconstruct function in MDWF is not tested yet.
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    bool reset1 = newTmp(&tmp1, x);

    // create full solution
    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // psi_e = M5inv in_e - k_b M5inv D4_eo psi_o
      Dslash4pre(x.Even(), x.Odd(), QUDA_ODD_PARITY);
      Dslash4Xpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Even(), -1.0); 
      Dslash5inv(x.Even(), *tmp1, QUDA_ODD_PARITY, kappa5); //kappa5 is dummy
    } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // psi_o = M5inv in_o - k_b M5inv D4_oe psi_e
      Dslash4pre(x.Odd(), x.Even(), QUDA_EVEN_PARITY);
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Odd(), -1.0); 
      Dslash5inv(x.Odd(), *tmp1, QUDA_EVEN_PARITY, kappa5); //kappa5 is dummy
    } else {
      errorQuda("MatPCType %d not valid for DiracMobiusDomainWallPC", matpcType);
    }
    deleteTmp(&tmp1, reset1);
  }

} // namespace quda

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
    mobius::initConstants(*dirac.gauge, profile);
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
  void DiracMobiusDomainWallPC::Dslash4(ColorSpinorField &out, const ColorSpinorField &in,
					const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, commDim, 0, profile);   

    flops += 1320LL*(long long)in.Volume();
  }
  
  void DiracMobiusDomainWallPC::Dslash4pre(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, commDim, 1, profile);   

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 72LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracMobiusDomainWallPC::Dslash5(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    
    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, commDim, 2, profile);   

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 48LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracMobiusDomainWallPC::Dslash5inv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda

    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, 0, mass, 0, commDim, 3, profile);

    long long Ls = in.X(4);
    flops += 144LL*(long long)in.Volume()*Ls + 3LL*Ls*(Ls-1LL);
  }

  // Modification for the 4D preconditioned Mobius domain wall operator
  void DiracMobiusDomainWallPC::Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in,
					    const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda

    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, commDim, 0, profile);

    flops += (1320LL+48LL)*(long long)in.Volume();
  }

  void DiracMobiusDomainWallPC::Dslash4preXpay(ColorSpinorField &out, const ColorSpinorField &in,
					       const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda

    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, commDim, 1, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (72LL+48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracMobiusDomainWallPC::Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in,
				   const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda

    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, commDim, 2, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (96LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracMobiusDomainWallPC::Dslash5invXpay(ColorSpinorField &out, const ColorSpinorField &in,
					       const QudaParity parity, const ColorSpinorField &x, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    mobius::initMDWFConstants(b_5, c_5, in.X(4), m5, profile);
    mobius::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda

    MDWFDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
		   &static_cast<const cudaColorSpinorField&>(in),
		   parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
		   mass, k, commDim, 3, profile);

    long long Ls = in.X(4);
    flops +=  (144LL*Ls + 48LL)*(long long)in.Volume() + 3LL*Ls*(Ls-1LL);
  }

  // Apply the even-odd preconditioned mobius DWF operator
  //Actually, Dslash5 will return M5 operation and M5 = 1 + 0.5*kappa_b/kappa_c * D5
  void DiracMobiusDomainWallPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

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
      Dslash5invXpay(out, *tmp1, parity[1], in, 1.0);
    } else if (symmetric && dagger) {
      Dslash5inv(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash4pre(*tmp1, out, parity[0]);
      Dslash5inv(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash4preXpay(out, *tmp1, parity[1], in, 1.0);
    } else if (!symmetric && !dagger) {
      Dslash4pre(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash5Xpay(out, in, parity[1], *tmp1, 1.0);
    } else if (!symmetric && dagger) {
      Dslash4(*tmp1, in, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4(out, *tmp1, parity[1]);
      Dslash4pre(*tmp1, out, parity[1]);
      Dslash5Xpay(out, in, parity[1], *tmp1, 1.0);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracMobiusDomainWallPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracMobiusDomainWallPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
      ColorSpinorField &x, ColorSpinorField &b, 
      const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {  
      // prepare function in MDWF is not tested yet.
      bool reset = newTmp(&tmp1, b.Even());

      // we desire solution to full system
      if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo * D4pre * D5inv b_o
        Dslash5inv(x.Odd(), b.Odd(), QUDA_ODD_PARITY);
        Dslash4pre(*tmp1, x.Odd(), QUDA_ODD_PARITY);
        Dslash4Xpay(x.Odd(), *tmp1, QUDA_EVEN_PARITY, b.Even(), 1.0);
        src = &(x.Odd());
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe * D4pre * D5inv b_e
        Dslash5inv(x.Even(), b.Even(), QUDA_EVEN_PARITY);
        Dslash4pre(*tmp1, x.Even(), QUDA_EVEN_PARITY);
        Dslash4Xpay(x.Even(), *tmp1, QUDA_ODD_PARITY, b.Odd(), 1.0);
        src = &(x.Even());
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracMobiusDomainWallPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want

      deleteTmp(&tmp1, reset);
    }
  }

  void DiracMobiusDomainWallPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
      const QudaSolutionType solType) const
  {
    // reconstruct function in MDWF is not tested yet.
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    bool reset1 = newTmp(&tmp1, x.Even());

    // TODO check -1.0 factor here is correct

    // create full solution
    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // psi_o = M5inv in_o - k_b M5inv D4_oe D4pre psi_e
      Dslash4pre(x.Even(), x.Odd(), QUDA_EVEN_PARITY);
      Dslash4Xpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Even(), -1.0); 
      Dslash5inv(x.Even(), *tmp1, QUDA_ODD_PARITY);
    } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // psi_e = M5inv in_e - k_b M5inv D4_eo D4pre psi_o
      Dslash4pre(x.Odd(), x.Even(), QUDA_ODD_PARITY);
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Odd(), -1.0); 
      Dslash5inv(x.Odd(), *tmp1, QUDA_EVEN_PARITY);
    } else {
      errorQuda("MatPCType %d not valid for DiracMobiusDomainWallPC", matpcType);
    }
    deleteTmp(&tmp1, reset1);
  }

} // namespace quda

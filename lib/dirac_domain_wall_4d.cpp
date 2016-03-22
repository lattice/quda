#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  namespace domainwall4d {
#include <dslash_init.cuh>
  }

// Modification for the 4D preconditioned domain wall operator
  DiracDomainWall4DPC::DiracDomainWall4DPC(const DiracParam &param)
    : DiracDomainWallPC(param)
  {
    domainwall4d::initConstants(*param.gauge, profile);
  }

  DiracDomainWall4DPC::DiracDomainWall4DPC(const DiracDomainWall4DPC &dirac) 
    : DiracDomainWallPC(dirac)
  {
    domainwall4d::initConstants(*dirac.gauge, profile);
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

// Modification for the 4D preconditioned domain wall operator
  void DiracDomainWall4DPC::Dslash4(ColorSpinorField &out, const ColorSpinorField &in, 
                            const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    domainwall4d::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			 &static_cast<const cudaColorSpinorField&>(in),
			 parity, dagger, 0, mass, 0, 0, commDim, 0, profile);

    flops += 1320LL*(long long)in.Volume();
  }

  void DiracDomainWall4DPC::Dslash5(ColorSpinorField &out, const ColorSpinorField &in, 
                            const QudaParity parity) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    domainwall4d::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  

    domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			 &static_cast<const cudaColorSpinorField&>(in),
			 parity, dagger, 0, mass, 0, 0, commDim, 1, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 96LL*bulk + 120LL*wall;
  }

  // I think this is misnamed and should be M5inv
  void DiracDomainWall4DPC::Dslash5inv(ColorSpinorField &out, const ColorSpinorField &in,
				       const QudaParity parity, const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
 
    domainwall4d::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda  
    domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			 &static_cast<const cudaColorSpinorField&>(in),
			 parity, dagger, 0, mass, k, 0, commDim, 2, profile);

  
    long long Ls = in.X(4);
    flops +=  144LL*(long long)in.Volume()*Ls + 3LL*Ls*(Ls-1LL);
  }

  // Modification for the 4D preconditioned domain wall operator
  void DiracDomainWall4DPC::Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in,
					const QudaParity parity, const ColorSpinorField &x,
					const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    domainwall4d::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda
    domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			 &static_cast<const cudaColorSpinorField&>(in),
			 parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
			 mass, k, 0, commDim, 0, profile);

    flops += (1320LL+48LL)*(long long)in.Volume();
  }

  void DiracDomainWall4DPC::Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in,
					const QudaParity parity, const ColorSpinorField &x,
					const double &k) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    domainwall4d::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda
    domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			 &static_cast<const cudaColorSpinorField&>(in),
			 parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
			 mass, k, 0, commDim, 1, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracDomainWall4DPC::Dslash5invXpay(ColorSpinorField &out, const ColorSpinorField &in,
					   const QudaParity parity, const double &a,
					   const ColorSpinorField &x, const double &b) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    domainwall4d::setFace(face1,face2); // FIXME: temporary hack maintain C linkage for dslashCuda
    domainWallDslashCuda(&static_cast<cudaColorSpinorField&>(out), *gauge,
			 &static_cast<const cudaColorSpinorField&>(in),
			 parity, dagger, &static_cast<const cudaColorSpinorField&>(x),
			 mass, a, b, commDim, 2, profile);

    long long Ls = in.X(4);
    flops +=  (144LL*Ls + 48LL)*(long long)in.Volume() + 3LL*Ls*(Ls-1LL);
  }

  // Apply the 4D even-odd preconditioned domain-wall Dirac operator
  void DiracDomainWall4DPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    double kappa2 = kappa5*kappa5;

    bool reset1 = newTmp(&tmp1, in);

    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    bool symmetric =(matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    if (symmetric && !dagger) {
      // 1 - k^2 M5^-1 D4 M5^-1 D4
      Dslash4(*tmp1, in, parity[0]);
      Dslash5inv(out, *tmp1, parity[0], kappa5);
      Dslash4(*tmp1, out, parity[1]);
      Dslash5invXpay(out, *tmp1, parity[1], kappa5, in, -kappa2);
    } else if (symmetric && dagger) {
      // 1 - k^2 D4 M5^-1 D4 M5^-1
      Dslash5inv(*tmp1, in, parity[1], kappa5);
      Dslash4(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0], kappa5);
      Dslash4Xpay(out, *tmp1, parity[1], in, -kappa2);
    } else {
      // 1 - k D5 - k^2 D4 M5^-1 D4_oe
      Dslash4(*tmp1, in, parity[0]);
      Dslash5inv(out, *tmp1, parity[0], kappa5);
      Dslash4Xpay(*tmp1, out, parity[1], in, -kappa2);
      Dslash5Xpay(out, in, parity[1], *tmp1, -kappa5);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracDomainWall4DPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracDomainWall4DPC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				    ColorSpinorField &x, ColorSpinorField &b, 
				    const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {  // we desire solution to full system
      bool reset = newTmp(&tmp1, b.Even());

      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = M5^-1 (b_e + k D4_eo*M5^-1 b_o)
        src = &(x.Odd());
        Dslash5inv(*src, b.Odd(), QUDA_ODD_PARITY, kappa5);
        Dslash4Xpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa5);
	Dslash5inv(*src, *tmp1, QUDA_EVEN_PARITY, kappa5);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = M5^-1 (b_o + k D4_oe*M5^-1 b_e)
        src = &(x.Even());
        Dslash5inv(*src, b.Even(), QUDA_EVEN_PARITY, kappa5);
        Dslash4Xpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa5);
	Dslash5inv(*src, *tmp1, QUDA_ODD_PARITY, kappa5);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo*M5^-1 b_o
        src = &(x.Odd());
        Dslash5inv(*tmp1, b.Odd(), QUDA_ODD_PARITY, kappa5);
        Dslash4Xpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa5);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe*M5^-1 b_e
        src = &(x.Even());
        Dslash5inv(*tmp1, b.Even(), QUDA_EVEN_PARITY, kappa5);
        Dslash4Xpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa5);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want

      deleteTmp(&tmp1, reset);
    }
  }

  void DiracDomainWall4DPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
					const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    checkFullSpinor(x, b);

    bool reset1 = newTmp(&tmp1, b.Even());

    // create full solution

    if (matpcType == QUDA_MATPC_EVEN_EVEN ||
	matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // x_o = M5^-1 (b_o + k D4_oe x_e)
      Dslash4Xpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa5);
      Dslash5inv(x.Odd(), *tmp1, QUDA_ODD_PARITY, kappa5);
    } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	       matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // x_e = M5^-1 (b_e + k D4_eo x_o)
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa5);
      Dslash5inv(x.Even(), *tmp1, QUDA_EVEN_PARITY, kappa5);
    } else {
      errorQuda("MatPCType %d not valid for DiracDomainWall4DPC", matpcType);
    }

    deleteTmp(&tmp1, reset1);
  }

} // end namespace quda

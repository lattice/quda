#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>

namespace quda {

  DiracWilson::DiracWilson(const DiracParam &param) : 
    Dirac(param), face(param.gauge->X(), 4, 12, 1, param.gauge->Precision()) { }

  DiracWilson::DiracWilson(const DiracWilson &dirac) : 
    Dirac(dirac), face(dirac.face) { }

  DiracWilson::DiracWilson(const DiracParam &param, const int nDims) : 
    Dirac(param), face(param.gauge->X(), nDims, 12, 1, param.gauge->Precision(), param.Ls) { }//temporal hack (for DW and TM operators) 

  DiracWilson::~DiracWilson() { }

  DiracWilson& DiracWilson::operator=(const DiracWilson &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      face = dirac.face;
    }
    return *this;
  }

  void DiracWilson::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			   const QudaParity parity) const
  {
    initSpinorConstants(in, profile);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda

    wilsonDslashCuda(&out, gauge, &in, parity, dagger, 0, 0.0, commDim, profile);

    flops += 1320ll*in.Volume();
  }

  void DiracWilson::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			       const QudaParity parity, const cudaColorSpinorField &x,
			       const double &k) const
  {
    initSpinorConstants(in, profile);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda

    wilsonDslashCuda(&out, gauge, &in, parity, dagger, &x, k, commDim, profile);

    flops += 1368ll*in.Volume();
  }

  void DiracWilson::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, in.Odd(), -kappa);
    DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, in.Even(), -kappa);
  }

  void DiracWilson::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);
    checkFullSpinor(*tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracWilson::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			    cudaColorSpinorField &x, cudaColorSpinorField &b, 
			    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracWilson::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				const QudaSolutionType solType) const
  {
    // do nothing
  }

  /* Creates the coarse grid dirac operator
  Takes: multigrid transfer class, which knows
  about the coarse grid blocking, as well as
  having prolongate and restrict member functions
  
  Returns: Color matrices Y[0..2*dim] corresponding
  to the coarse grid operator.  The first 2*dim
  matrices correspond to the forward/backward
  hopping terms on the coarse grid.  Y[2*dim] is
  the color matrix that is diagonal on the coarse
  grid
  */

  void DiracWilson::createCoarseOp(Transfer &T, void *Y[], QudaPrecision precision) const {
	CoarseOp(T, Y, precision, gauge);
#if 0
    //First make a cpu gauge field from
    // the cuda gauge field

    int pad = 0;
    GaugeFieldParam gf_param(gauge.X(), precision, gauge.Reconstruct(), pad = 0, gauge.Geometry());
    gf_param.order = QUDA_QDP_GAUGE_ORDER;
    gf_param.fixed = gauge.GaugeFixed();
    gf_param.link_type = gauge.LinkType();
    gf_param.t_boundary = gauge.TBoundary();
    gf_param.anisotropy = gauge.Anisotropy();
    gf_param.gauge = NULL;
    gf_param.create = QUDA_NULL_FIELD_CREATE;

    cpuGaugeField g(gf_param);

  //Copy the cuda gauge field to the cpu
  gauge.saveCPUField(g, QUDA_CPU_FIELD_LOCATION);

  int ndim = g.Ndim();
  int geo_bs[QUDA_MAX_DIM];
  int spin_bs = T.Spin_bs();
  int nvec = T.nvec();
  int x[QUDA_MAX_DIM];
  int xc[QUDA_MAX_DIM];
  for(int d = 0; d < ndim; d++) {
    x[d] = g.X()[d];
    geo_bs[d] = T.Geo_bs()[d];
    xc[d] = x[d]/geo_bs[d];
  }


  void *vOrder;
  if (precision == QUDA_DOUBLE_PRECISION) {
    vOrder = (ColorSpinorFieldOrder<double> *) createOrder<double>(T.Vectors(),nvec);
  }
  else {
    vOrder = (ColorSpinorFieldOrder<float> *) createOrder<float>(T.Vectors(), nvec);
  }

  #endif 
  } 

  DiracWilsonPC::DiracWilsonPC(const DiracParam &param)
    : DiracWilson(param)
  {

  }

  DiracWilsonPC::DiracWilsonPC(const DiracWilsonPC &dirac) 
    : DiracWilson(dirac)
  {

  }

  DiracWilsonPC::~DiracWilsonPC()
  {

  }

  DiracWilsonPC& DiracWilsonPC::operator=(const DiracWilsonPC &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
    }
    return *this;
  }

  void DiracWilsonPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
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
      errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
    }

    deleteTmp(&tmp1, reset);
  }

  void DiracWilsonPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
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

  void DiracWilsonPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
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
	DslashXpay(x.Odd(), b.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
	src = &(x.Odd());
	sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
	// src = b_o + k D_oe b_e
	DslashXpay(x.Even(), b.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
	src = &(x.Even());
	sol = &(x.Odd());
      } else {
	errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
    }

  }

  void DiracWilsonPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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
      errorQuda("MatPCType %d not valid for DiracWilsonPC", matpcType);
    }
  }

} // namespace quda

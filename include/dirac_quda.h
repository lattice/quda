#ifndef _DIRAC_QUDA_H
#define _DIRAC_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <dslash_quda.h>
#include <face_quda.h>
#include <blas_quda.h>

#include <typeinfo>

namespace quda {

  // Params for Dirac operator
  class DiracParam {

  public:
    QudaDiracType type;
    double kappa;
    double mass;
    double m5; // used by domain wall only
    int Ls;    //!NEW: used by domain wall only and twisted mass  
    MatPCType matpcType;
    DagType dagger;
    cudaGaugeField *gauge;
    cudaGaugeField *fatGauge;  // used by staggered only
    cudaGaugeField *longGauge; // used by staggered only
    cudaCloverField *clover;
  
    double mu; // used by twisted mass only
    double epsilon; //2nd tm parameter (used by twisted mass only)

    ColorSpinorField *tmp1;
    ColorSpinorField *tmp2; // used by Wilson-like kernels only

    QudaVerbosity verbose;

    int commDim[QUDA_MAX_DIM]; // whether to do comms or not

  DiracParam() 
    : type(QUDA_INVALID_DIRAC), kappa(0.0), m5(0.0), matpcType(QUDA_MATPC_INVALID),
      dagger(QUDA_DAG_INVALID), gauge(0), clover(0), mu(0.0), epsilon(0.0),
      tmp1(0), tmp2(0), verbose(QUDA_SILENT)
    {

    }

    void print() {
      printfQuda("Printing DslashParam\n");
      printfQuda("type = %d\n", type);
      printfQuda("kappa = %g\n", kappa);
      printfQuda("mass = %g\n", mass);
      printfQuda("m5 = %g\n", m5);
      printfQuda("Ls = %d\n", Ls);
      printfQuda("matpcType = %d\n", matpcType);
      printfQuda("dagger = %d\n", dagger);
      printfQuda("mu = %g\n", mu);
      printfQuda("epsilon = %g\n", epsilon);
      for (int i=0; i<QUDA_MAX_DIM; i++) printfQuda("commDim[%d] = %d\n", i, commDim[i]);
    }
  };

  void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);
  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);

  // forward declarations
  class DiracMatrix;
  class DiracM;
  class DiracMdagM;
  class DiracMdag;

  // Abstract base class
  class Dirac {

    friend class DiracMatrix;
    friend class DiracM;
    friend class DiracMdagM;
    friend class DiracMdag;
    friend class DiracCoarse;

  protected:
    cudaGaugeField &gauge;
    double kappa;
    double mass;
    MatPCType matpcType;
    mutable DagType dagger; // mutable to simplify implementation of Mdag
    mutable unsigned long long flops;
    mutable ColorSpinorField *tmp1; // temporary hack
    mutable ColorSpinorField *tmp2; // temporary hack

    bool newTmp(ColorSpinorField **, const ColorSpinorField &) const;
    void deleteTmp(ColorSpinorField **, const bool &reset) const;

    QudaTune tune;
    QudaVerbosity verbose;  

    int commDim[QUDA_MAX_DIM]; // whether do comms or not

    mutable TimeProfile profile;

  public:
    Dirac(const DiracParam &param);
    Dirac(const Dirac &dirac);
    virtual ~Dirac();
    Dirac& operator=(const Dirac &dirac);

    virtual void checkParitySpinor(const ColorSpinorField &, const ColorSpinorField &) const;
    virtual void checkFullSpinor(const ColorSpinorField &, const ColorSpinorField &) const;
    void checkSpinorAlias(const ColorSpinorField &, const ColorSpinorField &) const;

    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			const QudaParity parity) const = 0;
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			    const QudaParity parity, const ColorSpinorField &x,
			    const double &k) const = 0;
    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const = 0;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const = 0;
    void Mdag(ColorSpinorField &out, const ColorSpinorField &in) const;

    // required methods to use e-o preconditioning for solving full system
    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b, 
			 const QudaSolutionType) const = 0;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const = 0;
    void setMass(double mass){ this->mass = mass;}
    // Dirac operator factory
    static Dirac* create(const DiracParam &param);

    unsigned long long Flops() const { unsigned long long rtn = flops; flops = 0; return rtn; }
    QudaVerbosity Verbose() const { return verbose; }
  };

  //Forward declaration of multigrid Transfer class
  class Transfer;

  // Full Wilson
  class DiracWilson : public Dirac {

  protected:
    FaceBuffer face; // multi-gpu communication buffers

  public:
    DiracWilson(const DiracParam &param);
    DiracWilson(const DiracWilson &dirac);
    DiracWilson(const DiracParam &param, const int nDims);//to correctly adjust face for DW and non-deg twisted mass   
  
    virtual ~DiracWilson();
    DiracWilson& operator=(const DiracWilson &dirac);

    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			const QudaParity parity) const;
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			    const QudaParity parity, const ColorSpinorField &x, const double &k) const;
    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;

    virtual void createCoarseOp(const Transfer &T, void *Y[], QudaPrecision precision) const;
    virtual void applyCoarse(ColorSpinorField &out, const ColorSpinorField &in, void *Y[], QudaPrecision precision) const; 
  };

  void CoarseOp(const Transfer &T, void *Y[], QudaPrecision precision, const cudaGaugeField &gauge);
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, void *Y[], QudaPrecision precision, double kappa);

  // Even-odd preconditioned Wilson
  class DiracWilsonPC : public DiracWilson {

  private:

  public:
    DiracWilsonPC(const DiracParam &param);
    DiracWilsonPC(const DiracWilsonPC &dirac);
    virtual ~DiracWilsonPC();
    DiracWilsonPC& operator=(const DiracWilsonPC &dirac);

    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
		 ColorSpinorField &x, ColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
		     const QudaSolutionType) const;
  };

  // Full clover
  class DiracClover : public DiracWilson {

  protected:
    cudaCloverField &clover;
    void checkParitySpinor(const ColorSpinorField &, const ColorSpinorField &) const;

  public:
    DiracClover(const DiracParam &param);
    DiracClover(const DiracClover &dirac);
    virtual ~DiracClover();
    DiracClover& operator=(const DiracClover &dirac);

    void Clover(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
			    const ColorSpinorField &x, const double &k) const;
    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Even-odd preconditioned clover
  class DiracCloverPC : public DiracClover {

  public:
    DiracCloverPC(const DiracParam &param);
    DiracCloverPC(const DiracCloverPC &dirac);
    virtual ~DiracCloverPC();
    DiracCloverPC& operator=(const DiracCloverPC &dirac);

    void CloverInv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    void Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
		const QudaParity parity) const;
    void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
		    const QudaParity parity, const ColorSpinorField &x, const double &k) const;

    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
		 ColorSpinorField &x, ColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
		     const QudaSolutionType) const;
  };



  // Full domain wall 
  class DiracDomainWall : public DiracWilson {

  protected:
    double m5;
    double kappa5;

  public:
    DiracDomainWall(const DiracParam &param);
    DiracDomainWall(const DiracDomainWall &dirac);
    virtual ~DiracDomainWall();
    DiracDomainWall& operator=(const DiracDomainWall &dirac);

    void Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
		const QudaParity parity) const;
    void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
		    const QudaParity parity, const ColorSpinorField &x, const double &k) const;

    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // 5d Even-odd preconditioned domain wall
  class DiracDomainWallPC : public DiracDomainWall {

  private:

  public:
    DiracDomainWallPC(const DiracParam &param);
    DiracDomainWallPC(const DiracDomainWallPC &dirac);
    virtual ~DiracDomainWallPC();
    DiracDomainWallPC& operator=(const DiracDomainWallPC &dirac);

    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
		 ColorSpinorField &x, ColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
		     const QudaSolutionType) const;
  };

  // Full twisted mass
  class DiracTwistedMass : public DiracWilson {

  protected:
    double mu;
    double epsilon;
    void TwistedApply(ColorSpinorField &out, const ColorSpinorField &in, 
		      const double &a, const double &b, const double &c,
		      const QudaTwistGamma5Type twistType) const;

    static int initTMFlag;
    void initConstants(const ColorSpinorField &in) const;

    // internal wrapper to twistedMassDslashCuda
    void TwistedDslash(ColorSpinorField &out, const ColorSpinorField &in, 
		       const int parity, QudaTwistDslashType type,
		       const double &a, const double &b, const double &c, const double &d) const;

    void TwistedDslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			   const int parity, const ColorSpinorField &x, QudaTwistDslashType type,
			   const double &a, const double &b, const double &c, const double &d) const;

  public:
    DiracTwistedMass(const DiracTwistedMass &dirac);
    DiracTwistedMass(const DiracParam &param, const int nDim);
    virtual ~DiracTwistedMass();
    DiracTwistedMass& operator=(const DiracTwistedMass &dirac);

    void Twist(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Even-odd preconditioned twisted mass
  class DiracTwistedMassPC : public DiracTwistedMass {

  public:
    DiracTwistedMassPC(const DiracTwistedMassPC &dirac);
    DiracTwistedMassPC(const DiracParam &param, const int nDim);

    virtual ~DiracTwistedMassPC();
    DiracTwistedMassPC& operator=(const DiracTwistedMassPC &dirac);

    void TwistInv(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			const QudaParity parity) const;
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			    const QudaParity parity, const ColorSpinorField &x, const double &k) const;
    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
		 ColorSpinorField &x, ColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
		     const QudaSolutionType) const;
  };

  // Full staggered
  class DiracStaggered : public Dirac {

  protected:
    cudaGaugeField &fatGauge;
    cudaGaugeField &longGauge;
    FaceBuffer face; // multi-gpu communication buffers

  public:
    DiracStaggered(const DiracParam &param);
    DiracStaggered(const DiracStaggered &dirac);
    virtual ~DiracStaggered();
    DiracStaggered& operator=(const DiracStaggered &dirac);

    virtual void checkParitySpinor(const ColorSpinorField &, const ColorSpinorField &) const;
  
    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			const QudaParity parity) const;
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
			    const QudaParity parity, const ColorSpinorField &x, const double &k) const;
    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Even-odd preconditioned staggered
  class DiracStaggeredPC : public DiracStaggered {

  protected:

  public:
    DiracStaggeredPC(const DiracParam &param);
    DiracStaggeredPC(const DiracStaggeredPC &dirac);
    virtual ~DiracStaggeredPC();
    DiracStaggeredPC& operator=(const DiracStaggeredPC &dirac);

    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Functor base class for applying a given Dirac matrix (M, MdagM, etc.)
  class DiracMatrix {

  protected:
    const Dirac *dirac;

  public:
  DiracMatrix(const Dirac &d) : dirac(&d) { }
  DiracMatrix(const Dirac *d) : dirac(d) { }
    virtual ~DiracMatrix() = 0;

    virtual void operator()(ColorSpinorField &out, const ColorSpinorField &in) const = 0;
    virtual void operator()(ColorSpinorField &out, const ColorSpinorField &in,
			    ColorSpinorField &tmp) const = 0;
    virtual void operator()(ColorSpinorField &out, const ColorSpinorField &in,
			    ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const = 0;

    unsigned long long flops() const { return dirac->Flops(); }

    std::string Type() const { return typeid(*dirac).name(); }

    const Dirac* Expose() { return dirac; }
  };

  inline DiracMatrix::~DiracMatrix()
  {

  }

  class DiracM : public DiracMatrix {

  public:
  DiracM(const Dirac &d) : DiracMatrix(d) { }
  DiracM(const Dirac *d) : DiracMatrix(d) { }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      dirac->M(out, in);
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->M(out, in);
      dirac->tmp1 = NULL;
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
		    ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
    {
      dirac->tmp1 = &Tmp1;
      dirac->tmp2 = &Tmp2;
      dirac->M(out, in);
      dirac->tmp2 = NULL;
      dirac->tmp1 = NULL;
    }
  };

  class DiracMdagM : public DiracMatrix {

  public:
    DiracMdagM(const Dirac &d) : DiracMatrix(d), shift(0.0) { }
    DiracMdagM(const Dirac *d) : DiracMatrix(d), shift(0.0) { }

    //! Shift term added onto operator (M^dag M + shift)
    double shift;

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      dirac->MdagM(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->MdagM(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      dirac->tmp1 = NULL;
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
		    ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
    {
      dirac->tmp1 = &Tmp1;
      dirac->tmp2 = &Tmp2;
      dirac->MdagM(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      dirac->tmp2 = NULL;
      dirac->tmp1 = NULL;
    }
  };

  class DiracMdag : public DiracMatrix {

  public:
  DiracMdag(const Dirac &d) : DiracMatrix(d) { }
  DiracMdag(const Dirac *d) : DiracMatrix(d) { }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      dirac->Mdag(out, in);
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->Mdag(out, in);
      dirac->tmp1 = NULL;
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
		    ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
    {
      dirac->tmp1 = &Tmp1;
      dirac->tmp2 = &Tmp2;
      dirac->Mdag(out, in);
      dirac->tmp2 = NULL;
      dirac->tmp1 = NULL;
    }
  };

} // namespace quda

#endif // _DIRAC_QUDA_H

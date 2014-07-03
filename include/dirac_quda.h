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
    int Ls;    //!NEW: used by domain wall and twisted mass
    double *b_5;    //!NEW: used by mobius domain wall only  
    double *c_5;    //!NEW: used by mobius domain wall only
    QudaMatPCType matpcType;
    QudaDagType dagger;
    cudaGaugeField *gauge;
    cudaGaugeField *fatGauge;  // used by staggered only
    cudaGaugeField *longGauge; // used by staggered only
    cudaCloverField *clover;
    cudaCloverField *cloverInv;
  
    double mu; // used by twisted mass only
    double epsilon; //2nd tm parameter (used by twisted mass only)

    cudaColorSpinorField *tmp1;
    cudaColorSpinorField *tmp2; // used by Wilson-like kernels only

    int commDim[QUDA_MAX_DIM]; // whether to do comms or not

  DiracParam() 
    : type(QUDA_INVALID_DIRAC), kappa(0.0), m5(0.0), matpcType(QUDA_MATPC_INVALID),
      dagger(QUDA_DAG_INVALID), gauge(0), clover(0), cloverInv(0), mu(0.0), epsilon(0.0),
      tmp1(0), tmp2(0)
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
      for (int i=0; i<Ls; i++) printfQuda("b_5[%d] = %e\t c_5[%d] = %e\n", i,b_5[i],i,c_5[i]);
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

  protected:
    cudaGaugeField &gauge;
    double kappa;
    double mass;
    QudaMatPCType matpcType;
    mutable QudaDagType dagger; // mutable to simplify implementation of Mdag
    mutable unsigned long long flops;
    mutable cudaColorSpinorField *tmp1; // temporary hack
    mutable cudaColorSpinorField *tmp2; // temporary hack

    bool newTmp(cudaColorSpinorField **, const cudaColorSpinorField &) const;
    void deleteTmp(cudaColorSpinorField **, const bool &reset) const;

    QudaTune tune;

    int commDim[QUDA_MAX_DIM]; // whether do comms or not

    mutable TimeProfile profile;

  public:
    Dirac(const DiracParam &param);
    Dirac(const Dirac &dirac);
    virtual ~Dirac();
    Dirac& operator=(const Dirac &dirac);

    virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
    virtual void checkFullSpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
    void checkSpinorAlias(const cudaColorSpinorField &, const cudaColorSpinorField &) const;

    virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			const QudaParity parity) const = 0;
    virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			    const QudaParity parity, const cudaColorSpinorField &x,
			    const double &k) const = 0;
    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
    void Mdag(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    // required methods to use e-o preconditioning for solving full system
    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const = 0;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			     const QudaSolutionType) const = 0;
    void setMass(double mass){ this->mass = mass;}
    // Dirac operator factory
    static Dirac* create(const DiracParam &param);

    unsigned long long Flops() const { unsigned long long rtn = flops; flops = 0; return rtn; }


    QudaMatPCType getMatPCType() const { return matpcType; }
    void Dagger(QudaDagType dag) { dagger = dag; }
  };

  // Full Wilson
  class DiracWilson : public Dirac {

  protected:
    FaceBuffer face1, face2; // multi-gpu communication buffers

  public:
    DiracWilson(const DiracParam &param);
    DiracWilson(const DiracWilson &dirac);
    DiracWilson(const DiracParam &param, const int nDims);//to correctly adjust face for DW and non-deg twisted mass   
  
    virtual ~DiracWilson();
    DiracWilson& operator=(const DiracWilson &dirac);

    virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			const QudaParity parity) const;
    virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Even-odd preconditioned Wilson
  class DiracWilsonPC : public DiracWilson {

  private:

  public:
    DiracWilsonPC(const DiracParam &param);
    DiracWilsonPC(const DiracWilsonPC &dirac);
    virtual ~DiracWilsonPC();
    DiracWilsonPC& operator=(const DiracWilsonPC &dirac);

    void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		 cudaColorSpinorField &x, cudaColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		     const QudaSolutionType) const;
  };

  // Full clover
  class DiracClover : public DiracWilson {

  protected:
    cudaCloverField &clover;
    void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;

  public:
    DiracClover(const DiracParam &param);
    DiracClover(const DiracClover &dirac);
    virtual ~DiracClover();
    DiracClover& operator=(const DiracClover &dirac);

    void Clover(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
    virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity,
			    const cudaColorSpinorField &x, const double &k) const;
    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Even-odd preconditioned clover
  class DiracCloverPC : public DiracClover {

  public:
    DiracCloverPC(const DiracParam &param);
    DiracCloverPC(const DiracCloverPC &dirac);
    virtual ~DiracCloverPC();
    DiracCloverPC& operator=(const DiracCloverPC &dirac);

    void CloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
    void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		const QudaParity parity) const;
    void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

    void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		 cudaColorSpinorField &x, cudaColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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

    void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		const QudaParity parity) const;
    void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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

    void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		 cudaColorSpinorField &x, cudaColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		     const QudaSolutionType) const;
  };

// 4d Even-odd preconditioned domain wall
  class DiracDomainWall4DPC : public DiracDomainWallPC {

  private:

  public:
    DiracDomainWall4DPC(const DiracParam &param);
    DiracDomainWall4DPC(const DiracDomainWall4DPC &dirac);
    virtual ~DiracDomainWall4DPC();
    DiracDomainWall4DPC& operator=(const DiracDomainWall4DPC &dirac);
    void Dslash4(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		const QudaParity parity) const;
    void Dslash5(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
    void Dslash5inv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity, const double &k) const;
    void Dslash4Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
    void Dslash5Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

    void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		 cudaColorSpinorField &x, cudaColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		     const QudaSolutionType) const;
  };


// 4d Even-odd preconditioned Mobius domain wall
  class DiracMobiusDomainWallPC : public DiracDomainWallPC {
    
  protected:
    //Mobius coefficients
    double *b_5;
    double *c_5;

  private:

  public:
    DiracMobiusDomainWallPC(const DiracParam &param);
    DiracMobiusDomainWallPC(const DiracMobiusDomainWallPC &dirac);
    virtual ~DiracMobiusDomainWallPC();
    DiracMobiusDomainWallPC& operator=(const DiracMobiusDomainWallPC &dirac);
    void Dslash4(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		const QudaParity parity) const;
    void Dslash4pre(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		const QudaParity parity) const;
    void Dslash5(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
    void Dslash5inv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity, const double &k) const;
    void Dslash4Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
    void Dslash5Xpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

    void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void Mdag(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol, cudaColorSpinorField &x, 
		 cudaColorSpinorField &b, const QudaSolutionType) const;
    void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b, const QudaSolutionType) const;
  };

  // Full twisted mass
  class DiracTwistedMass : public DiracWilson {

  protected:
    double mu;
    double epsilon;
    void twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaTwistGamma5Type twistType) const;

    static int initTMFlag;
    void initConstants(const cudaColorSpinorField &in) const;

  public:
    DiracTwistedMass(const DiracTwistedMass &dirac);
    DiracTwistedMass(const DiracParam &param, const int nDim);
    virtual ~DiracTwistedMass();
    DiracTwistedMass& operator=(const DiracTwistedMass &dirac);

    void Twist(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Even-odd preconditioned twisted mass
  class DiracTwistedMassPC : public DiracTwistedMass {

  public:
    DiracTwistedMassPC(const DiracTwistedMassPC &dirac);
    DiracTwistedMassPC(const DiracParam &param, const int nDim);

    virtual ~DiracTwistedMassPC();
    DiracTwistedMassPC& operator=(const DiracTwistedMassPC &dirac);

    void TwistInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			const QudaParity parity) const;
    virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
    void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		 cudaColorSpinorField &x, cudaColorSpinorField &b, 
		 const QudaSolutionType) const;
    void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		     const QudaSolutionType) const;
  };

  // Full twisted mass with a clover term
  class DiracTwistedClover : public DiracWilson {

  protected:
    double mu;
    double epsilon;
    cudaCloverField &clover;
    cudaCloverField &cloverInv;
    void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
    void twistedCloverApply(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
          const QudaTwistGamma5Type twistType, const int parity) const;

    static int initTMCFlag;
    void initConstants(const cudaColorSpinorField &in) const;

  public:
    DiracTwistedClover(const DiracTwistedClover &dirac);
    DiracTwistedClover(const DiracParam &param, const int nDim);
    virtual ~DiracTwistedClover();
    DiracTwistedClover& operator=(const DiracTwistedClover &dirac);

    void TwistClover(cudaColorSpinorField &out, const cudaColorSpinorField &in, const int parity) const;	//IS PARITY REQUIRED???

    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
       cudaColorSpinorField &x, cudaColorSpinorField &b, 
       const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
           const QudaSolutionType) const;
  };

  // Even-odd preconditioned twisted mass with a clover term
  class DiracTwistedCloverPC : public DiracTwistedClover {

  public:
    DiracTwistedCloverPC(const DiracTwistedCloverPC &dirac);
    DiracTwistedCloverPC(const DiracParam &param, const int nDim);

    virtual ~DiracTwistedCloverPC();
    DiracTwistedCloverPC& operator=(const DiracTwistedCloverPC &dirac);

    void TwistCloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const int parity) const;

    virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
      const QudaParity parity) const;
    virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
          const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
    void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
     cudaColorSpinorField &x, cudaColorSpinorField &b, 
     const QudaSolutionType) const;
    void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
         const QudaSolutionType) const;
  };

  // Full staggered
  class DiracStaggered : public Dirac {

  protected:
    FaceBuffer face1, face2; // multi-gpu communication buffers

  public:
    DiracStaggered(const DiracParam &param);
    DiracStaggered(const DiracStaggered &dirac);
    virtual ~DiracStaggered();
    DiracStaggered& operator=(const DiracStaggered &dirac);

    virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  
    virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			const QudaParity parity) const;
    virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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

    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Full staggered
  class DiracImprovedStaggered : public Dirac {

  protected:
    cudaGaugeField &fatGauge;
    cudaGaugeField &longGauge;
    FaceBuffer face1, face2; // multi-gpu communication buffers

  public:
    DiracImprovedStaggered(const DiracParam &param);
    DiracImprovedStaggered(const DiracImprovedStaggered &dirac);
    virtual ~DiracImprovedStaggered();
    DiracImprovedStaggered& operator=(const DiracImprovedStaggered &dirac);

    virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  
    virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			const QudaParity parity) const;
    virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			    const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // Even-odd preconditioned staggered
  class DiracImprovedStaggeredPC : public DiracImprovedStaggered {

  protected:

  public:
    DiracImprovedStaggeredPC(const DiracParam &param);
    DiracImprovedStaggeredPC(const DiracImprovedStaggeredPC &dirac);
    virtual ~DiracImprovedStaggeredPC();
    DiracImprovedStaggeredPC& operator=(const DiracImprovedStaggeredPC &dirac);

    virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
    virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			 cudaColorSpinorField &x, cudaColorSpinorField &b, 
			 const QudaSolutionType) const;
    virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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

    virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
    virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in,
			    cudaColorSpinorField &tmp) const = 0;
    virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in,
			    cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const = 0;

    unsigned long long flops() const { return dirac->Flops(); }


    QudaMatPCType getMatPCType() const { return dirac->getMatPCType(); }

    std::string Type() const { return typeid(*dirac).name(); }
  };

  inline DiracMatrix::~DiracMatrix()
  {

  }

  class DiracM : public DiracMatrix {

  public:
  DiracM(const Dirac &d) : DiracMatrix(d) { }
  DiracM(const Dirac *d) : DiracMatrix(d) { }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
    {
      dirac->M(out, in);
    }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->M(out, in);
      dirac->tmp1 = NULL;
    }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
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

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
    {
      dirac->MdagM(out, in);
      if (shift != 0.0) axpyCuda(shift, const_cast<cudaColorSpinorField&>(in), out);
    }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->MdagM(out, in);
      if (shift != 0.0) axpyCuda(shift, const_cast<cudaColorSpinorField&>(in), out);
      dirac->tmp1 = NULL;
    }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
    {
      dirac->tmp1 = &Tmp1;
      dirac->tmp2 = &Tmp2;
      dirac->MdagM(out, in);
      if (shift != 0.0) axpyCuda(shift, const_cast<cudaColorSpinorField&>(in), out);
      dirac->tmp2 = NULL;
      dirac->tmp1 = NULL;
    }
  };

  class DiracMdag : public DiracMatrix {

  public:
  DiracMdag(const Dirac &d) : DiracMatrix(d) { }
  DiracMdag(const Dirac *d) : DiracMatrix(d) { }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
    {
      dirac->Mdag(out, in);
    }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->Mdag(out, in);
      dirac->tmp1 = NULL;
    }

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
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

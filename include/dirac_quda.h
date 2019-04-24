#ifndef _DIRAC_QUDA_H
#define _DIRAC_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <dslash_quda.h>
#include <blas_quda.h>

#include <typeinfo>

namespace quda {

  class Transfer;
  class Dirac;

  // Params for Dirac operator
  class DiracParam {

  public:
    QudaDiracType type;
    double kappa;
    double mass;
    double m5; // used by domain wall only
    int Ls;    // used by domain wall and twisted mass
    Complex b_5[QUDA_MAX_DWF_LS]; // used by mobius domain wall only
    Complex c_5[QUDA_MAX_DWF_LS]; // used by mobius domain wall only
    QudaMatPCType matpcType;
    QudaDagType dagger;
    cudaGaugeField *gauge;
    cudaGaugeField *fatGauge;  // used by staggered only
    cudaGaugeField *longGauge; // used by staggered only
    int laplace3D;
    cudaCloverField *clover;
  
    double mu; // used by twisted mass only
    double mu_factor; // used by multigrid only
    double epsilon; //2nd tm parameter (used by twisted mass only)

    ColorSpinorField *tmp1;
    ColorSpinorField *tmp2; // used by Wilson-like kernels only

    int commDim[QUDA_MAX_DIM]; // whether to do comms or not

    QudaPrecision halo_precision; // only does something for DiracCoarse at present

    // for multigrid only
    Transfer *transfer; 
    Dirac *dirac;

  DiracParam() 
    : type(QUDA_INVALID_DIRAC), kappa(0.0), m5(0.0), matpcType(QUDA_MATPC_INVALID),
      dagger(QUDA_DAG_INVALID), gauge(0), clover(0), mu(0.0), mu_factor(0.0), epsilon(0.0),
      tmp1(0), tmp2(0), halo_precision(QUDA_INVALID_PRECISION)
    {
      for (int i=0; i<QUDA_MAX_DIM; i++) commDim[i] = 1;
    }

    void print() {
      printfQuda("Printing DslashParam\n");
      printfQuda("type = %d\n", type);
      printfQuda("kappa = %g\n", kappa);
      printfQuda("mass = %g\n", mass);
      printfQuda("laplace3D = %d\n", laplace3D);
      printfQuda("m5 = %g\n", m5);
      printfQuda("Ls = %d\n", Ls);
      printfQuda("matpcType = %d\n", matpcType);
      printfQuda("dagger = %d\n", dagger);
      printfQuda("mu = %g\n", mu);
      printfQuda("epsilon = %g\n", epsilon);
      printfQuda("halo_precision = %d\n", halo_precision);
      for (int i=0; i<QUDA_MAX_DIM; i++) printfQuda("commDim[%d] = %d\n", i, commDim[i]);
      for (int i = 0; i < Ls; i++)
        printfQuda(
            "b_5[%d] = %e %e \t c_5[%d] = %e %e\n", i, b_5[i].real(), b_5[i].imag(), i, c_5[i].real(), c_5[i].imag());
    }

  };

  void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);
  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);

  // forward declarations
  class DiracMatrix;
  class DiracM;
  class DiracMdagM;
  class DiracMMdag;
  class DiracMdag;
  //Forward declaration of multigrid Transfer class
  class Transfer;

  // Abstract base class
  class Dirac : public Object {

    friend class DiracMatrix;
    friend class DiracM;
    friend class DiracMdagM;
    friend class DiracMMdag;
    friend class DiracMdag;

  protected:
    cudaGaugeField *gauge;
    double kappa;
    double mass;
    int laplace3D;
    QudaMatPCType matpcType;
    mutable QudaDagType dagger; // mutable to simplify implementation of Mdag
    mutable unsigned long long flops;
    mutable ColorSpinorField *tmp1; // temporary hack
    mutable ColorSpinorField *tmp2; // temporary hack
    QudaDiracType type; 
    mutable QudaPrecision halo_precision; // only does something for DiracCoarse at present

    bool newTmp(ColorSpinorField **, const ColorSpinorField &) const;
    void deleteTmp(ColorSpinorField **, const bool &reset) const;

    mutable int commDim[QUDA_MAX_DIM]; // whether do comms or not

    mutable TimeProfile profile;

  public:
    Dirac(const DiracParam &param);
    Dirac(const Dirac &dirac);
    virtual ~Dirac();
    Dirac& operator=(const Dirac &dirac);

    /**
       @brief Enable / disable communications for the Dirac operator
       @param[in] commDim_ Array of booleans which determines whether
       communications are enabled
     */
    void setCommDim(const int commDim_[QUDA_MAX_DIM]) const {
      for (int i=0; i<QUDA_MAX_DIM; i++) { commDim[i] = commDim_[i]; }
    }

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
    void MMdag(ColorSpinorField &out, const ColorSpinorField &in) const;

    // required methods to use e-o preconditioning for solving full system
    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b,
			 const QudaSolutionType) const = 0;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const = 0;
    void setMass(double mass){ this->mass = mass;}
    // Dirac operator factory
    static Dirac* create(const DiracParam &param);

    double Kappa() const { return kappa; }
    virtual double Mu() const { return 0.; }
    virtual double MuFactor() const { return 0.; }

    unsigned long long Flops() const { unsigned long long rtn = flops; flops = 0; return rtn; }


    QudaMatPCType getMatPCType() const { return matpcType; }
    int getStencilSteps() const;
    void Dagger(QudaDagType dag) const { dagger = dag; }
    void flipDagger() const { dagger = (dagger == QUDA_DAG_YES) ? QUDA_DAG_NO : QUDA_DAG_YES; }

    /**
     * @brief Create the coarse operator (virtual parent)
     *
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param T[in] Transfer operator defining the coarse grid
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator (gets explicitly built into clover, hard coded to zero for non-staggered ops)
     * @param mu TM mu parameter for the coarse operator
     * @param mu_factor multiplicative factor for the mu parameter
     */
    virtual void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
				double kappa, double mass=0., double mu=0., double mu_factor=0.) const
    {errorQuda("Not implemented");}

    QudaPrecision HaloPrecision() const { return halo_precision; }
    void setHaloPrecision(QudaPrecision halo_precision_) const { halo_precision = halo_precision_; }
  };

  // Full Wilson
  class DiracWilson : public Dirac {

  protected:
    void initConstants();

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

    /**
     * @brief Create the coarse Wilson operator
     *
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param T[in] Transfer operator defining the coarse grid
     * @param mass Mass parameter for the coarse operator (hard coded to 0 when CoarseOp is called)
     * @param kappa Kappa parameter for the coarse operator
     */
    virtual void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
				double kappa, double mass=0.,double mu=0., double mu_factor=0.) const;
  };

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
    void initConstants();

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

    /**
     * @brief Create the coarse clover operator
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator (hard coded to 0 when CoarseOp is called)
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass=0., double mu=0., double mu_factor=0.) const;
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

    /**
     * @brief Create the coarse even-odd preconditioned clover
     * operator.  Unlike the Wilson operator, the coarsening of the
     * preconditioned clover operator differs from that of the
     * unpreconditioned clover operator, so we need to specialize it.
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator (set to zero)
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass=0., double mu=0., double mu_factor=0.) const;
  };


  // Full domain wall
  class DiracDomainWall : public DiracWilson {

  protected:
    double m5;
    double kappa5;
    int Ls; // length of the fifth dimension
    void checkDWF(const ColorSpinorField &out, const ColorSpinorField &in) const;

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

  // Full domain wall, but with 4-d parity ordered fields
  class DiracDomainWall4D : public DiracDomainWall
  {

private:
public:
    DiracDomainWall4D(const DiracParam &param);
    DiracDomainWall4D(const DiracDomainWall4D &dirac);
    virtual ~DiracDomainWall4D();
    DiracDomainWall4D &operator=(const DiracDomainWall4D &dirac);

    void Dslash4(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    void Dslash5(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    void Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in,
		     const QudaParity parity, const ColorSpinorField &x, const double &k) const;
    void Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in,
		     const QudaParity parity, const ColorSpinorField &x, const double &k) const;

    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x, ColorSpinorField &b,
        const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType) const;
  };

  // 4d Even-odd preconditioned domain wall
  class DiracDomainWall4DPC : public DiracDomainWall4D
  {

private:
public:
    DiracDomainWall4DPC(const DiracParam &param);
    DiracDomainWall4DPC(const DiracDomainWall4DPC &dirac);
    virtual ~DiracDomainWall4DPC();
    DiracDomainWall4DPC &operator=(const DiracDomainWall4DPC &dirac);

    void Dslash5inv(
        ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity, const double &kappa5) const;
    void Dslash5invXpay(ColorSpinorField &out, const ColorSpinorField &in,
			const QudaParity parity, const double &kappa5, const ColorSpinorField &x, const double &k) const;

    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
		 ColorSpinorField &x, ColorSpinorField &b,
		 const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
		     const QudaSolutionType) const;
  };

  // Full Mobius
  class DiracMobius : public DiracDomainWall {

  protected:
    //Mobius coefficients
      Complex b_5[QUDA_MAX_DWF_LS];
      Complex c_5[QUDA_MAX_DWF_LS];

      /**
         Whether we are using classical Mobius with constant real-valued
         b and c coefficients, or zMobius with complex-valued variable
         coefficients
      */
      bool zMobius;

  public:
    DiracMobius(const DiracParam &param);
    DiracMobius(const DiracMobius &dirac);
    virtual ~DiracMobius();
    DiracMobius& operator=(const DiracMobius &dirac);

    void Dslash4(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    void Dslash4pre(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    void Dslash5(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;

    void Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in,
		     const QudaParity parity, const ColorSpinorField &x, const double &k) const;
    void Dslash4preXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
			const ColorSpinorField &x, const double &k) const;
    void Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in,
		     const QudaParity parity, const ColorSpinorField &x, const double &k) const;

    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b,
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  // 4d Even-odd preconditioned Mobius domain wall
  class DiracMobiusPC : public DiracMobius {

  protected:

  private:

  public:
    DiracMobiusPC(const DiracParam &param);
    DiracMobiusPC(const DiracMobiusPC &dirac);
    virtual ~DiracMobiusPC();
    DiracMobiusPC& operator=(const DiracMobiusPC &dirac);

    void Dslash5inv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;

    void Dslash5invXpay(ColorSpinorField &out, const ColorSpinorField &in,
			const QudaParity parity, const ColorSpinorField &x, const double &k) const;


    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;
    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol, ColorSpinorField &x, 
		 ColorSpinorField &b, const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType) const;
  };

  // Full twisted mass
  class DiracTwistedMass : public DiracWilson {

  protected:
      mutable double mu;
      mutable double epsilon;
      void twistedApply(ColorSpinorField &out, const ColorSpinorField &in, const QudaTwistGamma5Type twistType) const;
      virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity) const;
      virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, QudaParity parity,
          const ColorSpinorField &x, const double &k) const;

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

    double Mu() const { return mu; }

   /**
     * @brief Create the coarse twisted-mass operator
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator (gets explicitly built into clover, hard coded to zero for non-staggered ops)
     * @param mu TM mu parameter for the coarse operator
     * @param mu_factor multiplicative factor for the mu parameter
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass, double mu, double mu_factor=0.) const;
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
   /**
     * @brief Create the coarse even-odd preconditioned twisted-mass
     *        operator
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator (gets explicitly built into clover, hard coded to zero for non-staggered ops)
     * @param mu TM mu parameter for the coarse operator
     * @param mu_factor multiplicative factor for the mu parameter
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass, double mu, double mu_factor=0.) const;
  };

  // Full twisted mass with a clover term
  class DiracTwistedClover : public DiracWilson {

  protected:
    double mu;
    double epsilon;
    cudaCloverField &clover;
    void checkParitySpinor(const ColorSpinorField &, const ColorSpinorField &) const;
    void twistedCloverApply(ColorSpinorField &out, const ColorSpinorField &in, 
          const QudaTwistGamma5Type twistType, const int parity) const;

  public:
    DiracTwistedClover(const DiracTwistedClover &dirac);
    DiracTwistedClover(const DiracParam &param, const int nDim);
    virtual ~DiracTwistedClover();
    DiracTwistedClover& operator=(const DiracTwistedClover &dirac);

    void TwistClover(ColorSpinorField &out, const ColorSpinorField &in, const int parity) const;

    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
        const ColorSpinorField &x, const double &k) const;

    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
       ColorSpinorField &x, ColorSpinorField &b,
       const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
           const QudaSolutionType) const;

    double Mu() const { return mu; }

   /**
     * @brief Create the coarse twisted-clover operator
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator (gets explicitly built into clover, hard coded to zero for non-staggered ops)
     * @param mu TM mu parameter for the coarse operator
     * @param mu_factor multiplicative factor for the mu parameter
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass, double mu, double mu_factor=0.) const;
  };

  // Even-odd preconditioned twisted mass with a clover term
  class DiracTwistedCloverPC : public DiracTwistedClover {

    mutable bool reverse; /** swap the order of the derivative D and the diagonal inverse A^{-1} */

public:
    DiracTwistedCloverPC(const DiracTwistedCloverPC &dirac);
    DiracTwistedCloverPC(const DiracParam &param, const int nDim);

    virtual ~DiracTwistedCloverPC();
    DiracTwistedCloverPC& operator=(const DiracTwistedCloverPC &dirac);

    void TwistCloverInv(ColorSpinorField &out, const ColorSpinorField &in, const int parity) const;

    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
        const ColorSpinorField &x, const double &k) const;
    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
     ColorSpinorField &x, ColorSpinorField &b,
     const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
         const QudaSolutionType) const;

    /**
     * @brief Create the coarse even-odd preconditioned twisted-clover
     * operator.  Unlike the Wilson operator, the coarsening of the
     * preconditioned clover operator differs from that of the
     * unpreconditioned clover operator, so we need to specialize it.
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator (gets explicitly built into clover, hard coded to zero for non-staggered ops)
     * @param mu TM mu parameter for the coarse operator
     * @param mu_factor multiplicative factor for the mu parameter
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass, double mu, double mu_factor=0.) const;
  };

  // Full staggered
  class DiracStaggered : public Dirac {

  protected:

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

    /**
     * @brief Create the coarse staggered operator.  Unlike the Wilson operator,
     *        we assume a mass normalization, not a kappa normalization. Thus kappa
     *        gets ignored. 
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator (ignored, set to 1.0)
     * @param mass Mass parameter for the coarse operator (gets explicitly built into clover)
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
      double kappa, double mass, double mu=0., double mu_factor=0.) const;
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

  // Full staggered
  class DiracImprovedStaggered : public Dirac {

  protected:
    cudaGaugeField &fatGauge;
    cudaGaugeField &longGauge;

  public:
    DiracImprovedStaggered(const DiracParam &param);
    DiracImprovedStaggered(const DiracImprovedStaggered &dirac);
    virtual ~DiracImprovedStaggered();
    DiracImprovedStaggered& operator=(const DiracImprovedStaggered &dirac);

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
  class DiracImprovedStaggeredPC : public DiracImprovedStaggered {

  protected:

  public:
    DiracImprovedStaggeredPC(const DiracParam &param);
    DiracImprovedStaggeredPC(const DiracImprovedStaggeredPC &dirac);
    virtual ~DiracImprovedStaggeredPC();
    DiracImprovedStaggeredPC& operator=(const DiracImprovedStaggeredPC &dirac);

    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			 ColorSpinorField &x, ColorSpinorField &b,
			 const QudaSolutionType) const;
    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
			     const QudaSolutionType) const;
  };

  /**
     This class serves as a front-end to the coarse Dslash operator,
     similar to the other dslash operators.
   */
  class DiracCoarse : public Dirac {

  protected:
    double mu;
    double mu_factor;
    const Transfer *transfer; /** restrictor / prolongator defined here */
    const Dirac *dirac; /** Parent Dirac operator */

    mutable cpuGaugeField *Y_h; /** CPU copy of the coarse link field */
    mutable cpuGaugeField *X_h; /** CPU copy of the coarse clover term */
    mutable cpuGaugeField *Xinv_h; /** CPU copy of the inverse coarse clover term */
    mutable cpuGaugeField *Yhat_h; /** CPU copy of the preconditioned coarse link field */

    mutable cudaGaugeField *Y_d; /** GPU copy of the coarse link field */
    mutable cudaGaugeField *X_d; /** GPU copy of the coarse clover term */
    mutable cudaGaugeField *Xinv_d; /** GPU copy of inverse coarse clover term */
    mutable cudaGaugeField *Yhat_d; /** GPU copy of the preconditioned coarse link field */

    /**
       @brief Initialize the coarse gauge fields.  Location is
       determined by gpu_setup variable.
    */
    void initializeCoarse();

    /**
       @brief Create the CPU or GPU coarse gauge fields on demand
       (requires that the fields have been created in the other memory
       space)
    */
    void initializeLazy(QudaFieldLocation location) const;

    mutable bool enable_gpu; /** Whether the GPU links have been constructed */
    mutable bool enable_cpu; /** Whether the CPU links have been constructed */
    const bool gpu_setup; /** Where to do the coarse-operator construction*/
    mutable bool init_gpu; /** Whether this instance did the GPU allocation or not */
    mutable bool init_cpu; /** Whether this instance did the CPU allocation or not */
    const bool mapped; /** Whether we allocate Y and X GPU fields in mapped memory or not */

    /**
       @brief Allocate the Y and X fields
       @param[in] gpu Whether to allocate on gpu (true) or cpu (false)
       @param[in] mapped whether to put gpu allocations into mapped memory
     */
    void createY(bool gpu = true, bool mapped = false) const;

    /**
       @brief Allocate the Yhat and Xinv fields
       @param[in] gpu Whether to allocate on gpu (true) or cpu (false)
     */
    void createYhat(bool gpu = true) const;

  public:
    double Mu() const { return mu; }
    double MuFactor() const { return mu_factor; }

    /**
       @param[in] param Parameters defining this operator
       @param[in] gpu_setup Whether to do the setup on GPU or CPU
       @param[in] mapped Set to true to put Y and X fields in mapped memory
     */
    DiracCoarse(const DiracParam &param, bool gpu_setup=true, bool mapped=false);

    /**
       @param[in] param Parameters defining this operator
       @param[in] Y_h CPU coarse link field
       @param[in] X_h CPU coarse clover field
       @param[in] Xinv_h CPU coarse inverse clover field
       @param[in] Yhat_h CPU coarse preconditioned link field
       @param[in] Y_d GPU coarse link field
       @param[in] X_d GPU coarse clover field
       @param[in] Xinv_d GPU coarse inverse clover field
       @param[in] Yhat_d GPU coarse preconditioned link field
     */
    DiracCoarse(const DiracParam &param,
		cpuGaugeField *Y_h, cpuGaugeField *X_h, cpuGaugeField *Xinv_h, cpuGaugeField *Yhat_h,
		cudaGaugeField *Y_d=0, cudaGaugeField *X_d=0, cudaGaugeField *Xinv_d=0, cudaGaugeField *Yhat_d=0);

    /**
       @param[in] dirac Another operator instance to clone from (shallow copy)
       @param[in] param Parameters defining this operator
     */
    DiracCoarse(const DiracCoarse &dirac, const DiracParam &param);
    virtual ~DiracCoarse();

    /**
       @brief Apply the coarse clover operator
       @param[out] out Output field
       @param[in] in Input field
       @param[paraity] parity Parity which we are applying the operator to
     */
    void Clover(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;

    /**
       @brief Apply the inverse coarse clover operator
       @param[out] out Output field
       @param[in] in Input field
       @param[paraity] parity Parity which we are applying the operator to
     */
    void CloverInv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;

    /**
       @brief Apply DslashXpay out = (D * in)
       @param[out] out Output field
       @param[in] in Input field
       @param[paraity] parity Parity which we are applying the operator to
     */
    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in,
			const QudaParity parity) const;

    /**
       @brief Apply DslashXpay out = (D * in + A * x)
       @param[out] out Output field
       @param[in] in Input field
       @param[paraity] parity Parity which we are applying the operator to
     */
    virtual void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
			    const ColorSpinorField &x, const double &k) const;

    /**
       @brief Apply the full operator
       @param[out] out output vector, out = M * in
       @param[in] in input vector
     */
    virtual void M(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    virtual void prepare(ColorSpinorField* &src, ColorSpinorField* &sol, ColorSpinorField &x, ColorSpinorField &b,
			 const QudaSolutionType) const;

    virtual void reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType) const;

    /**
     * @brief Create the coarse operator from this coarse operator
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter (assumed to be zero, staggered mass gets built into clover)
     * @param mu TM mu parameter for the coarse operator
     * @param mu_factor multiplicative factor for the mu parameter
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass, double mu, double mu_factor=0.) const;


    /**
     * @brief Create the precondtioned coarse operator
     *
     * @param Yhat[out] Preconditioned coarse link field
     * @param Xinv[out] Coarse clover inversefield
     * @param Y[in] Coarse link field
     * @param X[in] Coarse clover inverse field
     */
    void createPreconditionedCoarseOp(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X);

  };

  /**
     Even-odd preconditioned variant of coarse Dslash operator
  */
  class DiracCoarsePC : public DiracCoarse {

  public:
    /**
       @param[in] param Parameters defining this operator
       @param[in] gpu_setup Whether to do the setup on GPU or CPU
     */
    DiracCoarsePC(const DiracParam &param, bool gpu_setup=true);

    /**
       @param[in] dirac Another operator instance to clone from (shallow copy)
       @param[in] param Parameters defining this operator
     */
    DiracCoarsePC(const DiracCoarse &dirac, const DiracParam &param);

    virtual ~DiracCoarsePC();

    void Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
    void DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
		    const ColorSpinorField &x, const double &k) const;
    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;
    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol, ColorSpinorField &x, ColorSpinorField &b,
		 const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType) const;

    /**
     * @brief Create the coarse even-odd preconditioned coarse
     * operator.  Unlike the Wilson operator, the coarsening of the
     * preconditioned coarse operator differs from that of the
     * unpreconditioned coarse operator, so we need to specialize it.
     *
     * @param T[in] Transfer operator defining the coarse grid
     * @param Y[out] Coarse link field
     * @param X[out] Coarse clover field
     * @param kappa Kappa parameter for the coarse operator
     * @param mass Mass parameter for the coarse operator, assumed to be zero
     * @param mu TM mu parameter for the coarse operator
     * @param mu_factor multiplicative factor for the mu parameter
     */
    void createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
			double kappa, double mass, double mu, double mu_factor=0.) const;
  };


  /**
     @brief Full Gauge Laplace operator.  Although not a Dirac
     operator per se, it's a linear operator so it's conventient to
     put in the Dirac operator abstraction.
  */
  class GaugeLaplace : public Dirac {

  public:
    GaugeLaplace(const DiracParam &param);
    GaugeLaplace(const GaugeLaplace &laplace);

    virtual ~GaugeLaplace();
    GaugeLaplace& operator=(const GaugeLaplace &laplace);

    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
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

  /**
     @brief Even-odd preconditioned Gauge Laplace operator
  */
  class GaugeLaplacePC : public GaugeLaplace {

  public:
    GaugeLaplacePC(const DiracParam &param);
    GaugeLaplacePC(const GaugeLaplacePC &laplace);
    virtual ~GaugeLaplacePC();
    GaugeLaplacePC& operator=(const GaugeLaplacePC &laplace);

    void M(ColorSpinorField &out, const ColorSpinorField &in) const;
    void MdagM(ColorSpinorField &out, const ColorSpinorField &in) const;

    void prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
		 ColorSpinorField &x, ColorSpinorField &b,
		 const QudaSolutionType) const;
    void reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType) const;
  };

  /**
     @brief Full Covariant Derivative operator.  Although not a Dirac
     operator per se, it's a linear operator so it's conventient to
     put in the Dirac operator abstraction.
  */
  class GaugeCovDev : public Dirac {

  public:
    GaugeCovDev(const DiracParam &param);
    GaugeCovDev(const GaugeCovDev &covDev);

    virtual ~GaugeCovDev();
    GaugeCovDev& operator=(const GaugeCovDev &covDev);

    virtual void DslashCD(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity, const int mu) const;
    virtual void MCD(ColorSpinorField &out, const ColorSpinorField &in, const int mu) const;
    virtual void MdagMCD(ColorSpinorField &out, const ColorSpinorField &in, const int mu) const;


    virtual void Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const;
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


  // Functor base class for applying a given Dirac matrix (M, MdagM, etc.)
  class DiracMatrix {

  protected:
    const Dirac *dirac;

  public:
    DiracMatrix(const Dirac &d) : dirac(&d), shift(0.0) { }
    DiracMatrix(const Dirac *d) : dirac(d), shift(0.0) { }
    DiracMatrix(const DiracMatrix &mat) : dirac(mat.dirac), shift(mat.shift) { }
    DiracMatrix(const DiracMatrix *mat) : dirac(mat->dirac), shift(mat->shift) { }
    virtual ~DiracMatrix() { }

    virtual void operator()(ColorSpinorField &out, const ColorSpinorField &in) const = 0;
    virtual void operator()(ColorSpinorField &out, const ColorSpinorField &in,
			    ColorSpinorField &tmp) const = 0;
    virtual void operator()(ColorSpinorField &out, const ColorSpinorField &in,
			    ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const = 0;


    unsigned long long flops() const { return dirac->Flops(); }


    QudaMatPCType getMatPCType() const { return dirac->getMatPCType(); }
    
    virtual int getStencilSteps() const = 0; 

    std::string Type() const { return typeid(*dirac).name(); }
    
    bool isStaggered() const {
      return (Type() == typeid(DiracStaggeredPC).name() ||
	      Type() == typeid(DiracStaggered).name()   ||
	      Type() == typeid(DiracImprovedStaggeredPC).name() ||
	      Type() == typeid(DiracImprovedStaggered).name()) ? true : false;
    }
    
    const Dirac* Expose() { return dirac; }

    //! Shift term added onto operator (M/M^dag M/M M^dag + shift)
    double shift;
  };

  class DiracM : public DiracMatrix {

  public:
  DiracM(const Dirac &d) : DiracMatrix(d) { }
  DiracM(const Dirac *d) : DiracMatrix(d) { }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      dirac->M(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
    {
      bool reset1 = false;
      if (!dirac->tmp1) { dirac->tmp1 = &tmp; reset1 = true; }
      dirac->M(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      if (reset1) { dirac->tmp1 = NULL; reset1 = false; }
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
			   ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
    {
      bool reset1 = false;
      bool reset2 = false;
      if (!dirac->tmp1) { dirac->tmp1 = &Tmp1; reset1 = true; }
      if (!dirac->tmp2) { dirac->tmp2 = &Tmp2; reset2 = true; }
      dirac->M(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      if (reset2) { dirac->tmp2 = NULL; reset2 = false; }
      if (reset1) { dirac->tmp1 = NULL; reset1 = false; }
    }

    int getStencilSteps() const
    {
      return dirac->getStencilSteps(); 
    }
  };

  class DiracMdagM : public DiracMatrix {

  public:
    DiracMdagM(const Dirac &d) : DiracMatrix(d) { }
    DiracMdagM(const Dirac *d) : DiracMatrix(d) { }

    

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
 
    int getStencilSteps() const
    {
      return 2*dirac->getStencilSteps(); // 2 for M and M dagger
    }
  };


  class DiracMMdag : public DiracMatrix {

  public:
    DiracMMdag(const Dirac &d) : DiracMatrix(d) { }
    DiracMMdag(const Dirac *d) : DiracMatrix(d) { }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      dirac->MMdag(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->MMdag(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      dirac->tmp1 = NULL;
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
			   ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
    {
      dirac->tmp1 = &Tmp1;
      dirac->tmp2 = &Tmp2;
      dirac->MMdag(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      dirac->tmp2 = NULL;
      dirac->tmp1 = NULL;
    }

    int getStencilSteps() const
    {
      return 2*dirac->getStencilSteps(); // 2 for M and M dagger
    }
  };

  class DiracMdag : public DiracMatrix {

  public:
  DiracMdag(const Dirac &d) : DiracMatrix(d) { }
  DiracMdag(const Dirac *d) : DiracMatrix(d) { }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      dirac->Mdag(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
    {
      dirac->tmp1 = &tmp;
      dirac->Mdag(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      dirac->tmp1 = NULL;
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
		    ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
    {
      dirac->tmp1 = &Tmp1;
      dirac->tmp2 = &Tmp2;
      dirac->Mdag(out, in);
      if (shift != 0.0) blas::axpy(shift, const_cast<ColorSpinorField&>(in), out);
      dirac->tmp2 = NULL;
      dirac->tmp1 = NULL;
    }

    int getStencilSteps() const
    {
      return dirac->getStencilSteps(); 
    }
  };

  class DiracDagger : public DiracMatrix {

  protected:
    const DiracMatrix &mat;

  public:
  DiracDagger(const DiracMatrix &mat) : DiracMatrix(mat), mat(mat) { }
  DiracDagger(const DiracMatrix *mat) : DiracMatrix(mat), mat(*mat) { }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      dirac->flipDagger();
      mat(out, in);
      dirac->flipDagger();
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &tmp) const
    {
      dirac->flipDagger();
      mat(out, in, tmp);
      dirac->flipDagger();
    }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
                    ColorSpinorField &Tmp1, ColorSpinorField &Tmp2) const
    {
      dirac->flipDagger();
      mat(out, in, Tmp1, Tmp2);
      dirac->flipDagger();
    }

    int getStencilSteps() const
    {
      return mat.getStencilSteps(); 
    }
  };

} // namespace quda

#endif // _DIRAC_QUDA_H

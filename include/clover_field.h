#ifndef _CLOVER_QUDA_H
#define _CLOVER_QUDA_H

#include <quda_internal.h>
#include <lattice_field.h>

namespace quda {

  struct CloverFieldParam : public LatticeFieldParam {
    bool direct; // whether to create the direct clover 
    bool inverse; // whether to create the inverse clover
    void *clover;
    void *norm;
    void *cloverInv;
    void *invNorm;

//for twisted mass only:
    bool twisted; // whether to create twisted mass clover
    double mu2;

    QudaCloverFieldOrder order;
    QudaFieldCreate create;
    void setPrecision(QudaPrecision precision) {
      this->precision = precision;
      order = (precision == QUDA_DOUBLE_PRECISION) ? 
	QUDA_FLOAT2_CLOVER_ORDER : QUDA_FLOAT4_CLOVER_ORDER;
    }
  };

  std::ostream& operator<<(std::ostream& output, const CloverFieldParam& param);

  class CloverField : public LatticeField {

  protected:
    size_t bytes; // bytes allocated per clover full field 
    size_t norm_bytes; // sizeof each norm full field
    int length;
    int real_length;
    int nColor;
    int nSpin;

    void *clover;
    void *norm;
    void *cloverInv;
    void *invNorm;
//new!
    bool twisted; 
    double mu2;

    QudaCloverFieldOrder order;
    QudaFieldCreate create;

    double *trlog;

  public:
    CloverField(const CloverFieldParam &param);
    virtual ~CloverField();

    void* V(bool inverse=false) { return inverse ? cloverInv : clover; }
    void* Norm(bool inverse=false) { return inverse ? invNorm : norm; }
    const void* V(bool inverse=false) const { return inverse ? cloverInv : clover; }
    const void* Norm(bool inverse=false) const { return inverse ? invNorm : norm; }

    /**
       This function returns true if the field is stored in an
       internal field order for the given precision.
    */
    bool isNative() const;

    double* TrLog() const { return trlog; }
    
    QudaCloverFieldOrder Order() const { return order; }
    size_t Bytes() const { return bytes; }
    size_t NormBytes() const { return norm_bytes; }
//new!
    bool Twisted() const {return twisted; }
    double Mu2() const {return mu2; }
  };

  class cudaCloverField : public CloverField {

  private:
    void *even, *odd;
    void *evenNorm, *oddNorm;

    void *evenInv, *oddInv;
    void *evenInvNorm, *oddInvNorm;

    // computes the clover field given the input gauge field
    void compute(const cudaGaugeField &gauge);

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t evenTex;
    cudaTextureObject_t evenNormTex;
    cudaTextureObject_t oddTex;
    cudaTextureObject_t oddNormTex;
    cudaTextureObject_t evenInvTex;
    cudaTextureObject_t evenInvNormTex;
    cudaTextureObject_t oddInvTex;
    cudaTextureObject_t oddInvNormTex;
    void createTexObject(cudaTextureObject_t &tex, cudaTextureObject_t &texNorm, void *field, void *norm);
    void destroyTexObject();
#endif

  public:
    // create a cudaCloverField from a CloverFieldParam
    cudaCloverField(const CloverFieldParam &param);

    virtual ~cudaCloverField();

#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t& EvenTex() const { return evenTex; }
    const cudaTextureObject_t& EvenNormTex() const { return evenNormTex; }
    const cudaTextureObject_t& OddTex() const { return oddTex; }
    const cudaTextureObject_t& OddNormTex() const { return oddNormTex; }
    const cudaTextureObject_t& EvenInvTex() const { return evenInvTex; }
    const cudaTextureObject_t& EvenInvNormTex() const { return evenInvNormTex; }
    const cudaTextureObject_t& OddInvTex() const { return oddInvTex; }
    const cudaTextureObject_t& OddInvNormTex() const { return oddInvNormTex; }
#endif

    /**
       Copy into this CloverField from the generic CloverField src
       @param src The clover field from which we want to copy
     */
    void copy(const CloverField &src, bool inverse=true);

    /**
       Copy into this CloverField from the cpuCloverField cpu
       @param cpu The cpu clover field from which we want to copy
     */
    void loadCPUField(const cpuCloverField &cpu);

  
    /**
      Copy from this CloverField into cpuCloverField cpu
      @param cpu The cpu clover destination field
    */
    void saveCPUField(cpuCloverField &cpu);  

    friend class DiracClover;
    friend class DiracCloverPC;
    friend struct FullClover;
  };

  // this is a place holder for a future host-side clover object
  class cpuCloverField : public CloverField {

  private:

  public:
    cpuCloverField(const CloverFieldParam &param);
    virtual ~cpuCloverField();
  };

  // lightweight struct used to send pointers to cuda driver code
  struct FullClover {
    void *even;
    void *odd;
    void *evenNorm;
    void *oddNorm;
    QudaPrecision precision;
    size_t bytes; // sizeof each clover field (per parity)
    size_t norm_bytes; // sizeof each norm field (per parity)
    int stride; // stride (volume + pad)

#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t &evenTex;
    const cudaTextureObject_t &evenNormTex;
    const cudaTextureObject_t &oddTex;
    const cudaTextureObject_t &oddNormTex;
    const cudaTextureObject_t& EvenTex() const { return evenTex; }
    const cudaTextureObject_t& EvenNormTex() const { return evenNormTex; }
    const cudaTextureObject_t& OddTex() const { return oddTex; }
    const cudaTextureObject_t& OddNormTex() const { return oddNormTex; }    
#endif

    FullClover(const cudaCloverField &clover, bool inverse=false) :
    precision(clover.precision), bytes(clover.bytes), norm_bytes(clover.norm_bytes), stride(clover.stride)
#ifdef USE_TEXTURE_OBJECTS
	, evenTex(inverse ? clover.evenInvTex : clover.evenTex)
	, evenNormTex(inverse ? clover.evenInvNormTex : clover.evenNormTex)
	, oddTex(inverse ? clover.oddInvTex : clover.oddTex)
	, oddNormTex(inverse ? clover.oddInvNormTex : clover.oddNormTex)
#endif
      { 
	if (inverse) {
	  even = clover.evenInv;
	  evenNorm = clover.evenInvNorm;
	  odd = clover.oddInv;	
	  oddNorm = clover.oddInvNorm;
	} else {
	  even = clover.even;
	  evenNorm = clover.evenNorm;
	  odd = clover.odd;	
	  oddNorm = clover.oddNorm;
	}
    }
  };

  // driver for computing the clover field from the gauge field
  void computeClover(CloverField &clover, const GaugeField &gauge, double coeff,  QudaFieldLocation location);


  /**
     This generic function is used for copying the clover field where
     in the input and output can be in any order and location.
     @param out The output field to which we are copying
     @param in The input field from which we are copying
     @param inverse Whether we are copying the inverse term or not
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
     @param outNorm The output norm buffer (optional)
     @param inNorm The input norm buffer (optional)
  */
  void copyGenericClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
			 void *Out=0, void *In=0, void *outNorm=0, void *inNorm=0);
  


  /**
     This function compute the Cholesky decomposition of each clover
     matrix and stores the clover inverse field.
     @param clover The clover field (contains both the field itself and its inverse)
     @param computeTraceLog Whether to compute the trace logarithm of the clover term
     @param location The location of the field
  */
  void cloverInvert(CloverField &clover, bool computeTraceLog, QudaFieldLocation location);

  /**
     Compute the force contribution from the solver solution fields
   
     Force(x, mu) = U(x, mu) * sum_i=1^nvec ( P_mu^+ x(x+mu) p(x)^\dag  +  P_mu^- p(x+mu) x(x)^\dag )

      M = A_even - kappa^2 * Dslash * A_odd^{-1} * Dslash
      x(even) = M^{-1} b(even)
      x(odd)  = A_odd^{-1} * Dslash * x(even)
      p(even) = M * x(even)
      p(odd)  = A_odd^{-1} * Dslash^dag * M * x(even). 

     @param force The resulting force field
     @param U The input gauge field
     @param x Solution field (both parities)
     @param p Intermediate vectors (both parities)
     @param coeff Multiplicative coefficient (e.g., dt * residue)
   */
  void computeCloverForce(cudaGaugeField& force, const cudaGaugeField& U,
			  cudaColorSpinorField& x, cudaColorSpinorField& p,
			  const double coeff);
  /**
     Compute the outer product from the solver solution fields arising
     from the diagonal term of the fermion bilinear in direction mu,nu.

     @param oprod Computed outer product field (scalar matrix field)
     @param x Solution field (both parities)
     @param p Intermediate vectors (both parities)
     @coeff coeff Multiplicative coefficient (e.g., dt * residiue)
     @param mu mu direction
     @param nu nu direction
     @param update If update>1 then read/write to oprod field else just over write
  */
  void computeCloverSigmaOprod(cudaGaugeField& oprod,
			       cudaColorSpinorField& x,  
			       cudaColorSpinorField& p,
			       const double coeff, int mu, int nu, int update);
  /**
     Compute the matrix field necessary for the force calculation from
     the clover trace action.

     @param output The computed matrix field (scalar)
     @param clover The input clover field
     @param mu mu direction
     @param nu nu direction
     @param location The location of the computation
   */
  void computeCloverSigmaTrace(GaugeField &output, const CloverField &clover, int mu, int nu, QudaFieldLocation location);

  /**
     Compute the derivative of the clover matrix in the direction
     mu,nu and compute the resulting force given the outer-product
     field

     @param force The compute force field (read/write update)
     @param gauge The input gauge field
     @param oprod The input outer-product field
     @param mu mu direction
     @param nu nu direction
     @param coeff Multiplicative coefficient (e.g., clover coefficient)
     @param parity The field parity we are working on 
     @param conjugate Whether to set oprod = oprod - conj(oprod) or not
   */
  void cloverDerivative(cudaGaugeField &force, cudaGaugeField& gauge, cudaGaugeField& oprod, int mu, int nu, double coeff, QudaParity parity, int conjugate);

} // namespace quda

#endif // _CLOVER_QUDA_H

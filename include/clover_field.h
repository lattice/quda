#pragma once

#include <array>
#include <quda_internal.h>
#include <lattice_field.h>
#include <comm_key.h>

namespace quda {

  namespace clover
  {

    inline bool isNative(QudaCloverFieldOrder order, QudaPrecision precision)
    {
      if (precision == QUDA_DOUBLE_PRECISION) {
        if (order == QUDA_FLOAT2_CLOVER_ORDER) return true;
      } else if (precision == QUDA_SINGLE_PRECISION || precision == QUDA_HALF_PRECISION
                 || precision == QUDA_QUARTER_PRECISION) {
        if (order == QUDA_FLOAT4_CLOVER_ORDER) return true;
      }
      return false;
    }

  } // namespace clover

  // Prefetch type
  enum class CloverPrefetchType {
    BOTH_CLOVER_PREFETCH_TYPE,    // clover and inverse
    CLOVER_CLOVER_PREFETCH_TYPE,  // clover only
    INVERSE_CLOVER_PREFETCH_TYPE, // inverse clover only
    INVALID_CLOVER_PREFETCH_TYPE = QUDA_INVALID_ENUM
  };

  struct CloverFieldParam : public LatticeFieldParam {
    bool direct; // whether to create the direct clover 
    bool inverse; // whether to create the inverse clover
    void *clover;
    void *norm;
    void *cloverInv;
    void *invNorm;
    double csw;   //! C_sw clover coefficient
    double coeff; //! Overall clover coefficient
    bool twisted; // whether to create twisted mass clover
    double mu2;
    double rho;

    QudaCloverFieldOrder order;
    QudaFieldCreate create;

    QudaFieldLocation location;

    /**
       @brief Helper function for setting the precision and corresponding
       field order for QUDA internal fields.
       @param precision The precision to use
       @param force_native Whether we should force the field order to be native
    */
    void setPrecision(QudaPrecision precision, bool force_native = false)
    {
      // is the current status in native field order?
      bool native = force_native ? true : clover::isNative(order, this->precision);
      this->precision = precision;
      this->ghost_precision = precision;

      if (native) {
        order = (precision == QUDA_DOUBLE_PRECISION) ? QUDA_FLOAT2_CLOVER_ORDER : QUDA_FLOAT4_CLOVER_ORDER;
      }
    }

    CloverFieldParam() :
      LatticeFieldParam(),
      direct(true),
      inverse(true),
      clover(nullptr),
      norm(nullptr),
      cloverInv(nullptr),
      invNorm(nullptr),
      twisted(false),
      mu2(0.0),
      rho(0.0),
      location(QUDA_INVALID_FIELD_LOCATION)
    {
    }

    CloverFieldParam(const CloverFieldParam &param) :
      LatticeFieldParam(param),
      direct(param.direct),
      inverse(param.inverse),
      clover(param.clover),
      norm(param.norm),
      cloverInv(param.cloverInv),
      invNorm(param.invNorm),
      twisted(param.twisted),
      mu2(param.mu2),
      rho(param.rho),
      location(param.location)
    {
    }

    CloverFieldParam(const CloverField &field);
  };

  std::ostream& operator<<(std::ostream& output, const CloverFieldParam& param);

  class CloverField : public LatticeField {

  protected:
    size_t bytes; // bytes allocated per clover full field 
    size_t norm_bytes; // sizeof each norm full field
    size_t length;
    size_t real_length;
    int nColor;
    int nSpin;

    void *clover;
    void *norm;
    void *cloverInv;
    void *invNorm;

    double csw;
    double coeff;
    bool twisted; 
    double mu2;
    double rho;

    QudaCloverFieldOrder order;
    QudaFieldCreate create;

    mutable std::array<double, 2> trlog;

    /**
       @brief Set the vol_string and aux_string for use in tuning
    */
    void setTuningString();

  public:
    CloverField(const CloverFieldParam &param);
    virtual ~CloverField();

    static CloverField *Create(const CloverFieldParam &param);

    void* V(bool inverse=false) { return inverse ? cloverInv : clover; }
    void* Norm(bool inverse=false) { return inverse ? invNorm : norm; }
    const void* V(bool inverse=false) const { return inverse ? cloverInv : clover; }
    const void* Norm(bool inverse=false) const { return inverse ? invNorm : norm; }

    /**
     * Define the parameter type for this field.
     */
    using param_type = CloverFieldParam;

    /**
       @return True if the field is stored in an internal field order
       for the given precision.
    */
    bool isNative() const { return clover::isNative(order, precision); }

    /**
       @return Pointer to array storing trlog on each parity
    */
    std::array<double, 2> &TrLog() const { return trlog; }

    /**
       @return The order of the field
     */
    QudaCloverFieldOrder Order() const { return order; }

    /**
       @return The size of the fieldallocation
     */
    size_t Bytes() const { return bytes; }

    /**
       @return The size of the norm allocation
     */
    size_t NormBytes() const { return norm_bytes; }

    /**
       @return The total bytes of allocation
     */
    size_t TotalBytes() const
    {
      int direct = V(false) ? 1 : 0;
      int inverse = V(true) ? 1 : 0;
      return (direct + inverse) * (bytes + norm_bytes);
    }

    /**
       @return Number of colors
    */
    int Ncolor() const { return nColor; }

    /**
       @return Number of spins
    */
    int Nspin() const { return nSpin; }

    /**
       @return Csw coefficient (does not include kappa)
    */
    double Csw() const { return csw; }

    /**
       @return Clover coefficient (explicitly includes kappa)
    */
    double Coeff() const { return coeff; }

    /**
       @return If the clover field is associated with twisted-clover fermions
    */
    bool Twisted() const { return twisted; }

    /**
       @return mu^2 factor baked into inverse clover field (for twisted-clover inverse)
    */
    double Mu2() const { return mu2; }

    /**
       @return rho factor backed into the clover field, (for real
       diagonal additive Hasenbusch), e.g., A + rho
    */
    double Rho() const { return rho; }

    /**
       @brief Bakes in the rho factor into the clover field, (for real
       diagonal additive Hasenbusch), e.g., A + rho
    */
    void setRho(double rho);

    /**
       @brief Compute the L1 norm of the field
       @return L1 norm
     */
    double norm1(bool inverse = false) const;

    /**
       @brief Compute the L2 norm squared of the field
       @return L2 norm squared
     */
    double norm2(bool inverse = false) const;

    /**
       @brief Compute the absolute maximum of the field (Linfinity norm)
       @return Absolute maximum value
     */
    double abs_max(bool inverse = false) const;

    /**
       @brief Compute the absolute minimum of the field
       @return Absolute minimum value
     */
    double abs_min(bool inverse = false) const;

    /**
       @brief Backs up the CloverField
    */
    void backup() const;

    /**
       @brief Restores the CloverField
    */
    void restore() const;

    virtual int full_dim(int d) const { return x[d]; }
  };

  class cudaCloverField : public CloverField {

  private:
    void *even, *odd;
    void *evenNorm, *oddNorm;

    void *evenInv, *oddInv;
    void *evenInvNorm, *oddInvNorm;

    // computes the clover field given the input gauge field
    void compute(const cudaGaugeField &gauge);

  public:
    // create a cudaCloverField from a CloverFieldParam
    cudaCloverField(const CloverFieldParam &param);

    virtual ~cudaCloverField();

    /**
       @brief Copy into this CloverField from the generic CloverField src
       @param src The clover field from which we want to copy
       @param inverse Are we copying the inverse or direct field
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
    void saveCPUField(cpuCloverField &cpu) const;

    /**
      @brief If managed memory and prefetch is enabled, prefetch
      the clover, the norm field (as appropriate), and the inverse
      fields (as appropriate) to the CPU or the GPU.
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in (default 0)
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream = device::get_default_stream()) const;

    /**
      @brief If managed memory and prefetch is enabled, prefetch
      the clover, norm field and/or the inverse
      fields as specified to the CPU or the GPU.
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in
      @param[in] type Whether to grab the clover, inverse, or both
      @param[in] parity Whether to grab the full clover or just the even/odd parity
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream, CloverPrefetchType type,
                  QudaParity parity = QUDA_INVALID_PARITY) const;

    /**
      @brief Copy all contents of the field to a host buffer.
      @param[in] the host buffer to copy to.
    */
    virtual void copy_to_buffer(void *buffer) const;

    /**
      @brief Copy all contents of the field from a host buffer to this field.
      @param[in] the host buffer to copy from.
    */
    virtual void copy_from_buffer(void *buffer);

    friend class DiracClover;
    friend class DiracCloverPC;
    friend class DiracTwistedClover;
    friend class DiracTwistedCloverPC;
    friend struct FullClover;
  };

  // this is a place holder for a future host-side clover object
  class cpuCloverField : public CloverField {

  private:

  public:
    cpuCloverField(const CloverFieldParam &param);
    virtual ~cpuCloverField();

    /**
      @brief Copy all contents of the field to a host buffer.
      @param[in] the host buffer to copy to.
    */
    virtual void copy_to_buffer(void *buffer) const;

    /**
      @brief Copy all contents of the field from a host buffer to this field.
      @param[in] the host buffer to copy from.
    */
    virtual void copy_from_buffer(void *buffer);
  };

  /**
     This is a debugging function, where we cast a clover field into a
     spinor field so we can compute its L1 norm.
     @param a The clover field that we want the norm of
     @return The L1 norm of the gauge field
  */
  double norm1(const CloverField &u, bool inverse=false);

  /**
     This is a debugging function, where we cast a clover field into a
     spinor field so we can compute its L2 norm.
     @param a The clover field that we want the norm of
     @return The L2 norm squared of the gauge field
  */
  double norm2(const CloverField &a, bool inverse=false);


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
    double rho; // rho additive factor

    FullClover(const cudaCloverField &clover, bool inverse = false) :
      precision(clover.precision),
      bytes(clover.bytes),
      norm_bytes(clover.norm_bytes),
      stride(clover.stride),
      rho(clover.rho)
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

  /**
     @brief Driver for computing the clover field from the field
     strength tensor.
     @param[out] clover Compute clover field
     @param[in] fmunu Field strength tensor
     @param[in] coefft Clover coefficient
  */
  void computeClover(CloverField &clover, const GaugeField &fmunu, double coeff);

  /**
     @brief This generic function is used for copying the clover field where
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
  void copyGenericClover(CloverField &out, const CloverField &in, bool inverse,
			 QudaFieldLocation location, void *Out=0, void *In=0, void *outNorm=0, void *inNorm=0);
  


  /**
     @brief This function compute the Cholesky decomposition of each clover
     matrix and stores the clover inverse field.

     @param clover The clover field (contains both the field itself and its inverse)
     @param computeTraceLog Whether to compute the trace logarithm of the clover term
  */
  void cloverInvert(CloverField &clover, bool computeTraceLog);

  /**
     @brief This function adds a real scalar onto the clover diagonal (only to the direct field not the inverse)

     @param clover The clover field
     @param rho Real scalar to be added on
  */
  void cloverRho(CloverField &clover, double rho);

  /**
     @brief Compute the force contribution from the solver solution fields
   
     Force(x, mu) = U(x, mu) * sum_i=1^nvec ( P_mu^+ x(x+mu) p(x)^\dag  +  P_mu^- p(x+mu) x(x)^\dag )

      M = A_even - kappa^2 * Dslash * A_odd^{-1} * Dslash
      x(even) = M^{-1} b(even)
      x(odd)  = A_odd^{-1} * Dslash * x(even)
      p(even) = M * x(even)
      p(odd)  = A_odd^{-1} * Dslash^dag * M * x(even). 

     @param force[out,in] The resulting force field
     @param U The input gauge field
     @param x Solution field (both parities)
     @param p Intermediate vectors (both parities)
     @param coeff Multiplicative coefficient (e.g., dt * residue)
   */
  void computeCloverForce(GaugeField& force, const GaugeField& U,
			  std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &p,
			  std::vector<double> &coeff);
  /**
     @brief Compute the outer product from the solver solution fields
     arising from the diagonal term of the fermion bilinear in
     direction mu,nu and sum to outer product field.

     @param oprod[out,in] Computed outer product field (tensor matrix field)
     @param x[in] Solution field (both parities)
     @param p[in] Intermediate vectors (both parities)
     @coeff coeff[in] Multiplicative coefficient (e.g., dt * residiue), one for each parity
  */
  void computeCloverSigmaOprod(GaugeField& oprod,
			       std::vector<ColorSpinorField*> &x,
			       std::vector<ColorSpinorField*> &p,
			       std::vector< std::vector<double> > &coeff);
  /**
     @brief Compute the matrix tensor field necessary for the force calculation from
     the clover trace action.  This computes a tensor field [mu,nu].

     @param output The computed matrix field (tensor matrix field)
     @param clover The input clover field
     @param coeff  Scalar coefficient multiplying the result (e.g., stepsize)
   */
  void computeCloverSigmaTrace(GaugeField &output, const CloverField &clover, double coeff);

  /**
     @brief Compute the derivative of the clover matrix in the direction
     mu,nu and compute the resulting force given the outer-product
     field

     @param force The computed force field (read/write update)
     @param gauge The input gauge field
     @param oprod The input outer-product field (tensor matrix field)
     @param coeff Multiplicative coefficient (e.g., clover coefficient)
     @param parity The field parity we are working on 
   */
  void cloverDerivative(GaugeField &force, GaugeField &gauge, GaugeField &oprod, double coeff, QudaParity parity);

  /**
    @brief This function is used for copying from a source clover field to a destination clover field
      with an offset.
    @param out The output field to which we are copying
    @param in The input field from which we are copying
    @param offset The offset for the larger field between out and in.
    @param pc_type Whether the field order uses 4d or 5d even-odd preconditioning.
 */
  void copyFieldOffset(CloverField &out, const CloverField &in, CommKey offset, QudaPCType pc_type);

  /**
     @brief Helper function that returns whether we have enabled
     dynamic clover inversion or not.
   */
  constexpr bool dynamic_clover_inverse()
  {
#ifdef DYNAMIC_CLOVER
    return true;
#else
    return false;
#endif
  }

} // namespace quda

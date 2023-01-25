#pragma once

#include <quda_internal.h>
#include <lattice_field.h>
#include <comm_key.h>
#include <array.h>

namespace quda {

  namespace clover
  {

    /**
       @brief Helper function that returns whether we have enabled
       dynamic clover inversion or not.
    */
    constexpr bool dynamic_inverse()
    {
#ifdef DYNAMIC_CLOVER
      return true;
#else
      return false;
#endif
    }

    /**
       @brief Precision mapper that is used for the Cholesky
       factorization when inverting the clover matrices.  If
       CLOVER_PROMOTE_CHOLESKY is set, then we always use double
       precision, else we use the same precision as the type.  For
       fixed-point types we always use single precision regardless.
    */
#ifdef CLOVER_PROMOTE_CHOLESKY
    template <typename T> struct cholesky_mapper {
      using type = double;
    };
#else
    template <typename T> struct cholesky_mapper {
      using type = T;
    };
#endif
    template <> struct cholesky_mapper<short> {
      using type = float;
    };
    template <> struct cholesky_mapper<int8_t> {
      using type = float;
    };

    template <typename T> using cholesky_t = typename cholesky_mapper<T>::type;

    /**
       @brief Helper function that returns whether we have enabled
       clover reconstruction or not.
    */
    constexpr bool reconstruct()
    {
#ifdef RECONSTRUCT_CLOVER
      return true;
#else
      return false;
#endif
    }

    template <typename T> constexpr auto getNative() { return QUDA_FLOAT2_CLOVER_ORDER; }
    template <> constexpr auto getNative<float>() { return QUDA_FLOAT4_CLOVER_ORDER; }
    template <> constexpr auto getNative<short>() { return static_cast<QudaCloverFieldOrder>(QUDA_ORDER_FP); }
    template <> constexpr auto getNative<int8_t>() { return static_cast<QudaCloverFieldOrder>(QUDA_ORDER_FP); }

    constexpr QudaCloverFieldOrder getNative(QudaPrecision precision)
    {
      switch (precision) {
      case QUDA_DOUBLE_PRECISION: return getNative<double>();
      case QUDA_SINGLE_PRECISION: return getNative<float>();
      case QUDA_HALF_PRECISION: return getNative<short>();
      case QUDA_QUARTER_PRECISION: return getNative<int8_t>();
      default: return QUDA_INVALID_CLOVER_ORDER;
      }
    }

    constexpr bool isNative(QudaCloverFieldOrder order, QudaPrecision precision)
    {
      return order == getNative(precision);
    }

  } // namespace clover

  // Prefetch type
  enum class CloverPrefetchType {
    BOTH_CLOVER_PREFETCH_TYPE,     /** clover and inverse */
      CLOVER_CLOVER_PREFETCH_TYPE, /** clover only */
      INVERSE_CLOVER_PREFETCH_TYPE,/** inverse clover only */
    INVALID_CLOVER_PREFETCH_TYPE = QUDA_INVALID_ENUM
  };

  struct CloverFieldParam : public LatticeFieldParam {
    bool reconstruct
      = clover::reconstruct(); /** Whether to create a compressed clover field that requires reconstruction */
    bool inverse = true;       /** Whether to create the inverse clover field */
    void *clover = nullptr;    /** Pointer to the clover field */
    void *cloverInv = nullptr; /** Pointer to the clover inverse field */
    double csw = 0.0;          /** C_sw clover coefficient */
    double coeff = 0.0;        /** Overall clover coefficient */
    QudaTwistFlavorType twist_flavor = QUDA_TWIST_INVALID; /** Twisted-mass flavor type */
    bool twisted = false;                                  /** Whether to create twisted mass clover */
    double mu2 = 0.0;                                      /** Chiral twisted mass term */
    double epsilon2 = 0.0;                                 /** Flavor twisted mass term */
    double rho = 0.0;                                      /** Hasenbusch rho term */

    QudaCloverFieldOrder order = QUDA_INVALID_CLOVER_ORDER; /** Field order */
    QudaFieldCreate create = QUDA_INVALID_FIELD_CREATE;     /** Creation type */

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

      if (native) order = clover::getNative(precision);
    }

    CloverFieldParam() = default;

    CloverFieldParam(const CloverFieldParam &param) :
      LatticeFieldParam(param),
      reconstruct(param.reconstruct),
      inverse(param.inverse),
      clover(param.clover),
      cloverInv(param.cloverInv),
      twist_flavor(param.twist_flavor),
      mu2(param.mu2),
      epsilon2(param.epsilon2),
      rho(param.rho)
    {
    }

    CloverFieldParam(const QudaInvertParam &inv_param, const lat_dim_t &x) :
      LatticeFieldParam(),
      reconstruct(clover::reconstruct()),
      inverse(true),
      clover(nullptr),
      cloverInv(nullptr),
      csw(inv_param.clover_csw),
      // If clover_coeff is not set manually, then it is the product Csw * kappa.
      // If the user has set the clover_coeff manually, that value takes precedent.
      coeff(inv_param.clover_coeff == 0.0 ? inv_param.kappa * inv_param.clover_csw : inv_param.clover_coeff),
      twist_flavor(inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH ? inv_param.twist_flavor : QUDA_TWIST_NO),
      mu2(twist_flavor != QUDA_TWIST_NO ? 4. * inv_param.kappa * inv_param.kappa * inv_param.mu * inv_param.mu : 0.0),
      epsilon2(twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ?
                 4.0 * inv_param.kappa * inv_param.kappa * inv_param.epsilon * inv_param.epsilon :
                 0.0),
      rho(inv_param.clover_rho)
    {
      siteSubset = QUDA_FULL_SITE_SUBSET;
      this->x = x;
    }

    CloverFieldParam(const CloverField &field);
  };

  std::ostream& operator<<(std::ostream& output, const CloverFieldParam& param);

  class CloverField : public LatticeField {

  protected:
    const bool reconstruct = clover::reconstruct(); /** Whether this field is compressed and requires reconstruction */

    size_t bytes = 0; // bytes allocated per clover full field
    size_t length = 0;
    size_t real_length = 0;
    size_t compressed_block = 0; /** Length of compressed chiral block */
    int nColor = 0;
    int nSpin = 0;

    void *clover = nullptr;
    void *cloverInv = nullptr;

    double diagonal = 0.0;
    array<double, 2> max = {};

    double csw = 0.0;
    double coeff = 0.0;
    QudaTwistFlavorType twist_flavor = QUDA_TWIST_INVALID;
    double mu2 = 0.0;      // chiral twisted mass squared
    double epsilon2 = 0.0; // flavour twisted mass squared
    double rho = 0.0;

    QudaCloverFieldOrder order = QUDA_INVALID_CLOVER_ORDER;
    QudaFieldCreate create = QUDA_INVALID_FIELD_CREATE;

    mutable array<double, 2> trlog = {};

    /**
       @brief Set the vol_string and aux_string for use in tuning
    */
    void setTuningString();

    /**
       @brief Backs up the CloverField (called by public backup function)
    */
    void backup(bool which) const;

    /**
       @brief Restores the CloverField (called by public restore function)
    */
    void restore(bool which) const;

  public:
    CloverField(const CloverFieldParam &param);
    virtual ~CloverField();

    static CloverField *Create(const CloverFieldParam &param);

    void* V(bool inverse=false) { return inverse ? cloverInv : clover; }
    const void* V(bool inverse=false) const { return inverse ? cloverInv : clover; }

    /**
       @return diagonal scaling factor applied to the identity
    */
    double Diagonal() const { return diagonal; }

    /**
       @return set diagonal scaling factor applied to the identity
    */
    void Diagonal(double diagonal) { this->diagonal = diagonal; }

    /**
       @return max element in the clover field for fixed-point scaling
    */
    auto max_element(bool inverse) const { return max[inverse]; }

    /**
     * Define the parameter type for this field.
     */
    using param_type = CloverFieldParam;

    /**
       @return If the field is compressed and requires reconstruction
    */
    bool Reconstruct() const { return reconstruct; }

    /**
       @return True if the field is stored in an internal field order
       for the given precision.
    */
    bool isNative() const { return clover::isNative(order, precision); }

    /**
       @return Array storing trlog on each parity
    */
    auto &TrLog() const { return trlog; }

    /**
       @return The order of the field
     */
    QudaCloverFieldOrder Order() const { return order; }

    /**
       @return The size of the field allocation
     */
    size_t Bytes() const { return bytes; }

    /**
       @return The total bytes of allocation
     */
    size_t TotalBytes() const { return total_bytes; }

    /**
       @return The storage length of the compressed chiral block
     */
    size_t compressed_block_size() const { return compressed_block; }

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
       @return If the clover field is associated with twisted-clover fermions and which flavor type thereof
    */
    QudaTwistFlavorType TwistFlavor() const { return twist_flavor; }

    /**
       @return mu^2 factor baked into inverse clover field (for twisted-clover inverse)
    */
    double Mu2() const { return mu2; }

    /**
       @return epsilon^2 factor baked into inverse clover field (for non-deg twisted-clover inverse)
    */
    double Epsilon2() const { return epsilon2; }

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
       @brief Copy into this CloverField from CloverField src
       @param src The clover field from which we want to copy
       @param inverse Are we copying the inverse or direct field?
     */
    void copy(const CloverField &src, bool inverse);

    /**
       @brief Copy into this CloverField from CloverField src.  Will
       copy both the field and its inverse (if it exists).
       @param src The clover field from which we want to copy
     */
    void copy(const CloverField &src);

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

    /**
      @brief If managed memory and prefetch is enabled, prefetch the
      clover and the inverse fields (as appropriate) to the CPU or the
      GPU.
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in (default 0)
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream = device::get_default_stream()) const;

    /**
      @brief If managed memory and prefetch is enabled, prefetch the
      clover, and/or the inverse fields as specified to the CPU or the
      GPU.
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in
      @param[in] type Whether to grab the clover, inverse, or both
      @param[in] parity Whether to grab the full clover or just the even/odd parity
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream, CloverPrefetchType type,
                  QudaParity parity = QUDA_INVALID_PARITY) const;

    int full_dim(int d) const { return x[d]; }

    /**
      @brief Copy all contents of the field to a host buffer.
      @param[in] the host buffer to copy to.
    */
    void copy_to_buffer(void *buffer) const;

    /**
      @brief Copy all contents of the field from a host buffer to this field.
      @param[in] the host buffer to copy from.
    */
    void copy_from_buffer(void *buffer);

    friend class DiracClover;
    friend class DiracCloverPC;
    friend class DiracTwistedClover;
    friend class DiracTwistedCloverPC;
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
  */
  void copyGenericClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
                         void *Out = 0, const void *In = 0);

  /**
     @brief This function compute the Cholesky decomposition of each clover
     matrix and stores the clover inverse field.

     @param clover The clover field (contains both the field itself and its inverse)
     @param computeTraceLog Whether to compute the trace logarithm of the clover term
  */
  void cloverInvert(CloverField &clover, bool computeTraceLog);

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

} // namespace quda

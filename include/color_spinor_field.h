#pragma once

#include <iostream>
#include <quda_internal.h>
#include <quda.h>
#include <lattice_field.h>
#include <field_cache.h>
#include <comm_key.h>

namespace quda
{

  namespace colorspinor
  {

    template <typename T, int nSpin> constexpr auto getNative() { return QUDA_FLOAT2_FIELD_ORDER; }
    template <> constexpr auto getNative<float, 4>() { return QUDA_FLOAT4_FIELD_ORDER; }

    // fixed-point Wilson fields
    template <> constexpr auto getNative<short, 4>() { return static_cast<QudaFieldOrder>(QUDA_ORDER_FP); }
    template <> constexpr auto getNative<int8_t, 4>() { return static_cast<QudaFieldOrder>(QUDA_ORDER_FP); }

    // fp32 multigrid fields
    template <> constexpr auto getNative<float, 2>() { return static_cast<QudaFieldOrder>(QUDA_ORDER_SP_MG); }

    // fixed-point multigrid fields
    template <> constexpr auto getNative<short, 2>() { return static_cast<QudaFieldOrder>(QUDA_ORDER_FP_MG); }
    template <> constexpr auto getNative<int8_t, 2>() { return static_cast<QudaFieldOrder>(QUDA_ORDER_FP_MG); }

    template <typename T> constexpr auto getNative(int) { return QUDA_INVALID_FIELD_ORDER; }

    template <> constexpr auto getNative<double>(int nSpin)
    {
      return nSpin == 1 ? getNative<double, 1>() : nSpin == 2 ? getNative<double, 2>() : getNative<double, 4>();
    }

    template <> constexpr auto getNative<float>(int nSpin)
    {
      return nSpin == 1 ? getNative<float, 1>() : nSpin == 2 ? getNative<float, 2>() : getNative<float, 4>();
    }

    template <> constexpr auto getNative<short>(int nSpin)
    {
      return nSpin == 1 ? getNative<short, 1>() : nSpin == 2 ? getNative<short, 2>() : getNative<short, 4>();
    }

    template <> constexpr auto getNative<int8_t>(int nSpin)
    {
      return nSpin == 1 ? getNative<int8_t, 1>() : nSpin == 2 ? getNative<int8_t, 2>() : getNative<int8_t, 4>();
    }

    constexpr QudaFieldOrder getNative(QudaPrecision precision, int nSpin)
    {
      switch (precision) {
      case QUDA_DOUBLE_PRECISION: return getNative<double>(nSpin);
      case QUDA_SINGLE_PRECISION: return getNative<float>(nSpin);
      case QUDA_HALF_PRECISION: return getNative<short>(nSpin);
      case QUDA_QUARTER_PRECISION: return getNative<int8_t>(nSpin);
      default: return QUDA_INVALID_FIELD_ORDER;
      }
    }

    constexpr bool isNative(QudaFieldOrder order, QudaPrecision precision, int nSpin, int)
    {
      return order == getNative(precision, nSpin);
    }

  } // namespace colorspinor

  enum MemoryLocation { Device = 1, Host = 2, Remote = 4, Shmem = 8 };

  /**
     @brief Helper function for getting the implied spinor parity from a matrix preconditioning type.
     @param[in] matpc_type The matrix preconditioning type
     @return Even or Odd as appropriate, invalid if the preconditioning type is invalid (implicitly non-preconditioned)
   */
  constexpr QudaParity impliedParityFromMatPC(const QudaMatPCType &matpc_type)
  {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      return QUDA_EVEN_PARITY;
    } else if (matpc_type == QUDA_MATPC_ODD_ODD || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      return QUDA_ODD_PARITY;
    } else {
      return QUDA_INVALID_PARITY;
    }
  }

  /** Typedef for a set of spinors. Can be further divided into subsets ,e.g., with different precisions (not implemented currently) */
  typedef std::vector<ColorSpinorField *> CompositeColorSpinorField;

  /**
     Any spinor object can be qualified in the following categories:
     1. A regular spinor field (is_composite = false , is_component = false)
     2. A composite spinor field, i.e., a collection of spinor fields (is_composite = true , is_component = false)
     3. An individual component of a composite spinor field (is_composite = false , is_component = true)
     4. A subset of a composite spinor field (e.g., based on index range or field precision) : currently not implemented
  */
  struct CompositeColorSpinorFieldDescriptor {

    bool is_composite = false; // set to 'false' for a regular spinor field
    bool is_component
      = false; // set to 'true' if we want to work with an individual component (otherwise will work with the whole set)

    int dim = 0; // individual component has dim = 0
    int id = 0;

    size_t volume = 0;   // volume of a single eigenvector
    size_t volumeCB = 0; // CB volume of a single eigenvector
    size_t length = 0;   // length (excluding norm))
    size_t bytes = 0;    // size in bytes of spinor field

    CompositeColorSpinorFieldDescriptor() = default;

    CompositeColorSpinorFieldDescriptor(bool is_composite, int dim, bool is_component = false, int id = 0) :
      is_composite(is_composite), is_component(is_component), dim(dim), id(id), volume(0), volumeCB(0), length(0), bytes(0)
    {
      if (is_composite && is_component)
        errorQuda("Composite type is not implemented");
      else if (is_composite && dim == 0)
        is_composite = false;
    }
  };

  class ColorSpinorParam : public LatticeFieldParam
  {

  public:
    int nColor = 0; // Number of colors of the field
    int nSpin = 0;  // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor
    int nVec = 1;   // number of packed vectors (for multigrid transfer operator)

    QudaTwistFlavorType twistFlavor = QUDA_TWIST_INVALID; // used by twisted mass

    QudaSiteOrder siteOrder = QUDA_INVALID_SITE_ORDER; // defined for full fields

    QudaFieldOrder fieldOrder = QUDA_INVALID_FIELD_ORDER; // Float, Float2, Float4 etc.
    QudaGammaBasis gammaBasis = QUDA_INVALID_GAMMA_BASIS;
    QudaFieldCreate create = QUDA_INVALID_FIELD_CREATE;

    QudaPCType pc_type = QUDA_PC_INVALID; // used to select preconditioning method in DWF

    /** Used to specify whether a single parity field is even/odd
     * By construction not enforced, this is more of an optional
     * metadata to specify, for ex, if an eigensolver is for an
     * even or odd parity. */
    QudaParity suggested_parity = QUDA_INVALID_PARITY;

    ColorSpinorField *field = nullptr;
    void *v = nullptr; // pointer to field
    size_t norm_offset = 0;

    //! for deflation solvers:
    bool is_composite = false;
    int composite_dim = 0; // e.g., number of eigenvectors in the set
    bool is_component = false;
    int component_id = 0; // eigenvector index

    /**
       If using CUDA native fields, this function will ensure that the
       field ordering is appropriate for the new precision setting to
       maintain this status
       @param precision_ New precision value
       @param ghost_precision_ New ghost precision value
     */
    void setPrecision(QudaPrecision precision, QudaPrecision ghost_precision = QUDA_INVALID_PRECISION,
                      bool force_native = false)
    {
      // is the current status in native field order?
      bool native = force_native ? true : colorspinor::isNative(fieldOrder, this->precision, nSpin, nColor);
      this->precision = precision;
      this->ghost_precision = (ghost_precision == QUDA_INVALID_PRECISION) ? precision : ghost_precision;

      // if this is a native field order, let's preserve that status, else keep the same field order
      if (native) fieldOrder = colorspinor::getNative(precision, nSpin);
    }

    ColorSpinorParam(const ColorSpinorField &a);

    ColorSpinorParam() = default;

    // used to create cpu params

    ColorSpinorParam(void *V, QudaInvertParam &inv_param, const lat_dim_t &X, const bool pc_solution,
                     QudaFieldLocation location = QUDA_CPU_FIELD_LOCATION) :
      LatticeFieldParam(4, X, 0, location, inv_param.cpu_prec),
      nColor(3),
      nSpin((inv_param.dslash_type == QUDA_ASQTAD_DSLASH || inv_param.dslash_type == QUDA_STAGGERED_DSLASH
             || inv_param.dslash_type == QUDA_LAPLACE_DSLASH) ?
              1 :
              4),
      nVec(1),
      twistFlavor(inv_param.twist_flavor),
      siteOrder(QUDA_INVALID_SITE_ORDER),
      fieldOrder(QUDA_INVALID_FIELD_ORDER),
      gammaBasis(inv_param.gamma_basis),
      create(QUDA_REFERENCE_FIELD_CREATE),
      pc_type(inv_param.dslash_type == QUDA_DOMAIN_WALL_DSLASH ? QUDA_5D_PC : QUDA_4D_PC),
      v(V),
      is_composite(false),
      composite_dim(0),
      is_component(false),
      component_id(0)
    {

      if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
      for (int d = 0; d < nDim; d++) x[d] = X[d];

      if (!pc_solution) {
        siteSubset = QUDA_FULL_SITE_SUBSET;
      } else {
        x[0] /= 2; // X defined the full lattice dimensions
        siteSubset = QUDA_PARITY_SITE_SUBSET;
      }

      suggested_parity = impliedParityFromMatPC(inv_param.matpc_type);

      if (inv_param.dslash_type == QUDA_DOMAIN_WALL_DSLASH || inv_param.dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
          || inv_param.dslash_type == QUDA_MOBIUS_DWF_DSLASH || inv_param.dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
        nDim++;
        x[4] = inv_param.Ls;
      } else if ((inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH || inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
                 && twistFlavor == QUDA_TWIST_NONDEG_DOUBLET) {
        nDim++;
        x[4] = 2; // for two flavors
      } else {
        x[4] = 1;
      }

      if (inv_param.dirac_order == QUDA_INTERNAL_DIRAC_ORDER) {
        setPrecision(precision, precision, true);
        siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
      } else if (inv_param.dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
        fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        siteOrder = QUDA_ODD_EVEN_SITE_ORDER;
      } else if (inv_param.dirac_order == QUDA_QDP_DIRAC_ORDER) {
        fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
        siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
      } else if (inv_param.dirac_order == QUDA_DIRAC_ORDER) {
        fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
      } else if (inv_param.dirac_order == QUDA_QDPJIT_DIRAC_ORDER) {
        fieldOrder = QUDA_QDPJIT_FIELD_ORDER;
        siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
      } else if (inv_param.dirac_order == QUDA_TIFR_PADDED_DIRAC_ORDER) {
        fieldOrder = QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER;
        siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
      } else {
        errorQuda("Dirac order %d not supported", inv_param.dirac_order);
      }
    }

    // normally used to create cuda param from a cpu param
    ColorSpinorParam(ColorSpinorParam &cpuParam, QudaInvertParam &inv_param, QudaFieldLocation location) :
      LatticeFieldParam(cpuParam.nDim, cpuParam.x, 0, location, inv_param.cuda_prec),
      nColor(cpuParam.nColor),
      nSpin(cpuParam.nSpin),
      nVec(cpuParam.nVec),
      twistFlavor(cpuParam.twistFlavor),
      siteOrder(QUDA_EVEN_ODD_SITE_ORDER),
      fieldOrder(QUDA_INVALID_FIELD_ORDER),
      gammaBasis(nSpin == 4 ? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS),
      create(QUDA_NULL_FIELD_CREATE),
      pc_type(cpuParam.pc_type),
      suggested_parity(cpuParam.suggested_parity),
      v(0),
      is_composite(cpuParam.is_composite),
      composite_dim(cpuParam.composite_dim),
      is_component(false),
      component_id(0)
    {
      siteSubset = cpuParam.siteSubset;
      setPrecision(precision, precision, true);
      for (int d = 0; d < QUDA_MAX_DIM; d++) x[d] = cpuParam.x[d];
    }

    void print()
    {
      printfQuda("nColor = %d\n", nColor);
      printfQuda("nSpin = %d\n", nSpin);
      printfQuda("twistFlavor = %d\n", twistFlavor);
      printfQuda("nDim = %d\n", nDim);
      for (int d = 0; d < nDim; d++) printfQuda("x[%d] = %d\n", d, x[d]);
      printfQuda("precision = %d\n", precision);
      printfQuda("ghost_precision = %d\n", ghost_precision);
      printfQuda("siteSubset = %d\n", siteSubset);
      printfQuda("siteOrder = %d\n", siteOrder);
      printfQuda("fieldOrder = %d\n", fieldOrder);
      printfQuda("gammaBasis = %d\n", gammaBasis);
      printfQuda("create = %d\n", create);
      printfQuda("pc_type = %d\n", pc_type);
      printfQuda("suggested_parity = %d\n", suggested_parity);
      printfQuda("v = %lx\n", (unsigned long)v);
      printfQuda("norm_offset = %lu\n", (unsigned long)norm_offset);
      //! for deflation etc.
      if (is_composite) printfQuda("Number of elements = %d\n", composite_dim);
    }
  };

  struct DslashConstant;

  class ColorSpinorField : public LatticeField
  {
  private:
    /**
       @brief Create the field as specified by the param
       @param[in] Parameter struct
    */
    void create(const ColorSpinorParam &param);

    /**
       @brief Move the contents of a field to this
       @param[in,out] other Field we are moving from
    */
    void move(ColorSpinorField &&other);

    /**
       @brief Destroy the field
    */
    void destroy();

  protected:
    bool init = false;
    bool alloc = false;     // whether we allocated memory
    bool reference = false; // whether the field is a reference or not
    bool ghost_only = false; // whether the field is only a ghost wrapper

    /** Used to keep local track of allocated ghost_precision in createGhostZone */
    mutable QudaPrecision ghost_precision_allocated = QUDA_INVALID_PRECISION;

    int nColor = 0;
    int nSpin = 0;
    int nVec = 0;

    QudaTwistFlavorType twistFlavor = QUDA_TWIST_INVALID;

    QudaPCType pc_type = QUDA_PC_INVALID; // used to select preconditioning method in DWF

    /** Used to specify whether a single parity field is even/odd
     * By construction not enforced, this is more of an optional
     * metadata to specify, for ex, if an eigensolver is for an
     * even or odd parity. */
    QudaParity suggested_parity = QUDA_INVALID_PARITY;

    size_t length = 0; // length including pads, but not norm zone

    void *v = nullptr;      // the field elements
    void *v_h = nullptr;    // the field elements
    size_t norm_offset = 0; /** offset to the norm (if applicable) */

    // multi-GPU parameters
    array_2d<void *, 2, QUDA_MAX_DIM> ghost = {};          // pointers to the ghost regions - NULL by default
    mutable lat_dim_t ghostFace = {};                      // the size of each face
    mutable lat_dim_t ghostFaceCB = {};                    // the size of each checkboarded face
    mutable array<void *, 2 *QUDA_MAX_DIM> ghost_buf = {}; // wrapper that points to current ghost zone

    mutable DslashConstant *dslash_constant = nullptr; // constants used by dslash and packing kernels

    size_t bytes = 0;     // size in bytes of spinor field
    size_t bytes_raw = 0; // actual data size neglecting alignment

    QudaSiteOrder siteOrder = QUDA_INVALID_SITE_ORDER;
    QudaFieldOrder fieldOrder = QUDA_INVALID_FIELD_ORDER;
    QudaGammaBasis gammaBasis = QUDA_INVALID_GAMMA_BASIS;

    // in the case of full fields, these are references to the even / odd sublattices
    ColorSpinorField *even = nullptr;
    ColorSpinorField *odd = nullptr;

    //! used for deflation eigenvector sets etc.:
    CompositeColorSpinorFieldDescriptor composite_descr; // contains info about the set
    //
    CompositeColorSpinorField components;

    /**
       Compute the required extended ghost zone sizes and offsets
       @param[in] nFace The depth of the halo
       @param[in] spin_project Whether we are spin projecting
    */
    void createGhostZone(int nFace, bool spin_project = true) const;

    /**
       @brief Fills the param with this field's meta data (used for
       creating a cloned field)
       @param[in] param The parameter we are filling
    */
    void fill(ColorSpinorParam &) const;

    /**
       @brief Set the vol_string and aux_string for use in tuning
    */
    void setTuningString();

  public:
    inline static void *fwdGhostFaceBuffer[QUDA_MAX_DIM] = {};      // cpu memory
    inline static void *backGhostFaceBuffer[QUDA_MAX_DIM] = {};     // cpu memory
    inline static void *fwdGhostFaceSendBuffer[QUDA_MAX_DIM] = {};  // cpu memory
    inline static void *backGhostFaceSendBuffer[QUDA_MAX_DIM] = {}; // cpu memory
    inline static int initGhostFaceBuffer = 0;
    inline static size_t ghostFaceBytes[QUDA_MAX_DIM] = {};
    static void freeGhostBuffer(void);

    /**
       @brief Default constructor
    */
    ColorSpinorField() = default;

    /**
       @brief Copy constructor for creating a ColorSpinorField from another ColorSpinorField
       @param[in] field Instance of ColorSpinorField from which we are cloning
    */
    ColorSpinorField(const ColorSpinorField &field) noexcept;

    /**
       @brief Move constructor for creating a ColorSpinorField from another ColorSpinorField
       @param[in] field Instance of ColorSpinorField from which we are moving
    */
    ColorSpinorField(ColorSpinorField &&field) noexcept;

    /**
       @brief Constructor for creating a ColorSpinorField from a ColorSpinorParam
       @param param Contains the metadata for creating the field
    */
    ColorSpinorField(const ColorSpinorParam &param);

    /**
       @brief Destructor for ColorSpinorField
    */
    virtual ~ColorSpinorField();

    /**
       @brief Copy assignment operator
       @param[in] field Instance from which we are copying
       @return Reference to this field
     */
    ColorSpinorField &operator=(const ColorSpinorField &field);

    /**
       @brief Move assignment operator
       @param[in] field Instance from which we are moving
       @return Reference to this field
     */
    ColorSpinorField &operator=(ColorSpinorField &&field);

    /**
       @brief Copy the source field contents into this
       @param[in] src Source from which we are copying
     */
    void copy(const ColorSpinorField &src);

    /**
       @brief Zero all elements of this field
     */
    void zero();

    /**
       @brief Zero the padded regions added on to the field.  Ensures
       correct reductions and silences false positive warnings
       regarding uninitialized memory.
     */
    void zeroPad();

    int Ncolor() const { return nColor; }
    int Nspin() const { return nSpin; }
    int Nvec() const { return nVec; }
    QudaTwistFlavorType TwistFlavor() const { return twistFlavor; }
    int Ndim() const { return nDim; }
    const int *X() const { return x.data; }
    int X(int d) const { return x[d]; }
    size_t Length() const { return length; }
    size_t Bytes() const { return bytes; }
    size_t TotalBytes() const { return bytes; }
    size_t GhostBytes() const { return ghost_bytes; }
    size_t GhostFaceBytes(int i) const { return ghost_face_bytes[i]; }
    size_t GhostNormBytes() const { return ghost_bytes; }
    void PrintDims() const { printfQuda("dimensions=%d %d %d %d\n", x[0], x[1], x[2], x[3]); }

    /**
       @brief Return pointer to the field allocation
    */
    void *V()
    {
      if (ghost_only) errorQuda("Not defined for ghost-only field");
      return v;
    }

    /**
       @brief Return pointer to the field allocation
    */
    const void *V() const
    {
      if (ghost_only) errorQuda("Not defined for ghost-only field");
      return v;
    }

    /**
       @brief Return pointer to the norm base pointer in the field allocation
    */
    void *Norm()
    {
      if (ghost_only) errorQuda("Not defined for ghost-only field");
      return static_cast<char *>(v) + norm_offset;
    }

    /**
       @brief Return pointer to the norm base pointer in the field allocation
    */
    const void *Norm() const
    {
      if (ghost_only) errorQuda("Not defined for ghost-only field");
      return static_cast<char *>(v) + norm_offset;
    }

    size_t NormOffset() const { return norm_offset; }

    /**
       @brief Returns the full lattice dimension regardless if this
       field is a subset or not
       @param[in] d Dimension we are querying
       @return The full lattice dimension in dimension d
    */
    int full_dim(int d) const { return (d == 0 && siteSubset == 1) ? x[d] * 2 : x[d]; }

    /**
     * Define the parameter type for this field.
     */
    using param_type = ColorSpinorParam;

    /**
       @brief Allocate the ghost buffers
       @param[in] nFace Depth of each halo
       @param[in] spin_project Whether the halos are spin projected (Wilson-type fermions only)
    */
    void allocateGhostBuffer(int nFace, bool spin_project = true) const;

    /**
       @brief Create the communication handlers and buffers
       @param[in] nFace Depth of each halo
       @param[in] spin_project Whether the halos are spin projected (Wilson-type fermions only)
    */
    void createComms(int nFace, bool spin_project = true);

    /**
       @brief Packs the ColorSpinorField's ghost zone
       @param[in] nFace How many faces to pack (depth)
       @param[in] parity Parity of the field
       @param[in] dagger Whether the operator is the Hermitian conjugate or not
       @param[in] stream Which stream to use for the kernel
       @param[out] buffer Optional parameter where the ghost should be
       stored (default is to use ColorSpinorField::ghostFaceBuffer)
       @param[in] location Are we packing directly into local device memory, zero-copy memory or remote memory
       @param[in] location_label Consistent label used for labeling
       the packing tunekey since location can be difference for each process
       @param[in] spin_project Whether we are spin projecting when face packing
       @param[in] a Twisted mass parameter (scale factor, default=0)
       @param[in] b Twisted mass parameter (flavor twist factor, default=0)
       @param[in] c Twisted mass parameter (chiral twist factor, default=0)
      */
    void packGhost(const int nFace, const QudaParity parity, const int dagger, const qudaStream_t &stream,
                   MemoryLocation location[2 * QUDA_MAX_DIM], MemoryLocation location_label, bool spin_project,
                   double a = 0, double b = 0, double c = 0, int shmem = 0);

    /**
       Pack the field halos in preparation for halo exchange, e.g., for Dslash
       @param[in] nFace Depth of faces
       @param[in] parity Field parity
       @param[in] dagger Whether this exchange is for the conjugate operator
       @param[in] stream Stream to be used for packing kernel
       @param[in] location Array of field locations where each halo
       will be sent (Host, Device or Remote)
       @param[in] location_label Consistent label used for labeling
       the packing tunekey since location can be difference for each
       process
       @param[in] spin_project Whether we are spin projecting when face packing
       @param[in] a Used for twisted mass (scale factor)
       @param[in] b Used for twisted mass (chiral twist factor)
       @param[in] c Used for twisted mass (flavor twist factor)
    */
    void pack(int nFace, int parity, int dagger, const qudaStream_t &stream, MemoryLocation location[2 * QUDA_MAX_DIM],
              MemoryLocation location_label, bool spin_project = true, double a = 0, double b = 0, double c = 0,
              int shmem = 0);

    /**
      @brief Initiate the gpu to cpu send of the ghost zone (halo)
      @param ghost_spinor Where to send the ghost zone
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param stream The array of streams to use
      */
    void sendGhost(void *ghost_spinor, const int dim, const QudaDirection dir, const qudaStream_t &stream);

    /**
      Initiate the cpu to gpu send of the ghost zone (halo)
      @param ghost_spinor Source of the ghost zone
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param stream The array of streams to use
      */
    void unpackGhost(const void *ghost_spinor, const int dim, const QudaDirection dir, const qudaStream_t &stream);

    /**
       @brief Copies the ghost to the host from the device, prior to
       communication.
       @param[in] d d=[2*dim+dir], where dim is dimension and dir is
       the scatter-centric direction (0=backwards,1=forwards)
       @param[in] stream The stream in which to do the copy
     */
    void gather(int dir, const qudaStream_t &stream);

    /**
       @brief Initiate halo communication receive
       @param[in] d d=[2*dim+dir], where dim is dimension and dir is
       the scatter-centric direction (0=backwards,1=forwards)
       @param[in] gdr Whether we are using GDR on the receive side
    */
    void recvStart(int dir, const qudaStream_t &stream, bool gdr = false);

    /**
       @brief Initiate halo communication sending
       @param[in] d d=[2*dim+dir], where dim is dimension and dir is
       the scatter-centric direction (0=backwards,1=forwards)
       @param[in] stream_idx The stream in which to do the copy.  If
       -1 is passed then the copy will be issied to the d^th stream
       @param[in] gdr Whether we are using GDR on the send side
       @param[in] remote_write Whether we are writing direct to remote memory (or using copy engines)
    */
    void sendStart(int d, const qudaStream_t &stream, bool gdr = false, bool remote_write = false);

    /**
       @brief Initiate halo communication
       @param[in] d d=[2*dim+dir], where dim is dimension and dir is
       the scatter-centric direction (0=backwards,1=forwards)
       @param[in] stream (presently unused)
       @param[in] gdr_send Whether we are using GDR on the send side
       @param[in] gdr_recv Whether we are using GDR on the receive side
    */
    void commsStart(int d, const qudaStream_t &stream, bool gdr_send = false, bool gdr_recv = false);

    /**
       @brief Non-blocking query if the halo communication has completed
       @param[in] d d=[2*dim+dir], where dim is dimension and dir is
       the scatter-centric direction (0=backwards,1=forwards)
       @param[in] stream (presently unused)
       @param[in] gdr_send Whether we are using GDR on the send side
       @param[in] gdr_recv Whether we are using GDR on the receive side
    */
    int commsQuery(int d, const qudaStream_t &stream, bool gdr_send = false, bool gdr_recv = false);

    /**
       @brief Wait on halo communication to complete
       @param[in] d d=[2*dim+dir], where dim is dimension and dir is
       the scatter-centric direction (0=backwards,1=forwards)
       @param[in] stream (unused)
       @param[in] gdr_send Whether we are using GDR on the send side
       @param[in] gdr_recv Whether we are using GDR on the receive side
    */
    void commsWait(int d, const qudaStream_t &stream, bool gdr_send = false, bool gdr_recv = false);

    /**
       @brief Unpacks the ghost from host to device after
       communication has finished.
       @param[in] d d=[2*dim+dir], where dim is dimension and dir is
       the scatter-centric direction (0=backwards,1=forwards)
       @param[in] stream The stream in which to do the copy.  If
       -1 is passed then the copy will be issied to the d^th stream
     */
    void scatter(int d, const qudaStream_t &stream);

    /**
       Do the exchange between neighbouring nodes of the data in
       sendbuf storing the result in recvbuf.  The arrays are ordered
       (2*dim + dir).
       @param recvbuf Packed buffer where we store the result
       @param sendbuf Packed buffer from which we're sending
       @param nFace Number of layers we are exchanging
     */
    void exchange(void **ghost, void **sendbuf, int nFace = 1) const;

    /**
       This is a unified ghost exchange function for doing a complete
       halo exchange regardless of the type of field.  All dimensions
       are exchanged and no spin projection is done in the case of
       Wilson fermions.
       @param[in] parity Field parity
       @param[in] nFace Depth of halo exchange
       @param[in] dagger Is this for a dagger operator (only relevant for spin projected Wilson)
       @param[in] pack_destination Destination of the packing buffer
       @param[in] halo_location Destination of the halo reading buffer
       @param[in] gdr_send Are we using GDR for sending
       @param[in] gdr_recv Are we using GDR for receiving
       @param[in] ghost_precision The precision used for the ghost exchange
       @param[in] shmem The type of shmem communication (if applicable)
       @param[in] v Vector of fields to be used for batched exchange
     */
    void exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination = nullptr,
                       const MemoryLocation *halo_location = nullptr, bool gdr_send = false, bool gdr_recv = false,
                       QudaPrecision ghost_precision = QUDA_INVALID_PRECISION, int shmem = 0,
                       cvector_ref<const ColorSpinorField> v = {}) const;

    /**
      This function returns true if the field is stored in an internal
      field order, given the precision and the length of the spin
      dimension.
      */
    bool isNative() const { return colorspinor::isNative(fieldOrder, precision, nSpin, nColor); }

    bool IsComposite() const { return composite_descr.is_composite; }
    bool IsComponent() const { return composite_descr.is_component; }

    int CompositeDim() const { return composite_descr.dim; }
    int ComponentId() const { return composite_descr.id; }
    int ComponentVolume() const { return composite_descr.volume; }
    int ComponentVolumeCB() const { return composite_descr.volumeCB; }
    size_t ComponentLength() const { return composite_descr.length; }

    size_t ComponentBytes() const { return composite_descr.bytes; }

    QudaPCType PCType() const { return pc_type; }
    QudaParity SuggestedParity() const { return suggested_parity; }
    void setSuggestedParity(QudaParity suggested_parity) { this->suggested_parity = suggested_parity; }

    QudaSiteSubset SiteSubset() const { return siteSubset; }
    QudaSiteOrder SiteOrder() const { return siteOrder; }
    QudaFieldOrder FieldOrder() const { return fieldOrder; }
    QudaGammaBasis GammaBasis() const { return gammaBasis; }

    const int *GhostFace() const { return ghostFace.data; }
    const int *GhostFaceCB() const { return ghostFaceCB.data; }

    /**
       Return the offset in bytes to the start of the ghost zone in a
       given dimension and direction
       @param[in] dim The dimension of the ghost
       @param[in] dir The direction of the ghost
     */
    size_t GhostOffset(const int dim, const int dir) const { return ghost_offset[dim][dir]; }

    const void *Ghost2() const;

    /**
       Return array of pointers to the ghost zones (ordering dim*2+dir)
     */
    void *const *Ghost() const;

    /**
       @brief Get the dslash_constant structure from this field
    */
    const DslashConstant &getDslashConstant() const { return *dslash_constant; }

    const ColorSpinorField &Even() const;
    const ColorSpinorField &Odd() const;

    ColorSpinorField &Even();
    ColorSpinorField &Odd();

    CompositeColorSpinorField &Components() { return components; };

    /**
       @brief Return the idx^th component of the composite field.  An
       error will be thrown if the field is not a composite field, or
       if an out of bounds idx is requested.
       @param[in] idx Component index
       @return Component reference
    */
    ColorSpinorField &Component(int idx);

    /**
       @brief Return the idx^th component of the composite field.  An
       error will be thrown if the field is not a composite field, or
       if an out of bounds idx is requested.
       @param[in] idx Component index
       @return Component const reference
    */
    const ColorSpinorField &Component(int idx) const;

    /**
     * Compute the n-dimensional site index given the 1-d offset index
     * @param y n-dimensional site index
     * @param i 1-dimensional site index
     */
    void LatticeIndex(int *y, int i) const;

    /**
     * Compute the 1-d offset index given the n-dimensional site index
     * @param i 1-dimensional site index
     * @param y n-dimensional site index
     */
    void OffsetIndex(int &i, int *y) const;

    static ColorSpinorField *Create(const ColorSpinorParam &param) { return new ColorSpinorField(param); }

    /**
      @brief Create a dummy field used for batched communication
      @param[in] v Vector of fields we which to batch together
      @return Dummy (nDim+1)-dimensional field
     */
    static FieldTmp<ColorSpinorField> create_comms_batch(cvector_ref<const ColorSpinorField> &v);

    /**
       @brief Create a field that aliases this field's storage.  The
       alias field can use a different precision than this field,
       though it cannot be greater.  This functionality is useful for
       the case where we have multiple temporaries in different
       precisions, but do not need them simultaneously.  Use this functionality with caution.
       @param[in] param Parameters for the alias field
    */
    ColorSpinorField create_alias(const ColorSpinorParam &param = ColorSpinorParam());

    /**
       @brief Create a field that aliases this field's storage.  The
       alias field can use a different precision than this field,
       though it cannot be greater.  This functionality is useful for
       the case where we have multiple temporaries in different
       precisions, but do not need them simultaneously.  Use this functionality with caution.
       @param[in] param Parameters for the alias field
    */
    ColorSpinorField *CreateAlias(const ColorSpinorParam &param);

    /**
       @brief Create a coarse color-spinor field, using this field to set the meta data
       @param[in] geoBlockSize Geometric block size that defines the coarse grid dimensions
       @param[in] spinlockSize Geometric block size that defines the coarse spin dimension
       @param[in] Nvec Number of coarse color degrees of freedom per grid point
       @param[in] precision Optionally set the precision of the fine field
       @param[in] location Optionally set the location of the coarse field
       @param[in] mem_type Optionally set the memory type used (e.g., can override with mapped memory)
    */
    ColorSpinorField *CreateCoarse(const int *geoBlockSize, int spinBlockSize, int Nvec,
                                   QudaPrecision precision = QUDA_INVALID_PRECISION,
                                   QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION,
                                   QudaMemoryType mem_Type = QUDA_MEMORY_INVALID);

    /**
       @brief Create a fine color-spinor field, using this field to set the meta data
       @param[in] geoBlockSize Geometric block size that defines the fine grid dimensions
       @param[in] spinlockSize Geometric block size that defines the fine spin dimension
       @param[in] Nvec Number of fine color degrees of freedom per grid point
       @param[in] precision Optionally set the precision of the fine field
       @param[in] location Optionally set the location of the fine field
       @param[in] mem_type Optionally set the memory type used (e.g., can override with mapped memory)
    */
    ColorSpinorField *CreateFine(const int *geoblockSize, int spinBlockSize, int Nvec,
                                 QudaPrecision precision = QUDA_INVALID_PRECISION,
                                 QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION,
                                 QudaMemoryType mem_type = QUDA_MEMORY_INVALID);

    /**
       @brief Backs up the ColorSpinorField
    */
    void backup() const;

    /**
       @brief Restores the ColorSpinorField
    */
    void restore() const;

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

    /**
      @brief If managed memory and prefetch is enabled, prefetch
      the spinor, to the CPU or the GPU
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in (default 0)
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream = device::get_default_stream()) const;

    /**
       @brief Fill the field with a defined source type
       @param[in] sourceType The type of source
       @param[in] x local site index
       @param[in] s spin index
       @param[in] c color index
    */
    void Source(QudaSourceType sourceType, unsigned int x = 0, int s = 0, int c = 0);

    /**
     * @brief Print the site vector
     * @param[in] a The field we are printing from
     * @param[in] parity Parity index
     * @param[in] x_cb Checkerboard space-time index
     * @param[in] rank The rank we are requesting from (default is rank = 0)
     */
    void PrintVector(int parity, unsigned int x_cb, int rank = 0) const;

    /**
       @brief Perform a component by component comparison of two
       color-spinor fields.  In doing we normalize with respect to the
       first colorspinor field, e.g., we compare || a_i - b_i || / || a ||
       @param[in] a Ground truth color spinor field
       @param[in] b Field we are checking

       @param[in] resolution How many bins per order of magnitude to
       use.  The default resolution=1 means that we have 16 bins
       covering the range [1e-15,1.0].
     */
    static int Compare(const ColorSpinorField &a, const ColorSpinorField &b, const int resolution = 1);

    /**
       @brief Check if two instances are compatible
       @param[in] a Input field
       @param[in] b Input field
       @return Return true if two fields are compatible
     */
    static bool are_compatible(const ColorSpinorField &a, const ColorSpinorField &b);

    /**
       @brief Check if two instances are weakly compatible (precision
       and order can differ)
       @param[in] a Input field
       @param[in] b Input field
       @return Return true if two fields are compatible
     */
    static bool are_compatible_weak(const ColorSpinorField &a, const ColorSpinorField &b);

    /**
       @brief Test if two instances are compatible.  Throws an error
       if test fails.
       @param[in] a Input field
       @param[in] b Input field
     */
    static void test_compatible(const ColorSpinorField &a, const ColorSpinorField &b);

    /**
       @brief Test if two instances are weakly compatible (precision
       and order can differ).  Throws an error if test fails.
       @param[in] a Input field
       @param[in] b Input field
     */
    static void test_compatible_weak(const ColorSpinorField &a, const ColorSpinorField &b);

    friend std::ostream &operator<<(std::ostream &out, const ColorSpinorField &);
    friend class ColorSpinorParam;
  };

  /**
     @brief Specialization of is_field to allow us to make sets of ColorSpinorField
   */
  template <> struct is_field<ColorSpinorField> : std::true_type {
  };

  /**
     @brief Helper function to resize a std::vector of
     ColorSpinorFields.  This should be favored over using
     std::vector::resize, since it avoids unnecessary copies.

     @param[in,out] v The vector we are resizing
     @param[in] new_size The size we are resizing the vector to
     @param[in] param The parameter struct used to create the new
     elements
   */
  void resize(std::vector<ColorSpinorField> &v, size_t new_size, const ColorSpinorParam &param);

  /**
     @brief Helper function to resize a std::vector of
     ColorSpinorFields.  This should be favored over using
     std::vector::resize, since it avoids unnecessary copies.  If no
     src vector is passed, the meta data for the newly constructed
     fields will be sourced from element 0.

     @param[in,out] v The vector we are resizing
     @param[in] new_size The size we are resizing the vector to
     @param[in] create The create type we using for the field
     @param[in] src Any src vector from which we are copying from,
     referencing to or obtaining any meta data from
   */
  void resize(std::vector<ColorSpinorField> &v, size_t new_size, QudaFieldCreate create,
              const ColorSpinorField &src = ColorSpinorField());

  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, QudaFieldLocation location,
                              void *Dst = nullptr, const void *Src = nullptr);

  void genericSource(ColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c);
  int genericCompare(const ColorSpinorField &a, const ColorSpinorField &b, int tol);

  /**
    @brief This function is used for copying from a source colorspinor field to a destination field
      with an offset.
    @param out The output field to which we are copying
    @param in The input field from which we are copying
    @param offset The offset for the larger field between out and in.
    @param pc_type Whether the field order uses 4d or 5d even-odd preconditioning.
  */
  void copyFieldOffset(ColorSpinorField &out, const ColorSpinorField &in, CommKey offset, QudaPCType pc_type);

  /**
     @brief Print the value of the field at the requested coordinates
     @param[in] a The field we are printing from
     @param[in] parity Parity index
     @param[in] x_cb Checkerboard space-time index
     @param[in] rank The rank we are requesting from (default is rank = 0)
  */
  void genericPrintVector(const ColorSpinorField &a, int parity, unsigned int x_cb, int rank = 0);

  /**
     @brief Generic ghost packing routine

     @param[out] ghost Array of packed ghosts with array ordering [2*dim+dir]
     @param[in] a Input field that is being packed
     @param[in] parity Which parity are we packing
     @param[in] nFace The depth of the face in each dimension and direction
     @param[in] dagger Is for a dagger operator (presently ignored)
     @param[in] destination Array specifying the memory location of each resulting ghost [2*dim+dir]
     @param[in] shmem The shmem type to use
     @param[in] v Vector fields to batch into ghost (if v.size() > 0)
  */
  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity, int nFace, int dagger,
                        MemoryLocation *destination = nullptr, int shmem = 0, cvector_ref<const ColorSpinorField> v = {});

  /**
     @brief pre-declaration of RNG class (defined in non-device-safe random_quda.h)
  */
  class RNG;

  /**
     @brief Generate a random noise spinor.  This variant allows the user to manage the RNG state.
     @param src The colorspinorfield
     @param randstates Random state
     @param type The type of noise to create (QUDA_NOISE_GAUSSIAN or QUDA_NOISE_UNIFORM)
  */
  void spinorNoise(ColorSpinorField &src, RNG &randstates, QudaNoiseType type);

  /**
     @brief Generate a random noise spinor.  This variant just
     requires a seed and will create and destroy the random number state.
     @param src The colorspinorfield
     @param seed Seed
     @param type The type of noise to create (QUDA_NOISE_GAUSSIAN or QUDA_NOISE_UNIFORM)
  */
  void spinorNoise(ColorSpinorField &src, unsigned long long seed, QudaNoiseType type);

  /**
     @brief Generate a set of diluted color spinors from a single source.
     @param v Diluted vector set
     @param src The input source
     @param type The type of dilution to apply (QUDA_DILUTION_SPIN_COLOR, etc.)
  */
  void spinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type);

  /**
     @brief Helper function for determining if the preconditioning
     type of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If PCType is unique return this
   */
  inline QudaPCType PCType_(const char *func, const char *file, int line, const ColorSpinorField &a,
                            const ColorSpinorField &b)
  {
    QudaPCType type = QUDA_PC_INVALID;
    if (a.PCType() == b.PCType())
      type = a.PCType();
    else
      errorQuda("PCTypes %d %d do not match (%s:%d in %s())\n", a.PCType(), b.PCType(), file, line, func);
    return type;
  }

  /**
     @brief Helper function for determining if the precision of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check precision on
     @return If precision is unique return the precision
   */
  template <typename... Args>
  inline QudaPCType PCType_(const char *func, const char *file, int line, const ColorSpinorField &a,
                            const ColorSpinorField &b, const Args &...args)
  {
    return static_cast<QudaPCType>(PCType_(func, file, line, a, b) & PCType_(func, file, line, a, args...));
  }

#define checkPCType(...) PCType_(__func__, __FILE__, __LINE__, __VA_ARGS__)

  /**
     @brief Helper function for determining if the order of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If order is unique return the order
   */
  inline QudaFieldOrder Order_(const char *func, const char *file, int line, const ColorSpinorField &a,
                               const ColorSpinorField &b)
  {
    QudaFieldOrder order = QUDA_INVALID_FIELD_ORDER;
    if (a.FieldOrder() == b.FieldOrder())
      order = a.FieldOrder();
    else
      errorQuda("Orders %d %d do not match  (%s:%d in %s())\n", a.FieldOrder(), b.FieldOrder(), file, line, func);
    return order;
  }

  /**
     @brief Helper function for determining if the order of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check order on
     @return If order is unique return the order
   */
  template <typename... Args>
  inline QudaFieldOrder Order_(const char *func, const char *file, int line, const ColorSpinorField &a,
                               const ColorSpinorField &b, const Args &...args)
  {
    return static_cast<QudaFieldOrder>(Order_(func, file, line, a, b) & Order_(func, file, line, a, args...));
  }

#define checkOrder(...) Order_(__func__, __FILE__, __LINE__, __VA_ARGS__)

  /**
     @brief Helper function for determining if the length of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If length is unique return the length
   */
  inline int Length_(const char *func, const char *file, int line, const ColorSpinorField &a, const ColorSpinorField &b)
  {
    int length = 0;
    if (a.Length() == b.Length())
      length = a.Length();
    else
      errorQuda("Lengths %lu %lu do not match  (%s:%d in %s())\n", a.Length(), b.Length(), file, line, func);
    return length;
  }

  /**
     @brief Helper function for determining if the length of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check length on
     @return If length is unique return the length
   */
  template <typename... Args>
  inline int Length_(const char *func, const char *file, int line, const ColorSpinorField &a, const ColorSpinorField &b,
                     const Args &...args)
  {
    return static_cast<int>(Length_(func, file, line, a, b) & Length_(func, file, line, a, args...));
  }

#define checkLength(...) Length_(__func__, __FILE__, __LINE__, __VA_ARGS__)

} // namespace quda

#ifndef _TRANSFER_H
#define _TRANSFER_H

/**
 * @file transfer.h
 *
 * @section DESCRIPTION
 *
 * Defines the prolongation and restriction operators used to transfer
 * between grids.
 */

#include <color_spinor_field.h>
#include <vector>
#include <reference_wrapper_helper.h>

namespace quda {

  /**
     The transfer class defines the inter-grid operators that connect
     fine and coarse grids.  This implements both restriction and
     prologation methods.  The transfer operator is fully defined at
     object creation time, and defined by the null-space vectors and
     the coarsening pattern that are passed to it.

     At present only the restriction (R) and prolongation (P) methods
     have been offloaded to run on the GPU, with the block
     orthogonlization yet to be offloaded.
   */
  class Transfer {

  protected:

    /** The raw null space components */
    const std::vector<ColorSpinorField> &B;

    /** The number of null space components */
    const int Nvec;

    /** Precision to use for the GPU null-space components */
    const QudaPrecision null_precision;

    /** Block-normalized null-space components that define the prolongator */
    mutable ColorSpinorField V;

    /** The geometrical coase grid blocking */
    int *geo_bs = nullptr;

    /** The mapping onto coarse sites from fine sites.  This has
	length equal to the fine-grid volume, and is sorted into
	lexicographical fine-grid order, with each value corresponding
	to a coarse-grid offset. (CPU) */
    mutable int *fine_to_coarse_h = nullptr;

    /** The mapping onto fine sites from coarse sites. This has length
	equal to the fine-grid volume, and is sorted into lexicographical
	block order, with each value corresponding to a fine-grid offset. (CPU) */
    mutable int *coarse_to_fine_h = nullptr;

    /** The mapping onto coarse sites from fine sites.  This has
	length equal to the fine-grid volume, and is sorted into
	lexicographical fine-grid order, with each value corresponding
	to a coarse-grid offset. (GPU) */
    mutable int *fine_to_coarse_d = nullptr;

    /** The mapping onto fine sites from coarse sites. This has length
	equal to the fine-grid volume, and is sorted into lexicographical
	block order, with each value corresponding to a fine-grid offset. (GPU) */
    mutable int *coarse_to_fine_d = nullptr;

    /** The spin blocking. Defined as zero when the fine operator is staggered. */
    int spin_bs;

    /** The mapping onto coarse spin from fine spin (inner) and fine parity (outer), for staggered */
    int **spin_map = nullptr;

    /** Whether the transfer operator is to be applied to full fields or single parity fields */
    QudaSiteSubset site_subset = QUDA_FULL_SITE_SUBSET;

    /** The parity of any single-parity fine-grid fields that are passed into the transfer operator */
    QudaParity parity = QUDA_INVALID_PARITY;

    /** ColorSpinorParam corresponding to the fine-grid field */
    ColorSpinorParam fine_param;

    /** ColorSpinorParam corresponding to the coarse-grid field */
    ColorSpinorParam coarse_param;

    /**
     * @brief Creates the map between fine and coarse grids
     */
    void createGeoMap();

    /**
     * @brief Creates the map between fine spin and parity to coarse spin dimensions
     * @param n_fine_spin The number of fine spins
     */
    void createSpinMap(int n_fine_spin);

  public:
    /**
     * The constructor for Transfer
     * @param B Array of null-space vectors
     * @param Nvec Number of null-space vectors
     * @param spin_bs The spin block sizes to use
     * @param null_precision The precision to store the null-space basis vectors in
     */
    Transfer(const std::vector<ColorSpinorField> &B, int Nvec, int spin_bs, QudaPrecision null_precision);

    /** The destructor for Transfer */
    ~Transfer();

    /**
     @brief for resetting the Transfer when the null vectors have changed
     */
    virtual void reset() {};

    /**
     * Apply the prolongator
     * @param out The resulting field on the fine lattice
     * @param in The input field on the coarse lattice
     */
    virtual void P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const = 0;

    /**
     * Apply the restrictor
     * @param out The resulting field on the coarse lattice
     * @param in The input field on the fine lattice
     */
    virtual void R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const = 0;

    /**
     * @brief The precision of the packed null-space vectors
     */
    QudaPrecision NullPrecision() const { return null_precision; }

    /**
     * Returns a const reference to the V field
     * @param location Which memory space are we requesting
     * @return The V field const reference
     */
    const ColorSpinorField &Vectors() const { return V; }

    /**
     * Returns the number of near nullvectors
     * @return Nvec
     */
    int nvec() const {return Nvec;}

    /**
     * Returns the number of fine spins
     * @return fine_nspin
     */
    int fineNspin() const { return fine_param.nSpin; }

    /**
     * Returns the number of coarse spins
     * @return coarse_nspin
     */
    int coarseNspin() const { return coarse_param.nSpin; }

    /**
     * Returns the number of fine colors
     * @return fine_ncolor
     */
    int fineNcolor() const { return fine_param.nColor; }

    /**
     * Returns the number of coarse colors
     * @return coarse_ncolor
     */
    int coarseNcolor() const { return coarse_param.nColor; }

    /**
     * Returns the number of fine dimensions
     * @return fine_gamma_basis
     */
    int fineGammaBasis() const { return fine_param.gammaBasis; }

    /**
     * Returns the number of coarse dimensions
     * @return coarse_gamma_basis
     */
    int coarseGammaBasis() const { return coarse_param.gammaBasis; }

    /**
     * Returns the site subset of the fine field
     * @return fine_site_subset
     */
    QudaSiteSubset fineSiteSubset() const { return fine_param.siteSubset; }

    /**
     * Returns the site subset of the coarse field
     * @return coarse_site_subset
     */
    QudaSiteSubset coarseSiteSubset() const { return coarse_param.siteSubset; }

    /**
     * Verifies the compatibility of the fine field with the transfer operator
     * @param fine The fine field
     */
     void verifyFineCompatibility(const ColorSpinorField &fine) const;

    /**
     * Verifies the compatibility of the coarse field with the transfer operator
     * @param coarse The coarse field
     */
    void verifyCoarseCompatibility(const ColorSpinorField &coarse) const;

    /**
     * Returns the amount of spin blocking. Defined as zero when coarsening staggered.
     * @return spin_bs
     */
    int Spin_bs() const {return spin_bs;}

    /**
     * Returns the geometrical coarse grid blocking
     * @return geo_bs
     */
    const int *Geo_bs() const {return geo_bs;}

    /**
     * Returns the transfer type; used to inform staggered-type coarsenings
     * @return transfer_type
     */
    virtual QudaTransferType getTransferType() const = 0;

    /**
       @return Pointer to the lookup table to the fine-to-coarse map
    */
    const int* fineToCoarse(QudaFieldLocation location=QUDA_CPU_FIELD_LOCATION) const
    { return location == QUDA_CPU_FIELD_LOCATION ? fine_to_coarse_h : fine_to_coarse_d; }

    /**
       @return Pointer to the lookup table to the coarse-to-fine map
    */
    const int* coarseToFine(QudaFieldLocation location=QUDA_CPU_FIELD_LOCATION) const
    { return location == QUDA_CPU_FIELD_LOCATION ? coarse_to_fine_h : coarse_to_fine_d; }

    /**
     * @brief Sets whether the transfer operator is to act on full
     * fields or single parity fields, and if single-parity which
     * parity.
     * @param[in] site_subset The site_subset of the fine-grid fields
     * @param[in] parity The parity of the single-parity fields (if
     * applicable)
     */
    void setSiteSubset(QudaSiteSubset site_subset, QudaParity parity);

    /**
     * @brief Returns the ColorSpinorParam for the fine-grid field
     * @param precision Optionally sets the precision of the fine-grid field
     * @param new_location Optionally sets the location of the fine-grid field
     * @param new_mem_type Optionally sets the memory type of the fine-grid field
     * @return The ColorSpinorParam for the fine-grid field
     */
    ColorSpinorParam fineColorSpinorParam(QudaPrecision precision,
      QudaFieldLocation new_location = QUDA_INVALID_FIELD_LOCATION,
      QudaMemoryType new_mem_type = QUDA_MEMORY_INVALID) const;

    /**
     * @brief Returns the ColorSpinorParam for the coarse-grid field
     * @param precision Optionally sets the precision of the coarse-grid field
     * @param new_location Optionally sets the location of the coarse-grid field
     * @param new_mem_type Optionally sets the memory type of the coarse-grid field
     * @return The ColorSpinorParam for the coarse-grid field
     */
    ColorSpinorParam coarseColorSpinorParam(QudaPrecision precision,
      QudaFieldLocation new_location = QUDA_INVALID_FIELD_LOCATION,
      QudaMemoryType new_mem_type = QUDA_MEMORY_INVALID) const;
  };

  class TransferAggregate : public Transfer {

  protected:

    /** The number of times to Gram-Schmidt within block ortho */
    const int NblockOrtho;

    /** Whether we are doing the two-pass Block Orthogonalize */
    const bool blockOrthoTwoPass;

    /** Whether to apply the transfer operation with MMA */
    bool use_mma;

    /**
      * @brief Initialize the block sizes
      */
    void initializeBlockSizes(int *geo_bs);

    /**
      * @brief Allocate V field
      * @param[in] location Where to allocate the V field
      */
    void createV() const;

    /**
      * @brief Creates the ColorSpinorParam objects for the fine and coarse fields
      */
    void createColorSpinorParams();

  public:
    /**
      * The constructor for TransferAggregate
      * @param B Array of null-space vectors
      * @param Nvec Number of null-space vectors
      * @param NblockOrtho Number of times to Gram-Schmidt within block ortho
      * @param blockOrthoTwoPass Whether to do a two pass block orthogonalization
      * @param geo_bs The geometric block sizes to use
      * @param spin_bs The spin block sizes to use
      * @param null_precision The precision to store the null-space basis vectors in
      * @param use_mma Whether to apply the transfer operation with MMA
      */
    TransferAggregate(const std::vector<ColorSpinorField> &B, int Nvec, int NblockOrtho, bool blockOrthoTwoPass, int *geo_bs,
              int spin_bs, QudaPrecision null_precision, bool use_mma);

    /**
      @brief for resetting the Transfer when the null vectors have changed
      */
    void reset() override;

    /**
      * Apply the prolongator
      * @param out The resulting field on the fine lattice
      * @param in The input field on the coarse lattice
      */
    virtual void P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const override;

    /**
      * Apply the restrictor
      * @param out The resulting field on the coarse lattice
      * @param in The input field on the fine lattice
      */
    virtual void R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const override;

    /**
      * Returns the transfer type; used to inform staggered-type coarsenings
      * @return transfer_type
      */
    virtual QudaTransferType getTransferType() const override { return QUDA_TRANSFER_AGGREGATE; }

  };

  class TransferCopy : public Transfer {

  protected:

    /** Implies whether or not the fine level is a staggered operator, in which
    case we don't actually need to allocate any memory. */
    mutable QudaTransferType transfer_type;

    /**
      * @brief Initialize the block sizes
      */
    void initializeBlockSizes(int *geo_bs);

    /**
      * @brief Allocate V field
      * @param[in] location Where to allocate the V field
      */
    void createV() const;

    /**
      * @brief Creates the ColorSpinorParam objects for the fine and coarse fields
      */
    void createColorSpinorParams();

  public:
    /**
      * The constructor for Transfer
      * @param B Array of null-space vectors
      * @param Nvec Number of null-space vectors
      * @param geo_bs The geometric block sizes to use
      * @param spin_bs The spin block sizes to use
      * @param null_precision The precision to store the null-space basis vectors in
      * @param transfer_type The type of transfer operator to use
      */
    TransferCopy(const std::vector<ColorSpinorField> &B, int Nvec, int *geo_bs,
              int spin_bs, QudaPrecision null_precision, const QudaTransferType transfer_type);

    /**
      * Apply the prolongator
      * @param out The resulting field on the fine lattice
      * @param in The input field on the coarse lattice
      */
    virtual void P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const override;

    /**
      * Apply the restrictor
      * @param out The resulting field on the coarse lattice
      * @param in The input field on the fine lattice
      */
    virtual void R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const override;

    /**
      * Returns the transfer type; used to inform staggered-type coarsenings
      * @return transfer_type
      */
    virtual QudaTransferType getTransferType() const override { return transfer_type; }

  };

  class TransferCoarseKD : public Transfer {

  protected:

    /**
      * @brief Initialize the block sizes
      */
    void initializeBlockSizes(int *geo_bs);

    /**
      * @brief Allocate V field
      * @param[in] location Where to allocate the V field
      */
    void createV() const;

    /**
      * @brief Creates the ColorSpinorParam objects for the fine and coarse fields
      */
    void createColorSpinorParams();

  public:
    /**
      * The constructor for Transfer
      * @param B Array of null-space vectors
      * @param Nvec Number of null-space vectors
      * @param geo_bs The geometric block sizes to use
      * @param spin_bs The spin block sizes to use
      * @param null_precision The precision to store the null-space basis vectors in
      */
    TransferCoarseKD(const std::vector<ColorSpinorField> &B, int Nvec, int *geo_bs,
              int spin_bs, QudaPrecision null_precision);

    /**
      * Apply the prolongator
      * @param out The resulting field on the fine lattice
      * @param in The input field on the coarse lattice
      */
    virtual void P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const override;

    /**
      * Apply the restrictor
      * @param out The resulting field on the coarse lattice
      * @param in The input field on the fine lattice
      */
    virtual void R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const override;

    /**
      * Returns the transfer type; used to inform staggered-type coarsenings
      * @return transfer_type
      */
    virtual QudaTransferType getTransferType() const override { return QUDA_TRANSFER_COARSE_KD; }
  };

  /**
     @brief Block orthogonnalize the matrix field, where the blocks are
     defined by lookup tables that map the fine grid points to the
     coarse grid points, and similarly for the spin degrees of
     freedom.
     @param[in,out] V Matrix field to be orthgonalized
     @param[in] B input vectors
     @param[in] geo_bs Geometric block size
     @param[in] fine_to_coarse Fine-to-coarse lookup table (linear indices)
     @param[in] coarse_to_fine Coarse-to-fine lookup table (linear indices)
     @param[in] spin_bs Spin block size
     @param[in] n_block_ortho Number of times to Gram-Schmidt
     @param[in] two_pass Whether we use a two-pass algorithm: first
     pass is a dummy run to set the scale, second does the final
     calculation.  This this provides better accuracy in fixed-point
     precision.
   */
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass);

  template <int coarseColor, int fineColor>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, int spin_bs, int n_block_ortho, bool two_pass);

  /**
     @brief Transpose the B vectors into a composite V field:
       - B: nVec -> spatial/N -> spin/color -> N, where N is for that in floatN
       - V: spatial -> spin/color -> nVec
     @param[out] The output V Matrix field
     @param[in] B input vectors
     @param[in] from_non_rel whether or not transform B from non-relativistic basis
   */
  void BlockTransposeForward(ColorSpinorField &V, const cvector_ref<const ColorSpinorField> &B,
                             bool from_non_rel = false);

  /**
     @brief Transpose the a composite V field into B vectors:
       - B: nVec -> spatial/N -> spin/color -> N, where N is for that in floatN
       - V: spatial -> spin/color -> nVec
     @param[in] The output V Matrix field
     @param[out] B input vectors
     @param[in] from_non_rel whether or not transform B to non-reletivistic basis
   */
  void BlockTransposeBackward(const ColorSpinorField &V, const cvector_ref<ColorSpinorField> &B, bool to_non_rel = false);

  /**
     @brief Apply the prolongation operator
     @param[out] out Resulting fine grid field
     @param[in] in Input field on coarse grid
     @param[in] v Matrix field containing the null-space components
     @param[in] fine_to_coarse Fine-to-coarse lookup table (linear indices)
     @param[in] spin_map Spin blocking lookup table
     @param[in] parity of the output fine field (if single parity output field)
   */
  void Prolongate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int *const *spin_map, bool use_mma, int parity = QUDA_INVALID_PARITY);

  template <int coarseColor, int fineColor>
  void Prolongate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int *const *spin_map, int parity = QUDA_INVALID_PARITY);

  template <int fineColor, int coarseColor, int nVec>
  void ProlongateMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                     const int *fine_to_coarse, const int *const *spin_map, int parity);

  /**
     @brief Apply the restriction operator
     @param[out] out Resulting coarsened field
     @param[in] in Input field on fine grid
     @param[in] v Matrix field containing the null-space components
     @param[in] Nvec Number of null-space components
     @param[in] fine_to_coarse Fine-to-coarse lookup table (linear indices)
     @param[in] spin_map Spin blocking lookup table
     @param[in] parity of the input fine field (if single parity input field)
   */
  void Restrict(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, bool use_mma,
                int parity = QUDA_INVALID_PARITY);

  template <int coarseColor, int fineColor>
  void Restrict(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity = QUDA_INVALID_PARITY);

  template <int coarseColor, int fineColor, int nVec>
  void RestrictMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map,
                   int parity = QUDA_INVALID_PARITY);

  /**
     @brief Apply the unitary "prolongation" operator for Kahler-Dirac preconditioning
     @param[out] out Resulting fine grid field
     @param[in] in Input field on coarse grid
     @param[in] fine_to_coarse Fine-to-coarse lookup table (linear indices)
     @param[in] spin_map Spin blocking lookup table
     @param[in] parity of the output fine field (if single parity output field)
   */
  void StaggeredProlongate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const int *fine_to_coarse,
                           const int *const *spin_map, int parity = QUDA_INVALID_PARITY);

  /**
     @brief Apply the unitary "restriction" operator for Kahler-Dirac preconditioning
     @param[out] out Resulting coarse grid field
     @param[in] in Input field on fine grid
     @param[in] fine_to_coarse Fine-to-coarse lookup table (linear indices)
     @param[in] spin_map Spin blocking lookup table
     @param[in] parity of the output fine field (if single parity output field)
   */
  void StaggeredRestrict(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const int *fine_to_coarse,
                         const int *const *spin_map, int parity = QUDA_INVALID_PARITY);

} // namespace quda
#endif // _TRANSFER_H

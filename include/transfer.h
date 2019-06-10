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

  private:

    /** The raw null space components */
    const std::vector<ColorSpinorField*> &B;

    /** The number of null space components */
    const int Nvec;

    /** The number of times to Gram-Schmidt within block ortho */
    const int NblockOrtho;

    /** Precision to use for the GPU null-space components */
    const QudaPrecision null_precision;

    /** CPU copy of the block-normalized null-space components that define the prolongator */
    mutable ColorSpinorField *V_h;

    /** GPU copy of the block-normalized null-space components that define the prolongator */
    mutable ColorSpinorField *V_d;

    /** A CPU temporary field with fine geometry and fine color we use for changing gamma basis */
    mutable ColorSpinorField *fine_tmp_h;

    /** A GPU temporary field with fine geometry and fine color we use for changing gamma basis */
    mutable ColorSpinorField *fine_tmp_d;

    /** A CPU temporary field with coarse geometry and coarse color */
    mutable ColorSpinorField *coarse_tmp_h;

    /** A GPU temporary field with coarse geometry and coarse color we use for CPU input / output */
    mutable ColorSpinorField *coarse_tmp_d;

    /** The geometrical coase grid blocking */
    int *geo_bs;

    /** The mapping onto coarse sites from fine sites.  This has
	length equal to the fine-grid volume, and is sorted into
	lexicographical fine-grid order, with each value corresponding
	to a coarse-grid offset. (CPU) */
    mutable int *fine_to_coarse_h;

    /** The mapping onto fine sites from coarse sites. This has length
	equal to the fine-grid volume, and is sorted into lexicographical
	block order, with each value corresponding to a fine-grid offset. (CPU) */
    mutable int *coarse_to_fine_h;

    /** The mapping onto coarse sites from fine sites.  This has
	length equal to the fine-grid volume, and is sorted into
	lexicographical fine-grid order, with each value corresponding
	to a coarse-grid offset. (GPU) */
    mutable int *fine_to_coarse_d;

    /** The mapping onto fine sites from coarse sites. This has length
	equal to the fine-grid volume, and is sorted into lexicographical
	block order, with each value corresponding to a fine-grid offset. (GPU) */
    mutable int *coarse_to_fine_d;

    /** The spin blocking. Defined as zero when the fine operator is staggered. */
    int spin_bs;

    /** The mapping onto coarse spin from fine spin (inner) and fine parity (outer), for staggered */
    int **spin_map;

    /** Nspin for the fine level. Required for deallocating spin_map. */
    const int nspin_fine;

    /** Whether the transfer operator is to be applied to full fields or single parity fields */
    QudaSiteSubset site_subset;

    /** The parity of any single-parity fine-grid fields that are passed into the transfer operator */
    QudaParity parity;

    /** Whether the GPU transfer operator has been constructed */
    mutable bool enable_gpu;

    /** Whether the CPU transfer operator has been constructed */
    mutable bool enable_cpu;

    /** Whether to apply the transfer operaton the GPU (requires
	enable_gpu=true in the constructor) */
    mutable bool use_gpu;

    /**
     * @brief Allocate V field
     * @param[in] location Where to allocate the V field
     */
    void createV(QudaFieldLocation location) const;

    /**
     * @brief Allocate temporaries used when applying transfer operators
     * @param[in] location Where to allocate the temporaries
     */
    void createTmp(QudaFieldLocation location) const;

    /**
     * @brief Creates the map between fine and coarse grids
     * @param geo_bs An array storing the block size in each geometric dimension
     */
    void createGeoMap(int *geo_bs);

    /** 
     * @brief Creates the map between fine spin and parity to coarse spin dimensions
     * @param spin_bs The spin block size
     */
    void createSpinMap(int spin_bs);

    /**
     * @brief Lazy allocation of the transfer operator in a given location
     * @param[in] location Where to allocate the temporaries
     */
    void initializeLazy(QudaFieldLocation location) const;

    /**
     * Internal flops accumulator
     */
    mutable double flops_;

    /**
     * Reference to profile kept in the corresponding MG instance.
     * Use this to record restriction and prolongation overhead.
     */
    TimeProfile &profile;

  public:
      /**
       * The constructor for Transfer
       * @param B Array of null-space vectors
       * @param Nvec Number of null-space vectors
       * @param NblockOrtho Number of times to Gram-Schmidt within block ortho
       * @param d The Dirac operator to which these null-space vectors correspond
       * @param geo_bs The geometric block sizes to use
       * @param spin_bs The spin block sizes to use
       * @param parity For single-parity fields are these QUDA_EVEN_PARITY or QUDA_ODD_PARITY
       * @param null_precision The precision to store the null-space basis vectors in
       * @param enable_gpu Whether to enable this to run on GPU (as well as CPU)
       */
      Transfer(const std::vector<ColorSpinorField *> &B, int Nvec, int NblockOrtho, int *geo_bs, int spin_bs,
               QudaPrecision null_precision, TimeProfile &profile);

      /** The destructor for Transfer */
      virtual ~Transfer();

      /**
       @brief for resetting the Transfer when the null vectors have changed
       */
      void reset();

      /**
       * Apply the prolongator
       * @param out The resulting field on the fine lattice
       * @param in The input field on the coarse lattice
       */
      void P(ColorSpinorField &out, const ColorSpinorField &in) const;

      /**
       * Apply the restrictor
       * @param out The resulting field on the coarse lattice
       * @param in The input field on the fine lattice
       */
      void R(ColorSpinorField &out, const ColorSpinorField &in) const;

      /**
       * @brief The precision of the packed null-space vectors
       */
      QudaPrecision NullPrecision(QudaFieldLocation location) const
      {
        return location == QUDA_CUDA_FIELD_LOCATION ? null_precision : std::max(B[0]->Precision(), QUDA_SINGLE_PRECISION);
      }

    /**
     * Returns a const reference to the V field
     * @param location Which memory space are we requesting
     * @return The V field const reference
     */
    const ColorSpinorField& Vectors(QudaFieldLocation location=QUDA_INVALID_FIELD_LOCATION) const {
      if (location == QUDA_INVALID_FIELD_LOCATION) {
        // if not set then we return the memory space where the input vectors are stored
        return B[0]->Location() == QUDA_CUDA_FIELD_LOCATION ? *V_d : *V_h;
      } else {
        return location == QUDA_CUDA_FIELD_LOCATION ? *V_d : *V_h;
      }
    }

    /**
     * Returns the number of near nullvectors
     * @return Nvec
     */
    int nvec() const {return Nvec;}

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
     * Sets where the prolongator / restrictor should take place
     * @param location Location where the transfer operator should be computed
     */
    void setTransferGPU(bool use_gpu) const { this->use_gpu = use_gpu; }

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
     * Return flops
     * @return flops expended by this operator
     */
    double flops() const;
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
   */
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField *> &B, const int *fine_to_coarse,
                          const int *coarse_to_fine, const int *geo_bs, const int spin_bs, const int n_block_ortho);

  /**
     @brief Apply the prolongation operator
     @param[out] out Resulting fine grid field
     @param[in] in Input field on coarse grid
     @param[in] v Matrix field containing the null-space components
     @param[in] Nvec Number of null-space components
     @param[in] fine_to_coarse Fine-to-coarse lookup table (linear indices)
     @param[in] spin_map Spin blocking lookup table
     @param[in] parity of the output fine field (if single parity output field)
   */
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v, 
		  int Nvec, const int *fine_to_coarse, const int * const *spin_map,
		  int parity=QUDA_INVALID_PARITY);

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
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v, 
		int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int * const *spin_map,
		int parity=QUDA_INVALID_PARITY);
  

} // namespace quda
#endif // _TRANSFER_H

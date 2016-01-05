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
#include <dirac_quda.h>
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

    /** CPU copy of the block-normalized null-space components that define the prolongator */
    ColorSpinorField *V_h;

    /** GPU copy of the block-normalized null-space components that define the prolongator */
    ColorSpinorField *V_d;

    /** A CPU temporary field with fine geometry and fine color we use for changing gamma basis */
    ColorSpinorField *fine_tmp_h;

    /** A GPU temporary field with fine geometry and fine color we use for changing gamma basis */
    ColorSpinorField *fine_tmp_d;

    /** A CPU temporary field with coarse geometry and coarse color */
    ColorSpinorField *coarse_tmp_h; 

    /** A GPU temporary field with coarse geometry and coarse color we use for CPU input / output */
    ColorSpinorField *coarse_tmp_d; 

    /** The geometrical coase grid blocking */
    int *geo_bs;

    /** The mapping onto coarse sites from fine sites.  This has
	length equal to the fine-grid volume, and is sorted into
	lexicographical fine-grid order, with each value corresponding
	to a coarse-grid offset. (CPU) */
    int *fine_to_coarse_h;

    /** The mapping onto fine sites from coarse sites. This has length
	equal to the fine-grid volume, and is sorted into lexicographical
	block order, with each value corresponding to a fine-grid offset. (CPU) */
    int *coarse_to_fine_h;

    /** The mapping onto coarse sites from fine sites.  This has
	length equal to the fine-grid volume, and is sorted into
	lexicographical fine-grid order, with each value corresponding
	to a coarse-grid offset. (GPU) */
    int *fine_to_coarse_d;

    /** The mapping onto fine sites from coarse sites. This has length
	equal to the fine-grid volume, and is sorted into lexicographical
	block order, with each value corresponding to a fine-grid offset. (GPU) */
    int *coarse_to_fine_d;

    /** The spin blocking */
    int spin_bs;

    /** The mapping onto coarse spin from fine spin 
        A.S.: For the top-level staggered this map is underfined (NULL) 
        and is trivial one-to-one for the other levels*/
    int *spin_map;

    /** The length of the fine lattice */
    int fine_length;

    /** The length of the coarse lattice */
    int coarse_length;

    /** Whether to enable transfer operator on the GPU */
    bool enable_gpu;

    /** Whether to apply the transfer operaton the GPU (requires
	enable_gpu=true in the constructor) */
    mutable bool use_gpu;

    /**
     * Copies the null-space vector components into the V-field
     */
    void fillV(ColorSpinorField &V);

    /** 
     * Creates the map between fine and coarse grids 
     * @param geo_bs An array storing the block size in each geometric dimension
     */
    void createGeoMap(int *geo_bs);

    /**
     * Creates a block-ordered version of the color-spinor field V
     * N.B. in must be the accessor to the color-spinor field V
     * @param out A Complex array storing the block-ordered fields
     * @param in  Accessor for the color-spinor field V
     */
    //template <class Complex, class FieldOrder>
    //void blockOrderV(Complex *out, const FieldOrder &in, int *geo_bs, int spin_bs);

  /**
     * Copies elements from the block-ordered field in back to the color-spinor field V
     * N.B. out must be the accessor to the color-spinor field V
     * @param out The full lattice color spinor field, not block-ordered
     * @param in  A Complex array storing the block-ordered fields
     */
    //template <class FieldOrder, class Complex>
    //void undoblockOrderV(FieldOrder &out, Complex *in, int *geo_bs, int spin_bs);   

  /**
   * Does Gram-Schmidt orthogonalization.
   * @param v The block-ordered vectors
   * @param nBlocks
   * @param Nc
   * @param blockSize
   */
  //template <class Complex>
    //void blockGramSchmidt(Complex *v, int nBlocks, int Nc, int blockSize);

    /** 
     * Creates the map between fine and coarse spin dimensions
     * @param spin_bs The spin block size
     */
    void createSpinMap(int spin_bs);

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
     * @param d The Dirac operator to which these null-space vectors correspond
     * @param geo_bs The geometric block sizes to use
     * @param spin_bs The spin block sizes to use (default is 1)
     * @param enable_gpu Whether to enable this to run on GPU (as well as CPU)
     */
    Transfer(const std::vector<ColorSpinorField*> &B, int Nvec, int *geo_bs, int spin_bs,
	     bool enable_gpu, TimeProfile &profile);

    /** The destructor for Transfer */
    virtual ~Transfer();

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
     * Returns a const reference to the V field
     * @return The V field const reference
     */
    const ColorSpinorField& Vectors() const { return *V_h; }

    /**
     * Returns the number of near nullvectors
     * @return Nvec
     */
    int nvec() const {return Nvec;}

    /**
     * Returns the amount of spin blocking
     * @return spin_bs
     */
    int Spin_bs() const {return spin_bs;}

    /**
     * Returns the geometrical coarse grid blocking
     * @return geo_bs
     */
    const int *Geo_bs() const {return geo_bs;}
    
    /**
     * Sets where the prolongator / restrictor should take place
     * @param location Location where the transfer operator should be computed
     */
    void setTransferGPU(bool use_gpu) const { this->use_gpu = use_gpu; }

    /**
     * Return flops
     * @return flops expended by this operator
     */
    double flops() const;
  };

  /**
     Helper method that takes a vector of ColorSpinorFields and packes them into a single matrix field.
     @param[out] V The resulting packed matrix field
     @param[in] B Vector of ColorSpinorFields to be packed
     @param[in] Nvec Vector length
   */
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec);

  /**
     @brief Block orthogonnalize the matrix field, where the blocks are
     defined by lookup tables that map the fine grid points to the
     coarse grid points, and similarly for the spin degrees of
     freedom.
     @param[in,out] V Matrix field to be orthgonalized
     @param[in] Nvec Vector length
     @param[in] geo_bs Geometric block size
     @param[in] fine_to_coarse Fine-to-coarse lookup table (linear indices)
     @param[in] spin_bs Spin block size
   */
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, const int *geo_bs, 
			  const int *fine_to_coarse, int spin_bs);

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
		  int Nvec, const int *fine_to_coarse, const int *spin_map,
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
		int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int *spin_map,
		int parity=QUDA_INVALID_PARITY);
  

} // namespace quda
#endif // _TRANSFER_H

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

  class Transfer {

  private:

    /** The raw null space components */
    const std::vector<ColorSpinorField*> &B;

    /** The number of null space components */
    const int Nvec;

    /** The block-normalized null-space components that define the prolongator */
    ColorSpinorField *V;

    /** A temporary field with fine geometry but coarse color */
    ColorSpinorField *tmp;

    /** The mapping onto coarse sites from fine sites */
    int *geo_map;

    /** The mapping onto coarse spin from fine spin */
    int *spin_map;

    /** The length of the fine lattice */
    int fine_length;

    /** The length of the coarse lattice */
    int coarse_length;

    /**
     * Copies the null-space vector components into the V-field
     */
    void fillV();

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

    /** The single-precision accessor class */
    ColorSpinorFieldOrder<double> *order_double;

    /** The double-precision accessor class */
    ColorSpinorFieldOrder<float> *order_single;


  public:

    /** 
     * The constructor for Transfer
     * @param B Array of null-space vectors
     * @param Nvec Number of null-space vectors
     * @param d The Dirac operator to which these null-space vectors correspond
     * @param geo_map Geometric mapping from fine grid to coarse grid 
     * @param spin_map Mapping from fine spin to coarse spin 
     */
    Transfer(const std::vector<ColorSpinorField*> &B, int Nvec, int *geo_bs, int spin_bs);

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
    const ColorSpinorField& Vectors() { return *V; }

  };

  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec);

  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, const int *geo_bs, 
			  const int *geo_map, int spin_bs);

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, 
		  const ColorSpinorField &v, ColorSpinorField &tmp, 
		  int Nvec, const int *geo_map, const int *spin_map);

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, 
		const ColorSpinorField &v, ColorSpinorField &tmp,
		int Nvec, const int *geo_map, const int *spin_map);
  

} // namespace quda
#endif // _TRANSFER_H

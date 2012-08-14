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
#include <color_spinor_field_order.h>

#include <dirac_quda.h>

namespace quda {

  class Transfer {

  private:

    /** The raw null space components */
    cpuColorSpinorField *B;

    /** The block-normalized null-space components that define the prolongator */
    cpuColorSpinorField *V;

    /** A temporary field with fine geometry but coarse color */
    cpuColorSpinorField *tmp;

    /** The mapping onto coarse sites from fine sites */
    int *geo_map;

    /** The mapping onto coarse spin from fine spin */
    int *spin_map;

    /** 
     * Creates the map between fine and coarse grids 
     * @param geo_bs An array storing the block size in each geometric dimension
     */
    void createGeoMap(int *geo_bs);

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
    Transfer(cpuColorSpinorField *B, int Nvec, Dirac &d, int *geo_bs, int spin_bs);

    /** The destructor for Transfer */
    virtual ~Transfer();

    /** 
     * Apply the prolongator
     * @param out The resulting field on the fine lattice
     * @param in The input field on the coarse lattice
     */
    void P(cpuColorSpinorField &out, const cpuColorSpinorField &in);

    /** 
     * Apply the restrictor 
     * @param out The resulting field on the coarse lattice
     * @param in The input field on the fine lattice   
     */
    void R(cpuColorSpinorField &out, const cpuColorSpinorField &in);

  };

} // namespace quda
#endif // _TRANSFER_H

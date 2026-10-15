#pragma once

// Unitarity check and Gram-Schmidt reunitarization of host gauge fields, following MILC

#include <cstddef>
#include <enum_quda.h>

namespace quda
{

  namespace io
  {

    /**
       The largest deviation from unitarity that a gauge field may
       have and still be reunitarized.  This is MILC's TOLERANCE in
       generic/check_unitarity.c and generic/reunitarize2.c.
     */
    constexpr double milc_unitarity_tolerance = 1e-4;

    /**
       @brief The largest deviation from unitarity of this rank's links,
       measured as MILC's check_su3 does: the largest of each row's
       norm error and the overlap of each pair of rows.  A link with a
       NaN or infinite element counts as infinitely far from unitary.
       @param[in] gauge The host gauge field in QDP order, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] volume The number of local sites
     */
    double unitarity_deviation(const void *const gauge[], QudaPrecision prec, size_t volume);

    /**
       @brief Reunitarize this rank's links as MILC's reunit_su3 does,
       in the same order of operations: normalize row 0, orthogonalize
       row 1 to row 0 and normalize it, and set row 2 to the complex
       conjugate of the cross product of rows 0 and 1.
       @param[in,out] gauge The host gauge field in QDP order, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] volume The number of local sites
     */
    void reunitarize_gram_schmidt(void *const gauge[], QudaPrecision prec, size_t volume);

  } // namespace io

} // namespace quda

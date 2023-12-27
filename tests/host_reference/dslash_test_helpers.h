#pragma once

#include <quda.h>

enum class dslash_test_type {
  Dslash = 0,
  MatPC,
  Mat,
  MatPCDagMatPC,
  MatDagMat,
  M5,
  M5inv,
  Dslash4pre,
  MatPCLocal,
  MatLocal,
  MatPCDagMatPCLocal,
  MatDagMatLocal
};

/**
 * Determine if the dslash is full parity or single parity based on
 * the dslash_test_type
 * @param type  Dslash test type
 * @return True for a single parity operator, false for a full parity operator
 */
bool is_pc_dslash_test_type(dslash_test_type type);

/**
 * Apply the Dslash operator (D_{eo} or D_{oe}) for 4D EO preconditioned DWF.
 * @param h_out  Result spinor field
 * @param h_in   Input spinor field
 * @param param  Contains all metadata regarding host and device
 *               storage
 * @param parity The destination parity of the field
 * @param test_type Choose a type of dslash operators
 */
void dslashQuda_4dpc(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, dslash_test_type test_type);

/**
 * Apply the Dslash operator (D_{eo} or D_{oe}) for Mobius DWF.
 * @param h_out  Result spinor field
 * @param h_in   Input spinor field
 * @param param  Contains all metadata regarding host and device
 *               storage
 * @param parity The destination parity of the field
 * @param test_type Choose a type of dslash operators
 */
void dslashQuda_mdwf(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, dslash_test_type test_type);

void dslashQuda_mobius_eofa(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity,
                            dslash_test_type test_type);

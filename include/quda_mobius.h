#ifndef _QUDA_MOBIUS_H
#define _QUDA_MOBIUS_H

/**
 * @file  quda_mobius.h
 * @brief NOT the Main header file for the QUDA library
 *
 * Note to QUDA developers: When adding new members to QudaGaugeParam
 * and QudaInvertParam, be sure to update lib/check_params.h as well
 * as the Fortran interface in lib/quda_fortran.F90.
 */

#include <quda.h>
#include <enum_quda.h>
#include <stdio.h> /* for FILE */
#include <quda_constants.h>

#ifdef __cplusplus
extern "C" {
#endif

  void ritzQuda(void *hp_x, void *hp_b, QudaInvertParam& inv_param, bool find_min = false);

#ifdef __cplusplus
}
#endif

#endif /* _QUDA_H */

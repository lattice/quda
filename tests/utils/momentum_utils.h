#pragma once

#include "quda.h"

/**
 * @brief Create a momentum field in a MILC data layout
 *
 * @param[out] mom Momentum field
 * @param[in] precision Floating-point precision of field
 * @param[in] max_val Maximum value of field
 */
void createMomCPU(void *mom, QudaPrecision precision, double max_val = 1.0);

/**
 * @brief Compute and print a robust comparison of agreement between two
 *        momentum fields
 *
 * @param[in] momA First momentum field
 * @param[in] momB Second momentum field
 * @param[in] len Length of the momentum field
 * @param[in] precision Floating-point precision of field
 */
int strong_check_mom(const void *momA, const void *momB, int len, QudaPrecision prec);

/**
 * @brief Host reference implementation of the momentum action
 * contribution, including the MILC convention of subtracting 4
 * from each site norm to improve stability.
 *
 * @param[in] mom Momentum field
 * @param[in] len Length of the momentum field
 * @param[in] precision Floating-point precision of field
 */
double momentumActionCPU(const void *mom, int len, QudaPrecision prec);

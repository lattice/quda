#pragma once 

#include <blas_reference.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

int contraction_reference(void *spinorX, void *spinorY, void *result, QudaContractGamma cGamma, QudaPrecision cpu_prec, int X[]);

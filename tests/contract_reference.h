#pragma once 
#include <blas_reference.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void contraction_reference();

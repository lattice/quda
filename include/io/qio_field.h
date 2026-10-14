#pragma once

// Low-level QIO routines, used only by the GaugeIO and VectorIO front-ends

#include <enum_quda.h>

#ifdef HAVE_QIO
void read_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X, QudaFileIOMode io_mode);
void write_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X, QudaFileIOMode io_mode);
void read_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X, QudaSiteSubset subset,
                       QudaParity parity, int nColor, int nSpin, int Nvec, QudaFileIOMode io_mode);
void write_spinor_field(const char *filename, const void *V[], QudaPrecision precision, const int *X,
                        QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin, int Nvec, bool partfile,
                        QudaFileIOMode io_mode);
#endif

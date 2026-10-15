#pragma once

// Low-level routines of the external QIO backend, used only by the GaugeIO and VectorIO front-ends

#include <array>
#include <string>
#include <enum_quda.h>
#include <io/scidac_io.h>

#ifdef HAVE_QIO
void read_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X, QudaFileIOMode io_mode,
                      const std::array<int, 4> &io_partition_grid, quda::scidac::GaugeInfo &info);
quda::io::Checksum write_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X,
                                     QudaFileIOMode io_mode, const std::array<int, 4> &io_partition_grid,
                                     const quda::scidac::GaugeInfo &info);
void read_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X, QudaSiteSubset subset,
                       QudaParity parity, int nColor, int nSpin, int Nvec, QudaFileIOMode io_mode);
void write_spinor_field(const char *filename, const void *V[], QudaPrecision precision, const int *X,
                        QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin, int Nvec, bool partfile,
                        QudaFileIOMode io_mode);
#endif

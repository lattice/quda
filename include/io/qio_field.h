#pragma once

// Low-level routines of the external QIO backend, used only by the GaugeIO and VectorIO front-ends

#include <array>
#include <string>
#include <utility>
#include <enum_quda.h>
#include <io/scidac_io.h>

#ifdef HAVE_QIO
void read_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X, QudaFileIOMode io_mode,
                      const std::array<int, 4> &io_partition_grid, quda::scidac::GaugeInfo &info);
quda::io::Checksum write_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X,
                                     QudaFileIOMode io_mode, const std::array<int, 4> &io_partition_grid,
                                     const quda::scidac::GaugeInfo &info);
// reads the first records of a vector file, one per host field; milc means MILC's eigenvector records, which have no
// spins
std::vector<quda::scidac::FieldChecksum> read_spinor_field(const char *filename,
                                                           const std::vector<quda::io::HostField> &fields, const int *X,
                                                           QudaSiteSubset subset, int nColor, int nSpin, bool milc,
                                                           QudaFileIOMode io_mode,
                                                           const std::array<int, 4> &io_partition_grid);
// writes one record per host field, with the metadata of quda::scidac::vector_records
std::vector<quda::io::Checksum> write_spinor_field(const char *filename, const quda::scidac::VectorRecords &records,
                                                   const std::vector<quda::io::HostField> &fields, const int *X,
                                                   QudaSiteSubset subset, bool partfile, QudaFileIOMode io_mode,
                                                   const std::array<int, 4> &io_partition_grid);
#endif

#pragma once

// Low-level QIO routines, used only by the GaugeIO and VectorIO front-ends

#include <string>
#include <enum_quda.h>

/**
   @brief Gauge field metadata beyond the links themselves.  On save
   these describe what to write; on load they report what was found.
 */
struct QIOGaugeInfo {
  bool ildg = false;           /** Whether the file has (or should have) the ILDG records */
  std::string ildg_lfn;        /** ILDG logical file name to write; empty means none (save only) */
  bool has_usqcd_info = false; /** Whether plaq and linktr are (or should be) in a usqcdInfo record */
  double plaq = 0.0;           /** Average Re Tr U_P / 3 */
  double linktr = 0.0;         /** Average Re Tr U_mu / 3 */
  QudaPrecision file_prec = QUDA_INVALID_PRECISION; /** Precision of the data in the file (load only) */
};

#ifdef HAVE_QIO
void read_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X, QudaFileIOMode io_mode,
                      QIOGaugeInfo &info);
void write_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X, QudaFileIOMode io_mode,
                       const QIOGaugeInfo &info);
void read_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X, QudaSiteSubset subset,
                       QudaParity parity, int nColor, int nSpin, int Nvec, QudaFileIOMode io_mode);
void write_spinor_field(const char *filename, const void *V[], QudaPrecision precision, const int *X,
                        QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin, int Nvec, bool partfile,
                        QudaFileIOMode io_mode);
#endif

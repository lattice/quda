#pragma once

#include <optional>
#include <string>
#include <vector>
#include <color_spinor_field.h>
#include <reference_wrapper_helper.h>
#include <io/field_io.h>
#include <io/scidac_io.h>
#include <io/site_io.h>

namespace quda
{

  /** What a staggered source or propagator file held, besides the fields */
  struct KSFileInfo {
    KSFileType type = KSFileType::VectorSource;
    QudaPrecision file_prec = QUDA_INVALID_PRECISION; /** Of the records loaded */
    std::string file_info;                            /** The <info> of usqcdKSPropFile, or a source file's file XML */
    std::vector<std::optional<io::Hypercube>> boxes;  /** Per loaded source: the record's box, or none (full lattice) */
    std::vector<std::string> source_info;             /** Per loaded source: the record's <info> */
    std::vector<std::string> solution_info;           /** Per loaded solution: the record's <info> (propagators) */
  };

  /**
     @brief Loads the sources of MILC's staggered field files: color-vector
     source files, the source records of USQCD KS propagator files of
     source/solution pairs, and complex-field source files.  The fields
     are 4-d staggered fields (1 spin, 3 colors) of any precision, on the
     host or the device, full or of a single parity, which then takes its
     parity's sites of the file's full lattice.  QUDA does not save these
     files.
   */
  class KSSourceIO
  {
    const std::string filename;
    const FileIOParam param;

  public:
    /**
       @param[in] filename The file
       @param[in] param File I/O options: the format (AUTO or QIO, or
       the external QIO backend), the I/O mode, the I/O partition grid
       and whether a checksum mismatch is allowed
     */
    KSSourceIO(const std::string &filename, const FileIOParam &param = {});

    /**
       @brief Load the first sources.size() sources of a color-vector
       source file or of a VV-pair file's source records, or
       sources.size() <= 3 colors of a complex-field file (source c in
       color c, the other colors zero).  Sites outside a record's box
       are zero.
     */
    KSFileInfo load(cvector_ref<ColorSpinorField> &sources);
  };

  /**
     @brief Loads USQCD KS propagator files of source/solution pairs (VV
     pairs), as MILC writes them: a source record and a solution record
     for each source color.  The fields are as for KSSourceIO.  QUDA does
     not save these files.
   */
  class KSPropagatorIO
  {
    const std::string filename;
    const FileIOParam param;

  public:
    /**
       @param[in] filename The file
       @param[in] param File I/O options, as for KSSourceIO
     */
    KSPropagatorIO(const std::string &filename, const FileIOParam &param = {});

    /**
       @brief Load the first solutions.size() pairs of a VV-pair file.
       With sources empty the source records are checked but not loaded;
       otherwise there is one source per solution.
     */
    KSFileInfo load(cvector_ref<ColorSpinorField> &solutions, cvector_ref<ColorSpinorField> &sources);
  };

} // namespace quda

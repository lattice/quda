#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <io/field_io.h>
#include <io/host_field.h>
#include <io/lime.h>
#include <io/site_io.h>

namespace quda
{

  namespace scidac
  {

    // LIME record types
    constexpr const char *private_file_xml = "scidac-private-file-xml";
    constexpr const char *file_xml = "scidac-file-xml";
    constexpr const char *private_record_xml = "scidac-private-record-xml";
    constexpr const char *record_xml = "scidac-record-xml";
    constexpr const char *binary_data = "scidac-binary-data";
    constexpr const char *checksum = "scidac-checksum";
    constexpr const char *ildg_format = "ildg-format";
    constexpr const char *ildg_data_lfn = "ildg-data-lfn";
    constexpr const char *ildg_binary_data = "ildg-binary-data";

    /**
       @brief The private file XML
       @param[in] dims The global lattice dimensions
       @param[in] volfmt 0 for a single file, 2 for a partfile
     */
    std::string file_info_xml(const std::array<int, 4> &dims, int volfmt);

    /**
       @brief The private record XML of a lattice field, dated now.
       Colors, spins, typesize and datacount are omitted when zero.
       @param[in] typesize Bytes per item and site
       @param[in] datacount Items per site
     */
    std::string record_info_xml(const std::string &datatype, QudaPrecision prec, int colors, int spins, int typesize,
                                int datacount);

    /**
       @brief The checksum record XML
     */
    std::string checksum_xml(uint32_t suma, uint32_t sumb);

    /**
       @brief The ildg-format record XML of a gauge field
     */
    std::string ildg_format_xml(QudaPrecision prec, const std::array<int, 4> &dims);

    /**
       @brief Group the records of a LIME file into messages, using their MB and ME flags
     */
    std::vector<std::vector<lime::Record>> group_messages(const std::vector<lime::Record> &records);

    /**
       @brief Find the first record of a type in a message
     */
    const lime::Record *find_record(const std::vector<lime::Record> &message, const std::string &type);

    /**
       @brief The metadata of the lattice field in a SciDAC or ILDG
       single file: what the private record XML, the user record XML
       and the ILDG records hold (or should hold)
     */
    struct FieldMetadata {
      std::string datatype;
      QudaPrecision prec = QUDA_INVALID_PRECISION;
      int colors = 0;
      int spins = 0;
      int typesize = 0;     /** Bytes per item and site */
      int datacount = 0;    /** Items per site */
      std::string user_xml; /** The user record XML */
      bool ildg = false;    /** Whether the file has (or should have) the ILDG records */
      std::string ildg_lfn; /** ILDG logical file name to write; empty means none (save only) */
    };

    /**
       @brief Where the lattice field of a SciDAC or ILDG file is, as found on load
     */
    struct FieldRecords {
      std::string file;           /** The file that holds the field */
      bool native = false;        /** Whether the file has the SciDAC records (ILDG files from other codes need not) */
      std::array<int, 4> dims {}; /** The lattice of the file */
      FieldMetadata metadata;     /** From the private record XML, or the ildg-format record */
      int n_binary = 0;           /** Number of messages with binary data in the file */
      lime::Record data;          /** The field's binary data */
      std::optional<lime::Record> checksum; /** The first checksum record after the data */
    };

    /**
       @brief Find the field of a SciDAC or ILDG single file: the
       first message with binary data.  As QIO, this reads <filename>,
       or <filename>.vol0000 if there is no such file.  Errors if the
       file is not a single file or not a lattice field.
     */
    FieldRecords find_field(const std::string &filename);

    /**
       @brief The checksum of a field's binary data
     */
    struct FieldChecksum {
      io::Checksum sum;              /** Computed from the data */
      bool has_checksum = false;     /** Whether the file has a checksum (not in ILDG files from other codes) */
      bool checksum_matches = false; /** Whether it matches the data */
    };

    /**
       @brief Read the field found by find_field into a host field,
       after checking its dimensions, typesize and length.  The
       checksum is verified, and a mismatch is an error unless param
       allows it.  The datacount is the caller's to check.
       @param[in] records The field's records
       @param[out] field The host field, one item per datacount
       @param[in] X The local lattice dimensions of a full field
       @param[in] parity Even or odd for a single-parity field, stored on
       a lattice of half the global x extent; otherwise the full lattice
       @param[in] param File I/O options
     */
    FieldChecksum read_field(const FieldRecords &records, const io::HostField &field, const int *X, QudaParity parity,
                             const FileIOParam &param);

    /**
       @brief Save a field as a SciDAC single file with one record: a
       first message with the private file XML and QUDA's file XML,
       then one with the private record XML, the user record XML, the
       ILDG records if requested, the binary data and its checksum
       @param[in] filename The file
       @param[in] metadata What to write in the metadata records; the field is saved in its precision
       @param[in] field The host field, one item per datacount
       @param[in] X The local lattice dimensions of a full field
       @param[in] parity Even or odd for a single-parity field, stored on
       a lattice of half the global x extent; otherwise the full lattice
       @param[in] param File I/O options (the I/O mode and partitions)
       @return The checksums written
     */
    io::Checksum write_field(const std::string &filename, const FieldMetadata &metadata, const io::HostField &field,
                             const int *X, QudaParity parity, const FileIOParam &param);

    /**
       @brief Gauge field metadata beyond the links themselves, for
       both the native and the external QIO backend.  On save these
       describe what to write; on load they report what was found.
     */
    struct GaugeInfo {
      bool ildg = false;           /** Whether the file has (or should have) the ILDG records */
      std::string ildg_lfn;        /** ILDG logical file name to write; empty means none (save only) */
      bool has_usqcd_info = false; /** Whether plaq and linktr are (or should be) in a usqcdInfo record */
      double plaq = 0.0;           /** Average Re Tr U_P / 3 */
      double linktr = 0.0;         /** Average Re Tr U_mu / 3 */
      QudaPrecision file_prec = QUDA_INVALID_PRECISION; /** Precision of the data in the file (load only) */
      bool has_checksum = false; /** Whether the file has a checksum (load only; not in ILDG files from other codes) */
      bool checksum_matches = false; /** Whether it matches the data (load only) */
      io::Checksum sum;              /** The checksums suma and sumb computed from the data (load only) */
    };

    /**
       @brief The datatype that QUDA writes in the private record XML of a gauge field
       @param[in] file_prec Precision of the file
       @param[in] ildg Whether the file is an ILDG file, which requires a USQCD datatype
     */
    std::string gauge_datatype(QudaPrecision file_prec, bool ildg);

    /**
       @brief The datatype that QUDA writes in the private record XML of a set of vectors
       @param[in] file_prec Precision of the file
     */
    std::string vector_datatype(QudaPrecision file_prec, int nSpin, int nColor);

    /**
       @brief QUDA's own record XML for a field, written as the user record XML
       @param[in] len Reals per site of each field (18 for gauge fields)
       @param[in] datatype The datatype of the private record XML
     */
    std::string quda_record_xml(int len, const std::string &datatype, QudaSiteSubset subset, QudaParity parity,
                                int nColor, int nSpin);

    /**
       @brief The USQCD gauge field record XML, usqcdInfo, with the average plaquette and link trace
     */
    std::string usqcd_info_xml(double plaq, double linktr);

    /**
       @brief Find the content of the first element <tag>...</tag> of an
       XML string.  The opening tag may have attributes; elements of the
       same name nested within it are skipped over.
       @return The content, or nothing if there is no such element
     */
    std::optional<std::string> xml_value(const std::string &xml, const std::string &tag);

    /**
       @brief Load a SciDAC or ILDG gauge field.  The checksum is
       verified, and a mismatch is an error unless param allows it.
       @param[in] filename The file
       @param[out] gauge The host gauge field in QDP order, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
       @param[in] param File I/O options
       @return Metadata found in the file
     */
    GaugeInfo load_gauge(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                         const FileIOParam &param);

    /**
       @brief Save a SciDAC or ILDG gauge field in the file precision equal to the host precision
       @param[in] filename The file
       @param[in] gauge The host gauge field in QDP order, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
       @param[in] param File I/O options (the I/O mode and partitions)
       @param[in] info Whether to write ILDG records and a usqcdInfo record
       @return The checksums written
     */
    io::Checksum save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, const GaugeInfo &info);

    /**
       @brief Load a set of vectors from a single file in QUDA's layout,
       in which each site holds all the vectors one after another.
       The checksum is verified, and a mismatch is an error unless
       param allows it.
       @param[in] filename The file
       @param[out] V The host vectors, each [spin][color][complex] at the QUDA host index
       @param[in] prec The precision of the host vectors
       @param[in] X The local lattice dimensions of the vectors (half the x extent for single-parity vectors)
       @param[in] subset Whether the vectors are full or single parity; single-parity files have half the x extent
       @param[in] parity The parity of single-parity vectors
       @param[in] Nvec Number of vectors
       @param[in] param File I/O options
       @return The file precision and the checksums
     */
    std::pair<QudaPrecision, FieldChecksum> load_vectors(const std::string &filename, void *const V[], QudaPrecision prec,
                                                         const int *X, QudaSiteSubset subset, QudaParity parity,
                                                         int nColor, int nSpin, int Nvec, const FileIOParam &param);

    /**
       @brief Save a set of vectors to a single file in QUDA's layout, in the file precision equal to the host precision
       @param[in] filename The file
       @param[in] V The host vectors, each [spin][color][complex] at the QUDA host index
       @param[in] prec The precision of the host vectors
       @param[in] X The local lattice dimensions of the vectors (half the x extent for single-parity vectors)
       @param[in] subset Whether the vectors are full or single parity; single-parity files have half the x extent
       @param[in] parity The parity of the vectors, recorded in the user record XML
       @param[in] Nvec Number of vectors
       @param[in] param File I/O options (the I/O mode and partitions; partfiles are not supported yet)
       @return The checksums written
     */
    io::Checksum save_vectors(const std::string &filename, const void *const V[], QudaPrecision prec, const int *X,
                              QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin, int Nvec,
                              const FileIOParam &param);

  } // namespace scidac

} // namespace quda

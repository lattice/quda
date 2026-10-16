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
    constexpr const char *sitelist = "scidac-sitelist";

    // volume formats of the private file XML
    constexpr int single_file_volfmt = 0;
    constexpr int partfile_volfmt = 2;

    /**
       @brief The private file XML
       @param[in] dims The global lattice dimensions
       @param[in] volfmt single_file_volfmt or partfile_volfmt
     */
    std::string file_info_xml(const std::array<int, 4> &dims, int volfmt);

    /**
       @brief The name of volume n of a partfile: <filename>.volNNNN, as QIO names them
     */
    std::string volume_name(const std::string &filename, int n);

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
       file: what the private record XML, the user record XML and the
       ILDG records hold (or should hold)
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

    // the user file XML that QUDA writes
    constexpr const char *quda_user_file_xml = "Dummy user file XML";

    /**
       @brief A lattice field of a SciDAC or ILDG file, as found on load: a message with binary data
     */
    struct FieldRecord {
      FieldMetadata metadata;               /** From the private record XML, or the ildg-format record */
      lime::Record data;                    /** The binary data (in a partfile, that of the first volume) */
      std::optional<io::Checksum> checksum; /** From the first checksum record after the data */
    };

    /**
       @brief Where the lattice fields of a SciDAC or ILDG file are, as found on load
     */
    struct FieldRecords {
      std::string filename; /** The name the fields were loaded by */
      std::string file;     /** The file that holds the fields, or the first volume of a partfile */
      bool native = false;  /** Whether the file has the SciDAC records (ILDG files from other codes need not) */
      int volfmt = single_file_volfmt;
      std::array<int, 4> dims {};      /** The lattice of the file */
      std::string user_file_xml;       /** The user file XML (SciDAC files only) */
      std::vector<FieldRecord> fields; /** Every message with binary data, in order (only the first in ILDG
                                           files without the SciDAC records) */
    };

    /**
       @brief Find the fields of a SciDAC or ILDG file: every message
       with binary data.  As QIO, this reads <filename>, or
       <filename>.vol0000 if there is no such file, which holds the
       metadata of a partfile.  Rank 0 reads every metadata record in
       one pass.  Errors if the file is neither a single file nor a
       partfile, or holds no lattice field.
     */
    FieldRecords find_field(const std::string &filename);

    /**
       @brief Find the volumes of a partfile and check that they tile
       the lattice.  Rank 0 lists the directory for <filename>.vol
       followed by four or more digits, as the volumes' numbers have
       gaps when they were saved with I/O partitions.  Each volume's
       site list must be a box of the lattice in lexicographic order,
       the boxes must not overlap and must cover the lattice, and every
       volume must hold as many binary records as the first, of the same
       size.  Collective.
       @param[in] records The fields, as found by find_field
       @param[in] site_bytes Size of a site of each record in the files
       @return For each record, the volumes with their data and boxes
     */
    std::vector<std::vector<io::FilePart>> find_volumes(const FieldRecords &records, size_t site_bytes);

    /**
       @brief The checksum of a field's binary data
     */
    struct FieldChecksum {
      io::Checksum sum;              /** Computed from the data */
      bool has_checksum = false;     /** Whether the file has a checksum (not in ILDG files from other codes) */
      bool checksum_matches = false; /** Whether it matches the data */
    };

    /**
       @brief Read the first fields found by find_field, from a single
       file or a partfile, into host fields, one record at a time,
       after checking the dimensions, and each record's typesize and
       length.  Each record's checksum is verified, and a mismatch is
       an error unless param allows it.  The datacounts are the
       caller's to check.
       @param[in] records The fields' records
       @param[out] fields The host fields, one per record from the first, each with one item per datacount
       @param[in] X The local lattice dimensions of a full field
       @param[in] parity Even or odd for single-parity fields, stored on
       a lattice of half the global x extent; otherwise the full lattice
       @param[in] param File I/O options
       @return The checksum of each record read
     */
    std::vector<FieldChecksum> read_fields(const FieldRecords &records, const std::vector<io::HostField> &fields,
                                           const int *X, QudaParity parity, const FileIOParam &param);

    /**
       @brief Read the first field found by find_field; see read_fields
     */
    FieldChecksum read_field(const FieldRecords &records, const io::HostField &field, const int *X, QudaParity parity,
                             const FileIOParam &param);

    /**
       @brief Save fields as a SciDAC file with one record each: a
       first message with the private file XML and the user file XML,
       then per field one with the private record XML, the user record
       XML, the ILDG records if requested, the binary data and its
       checksum.  With param.partfile on more than one rank, the fields
       are saved as QIO saves a partfile: one volume per I/O partition,
       named after its leader, each with a site list and then each
       record's data, and the metadata and the checksums only in the
       first volume.  On one rank, as in QIO, a single file.
       @param[in] filename The file
       @param[in] user_file_xml The user file XML
       @param[in] metadata What to write in each record's metadata records; each field is saved in its precision
       @param[in] fields The host fields, one per record, each with one item per datacount
       @param[in] X The local lattice dimensions of a full field
       @param[in] parity Even or odd for single-parity fields, stored on
       a lattice of half the global x extent; otherwise the full lattice
       @param[in] param File I/O options (the I/O mode, partitions and partfile options)
       @return The checksums written, one per record
     */
    std::vector<io::Checksum> write_fields(const std::string &filename, const std::string &user_file_xml,
                                           const std::vector<FieldMetadata> &metadata,
                                           const std::vector<io::HostField> &fields, const int *X, QudaParity parity,
                                           const FileIOParam &param);

    /**
       @brief Save a field as a SciDAC file with one record and QUDA's user file XML; see write_fields
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
       @param[in] eval An eigenvalue, written as <eigVal> in the <info> element if given
       @param[in] resid A residual, written as <resid> in the <info> element if given
     */
    std::string quda_record_xml(int len, const std::string &datatype, QudaSiteSubset subset, QudaParity parity,
                                int nColor, int nSpin, std::optional<double> eval = {}, std::optional<double> resid = {});

    /**
       @brief The datatype of MILC's eigenvector records, USQCD_[FD]3_ColorVector
     */
    std::string milc_vector_datatype(QudaPrecision file_prec);

    /**
       @brief The user file XML of MILC's eigenvector files, with the number of eigenvectors
     */
    std::string milc_eigen_file_xml(int n_vectors);

    /**
       @brief The user record XML of an eigenvector in MILC's files, with its eigenvalue and residual
     */
    std::string milc_eigen_record_xml(double eval, double resid);

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
       @brief The strings a vector file carries: QUDA's, or those of MILC's eigenvector files
     */
    enum class VectorStrings { QUDA, MILC };

    /**
       @brief A vector file, as found on load
     */
    struct VectorFile {
      FieldRecords records;
      VectorStrings strings = VectorStrings::QUDA;
      std::vector<double> evals; /** The <eigVal> of every record, or none if a record has none */
    };

    /**
       @brief Find the vectors of a SciDAC file, with its strings.  A
       file is MILC's eigenvector format if its user file XML has
       MILC's title and a positive <Nvecs>; every record must then be
       one of MILC's eigenvectors.  Files of USQCD KS propagators and
       MILC's packed eigenvectors are errors.  Collective.
       @param[in] filename The file
     */
    VectorFile find_vectors(const std::string &filename);

    /**
       @brief The host fields that the first vectors of a file are read
       into, one per record to read: a record each, or, from a file with
       all its vectors in one record (QUDA's layout before one record per
       vector), that record, with the vectors not loaded skipped.
       Errors if the file holds fewer vectors.
       @param[in] file The file
       @param[in] field The host vectors, Ls items each
       @param[in] n_vectors The number of vectors to load
     */
    std::vector<io::HostField> vector_fields(const VectorFile &file, const io::HostField &field, int n_vectors);

    /**
       @brief Load the first vectors of a file found by find_vectors,
       from a single file or a partfile saved with any rank layout and
       partition grid, after checking the records' shapes.  The
       checksums are verified, and a mismatch is an error unless param
       allows it.
       @param[in] file The file
       @param[out] fields The host fields of vector_fields, each item a 4-d vector (a vector, or a slice of a 5-d
       one), [spin][color][complex] at the QUDA host index
       @param[in] X The local lattice dimensions of the file's lattice (half the x extent for single-parity files)
       @param[in] subset Whether the file's lattice is the full lattice or half of it in x
       @param[in] parity The parity of a single-parity file
       @param[in] param File I/O options
       @return The checksum of each record read
     */
    std::vector<FieldChecksum> load_vectors(const VectorFile &file, const std::vector<io::HostField> &fields,
                                            const int *X, QudaSiteSubset subset, QudaParity parity, int nColor,
                                            int nSpin, const FileIOParam &param);

    /**
       @brief What to save with a set of vectors
     */
    struct VectorSaveInfo {
      VectorStrings strings = VectorStrings::QUDA;
      bool interleaved = false;   /** All vectors in one record, as QUDA saved them before (QUDA's strings only) */
      std::vector<double> evals;  /** One eigenvalue per vector, or none */
      std::vector<double> resids; /** One residual per vector, or none */
    };

    /**
       @brief The metadata records of a set of vectors: the user file XML, and each record's metadata
     */
    struct VectorRecords {
      std::string user_file_xml;
      std::vector<FieldMetadata> records;
    };

    /**
       @brief The metadata records to save a set of vectors with, one
       record per vector (or one for them all, if interleaved), for both
       backends
       @param[in] prec The file precision
       @param[in] n_vectors The number of vectors
       @param[in] Ls Items per vector
       @param[in] subset Whether the file's lattice is the full lattice or half of it in x
       @param[in] parity The parity of the vectors, recorded in QUDA's user record XML
     */
    VectorRecords vector_records(QudaPrecision prec, int n_vectors, int Ls, QudaSiteSubset subset, QudaParity parity,
                                 int nColor, int nSpin, const VectorSaveInfo &info);

    /**
       @brief Save a set of vectors, one record per vector, as a single
       file or a partfile, in the file precision equal to the host
       precision
       @param[in] filename The file
       @param[in] field The host vectors, Ls items each, every item a 4-d vector (a vector, or a slice of a 5-d one),
       [spin][color][complex] at the QUDA host index
       @param[in] n_vectors The number of vectors
       @param[in] X The local lattice dimensions of the file's lattice (half the x extent for single-parity files)
       @param[in] subset Whether the file's lattice is the full lattice or half of it in x
       @param[in] parity The parity of the vectors, recorded in QUDA's user record XML
       @param[in] info The strings, eigenvalues and residuals
       @param[in] param File I/O options (the I/O mode, partitions and partfile options)
       @return The checksums written, one per record
     */
    std::vector<io::Checksum> save_vectors(const std::string &filename, const io::HostField &field, int n_vectors,
                                           const int *X, QudaSiteSubset subset, QudaParity parity, int nColor,
                                           int nSpin, const VectorSaveInfo &info, const FileIOParam &param);

  } // namespace scidac

} // namespace quda

#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <io/field_io.h>
#include <io/gauge_io.h>
#include <io/host_field.h>
#include <io/lime.h>
#include <io/site_io.h>

namespace quda
{

  /**
     @brief The staggered field files of MILC that QUDA loads: USQCD KS
     propagator files of source/solution pairs, color-vector source
     files, and complex-field source files
   */
  enum class KSFileType { Propagator, VectorSource, ComplexSource };

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
       @param[in] hypercube The box of a hypercube record (record type 2), or none for a lattice field
     */
    std::string record_info_xml(const std::string &datatype, QudaPrecision prec, int colors, int spins, int typesize,
                                int datacount, const std::optional<io::Hypercube> &hypercube = {});

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
      /** The ILDG logical file name: on save, the ildg-data-lfn record to write with the ILDG records, if any; on
          load, the one found in the record's message (or anywhere, in ILDG files from other codes) */
      std::optional<std::string> ildg_lfn;
      /** The box of the file's lattice that a hypercube record (record type 2) holds; none for a lattice field */
      std::optional<io::Hypercube> hypercube;
      std::optional<std::string> date; /** The <date> of the private record XML, verbatim (load only) */
    };

    /**
       @brief The box of a hypercube record, "l0 l1 l2 l3 .. u0 u1 u2 u3", for messages
     */
    std::string hypercube_str(const io::Hypercube &box);

    /**
       @brief QUDA's user file XML: <qudaFile> with its version, the git
       version of the QUDA library (if it was built with one) and a
       description of the content
       @param[in] content What the file holds, e.g. "gauge field" or "3 vectors"
     */
    std::string quda_file_xml(const std::string &content);

    // the user file XML that QUDA wrote before quda_file_xml (FileIOParam::legacy_strings)
    constexpr const char *legacy_file_xml = "Dummy user file XML";

    /**
       @brief A lattice field of a SciDAC or ILDG file, as found on load: a message with binary data
     */
    struct FieldRecord {
      FieldMetadata metadata;               /** From the private record XML, or the ildg-format record */
      lime::Record data;                    /** The binary data (in a partfile, that of the first volume) */
      std::optional<io::Checksum> checksum; /** From the first checksum record after the data */
      size_t index = 0;                     /** The record's place among the file's binary records */
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
       partfile, or holds no lattice field or hypercube record.
     */
    FieldRecords find_field(const std::string &filename);

    /**
       @brief Whether loading a file reads a partfile: there is no
       <filename>, but there is <filename>.vol0000.  Collective.
     */
    bool partfile_named(const std::string &filename);

    /**
       @brief Error if any of the first n binary records of a file is a
       hypercube record, which only staggered source files hold
       @param[in] n The number of records to check (all of them if negative)
     */
    void check_no_hypercube(const FieldRecords &records, int n = -1);

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
       length.  A hypercube record fills only its box's sites, and only
       in single files.  Each record's checksum is verified, and a
       mismatch is an error unless param allows it.  The datacounts and
       whether boxes are expected are the caller's to check.
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
       @param[in] metadata What to write in each record's metadata
       records; each field is saved in its precision, and as a
       hypercube record of its box's sites if it has one (single files
       only)
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
      bool legacy_strings = false;   /** Save with the strings of FileIOParam::legacy_strings (save only) */
      GaugeMetadata metadata;        /** The file's SciDAC and ILDG metadata (load only) */
    };

    /**
       @brief Set the user record XML of a gauge field's metadata, with
       whether it is a usqcdInfo record and the content of its <info>
       element, for both backends
       @param[out] metadata The metadata
       @param[in] user_xml The user record XML as found in the file
     */
    void set_gauge_record_xml(GaugeMetadata &metadata, const std::string &user_xml);

    /**
       @brief The metadata records of a gauge field: the user file XML and the record's metadata
     */
    struct GaugeRecords {
      std::string user_file_xml;
      FieldMetadata record;
    };

    /**
       @brief The metadata records to save a gauge field with, for both backends
       @param[in] file_prec Precision of the file
       @param[in] info Whether to write the ILDG records, a usqcdInfo record and the legacy strings
     */
    GaugeRecords gauge_records(QudaPrecision file_prec, const GaugeInfo &info);

    /**
       @brief The datatype that QUDA writes in the private record XML of a set of vectors: USQCD's for 3-color
       staggered and Wilson vectors, otherwise QUDA's own
       @param[in] file_prec Precision of the file
       @param[in] legacy Always QUDA's own, as QUDA wrote them before (FileIOParam::legacy_strings)
     */
    std::string vector_datatype(QudaPrecision file_prec, int nSpin, int nColor, bool legacy = false);

    /**
       @brief The datatype of 3-color vectors without spin, USQCD_[FD]3_ColorVector: that of MILC's
       eigenvectors and of QUDA's staggered vectors
     */
    std::string color_vector_datatype(QudaPrecision file_prec);

    /**
       @brief Whether a record's spins are those of vectors with nSpin spins.  Records of
       color_vector_datatype have no spins, as QIO and MILC write them, and hold 1-spin vectors.
     */
    bool spins_match(const std::string &datatype, int spins, int nSpin);

    /**
       @brief QUDA's own record XML for a field, written as the user record XML
       @param[in] len Reals per site of each field (18 for gauge fields)
       @param[in] eval An eigenvalue, written as <eigVal> in the <info> element if given
       @param[in] eval_imag Its imaginary part, written as <eigValImag> after <eigVal> if given
       @param[in] resid A residual, written as <resid> in the <info> element if given
       @param[in] sign_convention A staggered sign convention, written as <staggeredSignConvention> in the <info>
       element if not empty
     */
    std::string quda_record_xml(int len, QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin,
                                std::optional<double> eval = {}, std::optional<double> eval_imag = {},
                                std::optional<double> resid = {}, const std::string &sign_convention = "");

    /**
       @brief QUDA's record XML as it was written before quda_record_xml
       (FileIOParam::legacy_strings): version BETA, with the datatype as
       <type>, and without imaginary parts of eigenvalues
       @param[in] datatype The datatype of the private record XML
     */
    std::string legacy_quda_record_xml(int len, const std::string &datatype, QudaSiteSubset subset, QudaParity parity,
                                       int nColor, int nSpin, std::optional<double> eval = {},
                                       std::optional<double> resid = {});

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
       @param[in] info Whether to write ILDG records, a usqcdInfo record and the legacy strings
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
      /** The <eigVal> of every record with its <eigValImag> (0 if none), or none if a record has no <eigVal> */
      std::vector<std::complex<double>> evals;
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
      bool interleaved = false;    /** All vectors in one record, as QUDA saved them before (QUDA's strings only) */
      bool legacy_strings = false; /** QUDA's strings as before (FileIOParam::legacy_strings); implied by interleaved */
      std::vector<double> evals;   /** One eigenvalue per vector, or none */
      /** Their imaginary parts, or none; QUDA's strings hold them in every record if any is nonzero */
      std::vector<double> evals_imag;
      std::vector<double> resids;  /** One residual per vector, or none */
      std::string sign_convention; /** The staggered sign convention ("QUDA" or "MILC"), or empty (QUDA's strings) */
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

    /**
       @brief A staggered source or KS propagator file, as found on load
     */
    struct KSFile {
      FieldRecords records;
      KSFileType type = KSFileType::VectorSource;
      std::string file_info;         /** The <info> of usqcdKSPropFile, or a source file's whole file XML */
      std::vector<std::string> info; /** Each binary record's <info>, empty if it has none */
    };

    /**
       @brief The kind of staggered file whose records find_field found,
       by find_ks_file's rules, without checking the records; none if it
       is not one.  USQCD Dirac propagator files and MILC's Dirac source
       files are errors.
     */
    std::optional<KSFileType> ks_file_type(const FieldRecords &records);

    /**
       @brief Find the records of a staggered source or KS propagator
       file and check them: a USQCD KS propagator file of
       source/solution pairs (VV pairs), a file of complex fields (of
       which the first is the source), or a file of color-vector
       sources with color 0, 1, ... in turn.  Sources may be hypercube
       records.  Partfiles, other USQCD propagator files and any other
       file are errors.  Collective.
       @param[in] filename The file
     */
    KSFile find_ks_file(const std::string &filename);

  } // namespace scidac

} // namespace quda

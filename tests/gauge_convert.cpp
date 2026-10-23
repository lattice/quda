#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <quda.h>
#include <comm_quda.h>
#include <timer.h>
#include <util_quda.h>
#include <io/gauge_io.h>
#include <io/milc_io.h>
#include <io/reunitarize.h>
#include <io/scidac_io.h>
#include <io/site_io.h>

#include <command_line_params.h>
#include <gauge_utils.h>
#include <host_utils.h>
#include <misc.h>

// gauge_convert: load a gauge field (or generate one) and save it in another format, precision, style and I/O mode,
// forwarding the input's metadata, then reload the output and check it.

using namespace quda;

namespace
{
  constexpr int link_reals = 18;

  size_t volume() { return static_cast<size_t>(V); }

  // the longest usqcdInfo record QIO writes, without its NUL (QIO_STRINGALLOC - 1)
  constexpr size_t qio_max_record_xml = 1023;

  // the format of a file as the no-op rule sees it: the backend doesn't count
  enum class Family { MILC, SCIDAC, ILDG };

  const char *family_str(Family family)
  {
    switch (family) {
    case Family::MILC: return "MILC";
    case Family::SCIDAC: return "SciDAC";
    case Family::ILDG: return "ILDG";
    }
    return "unknown";
  }

  Family save_family(QudaFileFormat format)
  {
    switch (format) {
    case QUDA_FILE_FORMAT_MILC: return Family::MILC;
    case QUDA_FILE_FORMAT_ILDG:
    case QUDA_FILE_FORMAT_ILDG_EXTERNAL: return Family::ILDG;
    default: return Family::SCIDAC;
    }
  }

  Family load_family(const GaugeFileInfo &info)
  {
    if (info.format == QUDA_FILE_FORMAT_MILC) return Family::MILC;
    return info.ildg ? Family::ILDG : Family::SCIDAC;
  }

  const char *prec_str(QudaPrecision prec) { return prec == QUDA_DOUBLE_PRECISION ? "double" : "single"; }

  // a host gauge field in QDP order, one array per direction
  struct HostGauge {
    const QudaPrecision prec;
    std::array<std::vector<char>, 4> data;
    void *ptr[4];
    HostGauge(QudaPrecision prec, size_t volume) : prec(prec)
    {
      for (int d = 0; d < 4; d++) {
        data[d].resize(volume * link_reals * prec);
        ptr[d] = data[d].data();
      }
    }
    HostGauge(const HostGauge &) = delete;
    HostGauge &operator=(const HostGauge &) = delete;
  };

  // the input and output and their parameters, all checked before any file is touched
  struct Conversion {
    bool generated = false;
    std::string input;
    std::string output;
    QudaFileFormat save_format = QUDA_FILE_FORMAT_INVALID; // as given, _EXTERNAL included
    FileIOParam load_param;
    FileIOParam save_param;
    FileIOParam verify_param;
  };

  QudaFileIOParam public_param(QudaFileFormat format, QudaFileIOMode io_mode, const std::array<int, 4> &partitions)
  {
    QudaFileIOParam param = newQudaFileIOParam();
    param.format = format;
    param.io_mode = io_mode;
    for (int d = 0; d < 4; d++) param.io_partition_grid[d] = partitions[d];
    param.write_usqcd_info = gauge_write_usqcd_info ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    param.struct_size = sizeof(param);
    return param;
  }

  bool is_zero(const std::array<int, 4> &grid) { return grid == std::array<int, 4> {}; }

  Conversion check_options()
  {
    Conversion c;
    if (unit_gauge) errorQuda("gauge_convert takes --gauge-input unit rather than --unit-gauge");
    c.generated = gauge_input_explicit && gauge_input != GaugeInputMode::LOAD;
    if (c.generated && !latfile.empty())
      errorQuda("Give either --load-gauge or a generated --gauge-input, not both (--gauge-input %s)",
                get_gauge_input_str(gauge_input));
    if (!c.generated && latfile.empty())
      errorQuda("Give the field to convert, with --load-gauge or a generated --gauge-input haar|unit|gaussian-su3");
    if (gauge_input_width_explicit && !(c.generated && gauge_input == GaugeInputMode::GAUSSIAN_SU3))
      errorQuda("--gauge-input-width requires --gauge-input gaussian-su3");
    if (gauge_input_seed_given && !c.generated) errorQuda("--gauge-input-seed requires a generated --gauge-input");
    if (c.generated) {
      if (gauge_reunitarize != QUDA_GAUGE_REUNITARIZE_NONE) errorQuda("--gauge-reunitarize requires --load-gauge");
      if (gauge_allow_checksum_mismatch) errorQuda("--gauge-allow-checksum-mismatch requires --load-gauge");
      if (gauge_load_format_given || gauge_load_io_mode_given || gauge_load_io_partitions_given)
        errorQuda("--gauge-load-format, --gauge-load-io-mode and --gauge-load-io-partitions require --load-gauge");
    }
    c.input = latfile;
    c.output = gauge_outfile;

    auto load = public_param(gauge_load_format_given ? gauge_load_format : gauge_file_format,
                             gauge_load_io_mode_given ? gauge_load_io_mode : gauge_file_io_mode,
                             gauge_load_io_partitions_given ? gauge_load_io_partitions : gauge_file_io_partitions);
    load.allow_checksum_mismatch = gauge_allow_checksum_mismatch ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    load.reunitarize = gauge_reunitarize;
    if (!c.generated) c.load_param = FileIOParam(&load);

    if (c.output.empty()) {
      const std::vector<std::pair<bool, const char *>> save_options
        = {{gauge_save_format != QUDA_FILE_FORMAT_INVALID, "--gauge-save-format"},
           {gauge_save_prec != QUDA_INVALID_PRECISION, "--gauge-save-prec"},
           {gauge_save_io_mode_given, "--gauge-save-io-mode"},
           {gauge_save_io_partitions_given, "--gauge-save-io-partitions"},
           {!is_zero(gauge_save_partfile_grid), "--gauge-save-partfile-grid"},
           {gauge_convert_overwrite, "--gauge-convert-overwrite"},
           {!gauge_convert_verify, "--gauge-convert-verify"}};
      for (auto &[given, name] : save_options)
        if (given) warningQuda("%s is ignored without --save-gauge", name);
      return c;
    }
    if (gauge_save_format == QUDA_FILE_FORMAT_INVALID)
      errorQuda("--save-gauge requires --gauge-save-format (qio, ildg, milc, qio-external or ildg-external)");
    c.save_format = gauge_save_format;
    const auto family = save_family(c.save_format);
    if (family == Family::MILC && gauge_save_prec == QUDA_DOUBLE_PRECISION)
      errorQuda("MILC gauge files are single precision, so --gauge-save-prec double cannot be saved as MILC");

    const bool grid = !is_zero(gauge_save_partfile_grid);
    if (grid) {
      for (auto n : gauge_save_partfile_grid)
        if (n <= 0) errorQuda("--gauge-save-partfile-grid needs four positive numbers");
      if (c.save_format != QUDA_FILE_FORMAT_QIO)
        errorQuda("--gauge-save-partfile-grid needs --gauge-save-format qio: only the native backend lays out "
                  "partfiles for another process grid, and only SciDAC files have partfiles");
    }
    if ((gauge_save_partfile || grid) && family != Family::SCIDAC)
      errorQuda("Only SciDAC gauge files have partfiles, not %s files", family_str(family));

    auto save = public_param(c.save_format, gauge_save_io_mode_given ? gauge_save_io_mode : gauge_file_io_mode,
                             gauge_save_io_partitions_given ? gauge_save_io_partitions : gauge_file_io_partitions);
    safe_strcpy(save.ildg_lfn, gauge_ildg_lfn, 256, "gauge_ildg_lfn");
    save.partfile = gauge_save_partfile || grid ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    c.save_param = FileIOParam(&save);
    c.save_param.partfile_grid = gauge_save_partfile_grid;

    // the output is reloaded with the parameters it was saved with, as it was saved
    save.partfile = QUDA_BOOLEAN_FALSE;
    c.verify_param = FileIOParam(&save);
    return c;
  }

  bool exists(const std::filesystem::path &path)
  {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
  }

  // whether a file name is <base>.vol followed by 4 to 9 digits, as partfile volumes are named
  bool is_volume_name(const std::string &name, const std::string &base)
  {
    const auto prefix = base + ".vol";
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    const auto digits = name.substr(prefix.size());
    return digits.size() >= 4 && digits.size() <= 9
      && std::all_of(digits.begin(), digits.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
  }

  // the volumes of <file> that exist
  std::vector<std::string> volumes(const std::string &file)
  {
    namespace fs = std::filesystem;
    const fs::path path(file);
    const auto dir = path.has_parent_path() ? path.parent_path() : fs::path(".");
    const auto base = path.filename().string();
    std::vector<std::string> found;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
      const auto name = it->path().filename().string();
      if (is_volume_name(name, base)) found.push_back(path.has_parent_path() ? (dir / name).string() : name);
    }
    std::sort(found.begin(), found.end());
    return found;
  }

  // whether two paths name the same file: lexically, or as existing files
  bool same_file(const std::string &a, const std::string &b)
  {
    namespace fs = std::filesystem;
    if (fs::path(a).lexically_normal() == fs::path(b).lexically_normal()) return true;
    std::error_code ec;
    return exists(a) && exists(b) && fs::equivalent(a, b, ec);
  }

  // whether a file would be a volume of a partfile saved as <base>, lexically
  bool is_volume_of(const std::string &file, const std::string &base)
  {
    namespace fs = std::filesystem;
    const auto f = fs::path(file).lexically_normal(), b = fs::path(base).lexically_normal();
    return f.parent_path() == b.parent_path() && is_volume_name(f.filename().string(), b.filename().string());
  }

  std::string name_list(const std::vector<std::string> &names)
  {
    std::string list;
    for (auto i = 0u; i < names.size() && i < 3; i++) list += (i ? ", " : "") + io::excerpt(names[i]);
    if (names.size() > 3) list += " and " + std::to_string(names.size() - 3) + " more";
    return list;
  }

  /**
     @brief The hazards of the save, checked on rank 0: the save would
     write or remove one of the input's files (an error even with
     --gauge-convert-overwrite), or the output exists.  Errors on every
     rank.
     @return The files an overwrite removes (on rank 0)
   */
  std::vector<std::string> check_hazards(const Conversion &c)
  {
    std::string error;
    std::vector<std::string> existing;
    if (comm_rank() == 0) {
      const bool milc = save_family(c.save_format) == Family::MILC;
      if (exists(c.output)) existing.push_back(c.output);
      if (milc && exists(c.output + ".info")) existing.push_back(c.output + ".info");
      for (auto &v : volumes(c.output)) existing.push_back(v);

      if (!c.generated) {
        std::vector<std::string> inputs = {c.input};
        for (auto &v : volumes(c.input)) inputs.push_back(v);
        if (exists(c.input + ".info")) inputs.push_back(c.input + ".info");
        std::vector<std::string> outputs = existing;
        outputs.push_back(c.output);
        if (milc) outputs.push_back(c.output + ".info");

        for (auto &in : inputs) {
          bool clash = is_volume_of(in, c.output);
          for (auto &out : outputs) clash = clash || same_file(in, out);
          if (clash) {
            error = "Saving " + io::excerpt(c.output) + " would write or remove " + io::excerpt(in)
              + ", a file of the input " + io::excerpt(c.input);
            break;
          }
        }
      }
      if (error.empty() && !existing.empty() && !gauge_convert_overwrite)
        error = "The output exists: " + name_list(existing) + "; give --gauge-convert-overwrite to replace it";
    }
    io::broadcast(error);
    if (!error.empty()) errorQuda("%s", error.c_str());

    if (comm_size() == 1 && c.save_param.partfile && is_zero(c.save_param.partfile_grid))
      warningQuda("A partfile saved on one rank is a single file; give --gauge-save-partfile-grid for volumes");
    return existing;
  }

  void remove_existing(const std::vector<std::string> &existing)
  {
    if (comm_rank() == 0)
      for (auto &file : existing) {
        std::error_code ec;
        if (!std::filesystem::remove(file, ec) || ec)
          errorQuda("Could not remove %s: %s", io::excerpt(file).c_str(), ec.message().c_str());
        logQuda(QUDA_SUMMARIZE, "Removed %s (--gauge-convert-overwrite)\n", io::excerpt(file).c_str());
      }
    comm_barrier();
  }

  double unitarity_deviation(const HostGauge &gauge)
  {
    double deviation = io::unitarity_deviation(gauge.ptr, gauge.prec, volume());
    comm_allreduce_max(deviation);
    return deviation;
  }

  double gigabytes(QudaPrecision prec)
  {
    return 4.0 * link_reals * static_cast<int>(prec) * volume() * comm_size() / 1e9;
  }

  void report_field(const char *what, const HostGauge &gauge, const int *X)
  {
    const auto obs = gauge_observables(gauge.ptr, gauge.prec, X);
    printfQuda("%s: plaquette %.12f (spatial %.12f, temporal %.12f), link trace %.12e, largest deviation from "
               "unitarity %.2e\n",
               what, obs.plaq, obs.plaq_spatial, obs.plaq_temporal, obs.linktr, unitarity_deviation(gauge));
  }

  void generate(HostGauge &gauge)
  {
    if (gauge_input_seed_given) initRand(gauge_input_seed);
    // links without any boundary condition or anisotropy, as a file holds them
    switch (gauge_input) {
    case GaugeInputMode::HAAR: constructRandomSU3GaugeField(gauge.ptr, gauge.prec); break;
    case GaugeInputMode::UNIT: constructIdentityGaugeField(gauge.ptr, gauge.prec); break;
    case GaugeInputMode::GAUSSIAN_SU3: constructGaussianSU3GaugeField(gauge.ptr, gauge.prec, gauge_input_width); break;
    default: errorQuda("Unexpected gauge input %s", get_gauge_input_str(gauge_input));
    }
    printGaugeInput(gauge_input);
    if (gauge_input_seed_given) printfQuda("Gauge input seed: %lu\n", static_cast<unsigned long>(gauge_input_seed));
  }

  // MILC's .info keys that QUDA writes itself, and the four it computes
  const std::vector<std::string> milc_generic_keys
    = {"magic_number", "time_stamp", "checksums", "nx", "ny", "nz", "nt"};
  const std::vector<std::string> milc_computed_keys
    = {"gauge.ssplaq", "gauge.stplaq", "gauge.nersc_linktr", "gauge.nersc_checksum"};

  bool contains(const std::vector<std::string> &keys, const std::string &key)
  {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
  }

  std::string trim(const std::string &s)
  {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return std::string();
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
  }

  MILCInfoItem info_item(const std::string &key, const std::string &value) { return {key, key + " = " + value}; }

  std::string quoted(const std::string &s) { return "\"" + s + "\""; }

  // MILC's keys start with a lower-case letter, followed by letters, digits, dots and underscores
  bool is_info_key(const std::string &key)
  {
    if (key.empty() || key[0] < 'a' || key[0] > 'z') return false;
    return std::all_of(key.begin(), key.end(), [](char ch) {
      return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.';
    });
  }

  /**
     @brief Whether a text holds MILC .info lines: every line that isn't
     blank starts with a key followed by "=" ("key = value", "key =[3]
     a b c"), or by white space and one value, a word or a quoted string
     (MILC's older "key value")
   */
  bool is_info_text(const std::string &text)
  {
    const auto items = milc::info_items(text);
    if (items.empty()) return false;
    for (auto &item : items) {
      if (!is_info_key(item.key)) return false;
      const auto line = trim(item.line);
      const auto rest = trim(line.substr(item.key.size()));
      if (rest.empty()) return false;
      if (rest[0] == '=') continue;
      if (rest[0] == '"') {
        if (rest.size() < 2 || rest.back() != '"') return false;
      } else if (rest.find_first_of(" \t") != std::string::npos) {
        return false;
      }
    }
    return true;
  }

  // the content of a record XML that is a bare <info> element, with or without the XML declaration
  std::optional<std::string> bare_info(const std::string &xml)
  {
    auto s = trim(xml);
    if (s.compare(0, 5, "<?xml") == 0) {
      auto end = s.find("?>");
      if (end == std::string::npos) return std::nullopt;
      s = trim(s.substr(end + 2));
    }
    const std::string open = "<info>", close = "</info>";
    if (s.size() < open.size() + close.size() || s.compare(0, open.size(), open) != 0
        || s.compare(s.size() - close.size(), close.size(), close) != 0)
      return std::nullopt;
    return s.substr(open.size(), s.size() - open.size() - close.size());
  }

  std::string join_lines(const std::vector<MILCInfoItem> &items)
  {
    std::string text;
    for (auto &item : items) text += item.line + "\n";
    return text;
  }

  /**
     @brief Replace the lines of the given keys by the new ones, at the
     first line of each key, leaving later lines of those keys out
     @param[in] append Append the new lines whose key has no line
   */
  std::vector<MILCInfoItem> replace_items(const std::vector<MILCInfoItem> &items,
                                          const std::vector<MILCInfoItem> &replacements, bool append)
  {
    std::vector<MILCInfoItem> out;
    std::vector<std::string> placed;
    auto find = [&](const std::string &key) {
      return std::find_if(replacements.begin(), replacements.end(), [&](const auto &r) { return r.key == key; });
    };
    for (auto &item : items) {
      auto r = find(item.key);
      if (r == replacements.end()) {
        out.push_back(item);
      } else if (!contains(placed, item.key)) {
        out.push_back(*r);
        placed.push_back(item.key);
      }
    }
    if (append)
      for (auto &r : replacements)
        if (!contains(placed, r.key)) out.push_back(r);
    return out;
  }

  // MILC's four computed keys of the links, as milc::save_gauge writes them
  std::vector<MILCInfoItem> computed_items(const GaugeObservables &obs)
  {
    auto print = [](const char *fmt, auto value) {
      char buf[64];
      snprintf(buf, sizeof(buf), fmt, value);
      return std::string(buf);
    };
    return {info_item("gauge.ssplaq", print("%f", static_cast<double>(static_cast<float>(3.0 * obs.plaq_spatial)))),
            info_item("gauge.stplaq", print("%f", static_cast<double>(static_cast<float>(3.0 * obs.plaq_temporal)))),
            info_item("gauge.nersc_linktr", print("%e", static_cast<double>(static_cast<float>(obs.linktr)))),
            info_item("gauge.nersc_checksum", print("%u", static_cast<unsigned int>(obs.nersc_checksum)))};
  }

  // MILC's provenance keys: the source file, its time stamp (MILC) or record date (SciDAC), and its checksums
  std::vector<MILCInfoItem> provenance_items(const std::string &input, const GaugeFileInfo &info)
  {
    std::vector<MILCInfoItem> items = {info_item("gauge.previous.filename", quoted(input))};
    const auto &m = info.metadata;
    if (info.format == QUDA_FILE_FORMAT_MILC && !m.milc_time_stamp.empty())
      items.push_back(info_item("gauge.previous.time_stamp", quoted(m.milc_time_stamp)));
    else if (info.format != QUDA_FILE_FORMAT_MILC && m.record_date)
      items.push_back(info_item("gauge.previous.time_stamp", quoted(*m.record_date)));
    char sums[32];
    snprintf(sums, sizeof(sums), "%x %x", info.sum29, info.sum31);
    items.push_back(info_item("gauge.previous.checksums", quoted(sums)));
    return items;
  }

  bool is_quda_file_xml(const std::string &xml) { return xml.find("<qudaFile>") != std::string::npos; }
  bool is_quda_record_xml(const std::string &xml) { return xml.find("<qudaGaugeFieldFile>") != std::string::npos; }

  // what the mapping gives the save, and what it found
  struct Mapping {
    GaugeMetadata metadata;
    std::vector<std::string> notes;
    size_t provenance_bytes = 0; // the bytes of usqcdInfo's <info> that are provenance lines
  };

  /**
     @brief The reasons a conversion is not a no-op: none means it changes nothing in the content
   */
  std::vector<std::string> change_reasons(const Conversion &c, const GaugeFileInfo &info, QudaPrecision save_prec)
  {
    if (c.generated) return {"the field is generated"};
    std::vector<std::string> reasons;
    const auto in = load_family(info), out = save_family(c.save_format);
    if (in != out)
      reasons.push_back(std::string("the format changes from ") + family_str(in) + " to " + family_str(out));
    if (save_prec != info.file_prec)
      reasons.push_back(std::string("the precision changes from ") + prec_str(info.file_prec) + " to "
                        + prec_str(save_prec));
    switch (c.load_param.reunitarize) {
    case QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1:
      if (info.file_prec == QUDA_DOUBLE_PRECISION)
        reasons.push_back("the links were rounded to single precision (milc-gram-schmidt-precision-1)");
      break;
    case QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2:
    case QUDA_GAUGE_REUNITARIZE_POLAR:
      reasons.push_back(std::string("the links were reunitarized (")
                        + get_gauge_reunitarize_str(c.load_param.reunitarize) + ")");
      break;
    default: break;
    }
    if (info.has_checksum && !info.checksum_matches) reasons.push_back("the file's checksums don't match its data");
    if (out == Family::MILC && info.format == QUDA_FILE_FORMAT_MILC && info.byte_swapped)
      reasons.push_back("the input is in the other byte order, and MILC files are saved in the native one");
    return reasons;
  }

  /**
     @brief Map the input's metadata to what the save writes
     @param[in] saved The observables of the links as saved
   */
  Mapping map_metadata(const Conversion &c, const GaugeFileInfo &info, bool noop, const GaugeObservables &saved)
  {
    Mapping map;
    auto &out = map.metadata;
    if (c.generated) return map;
    const auto &in = info.metadata;
    const auto from = load_family(info), to = save_family(c.save_format);
    const auto provenance = noop ? std::vector<MILCInfoItem> {} : provenance_items(c.input, info);
    const bool from_milc = from == Family::MILC;

    if (to == Family::MILC) {
      if (noop) out.milc_time_stamp = in.milc_time_stamp;
      std::vector<MILCInfoItem> lines;
      if (from_milc) {
        lines = in.milc_info;
      } else {
        const auto bare = bare_info(in.user_record_xml);
        if (in.has_usqcd_info && in.usqcd_info && is_info_text(*in.usqcd_info)) {
          lines = milc::info_items(*in.usqcd_info);
          map.notes.push_back("the .info lines of the usqcdInfo record's <info> are carried");
        } else if (!in.has_usqcd_info && bare && is_info_text(*bare)) {
          lines = milc::info_items(*bare);
          map.notes.push_back("the .info lines of the record XML's <info> element are carried");
        } else if (!in.has_usqcd_info && is_info_text(in.user_record_xml)) {
          lines = milc::info_items(in.user_record_xml);
          map.notes.push_back("the .info lines of the plain-text record XML are carried");
        } else if (in.has_usqcd_info && in.usqcd_info && !in.usqcd_info->empty()) {
          map.notes.push_back("the usqcdInfo record's <info> text holds no .info lines, so it is not carried");
        } else if (!in.has_usqcd_info && !in.user_record_xml.empty() && !is_quda_record_xml(in.user_record_xml)) {
          map.notes.push_back("the record XML holds no .info lines, so it is not carried");
        }
        if (!in.user_file_xml.empty() && !is_quda_file_xml(in.user_file_xml))
          map.notes.push_back("the user file XML is not carried: MILC files have none");
      }
      out.milc_info = replace_items(lines, provenance, true);
      return map;
    }

    // SciDAC or ILDG output
    if (noop) {
      out.record_date = in.record_date;
      if (to == Family::ILDG && gauge_ildg_lfn.empty()) out.ildg_lfn = in.ildg_lfn;
    } else if (to == Family::ILDG && in.ildg_lfn && !in.ildg_lfn->empty() && gauge_ildg_lfn.empty()) {
      warningQuda("The ILDG LFN of %s, \"%s\", is not reused, since the conversion changes the field; %s gets its "
                  "file name (give --gauge-ildg-lfn to choose one)",
                  io::excerpt(c.input).c_str(), io::excerpt(*in.ildg_lfn).c_str(), io::excerpt(c.output).c_str());
    }

    if (from_milc) {
      std::vector<MILCInfoItem> lines;
      for (auto &item : in.milc_info)
        if (!contains(milc_generic_keys, item.key)) lines.push_back(item);
      lines = replace_items(lines, computed_items(saved), false);
      out.has_usqcd_info = true;
      out.usqcd_info = join_lines(replace_items(lines, provenance, true));
      map.provenance_bytes = join_lines(provenance).size();
      if (!in.milc_info.empty()) map.notes.push_back("the .info lines are carried in the usqcdInfo record's <info>");
      if (!gauge_write_usqcd_info)
        map.notes.push_back("--gauge-write-usqcd-info is false, so no usqcdInfo record, and the .info lines are lost");
      return map;
    }

    if (!in.user_file_xml.empty() && !is_quda_file_xml(in.user_file_xml)) out.user_file_xml = in.user_file_xml;

    if (in.has_usqcd_info) {
      out.has_usqcd_info = true;
      out.usqcd_info = in.usqcd_info;
      if (!noop) {
        const auto text = in.usqcd_info.value_or("");
        if (is_info_text(text)) {
          auto lines = replace_items(milc::info_items(text), computed_items(saved), false);
          out.usqcd_info = join_lines(replace_items(lines, provenance, true));
        } else if (trim(text).empty() || text == "QUDA") {
          out.usqcd_info = join_lines(provenance);
        } else {
          out.usqcd_info = text + (text.back() == '\n' ? "" : "\n") + join_lines(provenance);
        }
        map.provenance_bytes = join_lines(provenance).size();
      }
      if (!gauge_write_usqcd_info)
        map.notes.push_back("--gauge-write-usqcd-info is false, so the usqcdInfo record is not written");
    } else if (!in.user_record_xml.empty() && !is_quda_record_xml(in.user_record_xml)) {
      out.user_record_xml = in.user_record_xml;
      map.notes.push_back("the record XML isn't a usqcdInfo record, so it is copied as it is, and no usqcdInfo "
                          "record (nor gauge.previous.* lines) is written");
    } else if (!noop) {
      out.has_usqcd_info = true;
      out.usqcd_info = join_lines(provenance);
      map.provenance_bytes = out.usqcd_info->size();
      if (!gauge_write_usqcd_info)
        map.notes.push_back("--gauge-write-usqcd-info is false, so the gauge.previous.* lines are not written");
    }
    return map;
  }

  // QIO cuts usqcdInfo records at 1023 bytes, so the external backend refuses longer ones; say why, before saving
  void check_external_limit(const Conversion &c, const Mapping &map, const GaugeObservables &saved)
  {
    const auto &m = map.metadata;
    if (!c.save_param.external_qio || !gauge_write_usqcd_info) return;
    if (!m.has_usqcd_info && !m.user_record_xml.empty()) return;
    const auto xml = scidac::usqcd_info_xml(saved.plaq, saved.linktr, m.usqcd_info.value_or("QUDA"));
    if (xml.size() <= qio_max_record_xml) return;
    errorQuda(
      "The usqcdInfo record of %s would be %lu bytes, but the external QIO library writes at most %lu; %lu bytes "
      "of its <info> are the gauge.previous.* lines this conversion adds. Save with the native backend "
      "(--gauge-save-format qio or ildg)",
      io::excerpt(c.output).c_str(), xml.size(), qio_max_record_xml, map.provenance_bytes);
  }

  std::vector<MILCInfoItem> without_quda_keys(const std::vector<MILCInfoItem> &items)
  {
    std::vector<MILCInfoItem> out;
    for (auto &item : items)
      if (!contains(milc_generic_keys, item.key) && !contains(milc_computed_keys, item.key)) out.push_back(item);
    return out;
  }

  /**
     @brief Reload the output and compare it with what was saved
     @return What is wrong, empty if nothing
   */
  std::vector<std::string> verify(const Conversion &c, const HostGauge &saved, const GaugeMetadata &given, const int *X)
  {
    std::vector<std::string> failures;
    HostGauge loaded(saved.prec, volume());
    host_timer_t timer;
    timer.start();
    const auto info = GaugeIO(c.output, c.verify_param).load(loaded.ptr, loaded.prec, X);
    timer.stop();

    size_t differ = 0;
    for (int d = 0; d < 4; d++)
      for (size_t i = 0; i < volume() * link_reals; i++)
        if (memcmp(saved.data[d].data() + i * saved.prec, loaded.data[d].data() + i * saved.prec, saved.prec) != 0)
          differ++;
    comm_allreduce_sum(differ);
    if (differ > 0) failures.push_back(std::to_string(differ) + " of the reloaded reals differ from those saved");

    const auto family = save_family(c.save_format);
    if (info.file_prec != saved.prec) failures.push_back("the file is not of the precision saved");
    if (load_family(info) != family) failures.push_back("the file is not of the format saved");
    if (info.has_checksum && !info.checksum_matches) failures.push_back("its checksums don't match its data");
    if (info.has_plaq_linktr && !info.plaq_linktr_matches)
      failures.push_back("its plaquette and link trace don't match the field");

    const auto &m = info.metadata;
    if (family == Family::MILC) {
      if (!given.milc_time_stamp.empty() && m.milc_time_stamp != given.milc_time_stamp)
        failures.push_back("its time stamp is not the one given");
      const auto expected = without_quda_keys(given.milc_info), found = without_quda_keys(m.milc_info);
      bool same = expected.size() == found.size();
      for (auto i = 0u; same && i < expected.size(); i++) same = expected[i].line == found[i].line;
      if (!same) failures.push_back("its .info lines are not those given");
      if (gauge_write_usqcd_info && !info.has_plaq_linktr)
        failures.push_back("its .info file lacks the computed plaquettes and link trace");
    } else {
      if (given.user_file_xml.empty() ? !is_quda_file_xml(m.user_file_xml) : m.user_file_xml != given.user_file_xml)
        failures.push_back("its user file XML is not the one expected");
      if (!given.has_usqcd_info && !given.user_record_xml.empty()) {
        if (m.has_usqcd_info || m.user_record_xml != given.user_record_xml)
          failures.push_back("its record XML is not the one given");
      } else if (gauge_write_usqcd_info) {
        if (!m.has_usqcd_info || m.usqcd_info != given.usqcd_info.value_or("QUDA"))
          failures.push_back("its usqcdInfo record's <info> is not the one given");
        if (!info.has_plaq_linktr) failures.push_back("its usqcdInfo record has no plaquette and link trace");
      }
      if (given.record_date && m.record_date != given.record_date)
        failures.push_back("its record date is not the one given");
      if (family == Family::ILDG) {
        std::string lfn = given.ildg_lfn && !given.ildg_lfn->empty() ? *given.ildg_lfn :
          c.save_param.ildg_lfn.empty()                              ? c.output :
                                                                       c.save_param.ildg_lfn;
        if (m.ildg_lfn != lfn) failures.push_back("its ILDG LFN is not " + io::excerpt(lfn));
      }
    }

    printfQuda("Verified %s: reloaded in %g s (%.3f GB/s), checksums %s, %s\n", c.output.c_str(), timer.last(),
               gigabytes(saved.prec) / timer.last(), io::checksum_str({info.sum29, info.sum31}).c_str(),
               failures.empty() ? "every value and the metadata as saved" : "FAILED");
    return failures;
  }

  void convert(const Conversion &c, const std::vector<std::string> &existing, const int *X)
  {
    HostGauge gauge(QUDA_DOUBLE_PRECISION, volume());
    GaugeFileInfo info;
    if (c.generated) {
      generate(gauge);
      report_field("Generated field", gauge, X);
    } else {
      host_timer_t timer;
      timer.start();
      info = GaugeIO(c.input, c.load_param).load(gauge.ptr, gauge.prec, X);
      timer.stop();
      printfQuda("Loaded %s: %s%s, %s precision, checksums %s (%s), in %g s (%.3f GB/s)\n", c.input.c_str(),
                 family_str(load_family(info)), c.load_param.external_qio ? " (external QIO)" : "",
                 prec_str(info.file_prec), io::checksum_str({info.sum29, info.sum31}).c_str(),
                 !info.has_checksum      ? "the file has none" :
                   info.checksum_matches ? "verified" :
                                           "NOT verified",
                 timer.last(), gigabytes(info.file_prec) / timer.last());
      if (info.reunitarized_deviation >= 0)
        printfQuda("Reunitarized with %s: largest deviation from unitarity %.2e before\n",
                   get_gauge_reunitarize_str(c.load_param.reunitarize), info.unitarity_deviation);
      report_field("Loaded field", gauge, X);
    }
    if (c.output.empty()) return;

    const auto family = save_family(c.save_format);
    const auto file_prec = c.generated ? QUDA_DOUBLE_PRECISION : info.file_prec;
    const auto save_prec = gauge_save_prec != QUDA_INVALID_PRECISION ? gauge_save_prec :
      family == Family::MILC                                         ? QUDA_SINGLE_PRECISION :
                                                                       file_prec;
    const auto method = c.load_param.reunitarize;
    if (!c.generated && save_prec > file_prec && method != QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2
        && method != QUDA_GAUGE_REUNITARIZE_POLAR)
      warningQuda("Saving the single-precision links of %s in double precision: they are unitary only to about 1e-7 "
                  "(--gauge-reunitarize milc-gram-schmidt-precision-2 reunitarizes them as MILC's double-precision "
                  "builds do)",
                  c.input.c_str());

    // the links as saved: rounded in place for single precision, so that both copies hold the saved values
    HostGauge rounded(QUDA_SINGLE_PRECISION, save_prec == QUDA_SINGLE_PRECISION ? volume() : 0);
    if (save_prec == QUDA_SINGLE_PRECISION) {
      for (int d = 0; d < 4; d++) {
        auto u = static_cast<double *>(gauge.ptr[d]);
        auto f = static_cast<float *>(rounded.ptr[d]);
#pragma omp parallel for
        for (size_t i = 0; i < volume() * link_reals; i++) {
          f[i] = static_cast<float>(u[i]);
          u[i] = f[i];
        }
      }
    }
    const HostGauge &saved = save_prec == QUDA_SINGLE_PRECISION ? rounded : gauge;
    const auto observables = gauge_observables(gauge.ptr, gauge.prec, X);

    const auto reasons = change_reasons(c, info, save_prec);
    const bool noop = reasons.empty();
    if (noop) {
      printfQuda("The conversion is a no-op: the time stamp or record date and the ILDG LFN are kept\n");
    } else {
      std::string why;
      for (auto i = 0u; i < reasons.size(); i++) why += (i ? "; " : "") + reasons[i];
      printfQuda("The conversion is not a no-op: %s\n", why.c_str());
    }
    const auto map = map_metadata(c, info, noop, observables);
    for (auto &note : map.notes) printfQuda("Metadata: %s\n", note.c_str());
    check_external_limit(c, map, observables);

    remove_existing(existing);
    host_timer_t timer;
    timer.start();
    GaugeIO(c.output, c.save_param).save(saved.ptr, saved.prec, X, map.metadata);
    timer.stop();
    printfQuda("Saved %s: %s%s%s, %s precision, in %g s (%.3f GB/s)\n", c.output.c_str(), family_str(family),
               c.save_param.external_qio ? " (external QIO)" : "", c.save_param.partfile ? " partfile" : "",
               prec_str(save_prec), timer.last(), gigabytes(save_prec) / timer.last());

    if (!gauge_convert_verify) return;
    const auto failures = verify(c, saved, map.metadata, X);
    if (!failures.empty()) {
      for (auto &f : failures) printfQuda("Verification of %s: %s\n", c.output.c_str(), f.c_str());
      errorQuda("Verification of %s failed (%lu problems)", c.output.c_str(), failures.size());
    }
  }
} // namespace

int main(int argc, char **argv)
{
  auto app = make_app("Convert a gauge field between SciDAC, ILDG and MILC files", "gauge_convert");
  add_gauge_convert_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);
  setVerbosity(verbosity);
  int X[4] = {xdim, ydim, zdim, tdim};
  setDims(X);

  // every file parameter and hazard is checked before any file is touched
  const auto conversion = check_options();
  const auto existing = conversion.output.empty() ? std::vector<std::string> {} : check_hazards(conversion);

  initQuda(device_ordinal);
  convert(conversion, existing, X);
  endQuda();

  finalizeComms();
  return 0;
}

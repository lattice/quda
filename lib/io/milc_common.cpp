#include <ctime>
#include <fstream>
#include <sstream>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/milc_io.h>

namespace quda
{

  namespace milc
  {

    std::string time_stamp()
    {
      time_t now = time(nullptr);
      char buf[64] = {};
      if (!ctime_r(&now, buf)) return "";
      std::string stamp(buf);
      if (!stamp.empty() && stamp.back() == '\n') stamp.pop_back();
      return stamp;
    }

    void write_info(const std::string &filename, const std::vector<std::pair<std::string, std::string>> &items)
    {
      if (comm_rank() != 0) return;
      auto info_filename = filename + ".info";
      std::ofstream out(info_filename);
      for (auto &item : items) out << item.first << " = " << item.second << "\n";
      out.close();
      if (!out) warningQuda("Could not write %s", info_filename.c_str());
    }

    namespace
    {
      std::string trim(const std::string &s)
      {
        auto first = s.find_first_not_of(" \t\r");
        if (first == std::string::npos) return std::string();
        return s.substr(first, s.find_last_not_of(" \t\r") - first + 1);
      }

      // the key of a line of a .info file and its value, unquoted; no key if the line starts with a separator
      std::pair<std::string, std::string> parse_info_line(const std::string &raw)
      {
        const auto line = trim(raw);
        auto end = line.find_first_of(" \t=");
        if (line.empty() || end == 0) return {};
        auto key = line.substr(0, end);
        auto value = end == std::string::npos ? std::string() : trim(line.substr(end));
        if (!value.empty() && value[0] == '=') value = trim(value.substr(1));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') value = value.substr(1, value.size() - 2);
        return {key, value};
      }
    } // namespace

    std::optional<std::vector<MILCInfoItem>> read_info(const std::string &filename)
    {
      int found = 0;
      std::string contents;
      if (comm_rank() == 0) {
        std::ifstream in(filename + ".info");
        if (in) {
          std::stringstream ss;
          ss << in.rdbuf();
          contents = ss.str();
          found = 1;
        }
      }
      comm_broadcast(&found, sizeof(found));
      if (!found) return std::nullopt;
      io::broadcast(contents);

      std::vector<MILCInfoItem> items;
      std::istringstream lines(contents);
      std::string line;
      while (std::getline(lines, line)) {
        if (trim(line).empty()) continue;
        items.push_back({parse_info_line(line).first, line});
      }
      return items;
    }

    std::map<std::string, std::string> info_values(const std::vector<MILCInfoItem> &items)
    {
      std::map<std::string, std::string> values;
      for (auto &item : items) {
        auto [key, value] = parse_info_line(item.line);
        if (!key.empty()) values[key] = value;
      }
      return values;
    }

  } // namespace milc

} // namespace quda

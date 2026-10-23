#include <algorithm>
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

    void write_info(const std::string &filename, const std::vector<std::pair<std::string, std::string>> &items,
                    const std::vector<MILCInfoItem> &lines, const std::vector<std::string> &dropped)
    {
      if (comm_rank() != 0) return;
      auto has_key = [](const auto &keys, const std::string &key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
      };
      std::vector<std::string> keys;
      for (auto &line : lines) keys.push_back(line.key);

      auto info_filename = filename + ".info";
      std::ofstream out(info_filename);
      for (auto &item : items)
        if (!has_key(keys, item.first)) out << item.first << " = " << item.second << "\n";
      std::vector<std::string> placed;
      for (auto &line : lines) {
        auto item = std::find_if(items.begin(), items.end(), [&](const auto &i) { return i.first == line.key; });
        if (item == items.end() && !has_key(dropped, line.key)) {
          out << line.line << "\n";
        } else if (item != items.end() && !has_key(placed, line.key)) {
          out << item->first << " = " << item->second << "\n";
          placed.push_back(line.key);
        }
      }
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

    std::vector<MILCInfoItem> info_items(const std::string &text)
    {
      std::vector<MILCInfoItem> items;
      std::istringstream lines(text);
      std::string line;
      while (std::getline(lines, line)) {
        if (trim(line).empty()) continue;
        items.push_back({parse_info_line(line).first, line});
      }
      return items;
    }

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
      return info_items(contents);
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

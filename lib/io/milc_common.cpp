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

    std::optional<std::map<std::string, std::string>> read_info(const std::string &filename)
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

      auto trim = [](const std::string &s) {
        auto first = s.find_first_not_of(" \t\r");
        if (first == std::string::npos) return std::string();
        return s.substr(first, s.find_last_not_of(" \t\r") - first + 1);
      };

      std::map<std::string, std::string> items;
      std::istringstream lines(contents);
      std::string line;
      while (std::getline(lines, line)) {
        line = trim(line);
        auto end = line.find_first_of(" \t=");
        if (line.empty() || end == 0) continue;
        auto key = line.substr(0, end);
        auto value = end == std::string::npos ? std::string() : trim(line.substr(end));
        if (!value.empty() && value[0] == '=') value = trim(value.substr(1));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') value = value.substr(1, value.size() - 2);
        items[key] = value;
      }
      return items;
    }

  } // namespace milc

} // namespace quda

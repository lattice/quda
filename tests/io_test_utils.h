#pragma once

// Helpers shared by the file IO tests (gauge_io_test and vector_io_test)

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <enum_quda.h>

inline const char *get_io_mode_str(QudaFileIOMode io_mode)
{
  return io_mode == QUDA_FILE_IO_SERIAL ? "serial" : "parallel";
}

inline std::vector<char> read_file(const std::string &file)
{
  std::ifstream in(file, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline bool write_file(const std::string &file, const std::vector<char> &contents)
{
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out.write(contents.data(), contents.size());
  return static_cast<bool>(out);
}

/**
   @brief Overwrite the time of writing in a saved file's contents: the
   time stamp of a MILC v5 gauge file (bytes 20 to 83 of the header),
   or the content of every <date> element of a LIME file (in the
   SciDAC private record XML)
   @return Whether there was a time to overwrite
 */
inline bool mask_write_time(std::vector<char> &contents, bool milc)
{
  if (milc) {
    if (contents.size() < 96) return false;
    std::fill(contents.begin() + 20, contents.begin() + 84, '#');
    return true;
  }
  const std::string open = "<date>", close = "</date>";
  bool found = false;
  for (auto it = contents.begin();;) {
    auto start = std::search(it, contents.end(), open.begin(), open.end());
    auto end = std::search(start, contents.end(), close.begin(), close.end());
    if (end == contents.end()) break;
    std::fill(start + open.size(), end, '#');
    found = true;
    it = end;
  }
  return found;
}

/**
   @brief The lines of a MILC .info file other than its time stamp
 */
inline std::vector<std::string> read_info_without_time_stamp(const std::string &file)
{
  std::ifstream in(file);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);)
    if (line.rfind("time_stamp", 0) != 0) lines.push_back(line);
  return lines;
}

/**
   @brief Expect two saved files, and the .info files of MILC files, to
   be the same apart from the time of writing
 */
inline void expect_same_saved_file(const std::string &a, const std::string &b, bool milc)
{
  auto x = read_file(a), y = read_file(b);
  EXPECT_TRUE(mask_write_time(x, milc)) << "no time of writing in " << a;
  EXPECT_TRUE(mask_write_time(y, milc)) << "no time of writing in " << b;
  EXPECT_EQ(x.size(), y.size()) << a << " and " << b << " differ in size";
  auto n = std::min(x.size(), y.size());
  auto diff = std::mismatch(x.begin(), x.begin() + n, y.begin()).first - x.begin();
  EXPECT_EQ(diff, static_cast<std::ptrdiff_t>(n)) << a << " and " << b << " first differ at byte " << diff;
  if (milc) {
    auto info_a = read_info_without_time_stamp(a + ".info"), info_b = read_info_without_time_stamp(b + ".info");
    EXPECT_FALSE(info_a.empty()) << "no " << a << ".info";
    EXPECT_EQ(info_a, info_b) << "the .info files of " << a << " and " << b << " differ";
  }
}

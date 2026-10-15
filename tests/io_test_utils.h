#pragma once

// Helpers shared by the file IO tests (gauge_io_test and vector_io_test)

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

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

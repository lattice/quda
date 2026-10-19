#pragma once

// Helpers shared by the file IO tests (gauge_io_test and vector_io_test)

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <comm_quda.h>
#include <enum_quda.h>
#include <quda.h>
#include <io/field_io.h>
#include <io/site_io.h>

inline const char *get_io_mode_str(QudaFileIOMode io_mode)
{
  return io_mode == QUDA_FILE_IO_SERIAL ? "serial" : "parallel";
}

/**
   @brief The public file options for a format, mode, I/O partition
   grid and partfile saves, without the usqcdInfo record
 */
inline QudaFileIOParam quda_file_io_param(QudaFileFormat format, QudaFileIOMode io_mode,
                                          const std::array<int, 4> &io_partition_grid = {}, bool partfile = false)
{
  QudaFileIOParam param = newQudaFileIOParam();
  param.struct_size = sizeof(param);
  param.format = format;
  param.io_mode = io_mode;
  param.write_usqcd_info = QUDA_BOOLEAN_FALSE;
  for (int d = 0; d < 4; d++) param.io_partition_grid[d] = io_partition_grid[d];
  param.partfile = partfile ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  return param;
}

/**
   @brief File options as quda_file_io_param, through the public
   interface struct (which splits the _EXTERNAL formats)
 */
inline quda::FileIOParam file_io_param(QudaFileFormat format, QudaFileIOMode io_mode,
                                       const std::array<int, 4> &io_partition_grid = {}, bool partfile = false)
{
  auto param = quda_file_io_param(format, io_mode, io_partition_grid, partfile);
  return quda::FileIOParam(&param);
}

/**
   @brief The I/O partition grids to test: every grid whose values
   divide the process grid if there are at most 8; otherwise a single
   partition, a partition spanning each split dimension alone, and
   two 2-d blocks
 */
inline std::vector<std::array<int, 4>> partition_grids()
{
  std::vector<std::array<int, 4>> all = {{1, 1, 1, 1}};
  for (int d = 0; d < 4; d++) {
    std::vector<std::array<int, 4>> more;
    for (auto grid : all) {
      for (int n = 1; n <= quda::comm_dim(d); n++) {
        if (quda::comm_dim(d) % n != 0) continue;
        grid[d] = n;
        more.push_back(grid);
      }
    }
    all = more;
  }
  if (all.size() <= 8) return all;

  const std::array<int, 4> full = {quda::comm_dim(0), quda::comm_dim(1), quda::comm_dim(2), quda::comm_dim(3)};
  std::vector<std::array<int, 4>> grids = {{1, 1, 1, 1}};
  auto add = [&](const std::array<int, 4> &grid) {
    if (grid != full && std::find(grids.begin(), grids.end(), grid) == grids.end()) grids.push_back(grid);
  };
  for (int d = 0; d < 4; d++) {
    if (full[d] == 1) continue;
    auto grid = full;
    grid[d] = 1;
    add(grid);
  }
  add({1, 1, full[2], full[3]});
  add({full[0], full[1], 1, 1});
  return grids;
}

inline std::string grid_str(const std::array<int, 4> &grid)
{
  return std::to_string(grid[0]) + " " + std::to_string(grid[1]) + " " + std::to_string(grid[2]) + " "
    + std::to_string(grid[3]);
}

/**
   @brief The name of a QIO partfile's volume for a rank: <file>.volNNNN
 */
inline std::string volume_file(const std::string &file, int rank)
{
  char suffix[16];
  snprintf(suffix, sizeof(suffix), ".vol%04d", rank);
  return file + suffix;
}

/**
   @brief File options as file_io_param, saving partfiles
 */
inline quda::FileIOParam partfile_param(QudaFileFormat format, QudaFileIOMode io_mode,
                                        const std::array<int, 4> &grid = {})
{
  return file_io_param(format, io_mode, grid, true);
}

/**
   @brief The volumes of a partfile saved on this rank layout with an
   I/O partition grid: one per partition, named after its leader (on
   one rank, QIO and QUDA save a single file instead).  Collective.
 */
inline std::vector<std::string> expected_volumes(const std::string &file, const std::array<int, 4> &grid)
{
  const quda::io::Partitions partitions(QUDA_FILE_IO_PARALLEL, grid, true);
  std::vector<std::string> volumes;
  for (int leader : partitions.leaders()) volumes.push_back(volume_file(file, leader));
  return volumes;
}

/**
   @brief The partfile volumes of a saved file in its directory, every
   <file>.vol..., in ascending order (of their numbers, which have four
   digits up to 9999)
 */
inline std::vector<std::string> saved_volumes(const std::string &file)
{
  namespace fs = std::filesystem;
  const fs::path path(file);
  const std::string prefix = path.filename().string() + ".vol";
  std::vector<std::string> volumes;
  for (auto &entry : fs::directory_iterator(path.has_parent_path() ? path.parent_path() : fs::path("."))) {
    auto name = entry.path().filename().string();
    if (name.compare(0, prefix.size(), prefix) == 0) volumes.push_back((path.parent_path() / name).string());
  }
  std::sort(volumes.begin(), volumes.end());
  return volumes;
}

/**
   @brief Delete a saved file once every rank is done with it: the
   single file, or the volumes of a partfile, which are named after
   their partitions' leaders, so their numbers can have gaps
 */
inline void remove_saved_file(const std::string &file)
{
  quda::comm_barrier();
  if (quda::comm_rank() == 0) {
    int removed = std::remove(file.c_str()) == 0;
    for (auto &volume : saved_volumes(file)) removed += std::remove(volume.c_str()) == 0;
    EXPECT_GT(removed, 0) << "no saved file " << file;
  }
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
   @brief Expect two files to be the same, byte for byte
 */
inline void expect_same_file(const std::string &a, const std::string &b)
{
  auto x = read_file(a), y = read_file(b);
  EXPECT_FALSE(x.empty()) << "no file " << a;
  EXPECT_EQ(x.size(), y.size()) << a << " and " << b << " differ in size";
  auto n = std::min(x.size(), y.size());
  auto diff = std::mismatch(x.begin(), x.begin() + n, y.begin()).first - x.begin();
  EXPECT_EQ(diff, static_cast<std::ptrdiff_t>(n)) << a << " and " << b << " first differ at byte " << diff;
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

/**
   @brief Expect two saved partfiles to have the same volumes, the
   same apart from the time of writing (only the first volume has one)
 */
inline void expect_same_partfile(const std::string &a, const std::string &b)
{
  auto u = saved_volumes(a), v = saved_volumes(b);
  ASSERT_EQ(u.size(), v.size()) << a << " and " << b << " have different numbers of volumes";
  for (auto i = 0u; i < u.size(); i++) {
    if (i == 0)
      expect_same_saved_file(u[i], v[i], false);
    else
      expect_same_file(u[i], v[i]);
  }
}

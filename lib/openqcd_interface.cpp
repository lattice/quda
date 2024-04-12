#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <regex>
#include <type_traits>

#include <quda_openqcd_interface.h>
#include <quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <index_helper.cuh>
#include <multigrid.h>
#include <mpi.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static openQCD_QudaState_t qudaState = {false, -1, -1, -1, -1, 0.0, 0.0, 0.0, {}, {}, nullptr, {}, {}, ""};

using namespace quda;

/**
 * code for NVTX taken from Jiri Kraus' blog post:
 * http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
 */

#ifdef INTERFACE_NVTX

#if QUDA_NVTX_VERSION == 3
#include "nvtx3/nvToolsExt.h"
#else
#include "nvToolsExt.h"
#endif

static const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff};
static const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                                                                          \
  {                                                                                                                    \
    int color_id = cid;                                                                                                \
    color_id = color_id % num_colors;                                                                                  \
    nvtxEventAttributes_t eventAttrib = {0};                                                                           \
    eventAttrib.version = NVTX_VERSION;                                                                                \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                                  \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                                                           \
    eventAttrib.color = colors[color_id];                                                                              \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                                                 \
    eventAttrib.message.ascii = name;                                                                                  \
    nvtxRangePushEx(&eventAttrib);                                                                                     \
  }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#define QUDA_OPENQCD_VERBOSE 1

template <bool start> void inline qudaopenqcd_called(const char *func, QudaVerbosity verb)
{
  /* add NVTX markup if enabled */
  if (start) {
    PUSH_RANGE(func, 1);
  } else {
    POP_RANGE;
  }

#ifdef QUDA_OPENQCD_VERBOSE
  if (verb >= QUDA_VERBOSE) {
    if (start) {
      printfQuda("QUDA_OPENQCD_INTERFACE: %s (called) \n", func);
    } else {
      printfQuda("QUDA_OPENQCD_INTERFACE: %s (return) \n", func);
    }
  }
#endif
}

template <bool start> void inline qudaopenqcd_called(const char *func)
{
  qudaopenqcd_called<start>(func, getVerbosity());
}

/**
 * Mapping of enums to their actual values. We have this mapping such that we
 * can use the named parameters in our input files rather than the number. this
 * makes reading and writing the configuration more understandable.
 */
std::unordered_map<std::string, std::string> enum_map
  = {{"QUDA_CG_INVERTER", std::to_string(QUDA_CG_INVERTER)},
     {"QUDA_BICGSTAB_INVERTER", std::to_string(QUDA_BICGSTAB_INVERTER)},
     {"QUDA_GCR_INVERTER", std::to_string(QUDA_GCR_INVERTER)},
     {"QUDA_MR_INVERTER", std::to_string(QUDA_MR_INVERTER)},
     {"QUDA_SD_INVERTER", std::to_string(QUDA_SD_INVERTER)},
     {"QUDA_PCG_INVERTER", std::to_string(QUDA_PCG_INVERTER)},
     {"QUDA_EIGCG_INVERTER", std::to_string(QUDA_EIGCG_INVERTER)},
     {"QUDA_INC_EIGCG_INVERTER", std::to_string(QUDA_INC_EIGCG_INVERTER)},
     {"QUDA_GMRESDR_INVERTER", std::to_string(QUDA_GMRESDR_INVERTER)},
     {"QUDA_GMRESDR_PROJ_INVERTER", std::to_string(QUDA_GMRESDR_PROJ_INVERTER)},
     {"QUDA_GMRESDR_SH_INVERTER", std::to_string(QUDA_GMRESDR_SH_INVERTER)},
     {"QUDA_FGMRESDR_INVERTER", std::to_string(QUDA_FGMRESDR_INVERTER)},
     {"QUDA_MG_INVERTER", std::to_string(QUDA_MG_INVERTER)},
     {"QUDA_BICGSTABL_INVERTER", std::to_string(QUDA_BICGSTABL_INVERTER)},
     {"QUDA_CGNE_INVERTER", std::to_string(QUDA_CGNE_INVERTER)},
     {"QUDA_CGNR_INVERTER", std::to_string(QUDA_CGNR_INVERTER)},
     {"QUDA_CG3_INVERTER", std::to_string(QUDA_CG3_INVERTER)},
     {"QUDA_CG3NE_INVERTER", std::to_string(QUDA_CG3NE_INVERTER)},
     {"QUDA_CG3NR_INVERTER", std::to_string(QUDA_CG3NR_INVERTER)},
     {"QUDA_CA_CG_INVERTER", std::to_string(QUDA_CA_CG_INVERTER)},
     {"QUDA_CA_CGNE_INVERTER", std::to_string(QUDA_CA_CGNE_INVERTER)},
     {"QUDA_CA_CGNR_INVERTER", std::to_string(QUDA_CA_CGNR_INVERTER)},
     {"QUDA_CA_GCR_INVERTER", std::to_string(QUDA_CA_GCR_INVERTER)},
     {"QUDA_INVALID_INVERTER", std::to_string(QUDA_INVALID_INVERTER)},
     {"QUDA_MAT_SOLUTION", std::to_string(QUDA_MAT_SOLUTION)},
     {"QUDA_MATDAG_MAT_SOLUTION", std::to_string(QUDA_MATDAG_MAT_SOLUTION)},
     {"QUDA_MATPC_SOLUTION", std::to_string(QUDA_MATPC_SOLUTION)},
     {"QUDA_MATPC_DAG_SOLUTION", std::to_string(QUDA_MATPC_DAG_SOLUTION)},
     {"QUDA_MATPCDAG_MATPC_SOLUTION", std::to_string(QUDA_MATPCDAG_MATPC_SOLUTION)},
     {"QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION", std::to_string(QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION)},
     {"QUDA_INVALID_SOLUTION", std::to_string(QUDA_INVALID_SOLUTION)},
     {"QUDA_DIRECT_SOLVE", std::to_string(QUDA_DIRECT_SOLVE)},
     {"QUDA_NORMOP_SOLVE", std::to_string(QUDA_NORMOP_SOLVE)},
     {"QUDA_DIRECT_PC_SOLVE", std::to_string(QUDA_DIRECT_PC_SOLVE)},
     {"QUDA_NORMOP_PC_SOLVE", std::to_string(QUDA_NORMOP_PC_SOLVE)},
     {"QUDA_NORMERR_SOLVE", std::to_string(QUDA_NORMERR_SOLVE)},
     {"QUDA_NORMERR_PC_SOLVE", std::to_string(QUDA_NORMERR_PC_SOLVE)},
     {"QUDA_NORMEQ_SOLVE", std::to_string(QUDA_NORMEQ_SOLVE)},
     {"QUDA_NORMEQ_PC_SOLVE", std::to_string(QUDA_NORMEQ_PC_SOLVE)},
     {"QUDA_INVALID_SOLVE", std::to_string(QUDA_INVALID_SOLVE)},
     {"QUDA_MATPC_EVEN_EVEN", std::to_string(QUDA_MATPC_EVEN_EVEN)},
     {"QUDA_MATPC_ODD_ODD", std::to_string(QUDA_MATPC_ODD_ODD)},
     {"QUDA_MATPC_EVEN_EVEN_ASYMMETRIC", std::to_string(QUDA_MATPC_EVEN_EVEN_ASYMMETRIC)},
     {"QUDA_MATPC_ODD_ODD_ASYMMETRIC", std::to_string(QUDA_MATPC_ODD_ODD_ASYMMETRIC)},
     {"QUDA_MATPC_INVALID", std::to_string(QUDA_MATPC_INVALID)},
     {"QUDA_DEFAULT_NORMALIZATION", std::to_string(QUDA_DEFAULT_NORMALIZATION)},
     {"QUDA_SOURCE_NORMALIZATION", std::to_string(QUDA_SOURCE_NORMALIZATION)},
     {"QUDA_QUARTER_PRECISION", std::to_string(QUDA_QUARTER_PRECISION)},
     {"QUDA_HALF_PRECISION", std::to_string(QUDA_HALF_PRECISION)},
     {"QUDA_SINGLE_PRECISION", std::to_string(QUDA_SINGLE_PRECISION)},
     {"QUDA_DOUBLE_PRECISION", std::to_string(QUDA_DOUBLE_PRECISION)},
     {"QUDA_INVALID_PRECISION", std::to_string(QUDA_INVALID_PRECISION)},
     {"QUDA_BOOLEAN_FALSE", std::to_string(QUDA_BOOLEAN_FALSE)},
     {"false", std::to_string(QUDA_BOOLEAN_FALSE)},
     {"FALSE", std::to_string(QUDA_BOOLEAN_FALSE)},
     {"no", std::to_string(QUDA_BOOLEAN_FALSE)},
     {"n", std::to_string(QUDA_BOOLEAN_FALSE)},
     {"off", std::to_string(QUDA_BOOLEAN_FALSE)},
     {"QUDA_BOOLEAN_TRUE", std::to_string(QUDA_BOOLEAN_TRUE)},
     {"true", std::to_string(QUDA_BOOLEAN_TRUE)},
     {"TRUE", std::to_string(QUDA_BOOLEAN_TRUE)},
     {"yes", std::to_string(QUDA_BOOLEAN_TRUE)},
     {"y", std::to_string(QUDA_BOOLEAN_TRUE)},
     {"on", std::to_string(QUDA_BOOLEAN_TRUE)},
     {"QUDA_BOOLEAN_INVALID", std::to_string(QUDA_BOOLEAN_INVALID)},
     {"QUDA_COMPUTE_NULL_VECTOR_NO", std::to_string(QUDA_COMPUTE_NULL_VECTOR_NO)},
     {"QUDA_COMPUTE_NULL_VECTOR_YES", std::to_string(QUDA_COMPUTE_NULL_VECTOR_YES)},
     {"QUDA_COMPUTE_NULL_VECTOR_INVALID", std::to_string(QUDA_COMPUTE_NULL_VECTOR_INVALID)},
     {"QUDA_MG_CYCLE_VCYCLE", std::to_string(QUDA_MG_CYCLE_VCYCLE)},
     {"QUDA_MG_CYCLE_FCYCLE", std::to_string(QUDA_MG_CYCLE_FCYCLE)},
     {"QUDA_MG_CYCLE_WCYCLE", std::to_string(QUDA_MG_CYCLE_WCYCLE)},
     {"QUDA_MG_CYCLE_RECURSIVE", std::to_string(QUDA_MG_CYCLE_RECURSIVE)},
     {"QUDA_MG_CYCLE_INVALID", std::to_string(QUDA_MG_CYCLE_INVALID)},
     {"QUDA_CPU_FIELD_LOCATION", std::to_string(QUDA_CPU_FIELD_LOCATION)},
     {"QUDA_CUDA_FIELD_LOCATION", std::to_string(QUDA_CUDA_FIELD_LOCATION)},
     {"QUDA_INVALID_FIELD_LOCATION", std::to_string(QUDA_INVALID_FIELD_LOCATION)},
     {"QUDA_TWIST_SINGLET", std::to_string(QUDA_TWIST_SINGLET)},
     {"QUDA_TWIST_NONDEG_DOUBLET", std::to_string(QUDA_TWIST_NONDEG_DOUBLET)},
     {"QUDA_TWIST_NO", std::to_string(QUDA_TWIST_NO)},
     {"QUDA_TWIST_INVALID", std::to_string(QUDA_TWIST_INVALID)},
     {"QUDA_DAG_NO", std::to_string(QUDA_DAG_NO)},
     {"QUDA_DAG_YES", std::to_string(QUDA_DAG_YES)},
     {"QUDA_DAG_INVALID", std::to_string(QUDA_DAG_INVALID)},
     {"QUDA_KAPPA_NORMALIZATION", std::to_string(QUDA_KAPPA_NORMALIZATION)},
     {"QUDA_MASS_NORMALIZATION", std::to_string(QUDA_MASS_NORMALIZATION)},
     {"QUDA_ASYMMETRIC_MASS_NORMALIZATION", std::to_string(QUDA_ASYMMETRIC_MASS_NORMALIZATION)},
     {"QUDA_INVALID_NORMALIZATION", std::to_string(QUDA_INVALID_NORMALIZATION)},
     {"QUDA_PRESERVE_SOURCE_NO", std::to_string(QUDA_PRESERVE_SOURCE_NO)},
     {"QUDA_PRESERVE_SOURCE_YES", std::to_string(QUDA_PRESERVE_SOURCE_YES)},
     {"QUDA_PRESERVE_SOURCE_INVALID", std::to_string(QUDA_PRESERVE_SOURCE_INVALID)},
     {"QUDA_USE_INIT_GUESS_NO", std::to_string(QUDA_USE_INIT_GUESS_NO)},
     {"QUDA_USE_INIT_GUESS_YES", std::to_string(QUDA_USE_INIT_GUESS_YES)},
     {"QUDA_USE_INIT_GUESS_INVALID", std::to_string(QUDA_USE_INIT_GUESS_INVALID)},
     {"QUDA_SILENT", std::to_string(QUDA_SILENT)},
     {"QUDA_SUMMARIZE", std::to_string(QUDA_SUMMARIZE)},
     {"QUDA_VERBOSE", std::to_string(QUDA_VERBOSE)},
     {"QUDA_DEBUG_VERBOSE", std::to_string(QUDA_DEBUG_VERBOSE)},
     {"QUDA_INVALID_VERBOSITY", std::to_string(QUDA_INVALID_VERBOSITY)},
     {"QUDA_TUNE_NO", std::to_string(QUDA_TUNE_NO)},
     {"QUDA_TUNE_YES", std::to_string(QUDA_TUNE_YES)},
     {"QUDA_TUNE_INVALID", std::to_string(QUDA_TUNE_INVALID)},
     {"QUDA_POWER_BASIS", std::to_string(QUDA_POWER_BASIS)},
     {"QUDA_CHEBYSHEV_BASIS", std::to_string(QUDA_CHEBYSHEV_BASIS)},
     {"QUDA_INVALID_BASIS", std::to_string(QUDA_INVALID_BASIS)},
     {"QUDA_ADDITIVE_SCHWARZ", std::to_string(QUDA_ADDITIVE_SCHWARZ)},
     {"QUDA_MULTIPLICATIVE_SCHWARZ", std::to_string(QUDA_MULTIPLICATIVE_SCHWARZ)},
     {"QUDA_INVALID_SCHWARZ", std::to_string(QUDA_INVALID_SCHWARZ)},
     {"QUDA_MADWF_ACCELERATOR", std::to_string(QUDA_MADWF_ACCELERATOR)},
     {"QUDA_INVALID_ACCELERATOR", std::to_string(QUDA_INVALID_ACCELERATOR)},
     {"QUDA_L2_RELATIVE_RESIDUAL", std::to_string(QUDA_L2_RELATIVE_RESIDUAL)},
     {"QUDA_L2_ABSOLUTE_RESIDUAL", std::to_string(QUDA_L2_ABSOLUTE_RESIDUAL)},
     {"QUDA_HEAVY_QUARK_RESIDUAL", std::to_string(QUDA_HEAVY_QUARK_RESIDUAL)},
     {"QUDA_INVALID_RESIDUAL", std::to_string(QUDA_INVALID_RESIDUAL)},
     {"QUDA_NULL_VECTOR_SETUP", std::to_string(QUDA_NULL_VECTOR_SETUP)},
     {"QUDA_TEST_VECTOR_SETUP", std::to_string(QUDA_TEST_VECTOR_SETUP)},
     {"QUDA_INVALID_SETUP_TYPE", std::to_string(QUDA_INVALID_SETUP_TYPE)},
     {"QUDA_TRANSFER_AGGREGATE", std::to_string(QUDA_TRANSFER_AGGREGATE)},
     {"QUDA_TRANSFER_COARSE_KD", std::to_string(QUDA_TRANSFER_COARSE_KD)},
     {"QUDA_TRANSFER_OPTIMIZED_KD", std::to_string(QUDA_TRANSFER_OPTIMIZED_KD)},
     {"QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG", std::to_string(QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)},
     {"QUDA_TRANSFER_INVALID", std::to_string(QUDA_TRANSFER_INVALID)},
     {"QUDA_EIG_TR_LANCZOS", std::to_string(QUDA_EIG_TR_LANCZOS)},
     {"QUDA_EIG_BLK_TR_LANCZOS", std::to_string(QUDA_EIG_BLK_TR_LANCZOS)},
     {"QUDA_EIG_IR_ARNOLDI", std::to_string(QUDA_EIG_IR_ARNOLDI)},
     {"QUDA_EIG_BLK_IR_ARNOLDI", std::to_string(QUDA_EIG_BLK_IR_ARNOLDI)},
     {"QUDA_EIG_INVALID", std::to_string(QUDA_EIG_INVALID)},
     {"QUDA_SPECTRUM_LM_EIG", std::to_string(QUDA_SPECTRUM_LM_EIG)},
     {"QUDA_SPECTRUM_SM_EIG", std::to_string(QUDA_SPECTRUM_SM_EIG)},
     {"QUDA_SPECTRUM_LR_EIG", std::to_string(QUDA_SPECTRUM_LR_EIG)},
     {"QUDA_SPECTRUM_SR_EIG", std::to_string(QUDA_SPECTRUM_SR_EIG)},
     {"QUDA_SPECTRUM_LI_EIG", std::to_string(QUDA_SPECTRUM_LI_EIG)},
     {"QUDA_SPECTRUM_SI_EIG", std::to_string(QUDA_SPECTRUM_SI_EIG)},
     {"QUDA_SPECTRUM_INVALID", std::to_string(QUDA_SPECTRUM_INVALID)},
     {"QUDA_MEMORY_DEVICE", std::to_string(QUDA_MEMORY_DEVICE)},
     {"QUDA_MEMORY_DEVICE_PINNED", std::to_string(QUDA_MEMORY_DEVICE_PINNED)},
     {"QUDA_MEMORY_HOST", std::to_string(QUDA_MEMORY_HOST)},
     {"QUDA_MEMORY_HOST_PINNED", std::to_string(QUDA_MEMORY_HOST_PINNED)},
     {"QUDA_MEMORY_MAPPED", std::to_string(QUDA_MEMORY_MAPPED)},
     {"QUDA_MEMORY_MANAGED", std::to_string(QUDA_MEMORY_MANAGED)},
     {"QUDA_MEMORY_INVALID", std::to_string(QUDA_MEMORY_INVALID)},
     {"QUDA_CUSOLVE_EXTLIB", std::to_string(QUDA_CUSOLVE_EXTLIB)},
     {"QUDA_EIGEN_EXTLIB", std::to_string(QUDA_EIGEN_EXTLIB)},
     {"QUDA_EXTLIB_INVALID", std::to_string(QUDA_EXTLIB_INVALID)}};

/**
 * @brief      Just a simple key-value store
 */
class KeyValueStore
{
private:
  std::unordered_map<std::string, std::unordered_map<std::string, std::tuple<std::string, std::string>>> store;
  std::unordered_map<std::string, std::string> *map = nullptr;
  std::string filename = "<not set>";

public:
  /**
   * @brief      Sets a key value pair
   *
   * @param[in]  section  The section
   * @param[in]  key      The key
   * @param[in]  value    The value
   */
  void set(const std::string &section, const std::string &key, const std::string &value)
  {
    if (map != nullptr) {
      auto mvalue = map->find(value);
      if (mvalue != map->end()) {
        std::get<0>(store[section][key]) = mvalue->second;
        std::get<1>(store[section][key]) = value;
        return;
      }
    }
    std::get<0>(store[section][key]) = value;
    std::get<1>(store[section][key]) = value;
  }

  void set_map(std::unordered_map<std::string, std::string> *_map) { map = _map; }

  bool section_exists(const std::string &section) { return store.find(section) != store.end(); }

  /**
   * @brief      Gets the specified key.
   *
   * @param[in]  section        The section
   * @param[in]  key            The key
   * @param[in]  default_value  The default value if section/key is absent
   *
   * @tparam     T              Desired return type
   *
   * @return     The corresponding value
   */
  template <typename T>
  T get(const std::string &section, const std::string &key, T default_value = T(), bool fail = false)
  {
    int idx;
    std::string rkey;
    std::smatch match;
    std::regex p_key("([^\\[]+)\\[(\\d+)\\]"); /* key[idx] */
    auto sec = store.find(section);

    if (sec != store.end()) {
      if (std::regex_search(key, match, p_key)) {
        rkey = match[1];
        idx = std::stoi(match[2]);
      } else {
        rkey = key;
        idx = 0;
      }

      auto item = sec->second.find(rkey);
      if (item != sec->second.end()) {
        std::stringstream ss(std::get<0>(item->second));
        if constexpr (std::is_enum_v<T>) {
          typename std::underlying_type<T>::type result, dummy;
          for (int i = 0; i < idx; i++) { ss >> dummy; }
          if (ss >> result) { return static_cast<T>(result); }
        } else {
          T result, dummy;
          for (int i = 0; i < idx; i++) { ss >> dummy; }
          if (ss >> result) { return result; }
        }
      }
    }
    if (fail) {
      errorQuda("Key \"%s\" in section \"%s\" in file %s does not exist.", key.c_str(), section.c_str(),
                filename.c_str());
    }
    return default_value; /* Return default value for non-existent keys */
  }

  /**
   * @brief      Fill the store with entries from an ini-file
   *
   * @param[in]  fname  The fname
   */
  void load(const std::string &fname)
  {
    std::string line, section;
    std::smatch match;
    filename = fname;
    std::ifstream file(filename.c_str());

    std::regex p_section("^\\s*\\[([\\w\\ ]+)\\].*$");      /* [section] */
    std::regex p_comment("^[^#]*(\\s*#.*)$");               /* line # comment */
    std::regex p_key_val("^([^\\s]+)\\s+(.*[^\\s]+)\\s*$"); /* key value */

    if (file.is_open()) {

      while (std::getline(file, line)) {

        /* remove all comments */
        if (std::regex_search(line, match, p_comment)) { line.erase(match.position(1)); }

        if (std::regex_search(line, match, p_section)) {
          section = match[1];
        } else if (std::regex_search(line, match, p_key_val)) {
          std::string key = match[1];
          std::string val = match[2];
          this->set(section, key, val);
        }
      }

      file.close();
    } else {
      std::cerr << "Error opening file: " << filename << std::endl;
    }
  }

  /**
   * @brief      Dumps all entries in the store.
   */
  void dump(std::string _section = "")
  {
    for (const auto &section : store) {
      if (_section == "" || _section == section.first) {
        std::cout << "[" << section.first << "]" << std::endl;
        for (const auto &pair : section.second) {
          std::cout << "  " << pair.first << " = " << std::get<1>(pair.second);
          if (std::get<0>(pair.second) != std::get<1>(pair.second)) { std::cout << " # " << std::get<0>(pair.second); }
          std::cout << std::endl;
        }
      }
    }
  }
};

/**
 * @brief      Returns the local lattice dimensions as lat_dim_t
 *
 * @return     The local dimensions.
 */
static lat_dim_t get_local_dims(int *fill = nullptr)
{
  lat_dim_t X;

  for (int i = 0; i < 4; i++) {
    if (fill) {
      fill[i] = qudaState.layout.L[i];
    } else {
      X[i] = qudaState.layout.L[i];
    }
  }

  return X;
}

/**
 * @brief      Calculate the rank from coordinates.
 *
 * @param[in]  coords  coords is the 4D cartesian coordinate of a rank
 * @param[in]  fdata   should point to an instance of qudaLayout.ranks,
 *                     @see struct openQCD_QudaLayout_t in
 *                     @file include/quda_openqcd_interface.h
 *
 * @return     rank
 */
int rankFromCoords(const int *coords, void *fdata)
{
  int *base = static_cast<int *>(fdata);
  int *NPROC = base + 1;
  int *ranks = base + 5;
  int i;

  i = coords[3] + NPROC[3] * (coords[2] + NPROC[2] * (coords[1] + NPROC[1] * (coords[0])));
  return ranks[i];
}

/**
 * Set set the local dimensions and machine topology for QUDA to use
 *
 * @param      layout  Struct defining local dimensions and machine topology
 * @param      infile  Input file
 */
void openQCD_qudaSetLayout(openQCD_QudaLayout_t layout, char *infile)
{
  int my_rank;
  char prefix[20];

  for (int dir = 0; dir < 4; ++dir) {
    if (layout.N[dir] % 2 != 0) {
      errorQuda("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }

#ifdef MULTI_GPU
/* TODO: would we ever want to run with QMP COMMS? */
#ifdef QMP_COMMS
  initCommsGridQuda(4, layout.nproc, nullptr, nullptr);
#else
  initCommsGridQuda(4, layout.nproc, rankFromCoords, (void *)(layout.data));
#endif
  static int device = -1; /* enable a default allocation of devices to processes */
#else
  static int device = layout.device;
#endif

  /* must happen *after* communication initialization */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  sprintf(prefix, "QUDA (rank=%d): ", my_rank);

  strcpy(qudaState.infile, infile);
  if (my_rank == 0) {
    KeyValueStore kv;
    kv.set_map(&enum_map);
    kv.load(qudaState.infile);
    qudaState.init.verbosity = kv.get<QudaVerbosity>("QUDA", "verbosity", qudaState.init.verbosity);
  }

  MPI_Bcast((void *)&qudaState.init.verbosity, sizeof(qudaState.init.verbosity), MPI_INT, 0, MPI_COMM_WORLD);
  setVerbosityQuda(qudaState.init.verbosity, prefix, qudaState.init.logfile);
  initQuda(device);
}

static int getLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1] * dim[2] * dim[3] / 2, dim[0] * dim[2] * dim[3] / 2);
  padding = MAX(padding, dim[0] * dim[1] * dim[3] / 2);
  padding = MAX(padding, dim[0] * dim[1] * dim[2] / 2);
  return padding;
}

/**
 * @brief      Creates a new quda parameter struct
 *
 * @return     The quda parameter struct.
 */
static QudaInvertParam newOpenQCDParam(void)
{
  static const QudaVerbosity verbosity = getVerbosity();

  QudaInvertParam param = newQudaInvertParam();

  param.verbosity = verbosity;

  param.cpu_prec = QUDA_DOUBLE_PRECISION;              /* The precision used by the input fermion fields */
  param.cuda_prec = QUDA_DOUBLE_PRECISION;             /* The precision used by the QUDA solver */
  param.cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION; /* The precision used by the QUDA eigensolver */

  param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;     /* The precision used by the QUDA solver */
  param.cuda_prec_precondition = QUDA_HALF_PRECISION; /* The precision used by the QUDA solver */

  /**
   * The order of the input and output fermion fields. Imposes fieldOrder =
   * QUDA_OPENQCD_FIELD_ORDER in color_spinor_field.h and
   * QUDA_OPENQCD_FIELD_ORDER makes quda to instantiate OpenQCDDiracOrder.
   */
  param.dirac_order = QUDA_OPENQCD_DIRAC_ORDER;

  /**
   * Gamma basis of the input and output host fields. Specifies the basis change
   * into QUDAs internal gamma basis. Note that QUDA applies the basis change U
   * to a spinor field when uploading and U^dagger when downloading.
   */
  param.gamma_basis = QUDA_OPENQCD_GAMMA_BASIS;

  return param;
}

/**
 * @brief      Initialize quda gauge param struct
 *
 * @param[in]  prec        Precision
 * @param[in]  rec         QUDA internal gauge field format
 * @param[in]  t_boundary  Time boundary condition
 *
 * @return     The quda gauge parameter struct.
 */
static QudaGaugeParam newOpenQCDGaugeParam(QudaPrecision prec, QudaReconstructType rec, QudaTboundary t_boundary)
{
  QudaGaugeParam param = newQudaGaugeParam();

  get_local_dims(param.X);
  param.cuda_prec_sloppy = param.cpu_prec = param.cuda_prec = prec;
  param.type = QUDA_SU3_LINKS;

  param.reconstruct_sloppy = param.reconstruct = rec;

  /* This makes quda to instantiate OpenQCDOrder */
  param.gauge_order = QUDA_OPENQCD_GAUGE_ORDER;

  param.t_boundary = t_boundary;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  param.scale = 1.0;
  param.anisotropy = 1.0;                 /* 1.0 means not anisotropic */
  param.ga_pad = getLinkPadding(param.X); /* Why this? */

  return param;
}

void openQCD_qudaInit(openQCD_QudaInitArgs_t init, openQCD_QudaLayout_t layout, char *infile)
{
  if (qudaState.initialized) return;
  qudaState.init = init;
  qudaState.layout = layout;

  qudaopenqcd_called<true>(__func__);
  openQCD_qudaSetLayout(qudaState.layout, infile);
  qudaopenqcd_called<false>(__func__);
  qudaState.initialized = true;
}

void openQCD_qudaFinalize(void)
{
  for (int id = 0; id < OPENQCD_MAX_INVERTERS; ++id) {
    if (qudaState.inv_handles[id] != nullptr) { openQCD_qudaSolverDestroy(id); }
  }

  for (int id = 0; id < OPENQCD_MAX_EIGENSOLVERS; ++id) {
    if (qudaState.eig_handles[id] != nullptr) { openQCD_qudaEigensolverDestroy(id); }
  }

  qudaState.initialized = false;
  endQuda();
}

double openQCD_qudaPlaquette(void)
{
  double plaq[3];

  plaqQuda(plaq);

  /* Note different Nc normalization wrt openQCD! */
  return 3.0 * plaq[0];
}

void openQCD_qudaGaugeLoad(void *gauge, QudaPrecision prec, QudaReconstructType rec, QudaTboundary t_boundary)
{
  QudaGaugeParam param = newOpenQCDGaugeParam(prec, rec, t_boundary);
  loadGaugeQuda(gauge, &param);
}

void openQCD_qudaGaugeSave(void *gauge, QudaPrecision prec, QudaReconstructType rec, QudaTboundary t_boundary)
{
  QudaGaugeParam param = newOpenQCDGaugeParam(prec, rec, t_boundary);

  void *buffer = pool_pinned_malloc((4 * qudaState.init.volume + 7 * qudaState.init.bndry / 4) * 18 * prec);
  saveGaugeQuda(buffer, &param);
  qudaState.init.reorder_gauge_quda_to_openqcd(buffer, gauge);
  pool_pinned_free(buffer);
}

void openQCD_qudaGaugeFree(void) { freeGaugeQuda(); }

void openQCD_qudaCloverLoad(void *clover, double kappa, double csw)
{
  QudaInvertParam param = newOpenQCDParam();
  param.clover_order = QUDA_OPENQCD_CLOVER_ORDER;
  param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
  param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;
  param.clover_cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;

  param.kappa = kappa;
  param.clover_csw = csw;
  param.clover_coeff = 0.0;

  loadCloverQuda(clover, NULL, &param);
}

void openQCD_qudaCloverFree(void) { freeCloverQuda(); }

/**
 * @brief      Set the su3csw corfficient and all related properties.
 *
 * @param      param   The parameter struct
 * @param[in]  su3csw  The su3csw coefficient
 */
inline void set_su3csw(QudaInvertParam *param, double su3csw)
{
  param->clover_csw = su3csw;
  if (su3csw != 0.0) {
    param->clover_location = QUDA_CUDA_FIELD_LOCATION; /* seems to have no effect? */
    param->clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    param->clover_cuda_prec = QUDA_DOUBLE_PRECISION;
    param->clover_cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;

    param->clover_coeff = 0.0;

    /* Set to Wilson Dirac operator with Clover term */
    param->dslash_type = QUDA_CLOVER_WILSON_DSLASH;

    if (qudaState.layout.flds_parms().gauge == OPENQCD_GAUGE_SU3) {
      param->clover_order = QUDA_FLOAT8_CLOVER_ORDER; /* what implication has this? */
      param->compute_clover = true;
    } else {
      param->clover_order = QUDA_OPENQCD_CLOVER_ORDER;
    }
  }
}

/**
 * @brief      Creates a new quda Dirac parameter struct
 *
 * @param[in]  p     OpenQCD Dirac parameter struct
 *
 * @return     The quda Dirac parameter struct.
 */
static QudaInvertParam newOpenQCDDiracParam(openQCD_QudaDiracParam_t p)
{
  QudaInvertParam param = newOpenQCDParam();

  param.dslash_type = QUDA_WILSON_DSLASH;
  param.kappa = p.kappa;
  param.mu = p.mu;
  param.dagger = QUDA_DAG_NO;

  set_su3csw(&param, p.su3csw);

  param.inv_type = QUDA_CG_INVERTER; /* just set some, needed? */

  /* What is the difference? only works with QUDA_MASS_NORMALIZATION */
  param.mass_normalization = QUDA_MASS_NORMALIZATION;

  /* Extent of the 5th dimension (for domain wall) */
  param.Ls = 1;

  return param;
}

void openQCD_back_and_forth(void *h_in, void *h_out)
{
  /* sets up the necessary parameters */
  QudaInvertParam param = newOpenQCDParam();

  /* creates a field on the CPU */
  ColorSpinorParam cpuParam(h_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  /* creates a field on the GPU with the same parameter set as the CPU field */
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  /* transfer the CPU field to GPU */
  in = in_h;

  /* creates a field on the CPU */
  cpuParam.v = h_out;
  cpuParam.location = QUDA_CPU_FIELD_LOCATION;
  ColorSpinorField out_h(cpuParam);

  /* creates a zero-field on the GPU */
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

  out = in;

  /* transfer the GPU field back to CPU */
  out_h = out;
}

int openQCD_qudaIndexIpt(const int *x)
{
  int L_openqcd[4];
  openqcd::rotate_coords(qudaState.layout.L, L_openqcd);
  return openqcd::ipt(x, L_openqcd);
}

int openQCD_qudaIndexIup(const int *x, const int mu)
{
  int L_openqcd[4], nproc_openqcd[4];
  openqcd::rotate_coords(qudaState.layout.L, L_openqcd);
  openqcd::rotate_coords(qudaState.layout.nproc, nproc_openqcd);
  return openqcd::iup(x, mu, L_openqcd, nproc_openqcd);
}

double openQCD_qudaNorm(void *h_in)
{
  QudaInvertParam param = newOpenQCDParam();

  ColorSpinorParam cpuParam(h_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  in = in_h;

  return blas::norm2(in);
}

double openQCD_qudaNorm_NoLoads(void *d_in) { return blas::norm2(*reinterpret_cast<ColorSpinorField *>(d_in)); }

void openQCD_qudaGamma(const int dir, void *openQCD_in, void *openQCD_out)
{
  /* sets up the necessary parameters */
  QudaInvertParam param = newOpenQCDParam();

  /* creates a field on the CPU */
  ColorSpinorParam cpuParam(openQCD_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  /* creates a field on the GPU with the same parameter set as the CPU field */
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  /* transfer the CPU field to GPU */
  in = in_h;

  /* creates a zero-field on the GPU */
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

  /* gamma_i run within QUDA using QUDA fields */
  switch (dir) {
  case 0: /* t direction */ gamma3(out, in); break;
  case 1: /* x direction */ gamma0(out, in); break;
  case 2: /* y direction */ gamma1(out, in); break;
  case 3: /* z direction */ gamma2(out, in); break;
  case 4:
  case 5:
    gamma5(out, in);
    /* UKQCD uses a different convention for Gamma matrices:
     * gamma5_ukqcd = gammax gammay gammaz gammat,
     * gamma5_openqcd = gammat gammax gammay gammaz,
     * and thus
     * gamma5_openqcd = -1 * U gamma5_ukqcd U^dagger,
     * with U the transformation matrix from OpenQCD to UKQCD. */
    blas::ax(-1.0, out);
    break;
  default: errorQuda("Unknown gamma: %d\n", dir);
  }

  /* creates a field on the CPU */
  cpuParam.v = openQCD_out;
  cpuParam.location = QUDA_CPU_FIELD_LOCATION;
  ColorSpinorField out_h(cpuParam);

  /* transfer the GPU field back to CPU */
  out_h = out;
}

void *openQCD_qudaH2D(void *openQCD_field)
{
  /* sets up the necessary parameters */
  QudaInvertParam param = newOpenQCDParam();

  /* creates a field on the CPU */
  ColorSpinorParam cpuParam(openQCD_field, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  /* creates a field on the GPU with the same parameter set as the CPU field */
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField *in = new ColorSpinorField(cudaParam);

  *in = in_h; /* transfer the CPU field to GPU */

  return in;
}

void openQCD_qudaSpinorFree(void **quda_field)
{
  delete reinterpret_cast<ColorSpinorField *>(*quda_field);
  *quda_field = nullptr;
}

void openQCD_qudaD2H(void *quda_field, void *openQCD_field)
{
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* sets up the necessary parameters */
  QudaInvertParam param = newOpenQCDParam();

  /* creates a field on the CPU */
  ColorSpinorParam cpuParam(openQCD_field, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField out_h(cpuParam);

  /* transfer the GPU field to CPU */
  out_h = *reinterpret_cast<ColorSpinorField *>(quda_field);
}

/**
 * @brief      Check whether the gauge field from openQCD is in sync with the
 *             one from QUDA.
 *
 * @return     true/false
 */
inline bool gauge_field_get_up2date(void)
{
  int ud_rev = -2, ad_rev = -2;

  /* get current residing gauge field revision (residing in openqxd) */
  qudaState.layout.get_gfld_flags(&ud_rev, &ad_rev);
  return ud_rev == qudaState.ud_rev && ad_rev == qudaState.ad_rev;
}

/**
 * @brief      Check whether the gauge field is not (yet) set in openQCD.
 *
 * @return     true/false
 */
inline bool gauge_field_get_unset(void)
{
  int ud_rev = -2, ad_rev = -2;
  qudaState.layout.get_gfld_flags(&ud_rev, &ad_rev);
  return ud_rev == 0 && ad_rev == 0;
}

/**
 * @brief      Check if the current SW field needs to update wrt the parameters from openQCD.
 *
 * @return     true/false
 */
inline bool clover_field_get_up2date(void)
{
  return (gauge_field_get_up2date() && qudaState.swd_ud_rev == qudaState.ud_rev && qudaState.swd_ad_rev == qudaState.ad_rev
          && qudaState.swd_kappa == 1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0))
          && qudaState.swd_su3csw == qudaState.layout.dirac_parms().su3csw
          && qudaState.swd_u1csw == qudaState.layout.dirac_parms().u1csw);
}

/**
 * @brief      Check whether the multigrid instance associated to the parameter
 *             struct is up to date with the global gauge field revision,
 *             parameters are in sync, and clover/gauge fields are up to date.
 *
 * @param      param  The parameter struct
 *
 * @return     true/false
 */
inline bool mg_get_up2date(QudaInvertParam *param)
{
  openQCD_QudaSolver *additional_prop = static_cast<openQCD_QudaSolver *>(param->additional_prop);

  return (param->preconditioner != nullptr && gauge_field_get_up2date() && clover_field_get_up2date()
          && additional_prop->mg_ud_rev == qudaState.ud_rev && additional_prop->mg_ad_rev == qudaState.ad_rev
          && additional_prop->mg_kappa == 1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0))
          && additional_prop->mg_su3csw == qudaState.layout.dirac_parms().su3csw
          && additional_prop->mg_u1csw == qudaState.layout.dirac_parms().u1csw);
}

/**
 * @brief      Sets the multigrid instance associated to the parameter struct to
 *             be in sync with openQxD.
 *
 * @param      param  The parameter struct
 */
inline void mg_set_revision(QudaInvertParam *param)
{
  openQCD_QudaSolver *additional_prop = static_cast<openQCD_QudaSolver *>(param->additional_prop);

  additional_prop->mg_ud_rev = qudaState.ud_rev;
  additional_prop->mg_ad_rev = qudaState.ad_rev;
  additional_prop->mg_kappa = 1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0));
  additional_prop->mg_su3csw = qudaState.layout.dirac_parms().su3csw;
  additional_prop->mg_u1csw = qudaState.layout.dirac_parms().u1csw;
}

/**
 * @brief      Set the global revisions numners for the SW field.
 */
inline void clover_field_set_revision(void)
{
  qudaState.swd_ud_rev = qudaState.ud_rev;
  qudaState.swd_ad_rev = qudaState.ad_rev;
  qudaState.swd_kappa = 1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0));
  qudaState.swd_su3csw = qudaState.layout.dirac_parms().su3csw;
  qudaState.swd_u1csw = qudaState.layout.dirac_parms().u1csw;
}

/**
 * @brief      Set the global revisions numners for the gauge field.
 */
inline void gauge_field_set_revision(void) { qudaState.layout.get_gfld_flags(&qudaState.ud_rev, &qudaState.ad_rev); }

/**
 * @brief      Check if the solver parameters are in sync with the parameters
 *             from openQCD.
 *
 * @param      param_  The parameter struct
 *
 * @return     Whether parameters are in sync or not
 */
int openQCD_qudaInvertParamCheck(void *param_)
{
  QudaInvertParam *param = static_cast<QudaInvertParam *>(param_);
  openQCD_QudaSolver *additional_prop = static_cast<openQCD_QudaSolver *>(param->additional_prop);
  bool ret = true;

  if (param->kappa != (1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0)))) {
    logQuda(
      QUDA_VERBOSE, "Property m0/kappa does not match in QudaInvertParam struct and openQxD:dirac_parms (openQxD: %.6e, QUDA: %.6e)\n",
      (1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0))), param->kappa);
    ret = false;
  }

  if (additional_prop->u1csw != qudaState.layout.dirac_parms().u1csw) {
    logQuda(
      QUDA_VERBOSE,
      "Property u1csw does not match in QudaInvertParam struct and openQxD:dirac_parms (openQxD: %.6e, QUDA: %.6e)\n",
      qudaState.layout.dirac_parms().u1csw, additional_prop->u1csw);
    ret = false;
  }

  if (param->clover_csw != qudaState.layout.dirac_parms().su3csw) {
    logQuda(
      QUDA_VERBOSE, "Property su3csw/clover_csw does not match in QudaInvertParam struct and openQxD:dirac_parms (openQxD: %.6e, QUDA: %.6e)\n",
      qudaState.layout.dirac_parms().su3csw, param->clover_csw);
    ret = false;
  }

  return ret;
}

/**
 * @brief      Check if the solver identifier is in bounds
 *
 * @param[in]  id    The identifier
 */
void inline check_solver_id(int id)
{
  if (id < -1 || id > OPENQCD_MAX_INVERTERS - 1) {
    errorQuda("Solver id %d is out of range [%d, %d).", id, -1, OPENQCD_MAX_INVERTERS);
  }
}

/**
 * @brief      Check if the eigen-solver identifier is in bounds
 *
 * @param[in]  id    The identifier
 */
void inline check_eigensolver_id(int id)
{
  if (id < 0 || id > OPENQCD_MAX_EIGENSOLVERS - 1) {
    errorQuda("Eigensolver id %d is out of range [%d, %d).", id, 0, OPENQCD_MAX_EIGENSOLVERS);
  }
}

/**
 * @brief      Transfer the gauge field if the gauge field was updated in
 *             openQxD. (Re-)calculate or transfer the clover field if
 *             parameters have changed or gauge field was updated. Update the
 *             settings kappa, su3csw and u1csw in the QudaInvertParam struct
 *             such that they are in sync with openQxD. Set up or update the
 *             multigrid instance if set in QudaInvertParam and if gauge- or
 *             clover-fields or parameters have changed.
 *
 * @param      param_  The parameter struct, where in param->additional_prop a
 *                     pointer to the QudaMultigridParam struct was placed.
 */
void openQCD_qudaSolverUpdate(void *param_)
{
  if (param_ == nullptr) { errorQuda("Solver handle is NULL."); }

  QudaInvertParam *param = static_cast<QudaInvertParam *>(param_);
  openQCD_QudaSolver *additional_prop = static_cast<openQCD_QudaSolver *>(param->additional_prop);

  bool do_param_update = !openQCD_qudaInvertParamCheck(param_);
  bool do_gauge_transfer = !gauge_field_get_up2date() && !gauge_field_get_unset();
  bool do_clover_update = !clover_field_get_up2date() && !gauge_field_get_unset();
  bool do_multigrid_update = param_ != qudaState.dirac_handle && param->inv_type_precondition == QUDA_MG_INVERTER
    && !mg_get_up2date(param) && !gauge_field_get_unset();
  bool do_multigrid_fat_update = do_multigrid_update
    && (do_gauge_transfer || additional_prop->mg_ud_rev != qudaState.ud_rev
        || additional_prop->mg_ad_rev != qudaState.ad_rev);

  if (do_gauge_transfer) {
    if (qudaState.layout.h_gauge == nullptr) { errorQuda("qudaState.layout.h_gauge is not set."); }
    logQuda(QUDA_VERBOSE, "Loading gauge field from openQCD ...\n");
    void *h_gauge = qudaState.layout.h_gauge();
    PUSH_RANGE("openQCD_qudaGaugeLoad", 3);
    QudaReconstructType rec
      = qudaState.layout.flds_parms().gauge == OPENQCD_GAUGE_SU3 ? QUDA_RECONSTRUCT_8 : QUDA_RECONSTRUCT_9;

    /**
     * We set t_boundary = QUDA_ANTI_PERIODIC_T. This setting is a label that
     * tells QUDA the current state of the residing gauge field, that is the
     * same state as the one we transfer from openqxd. In openqxd the hdfld
     * exhibits phases of -1 for the temporal time boundaries, meaning that the
     * gauge fields are explicitly multiplied by -1 on the t=0 time slice, see
     * chs_hd0() in hflds.c. The QUDA_ANTI_PERIODIC_T flag says that these
     * phases are incorporated into the field and that QUDA has to add these
     * phases on the t=0 time slice when reconstructing the field from
     * QUDA_RECONSTRUCT_8/12, but not from QUDA_RECONSTRUCT_NO. In case of
     * QUDA_RECONSTRUCT_NO the value if t_boundary has no effect.
     *
     * @see        https://github.com/lattice/quda/issues/1315
     * @see        Reconstruct#Unpack() in gauge_field_order.h
     * @see        Reconstruct<8,...>#Unpack() in gauge_field_order.h
     * @see        Reconstruct<12,...>#Unpack() in gauge_field_order.h
     */
    openQCD_qudaGaugeLoad(h_gauge, QUDA_DOUBLE_PRECISION, rec, QUDA_ANTI_PERIODIC_T);
    gauge_field_set_revision();
    POP_RANGE;
  }

  if (do_param_update) {
    logQuda(QUDA_VERBOSE, "Syncing kappa, su3csw, u1csw values from openQCD ...\n");
    param->kappa = 1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0));
    additional_prop->u1csw = qudaState.layout.dirac_parms().u1csw;
    set_su3csw(param, qudaState.layout.dirac_parms().su3csw);

    QudaInvertParam *mg_inv_param = additional_prop->mg_param->invert_param;
    mg_inv_param->kappa = 1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0));
    set_su3csw(mg_inv_param, qudaState.layout.dirac_parms().su3csw);
  }

  if (do_clover_update) {
    if (param->clover_csw == 0.0) {
      logQuda(QUDA_VERBOSE, "Deallocating Clover field in QUDA ...\n");
      freeCloverQuda();
      qudaState.swd_ud_rev = 0;
      qudaState.swd_ad_rev = 0;
      qudaState.swd_kappa = 0.0;
      qudaState.swd_su3csw = 0.0;
      qudaState.swd_u1csw = 0.0;
    } else {
      if (qudaState.layout.flds_parms().gauge == OPENQCD_GAUGE_SU3) {
        /**
         * SU3 case:
         * Leaving both h_clover = h_clovinv = NULL allocates the clover field on
         * the GPU and finally calls @createCloverQuda to calculate the clover
         * field.
         */
        logQuda(QUDA_VERBOSE, "Generating Clover field in QUDA ...\n");
        PUSH_RANGE("loadCloverQuda", 3);
        loadCloverQuda(NULL, NULL, param);
        POP_RANGE;
        clover_field_set_revision();
      } else {
        /**
         * U3 case: Transfer the SW-field from openQCD.
         */

        if (qudaState.layout.h_sw == nullptr) { errorQuda("qudaState.layout.h_sw is not set."); }

        logQuda(QUDA_VERBOSE, "Loading Clover field from openQCD ...\n");
        void *h_sw = qudaState.layout.h_sw();
        PUSH_RANGE("openQCD_qudaCloverLoad", 3);
        openQCD_qudaCloverLoad(h_sw, param->kappa, param->clover_csw);
        POP_RANGE;
        clover_field_set_revision();

        /*loadCloverQuda(qudaState.layout.h_sw(), NULL, param);*/
        /* TODO: The above line would be prefered over openQCD_qudaCloverLoad, but throws this error, no idea why?
        QUDA: ERROR: qudaEventRecord_ returned CUDA_ERROR_ILLEGAL_ADDRESS
         (timer.h:82 in start())
         (rank 0, host yoshi, quda_api.cpp:72 in void quda::target::cuda::set_driver_error(CUresult, const char*, const
        char*, const char*, const char*, bool)()) QUDA:        last kernel called was
        (name=N4quda10CopyCloverINS_6clover11FloatNOrderIdLi72ELi2ELb0ELb1ELb0EEENS1_12OpenQCDOrderIdLi72EEEddEE,volume=32x16x16x64,aux=GPU-offline,vol=524288precision=8Nc=3,compute_diagonal)*/
      }
    }
  }

  /* setup/update the multigrid instance or do nothing */
  if (do_multigrid_update) {
    QudaMultigridParam *mg_param = additional_prop->mg_param;

    if (mg_param == nullptr) { errorQuda("No multigrid parameter struct set."); }

    if (do_multigrid_fat_update && param->preconditioner != nullptr) {
      logQuda(QUDA_VERBOSE, "Destroying existing multigrid instance ...\n");
      PUSH_RANGE("destroyMultigridQuda", 4);
      destroyMultigridQuda(param->preconditioner);
      param->preconditioner = nullptr;
      POP_RANGE;

      additional_prop->mg_ud_rev = 0;
      additional_prop->mg_ad_rev = 0;
      additional_prop->mg_kappa = 0.0;
      additional_prop->mg_su3csw = 0.0;
      additional_prop->mg_u1csw = 0.0;
    }

    if (param->preconditioner == nullptr) {
      logQuda(QUDA_VERBOSE, "Setting up multigrid instance ...\n");
      PUSH_RANGE("newMultigridQuda", 4);
      param->preconditioner = newMultigridQuda(mg_param);
      POP_RANGE;
      mg_set_revision(param);
    } else {
      logQuda(QUDA_VERBOSE, "Updating existing multigrid instance ...\n");
      PUSH_RANGE("updateMultigridQuda", 4);
      updateMultigridQuda(param->preconditioner, mg_param);
      POP_RANGE;
      mg_set_revision(param);
    }
  }
}

void *openQCD_qudaSolverReadIn(int id)
{
  int my_rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Allocate on the heap */
  QudaInvertParam *param = new QudaInvertParam(newQudaInvertParam());
  QudaInvertParam *invert_param_mg = new QudaInvertParam(newQudaInvertParam());
  QudaMultigridParam *multigrid_param = new QudaMultigridParam(newQudaMultigridParam());
  std::string section = "Solver " + std::to_string(id);

  /* Some default settings */
  /* Some of them should not be changed */
  param->verbosity = QUDA_SUMMARIZE;
  param->cpu_prec = QUDA_DOUBLE_PRECISION;
  param->cuda_prec = QUDA_DOUBLE_PRECISION;
  param->cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;
  param->cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  param->cuda_prec_precondition = QUDA_HALF_PRECISION;
  param->dirac_order = QUDA_OPENQCD_DIRAC_ORDER;
  param->gamma_basis = QUDA_OPENQCD_GAMMA_BASIS;
  param->dslash_type = QUDA_WILSON_DSLASH;
  param->kappa = 1.0 / (2.0 * (qudaState.layout.dirac_parms().m0 + 4.0));
  param->mu = 0.0;
  param->dagger = QUDA_DAG_NO;
  param->solution_type = QUDA_MAT_SOLUTION;
  param->solve_type = QUDA_DIRECT_SOLVE;
  param->matpc_type = QUDA_MATPC_EVEN_EVEN;
  param->solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  param->inv_type_precondition = QUDA_INVALID_INVERTER; /* disables any preconditioning   */
  param->mass_normalization = QUDA_MASS_NORMALIZATION;

  set_su3csw(param, qudaState.layout.dirac_parms().su3csw);

  if (my_rank == 0 && id != -1) {

    KeyValueStore kv;
    kv.set_map(&enum_map);
    kv.load(qudaState.infile);

    if (!kv.section_exists(section)) {
      errorQuda("Solver section \"%s\" in file %s does not exist.", section.c_str(), qudaState.infile);
    }

    param->verbosity = kv.get<QudaVerbosity>(section, "verbosity", param->verbosity);

    if (param->verbosity >= QUDA_DEBUG_VERBOSE) { kv.dump(); }

    if (kv.get<std::string>(section, "solver") != "QUDA") {
      errorQuda("Solver section \"%s\" in file %s is not a valid quda-solver section (solver = %s).", section.c_str(),
                qudaState.infile, kv.get<std::string>(section, "solver").c_str());
    }

    /* both fields reside on the CPU */
    param->input_location = kv.get<QudaFieldLocation>(section, "input_location", QUDA_CPU_FIELD_LOCATION);
    param->output_location = kv.get<QudaFieldLocation>(section, "output_location", QUDA_CPU_FIELD_LOCATION);

    param->inv_type = kv.get<QudaInverterType>(section, "inv_type", param->inv_type);
    param->kappa = kv.get<double>(section, "kappa", param->kappa);
    param->mu = kv.get<double>(section, "mu", param->mu);
    param->tm_rho = kv.get<double>(section, "tm_rho", param->tm_rho);
    param->epsilon = kv.get<double>(section, "epsilon", param->epsilon);
    param->twist_flavor = kv.get<QudaTwistFlavorType>(section, "twist_flavor", param->twist_flavor);
    param->laplace3D = kv.get<int>(section, "laplace3D", param->laplace3D);

    /* Solver settings */
    param->tol = kv.get<double>(section, "tol", param->tol);
    param->tol_restart = kv.get<double>(section, "tol_restart", param->tol_restart);
    param->tol_hq = kv.get<double>(section, "tol_hq", param->tol_hq);

    param->compute_true_res = kv.get<int>(section, "compute_true_res", param->compute_true_res);
    param->true_res = kv.get<double>(section, "true_res", param->true_res);
    param->true_res_hq = kv.get<double>(section, "true_res_hq", param->true_res_hq);
    param->maxiter = kv.get<int>(section, "maxiter", param->maxiter);
    param->reliable_delta = kv.get<double>(section, "reliable_delta", param->reliable_delta);
    param->reliable_delta_refinement
      = kv.get<double>(section, "reliable_delta_refinement", param->reliable_delta_refinement);
    param->use_alternative_reliable = kv.get<int>(section, "use_alternative_reliable", param->use_alternative_reliable);
    param->use_sloppy_partial_accumulator
      = kv.get<int>(section, "use_sloppy_partial_accumulator", param->use_sloppy_partial_accumulator);

    param->solution_accumulator_pipeline
      = kv.get<int>(section, "solution_accumulator_pipeline", param->solution_accumulator_pipeline);

    param->max_res_increase = kv.get<int>(section, "max_res_increase", param->max_res_increase);
    param->max_res_increase_total = kv.get<int>(section, "max_res_increase_total", param->max_res_increase_total);
    param->max_hq_res_increase = kv.get<int>(section, "max_hq_res_increase", param->max_hq_res_increase);
    param->max_hq_res_restart_total = kv.get<int>(section, "max_hq_res_restart_total", param->max_hq_res_restart_total);

    param->heavy_quark_check = kv.get<int>(section, "heavy_quark_check", param->heavy_quark_check);

    param->pipeline = kv.get<int>(section, "pipeline", param->pipeline);
    param->num_offset = kv.get<int>(section, "num_offset", param->num_offset);
    param->num_src = kv.get<int>(section, "num_src", param->num_src);
    param->num_src_per_sub_partition
      = kv.get<int>(section, "num_src_per_sub_partition", param->num_src_per_sub_partition);

    param->split_grid[0] = kv.get<int>(section, "split_grid[1]", param->split_grid[0]);
    param->split_grid[1] = kv.get<int>(section, "split_grid[2]", param->split_grid[1]);
    param->split_grid[2] = kv.get<int>(section, "split_grid[3]", param->split_grid[2]);
    param->split_grid[3] = kv.get<int>(section, "split_grid[0]", param->split_grid[3]);

    param->overlap = kv.get<int>(section, "overlap", param->overlap);

    for (int i = 0; i < param->num_offset; i++) {
      std::string sub_key = "offset[" + std::to_string(i) + "]";
      param->offset[i] = kv.get<double>(section, sub_key, param->offset[i]);
      sub_key = "tol_offset[" + std::to_string(i) + "]";
      param->tol_offset[i] = kv.get<double>(section, sub_key, param->tol_offset[i]);
      sub_key = "tol_hq_offset[" + std::to_string(i) + "]";
      param->tol_hq_offset[i] = kv.get<double>(section, sub_key, param->tol_hq_offset[i]);
    }

    param->compute_action = kv.get<int>(section, "compute_action", param->compute_action);

    param->solution_type = kv.get<QudaSolutionType>(section, "solution_type", param->solution_type);
    param->solve_type = kv.get<QudaSolveType>(section, "solve_type", param->solve_type);
    param->matpc_type = kv.get<QudaMatPCType>(section, "matpc_type", param->matpc_type);
    param->dagger = kv.get<QudaDagType>(section, "dagger", param->dagger);
    param->mass_normalization = kv.get<QudaMassNormalization>(section, "mass_normalization", param->mass_normalization);
    param->solver_normalization
      = kv.get<QudaSolverNormalization>(section, "solver_normalization", param->solver_normalization);

    param->preserve_source = kv.get<QudaPreserveSource>(section, "preserve_source", param->preserve_source);

    param->cpu_prec = kv.get<QudaPrecision>(section, "cpu_prec", param->cpu_prec);
    param->cuda_prec = kv.get<QudaPrecision>(section, "cuda_prec", param->cuda_prec);
    param->cuda_prec_sloppy = kv.get<QudaPrecision>(section, "cuda_prec_sloppy", param->cuda_prec_sloppy);
    param->cuda_prec_refinement_sloppy
      = kv.get<QudaPrecision>(section, "cuda_prec_refinement_sloppy", param->cuda_prec_refinement_sloppy);
    param->cuda_prec_precondition
      = kv.get<QudaPrecision>(section, "cuda_prec_precondition", param->cuda_prec_precondition);
    param->cuda_prec_eigensolver = kv.get<QudaPrecision>(section, "cuda_prec_eigensolver", param->cuda_prec_eigensolver);

    param->clover_location = kv.get<QudaFieldLocation>(section, "clover_location", param->clover_location);
    param->clover_cpu_prec = kv.get<QudaPrecision>(section, "clover_cpu_prec", param->clover_cpu_prec);
    param->clover_cuda_prec = kv.get<QudaPrecision>(section, "clover_cuda_prec", param->clover_cuda_prec);
    param->clover_cuda_prec_sloppy
      = kv.get<QudaPrecision>(section, "clover_cuda_prec_sloppy", param->clover_cuda_prec_sloppy);
    param->clover_cuda_prec_refinement_sloppy
      = kv.get<QudaPrecision>(section, "clover_cuda_prec_refinement_sloppy", param->clover_cuda_prec_refinement_sloppy);
    param->clover_cuda_prec_precondition
      = kv.get<QudaPrecision>(section, "clover_cuda_prec_precondition", param->clover_cuda_prec_precondition);
    param->clover_cuda_prec_eigensolver
      = kv.get<QudaPrecision>(section, "clover_cuda_prec_eigensolver", param->clover_cuda_prec_eigensolver);

    param->use_init_guess = kv.get<QudaUseInitGuess>(section, "use_init_guess", param->use_init_guess);

    param->clover_csw = kv.get<double>(section, "clover_csw", param->clover_csw);
    param->clover_coeff = kv.get<double>(section, "clover_coeff", param->clover_coeff);
    param->clover_rho = kv.get<double>(section, "clover_rho", param->clover_rho);
    param->compute_clover_trlog = kv.get<int>(section, "compute_clover_trlog", param->compute_clover_trlog);
    param->tune = kv.get<QudaTune>(section, "tune", param->tune);
    param->Nsteps = kv.get<int>(section, "Nsteps", param->Nsteps);
    param->gcrNkrylov = kv.get<int>(section, "gcrNkrylov", param->gcrNkrylov);

    param->inv_type_precondition
      = kv.get<QudaInverterType>(section, "inv_type_precondition", param->inv_type_precondition);
    param->deflate = kv.get<QudaBoolean>(section, "deflate", param->deflate);
    param->verbosity_precondition
      = kv.get<QudaVerbosity>(section, "verbosity_precondition", param->verbosity_precondition);
    param->tol_precondition = kv.get<double>(section, "tol_precondition", param->tol_precondition);
    param->maxiter_precondition = kv.get<int>(section, "maxiter_precondition", param->maxiter_precondition);
    param->omega = kv.get<double>(section, "omega", param->omega);
    param->ca_basis = kv.get<QudaCABasis>(section, "ca_basis", param->ca_basis);
    param->ca_lambda_min = kv.get<double>(section, "ca_lambda_min", param->ca_lambda_min);
    param->ca_lambda_max = kv.get<double>(section, "ca_lambda_max", param->ca_lambda_max);
    param->ca_basis_precondition = kv.get<QudaCABasis>(section, "ca_basis_precondition", param->ca_basis_precondition);
    param->ca_lambda_min_precondition
      = kv.get<double>(section, "ca_lambda_min_precondition", param->ca_lambda_min_precondition);
    param->ca_lambda_max_precondition
      = kv.get<double>(section, "ca_lambda_max_precondition", param->ca_lambda_max_precondition);
    param->precondition_cycle = kv.get<int>(section, "precondition_cycle", param->precondition_cycle);
    param->schwarz_type = kv.get<QudaSchwarzType>(section, "schwarz_type", param->schwarz_type);
    param->accelerator_type_precondition
      = kv.get<QudaAcceleratorType>(section, "accelerator_type_precondition", param->accelerator_type_precondition);

    param->madwf_diagonal_suppressor
      = kv.get<double>(section, "madwf_diagonal_suppressor", param->madwf_diagonal_suppressor);
    param->madwf_ls = kv.get<int>(section, "madwf_ls", param->madwf_ls);
    param->madwf_null_miniter = kv.get<int>(section, "madwf_null_miniter", param->madwf_null_miniter);
    param->madwf_null_tol = kv.get<double>(section, "madwf_null_tol", param->madwf_null_tol);
    param->madwf_train_maxiter = kv.get<int>(section, "madwf_train_maxiter", param->madwf_train_maxiter);

    param->madwf_param_load = kv.get<QudaBoolean>(section, "madwf_param_load", param->madwf_param_load);
    param->madwf_param_save = kv.get<QudaBoolean>(section, "madwf_param_save", param->madwf_param_save);
    strcpy(param->madwf_param_infile,
           kv.get<std::string>(section, "madwf_param_infile", param->madwf_param_infile).c_str());
    strcpy(param->madwf_param_outfile,
           kv.get<std::string>(section, "madwf_param_outfile", param->madwf_param_outfile).c_str());

    param->residual_type = kv.get<QudaResidualType>(section, "residual_type", param->residual_type);

    if (param->inv_type_precondition == QUDA_MG_INVERTER) {

      std::string mg_section = section + " Multigrid";

      if (!kv.section_exists(mg_section)) {
        errorQuda("Solver section \"%s\" in file %s does not exist.", mg_section.c_str(), qudaState.infile);
      }

      /* (shallow) copy the struct */
      *invert_param_mg = *param;

      /* these have to be fixed, and cannot be overwritten by the input file */
      invert_param_mg->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
      invert_param_mg->dirac_order = QUDA_DIRAC_ORDER;

      multigrid_param->n_level = kv.get<int>(mg_section, "n_level", multigrid_param->n_level, true);
      multigrid_param->setup_type = kv.get<QudaSetupType>(mg_section, "setup_type", multigrid_param->setup_type);
      multigrid_param->pre_orthonormalize
        = kv.get<QudaBoolean>(mg_section, "pre_orthonormalize", multigrid_param->pre_orthonormalize);
      multigrid_param->post_orthonormalize
        = kv.get<QudaBoolean>(mg_section, "post_orthonormalize", multigrid_param->post_orthonormalize);
      multigrid_param->setup_minimize_memory
        = kv.get<QudaBoolean>(mg_section, "setup_minimize_memory", multigrid_param->setup_minimize_memory);
      multigrid_param->compute_null_vector
        = kv.get<QudaComputeNullVector>(mg_section, "compute_null_vector", multigrid_param->compute_null_vector);
      multigrid_param->generate_all_levels
        = kv.get<QudaBoolean>(mg_section, "generate_all_levels", multigrid_param->generate_all_levels);
      multigrid_param->run_verify = kv.get<QudaBoolean>(mg_section, "run_verify", multigrid_param->run_verify);
      multigrid_param->run_low_mode_check
        = kv.get<QudaBoolean>(mg_section, "run_low_mode_check", multigrid_param->run_low_mode_check);
      multigrid_param->run_oblique_proj_check
        = kv.get<QudaBoolean>(mg_section, "run_oblique_proj_check", multigrid_param->run_oblique_proj_check);
      multigrid_param->coarse_guess = kv.get<QudaBoolean>(mg_section, "coarse_guess", multigrid_param->coarse_guess);
      multigrid_param->preserve_deflation
        = kv.get<QudaBoolean>(mg_section, "preserve_deflation", multigrid_param->preserve_deflation);
      multigrid_param->allow_truncation
        = kv.get<QudaBoolean>(mg_section, "allow_truncation", multigrid_param->allow_truncation);
      multigrid_param->staggered_kd_dagger_approximation = kv.get<QudaBoolean>(
        mg_section, "staggered_kd_dagger_approximation", multigrid_param->staggered_kd_dagger_approximation);
      multigrid_param->thin_update_only
        = kv.get<QudaBoolean>(mg_section, "thin_update_only", multigrid_param->thin_update_only);

      for (int i = 0; i < multigrid_param->n_level; i++) {
        std::string subsection = section + " Multigrid Level " + std::to_string(i);

        if (!kv.section_exists(subsection)) {
          errorQuda("Solver section \"%s\" in file %s does not exist.", subsection.c_str(), qudaState.infile);
        }

        multigrid_param->geo_block_size[i][0]
          = kv.get<int>(subsection, "geo_block_size[1]", multigrid_param->geo_block_size[i][0]);
        multigrid_param->geo_block_size[i][1]
          = kv.get<int>(subsection, "geo_block_size[2]", multigrid_param->geo_block_size[i][1]);
        multigrid_param->geo_block_size[i][2]
          = kv.get<int>(subsection, "geo_block_size[3]", multigrid_param->geo_block_size[i][2]);
        multigrid_param->geo_block_size[i][3]
          = kv.get<int>(subsection, "geo_block_size[0]", multigrid_param->geo_block_size[i][3]);

        if (i == 0) {
          multigrid_param->geo_block_size[i][0] = 4;
          multigrid_param->geo_block_size[i][1] = 4;
          multigrid_param->geo_block_size[i][2] = 4;
          multigrid_param->geo_block_size[i][3] = 4;
        }

        multigrid_param->spin_block_size[i]
          = kv.get<int>(subsection, "spin_block_size", multigrid_param->spin_block_size[i]);
        multigrid_param->n_vec[i] = kv.get<int>(subsection, "n_vec", multigrid_param->n_vec[i]);
        multigrid_param->precision_null[i]
          = kv.get<QudaPrecision>(subsection, "precision_null", multigrid_param->precision_null[i]);
        multigrid_param->n_block_ortho[i] = kv.get<int>(subsection, "n_block_ortho", multigrid_param->n_block_ortho[i]);
        multigrid_param->block_ortho_two_pass[i]
          = kv.get<QudaBoolean>(subsection, "block_ortho_two_pass", multigrid_param->block_ortho_two_pass[i]);
        multigrid_param->verbosity[i] = kv.get<QudaVerbosity>(subsection, "verbosity", multigrid_param->verbosity[i]);
        multigrid_param->setup_inv_type[i]
          = kv.get<QudaInverterType>(subsection, "setup_inv_type", multigrid_param->setup_inv_type[i]);
        multigrid_param->setup_use_mma[i]
          = kv.get<QudaBoolean>(subsection, "setup_use_mma", multigrid_param->setup_use_mma[i]);
        multigrid_param->dslash_use_mma[i]
          = kv.get<QudaBoolean>(subsection, "dslash_use_mma", multigrid_param->dslash_use_mma[i]);
        multigrid_param->num_setup_iter[i]
          = kv.get<int>(subsection, "num_setup_iter", multigrid_param->num_setup_iter[i]);
        multigrid_param->setup_tol[i] = kv.get<double>(subsection, "setup_tol", multigrid_param->setup_tol[i]);
        multigrid_param->setup_maxiter[i] = kv.get<int>(subsection, "setup_maxiter", multigrid_param->setup_maxiter[i]);
        multigrid_param->setup_maxiter_refresh[i]
          = kv.get<int>(subsection, "setup_maxiter_refresh", multigrid_param->setup_maxiter_refresh[i]);
        multigrid_param->setup_ca_basis[i]
          = kv.get<QudaCABasis>(subsection, "setup_ca_basis", multigrid_param->setup_ca_basis[i]);
        multigrid_param->setup_ca_basis_size[i]
          = kv.get<int>(subsection, "setup_ca_basis_size", multigrid_param->setup_ca_basis_size[i]);
        multigrid_param->setup_ca_lambda_min[i]
          = kv.get<double>(subsection, "setup_ca_lambda_min", multigrid_param->setup_ca_lambda_min[i]);
        multigrid_param->setup_ca_lambda_max[i]
          = kv.get<double>(subsection, "setup_ca_lambda_max", multigrid_param->setup_ca_lambda_max[i]);

        multigrid_param->coarse_solver[i]
          = kv.get<QudaInverterType>(subsection, "coarse_solver", multigrid_param->coarse_solver[i]);
        multigrid_param->coarse_solver_tol[i]
          = kv.get<double>(subsection, "coarse_solver_tol", multigrid_param->coarse_solver_tol[i]);
        multigrid_param->coarse_solver_maxiter[i]
          = kv.get<int>(subsection, "coarse_solver_maxiter", multigrid_param->coarse_solver_maxiter[i]);
        multigrid_param->coarse_solver_ca_basis[i]
          = kv.get<QudaCABasis>(subsection, "coarse_solver_ca_basis", multigrid_param->coarse_solver_ca_basis[i]);
        multigrid_param->coarse_solver_ca_basis_size[i]
          = kv.get<int>(subsection, "coarse_solver_ca_basis_size", multigrid_param->coarse_solver_ca_basis_size[i]);
        multigrid_param->coarse_solver_ca_lambda_min[i]
          = kv.get<double>(subsection, "coarse_solver_ca_lambda_min", multigrid_param->coarse_solver_ca_lambda_min[i]);
        multigrid_param->coarse_solver_ca_lambda_max[i]
          = kv.get<double>(subsection, "coarse_solver_ca_lambda_max", multigrid_param->coarse_solver_ca_lambda_max[i]);
        multigrid_param->smoother[i] = kv.get<QudaInverterType>(subsection, "smoother", multigrid_param->smoother[i]);
        multigrid_param->smoother_tol[i] = kv.get<double>(subsection, "smoother_tol", multigrid_param->smoother_tol[i]);
        multigrid_param->nu_pre[i] = kv.get<int>(subsection, "nu_pre", multigrid_param->nu_pre[i]);
        multigrid_param->nu_post[i] = kv.get<int>(subsection, "nu_post", multigrid_param->nu_post[i]);
        multigrid_param->smoother_solver_ca_basis[i]
          = kv.get<QudaCABasis>(subsection, "smoother_solver_ca_basis", multigrid_param->smoother_solver_ca_basis[i]);
        multigrid_param->smoother_solver_ca_lambda_min[i] = kv.get<double>(
          subsection, "smoother_solver_ca_lambda_min", multigrid_param->smoother_solver_ca_lambda_min[i]);
        multigrid_param->smoother_solver_ca_lambda_max[i] = kv.get<double>(
          subsection, "smoother_solver_ca_lambda_max", multigrid_param->smoother_solver_ca_lambda_max[i]);
        multigrid_param->omega[i] = kv.get<double>(subsection, "omega", multigrid_param->omega[i]);
        multigrid_param->smoother_halo_precision[i]
          = kv.get<QudaPrecision>(subsection, "smoother_halo_precision", multigrid_param->smoother_halo_precision[i]);
        multigrid_param->smoother_schwarz_type[i]
          = kv.get<QudaSchwarzType>(subsection, "smoother_schwarz_type", multigrid_param->smoother_schwarz_type[i]);
        multigrid_param->smoother_schwarz_cycle[i]
          = kv.get<int>(subsection, "smoother_schwarz_cycle", multigrid_param->smoother_schwarz_cycle[i]);
        multigrid_param->coarse_grid_solution_type[i] = kv.get<QudaSolutionType>(
          subsection, "coarse_grid_solution_type", multigrid_param->coarse_grid_solution_type[i]);
        multigrid_param->smoother_solve_type[i]
          = kv.get<QudaSolveType>(subsection, "smoother_solve_type", multigrid_param->smoother_solve_type[i]);
        multigrid_param->cycle_type[i]
          = kv.get<QudaMultigridCycleType>(subsection, "cycle_type", multigrid_param->cycle_type[i]);
        multigrid_param->global_reduction[i]
          = kv.get<QudaBoolean>(subsection, "global_reduction", multigrid_param->global_reduction[i]);
        multigrid_param->location[i] = kv.get<QudaFieldLocation>(subsection, "location", multigrid_param->location[i]);
        multigrid_param->setup_location[i]
          = kv.get<QudaFieldLocation>(subsection, "setup_location", multigrid_param->setup_location[i]);
        multigrid_param->use_eig_solver[i]
          = kv.get<QudaBoolean>(subsection, "use_eig_solver", multigrid_param->use_eig_solver[i]);

        multigrid_param->vec_load[i] = kv.get<QudaBoolean>(subsection, "vec_load", multigrid_param->vec_load[i]);
        multigrid_param->vec_store[i] = kv.get<QudaBoolean>(subsection, "vec_store", multigrid_param->vec_store[i]);
        /*strcpy(multigrid_param->vec_infile[i], kv.get<std::string>(subsection, "vec_infile",
        multigrid_param->vec_infile[i]).c_str()); strcpy(multigrid_param->vec_outfile[i],
        kv.get<std::string>(subsection, "vec_outfile", multigrid_param->vec_outfile[i]).c_str());*/

        multigrid_param->mu_factor[i] = kv.get<double>(subsection, "mu_factor", multigrid_param->mu_factor[i]);
        multigrid_param->transfer_type[i]
          = kv.get<QudaTransferType>(subsection, "transfer_type", multigrid_param->transfer_type[i]);
      }
    }
  }

  /* transfer of the struct to all processes */
  MPI_Bcast((void *)param, sizeof(*param), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)invert_param_mg, sizeof(*invert_param_mg), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)multigrid_param, sizeof(*multigrid_param), MPI_BYTE, 0, MPI_COMM_WORLD);
  multigrid_param->invert_param = invert_param_mg;

  /**
   * We need a void* to store the multigrid_param (QudaMultigridParam) struct,
   * such that we can access it and setup multigrid in a later stage, for
   * instance right before calling invertQuda() if multigrid was not
   * instantiated until then.
   */
  openQCD_QudaSolver *additional_prop = new openQCD_QudaSolver();
  sprintf(additional_prop->infile, "%s", qudaState.infile);
  additional_prop->id = id;
  additional_prop->mg_param = multigrid_param;
  additional_prop->u1csw = qudaState.layout.dirac_parms().u1csw;
  param->additional_prop = reinterpret_cast<void *>(additional_prop);

  return (void *)param;
}

void *openQCD_qudaSolverGetHandle(int id)
{
  check_solver_id(id);
  if (qudaState.inv_handles[id] == nullptr) {
    if (id != -1) {
      logQuda(QUDA_VERBOSE, "Read in solver parameters from file %s for solver (id=%d)\n", qudaState.infile, id);
    }
    qudaState.inv_handles[id] = openQCD_qudaSolverReadIn(id);
  }

  openQCD_qudaSolverUpdate(qudaState.inv_handles[id]);
  return qudaState.inv_handles[id];
}

void openQCD_qudaDw_deprecated(void *src, void *dst, openQCD_QudaDiracParam_t p)
{
  QudaInvertParam param = newOpenQCDDiracParam(p);

  /* both fields reside on the CPU */
  param.input_location = QUDA_CPU_FIELD_LOCATION;
  param.output_location = QUDA_CPU_FIELD_LOCATION;

  MatQuda(static_cast<char *>(dst), static_cast<char *>(src), &param);

  logQuda(QUDA_DEBUG_VERBOSE, "MatQuda()\n");
  logQuda(QUDA_DEBUG_VERBOSE, "  gflops      = %.2e\n", param.gflops);
  logQuda(QUDA_DEBUG_VERBOSE, "  secs        = %.2e\n", param.secs);
}

void openQCD_qudaDw(double mu, void *in, void *out)
{
  if (gauge_field_get_unset()) { errorQuda("Gauge field not populated in openQxD."); }

  QudaInvertParam *param = static_cast<QudaInvertParam *>(openQCD_qudaSolverGetHandle(-1));
  param->mu = mu;

  if (!openQCD_qudaInvertParamCheck(param)) {
    errorQuda("QudaInvertParam struct check failed, parameters/fields between openQxD and QUDA are not in sync.");
  }

  /* both fields reside on the CPU */
  param->input_location = QUDA_CPU_FIELD_LOCATION;
  param->output_location = QUDA_CPU_FIELD_LOCATION;

  MatQuda(static_cast<char *>(out), static_cast<char *>(in), param);
}

/**
 * @brief      Take the string-hash over a struct using std::hash.
 *
 * @param[in]  in    Input struct
 *
 * @tparam     T     Type of input struct
 *
 * @return     Hash value
 */
template <typename T> int hash_struct(T *in)
{
  int hash = 0;
  char *cstruct = reinterpret_cast<char *>(in);

  for (char *c = cstruct; c < cstruct + sizeof(T); c += strlen(c) + 1) {
    if (strlen(c) != 0) { hash ^= (std::hash<std::string> {}(std::string(c)) << 1); }
  }

  return hash;
}

int openQCD_qudaSolverGetHash(int id)
{
  check_solver_id(id);
  if (qudaState.inv_handles[id] != nullptr) {
    QudaInvertParam *param = reinterpret_cast<QudaInvertParam *>(qudaState.inv_handles[id]);
    QudaInvertParam hparam = newQudaInvertParam();
    memset(&hparam, '\0', sizeof(QudaInvertParam)); /* set everything to zero */

    /* Set some properties we want to take the hash over */
    hparam.inv_type = param->inv_type;
    hparam.tol = param->tol;
    hparam.tol_restart = param->tol_restart;
    hparam.tol_hq = param->tol_hq;
    hparam.maxiter = param->maxiter;
    hparam.reliable_delta = param->reliable_delta;
    hparam.solution_type = param->solution_type;
    hparam.solve_type = param->solve_type;
    hparam.matpc_type = param->matpc_type;
    hparam.dagger = param->dagger;
    hparam.mass_normalization = param->mass_normalization;
    hparam.solver_normalization = param->solver_normalization;
    hparam.cpu_prec = param->cpu_prec;
    hparam.cuda_prec = param->cuda_prec;
    hparam.use_init_guess = param->use_init_guess;
    hparam.gcrNkrylov = param->gcrNkrylov;

    return hash_struct<QudaInvertParam>(&hparam);
  } else {
    return 0;
  }
}

void openQCD_qudaSolverPrintSetup(int id)
{
  check_solver_id(id);
  if (qudaState.inv_handles[id] != nullptr) {
    QudaInvertParam *param = static_cast<QudaInvertParam *>(qudaState.inv_handles[id]);
    openQCD_QudaSolver *additional_prop = static_cast<openQCD_QudaSolver *>(param->additional_prop);

    printQudaInvertParam(param);
    printfQuda("additional_prop->infile = %s\n", additional_prop->infile);
    printfQuda("additional_prop->id = %d\n", additional_prop->id);
    printfQuda("additional_prop->mg_param = %p\n", additional_prop->mg_param);
    printfQuda("additional_prop->u1csw = %.6e\n", additional_prop->u1csw);
    printfQuda("additional_prop->mg_ud_rev = %d\n", additional_prop->mg_ud_rev);
    printfQuda("additional_prop->mg_ad_rev = %d\n", additional_prop->mg_ad_rev);
    printfQuda("additional_prop->mg_kappa = %.6e\n", additional_prop->mg_kappa);
    printfQuda("additional_prop->mg_su3csw = %.6e\n", additional_prop->mg_su3csw);
    printfQuda("additional_prop->mg_u1csw = %.6e\n", additional_prop->mg_u1csw);
    printfQuda("handle = %p\n", param);
    printfQuda("hash = %d\n", openQCD_qudaSolverGetHash(id));

    printfQuda("inv_type_precondition = %d\n", param->inv_type_precondition);

    if (param->inv_type_precondition == QUDA_MG_INVERTER) { printQudaMultigridParam(additional_prop->mg_param); }
  } else {
    printfQuda("<Solver is not initialized yet>\n");
  }
}

double openQCD_qudaInvert(int id, double mu, void *source, void *solution, int *status)
{
  if (gauge_field_get_unset()) { errorQuda("Gauge field not populated in openQxD."); }

  /**
   * This is to make sure we behave in the same way as openQCDs solvers, we call
   * h_sw() which in turn calls sw_term(). We have to make sure that the SW-term
   * in openQxD is setup and in sync with QUDAs.
   */
  if (qudaState.layout.h_sw != nullptr) {
    qudaState.layout.h_sw();
  } else {
    errorQuda("qudaState.layout.h_sw is not set.");
  }

  QudaInvertParam *param = static_cast<QudaInvertParam *>(openQCD_qudaSolverGetHandle(id));
  param->mu = mu;

  if (!openQCD_qudaInvertParamCheck(param)) {
    errorQuda("Solver check failed, parameters/fields between openQxD and QUDA are not in sync.");
  }

  logQuda(QUDA_VERBOSE, "Calling invertQuda() ...\n");
  PUSH_RANGE("invertQuda", 5);
  invertQuda(static_cast<char *>(solution), static_cast<char *>(source), param);
  POP_RANGE;

  *status = param->true_res <= param->tol ? param->iter : -1;

  logQuda(QUDA_VERBOSE, "openQCD_qudaInvert()\n");
  logQuda(QUDA_VERBOSE, "  true_res    = %.2e\n", param->true_res);
  logQuda(QUDA_VERBOSE, "  true_res_hq = %.2e\n", param->true_res_hq);
  logQuda(QUDA_VERBOSE, "  iter        = %d\n", param->iter);
  logQuda(QUDA_VERBOSE, "  gflops      = %.2e\n", param->gflops);
  logQuda(QUDA_VERBOSE, "  secs        = %.2e\n", param->secs);
  logQuda(QUDA_VERBOSE, "  status      = %d\n", *status);

  return param->true_res;
}

void openQCD_qudaSolverDestroy(int id)
{
  check_solver_id(id);
  if (qudaState.inv_handles[id] != nullptr) {
    QudaInvertParam *param = static_cast<QudaInvertParam *>(qudaState.inv_handles[id]);

    if (param->inv_type_precondition == QUDA_MG_INVERTER) { destroyMultigridQuda(param->preconditioner); }

    delete static_cast<openQCD_QudaSolver *>(param->additional_prop)->mg_param;
    delete static_cast<openQCD_QudaSolver *>(param->additional_prop);
    delete param;
    qudaState.inv_handles[id] = nullptr;
  }
}

void *openQCD_qudaEigensolverReadIn(int id, int solver_id)
{
  int my_rank;
  QudaEigParam *param;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  QudaVerbosity verbosity = QUDA_SUMMARIZE;

  /* Allocate on the heap */
  if (qudaState.eig_handles[id] == nullptr) {
    param = new QudaEigParam(newQudaEigParam());
  } else {
    param = static_cast<QudaEigParam *>(qudaState.eig_handles[id]);
  }

  if (my_rank == 0) {

    KeyValueStore kv;
    kv.set_map(&enum_map);
    kv.load(qudaState.infile);

    std::string section = "Eigensolver " + std::to_string(id);

    verbosity = kv.get<QudaVerbosity>(section, "verbosity", verbosity);

    if (verbosity >= QUDA_DEBUG_VERBOSE) { kv.dump(); }

    if (kv.get<std::string>(section, "solver") != "QUDA") {
      errorQuda("Eigensolver section \"%s\" in file %s is not a valid quda-eigensolver section (solver = %s)\n",
                section.c_str(), qudaState.infile, kv.get<std::string>(section, "solver").c_str());
    }

    param->eig_type = kv.get<QudaEigType>(section, "eig_type", param->eig_type);
    param->use_poly_acc = kv.get<QudaBoolean>(section, "use_poly_acc", param->use_poly_acc);
    param->poly_deg = kv.get<int>(section, "poly_deg", param->poly_deg);
    param->a_min = kv.get<double>(section, "a_min", param->a_min);
    param->a_max = kv.get<double>(section, "a_max", param->a_max);
    param->preserve_deflation = kv.get<QudaBoolean>(section, "preserve_deflation", param->preserve_deflation);
    /*param->*preserve_deflation_space = kv.get<void>(section, *"*preserve_deflation_space", param->preserve_deflation_space);*/
    param->preserve_evals = kv.get<QudaBoolean>(section, "preserve_evals", param->preserve_evals);
    param->use_dagger = kv.get<QudaBoolean>(section, "use_dagger", param->use_dagger);
    param->use_norm_op = kv.get<QudaBoolean>(section, "use_norm_op", param->use_norm_op);
    param->use_pc = kv.get<QudaBoolean>(section, "use_pc", param->use_pc);
    param->use_eigen_qr = kv.get<QudaBoolean>(section, "use_eigen_qr", param->use_eigen_qr);
    param->compute_svd = kv.get<QudaBoolean>(section, "compute_svd", param->compute_svd);
    param->compute_gamma5 = kv.get<QudaBoolean>(section, "compute_gamma5", param->compute_gamma5);
    param->require_convergence = kv.get<QudaBoolean>(section, "require_convergence", param->require_convergence);
    param->spectrum = kv.get<QudaEigSpectrumType>(section, "spectrum", param->spectrum);
    param->n_ev = kv.get<int>(section, "n_ev", param->n_ev);
    param->n_kr = kv.get<int>(section, "n_kr", param->n_kr);
    param->n_conv = kv.get<int>(section, "n_conv", param->n_conv);
    param->n_ev_deflate = kv.get<int>(section, "n_ev_deflate", param->n_ev_deflate);
    param->tol = kv.get<double>(section, "tol", param->tol);
    param->qr_tol = kv.get<double>(section, "qr_tol", param->qr_tol);
    param->check_interval = kv.get<int>(section, "check_interval", param->check_interval);
    param->max_restarts = kv.get<int>(section, "max_restarts", param->max_restarts);
    param->batched_rotate = kv.get<int>(section, "batched_rotate", param->batched_rotate);
    param->block_size = kv.get<int>(section, "block_size", param->block_size);
    param->arpack_check = kv.get<QudaBoolean>(section, "arpack_check", param->arpack_check);
    strcpy(param->QUDA_logfile, kv.get<std::string>(section, "QUDA_logfile", param->QUDA_logfile).c_str());
    strcpy(param->arpack_logfile, kv.get<std::string>(section, "arpack_logfile", param->arpack_logfile).c_str());

    param->nk = kv.get<int>(section, "nk", param->nk);
    param->np = kv.get<int>(section, "np", param->np);
    param->import_vectors = kv.get<QudaBoolean>(section, "import_vectors", param->import_vectors);
    param->cuda_prec_ritz = kv.get<QudaPrecision>(section, "cuda_prec_ritz", param->cuda_prec_ritz);
    param->mem_type_ritz = kv.get<QudaMemoryType>(section, "mem_type_ritz", param->mem_type_ritz);
    param->location = kv.get<QudaFieldLocation>(section, "location", param->location);
    param->run_verify = kv.get<QudaBoolean>(section, "run_verify", param->run_verify);
    /*strcpy(param->vec_infile, kv.get<std::string>(section, "vec_infile", param->vec_infile).c_str());*/
    /*strcpy(param->vec_outfile, kv.get<std::string>(section, "vec_outfile", param->vec_outfile).c_str());*/
    param->vec_outfile[0] = '\0';
    param->vec_infile[0] = '\0';
    param->save_prec = kv.get<QudaPrecision>(section, "save_prec", param->save_prec);
    param->io_parity_inflate = kv.get<QudaBoolean>(section, "io_parity_inflate", param->io_parity_inflate);
    param->extlib_type = kv.get<QudaExtLibType>(section, "extlib_type", param->extlib_type);
  }

  /* transfer of the struct to all the processes */
  MPI_Bcast((void *)param, sizeof(*param), MPI_BYTE, 0, MPI_COMM_WORLD);

  void *inv_param = openQCD_qudaSolverGetHandle(solver_id);
  param->invert_param = static_cast<QudaInvertParam *>(inv_param);

  param->invert_param->verbosity = std::max(param->invert_param->verbosity, verbosity);

  if (solver_id != -1 && param->invert_param->verbosity >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(param->invert_param);
  }

  if (param->invert_param->verbosity >= QUDA_DEBUG_VERBOSE) { printQudaEigParam(param); }

  return (void *)param;
}

void *openQCD_qudaEigensolverGetHandle(int id, int solver_id)
{
  check_eigensolver_id(id);
  check_solver_id(solver_id);

  if (qudaState.eig_handles[id] == nullptr) {
    logQuda(QUDA_VERBOSE, "Read in eigensolver parameters from file %s for eigensolver (id=%d)\n", qudaState.infile, id);
    qudaState.eig_handles[id] = openQCD_qudaEigensolverReadIn(id, solver_id);
  }

  openQCD_qudaSolverUpdate(static_cast<QudaEigParam *>(qudaState.eig_handles[id])->invert_param);
  return qudaState.eig_handles[id];
}

void openQCD_qudaEigensolverPrintSetup(int id, int solver_id)
{
  check_eigensolver_id(id);
  check_solver_id(solver_id);

  if (qudaState.eig_handles[id] != nullptr) {
    QudaEigParam *param = static_cast<QudaEigParam *>(qudaState.eig_handles[id]);
    printQudaEigParam(param);
    printfQuda("\n");
    openQCD_qudaSolverPrintSetup(solver_id);
  } else {
    printfQuda("<Eigensolver is not initialized yet>\n");
  }
}

void openQCD_qudaEigensolve(int id, int solver_id, void **h_evecs, void *h_evals)
{
  if (gauge_field_get_unset()) { errorQuda("Gauge field not populated in openQxD."); }

  if (qudaState.layout.h_sw != nullptr) {
    qudaState.layout.h_sw();
  } else {
    errorQuda("qudaState.layout.h_sw is not set.");
  }

  QudaEigParam *eig_param = static_cast<QudaEigParam *>(openQCD_qudaEigensolverGetHandle(id, solver_id));

  if (!openQCD_qudaInvertParamCheck(eig_param->invert_param)) {
    errorQuda("Solver check failed, parameters/fields between openQxD and QUDA are not in sync.");
  }

  logQuda(QUDA_VERBOSE, "Calling eigensolveQuda() ...\n");
  PUSH_RANGE("eigensolveQuda", 6);
  eigensolveQuda(h_evecs, static_cast<openqcd_complex_dble *>(h_evals), eig_param);
  POP_RANGE;

  logQuda(QUDA_SUMMARIZE, "openQCD_qudaEigensolve()\n");
  logQuda(QUDA_SUMMARIZE, "  gflops      = %.2e\n", eig_param->invert_param->gflops);
  logQuda(QUDA_SUMMARIZE, "  secs        = %.2e\n", eig_param->invert_param->secs);
  logQuda(QUDA_SUMMARIZE, "  iter        = %d\n", eig_param->invert_param->iter);
}

void openQCD_qudaEigensolverDestroy(int id)
{
  check_eigensolver_id(id);

  if (qudaState.eig_handles[id] != nullptr) {
    QudaEigParam *eig_param = static_cast<QudaEigParam *>(qudaState.eig_handles[id]);

    delete eig_param;
    qudaState.eig_handles[id] = nullptr;
  }
}

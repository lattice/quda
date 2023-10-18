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
#include <mpi.h>

#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static openQCD_QudaState_t qudaState = {false, false, false, false, {}, {}};

using namespace quda;

// code for NVTX taken from Jiri Kraus' blog post:
// http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

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
  // add NVTX markup if enabled
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

template <bool start> void inline qudaopenqcd_called(const char *func) { qudaopenqcd_called<start>(func, getVerbosity()); }


/**
 * @brief      Just a simple key-value store
 */
class KeyValueStore {
private:
    std::unordered_map<std::string, std::unordered_map<std::string, std::tuple<std::string, std::string>>> store;
    std::unordered_map<std::string, std::string> *map = nullptr;

public:
    /**
     * @brief      Sets a key value pair
     *
     * @param[in]  section  The section
     * @param[in]  key      The key
     * @param[in]  value    The value
     */
    void set(const std::string& section, const std::string& key, const std::string& value) {
        if (map != nullptr) {
            auto mvalue = map->find(value);
            if (mvalue != map->end()) {
                //store[section][key] = mvalue->second;
                std::get<0>(store[section][key]) = mvalue->second;
                std::get<1>(store[section][key]) = value;
                return;
            }
        }
        std::get<0>(store[section][key]) = value;
        std::get<1>(store[section][key]) = value;
        //store[section][key] = value;
    }

    void set_map(std::unordered_map<std::string, std::string> *_map) {
      map = _map;
    }

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
    template<typename T>
    T get(const std::string& section, const std::string& key, T default_value = T()) {
        int idx;
        std::string rkey;
        std::smatch match;
        std::regex p_key("([^\\[]+)\\[(\\d+)\\]"); // key[idx]
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
                  for (int i=0; i<idx; i++) {
                    ss >> dummy;
                  }
                  if (ss >> result) {
                    return static_cast<T>(result);
                  }
                } else {
                  T result, dummy;
                  for (int i=0; i<idx; i++) {
                    ss >> dummy;
                  }
                  if (ss >> result) {
                    return result;
                  }
                }
            }
        }
        return default_value; // Return default value for non-existent keys
    }

    /**
     * @brief      Fill the store with entries from an ini-file
     *
     * @param[in]  filename  The filename
     */
    void load(const std::string& filename) {
        std::string line, section;
        std::smatch match;
        std::ifstream file(filename.c_str());

        std::regex p_section("^\\s*\\[([\\w\\ ]+)\\].*$"); // [section]
        std::regex p_comment("^[^#]*(\\s*#.*)$"); // line # comment
        std::regex p_key_val("^([^\\s]+)\\s+(.*[^\\s]+)\\s*$"); // key value

        if (file.is_open()) {

            while (std::getline(file, line)) {

                // remove all comments
                if (std::regex_search(line, match, p_comment)) {
                    line.erase(match.position(1));
                }

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
    void dump(std::string _section = "") {
        for (const auto& section : store) {
            if (_section == "" || _section == section.first) {
                std::cout << "[" << section.first << "]" << std::endl;
                for (const auto& pair : section.second) {
                    std::cout << "  " << pair.first << " = " << std::get<1>(pair.second);
                    if (std::get<0>(pair.second) != std::get<1>(pair.second)) {
                      std::cout << " # " << std::get<0>(pair.second);
                    }
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

  for (int i=0; i<4; i++) {
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
static int rankFromCoords(const int *coords, void *fdata) // TODO:
{
  int *base = static_cast<int *>(fdata);
  int *NPROC = base + 1;
  int *ranks = base + 5;
  int i;

  i = coords[3] + NPROC[3]*(coords[2] + NPROC[2]*(coords[1] + NPROC[1]*(coords[0])));
  return ranks[i];
}


/**
 * Set set the local dimensions and machine topology for QUDA to use
 *
 * @param layout Struct defining local dimensions and machine topology
 */
void openQCD_qudaSetLayout(openQCD_QudaLayout_t layout)
{
  for (int dir = 0; dir < 4; ++dir) {
    if (layout.N[dir] % 2 != 0) {
      errorQuda("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }

#ifdef MULTI_GPU
// TODO: would we ever want to run with QMP COMMS?
#ifdef QMP_COMMS
  initCommsGridQuda(4, layout.nproc, nullptr, nullptr);
#else
  initCommsGridQuda(4, layout.nproc, rankFromCoords, (void *)(layout.data));
#endif
  static int device = -1; // enable a default allocation of devices to processes 
#else
  static int device = layout.device;
#endif

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

  param.cpu_prec = QUDA_DOUBLE_PRECISION;  // The precision used by the input fermion fields
  param.cuda_prec = QUDA_DOUBLE_PRECISION; // The precision used by the QUDA solver

  /* AA: This breaks GCR */
  // /* TH added for MG support */
  param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION; // The precision used by the QUDA solver
  param.cuda_prec_precondition = QUDA_HALF_PRECISION; // The precision used by the QUDA solver

  /**
   * The order of the input and output fermion fields. Imposes fieldOrder =
   * QUDA_OPENQCD_FIELD_ORDER in color_spinor_field.h and
   * QUDA_OPENQCD_FIELD_ORDER makes quda to instantiate OpenQCDDiracOrder.
   */
  param.dirac_order = QUDA_OPENQCD_DIRAC_ORDER;

  // Gamma basis of the input and output host fields
  param.gamma_basis = QUDA_OPENQCD_GAMMA_BASIS;

  return param;
}


/**
 * @brief      Initialize quda gauge param struct
 *
 * @param[in]  prec  precision
 *
 * @return     The quda gauge parameter struct.
 */
static QudaGaugeParam newOpenQCDGaugeParam(QudaPrecision prec)
{
  QudaGaugeParam param = newQudaGaugeParam();

  get_local_dims(param.X);
  param.cuda_prec_sloppy = param.cpu_prec = param.cuda_prec = prec;
  param.type = QUDA_SU3_LINKS;

  param.reconstruct_sloppy = param.reconstruct = QUDA_RECONSTRUCT_NO;

  // This make quda to instantiate OpenQCDOrder
  param.gauge_order = QUDA_OPENQCD_GAUGE_ORDER;

  // Seems to have no effect ...
  param.t_boundary = QUDA_PERIODIC_T;

  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  param.scale = 1.0;
  param.anisotropy = 1.0; // 1.0 means not anisotropic
  param.ga_pad = getLinkPadding(param.X); // Why this?

  checkGaugeParam(&param);

  return param;
}


void openQCD_qudaInit(openQCD_QudaInitArgs_t init, openQCD_QudaLayout_t layout)
{
  if (qudaState.initialized) return;
  qudaState.init = init;
  qudaState.layout = layout;

  setVerbosityQuda(qudaState.init.verbosity, "QUDA: ", qudaState.init.logfile);
  qudaopenqcd_called<true>(__func__);
  openQCD_qudaSetLayout(qudaState.layout);
  qudaopenqcd_called<false>(__func__);
  qudaState.initialized = true;
}

void openQCD_qudaFinalize() {
  qudaState.initialized = false;
  endQuda();
}


double openQCD_qudaPlaquette(void)
{
  double plaq[3];

  if (!qudaState.gauge_loaded) {
    errorQuda("Gauge field not loaded into QUDA, cannot calculate plaquette. Call openQCD_qudaGaugeLoad() first.");
    return 0.0;
  }

  plaqQuda(plaq);

  // Note different Nc normalization wrt openQCD!
  return 3.0*plaq[0];
}


void openQCD_qudaGaugeLoad(void *gauge, QudaPrecision prec)
{
  QudaGaugeParam param = newOpenQCDGaugeParam(prec);

  /* Matthias Wagner: optimize that */
  void* buffer = pool_pinned_malloc(4*qudaState.init.volume*18*prec);
  qudaState.init.reorder_gauge_openqcd_to_quda(gauge, buffer);
  loadGaugeQuda(buffer, &param);
  pool_pinned_free(buffer);

  qudaState.gauge_loaded = true;
}


void openQCD_qudaGaugeSave(void *gauge, QudaPrecision prec)
{
  QudaGaugeParam param = newOpenQCDGaugeParam(prec);

  void* buffer = pool_pinned_malloc(4*qudaState.init.volume*18*prec);
  saveGaugeQuda(buffer, &param);
  qudaState.init.reorder_gauge_quda_to_openqcd(buffer, gauge);
  pool_pinned_free(buffer);
}


void openQCD_qudaGaugeFree(void)
{
  freeGaugeQuda();
  qudaState.gauge_loaded = false;
}


void openQCD_qudaCloverLoad(void *clover, double kappa, double csw)
{
  QudaInvertParam param = newOpenQCDParam();
  param.clover_order = QUDA_OPENQCD_CLOVER_ORDER;
  param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
  param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;

  param.kappa = kappa;
  param.clover_csw = csw;
  param.clover_coeff = 0.0;

  loadCloverQuda(clover, NULL, &param);
  qudaState.clover_loaded = true;
}


void openQCD_qudaCloverFree(void)
{
  freeCloverQuda();
  qudaState.clover_loaded = false;
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
  if (!qudaState.gauge_loaded) {
    errorQuda("Gauge field not loaded into QUDA, cannot setup Dirac operator / Clover term. Call openQCD_qudaGaugeLoad() first.");
  }

  QudaInvertParam param = newOpenQCDParam();

  param.dslash_type = QUDA_WILSON_DSLASH;
  param.kappa = p.kappa;
  param.mu = p.mu;
  param.dagger = QUDA_DAG_NO;

  if (p.su3csw != 0.0) {
    param.clover_location = QUDA_CUDA_FIELD_LOCATION; // seems to have no effect?
    param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;
    param.clover_order = QUDA_FLOAT8_CLOVER_ORDER; // what implication has this?

    param.compute_clover = true;
    param.clover_csw = p.su3csw;
    param.clover_coeff = 0.0;

    // Set to Wilson Dirac operator with Clover term
    param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;

    if (!qudaState.clover_loaded) {
      /**
       * Leaving both h_clover = h_clovinv = NULL allocates the clover field on
       * the GPU and finally calls @createCloverQuda to calculate the clover
       * field.
       */
      loadCloverQuda(NULL, NULL, &param); // Create the clover field
      qudaState.clover_loaded = true;
    }
  }

  param.inv_type = QUDA_CG_INVERTER; // just set some, needed?

  // What is the difference? only works with QUDA_MASS_NORMALIZATION
  param.mass_normalization = QUDA_MASS_NORMALIZATION;

  // Extent of the 5th dimension (for domain wall)
  param.Ls = 1;

  return param;
}


/**
 * @brief      Creates a new quda solver parameter struct
 *
 * @param[in]  p     OpenQCD Dirac parameter struct
 *
 * @return     The quda solver parameter struct.
 */
static QudaInvertParam newOpenQCDSolverParam(openQCD_QudaDiracParam_t p)
{
  QudaInvertParam param = newOpenQCDDiracParam(p);

  param.compute_true_res = true;

  param.solution_type = QUDA_MAT_SOLUTION;
  param.solve_type = QUDA_DIRECT_SOLVE;
  param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  param.inv_type_precondition = QUDA_INVALID_INVERTER; // disables any preconditioning

  return param;
}


void openQCD_back_and_forth(void *h_in, void *h_out)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(h_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  // creates a field on the GPU with the same parameter set as the CPU field
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  // transfer the CPU field to GPU
  in = in_h;

  // creates a field on the CPU
  cpuParam.v = h_out;
  cpuParam.location = QUDA_CPU_FIELD_LOCATION;
  ColorSpinorField out_h(cpuParam);

  // creates a zero-field on the GPU
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

  out = in;

  // transfer the GPU field back to CPU
  out_h = out;
}


/**
 * @brief      Calculates the norm of a spinor.
 *
 * @param[in]  h_in  input spinor of type spinor_dble[NSPIN]
 *
 * @return     norm
 */
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

double openQCD_qudaNorm_NoLoads(void *d_in)
{
  return blas::norm2(*reinterpret_cast<ColorSpinorField*>(d_in));
}


void openQCD_qudaGamma(const int dir, void *openQCD_in, void *openQCD_out)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(openQCD_in, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  // creates a field on the GPU with the same parameter set as the CPU field
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField in(cudaParam);

  // transfer the CPU field to GPU
  in = in_h;

  // creates a zero-field on the GPU
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField out(cudaParam);

  // gamma_i run within QUDA using QUDA fields
  switch (dir) {
  case 0: // t direction
    gamma3(out, in);
    break;
  case 1: // x direction
    gamma0(out, in);
    break;
  case 2: // y direction
    gamma1(out, in);
    break;
  case 3: // z direction
    gamma2(out, in);
    break;
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
  default:
    errorQuda("Unknown gamma: %d\n", dir);
  }

  // creates a field on the CPU
  cpuParam.v = openQCD_out;
  cpuParam.location = QUDA_CPU_FIELD_LOCATION;
  ColorSpinorField out_h(cpuParam);

  // transfer the GPU field back to CPU
  out_h = out;
}


void* openQCD_qudaH2D(void *openQCD_field)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(openQCD_field, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField in_h(cpuParam);

  // creates a field on the GPU with the same parameter set as the CPU field
  ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
  ColorSpinorField *in = new ColorSpinorField(cudaParam);

  *in = in_h; // transfer the CPU field to GPU

  return in;
}


void openQCD_qudaSpinorFree(void** quda_field)
{
  delete reinterpret_cast<ColorSpinorField*>(*quda_field);
  *quda_field = nullptr;
}

void openQCD_qudaD2H(void *quda_field, void *openQCD_field)
{
  // sets up the necessary parameters
  QudaInvertParam param = newOpenQCDParam();

  // creates a field on the CPU
  ColorSpinorParam cpuParam(openQCD_field, param, get_local_dims(), false, QUDA_CPU_FIELD_LOCATION);
  ColorSpinorField out_h(cpuParam);

  ColorSpinorField* in = reinterpret_cast<ColorSpinorField*>(quda_field);
  ColorSpinorField out(*in);

  out_h = out; // transfer the GPU field to CPU
}


void openQCD_qudaDw(void *src, void *dst, openQCD_QudaDiracParam_t p)
{
  QudaInvertParam param = newOpenQCDDiracParam(p);

  // both fields reside on the CPU
  param.input_location = QUDA_CPU_FIELD_LOCATION;
  param.output_location = QUDA_CPU_FIELD_LOCATION;

  MatQuda(static_cast<char *>(dst), static_cast<char *>(src), &param);
  /* AA: QUDA applies - Dw */
  /* blas::ax(-1.0, dst); */
}


double openQCD_qudaGCR(void *source, void *solution,
  openQCD_QudaDiracParam_t dirac_param, openQCD_QudaGCRParam_t gcr_param)
{
  QudaInvertParam param = newOpenQCDSolverParam(dirac_param);

  // both fields reside on the CPU
  param.input_location = QUDA_CPU_FIELD_LOCATION;
  param.output_location = QUDA_CPU_FIELD_LOCATION;

  param.inv_type = QUDA_GCR_INVERTER;
  param.tol = gcr_param.tol;
  param.maxiter = gcr_param.nmx;
  param.gcrNkrylov = gcr_param.nkv;
  param.reliable_delta = gcr_param.reliable_delta;

  invertQuda(static_cast<char *>(solution), static_cast<char *>(source), &param);

  printfQuda("true_res    = %.2e\n", param.true_res);
  printfQuda("true_res_hq = %.2e\n", param.true_res_hq);
  printfQuda("iter        = %d\n",   param.iter);
  printfQuda("gflops      = %.2e\n", param.gflops);
  printfQuda("secs        = %.2e\n", param.secs);

  return param.true_res;
}


void* openQCD_qudaSolverSetup(char *infile, char *section)
{
  int my_rank;
  void *mgprec;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Allocate on the heap
  QudaInvertParam* param = new QudaInvertParam(newQudaInvertParam());
  QudaInvertParam* invert_param_mg = new QudaInvertParam(newQudaInvertParam());
  QudaMultigridParam* multigrid_param = new QudaMultigridParam(newQudaMultigridParam());

  // Some default settings
  // Some of them should not be changed
  param->verbosity = QUDA_SUMMARIZE;
  param->cpu_prec = QUDA_DOUBLE_PRECISION;
  param->cuda_prec = QUDA_DOUBLE_PRECISION;
  param->cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  param->cuda_prec_precondition = QUDA_HALF_PRECISION;
  param->dirac_order = QUDA_OPENQCD_DIRAC_ORDER;
  param->gamma_basis = QUDA_OPENQCD_GAMMA_BASIS;
  param->dslash_type = QUDA_WILSON_DSLASH;
  param->kappa = 1.0/(2.0*(qudaState.layout.dirac_parms.m0+4.0));
  param->mu = 0.0;
  param->dagger = QUDA_DAG_NO;
  param->solution_type = QUDA_MAT_SOLUTION;
  param->solve_type = QUDA_DIRECT_SOLVE;
  param->matpc_type = QUDA_MATPC_EVEN_EVEN;
  param->solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  param->inv_type_precondition = QUDA_INVALID_INVERTER; // disables any preconditioning  
  param->mass_normalization = QUDA_MASS_NORMALIZATION;

  if (qudaState.layout.dirac_parms.su3csw != 0.0) {
    param->clover_location = QUDA_CUDA_FIELD_LOCATION; // seems to have no effect?
    param->clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    param->clover_cuda_prec = QUDA_DOUBLE_PRECISION;

    param->clover_csw = qudaState.layout.dirac_parms.su3csw;
    param->clover_coeff = 0.0;

    // Set to Wilson Dirac operator with Clover term
    param->dslash_type = QUDA_CLOVER_WILSON_DSLASH;

    if (qudaState.layout.flds_parms.gauge == OPENQCD_GAUGE_SU3) {
      param->clover_order = QUDA_FLOAT8_CLOVER_ORDER; // what implication has this?
      param->compute_clover = true;
    } else {
      param->clover_order = QUDA_OPENQCD_CLOVER_ORDER;
    }
  }

  if (my_rank == 0) {

    std::unordered_map<std::string, std::string> enum_map = {
      {"QUDA_CG_INVERTER",                      std::to_string(QUDA_CG_INVERTER)},
      {"QUDA_BICGSTAB_INVERTER",                std::to_string(QUDA_BICGSTAB_INVERTER)},
      {"QUDA_GCR_INVERTER",                     std::to_string(QUDA_GCR_INVERTER)},
      {"QUDA_MR_INVERTER",                      std::to_string(QUDA_MR_INVERTER)},
      {"QUDA_SD_INVERTER",                      std::to_string(QUDA_SD_INVERTER)},
      {"QUDA_PCG_INVERTER",                     std::to_string(QUDA_PCG_INVERTER)},
      {"QUDA_EIGCG_INVERTER",                   std::to_string(QUDA_EIGCG_INVERTER)},
      {"QUDA_INC_EIGCG_INVERTER",               std::to_string(QUDA_INC_EIGCG_INVERTER)},
      {"QUDA_GMRESDR_INVERTER",                 std::to_string(QUDA_GMRESDR_INVERTER)},
      {"QUDA_GMRESDR_PROJ_INVERTER",            std::to_string(QUDA_GMRESDR_PROJ_INVERTER)},
      {"QUDA_GMRESDR_SH_INVERTER",              std::to_string(QUDA_GMRESDR_SH_INVERTER)},
      {"QUDA_FGMRESDR_INVERTER",                std::to_string(QUDA_FGMRESDR_INVERTER)},
      {"QUDA_MG_INVERTER",                      std::to_string(QUDA_MG_INVERTER)},
      {"QUDA_BICGSTABL_INVERTER",               std::to_string(QUDA_BICGSTABL_INVERTER)},
      {"QUDA_CGNE_INVERTER",                    std::to_string(QUDA_CGNE_INVERTER)},
      {"QUDA_CGNR_INVERTER",                    std::to_string(QUDA_CGNR_INVERTER)},
      {"QUDA_CG3_INVERTER",                     std::to_string(QUDA_CG3_INVERTER)},
      {"QUDA_CG3NE_INVERTER",                   std::to_string(QUDA_CG3NE_INVERTER)},
      {"QUDA_CG3NR_INVERTER",                   std::to_string(QUDA_CG3NR_INVERTER)},
      {"QUDA_CA_CG_INVERTER",                   std::to_string(QUDA_CA_CG_INVERTER)},
      {"QUDA_CA_CGNE_INVERTER",                 std::to_string(QUDA_CA_CGNE_INVERTER)},
      {"QUDA_CA_CGNR_INVERTER",                 std::to_string(QUDA_CA_CGNR_INVERTER)},
      {"QUDA_CA_GCR_INVERTER",                  std::to_string(QUDA_CA_GCR_INVERTER)},
      {"QUDA_INVALID_INVERTER",                 std::to_string(QUDA_INVALID_INVERTER)},
      {"QUDA_MAT_SOLUTION",                     std::to_string(QUDA_MAT_SOLUTION)},
      {"QUDA_MATDAG_MAT_SOLUTION",              std::to_string(QUDA_MATDAG_MAT_SOLUTION)},
      {"QUDA_MATPC_SOLUTION",                   std::to_string(QUDA_MATPC_SOLUTION)},
      {"QUDA_MATPC_DAG_SOLUTION",               std::to_string(QUDA_MATPC_DAG_SOLUTION)},
      {"QUDA_MATPCDAG_MATPC_SOLUTION",          std::to_string(QUDA_MATPCDAG_MATPC_SOLUTION)},
      {"QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION",    std::to_string(QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION)},
      {"QUDA_INVALID_SOLUTION",                 std::to_string(QUDA_INVALID_SOLUTION)},
      {"QUDA_DIRECT_SOLVE",                     std::to_string(QUDA_DIRECT_SOLVE)},
      {"QUDA_NORMOP_SOLVE",                     std::to_string(QUDA_NORMOP_SOLVE)},
      {"QUDA_DIRECT_PC_SOLVE",                  std::to_string(QUDA_DIRECT_PC_SOLVE)},
      {"QUDA_NORMOP_PC_SOLVE",                  std::to_string(QUDA_NORMOP_PC_SOLVE)},
      {"QUDA_NORMERR_SOLVE",                    std::to_string(QUDA_NORMERR_SOLVE)},
      {"QUDA_NORMERR_PC_SOLVE",                 std::to_string(QUDA_NORMERR_PC_SOLVE)},
      {"QUDA_NORMEQ_SOLVE",                     std::to_string(QUDA_NORMEQ_SOLVE)},
      {"QUDA_NORMEQ_PC_SOLVE",                  std::to_string(QUDA_NORMEQ_PC_SOLVE)},
      {"QUDA_INVALID_SOLVE",                    std::to_string(QUDA_INVALID_SOLVE)},
      {"QUDA_MATPC_EVEN_EVEN",                  std::to_string(QUDA_MATPC_EVEN_EVEN)},
      {"QUDA_MATPC_ODD_ODD",                    std::to_string(QUDA_MATPC_ODD_ODD)},
      {"QUDA_MATPC_EVEN_EVEN_ASYMMETRIC",       std::to_string(QUDA_MATPC_EVEN_EVEN_ASYMMETRIC)},
      {"QUDA_MATPC_ODD_ODD_ASYMMETRIC",         std::to_string(QUDA_MATPC_ODD_ODD_ASYMMETRIC)},
      {"QUDA_MATPC_INVALID",                    std::to_string(QUDA_MATPC_INVALID)},
      {"QUDA_DEFAULT_NORMALIZATION",            std::to_string(QUDA_DEFAULT_NORMALIZATION)},
      {"QUDA_SOURCE_NORMALIZATION",             std::to_string(QUDA_SOURCE_NORMALIZATION)},
      {"QUDA_QUARTER_PRECISION",                std::to_string(QUDA_QUARTER_PRECISION)},
      {"QUDA_HALF_PRECISION",                   std::to_string(QUDA_HALF_PRECISION)},
      {"QUDA_SINGLE_PRECISION",                 std::to_string(QUDA_SINGLE_PRECISION)},
      {"QUDA_DOUBLE_PRECISION",                 std::to_string(QUDA_DOUBLE_PRECISION)},
      {"QUDA_INVALID_PRECISION",                std::to_string(QUDA_INVALID_PRECISION)},
      {"QUDA_BOOLEAN_FALSE",                    std::to_string(QUDA_BOOLEAN_FALSE)},
      {"QUDA_BOOLEAN_TRUE",                     std::to_string(QUDA_BOOLEAN_TRUE)},
      {"QUDA_BOOLEAN_INVALID",                  std::to_string(QUDA_BOOLEAN_INVALID)},
      {"QUDA_COMPUTE_NULL_VECTOR_NO",           std::to_string(QUDA_COMPUTE_NULL_VECTOR_NO)},
      {"QUDA_COMPUTE_NULL_VECTOR_YES",          std::to_string(QUDA_COMPUTE_NULL_VECTOR_YES)},
      {"QUDA_COMPUTE_NULL_VECTOR_INVALID",      std::to_string(QUDA_COMPUTE_NULL_VECTOR_INVALID)},
      {"QUDA_MG_CYCLE_VCYCLE",                  std::to_string(QUDA_MG_CYCLE_VCYCLE)},
      {"QUDA_MG_CYCLE_FCYCLE",                  std::to_string(QUDA_MG_CYCLE_FCYCLE)},
      {"QUDA_MG_CYCLE_WCYCLE",                  std::to_string(QUDA_MG_CYCLE_WCYCLE)},
      {"QUDA_MG_CYCLE_RECURSIVE",               std::to_string(QUDA_MG_CYCLE_RECURSIVE)},
      {"QUDA_MG_CYCLE_INVALID",                 std::to_string(QUDA_MG_CYCLE_INVALID)},
      {"QUDA_CPU_FIELD_LOCATION",               std::to_string(QUDA_CPU_FIELD_LOCATION)},
      {"QUDA_CUDA_FIELD_LOCATION",              std::to_string(QUDA_CUDA_FIELD_LOCATION)},
      {"QUDA_INVALID_FIELD_LOCATION",           std::to_string(QUDA_INVALID_FIELD_LOCATION)},
      {"QUDA_TWIST_SINGLET",                    std::to_string(QUDA_TWIST_SINGLET)},
      {"QUDA_TWIST_NONDEG_DOUBLET",             std::to_string(QUDA_TWIST_NONDEG_DOUBLET)},
      {"QUDA_TWIST_NO",                         std::to_string(QUDA_TWIST_NO)},
      {"QUDA_TWIST_INVALID",                    std::to_string(QUDA_TWIST_INVALID)},
      {"QUDA_DAG_NO",                           std::to_string(QUDA_DAG_NO)},
      {"QUDA_DAG_YES",                          std::to_string(QUDA_DAG_YES)},
      {"QUDA_DAG_INVALID",                      std::to_string(QUDA_DAG_INVALID)},
      {"QUDA_KAPPA_NORMALIZATION",              std::to_string(QUDA_KAPPA_NORMALIZATION)},
      {"QUDA_MASS_NORMALIZATION",               std::to_string(QUDA_MASS_NORMALIZATION)},
      {"QUDA_ASYMMETRIC_MASS_NORMALIZATION",    std::to_string(QUDA_ASYMMETRIC_MASS_NORMALIZATION)},
      {"QUDA_INVALID_NORMALIZATION",            std::to_string(QUDA_INVALID_NORMALIZATION)},
      {"QUDA_PRESERVE_SOURCE_NO",               std::to_string(QUDA_PRESERVE_SOURCE_NO)},
      {"QUDA_PRESERVE_SOURCE_YES",              std::to_string(QUDA_PRESERVE_SOURCE_YES)},
      {"QUDA_PRESERVE_SOURCE_INVALID",          std::to_string(QUDA_PRESERVE_SOURCE_INVALID)},
      {"QUDA_USE_INIT_GUESS_NO",                std::to_string(QUDA_USE_INIT_GUESS_NO)},
      {"QUDA_USE_INIT_GUESS_YES",               std::to_string(QUDA_USE_INIT_GUESS_YES)},
      {"QUDA_USE_INIT_GUESS_INVALID",           std::to_string(QUDA_USE_INIT_GUESS_INVALID)},
      {"QUDA_SILENT",                           std::to_string(QUDA_SILENT)},
      {"QUDA_SUMMARIZE",                        std::to_string(QUDA_SUMMARIZE)},
      {"QUDA_VERBOSE",                          std::to_string(QUDA_VERBOSE)},
      {"QUDA_DEBUG_VERBOSE",                    std::to_string(QUDA_DEBUG_VERBOSE)},
      {"QUDA_INVALID_VERBOSITY",                std::to_string(QUDA_INVALID_VERBOSITY)},
      {"QUDA_TUNE_NO",                          std::to_string(QUDA_TUNE_NO)},
      {"QUDA_TUNE_YES",                         std::to_string(QUDA_TUNE_YES)},
      {"QUDA_TUNE_INVALID",                     std::to_string(QUDA_TUNE_INVALID)},
      {"QUDA_POWER_BASIS",                      std::to_string(QUDA_POWER_BASIS)},
      {"QUDA_CHEBYSHEV_BASIS",                  std::to_string(QUDA_CHEBYSHEV_BASIS)},
      {"QUDA_INVALID_BASIS",                    std::to_string(QUDA_INVALID_BASIS)},
      {"QUDA_ADDITIVE_SCHWARZ",                 std::to_string(QUDA_ADDITIVE_SCHWARZ)},
      {"QUDA_MULTIPLICATIVE_SCHWARZ",           std::to_string(QUDA_MULTIPLICATIVE_SCHWARZ)},
      {"QUDA_INVALID_SCHWARZ",                  std::to_string(QUDA_INVALID_SCHWARZ)},
      {"QUDA_MADWF_ACCELERATOR",                std::to_string(QUDA_MADWF_ACCELERATOR)},
      {"QUDA_INVALID_ACCELERATOR",              std::to_string(QUDA_INVALID_ACCELERATOR)},
      {"QUDA_L2_RELATIVE_RESIDUAL",             std::to_string(QUDA_L2_RELATIVE_RESIDUAL)},
      {"QUDA_L2_ABSOLUTE_RESIDUAL",             std::to_string(QUDA_L2_ABSOLUTE_RESIDUAL)},
      {"QUDA_HEAVY_QUARK_RESIDUAL",             std::to_string(QUDA_HEAVY_QUARK_RESIDUAL)},
      {"QUDA_INVALID_RESIDUAL",                 std::to_string(QUDA_INVALID_RESIDUAL)},
      {"QUDA_NULL_VECTOR_SETUP",                std::to_string(QUDA_NULL_VECTOR_SETUP)},
      {"QUDA_TEST_VECTOR_SETUP",                std::to_string(QUDA_TEST_VECTOR_SETUP)},
      {"QUDA_INVALID_SETUP_TYPE",               std::to_string(QUDA_INVALID_SETUP_TYPE)},
      {"QUDA_TRANSFER_AGGREGATE",               std::to_string(QUDA_TRANSFER_AGGREGATE)},
      {"QUDA_TRANSFER_COARSE_KD",               std::to_string(QUDA_TRANSFER_COARSE_KD)},
      {"QUDA_TRANSFER_OPTIMIZED_KD",            std::to_string(QUDA_TRANSFER_OPTIMIZED_KD)},
      {"QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG",  std::to_string(QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)},
      {"QUDA_TRANSFER_INVALID",                 std::to_string(QUDA_TRANSFER_INVALID)}
    };

    KeyValueStore kv;
    kv.set_map(&enum_map);
    kv.load(infile);

    param->verbosity = kv.get<QudaVerbosity>(section, "verbosity", param->verbosity);
    setVerbosity(param->verbosity);

    if (param->verbosity >= QUDA_VERBOSE) {
      kv.dump();
    }

    if (kv.get<std::string>(section, "solver") != "QUDA") {
      errorQuda("Solver section %s is not a quda-solver section\n", section);
    }

    // both fields reside on the CPU
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
    param->reliable_delta_refinement = kv.get<double>(section, "reliable_delta_refinement", param->reliable_delta_refinement);
    param->use_alternative_reliable = kv.get<int>(section, "use_alternative_reliable", param->use_alternative_reliable);
    param->use_sloppy_partial_accumulator = kv.get<int>(section, "use_sloppy_partial_accumulator", param->use_sloppy_partial_accumulator);

    param->solution_accumulator_pipeline = kv.get<int>(section, "solution_accumulator_pipeline", param->solution_accumulator_pipeline);

    param->max_res_increase = kv.get<int>(section, "max_res_increase", param->max_res_increase);
    param->max_res_increase_total = kv.get<int>(section, "max_res_increase_total", param->max_res_increase_total);
    param->max_hq_res_increase = kv.get<int>(section, "max_hq_res_increase", param->max_hq_res_increase);
    param->max_hq_res_restart_total = kv.get<int>(section, "max_hq_res_restart_total", param->max_hq_res_restart_total);

    param->heavy_quark_check = kv.get<int>(section, "heavy_quark_check", param->heavy_quark_check);

    param->pipeline = kv.get<int>(section, "pipeline", param->pipeline);
    param->num_offset = kv.get<int>(section, "num_offset", param->num_offset);
    param->num_src = kv.get<int>(section, "num_src", param->num_src);
    param->num_src_per_sub_partition = kv.get<int>(section, "num_src_per_sub_partition", param->num_src_per_sub_partition);

    param->split_grid[0] = kv.get<int>(section, "split_grid[1]", param->split_grid[0]);
    param->split_grid[1] = kv.get<int>(section, "split_grid[2]", param->split_grid[1]);
    param->split_grid[2] = kv.get<int>(section, "split_grid[3]", param->split_grid[2]);
    param->split_grid[3] = kv.get<int>(section, "split_grid[0]", param->split_grid[3]);

    param->overlap = kv.get<int>(section, "overlap", param->overlap);

    /*param->offset = kv.get<double>(section, "offset", param->offset)[QUDA_MAX_MULTI_SHIFT];
    param->tol_offset = kv.get<double>(section, "tol_offset", param->tol_offset)[QUDA_MAX_MULTI_SHIFT];
    param->tol_hq_offset = kv.get<double>(section, "tol_hq_offset", param->tol_hq_offset)[QUDA_MAX_MULTI_SHIFT];*/

    param->compute_action = kv.get<int>(section, "compute_action", param->compute_action);

    param->solution_type = kv.get<QudaSolutionType>(section, "solution_type", param->solution_type);
    param->solve_type = kv.get<QudaSolveType>(section, "solve_type", param->solve_type);
    param->matpc_type = kv.get<QudaMatPCType>(section, "matpc_type", param->matpc_type);
    param->dagger = kv.get<QudaDagType>(section, "dagger", param->dagger);
    param->mass_normalization = kv.get<QudaMassNormalization>(section, "mass_normalization", param->mass_normalization);
    param->solver_normalization = kv.get<QudaSolverNormalization>(section, "solver_normalization", param->solver_normalization);

    param->preserve_source = kv.get<QudaPreserveSource>(section, "preserve_source", param->preserve_source);

    param->cpu_prec = kv.get<QudaPrecision>(section, "cpu_prec", param->cpu_prec);
    param->cuda_prec = kv.get<QudaPrecision>(section, "cuda_prec", param->cuda_prec);
    param->cuda_prec_sloppy = kv.get<QudaPrecision>(section, "cuda_prec_sloppy", param->cuda_prec_sloppy);
    param->cuda_prec_refinement_sloppy = kv.get<QudaPrecision>(section, "cuda_prec_refinement_sloppy", param->cuda_prec_refinement_sloppy);
    param->cuda_prec_precondition = kv.get<QudaPrecision>(section, "cuda_prec_precondition", param->cuda_prec_precondition);
    param->cuda_prec_eigensolver = kv.get<QudaPrecision>(section, "cuda_prec_eigensolver", param->cuda_prec_eigensolver);

    param->clover_location = kv.get<QudaFieldLocation>(section, "clover_location", param->clover_location);
    param->clover_cpu_prec = kv.get<QudaPrecision>(section, "clover_cpu_prec", param->clover_cpu_prec);
    param->clover_cuda_prec = kv.get<QudaPrecision>(section, "clover_cuda_prec", param->clover_cuda_prec);
    param->clover_cuda_prec_sloppy = kv.get<QudaPrecision>(section, "clover_cuda_prec_sloppy", param->clover_cuda_prec_sloppy);
    param->clover_cuda_prec_refinement_sloppy = kv.get<QudaPrecision>(section, "clover_cuda_prec_refinement_sloppy", param->clover_cuda_prec_refinement_sloppy);
    param->clover_cuda_prec_precondition = kv.get<QudaPrecision>(section, "clover_cuda_prec_precondition", param->clover_cuda_prec_precondition);
    param->clover_cuda_prec_eigensolver = kv.get<QudaPrecision>(section, "clover_cuda_prec_eigensolver", param->clover_cuda_prec_eigensolver);

    param->use_init_guess = kv.get<QudaUseInitGuess>(section, "use_init_guess", param->use_init_guess);

    param->clover_csw = kv.get<double>(section, "clover_csw", param->clover_csw);
    param->clover_coeff = kv.get<double>(section, "clover_coeff", param->clover_coeff);
    param->clover_rho = kv.get<double>(section, "clover_rho", param->clover_rho);
    param->compute_clover_trlog = kv.get<int>(section, "compute_clover_trlog", param->compute_clover_trlog);
    param->tune = kv.get<QudaTune>(section, "tune", param->tune);
    param->Nsteps = kv.get<int>(section, "Nsteps", param->Nsteps);
    param->gcrNkrylov = kv.get<int>(section, "gcrNkrylov", param->gcrNkrylov);

    param->inv_type_precondition = kv.get<QudaInverterType>(section, "inv_type_precondition", param->inv_type_precondition);
    param->deflate = kv.get<QudaBoolean>(section, "deflate", param->deflate);
    param->verbosity_precondition = kv.get<QudaVerbosity>(section, "verbosity_precondition", param->verbosity_precondition);
    param->tol_precondition = kv.get<double>(section, "tol_precondition", param->tol_precondition);
    param->maxiter_precondition = kv.get<int>(section, "maxiter_precondition", param->maxiter_precondition);
    param->omega = kv.get<double>(section, "omega", param->omega);
    param->ca_basis = kv.get<QudaCABasis>(section, "ca_basis", param->ca_basis);
    param->ca_lambda_min = kv.get<double>(section, "ca_lambda_min", param->ca_lambda_min);
    param->ca_lambda_max = kv.get<double>(section, "ca_lambda_max", param->ca_lambda_max);
    param->ca_basis_precondition = kv.get<QudaCABasis>(section, "ca_basis_precondition", param->ca_basis_precondition);
    param->ca_lambda_min_precondition = kv.get<double>(section, "ca_lambda_min_precondition", param->ca_lambda_min_precondition);
    param->ca_lambda_max_precondition = kv.get<double>(section, "ca_lambda_max_precondition", param->ca_lambda_max_precondition);
    param->precondition_cycle = kv.get<int>(section, "precondition_cycle", param->precondition_cycle);
    param->schwarz_type = kv.get<QudaSchwarzType>(section, "schwarz_type", param->schwarz_type);
    param->accelerator_type_precondition = kv.get<QudaAcceleratorType>(section, "accelerator_type_precondition", param->accelerator_type_precondition);

    param->madwf_diagonal_suppressor = kv.get<double>(section, "madwf_diagonal_suppressor", param->madwf_diagonal_suppressor);
    param->madwf_ls = kv.get<int>(section, "madwf_ls", param->madwf_ls);
    param->madwf_null_miniter = kv.get<int>(section, "madwf_null_miniter", param->madwf_null_miniter);
    param->madwf_null_tol = kv.get<double>(section, "madwf_null_tol", param->madwf_null_tol);
    param->madwf_train_maxiter = kv.get<int>(section, "madwf_train_maxiter", param->madwf_train_maxiter);

    param->madwf_param_load = kv.get<QudaBoolean>(section, "madwf_param_load", param->madwf_param_load);
    param->madwf_param_save = kv.get<QudaBoolean>(section, "madwf_param_save", param->madwf_param_save);

    /*param->madwf_param_infile = kv.get<char>(section, "madwf_param_infile", param->madwf_param_infile);
    param->madwf_param_outfile = kv.get<char>(section, "madwf_param_outfile", param->madwf_param_outfile);*/

    param->residual_type = kv.get<QudaResidualType>(section, "residual_type", param->residual_type);

    if (param->inv_type_precondition == QUDA_MG_INVERTER) {

      std::string mg_section = std::string(section) + " Multigrid";

      // (shallow) copy the struct
      *invert_param_mg = *param;

      // these have to be fixed
      invert_param_mg->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
      invert_param_mg->dirac_order = QUDA_DIRAC_ORDER;

      multigrid_param->n_level = kv.get<int>(mg_section, "n_level", multigrid_param->n_level);
      multigrid_param->setup_type = kv.get<QudaSetupType>(mg_section, "setup_type", multigrid_param->setup_type);
      multigrid_param->pre_orthonormalize = kv.get<QudaBoolean>(mg_section, "pre_orthonormalize", multigrid_param->pre_orthonormalize);
      multigrid_param->post_orthonormalize = kv.get<QudaBoolean>(mg_section, "post_orthonormalize", multigrid_param->post_orthonormalize);
      multigrid_param->setup_minimize_memory = kv.get<QudaBoolean>(mg_section, "setup_minimize_memory", multigrid_param->setup_minimize_memory);
      multigrid_param->compute_null_vector = kv.get<QudaComputeNullVector>(mg_section, "compute_null_vector", multigrid_param->compute_null_vector);
      multigrid_param->generate_all_levels = kv.get<QudaBoolean>(mg_section, "generate_all_levels", multigrid_param->generate_all_levels);
      multigrid_param->run_verify = kv.get<QudaBoolean>(mg_section, "run_verify", multigrid_param->run_verify);
      multigrid_param->run_low_mode_check = kv.get<QudaBoolean>(mg_section, "run_low_mode_check", multigrid_param->run_low_mode_check);
      multigrid_param->run_oblique_proj_check = kv.get<QudaBoolean>(mg_section, "run_oblique_proj_check", multigrid_param->run_oblique_proj_check);
      multigrid_param->coarse_guess = kv.get<QudaBoolean>(mg_section, "coarse_guess", multigrid_param->coarse_guess);
      multigrid_param->preserve_deflation = kv.get<QudaBoolean>(mg_section, "preserve_deflation", multigrid_param->preserve_deflation);
      multigrid_param->allow_truncation = kv.get<QudaBoolean>(mg_section, "allow_truncation", multigrid_param->allow_truncation);
      multigrid_param->staggered_kd_dagger_approximation = kv.get<QudaBoolean>(mg_section, "staggered_kd_dagger_approximation", multigrid_param->staggered_kd_dagger_approximation);
      multigrid_param->use_mma = kv.get<QudaBoolean>(mg_section, "use_mma", multigrid_param->use_mma);
      multigrid_param->thin_update_only = kv.get<QudaBoolean>(mg_section, "thin_update_only", multigrid_param->thin_update_only);

      for (int i=0; i<multigrid_param->n_level; i++) {
        std::string subsection = std::string(section) + " Multigrid Level " + std::to_string(i);

        multigrid_param->geo_block_size[i][0] = kv.get<int>(subsection, "geo_block_size[1]", multigrid_param->geo_block_size[i][0]);
        multigrid_param->geo_block_size[i][1] = kv.get<int>(subsection, "geo_block_size[2]", multigrid_param->geo_block_size[i][1]);
        multigrid_param->geo_block_size[i][2] = kv.get<int>(subsection, "geo_block_size[3]", multigrid_param->geo_block_size[i][2]);
        multigrid_param->geo_block_size[i][3] = kv.get<int>(subsection, "geo_block_size[0]", multigrid_param->geo_block_size[i][3]);

        if (i==0) {
          multigrid_param->geo_block_size[i][0] = 4;
          multigrid_param->geo_block_size[i][1] = 4;
          multigrid_param->geo_block_size[i][2] = 4;
          multigrid_param->geo_block_size[i][3] = 4;
        }

        multigrid_param->spin_block_size[i] = kv.get<int>(subsection, "spin_block_size", multigrid_param->spin_block_size[i]);
        multigrid_param->n_vec[i] = kv.get<int>(subsection, "n_vec", multigrid_param->n_vec[i]);
        multigrid_param->precision_null[i] = kv.get<QudaPrecision>(subsection, "precision_null", multigrid_param->precision_null[i]);
        multigrid_param->n_block_ortho[i] = kv.get<int>(subsection, "n_block_ortho", multigrid_param->n_block_ortho[i]);
        multigrid_param->block_ortho_two_pass[i] = kv.get<QudaBoolean>(subsection, "block_ortho_two_pass", multigrid_param->block_ortho_two_pass[i]);
        multigrid_param->verbosity[i] = kv.get<QudaVerbosity>(subsection, "verbosity", multigrid_param->verbosity[i]);
        multigrid_param->setup_inv_type[i] = kv.get<QudaInverterType>(subsection, "setup_inv_type", multigrid_param->setup_inv_type[i]);
        multigrid_param->num_setup_iter[i] = kv.get<int>(subsection, "num_setup_iter", multigrid_param->num_setup_iter[i]);
        multigrid_param->setup_tol[i] = kv.get<double>(subsection, "setup_tol", multigrid_param->setup_tol[i]);
        multigrid_param->setup_maxiter[i] = kv.get<int>(subsection, "setup_maxiter", multigrid_param->setup_maxiter[i]);
        multigrid_param->setup_maxiter_refresh[i] = kv.get<int>(subsection, "setup_maxiter_refresh", multigrid_param->setup_maxiter_refresh[i]);
        multigrid_param->setup_ca_basis[i] = kv.get<QudaCABasis>(subsection, "setup_ca_basis", multigrid_param->setup_ca_basis[i]);
        multigrid_param->setup_ca_basis_size[i] = kv.get<int>(subsection, "setup_ca_basis_size", multigrid_param->setup_ca_basis_size[i]);
        multigrid_param->setup_ca_lambda_min[i] = kv.get<double>(subsection, "setup_ca_lambda_min", multigrid_param->setup_ca_lambda_min[i]);
        multigrid_param->setup_ca_lambda_max[i] = kv.get<double>(subsection, "setup_ca_lambda_max", multigrid_param->setup_ca_lambda_max[i]);

        multigrid_param->coarse_solver[i] = kv.get<QudaInverterType>(subsection, "coarse_solver", multigrid_param->coarse_solver[i]);
        multigrid_param->coarse_solver_tol[i] = kv.get<double>(subsection, "coarse_solver_tol", multigrid_param->coarse_solver_tol[i]);
        multigrid_param->coarse_solver_maxiter[i] = kv.get<int>(subsection, "coarse_solver_maxiter", multigrid_param->coarse_solver_maxiter[i]);
        multigrid_param->coarse_solver_ca_basis[i] = kv.get<QudaCABasis>(subsection, "coarse_solver_ca_basis", multigrid_param->coarse_solver_ca_basis[i]);
        multigrid_param->coarse_solver_ca_basis_size[i] = kv.get<int>(subsection, "coarse_solver_ca_basis_size", multigrid_param->coarse_solver_ca_basis_size[i]);
        multigrid_param->coarse_solver_ca_lambda_min[i] = kv.get<double>(subsection, "coarse_solver_ca_lambda_min", multigrid_param->coarse_solver_ca_lambda_min[i]);
        multigrid_param->coarse_solver_ca_lambda_max[i] = kv.get<double>(subsection, "coarse_solver_ca_lambda_max", multigrid_param->coarse_solver_ca_lambda_max[i]);
        multigrid_param->smoother[i] = kv.get<QudaInverterType>(subsection, "smoother", multigrid_param->smoother[i]);
        multigrid_param->smoother_tol[i] = kv.get<double>(subsection, "smoother_tol", multigrid_param->smoother_tol[i]);
        multigrid_param->nu_pre[i] = kv.get<int>(subsection, "nu_pre", multigrid_param->nu_pre[i]);
        multigrid_param->nu_post[i] = kv.get<int>(subsection, "nu_post", multigrid_param->nu_post[i]);
        multigrid_param->smoother_solver_ca_basis[i] = kv.get<QudaCABasis>(subsection, "smoother_solver_ca_basis", multigrid_param->smoother_solver_ca_basis[i]);
        multigrid_param->smoother_solver_ca_lambda_min[i] = kv.get<double>(subsection, "smoother_solver_ca_lambda_min", multigrid_param->smoother_solver_ca_lambda_min[i]);
        multigrid_param->smoother_solver_ca_lambda_max[i] = kv.get<double>(subsection, "smoother_solver_ca_lambda_max", multigrid_param->smoother_solver_ca_lambda_max[i]);
        multigrid_param->omega[i] = kv.get<double>(subsection, "omega", multigrid_param->omega[i]);
        multigrid_param->smoother_halo_precision[i] = kv.get<QudaPrecision>(subsection, "smoother_halo_precision", multigrid_param->smoother_halo_precision[i]);
        multigrid_param->smoother_schwarz_type[i] = kv.get<QudaSchwarzType>(subsection, "smoother_schwarz_type", multigrid_param->smoother_schwarz_type[i]);
        multigrid_param->smoother_schwarz_cycle[i] = kv.get<int>(subsection, "smoother_schwarz_cycle", multigrid_param->smoother_schwarz_cycle[i]);
        multigrid_param->coarse_grid_solution_type[i] = kv.get<QudaSolutionType>(subsection, "coarse_grid_solution_type", multigrid_param->coarse_grid_solution_type[i]);
        multigrid_param->smoother_solve_type[i] = kv.get<QudaSolveType>(subsection, "smoother_solve_type", multigrid_param->smoother_solve_type[i]);
        multigrid_param->cycle_type[i] = kv.get<QudaMultigridCycleType>(subsection, "cycle_type", multigrid_param->cycle_type[i]);
        multigrid_param->global_reduction[i] = kv.get<QudaBoolean>(subsection, "global_reduction", multigrid_param->global_reduction[i]);
        multigrid_param->location[i] = kv.get<QudaFieldLocation>(subsection, "location", multigrid_param->location[i]);
        multigrid_param->setup_location[i] = kv.get<QudaFieldLocation>(subsection, "setup_location", multigrid_param->setup_location[i]);
        multigrid_param->use_eig_solver[i] = kv.get<QudaBoolean>(subsection, "use_eig_solver", multigrid_param->use_eig_solver[i]);

        /*multigrid_param->vec_load[i] = kv.get<QudaBoolean>(subsection, "vec_load", multigrid_param->vec_load[i]);
        multigrid_param->vec_infile[i] = kv.get<char>(subsection, "vec_infile", multigrid_param->vec_infile[i]);
        multigrid_param->vec_store[i] = kv.get<QudaBoolean>(subsection, "vec_store", multigrid_param->vec_store[i]);
        multigrid_param->vec_outfile[i] = kv.get<char>(subsection, "vec_outfile", multigrid_param->vec_outfile[i]);*/

        multigrid_param->mu_factor[i] = kv.get<double>(subsection, "mu_factor", multigrid_param->mu_factor[i]);
        multigrid_param->transfer_type[i] = kv.get<QudaTransferType>(subsection, "transfer_type", multigrid_param->transfer_type[i]);
      }
    }
  }

  // transfer of the struct to all the processes
  MPI_Bcast((void*) param,           sizeof(*param),           MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*) invert_param_mg, sizeof(*invert_param_mg), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*) multigrid_param, sizeof(*multigrid_param), MPI_BYTE, 0, MPI_COMM_WORLD);
  multigrid_param->invert_param = invert_param_mg;

  if (qudaState.layout.h_gauge != nullptr) {
    logQuda(QUDA_VERBOSE, "Loading gauge field from openQCD ...\n");
    PUSH_RANGE("openQCD_qudaGaugeLoad",3);
    openQCD_qudaGaugeLoad(qudaState.layout.h_gauge(), QUDA_DOUBLE_PRECISION);
    POP_RANGE;
  }

  if (qudaState.layout.dirac_parms.su3csw != 0.0) {
    if (qudaState.layout.flds_parms.gauge == OPENQCD_GAUGE_SU3) {
      /**
       * Leaving both h_clover = h_clovinv = NULL allocates the clover field on
       * the GPU and finally calls @createCloverQuda to calculate the clover
       * field.
       */
      logQuda(QUDA_VERBOSE, "Generating clover field in QUDA ...\n");
      PUSH_RANGE("loadCloverQuda",3);
      loadCloverQuda(NULL, NULL, param);
      POP_RANGE;
    } else {
      /**
       * Transfer the SW-field from openQCD.
       */
      logQuda(QUDA_VERBOSE, "Loading clover field from openQCD ...\n");
      PUSH_RANGE("openQCD_qudaCloverLoad",3);
      openQCD_qudaCloverLoad(qudaState.layout.h_sw(), param->kappa, param->clover_csw);
      POP_RANGE;

      //loadCloverQuda(qudaState.layout.h_sw(), NULL, param);
      // The above line would be prefered over openQCD_qudaCloverLoad, but throws this error, no idea why?
      //QUDA: ERROR: qudaEventRecord_ returned CUDA_ERROR_ILLEGAL_ADDRESS
      // (timer.h:82 in start())
      // (rank 0, host yoshi, quda_api.cpp:72 in void quda::target::cuda::set_driver_error(CUresult, const char*, const char*, const char*, const char*, bool)())
      //QUDA:        last kernel called was (name=N4quda10CopyCloverINS_6clover11FloatNOrderIdLi72ELi2ELb0ELb1ELb0EEENS1_12OpenQCDOrderIdLi72EEEddEE,volume=32x16x16x64,aux=GPU-offline,vol=524288precision=8Nc=3,compute_diagonal)
    }
  }

  if (param->inv_type_precondition == QUDA_MG_INVERTER) {
    PUSH_RANGE("newMultigridQuda",4);
    mgprec = newMultigridQuda(multigrid_param);
    param->preconditioner = mgprec;
    POP_RANGE;
  }

  checkInvertParam(param);
  if (param->verbosity >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(param);
  }

  if (param->inv_type_precondition == QUDA_MG_INVERTER) {
    checkMultigridParam(multigrid_param);
    if (param->verbosity >= QUDA_DEBUG_VERBOSE) {
      printQudaMultigridParam(multigrid_param);
    }
  }

  return (void*) param;
}

double openQCD_qudaInvert(void *param, double mu, void *source, void *solution, int *status)
{
  QudaInvertParam* invert_param = static_cast<QudaInvertParam*>(param);
  invert_param->mu = mu;

  logQuda(QUDA_VERBOSE, "Calling invertQuda() ...\n");
  PUSH_RANGE("invertQuda",5);
  invertQuda(static_cast<char *>(solution), static_cast<char *>(source), invert_param);
  POP_RANGE;

  if (invert_param->verbosity >= QUDA_VERBOSE) {
    logQuda(QUDA_VERBOSE, "openQCD_qudaInvert()\n");
    logQuda(QUDA_VERBOSE, "  true_res    = %.2e\n", invert_param->true_res);
    logQuda(QUDA_VERBOSE, "  true_res_hq = %.2e\n", invert_param->true_res_hq);
    logQuda(QUDA_VERBOSE, "  iter        = %d\n",   invert_param->iter);
    logQuda(QUDA_VERBOSE, "  gflops      = %.2e\n", invert_param->gflops);
    logQuda(QUDA_VERBOSE, "  secs        = %.2e\n", invert_param->secs);
  }

  *status = invert_param->true_res <= invert_param->tol ? invert_param->iter : -1;

  return invert_param->true_res;
}


void openQCD_qudaSolverDestroy(void *param)
{
  QudaInvertParam* invert_param = static_cast<QudaInvertParam*>(param);

  if (invert_param->inv_type_precondition == QUDA_MG_INVERTER) {
    destroyMultigridQuda(invert_param->preconditioner);
  }

  delete invert_param;
}


double openQCD_qudaMultigrid(void *source, void *solution, openQCD_QudaDiracParam_t dirac_param)
{
  QudaInvertParam invert_param = newOpenQCDSolverParam(dirac_param);
  QudaInvertParam invert_param_mg = newOpenQCDSolverParam(dirac_param);
  QudaMultigridParam multigrid_param = newQudaMultigridParam();

  //param.verbosity = QUDA_VERBOSE;
  invert_param.reliable_delta = 1e-5;
  invert_param.gcrNkrylov = 20;
  invert_param.maxiter = 2000;
  invert_param.tol = 1e-12;
  invert_param.inv_type = QUDA_GCR_INVERTER;
  invert_param.solution_type = QUDA_MAT_SOLUTION;
  invert_param.solve_type = QUDA_DIRECT_SOLVE;
  invert_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  invert_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  invert_param.inv_type_precondition = QUDA_MG_INVERTER;
  invert_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION; // The precision used by the QUDA solver
  invert_param.cuda_prec_precondition = QUDA_HALF_PRECISION; // The precision used by the QUDA solver

  invert_param_mg.reliable_delta = 1e-5;
  invert_param_mg.gcrNkrylov = 20;
  invert_param_mg.maxiter = 2000;
  invert_param_mg.tol = 1e-12;
  invert_param_mg.inv_type = QUDA_GCR_INVERTER;
  invert_param_mg.solution_type = QUDA_MAT_SOLUTION;
  invert_param_mg.solve_type = QUDA_DIRECT_SOLVE;
  invert_param_mg.matpc_type = QUDA_MATPC_EVEN_EVEN;
  invert_param_mg.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  invert_param_mg.inv_type_precondition = QUDA_MG_INVERTER;
  invert_param_mg.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invert_param_mg.dirac_order = QUDA_DIRAC_ORDER;

  // set the params, hard code the solver
  // parameters copied from recommended settings from Wiki
  multigrid_param.n_level = 2;
  multigrid_param.generate_all_levels = QUDA_BOOLEAN_TRUE;
  multigrid_param.run_verify = QUDA_BOOLEAN_FALSE;
  multigrid_param.invert_param = &invert_param_mg;
  multigrid_param.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;

  // try setting minimal parameters - leave rest to default
  // level 0 fine
  multigrid_param.geo_block_size[0][0] = 4; // xytz
  multigrid_param.geo_block_size[0][1] = 4;
  multigrid_param.geo_block_size[0][2] = 4;
  multigrid_param.geo_block_size[0][3] = 4;
  multigrid_param.n_vec[0] = 24;
  multigrid_param.spin_block_size[0] = 2;
  multigrid_param.precision_null[0] = QUDA_HALF_PRECISION; 
  multigrid_param.smoother[0] = QUDA_CA_GCR_INVERTER;
  multigrid_param.smoother_tol[0] = 0.25;
  multigrid_param.location[0] = QUDA_CUDA_FIELD_LOCATION;
  multigrid_param.nu_pre[0] = 0;
  multigrid_param.nu_post[0] = 8;
  multigrid_param.omega[0] = 0.8;
  multigrid_param.smoother_solve_type[0] = QUDA_DIRECT_PC_SOLVE;
  multigrid_param.cycle_type[0] = QUDA_MG_CYCLE_RECURSIVE;
  multigrid_param.coarse_solver[0] = QUDA_GCR_INVERTER;
  multigrid_param.coarse_solver_tol[0] = 0.25;
  multigrid_param.coarse_solver_maxiter[0] = 50;
  multigrid_param.coarse_grid_solution_type[0] = QUDA_MAT_SOLUTION;

  // level 1 coarse
  // no smoother required for innermost
  // so no blocks
  multigrid_param.precision_null[1] = QUDA_HALF_PRECISION;
  multigrid_param.coarse_solver[1] = QUDA_CA_GCR_INVERTER;
  multigrid_param.smoother[1] = QUDA_CA_GCR_INVERTER;
  multigrid_param.smoother_tol[1] = 0.25;
  multigrid_param.spin_block_size[1] = 1;
  multigrid_param.coarse_solver_tol[1] = 0.25;
  multigrid_param.coarse_solver_maxiter[1] = 50;
  multigrid_param.coarse_grid_solution_type[1] = QUDA_MATPC_SOLUTION;
  multigrid_param.smoother_solve_type[1] = QUDA_DIRECT_PC_SOLVE;
  multigrid_param.cycle_type[1] = QUDA_MG_CYCLE_RECURSIVE;
  multigrid_param.location[1] = QUDA_CUDA_FIELD_LOCATION;
  multigrid_param.nu_pre[1] = 0;
  multigrid_param.nu_post[1] = 8;
  multigrid_param.omega[1] = 0.8;

  PUSH_RANGE("newMultigridQuda",4);
  void *mgprec = newMultigridQuda(&multigrid_param);
  invert_param.preconditioner = mgprec;
  POP_RANGE;

  PUSH_RANGE("invertQUDA",5);
  invertQuda(static_cast<char *>(solution), static_cast<char *>(source), &invert_param);
  POP_RANGE;

  destroyMultigridQuda(mgprec);

  printfQuda("true_res    = %.2e\n", invert_param.true_res);
  printfQuda("true_res_hq = %.2e\n", invert_param.true_res_hq);
  printfQuda("iter        = %d\n",   invert_param.iter);
  printfQuda("gflops      = %.2e\n", invert_param.gflops);
  printfQuda("secs        = %.2e\n", invert_param.secs);

  return invert_param.true_res;
}

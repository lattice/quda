#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <blas_quda.h>
// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>
#include <filesystem>
#include "test.h"

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;
QudaMultigridParam mg_param;
QudaInvertParam mg_inv_param;
QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];
QudaEigParam eig_param;
bool use_split_grid = false;
bool use_multi_src = false;

int start_seed = 0;
std::string meas_vec_file_str = "";
std::string base_io_dir = std::filesystem::current_path().string();

// print instructions on how to run the old tests
bool print_legacy_info = false;

bool take_fwd_gflow = false;
bool take_adj_gflow = false;

GaugeField cpuFatQDP = {};
GaugeField cpuLongQDP = {};
GaugeField cpuFatMILC = {};
GaugeField cpuLongMILC = {};

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim);

  // Specific test
  printfQuda("\n%s smearing\n", get_gauge_smear_str(gauge_smear_type));
  switch (gauge_smear_type) {
  case QUDA_GAUGE_SMEAR_APE: printfQuda(" - alpha %f\n", gauge_smear_alpha); break;
  case QUDA_GAUGE_SMEAR_STOUT: printfQuda(" - rho %f\n", gauge_smear_rho); break;
  case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
    printfQuda(" - rho %f\n", gauge_smear_rho);
    printfQuda(" - epsilon %f\n", gauge_smear_epsilon);
    break;
  case QUDA_GAUGE_SMEAR_HYP:
    printfQuda(" - alpha1 %f\n", gauge_smear_alpha1);
    printfQuda(" - alpha2 %f\n", gauge_smear_alpha2);
    printfQuda(" - alpha3 %f\n", gauge_smear_alpha3);
    break;
  case QUDA_GAUGE_SMEAR_WILSON_FLOW:
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: printfQuda(" - epsilon %f\n", gauge_smear_epsilon); break;
  default: errorQuda("Undefined test type %d given", test_type);
  }
  printfQuda(" - smearing steps %d\n", gauge_smear_steps);
  printfQuda(" - smearing ignore direction %d\n", gauge_smear_dir_ignore);
  printfQuda(" - hierarchical nsave %d\n", gauge_n_save);
  printfQuda(" - Measurement interval %d\n", measurement_interval);
  printfQuda(" - n-naiks %d\n", n_naiks);
  printfQuda(" - eps-naiks %f\n", eps_naik);
  printfQuda(" - fermion mass %f\n", mass);
  // printfQuda(" - eps-naiks %f\n", eps_naik);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

void add_adj_hisq_option_group(std::shared_ptr<QUDAApp> quda_app)
{
    //   // Option group for SU(3) related options
  auto opgroup = quda_app->add_option_group("HISQ", "Options controlling adjoint + HISQ measurement parameters");
  opgroup
    ->add_option(
      "--n-naiks",
      n_naiks, "number of naiks term");
  opgroup
    ->add_option(
      "--take-fwd-gflow",
      take_fwd_gflow, "take forward gflow for testing purposes");
  opgroup
    ->add_option(
      "--take-adj-gflow",
      take_adj_gflow, "take adjoint gflow (default is flase)");
}

void add_meas_io_group(std::shared_ptr<QUDAApp> quda_app)
{
    auto opgroup = quda_app->add_option_group("Meas interval", "Options in controlling separate measurements");
    opgroup
    ->add_option(
      "--meas-vec-txt",
      meas_vec_file_str, "measurement interval text file");
    opgroup
    ->add_option(
      "--start-seed",
      start_seed, "start seed for random sources");
    opgroup
    ->add_option(
      "--base-io-dir",
      base_io_dir, "base directory to create data directory");
    
}

void check_naik(double &eps_naik, int &n_naiks)
{
    if (eps_naik != 0 && n_naiks != 2) {
     errorQuda("if eps naik is nonzero, nnaiks must be 2\n");
    }
    if (eps_naik == 0 && n_naiks == 2) {
     errorQuda("eps naik is zero, and nnaiks is 2, dont be lazy and go back and change the configs\n");
    }
}

std::vector<unsigned int> read_meas_int_vec()
{
    std::vector<unsigned int> res = {};
    std::ifstream file(meas_vec_file_str);
    if (!file.is_open()) {
        printfQuda("No measurement int file, constructing meas int vec from step size and total steps\n");
        for(int m=measurement_interval; m <= gauge_smear_steps; m = m + measurement_interval){
            res.push_back(m);
        }
    }
    else {
        printfQuda("Measurement int file registered\n");
        double number;
        while (file >> number) {
            res.push_back(number);
        }
        file.close();
    }
    printfQuda("Size of final meas int vec %lu\n", res.size());
    return res;
}

int linkIndex(const int* x, const int* X) {
    int idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    return idx;
}

void write_files(const QudaFermMeasurements &ferm_meas)
{
   std::cout << "File QUDA rank: " << quda::comm_rank() << "\n";
  std::filesystem::path latfile_path(latfile);
  std::string latfile_dir = latfile_path.parent_path().string();
  std::string latfile_filestr = latfile_path.filename().string();
  std::string deepest_dir = latfile_path.filename().string();  
  std::vector<std::string> dirstr_vec;
  for (const auto& entry : latfile_path.parent_path()){
      dirstr_vec.push_back(entry.string());
      printfQuda(("element added: " + entry.string() +"\n").c_str());
  }
  assert(dirstr_vec.size() >= 2);
  printfQuda(("our deepest dir is " + deepest_dir +"\n").c_str());
  std::filesystem::path base_output_dir(base_io_dir + "/data/");
  if (!std::filesystem::is_directory(base_output_dir)){
      std::filesystem::create_directory(base_output_dir);
  } 
  std::filesystem::path output_dir(base_output_dir.string()+"/"+dirstr_vec[dirstr_vec.size()-2] +"/"+deepest_dir);
  if (!std::filesystem::is_directory(output_dir)){
      std::filesystem::create_directories(output_dir);
  } 
  std::string quark_str = n_naiks > 1 ? "_cbarc" : "_sbars";
  std::string adj_str = take_adj_gflow ? "" : "N";
  std::string output_filestr;
  output_filestr = "_mq" + std::to_string(mass);
  output_filestr += "_naik" + std::to_string(eps_naik);
  output_filestr += "_start" + std::to_string(start_seed) + "_Nsrc" + std::to_string(Nsrc);

  std::string filename_allcon = output_dir.string()+"/"+adj_str+"allcon"+output_filestr;
  std::string filename_tslice = output_dir.string()+"/"+adj_str+"tslice"+output_filestr;

  auto* flow_int_pt = reinterpret_cast<std::vector<unsigned int>*>(ferm_meas.meas_int_vec);
  auto* ppb_data =reinterpret_cast<std::vector<std::vector<std::complex<double>>>*>(*ferm_meas.ppb);
  printfQuda("flow vec size %i \n",(*flow_int_pt).size());
  printfQuda("ppb data size %i \n",(*ppb_data).size());

  std::ofstream out_ppb(filename_allcon);
  std::vector<std::ofstream> out_ppb_vec, out_ppb_t_vec, out_pion_corr_vec;
  out_ppb_t_vec.reserve(ppb_data->size());
  for (int i = 0; i < ppb_data->size(); i++){
    std::string flowN_s = std::to_string(flow_int_pt->at(i));
    std::string epsF_s =  (flow_int_pt->at(i) == 0) ? "" : "_epsF" + std::to_string(gauge_smear_epsilon);
    std::string flow_file_name(filename_tslice+"_FT"+flowN_s+epsF_s);
    out_ppb_t_vec.emplace_back(flow_file_name);
    // out_pion_corr_vec.emplace_back(flow_file_name);
    if (!out_ppb_t_vec.back().is_open()) {
        printfQuda("Error: failed to open ppb_t flow file #%i\n",i);
    } else {
        printfQuda("Successfully opened flow file #%i\n",i);
    }
    // if (!out_pion_corr_vec.back().is_open()) {
    //     printfQuda("Error: failed to open pion_corr flow file #%i\n",i);
    // } else {
    //     printfQuda("Successfully opened flow file #%i\n",i);
    // }
  }
  
    //stuff works above
  if (!out_ppb.is_open()) {
      std::cerr << "Failed to open file: " << filename_allcon << std::endl;
  }
    
  for (const auto& row : *ppb_data) {
      for (const auto& elem : row) {
          out_ppb << elem.real()/(V*comm_size()) << " ";
      }
      out_ppb << "\n"; // Newline after each row
  }
  out_ppb.close();

  std::ofstream out_ppb_t(filename_tslice);
  if (!out_ppb_t.is_open()) {
      std::cerr << "Failed to open file: " << filename_tslice << std::endl;
  }
  auto* ppb_t_data = reinterpret_cast<std::vector<std::vector<std::vector<Complex>>>*>(ferm_meas.ppb_t);

  unsigned int flow_idx = 0;
    printfQuda("out vec size #%i\n",out_ppb_t_vec.size());
  for (const auto& flow_t: *ppb_t_data) {
      assert(out_ppb_t_vec[flow_idx].good());
      printfQuda("begin writing flow time #%i\n",flow_idx);
      for (const auto& s_src : flow_t) {
        // out_ppb_t << "next source\n";
        for (const auto& elem : s_src) {
          out_ppb_t << elem.real()/(V*comm_size()) << " ";
          out_ppb_t_vec[flow_idx] << elem.real()/(V*comm_size()) << " ";
        }
        out_ppb_t << "\n";
        out_ppb_t_vec[flow_idx] << "\n";
      }
    out_ppb_t << "\n";
    // out_ppb_t_vec[flow_idx].close();
    flow_idx += 1;
  }
  out_ppb_t.close();
}

void init()
{
  // Set QUDA internal parameters
  gauge_param = newQudaGaugeParam();
  setStaggeredGaugeParam(gauge_param);
  QudaGaugeSmearParam smear_param;
  if (gauge_smear) {
    smear_param = newQudaGaugeSmearParam();
    setGaugeSmearParam(smear_param);
  }
  printfQuda("quda prec is %d\n",gauge_param.cuda_prec);
  inv_param = newQudaInvertParam();
  mg_inv_param = newQudaInvertParam();
  mg_param = newQudaMultigridParam();
  eig_param = newQudaEigParam();

  if (inv_multigrid) {
    // Set some default values for MG solve types
    setQudaMgSolveTypes();
    setStaggeredMGInvertParam(inv_param);
    // Set sub structures
    mg_param.invert_param = &mg_inv_param;
    for (int i = 0; i < mg_levels; i++) {
      if (mg_eig[i]) {
        mg_eig_param[i] = newQudaEigParam();
        setMultigridEigParam(mg_eig_param[i], i);
        mg_param.eig_param[i] = &mg_eig_param[i];
      } else {
        mg_param.eig_param[i] = nullptr;
      }
    }
    // Set MG
    setStaggeredMultigridParam(mg_param);
  } else {
    setStaggeredInvertParam(inv_param);
  }

  if (inv_deflate) {
    setEigParam(eig_param);
    inv_param.eig_param = &eig_param;
    if (use_split_grid) { errorQuda("Split grid does not work with deflation yet.\n"); }
  } else {
    inv_param.eig_param = nullptr;
  }

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, 1);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuParam(gauge_param);
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField cpuIn = GaugeField(cpuParam);
  cpuFatQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuFatMILC = GaugeField(cpuParam);

  cpuParam.link_type = QUDA_ASQTAD_LONG_LINKS;
  cpuParam.nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuLongQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuLongMILC = GaugeField(cpuParam);

  void *qdp_inlink[4] = {cpuIn.data(0), cpuIn.data(1), cpuIn.data(2), cpuIn.data(3)};
  void *qdp_fatlink[4] = {cpuFatQDP.data(0), cpuFatQDP.data(1), cpuFatQDP.data(2), cpuFatQDP.data(3)};
  void *qdp_longlink[4] = {cpuLongQDP.data(0), cpuLongQDP.data(1), cpuLongQDP.data(2), cpuLongQDP.data(3)};
  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, 0, nullptr, true);

  // Reorder gauge fields to MILC order
  cpuFatMILC = cpuFatQDP;
  if (dslash_type == QUDA_ASQTAD_DSLASH) cpuLongMILC = cpuLongQDP;

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  // This needs to be called before `loadFatLongGaugeQuda` because this routine also loads the
  // gauge fields with different parameters.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);
    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  // freeGaugeQuda();

  loadFatLongGaugeQuda(cpuFatMILC.data(), cpuLongMILC.data(), gauge_param);

  // now copy back to QDP aliases, since these are used for the reference dslash
  // cpuFatQDP = cpuFatMILC;
  // cpuFatQDP.exchangeGhost();
  // cpuFatMILC.exchangeGhost();
  // if (dslash_type == QUDA_ASQTAD_DSLASH) {
  //   cpuLongQDP = cpuLongMILC;
  //   cpuLongQDP.exchangeGhost();
  //   cpuLongMILC.exchangeGhost();
  // }

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------
}
void cleanup()
{
  cpuFatQDP = {};
  cpuLongQDP = {};
  cpuFatMILC = {};
  cpuLongMILC = {};
}
int main(int argc, char **argv)
{
  // ::testing::InitGoogleTest(&argc, argv);
  // setQudaStaggeredDefaultInvTestParams();
  // setQudaDefaultMgTestParams();
  // Parse command line options
  auto app = make_app();
  add_su3_option_group(app);
  add_eigen_option_group(app);
  add_adj_hisq_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  add_comms_option_group(app);
  add_meas_io_group(app);
  app->add_option("--legacy-test-info", print_legacy_info,
                  "Print info on how to reproduce the old '--test #' behavior with flags, then exit");
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setVerbosity(verbosity);
  check_naik(eps_naik, n_naiks);

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  if (inv_deflate && inv_multigrid)
    errorQuda("Error: Cannot use both deflation and multigrid preconditioners on top level solve");

  initRand();

  // Only these fermions are supported in this file
  if constexpr (is_enabled_laplace()) {
    if (!is_staggered(dslash_type) && !is_laplace(dslash_type))
      errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  } else {
    if (is_laplace(dslash_type))
      errorQuda("The Laplace dslash is not enabled, cmake configure with -DQUDA_DIRAC_LAPLACE=ON");
    if (!is_staggered(dslash_type)) errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  }

  // Need to add support for LAPLACE MG?
  if (inv_multigrid) {
    if (!is_staggered(dslash_type)) {
      errorQuda("dslash_type %s not supported for multigrid preconditioner", get_dslash_str(dslash_type));
    }
  }

  display_test_info();

  initQuda(device_ordinal);



  init();

  // We here set all the problem parameters for all possible smearing types.
  QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
  smear_param.smear_type = gauge_smear_type;
  smear_param.n_steps = gauge_smear_steps;
  smear_param.adj_n_save = gauge_n_save;
  smear_param.hier_threshold = hier_threshold;
  smear_param.meas_interval = measurement_interval;
  smear_param.alpha = gauge_smear_alpha;
  smear_param.rho = gauge_smear_rho;
  smear_param.epsilon = gauge_smear_epsilon;
  smear_param.alpha1 = gauge_smear_alpha1;
  smear_param.alpha2 = gauge_smear_alpha2;
  smear_param.alpha3 = gauge_smear_alpha3;
  smear_param.dir_ignore = gauge_smear_dir_ignore;


    // params related to split grid.
  for (int i = 0; i < 4; i++) inv_param.split_grid[i] = grid_partition[i];
  int num_sub_partition = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
  use_split_grid = num_sub_partition > 1;
  use_multi_src = use_split_grid || (Nsrc_tile > 1);

if (Nsrc > QUDA_MAX_MULTI_SRC)
    errorQuda("Nsrc = %d which is great than QUDA_MAX_MULTI_SRC = %d\n", Nsrc, QUDA_MAX_MULTI_SRC);

  inv_param.num_src = Nsrc_tile;
  inv_param.num_src_per_sub_partition = Nsrc_tile / num_sub_partition;

  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &inv_param, &gauge_param);
    
  std::vector<quda::ColorSpinorField> in_raw(Nsrc,cs_param);
  std::vector<quda::ColorSpinorField> in(Nsrc,cs_param);
  std::vector<quda::ColorSpinorField> out(Nsrc,cs_param);
  std::vector<quda::ColorSpinorField> out_flowed(Nsrc,cs_param);


  std::vector<void *> in_raw_ptr(Nsrc);
  std::vector<void *> in_ptr(Nsrc);
  std::vector<void *> out_ptr(Nsrc);
  std::vector<void *> out_flowed_ptr(Nsrc);

    auto meas_int_vec = read_meas_int_vec();
    QudaFermMeasurements ferm_meas = newQudaFermMeasurements();
    ferm_meas.take_meas = QUDA_BOOLEAN_TRUE;
    ferm_meas.take_fwd_gflow = (QudaBoolean) take_fwd_gflow;
    ferm_meas.meas_int = measurement_interval;
    ferm_meas.meas_int_vec = (void *) &meas_int_vec;
    std::vector<std::vector<std::complex<double>>> ppb;
    std::vector<std::vector<std::vector<std::complex<double>>>> ppb_t;
    std::vector<std::vector<std::vector<std::complex<double>>>> pion_corr;
    void* ptr_ppb = &ppb;
    void** data_ppb = &ptr_ppb;
    ferm_meas.ppb = data_ppb;
    void* data_ppb_t = &ppb_t;
    ferm_meas.ppb_t = data_ppb_t;
    void* data_pion_corr = &pion_corr;
    ferm_meas.pion_corr = data_pion_corr;
    std::vector<int> meas_list;
    ferm_meas.meas_list = (void *) &meas_list;

    printfQuda("At start ppb has %li elements\n",ppb.size());
    printfQuda("At start ppb_t has %li elements\n",ppb_t.size());
    //simulates what user might do from external library

// Prepare rng, fill host spinors with random numbers
  //-----------------------------------------------------------------------------------

  std::vector<double> time(Nsrc);
  std::vector<double> gflops(Nsrc);
  std::vector<int> iter(Nsrc);

  for (int n = 0; n < Nsrc; n++) {
    // Populate the host spinor with random numbers.
    quda::spinorNoise(in_raw[n], n + start_seed, QUDA_NOISE_GAUSS);
    in_raw_ptr[n] = in_raw[n].data();
    in_ptr[n] = in[n].data();
    out_ptr[n] = out[n].data();
    out_flowed_ptr[n] = out_flowed[n].data();
  }
  quda::host_timer_t host_timer;
  host_timer.start();
  if (take_adj_gflow){
  performAdjGFlowHier(in_ptr.data(),in_raw_ptr.data(), &inv_param, &smear_param, &ferm_meas, Nsrc);
  }
  else {
  computeFlowedForwardPpb(in_ptr.data(),in_raw_ptr.data(), &inv_param, &smear_param, &ferm_meas, Nsrc);
  }
  host_timer.stop();
  printfQuda("At end ppb has %li elements\n",ppb.size());
  printfQuda("At end ppb_t has %li elements\n",ppb_t.size());
  printfQuda("Time elapsed for entire calculation procedure = %g secs\n", host_timer.last());
  printfQuda("Done: %d sub-partitions - %i total iter / %g secs = %g Gflops, %g secs per source\n", num_sub_partition,
                 inv_param.iter, inv_param.secs, inv_param.gflops / inv_param.secs, inv_param.secs / Nsrc_tile);
      if (inv_param.energy > 0) {
        printfQuda("Energy = %g J (%g J per source), Mean power = %g W, mean temp = %g C, mean clock = %f\n\n",
                   inv_param.energy, inv_param.energy / Nsrc_tile, inv_param.power, inv_param.temp, inv_param.clock);
      }

  in_raw = {};
  in = {};
  out = {};
  out_flowed = {};
  // tmp = {};
  // delete *rng;
  if ((quda::comm_rank() == 0) && (latfile.size() > 0 )){
  write_files(ferm_meas);}
  
  
  cleanup();

  // Finalize the QUDA library
  freeGaugeQuda();
  endQuda();
  finalizeComms();

  return 0;
}
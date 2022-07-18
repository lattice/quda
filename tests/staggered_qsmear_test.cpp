#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>

#include <blas_quda.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy solve_type S_dimension T_dimension "
             "Ls_dimension   dslash_type  normalization\n");
  printfQuda(
    "%6s   %6s          %d     %12s     %2s     %2s         %10s %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
    get_prec_str(prec), get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type), get_recon_str(link_recon),
    get_recon_str(link_recon_sloppy), get_solve_str(solve_type), xdim, ydim, zdim, tdim, Lsdim,
    get_dslash_str(dslash_type), get_mass_normalization_str(normalization));

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

int getLinkPadding_(const int dim[4])
{
  int padding = std::max(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
  padding = std::max(padding, dim[0]*dim[1]*dim[3]/2);
  padding = std::max(padding, dim[0]*dim[1]*dim[2]/2);
  return padding;
}

void initExtendedField(void* sitelink_ex[4], void* sitelink[4]) 
{	
  int X1=Z[0];
  int X2=Z[1];
  int X3=Z[2];
  int X4=Z[3];

  for(int i=0; i < V_ex; i++){
    int sid = i;
    int oddBit=0;
    if(i >= Vh_ex){
      sid = i - Vh_ex;
      oddBit = 1;
    }

    int za = sid/E1h;
    int x1h = sid - za*E1h;
    int zb = za/E2;
    int x2 = za - zb*E2;
    int x4 = zb/E3;
    int x3 = zb - x4*E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;


    if( x1< 2 || x1 >= X1 +2
        || x2< 2 || x2 >= X2 +2
        || x3< 2 || x3 >= X3 +2
        || x4< 2 || x4 >= X4 +2){

      continue;

    }

    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;

    int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
    if(oddBit){
      idx += Vh;
    }
    for(int dir= 0; dir < 4; dir++){
      char* src = (char*)sitelink[dir];
      char* dst = (char*)sitelink_ex[dir];
      memcpy(dst + i * gauge_site_size * host_gauge_data_type_size,
             src + idx * gauge_site_size * host_gauge_data_type_size, gauge_site_size * host_gauge_data_type_size);
    }//dir
  }//i
  return;
}


int main(int argc, char **argv)
{
  //setQudaDefaultMgTestParams();
  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  add_comms_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"full_ee_prec", 1}, {"full_oo_prec", 2}, {"even", 3},
                                          {"odd", 4},  {"mcg_even", 5},     {"mcg_odd", 6}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setVerbosity(verbosity);
  if (!inv_multigrid) solve_type = QUDA_INVALID_SOLVE;

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  initRand();

  // Only these fermions are supported in this file. Ensure a reasonable default,
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    printfQuda("dslash_type %s not supported, defaulting to %s\n", get_dslash_str(dslash_type),
               get_dslash_str(QUDA_ASQTAD_DSLASH));
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  // Deduce operator, solution, and operator preconditioning types
  setQudaStaggeredInvTestParams();

  display_test_info();

  // Set QUDA internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param  = newQudaInvertParam();
  
  setStaggeredGaugeParam(gauge_param);
  if (!inv_multigrid) setStaggeredInvertParam(inv_param);

  // params related to split grid.
  inv_param.split_grid[0] = grid_partition[0];
  inv_param.split_grid[1] = grid_partition[1];
  inv_param.split_grid[2] = grid_partition[2];
  inv_param.split_grid[3] = grid_partition[3];

  int num_sub_partition = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
  bool use_split_grid = num_sub_partition > 1;

  inv_param.eig_param = nullptr;

  // This must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device_ordinal);

  setDims(gauge_param.X);
  // Hack: use the domain wall dimensions so we may use the 5th dim for multi indexing
  dw_setDims(gauge_param.X, 1);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  void* qdp_inlink[4]    = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_twolnk[4]    = {nullptr,nullptr,nullptr,nullptr};  
  void* qdp_inlink_ex[4] = {nullptr,nullptr,nullptr,nullptr};
  
  void *milc_inlink       = nullptr;
  void *milc_twolnk       = nullptr;
  //
  GaugeField *cpuTwoLink  = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir]    = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_twolnk[dir]    = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_inlink_ex[dir] = safe_malloc(V_ex * gauge_site_size * host_gauge_data_type_size);    
  }
  //
  milc_inlink = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  milc_twolnk = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  // For load, etc
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  constructHostGaugeField(qdp_inlink, gauge_param, argc, argv);
  initExtendedField(qdp_inlink_ex, qdp_inlink);

  // Reorder gauge fields to MILC order
  reorderQDPtoMILC(milc_inlink, qdp_inlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  // This needs to be called before `loadFatLongGaugeQuda` because this routine also loads the
  // gauge fields with different parameters.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // Specific gauge parameters for MILC
  int pad_size = 0;
//#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
//#endif
  int link_pad = 3 * pad_size;

  gauge_param.reconstruct                   = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_sloppy            = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  gauge_param.ga_pad = link_pad;
  gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  //
  loadGaugeQuda(milc_inlink, &gauge_param);
  
  // Compute two link
  computeTwoLinkCPU(qdp_twolnk, qdp_inlink_ex, &gauge_param);

  // Create ghost gauge fields in case of multi GPU builds.
  reorderQDPtoMILC(milc_twolnk, qdp_twolnk, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  
  gauge_param.type =  QUDA_ASQTAD_LONG_LINKS;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuTwoLinkParam(gauge_param, milc_twolnk);
  cpuTwoLinkParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuTwoLink = GaugeField::Create(cpuTwoLinkParam);

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------

  // Staggered vector construct START
  //-----------------------------------------------------------------------------------
  std::vector<quda::ColorSpinorField *> spinor;
  std::vector<quda::ColorSpinorField *> ref_spinor;
  std::vector<quda::ColorSpinorField *> tmp;
  std::vector<quda::ColorSpinorField *> *tmp2;
  //
  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &inv_param, &gauge_param);

  for (int k = 0; k < Nsrc; k++) {
    spinor.emplace_back(quda::ColorSpinorField::Create(cs_param));
    tmp.emplace_back(quda::ColorSpinorField::Create(cs_param));
    ref_spinor.emplace_back(quda::ColorSpinorField::Create(cs_param));
    tmp2.emplace_back(quda::ColorSpinorField::Create(cs_param));
  }
  // Staggered vector construct END
  //-----------------------------------------------------------------------------------

  // Prepare rng
  auto *rng = new quda::RNG(*tmp2, 1234);

  // Performance measuring
  std::vector<double> time(Nsrc);
  std::vector<double> gflops(Nsrc);
  std::vector<int> iter(Nsrc);

  // QUDA invert test
  //----------------------------------------------------------------------------

  for (int k = 0; k < Nsrc; k++) { 
    quda::spinorNoise(*spinor[k], *rng, QUDA_NOISE_UNIFORM); 
    *tmp[k] = *spinor[k];
  }

  // smearing parameters
  double omega = 2.0;
  int n_steps  = 50;
  double smear_coeff = -1.0 * omega * omega / ( 4*n_steps );

  const int compute_2link = 1;
  const int delete_2link  = 0;
  const int t0            = -1;
  
  for (int k = 0; k < Nsrc; k++) {
    performTwoLinkGaussianSmearNStep(spinor[k]->V(), &inv_param, n_steps, smear_coeff, compute_2link, t0);
  }

  if (verify_results)
  {
    const double ftmp    = -(smear_coeff*smear_coeff)/(4.0*n_steps*4.0);
    const double msq     = 1. / ftmp;
    const double a       = inv_param.laplace3D * 2.0 + msq;

    for(int k = 0; k < Nsrc; k++){
      for (int i = 0; i < n_steps; i++) {
        if (i > 0) std::swap(*tmp[k], *ref_spinor[k]);
      
        quda::blas::ax(ftmp, *tmp[k]);
        quda::blas::axpy(a, *tmp[k], *tmp2[k]);
      
        staggeredTwoLinkGaussianSmear(ref_spinor[k]->Even(), qdp_twolnk, (void **)cpuTwoLink->Ghost(),  tmp[k]->Even(), &gauge_param, &inv_param, 0, smear_coeff, t0, gauge_param.cpu_prec);
        staggeredTwoLinkGaussianSmear(ref_spinor[k]->Odd(), qdp_twolnk,  (void **)cpuTwoLink->Ghost(),  tmp[k]->Odd(),  &gauge_param, &inv_param, 1, smear_coeff, t0, gauge_param.cpu_prec);
        blas::xpay(*tmp2[k], -1.0, *ref_spinor[k]);
        blas::zero(*tmp2[k]);
      }
    }
  }

  // Free RNG
  delete rng;

  // Clean up gauge fields
  for (int dir = 0; dir < 4; dir++) {
    host_free(qdp_inlink[dir]);
    host_free(qdp_inlink_ex[dir]);
    host_free(qdp_twolnk[dir]);
  }
  
  host_free(milc_inlink);
  host_free(milc_twolnk);

  if (cpuTwoLink != nullptr) { delete cpuTwoLink; cpuTwoLink = nullptr; }

  for (auto spinor_vec : spinor) { delete spinor_vec; }
  for (auto ref_vec : ref_spinor) { delete ref_vec; }
  for (auto tmp_vec : tmp ) { delete tmp_vec; }
  for (auto tmp_vec2: tmp2) { delete tmp_vec2; }

  // Finalize the QUDA library
  endQuda();

  // Finalize the communications layer
  finalizeComms();

  return 0;
}

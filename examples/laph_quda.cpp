//C++ headers 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//QUDA headers
#include <quda.h>
#include <gauge_field.h>
#include <comm_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <gauge_tools.h>
#include <misc.h>
#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

void constructGaugeField(QudaGaugeParam &gauge_param, cudaGaugeField *gaugeEx,
			 cudaGaugeField *gauge, RNG *randstates) {
  
  if (strcmp(latfile, "")) { // We loaded in a gauge field
    // copy internal extended field to gaugeEx
    copyExtendedResidentGaugeQuda((void*)gaugeEx);
  } else {
    if (heatbath_coldstart) InitGaugeField(*gaugeEx);
    else InitGaugeField(*gaugeEx, *randstates);
    
    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);
    
    // load the gauge field from gauge
    gauge_param.gauge_order = gauge->Order();
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;
    
    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
  }
}

void display_info()
{
  printfQuda("running the following simulation:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);
  
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

int main(int argc, char **argv)
{
  // command line options
  auto app = make_app();
  add_eigen_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Initialise QUDA
  //----------------------------------------------------------------------------
  // Set values for precisions via the command line.
  setQudaPrecisions();
  
  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // initialize the QUDA library
  initQuda(device_ordinal);
  
  // call srand() with a rank-dependent seed
  initRand();

  // Set verbosity
  setVerbosity(verbosity);

  // Set QUDA's internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Set the dimensions
  setDims(gauge_param.X);  

  // Allocate space on the host
  void *load_gauge[4];
  for (int dir = 0; dir < 4; dir++) { load_gauge[dir] = malloc(V * gauge_site_size * gauge_param.cpu_prec); }
  constructHostGaugeField(load_gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)load_gauge, &gauge_param);
  
  // All user inputs now defined
  display_info();
  //----------------------------------------------------------------------------

  // Construct an extended device gauge field
  //--------------------------------------------------------------------------
  //using namespace quda;
  GaugeFieldParam gParam(gauge_param);
  gParam.pad = 0;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = gauge_param.type;
  gParam.reconstruct = gauge_param.reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  cudaGaugeField *gauge = new cudaGaugeField(gParam);
  
  int pad = 0;
  int y[4];
  int R[4] = {0,0,0,0};
  for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
  for(int dir=0; dir<4; ++dir) y[dir] = gauge_param.X[dir] + 2 * R[dir];
  GaugeFieldParam gParamEx(y, prec, link_recon,
			   pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.order = gParam.order;
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = gParam.t_boundary;
  gParamEx.nFace = 1;
  for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
  cudaGaugeField *gaugeEx = new cudaGaugeField(gParamEx);
  
  // CURAND random generator initialization
  RNG *randstates = new RNG(*gauge, 1234);
  
  constructGaugeField(gauge_param, gaugeEx, gauge, randstates);
  //--------------------------------------------------------------------------
    
  // Plaquette and Q charge measurement
  //--------------------------------------------------------------------------
  // start the timer
  double time0 = -((double)clock());
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_plaquette = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;
  
  // Run the QUDA computation
  gaugeObservablesQuda(&param);
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  
  printfQuda("Computed plaquette is %.16e (spatial = %.16e, temporal = %.16e)\n", param.plaquette[0], param.plaquette[1], param.plaquette[2]);
  printfQuda("Computed q charge = %.16e\n", param.qcharge);
  //--------------------------------------------------------------------------

  time0 = -((double)clock());
  // (s)LapH set up
  //--------------------------------------------------------------------------
  // Only the Laplace3D dslash type is applicable. 
  if (dslash_type != QUDA_LAPLACE_DSLASH || laplace3D != 3) {
    printfQuda("dslash_type %s (ortho dim %d) not supported, defaulting to %s with orthodim 3\n", get_dslash_str(dslash_type), laplace3D, get_dslash_str(QUDA_LAPLACE_DSLASH));
    dslash_type = QUDA_LAPLACE_DSLASH;
    laplace3D = 3;
  }
  
  // Create eigenvector parameters for the Laplace operator
  setQudaStaggeredEigTestParams();
  
  // Set QUDA internal parameters
  // Though no inversions are performed, the inv_param
  // structure contains all the information we need to
  // construct the dirac operator. We encapsualte the
  // inv_param structure inside the eig_param structure
  // to avoid any confusion
  QudaInvertParam eig_inv_param = newQudaInvertParam();
  setStaggeredInvertParam(eig_inv_param);
  QudaEigParam eig_param = newQudaEigParam();
  setEigParam(eig_param);
  
  // We encapsulate the eigensolver parameters inside the invert parameter structure
  eig_param.invert_param = &eig_inv_param;

  // No ARPACK check for single prec solves
  if (eig_param.arpack_check && !(prec == QUDA_DOUBLE_PRECISION)) {
    errorQuda("ARPACK check only available in double precision");
  }
  
  // Create inverter parameters
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];
  if (inv_multigrid) {    
    setQudaMgSolveTypes();
    setMultigridInvertParam(inv_param);
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
    setMultigridParam(mg_param);
  } else {
    setInvertParam(inv_param);
  }
  inv_param.eig_param = nullptr;  
  //--------------------------------------------------------------------------

  // Vector construct START
  //----------------------------------------------------------------------------------
  // Allocate host side memory for the eigenspace
  void **host_evecs = (void **)safe_malloc(eig_n_conv * sizeof(void *));
  for (int i = 0; i < eig_param.n_conv; i++) {
    host_evecs[i] = (void *)safe_malloc(V * stag_spinor_site_size * eig_inv_param.cpu_prec);
  }
  int n_evals = eig_param.n_conv;
  if(eig_param.eig_type == QUDA_EIG_TR_LANCZOS_3D) n_evals *= tdim;  
  double _Complex *host_evals = (double _Complex *)safe_malloc(n_evals * sizeof(double _Complex));

  // Allocate host side memory for the inverter
  std::vector<quda::ColorSpinorField *> in(Nsrc);
  std::vector<quda::ColorSpinorField *> out(Nsrc);
  quda::ColorSpinorParam cs_param;
  constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
  
  for (int i = 0; i < Nsrc; i++) {
    in[i] = quda::ColorSpinorField::Create(cs_param);
    out[i] = quda::ColorSpinorField::Create(cs_param);
  }
  // Vector construct END
  //-----------------------------------------------------------------------------------

  // QUDA (s)LapH test BEGIN
  //----------------------------------------------------------------------------
  //
  // QUDA Laplace 3D eigensolver
  // The 3D eigensolver is a Thick Restarted Lanczos that uses the 3D Laplace
  // operator with the temporal dimension omitted. The 4D operator may be
  // applied to all T independent eigensystems stacked in a 4D Krylov space.
  // During the Lanczos step, the 4D vectors are broken into their respective
  // 3D spaces and the necessary linear algebra is perfromed. Then, the T Ritz
  // rotations are performed on each subsystem.
  // The results are stored in 4D ColorSpinorField arrays. Each t component
  // of the n^th 4D vector corresponds to the n^th eigevector of the t^th
  // eigensystem.
  double time = 0.0;
  time = -((double)clock());
  eigensolveQuda(host_evecs, host_evals, &eig_param);
  time += (double)clock();
  printfQuda("Time for %s solution = %f\n", eig_param.arpack_check ? "ARPACK" : "QUDA", time / CLOCKS_PER_SEC);
  
  // Compute Perambulators
  // 1. Sources
  // Sources are constructed from the above eigenvectors. A source emanates
  // from a timeslice t, and is composed of an n^th eigenvector (colour vector)
  // placed in the s^th spin position. All eigenvectors in the non-stochastic
  // variant are used to compute perambulators. in sLapH, one takes a stochastic
  // subset of the eigenvectors to produce sources. This introduces an extra
  // loop over a dilution index, but reduces the number of eigenvectors one
  // needs to employ.
  //----------------------------------------------------------------------------
  int t_sources = 1;
  int n_spin = 4;
  for(int t=0; t<t_sources; t++) {
    int source_t = 0;
    printfQuda("Source time t=%d\n", source_t);
    for(int s=0; s<n_spin; s++) {
      printfQuda("Source spin s=%d\n", s);
      
      // Construct the source for this iteration
      for(int n=0; n<eig_param.n_conv; n++) {
	createLAPHsource(in[0]->V(), host_evecs, source_t, s, n);
	invertQuda(out[0]->V(), in[0]->V(), &inv_param);
      }
    }
  }  
  //----------------------------------------------------------------------------
  
  // Deallocate host memory
  for (int i = 0; i < eig_n_conv; i++) host_free(host_evecs[i]);
  host_free(host_evecs);
  host_free(host_evals);
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("sLapH complete, total time = %g secs\n", time0);
   
  //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
  PGaugeExchangeFree();
  
  delete gauge;
  delete gaugeEx;
  delete randstates;
  for (int dir = 0; dir<4; dir++) free(load_gauge[dir]);
  //--------------------------------------------------------------------------
  
  // Finalize the QUDA library
  endQuda();
  finalizeComms();
  
  return 0;
}


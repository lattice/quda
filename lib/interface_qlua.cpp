/* Collection of functions required for applying         
 * the laplacian / performing WUppertal smearing.
 * The top level function will be called from a 
 * qlua C-interface.
 * October 2017
 */

#include <quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <comm_quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <interface_qlua_internal.h>
#include <mpi.h>
#include <blas_cublas.h>


using namespace quda;


QudaVerbosity parseVerbosity(const char *v){

  QudaVerbosity verbosity = QUDA_INVALID_VERBOSITY;
  
  if      (strcmp(v,"QUDA_SILENT")==0)        verbosity = QUDA_SILENT;
  else if (strcmp(v,"QUDA_SUMMARIZE")==0)     verbosity = QUDA_SUMMARIZE;
  else if (strcmp(v,"QUDA_VERBOSE")==0)       verbosity = QUDA_VERBOSE;
  else if (strcmp(v,"QUDA_DEBUG_VERBOSE")==0) verbosity = QUDA_DEBUG_VERBOSE;
  else if (strcmp(v,"INVALID_VERBOSITY")==0){
    printfQuda("Verbosity not set! Will set to QUDA_SUMMARIZE\n");
    verbosity = QUDA_SUMMARIZE;
  }
  else{
    printfQuda("Verbosity not set correctly (got \"%s\")! Will set to QUDA_SUMMARIZE\n",v);
    verbosity = QUDA_SUMMARIZE;
  }

  return verbosity;
}

qudaAPI_ContractId parseContractIdx(const char *v){
  
  qudaAPI_ContractId cId;
  
  if      (strcmp(v,"contract12")==0) cId = cntr12;
  else if (strcmp(v,"contract13")==0) cId = cntr13;
  else if (strcmp(v,"contract14")==0) cId = cntr14;
  else if (strcmp(v,"contract23")==0) cId = cntr23;
  else if (strcmp(v,"contract24")==0) cId = cntr24;
  else if (strcmp(v,"contract34")==0) cId = cntr34;
  else cId = cntr_INVALID;
  
  return cId;
}
  

/* topology in Quda is a global variable;
 * need to check for every lattice if the topology is the same */
static int
check_quda_comms(const qudaLattice *qS)
{
  int eq = 1;
  const Topology *qtopo = comm_default_topology();
  eq = eq && (comm_ndim(qtopo) == qS->rank);
  /* cannot check my_rank : no read method;
   * node coords are enough */
  if (!eq)
    return 1;
  for (int i = 0 ; i < qS->rank ; i++) {
    eq = eq && (comm_dims(qtopo)[i] == qS->net[i])
      && (comm_coords(qtopo)[i] == qS->net_coord[i]);
  }
  return (!eq);
}

//-- fill out QudaInvertParam
static void
init_QudaInvertParam_generic(QudaInvertParam& ip,
                             const QudaGaugeParam& gp, qudaAPI_Param paramAPI)
{

  ip  = newQudaInvertParam();

  ip.dslash_type              = QUDA_CLOVER_WILSON_DSLASH;
  ip.clover_cpu_prec          = QUDA_DOUBLE_PRECISION;
  ip.clover_cuda_prec         = QUDA_DOUBLE_PRECISION;
  ip.clover_cuda_prec_sloppy  = QUDA_HALF_PRECISION;
  ip.clover_order             = QUDA_PACKED_CLOVER_ORDER;
  ip.cpu_prec                 = QUDA_DOUBLE_PRECISION;
  ip.cuda_prec                = QUDA_DOUBLE_PRECISION;
  ip.cuda_prec_sloppy         = QUDA_HALF_PRECISION;
  ip.dagger                   = QUDA_DAG_NO;
  ip.dirac_order              = QUDA_QDP_DIRAC_ORDER;
  ip.gamma_basis              = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  ip.inv_type                 = QUDA_BICGSTAB_INVERTER;
  ip.mass_normalization       = QUDA_KAPPA_NORMALIZATION;
  ip.matpc_type               = QUDA_MATPC_EVEN_EVEN;
  ip.preserve_source          = QUDA_PRESERVE_SOURCE_NO;
  ip.reliable_delta           = 0.1;
  ip.solution_type            = QUDA_MAT_SOLUTION;
  ip.solve_type               = QUDA_DIRECT_PC_SOLVE;
  ip.sp_pad                   = gp.ga_pad;
  ip.cl_pad                   = gp.ga_pad;
  ip.tune                     = QUDA_TUNE_NO;
  ip.use_init_guess           = QUDA_USE_INIT_GUESS_NO;
  ip.verbosity                = paramAPI.verbosity;

  //-- FIXME: Need to change these
  ip.kappa                    = paramAPI.wParam.alpha[0];
  ip.clover_coeff             = paramAPI.wParam.alpha[1];
  ip.tol                      = paramAPI.wParam.alpha[2];
  ip.maxiter                  = int(paramAPI.wParam.alpha[3]);
}

//-- fill out QudaGaugeParam
static void
init_QudaGaugeParam_generic(QudaGaugeParam& gp, const qudaLattice *qS)
{
  gp = newQudaGaugeParam();
  
  gp.type               = QUDA_WILSON_LINKS;
  gp.gauge_order        = QUDA_QDP_GAUGE_ORDER;
  gp.gauge_fix          = QUDA_GAUGE_FIXED_NO;
  gp.cpu_prec           = QUDA_DOUBLE_PRECISION;
  gp.cuda_prec          = QUDA_DOUBLE_PRECISION;
  gp.reconstruct        = QUDA_RECONSTRUCT_NO;
  gp.cuda_prec_sloppy   = QUDA_HALF_PRECISION;
  gp.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  gp.anisotropy         = 1.0;
  gp.t_boundary         = QUDA_ANTI_PERIODIC_T;
  
  LONG_T max_face = 0;
  for (int mu = 0 ; mu < qS->rank ; mu++) {
    int locsize = qS->site_coord_hi[mu] - qS->site_coord_lo[mu];
    gp.X[mu] = locsize;
    LONG_T face = qS->locvol / (LONG_T) locsize;
    if (max_face < face)
      max_face = face;
  }
  gp.ga_pad            = max_face;
}


//-- load the gauge field
static GaugeField*
new_cudaGaugeField(QudaGaugeParam& gp, double *hbuf_u[])
{

  GaugeFieldParam gf_param(hbuf_u, gp);
  GaugeField *cpu_gf   = static_cast<GaugeField*>(new cpuGaugeField(gf_param));

  gf_param.create         = QUDA_NULL_FIELD_CREATE;
  gf_param.precision      = gp.cuda_prec;
  gf_param.reconstruct    = gp.reconstruct;
  gf_param.ghostExchange  = QUDA_GHOST_EXCHANGE_PAD;
  gf_param.pad            = gp.ga_pad;
  gf_param.order          = QUDA_FLOAT2_GAUGE_ORDER;

  GaugeField *cuda_gf = NULL;
  cuda_gf = new cudaGaugeField(gf_param);
  if (NULL == cuda_gf) return NULL;

  if (NULL != hbuf_u) {
    double t1 = MPI_Wtime();
    cuda_gf->copy(*cpu_gf); // C.K. This does ghost exchange as well
    double t2 = MPI_Wtime();
    printfQuda("TIMING - laplacianQuda: cuda_gf copy & exchangeGhost in %.6f sec.\n", t2-t1);
  }
  
  return cuda_gf;
}

//-- load a ColorSpinorField
static cudaColorSpinorField*
new_cudaColorSpinorField(QudaGaugeParam& gp, QudaInvertParam& ip,
			 int nColor, int nSpin,
			 QUDA_REAL *hbuf_x)
{
  ColorSpinorParam cpuParam(hbuf_x, ip, gp.X, false, QUDA_CPU_FIELD_LOCATION); // false stands for the pc_solution
  ColorSpinorParam cudaParam(cpuParam, ip, QUDA_CUDA_FIELD_LOCATION);
  
  cudaColorSpinorField *cuda_x = NULL;
  if (NULL != hbuf_x) {
    cudaParam.create = QUDA_COPY_FIELD_CREATE;
    ColorSpinorField *cpu_x = ColorSpinorField::Create(cpuParam);
    cuda_x = new cudaColorSpinorField(*cpu_x, cudaParam);
  }
  else{
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cuda_x = new cudaColorSpinorField(cudaParam);
  }
    
  return cuda_x;
}

//-- get back the resulting color spinor field
static void
save_cudaColorSpinorField(QUDA_REAL *hbuf_x,
			  QudaGaugeParam& gp, QudaInvertParam& ip,
			  int nColor, int nSpin,
			  ColorSpinorField &cuda_x)
{
  ColorSpinorParam cpuParam(hbuf_x, ip, gp.X, false); // false stands for the pc_solution
  cpuParam.nColor = nColor;  //
  cpuParam.nSpin  = nSpin;   // hack!
  cpuColorSpinorField cpu_x(cpuParam);                // cpuCSF wrapper for hbuf_x
  cpu_x = cuda_x;
}


static void createMomentaMatrix(int *momMatrix,
				int *Nmoms,
				int QsqMax,
				int totalL[])
{
  
  int imom = 0;
  int p_temp[3];
  for(int pz = 0; pz < totalL[2]; pz++){
    for(int py = 0; py < totalL[1]; py++)
      for(int px = 0; px < totalL[0]; px++){
        if(px < totalL[0]/2)
          p_temp[0] = px;
        else
          p_temp[0] = px - totalL[0];
	
        if(py < totalL[1]/2)
          p_temp[1] = py;
        else
          p_temp[1] = py - totalL[1];
	
        if(pz < totalL[2]/2)
          p_temp[2] = pz;
        else
          p_temp[2] = pz - totalL[2];
	
        if( (p_temp[0]*p_temp[0] + p_temp[1]*p_temp[1] + p_temp[2]*p_temp[2]) <= QsqMax ){
          for(int i=0;i<3;i++) momMatrix[i + 3*imom] = p_temp[i];
          imom++;
        }
      }
  }
  *Nmoms = imom;
  
}


static void createPhaseMatrix_CPU(complex<QUDA_REAL> *phaseMatrix,
				  const int *momMatrix,
				  momProjParam param,
				  int localL[], int totalL[])
{
  
  int lcoord[param.momDim];
  int gcoord[param.momDim];
  for(int iv=0;iv<param.V3;iv++){
    int a1 = iv / localL[0];
    int a2 = a1 / localL[1];
    lcoord[0] = iv - a1 * localL[0];
    lcoord[1] = a1 - a2 * localL[1];
    lcoord[2] = a2;
    gcoord[0] = lcoord[0] + comm_coord(0) * localL[0];
    gcoord[1] = lcoord[1] + comm_coord(1) * localL[1];
    gcoord[2] = lcoord[2] + comm_coord(2) * localL[2];
    
    QUDA_REAL f = (QUDA_REAL) param.expSgn;
    for(int im=0;im<param.Nmoms;im++){
      QUDA_REAL phase = 0.0;
      for(int id=0;id<param.momDim;id++)
	phase += momMatrix[id + param.momDim*im]*gcoord[id] / (QUDA_REAL)totalL[id];

      phaseMatrix[iv + param.V3*im].x =   cos(2.0*PI*phase);
      phaseMatrix[iv + param.V3*im].y = f*sin(2.0*PI*phase);
    }
  }//- iv

}


//-- top level function, calls quda-wuppertal smearing
EXTRN_C int
laplacianQuda(
	      QUDA_REAL *hv_out,
	      QUDA_REAL *hv_in,
	      QUDA_REAL *h_gauge[],
	      const qudaLattice *qS,
	      int nColor, int nSpin,
	      qudaAPI_Param paramAPI)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;
  if (QUDA_Nc != nColor)
    return 1;

  printfQuda("laplacianQuda: Will apply the Laplacian for %d steps with the parameters:\n", paramAPI.wParam.Nstep);
  for(int i=0; i< qS->rank; i++) printfQuda("  alpha[%d] = %.3f\n", i, paramAPI.wParam.alpha[i]);
  printfQuda("  beta = %.3f\n", paramAPI.wParam.beta);
  
  //-- Initialize the quda-gauge parameters
  QudaGaugeParam gp;
  init_QudaGaugeParam_generic(gp, qS);

  setVerbosity(paramAPI.verbosity);
  if(getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(&gp);

  //-- Initialize the inverter parameters
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);


  //-- Load the gauge field
  double t3 = MPI_Wtime();
  GaugeField *cuda_gf = NULL;
  cuda_gf = new_cudaGaugeField(gp, h_gauge);
  double t4 = MPI_Wtime();
  printfQuda("TIMING - laplacianQuda: cudaGaugeField loaded in %.6f sec.\n", t4-t3);

  //-- load the colorspinor fields
  double t5 = MPI_Wtime();
  cudaColorSpinorField *cuda_v_in  = NULL;
  cudaColorSpinorField *cuda_v_out = NULL;
  cuda_v_in  = new_cudaColorSpinorField(gp, ip, nColor, nSpin, hv_in);
  cuda_v_out = new_cudaColorSpinorField(gp, ip, nColor, nSpin, NULL);
  double t6 = MPI_Wtime();
  printfQuda("TIMING - laplacianQuda: cudaColorSpinorFields loaded in %.6f sec.\n", t6-t5);

  //-- Call the Wuppertal smearing Nstep times
  int parity = 0;  
  double t1 = MPI_Wtime();
  for (int i = 0 ; i < paramAPI.wParam.Nstep ; i++){
    wuppertalStep(*cuda_v_out, *cuda_v_in, parity, *cuda_gf, paramAPI.wParam.alpha, paramAPI.wParam.beta);    
    cudaDeviceSynchronize();
    checkCudaError();
    *cuda_v_in = *cuda_v_out;
  }
  double t2 = MPI_Wtime(); 
  printfQuda("TIMING - laplacianQuda: Wuppertal smearing for Nstep = %d done in %.6f sec.\n", paramAPI.wParam.Nstep, t2-t1);
 
  //-- extract
  double t7 = MPI_Wtime();
  save_cudaColorSpinorField(hv_out, gp, ip, nColor, nSpin, *cuda_v_out);
  double t8 = MPI_Wtime();
  printfQuda("TIMING - laplacianQuda: Field extraction done in %.6f sec.\n", t8-t7);
  
  //-- cleanup & return
  printfQuda("laplacianQuda: Finalizing...\n");
  delete_not_null(cuda_gf);
  delete_not_null(cuda_v_in);
  delete_not_null(cuda_v_out);

  saveTuneCache();

  printfQuda("laplacianQuda: Returning...\n");

  return status;
}


//-- top level function, performs di-quark contractions
EXTRN_C int
doQQ_contract_Quda(
	      QUDA_REAL *hprop_out,
	      QUDA_REAL *hprop_in1,
	      QUDA_REAL *hprop_in2,
	      const qudaLattice *qS,
	      int nColor, int nSpin,
	      qudaAPI_Param paramAPI)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;
  if (QUDA_Nc != nColor)
    return 1;

  //-- Check-print parameters
  if (paramAPI.cParam.cntrID == cntr_INVALID)
    errorQuda("doQQ_contract_Quda: Contract index not set correctly!\n");

  int nVec = paramAPI.cParam.nVec;  
  printfQuda("doQQ_contract_Quda: Got nVec = %d\n", nVec);
  printfQuda("doQQ_contract_Quda: Got contractID = %d\n", (int)paramAPI.cParam.cntrID);
  
  setVerbosity(paramAPI.verbosity);
  
  //-- Initialize the quda-gauge and invert parameters
  QudaGaugeParam gp;
  init_QudaGaugeParam_generic(gp, qS);

  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);

  //-- load the propagators
  LONG_T fld_lgh = qS->locvol * nColor * nSpin * 2;
  
  ColorSpinorField *cudaProp_in1[nVec];
  ColorSpinorField *cudaProp_in2[nVec];
  ColorSpinorField *cudaProp_out[nVec];
  
  double t5 = MPI_Wtime();
  for(int ivec=0;ivec<nVec;ivec++){
    cudaProp_in1[ivec] = new_cudaColorSpinorField(gp, ip, nColor, nSpin, &(hprop_in1[ivec * fld_lgh]) );
    cudaProp_in2[ivec] = new_cudaColorSpinorField(gp, ip, nColor, nSpin, &(hprop_in2[ivec * fld_lgh]) );
    cudaProp_out[ivec] = new_cudaColorSpinorField(gp, ip, nColor, nSpin, NULL );
    
    if((cudaProp_in1[ivec] == NULL) || (cudaProp_in2[ivec] == NULL) || (cudaProp_out[ivec] == NULL))
      errorQuda("doQQ_contract_Quda: Cannot allocate propagators. Exiting.\n");
    checkCudaError();
  }
  double t6 = MPI_Wtime();
  printfQuda("TIMING - doQQ_contract_Quda: Propagators loaded in %.6f sec.\n", t6-t5);
  
  //-- Call contractions kernel here
  int parity = 0;
  double t1 = MPI_Wtime();
  cudaContractQQ(cudaProp_out, cudaProp_in1, cudaProp_in2, parity, nColor, nSpin, paramAPI.cParam);
  checkCudaError();
  double t2 = MPI_Wtime();
  printfQuda("TIMING - doQQ_contract_Quda: Contractions in %.6f sec.\n", t2-t1);
  
  //-- extract
  double t7 = MPI_Wtime();
  for(int ivec=0;ivec<nVec;ivec++){
    save_cudaColorSpinorField(&(hprop_out[ivec * fld_lgh]), gp, ip, nColor, nSpin, *cudaProp_out[ivec]);
    checkCudaError();
  }
  double t8 = MPI_Wtime();
  printfQuda("TIMING - doQQ_contract_Quda: Propagator extraction done in %.6f sec.\n", t8-t7);
  
  //-- cleanup & return
  printfQuda("doQQ_contract_Quda: Finalizing...\n");
  for(int ivec=0;ivec<nVec;ivec++){
    delete_not_null(cudaProp_in1[ivec]);
    delete_not_null(cudaProp_in2[ivec]);
    delete_not_null(cudaProp_out[ivec]);
  }
  
  saveTuneCache();

  printfQuda("doQQ_contract_Quda: Returning...\n");

  return status;
}

//-- top level function, calls invertQuda
//-- Here, wParam holds inverter parameters
EXTRN_C int
Qlua_invertQuda(
		QUDA_REAL *hv_out,
		QUDA_REAL *hv_in,
		QUDA_REAL *h_gauge[],
		const qudaLattice *qS,
		int nColor, int nSpin,
		qudaAPI_Param paramAPI)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;
  if (QUDA_Nc != nColor)
    return 1;

  printfQuda("Qlua_invertQuda: Will perform inversion with the parameters:\n");
  printfQuda("kappa   = %lf\n", paramAPI.wParam.alpha[0]);
  printfQuda("Csw     = %lf\n", paramAPI.wParam.alpha[1]);
  printfQuda("tol     = %e\n",  paramAPI.wParam.alpha[2]);
  printfQuda("Maxiter = %d\n",  int(paramAPI.wParam.alpha[3]));

  //-- Initialize the quda-gauge parameters
  QudaGaugeParam gp;
  init_QudaGaugeParam_generic(gp, qS);

  setVerbosity(paramAPI.verbosity);
  if(getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(&gp);

  //-- Initialize the inverter parameters
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);

  //-- Inversion
  double x1 = MPI_Wtime();
  loadGaugeQuda(h_gauge, &gp);
  loadCloverQuda(NULL, NULL, &ip);
  double x2 = MPI_Wtime();
  printfQuda("TIMING - Qlua_invertQuda: loadGaugeQuda-loadCloverQuda in %.6f sec.\n", x2-x1);

  double x5 = MPI_Wtime();
  invertQuda(hv_out, hv_in, &ip);
  double x6 = MPI_Wtime();
  printfQuda("TIMING - Qlua_invertQuda: invertQuda in %.6f sec.\n", x6-x5);

  freeCloverQuda();
  freeGaugeQuda();

  saveTuneCache();

  printfQuda("Qlua_invertQuda: Returning...\n");

  return status;
}



//-- top level function, performs momentum projection
EXTRN_C int
momentumProjectionPropagator_Quda(
				  QUDA_REAL *corrOut,
				  QUDA_REAL *corrIn,
				  const qudaLattice *qS,
				  qudaAPI_Param paramAPI)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;

  //-- This is needed to avoid segfaults, propagator must hold the full 3d-volume
  if( comm_size() != comm_dim(3) )
    errorQuda("momentumProjectionPropagator_Quda: This function supports only T-direction partitioning!\n");

  
  int Nc = QUDA_Nc;
  int Ns = QUDA_Ns;

  //-- Check-print parameters
  int QsqMax = paramAPI.mpParam.QsqMax;
  int Ndata  = paramAPI.mpParam.Ndata;
  int expSgn = paramAPI.mpParam.expSgn;
  bool GPU_phaseMatrix = (paramAPI.mpParam.GPU_phaseMatrix == 1 ? true : false);
  if(expSgn != 1 && expSgn != -1)
    errorQuda("momentumProjectionPropagator_Quda: Got invalid exponential sign, expSgn = %d!\n",expSgn);

  printfQuda("momentumProjectionPropagator_Quda: Got QsqMax = %d\n", QsqMax);
  printfQuda("momentumProjectionPropagator_Quda: Got Ndata  = %d\n", Ndata);
  printfQuda("momentumProjectionPropagator_Quda: Got exponential sign %s\n", expSgn == 1 ? "PLUS" : "MINUS");
  printfQuda("momentumProjectionPropagator_Quda: Will create phase matrix on %s\n", GPU_phaseMatrix == true ? "GPU" : "CPU");
  
  setVerbosity(paramAPI.verbosity);
  
  //-- Initialize the quda-gauge and invert parameters
  QudaGaugeParam gp;
  init_QudaGaugeParam_generic(gp, qS);

  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);


  //-- Define useful topology quantities
  int nDim = qS->rank;
  int momDim = nDim - 1;
  int localL[nDim];
  int totalL[nDim];
  for(int mu=0; mu<nDim; mu++){
    localL[mu] = qS->site_coord_hi[mu] - qS->site_coord_lo[mu];
    totalL[mu] = localL[mu] * comm_dim(mu);
  }

  int Lt = localL[QUDA_TIME_AXIS];
  int totT = totalL[QUDA_TIME_AXIS];
  
  LONG_T totV3 = 1;
  LONG_T V3 = 1;
  for(int i=0;i<momDim;i++){
    totV3 *= totalL[i];
    V3    *= localL[i];
  }
    
  paramAPI.mpParam.momDim = momDim;
  paramAPI.mpParam.V3     = V3;
  printfQuda("momentumProjectionPropagator_Quda: totV3 = %lld , V3 = %lld\n", totV3, V3);
  //------------------------------------

  
  //-- Define the momenta matrix
  int QsqMaxLat = 0;
  for(int mu=0;mu<momDim;mu++)
    QsqMaxLat += pow(0.5*totalL[mu],2);
  if(QsqMax > QsqMaxLat)
    errorQuda("momentumProjectionPropagator_Quda: QsqMax = %d requested is greater than Maximum allowed for the lattice, QsqMaxLat = %d\n", QsqMax, QsqMaxLat);

  
  //-- Add this additional check, we want the full spectrum of momenta!
  if(QsqMax != QsqMaxLat)
    errorQuda("momentumProjectionPropagator_Quda: This function supports only the maximum Qsq allowed, QsqMaxLat = %d\n", QsqMaxLat);

  
  int Nmoms = 0;
  int *momMatrix = NULL;

  momMatrix = (int*) calloc(momDim*totV3, sizeof(int));
  if(momMatrix == NULL)
    errorQuda("momentumProjectionPropagator_Quda: Cannot allocate momMatrix. Exiting.\n");
  
  /* When this function returns, momMatrix contains
   * Nmoms pairs of momenta, giving Qsq <= QsqMax
   * If QsqMax is less than maximum allowed by lattice size,
   * the rest (totV3 - Nmoms) pair-entries of momMatrix contain zeros. 
   */
  createMomentaMatrix(momMatrix, &Nmoms, QsqMax, totalL);
  printfQuda("momentumProjectionPropagator_Quda: Momenta matrix created, Nmoms = %d\n", Nmoms);
  paramAPI.mpParam.Nmoms = Nmoms;
  //------------------------------------

  
  //-- Define the phase matrix
  /* In defining the phase matrix, only the non-zero entries of momMatrix
   * are used, hence the phase matrix has dimensions V3*Nmoms
   */
  complex<QUDA_REAL> *phaseMatrix_host = NULL;
  complex<QUDA_REAL> *phaseMatrix_dev  = NULL;

  cudaMalloc( (void**)&phaseMatrix_dev, sizeof(complex<QUDA_REAL>)*V3*Nmoms );
  checkCudaErrorNoSync();
  cudaMemset(phaseMatrix_dev, 0, sizeof(complex<QUDA_REAL>)*V3*Nmoms);
    
  double t1 = MPI_Wtime();
  if(GPU_phaseMatrix){
    createPhaseMatrix_GPU(phaseMatrix_dev, momMatrix, paramAPI.mpParam, localL, totalL);
  }
  else{
    phaseMatrix_host = (complex<QUDA_REAL>*) calloc(V3*Nmoms, sizeof(complex<QUDA_REAL>));
    if(phaseMatrix_host == NULL) errorQuda("momentumProjectionPropagator_Quda: Cannot allocate phaseMatrix on host. Exiting.\n");    
    createPhaseMatrix_CPU(phaseMatrix_host, momMatrix, paramAPI.mpParam, localL, totalL);
  }
  double t2 = MPI_Wtime();
  printfQuda("momentumProjectionPropagator_Quda: Phase matrix created in %f sec.\n", t2-t1);
  //------------------------------------
    
    
  //-- Perform momentum projection

  /* For testing purposes, I define a second pair
   * of input/output buffers in which I copy the propagator
   * data in a V3*Ndata*Lt format. This has to be column-major, i.e. the index
   * corresponding to V3 should run fastest.
   * For convenience, time should run the slowest
   */

  complex<QUDA_REAL> *corrIn_proj  = NULL;
  complex<QUDA_REAL> *corrOut_proj = NULL;

  corrIn_proj  = (complex<QUDA_REAL>*) calloc(V3*Ndata*Lt   , sizeof(complex<QUDA_REAL>));
  corrOut_proj = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*Lt, sizeof(complex<QUDA_REAL>));
  if(corrIn_proj == NULL || corrOut_proj == NULL)
    errorQuda("Cannot allocate proj buffers for corrIn or corrOut\n");
  
  //-- Not the fastest way to do it, but for testing is fair enough...
  for(int jc=0;jc<Nc;jc++){
    for(int js=0;js<Ns;js++)
      for(int ic=0;ic<Nc;ic++)
	for(int is=0;is<Ns;is++)
	  for(int it=0;it<Lt;it++)
	    for(int iv=0;iv<V3;iv++){
	      LONG_T ptr_f = 2*is + 2*Ns*ic + 2*Ns*Nc*iv + 2*Ns*Nc*V3*it + 2*Ns*Nc*V3*Lt*js + 2*Ns*Nc*V3*Lt*Ns*jc;
	      LONG_T ptr_t = iv + V3*is + V3*Ns*ic + V3*Ns*Nc*js + V3*Ns*Nc*Ns*jc + V3*Ns*Nc*Ns*Nc*it;
	      
	      corrIn_proj[ptr_t].x = corrIn[0 + ptr_f];
	      corrIn_proj[ptr_t].y = corrIn[1 + ptr_f];
	    }
  }  
  printfQuda("momentumProjectionPropagator_Quda: Input data transformed\n");

  
  //-- Define the device buffers
  complex<QUDA_REAL> *corrIn_dev = NULL;
  complex<QUDA_REAL> *corrOut_dev = NULL;

  complex<QUDA_REAL> al = complex<QUDA_REAL>{1.0,0.0};
  complex<QUDA_REAL> be = complex<QUDA_REAL>{1.0,0.0};
  
  cublasStatus_t stat;
  cublasHandle_t handle;
  
  cudaMalloc( (void**)&corrIn_dev,      sizeof(complex<QUDA_REAL>)*V3*Ndata*Lt );
  cudaMalloc( (void**)&corrOut_dev,     sizeof(complex<QUDA_REAL>)*Nmoms*Ndata*Lt );
  checkCudaErrorNoSync();

  //-- Copy matrices from host to the device
  stat = cublasCreate(&handle);
  
  if(!GPU_phaseMatrix){
    stat = cublasSetMatrix(V3, Nmoms, sizeof(complex<QUDA_REAL>), phaseMatrix_host, V3, phaseMatrix_dev, V3);
    if(stat != CUBLAS_STATUS_SUCCESS)
      errorQuda("momentumProjectionPropagator_Quda: phaseMatrix data copy to GPU failed!\n");
  }
  
  stat = cublasSetMatrix(V3, Ndata*Lt, sizeof(complex<QUDA_REAL>), corrIn_proj, V3, corrIn_dev, V3);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("momentumProjectionPropagator_Quda: corrIn data copy to GPU failed!\n");
  
  stat = cublasSetMatrix(Nmoms, Ndata*Lt, sizeof(complex<QUDA_REAL>), corrOut_proj, Nmoms, corrOut_dev, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("momentumProjectionPropagator_Quda: corrOut data copy to GPU failed!\n");

  
  //-- Perform projection
  /* Matrix Multiplication Out = PH^T * In.
   * phaseMatrix=(V3,Nmoms) is the phase matrix in column-major format, its transpose is used for multiplication
   * corrIn=(V3,Ndata*Lt) is the input correlation matrix
   * corrOut=(Nmoms,Ndata*Lt) is the output matrix in column-major format
   */
  double t3 = MPI_Wtime();
  stat = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmoms, Ndata*Lt, V3,
		     &al, phaseMatrix_dev, V3,
		     corrIn_dev , V3, &be,
		     corrOut_dev, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("momentumProjectionPropagator_Quda: Momentum projection failed!\n");
  double t4 = MPI_Wtime();
  printfQuda("momentumProjectionPropagator_Quda: Projection completed in %f sec.\n",t4-t3);

  
  //-- extract the result from GPU to CPU
  complex<QUDA_REAL> *corrOut_local  = NULL;
  complex<QUDA_REAL> *corrOut_global = NULL;
  corrOut_local  = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*Lt  , sizeof(complex<QUDA_REAL>));
  corrOut_global = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*totT, sizeof(complex<QUDA_REAL>));
  if(corrOut_local == NULL || corrOut_global == NULL)
    errorQuda("momentumProjectionPropagator_Quda: Cannot allocate output buffers\n");
  
  stat = cublasGetMatrix(Nmoms, Ndata*Lt, sizeof(complex<QUDA_REAL>), corrOut_dev, Nmoms, corrOut_local, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("momentumProjectionPropagator_Quda: corrOut data copy to CPU failed!\n");
  
  memcpy(corrOut_proj, corrOut_local, sizeof(complex<QUDA_REAL>)*Nmoms*Ndata*Lt);

  
  //-- Perform reduction over all processes
  /* Create a separate communicator
   * All processes with the same comm_coord(3) belong to COMM_TIME communicator.
   * When performing the reduction over the COMM_TIME communicator, the global sum
   * will be performed across all processes with the same time-coordinate,
   * and the result will be placed at the "root" of each of the "time" groups.
   * This means that the global result will exist only at the "time" processes, where each will
   * hold the sum for its corresponing time slices.
   * In the case where only the time-direction is partitioned,
   * MPI_Reduce is essentially a memcpy.
  */
  /*
  int time_rank, time_size;
  MPI_Comm COMM_TIME;
  MPI_Comm_split(MPI_COMM_WORLD, comm_coord(3), comm_rank(), &COMM_TIME);
  MPI_Comm_rank(COMM_TIME,&time_rank);
  MPI_Comm_size(COMM_TIME,&time_size);

  MPI_Reduce(corrOut_local, corrOut_proj, Nmoms*Ndata*Lt, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, COMM_TIME);  

  MPI_Gather(corrOut_proj  , Nmoms*Ndata*Lt, MPI_DOUBLE_COMPLEX,
	     corrOut_global, Nmoms*Ndata*Lt, MPI_DOUBLE_COMPLEX,
	     0, COMM_TIME);
  */
  
  //-- Not the fastest way to do it, but for testing is fair enough...
  // Works only when Nmoms = totV3 AND only the time direction is partitioned (such that V3 = totV3),
  // so that all entries of the output propagator get (non-zero) entries
  // and we don't go out of boundaries
  for(int jc=0;jc<Nc;jc++){
    for(int js=0;js<Ns;js++)
      for(int ic=0;ic<Nc;ic++)
	for(int is=0;is<Ns;is++)
	  for(int it=0;it<Lt;it++)
	    for(int im=0;im<Nmoms;im++){
	      LONG_T ptr_f = im   + Nmoms*is + Nmoms*Ns*ic + Nmoms*Ns*Nc*js + Nmoms*Ns*Nc*Ns*jc + Nmoms*Ns*Nc*Ns*Nc*it;
	      LONG_T ptr_t = 2*is + 2*Ns*ic  + 2*Ns*Nc*im  + 2*Ns*Nc*V3*it  + 2*Ns*Nc*V3*Lt*js  + 2*Ns*Nc*V3*Lt*Ns*jc;
	      
	      corrOut[0 + ptr_t] = corrOut_proj[ptr_f].real();
	      corrOut[1 + ptr_t] = corrOut_proj[ptr_f].imag();
	    }
  }
  printfQuda("momentumProjectionPropagator_Quda: Output data transformed\n");
  
  
  //-- cleanup & return  
  free(momMatrix);
  free(corrOut_local);
  free(corrIn_proj);
  free(corrOut_proj);
  if(!GPU_phaseMatrix) free(phaseMatrix_host);
  
  cudaFree(corrIn_dev);
  cudaFree(corrOut_dev);
  cudaFree(phaseMatrix_dev);

  cublasDestroy(handle);

  
  saveTuneCache();

  printfQuda("momentumProjectionPropagator_Quda: Returning...\n");
  
  return status;
}


//-- top level function, performs momentum projection
int momentumProjectCorr_Quda(XTRN_CPLX *corrOut, const complex<QUDA_REAL> *corrQuda_dev,
			     QluaUtilArg utilArg,
			     const qudaLattice *qS,
			     const int *momlist,
			     qudaAPI_Param paramAPI){
  int status = 0;

  char *func_name;
  asprintf(&func_name,"momentumProjectCorr_Quda");
  
  //-- Check-print parameters
  /* NOTE: In this function the QsqMax is not required, as the Momentum matrix comes as input */
  bool GPU_phaseMatrix = (paramAPI.mpParam.GPU_phaseMatrix == 1 ? true : false);
  LONG_T locvol = paramAPI.mpParam.locvol;
  int Nmoms     = paramAPI.mpParam.Nmoms;
  int Ndata     = paramAPI.mpParam.Ndata;
  double bc_t   = paramAPI.mpParam.bc_t;
  int expSgn    = paramAPI.mpParam.expSgn;
  if(expSgn != 1 && expSgn != -1)
    errorQuda("%s: Got invalid exponential sign, expSgn = %d!\n", func_name, expSgn);

  printfQuda("%s:\n", func_name);
  printfQuda("  Will create phase matrix on %s\n", GPU_phaseMatrix == true ? "GPU" : "CPU"); 
  printfQuda("  Got locvol = %lld\n", locvol);
  printfQuda("  Got Nmoms  = %d\n", Nmoms);
  printfQuda("  Got Ndata  = %d\n", Ndata);
  printfQuda("  Got bc_t   = %f\n", bc_t);
  printfQuda("  Got expSgn = %s\n", expSgn == 1 ? "PLUS" : "MINUS");
 
  setVerbosity(paramAPI.verbosity);
  
  //-- Initialize the quda-gauge and invert parameters
  QudaGaugeParam gp;
  init_QudaGaugeParam_generic(gp, qS);

  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);

  //-- Define useful topology quantities
  int nDim = qS->rank;
  int momDim = nDim - 1;
  int localL[nDim];
  int totalL[nDim];
  int csrc[nDim];
  LONG_T Qlocvol = 1;
  for(int mu=0; mu<nDim; mu++){
    localL[mu] = qS->site_coord_hi[mu] - qS->site_coord_lo[mu];
    totalL[mu] = localL[mu] * comm_dim(mu);
    csrc[mu] = paramAPI.mpParam.csrc[mu];
    Qlocvol *= localL[mu];
  }
  int Lt   = localL[QUDA_TIME_AXIS];
  int totT = totalL[QUDA_TIME_AXIS];

  LONG_T totV3 = 1;
  LONG_T V3 = 1;
  for(int i=0;i<momDim;i++){
    totV3 *= totalL[i];
    V3    *= localL[i];
  }
  paramAPI.mpParam.momDim = momDim;
  paramAPI.mpParam.V3     = V3;
  printfQuda("  V3    = %lld\n", V3);
  printfQuda("  totV3 = %lld\n", totV3);

  
  int tAxis = paramAPI.mpParam.tAxis;
  if(tAxis != QUDA_TIME_AXIS)
    errorQuda("%s: Time-axis between Quda-Qlua does not agree! Exiting.\n", func_name);
  if(totT != paramAPI.mpParam.Tdim)
    errorQuda("%s: Time-dimension between Quda-Qlua does not agree! Exiting.\n", func_name); 
  if(locvol != Qlocvol)
    errorQuda("%s: Local volume between Quda-Qlua does not agree! Exiting.\n", func_name);

  
  printfQuda("  Got source-coords (x,y,z,t) = (%d,%d,%d,%d)\n", csrc[0], csrc[1], csrc[2], csrc[3]);  
  printfQuda("  Got t_axis = %d\n", tAxis);
  printfQuda("  Got tlen   = %d\n", paramAPI.mpParam.Tdim);
  /* --------------------------------------------------------------------------------------- */
  /* --------------------------------------------------------------------------------------- */

  
  //-- Define the phase matrix
  //-- The phase matrix has dimensions V3*Nmoms
  //-- The device buffer needs to always be allocated as it's passed to cublasZgemm 
  complex<QUDA_REAL> *phaseMatrix_host = NULL;
  complex<QUDA_REAL> *phaseMatrix_dev  = NULL;
  
  cudaMalloc( (void**)&phaseMatrix_dev, sizeof(complex<QUDA_REAL>)*V3*Nmoms );
  checkCudaErrorNoSync();
  cudaMemset(phaseMatrix_dev, 0, sizeof(complex<QUDA_REAL>)*V3*Nmoms);
    
  if(GPU_phaseMatrix){
    createPhaseMatrix_GPU(phaseMatrix_dev, momlist, paramAPI.mpParam, localL, totalL);
  }
  else{
    phaseMatrix_host = (complex<QUDA_REAL>*) calloc(V3*Nmoms, sizeof(complex<QUDA_REAL>));
    if(phaseMatrix_host == NULL) errorQuda("%s: Cannot allocate phaseMatrix on host. Exiting.\n", func_name);    
    createPhaseMatrix_CPU(phaseMatrix_host, momlist, paramAPI.mpParam, localL, totalL);
  }
  printfQuda("%s: Phase matrix created.\n", func_name);
  /* --------------------------------------------------------------------------------------- */

  
  //-- Define output correlation buffers
  complex<QUDA_REAL> *corrOut_proj = NULL; //-- Final result (global summed, gathered) of momentum projection
  complex<QUDA_REAL> *corrOut_glob = NULL; //-- Globally summed momentum projection buffer
  complex<QUDA_REAL> *corrOut_host = NULL; //-- Host (local) output of cuBlas momentum projection
  
  corrOut_proj = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*totT, sizeof(complex<QUDA_REAL>));
  corrOut_glob = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*Lt  , sizeof(complex<QUDA_REAL>));
  corrOut_host = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*Lt  , sizeof(complex<QUDA_REAL>));
  if((corrOut_proj == NULL) ||
     (corrOut_glob == NULL) ||
     (corrOut_host == NULL))
    errorQuda("%s: Cannot allocate Output correlation function buffers\n", func_name);
  /* --------------------------------------------------------------------------------------- */

  
  /*---------- Perform momentum projection ----------*/
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  complex<QUDA_REAL> al = complex<QUDA_REAL>{1.0,0.0};
  complex<QUDA_REAL> be = complex<QUDA_REAL>{0.0,0.0};

  complex<QUDA_REAL> *corrInp_dev = NULL;
  complex<QUDA_REAL> *corrOut_dev = NULL;
  
  cudaMalloc( (void**)&corrInp_dev, sizeof(complex<QUDA_REAL>)*V3*Ndata*Lt );
  cudaMalloc( (void**)&corrOut_dev, sizeof(complex<QUDA_REAL>)*Nmoms*Ndata*Lt );
  checkCudaErrorNoSync();
  
  //-- Change volume site order from Quda-QDP to Qlua-QDP
  conv_siteOrder_QudaQdp_to_momproj(corrInp_dev, corrQuda_dev, utilArg);  

  stat = cublasSetMatrix(Nmoms, Ndata*Lt, sizeof(complex<QUDA_REAL>), corrOut_host, Nmoms, corrOut_dev, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: corrOut data copy to GPU failed!\n", func_name);

  if(!GPU_phaseMatrix){
    stat = cublasSetMatrix(V3, Nmoms, sizeof(complex<QUDA_REAL>), phaseMatrix_host, V3, phaseMatrix_dev, V3);
    if(stat != CUBLAS_STATUS_SUCCESS)
      errorQuda("%s: phaseMatrix data copy to GPU failed!\n", func_name);
  }
  
  //-- Perform projection
  /* Matrix Multiplication Out = PH^T * In.
   * phaseMatrix_dev=(V3,Nmoms) is the phase matrix in column-major format, its transpose is used for multiplication
   * corrInp_dev=(V3,Ndata*Lt) is the input correlation matrix
   * corrOut_dev=(Nmoms,Ndata*Lt) is the output matrix in column-major format
   */
  double t3 = MPI_Wtime();
  stat = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmoms, Ndata*Lt, V3,
		     &al, phaseMatrix_dev, V3,
		     corrInp_dev , V3, &be,
		     corrOut_dev, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: Momentum projection failed!\n", func_name);
  double t4 = MPI_Wtime();
  printfQuda("%s: cuBlas projection completed in %f sec.\n", func_name, t4-t3);

  
  //-- extract the result from GPU to CPU  
  stat = cublasGetMatrix(Nmoms, Ndata*Lt, sizeof(complex<QUDA_REAL>), corrOut_dev, Nmoms, corrOut_host, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: corrOut data copy to CPU failed!\n", func_name);


  //-- Perform reduction over all processes
  /* Create a separate communicator
   * All processes with the same comm_coord(3) belong to COMM_TIME communicator.
   * When performing the reduction over the COMM_TIME communicator, the global sum
   * will be performed across all processes with the same time-coordinate,
   * and the result will be placed at the "root" of each of the "time" groups.
   * This means that the global result will exist only at the "time" processes, where each will
   * hold the sum for its corresponing time slices.
   * (In the case where only the time-direction is partitioned, MPI_Reduce is essentially a memcpy).
   *
   * Then a Gathering is required, in order to put the global result from each of the "time" processes
   * into the final buffer (corrOut_proj).
   * MPI_Allgather takes care to put the result from each process to the corresponding location of the final buffer.
   * The final buffer is accesible from all ranks.
  */
  int time_rank, time_size;
  MPI_Comm COMM_TIME;
  MPI_Comm_split(MPI_COMM_WORLD, comm_coord(3), comm_rank(), &COMM_TIME);
  MPI_Comm_rank(COMM_TIME,&time_rank);
  MPI_Comm_size(COMM_TIME,&time_size);

  MPI_Reduce(corrOut_host, corrOut_glob, Nmoms*Ndata*Lt, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, COMM_TIME);  

  MPI_Allgather(corrOut_glob, Nmoms*Ndata*Lt, MPI_DOUBLE_COMPLEX,
		corrOut_proj, Nmoms*Ndata*Lt, MPI_DOUBLE_COMPLEX,
		COMM_TIME);
  
  /*
   * Now a transpose of the corrOut_proj is required such that it follows the Qlua-C
   * convention, T-inside-Nmoms-inside-Ndata. A shift of the source-time to zero is also required,
   * together with boundary condition application
   */
  for(int it=0;it<totT;it++){
    int itShf = (it + csrc[3]) % totT;
    complex<QUDA_REAL> bc_fct = ((it + csrc[3]) >= totT) ? complex<QUDA_REAL>{bc_t,0} : complex<QUDA_REAL>{1,0};
    for(int id=0;id<Ndata;id++){
      for(int im=0;im<Nmoms;im++){	
	int idx_from = im + Nmoms*id + Nmoms*Ndata*itShf;  //- This is how the indices of the Mom-projected buffer come out from cuBlas
	int idx_to   = it + totT*im  + totT*Nmoms*id;      //- This is how the indices of the Mom-projected buffer should be

	corrOut[idx_to] = bc_fct * corrOut_proj[idx_from];
      }
    }
  }


  for(int i=0;i<20;i++)
    printfQuda("corrOut[%d] , h, g, p, f: = %+.6lf %+.6f  ,  %+.6lf %+.6f  ,  %+.6lf %+.6f,  %+.6lf %+.6f \n",i,
	       corrOut_host[i].real(), corrOut_host[i].imag(),
	       corrOut_glob[i].real(), corrOut_glob[i].imag(),
	       corrOut_proj[i].real(), corrOut_proj[i].imag(),
	       corrOut[i].real()     , corrOut[i].imag());

  
  
  //-- cleanup & return  
  free(corrOut_proj);
  free(corrOut_glob);
  free(corrOut_host);

  if(!GPU_phaseMatrix) free(phaseMatrix_host);
  
  cudaFree(phaseMatrix_dev);
  cudaFree(corrInp_dev);
  cudaFree(corrOut_dev);

  cublasDestroy(handle);
  
  saveTuneCache();

  printfQuda("%s: Returning...\n", func_name);
  
  return status;
}



EXTRN_C int
baryon_sigma_twopt_asymsrc_gvec_momProj_Quda(XTRN_CPLX *momproj_buf, XTRN_CPLX *corrQuda, const qudaLattice *qS, const int *momlist,
					     QUDA_REAL *hprop1, QUDA_REAL *hprop2, QUDA_REAL *hprop3,
					     XTRN_CPLX *S2, XTRN_CPLX *S1,
                                             int Nc, int Ns, qudaAPI_Param paramAPI){

  int status = 0;

  char *func_name;
  asprintf(&func_name,"baryon_sigma_twopt_asymsrc_gvec_momProj_Quda");
  
  if (check_quda_comms(qS))
    return 1;

  if((Nc != QUDA_Nc) || (Ns != QUDA_Ns))
    return 1;

  //-- Load the parameters required for the CSFs
  QudaGaugeParam gp;
  init_QudaGaugeParam_generic(gp, qS);
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);
  setVerbosity(paramAPI.verbosity);

  
  //-- Read the propagators into cuda-CSFs, load into device pointers
  int nVec = 12;
  LONG_T fieldLgh = paramAPI.mpParam.locvol * Nc * Ns * 2;
  
  ColorSpinorField *cudaProp1[nVec];
  ColorSpinorField *cudaProp2[nVec];
  ColorSpinorField *cudaProp3[nVec];
  
  for(int ivec=0;ivec<nVec;ivec++){
    cudaProp1[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(hprop1[ivec * fieldLgh]) );
    cudaProp2[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(hprop2[ivec * fieldLgh]) );
    cudaProp3[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(hprop3[ivec * fieldLgh]) );

    if((cudaProp1[ivec] == NULL) ||
       (cudaProp2[ivec] == NULL) ||
       (cudaProp3[ivec] == NULL))
      errorQuda("%s: Cannot allocate propagators. Exiting.\n", func_name);

    checkCudaError();
  }
  printfQuda("%s: Color-spinor fields created.\n", func_name);
  /* --------------------------------------------------------------------------------------- */

  //-- Create a utility structure (required in momentum projection as well)
  QluaUtilArg utilArg(cudaProp1, paramAPI.mpParam.Ndata, paramAPI.mpParam.Ndata, paramAPI.mpParam.tAxis, sizeof(complex<QUDA_REAL>));

  //-- Check coordinate conventions
  int crdChkVal = QluaCoordCheck(utilArg);  
  if(crdChkVal == -1)
    errorQuda("Site mismatch! Exiting.\n");
  else if (crdChkVal == 0)
    printfQuda("Site order check PASSED.\n");
  /* --------------------------------------------------------------------------------------- */

  
  //-- GPU contractions  
  size_t corrSize = sizeof(complex<QUDA_REAL>) * paramAPI.mpParam.locvol * paramAPI.mpParam.Ndata;

  complex<QUDA_REAL> *corrQuda_dev = NULL;
  cudaMalloc((void**)&corrQuda_dev, corrSize);
  checkCudaErrorNoSync();
  cudaMemset(corrQuda_dev, 0, corrSize);

  double t1 = MPI_Wtime();
  contractGPU_baryon_sigma_twopt_asymsrc_gvec(corrQuda_dev,
					      cudaProp1, cudaProp2, cudaProp3,
					      (complex<QUDA_REAL>*)S2, (complex<QUDA_REAL>*)S1,
					      paramAPI.mpParam);
  double t2 = MPI_Wtime();
  printfQuda("%s: GPU Contractions completed in %f sec.\n", func_name, t2-t1);
  /* --------------------------------------------------------------------------------------- */

  
  //-- Copy the position space correlator back to CPU if required
  if(paramAPI.mpParam.push_res){
    cudaMemcpy(corrQuda, corrQuda_dev, corrSize, cudaMemcpyDeviceToHost);
    checkCudaErrorNoSync();
  }
  
  
  //-- Call momentum-projection function  
  int mpStat = momentumProjectCorr_Quda(momproj_buf, corrQuda_dev, utilArg, qS, momlist, paramAPI);
  if(mpStat != 0) return 1;

  
  //-- cleanup & return
  for(int ivec=0;ivec<nVec;ivec++){
    delete_not_null(cudaProp1[ivec]);
    delete_not_null(cudaProp2[ivec]);
    delete_not_null(cudaProp3[ivec]);
  }
  cudaFree(corrQuda_dev);
  
  saveTuneCache();
  
  return status;
}


  // int lV = paramAPI.mpParam.locvol;
  // int Ndata = paramAPI.mpParam.Ndata;

  // complex<QUDA_REAL> *corrQuda = NULL;
  // size_t corrSize = sizeof(complex<QUDA_REAL>) * lV * Ndata;
  // printfQuda("%s: corrSize = %lld bytes\n", func_name, (LONG_T)corrSize);
  
  // corrQuda = (complex<QUDA_REAL>*) malloc(corrSize);
  // if(corrQuda == NULL)
  //   errorQuda("%s: Cannot allocate correlator buffer. Exiting.\n", func_name);
  // memset(corrQuda, 0, corrSize);



  // int lL[4];
  // for(int mu=0; mu<4; mu++)
  //   lL[mu] = qS->site_coord_hi[mu] - qS->site_coord_lo[mu];


  // for(int id=0;id<paramAPI.mpParam.Ndata;id++){
  //   for(int iv=0;iv<paramAPI.mpParam.locvol;iv++){
  //     int idx = iv + qS->locvol * id;
  //     printf("corrPosSpc[%d] - (iv,id) = (%d,%d) : = %+lf  %+lf\n", idx, iv, id, corrPosSpc[idx].real(), corrPosSpc[idx].imag());
  //   }
  // } 
  
  //-- convert the volume index to QDP order
  // for(int id=0;id<Ndata;id++){
  //   for(int iv=0;iv<lV;iv++){
  //     int idx_from = qS->ind_qdp2quda[iv] + lV*id;
  //     int idx_to = iv + lV*id;
  //     printf("***** corrQuda[%d] - (iv,id) = (%d,%d) : = %+lf  %+lf\n", idx_to, iv, id, corrQuda[idx_from].real(), corrQuda[idx_from].imag());
  //   }
  // }
  // memcpy(corrPosSpc, corrQuda, corrSize);
  

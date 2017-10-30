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
#include <interface_qlua.h>
#include <mpi.h>

using namespace quda;

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
                             const QudaGaugeParam& gp, double *alpha)
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
  ip.tune                     = QUDA_TUNE_YES;
  ip.use_init_guess           = QUDA_USE_INIT_GUESS_NO;
  ip.verbosity                = QUDA_VERBOSE;

  //-- FIXME: Need to change these
  ip.kappa                    = alpha[0];
  ip.clover_coeff             = alpha[1];
  ip.tol                      = alpha[2];
  ip.maxiter                  = int(alpha[3]);
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
    cuda_gf->copy(*cpu_gf);
    cuda_gf->exchangeGhost();
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
  //cpuParam.nColor = nColor, cpuParam.nSpin = nSpin; // hack!

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
			  cudaColorSpinorField& cuda_x)
{
  ColorSpinorParam cpuParam(hbuf_x, ip, gp.X, false); // false stands for the pc_solution
  cpuParam.nColor = nColor;  //
  cpuParam.nSpin  = nSpin;   // hack!
  cpuColorSpinorField cpu_x(cpuParam);                // cpuCSF wrapper for hbuf_x
  cpu_x = cuda_x;
}

//-- top level function, calls quda-wuppertal smearing
EXTRN_C int
laplacianQuda(
	      QUDA_REAL *hv_out,
	      QUDA_REAL *hv_in,
	      QUDA_REAL *h_gauge[],
	      const qudaLattice *qS,
	      int nColor, int nSpin,
	      double *alpha, double beta, int Nstep)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;
  if (QUDA_Nc != nColor)
    return 1;

  printfQuda("laplacianQuda: Will apply the Laplacian for %d steps with the parameters:\n",Nstep);
  for(int i=0; i< qS->rank; i++) printfQuda("  alpha[%d] = %.3f\n",i,alpha[i]);
  printfQuda("  beta = %.3f\n",beta);

  //-- Initialize the quda-gauge parameters
  QudaGaugeParam gp;
  init_QudaGaugeParam_generic(gp, qS);

  //-- Initialize the inverter parameters
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, alpha);

  setVerbosity(ip.verbosity);
  if(getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(&gp);
  
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
  for (int i = 0 ; i < Nstep ; i++){
    wuppertalStep(*cuda_v_out, *cuda_v_in, parity, *cuda_gf, alpha, beta);    
    cudaDeviceSynchronize();
    checkCudaError();
    
    *cuda_v_in = *cuda_v_out;
  }
  double t2 = MPI_Wtime(); 
  printfQuda("TIMING - laplacianQuda: Wuppertal smearing for Nstep = %d done in %.6f sec.\n", Nstep, t2-t1);
 
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

#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <short.h>

#include <comm_quda.h>

// This contains the appropriate ifdef guards already
#include <mpi_comm_handle.h>

#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include <test_util.h>
#include <test_params.h>

#include <dslash_quda.h>
#include "misc.h"

// For reading in the gauge fields
#include <qio_field.h>

using namespace std;

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

#define MAX(a,b) ((a)>(b)?(a):(b))

int Z[4];
int V;
int Vh;
int Vs_x, Vs_y, Vs_z, Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
int faceVolume[4];

//extended volume, +4
int E1, E1h, E2, E3, E4;
int E[4];
int V_ex, Vh_ex;

double kappa5;
int Ls;
int V5;
int V5h;

int my_spinor_site_size;

extern float fat_link_max;

// Set some local QUDA precision variables
QudaPrecision local_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision &cpu_prec = local_prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_refinement_sloppy = prec_refinement_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;
QudaPrecision &cuda_prec_ritz = prec_ritz;

void setQudaDefaultPrecs() {
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (smoother_halo_prec == QUDA_INVALID_PRECISION) smoother_halo_prec = prec_null;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;
}

void setQudaDefaultMgSolveTypes() {
  for (int i =0; i < QUDA_MAX_MG_LEVEL; i++) {
    if (coarse_solve_type[i] == QUDA_INVALID_SOLVE) coarse_solve_type[i] = solve_type;
    if (smoother_solve_type[i] == QUDA_INVALID_SOLVE) smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE;
  }  
}

void verifyInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck, QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge, void *clover, void *clover_inv) {
      if (multishift) {
      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	errorQuda("Mass normalization not supported for multi-shift solver in invert_test");
      }

      void *spinorTmp = malloc(V*spinor_site_size*host_spinor_data_type_size*inv_param.Ls);
      printfQuda("Host residuum checks: \n");
      for(int i=0; i < inv_param.num_offset; i++) {
	ax(0, spinorCheck, V*spinor_site_size, inv_param.cpu_prec);
	
	if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	  if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
	    int tm_offset = Vh*spinor_site_size;
	    void *out0 = spinorCheck;
	    void *out1 = (char*)out0 + tm_offset*cpu_prec;
	    
	    void *tmp0 = spinorTmp;
	    void *tmp1 = (char*)tmp0 + tm_offset*cpu_prec;
	    
	    void *in0  = spinorOutMulti[i];
	    void *in1  = (char*)in0 + tm_offset*cpu_prec;
	    
	    tm_ndeg_matpc(tmp0, tmp1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	    tm_ndeg_matpc(out0, out1, gauge, tmp0, tmp1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	  } else {
	    tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		     inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	    tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		     inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	  }
	} else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	  if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	    errorQuda("Twisted mass solution type not supported");
	  tmc_matpc(spinorTmp, gauge, spinorOutMulti[i], clover, clover_inv, inv_param.kappa, inv_param.mu,
		    inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  tmc_matpc(spinorCheck, gauge, spinorTmp, clover, clover_inv, inv_param.kappa, inv_param.mu,
		    inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_WILSON_DSLASH) {
	  wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
		    inv_param.cpu_prec, gauge_param);
	  wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
		    inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	  clover_matpc(spinorTmp, gauge, clover, clover_inv, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
		       inv_param.cpu_prec, gauge_param);
	  clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
		       inv_param.cpu_prec, gauge_param);
	} else {
	  printfQuda("Domain wall not supported for multi-shift\n");
	  exit(-1);
	}
	
	axpy(inv_param.offset[i], spinorOutMulti[i], spinorCheck, Vh*spinor_site_size, inv_param.cpu_prec);
	mxpy(spinorIn, spinorCheck, Vh*spinor_site_size, inv_param.cpu_prec);
	double nrm2 = norm_2(spinorCheck, Vh*spinor_site_size, inv_param.cpu_prec);
	double src2 = norm_2(spinorIn, Vh*spinor_site_size, inv_param.cpu_prec);
	double l2r = sqrt(nrm2 / src2);
	
	printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
		   i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, 
		   inv_param.tol_hq_offset[i], inv_param.true_res_hq_offset[i]);
      }
      free(spinorTmp);
      
    } else {
      // Non-multishift workflow
      if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
	if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	  if(inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
	    tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
	  } else {
	    int tm_offset = V*spinor_site_size;
	    void *evenOut = spinorCheck;
	    void *oddOut  = (char*)evenOut + tm_offset*cpu_prec;
	    
	    void *evenIn  = spinorOut;
	    void *oddIn   = (char*)evenIn + tm_offset*cpu_prec;
	    
	    tm_ndeg_mat(evenOut, oddOut, gauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0, inv_param.cpu_prec, gauge_param);
	  }
	} else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	  tmc_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
		  inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_WILSON_DSLASH) {
	  wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	  clover_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	  dw_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
	} else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
	  dw_4d_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
	} else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
	  double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
	  double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
	  for(int xs = 0 ; xs < Lsdim ; xs++)
	    {
	      kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
	      kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
	    }
	  mdw_mat(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
	  free(kappa_b);
	  free(kappa_c);
	} else {
	  errorQuda("Unsupported dslash_type");
	}
	if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || 
	      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
	      dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
	    ax(0.5/kappa5, spinorCheck, V*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
	  } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH && twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
	    ax(0.5/inv_param.kappa, spinorCheck, 2*V*spinor_site_size, inv_param.cpu_prec);
	  } else {
	    ax(0.5/inv_param.kappa, spinorCheck, V*spinor_site_size, inv_param.cpu_prec);
	  }
	}
	
      } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {
	
	if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	  if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
	    int tm_offset = Vh*spinor_site_size;
	    void *out0 = spinorCheck;
	    void *out1 = (char*)out0 + tm_offset*cpu_prec;
	    
	    void *in0  = spinorOut;
	    void *in1  = (char*)in0 + tm_offset*cpu_prec;
	    
	    tm_ndeg_matpc(out0, out1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  } else {
	    tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		     inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  }
	} else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	  if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	    errorQuda("Twisted mass solution type not supported");
	  tmc_matpc(spinorCheck, gauge, spinorOut, clover, clover_inv, inv_param.kappa, inv_param.mu,
		    inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_WILSON_DSLASH) {
	  wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
		    inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	  clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
		       inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	  dw_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
	} else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
	  dw_4d_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
	} else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
	  double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
	  double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
	  for(int xs = 0 ; xs < Lsdim ; xs++)
	    {
	      kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
	      kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
	    }
	  mdw_matpc(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
	  free(kappa_b);
	  free(kappa_c);
	} else {
	  errorQuda("Unsupported dslash_type");
	}
	
	if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
	      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
	      dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
	    ax(0.25/(kappa5*kappa5), spinorCheck, V*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
	  } else {
	    ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, Vh*spinor_site_size, inv_param.cpu_prec);
	    
	  }
	}
	
      } else if (inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
	
	void *spinorTmp = malloc(V*spinor_site_size*host_spinor_data_type_size*inv_param.Ls);
	
	ax(0, spinorCheck, V*spinor_site_size, inv_param.cpu_prec);
	
	if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	  if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
	    int tm_offset = Vh*spinor_site_size;
	    void *out0 = spinorCheck;
	    void *out1 = (char*)out0 + tm_offset*cpu_prec;
	    
	    void *tmp0 = spinorTmp;
	    void *tmp1 = (char*)tmp0 + tm_offset*cpu_prec;
	    
	    void *in0  = spinorOut;
	    void *in1  = (char*)in0 + tm_offset*cpu_prec;
	    
	    tm_ndeg_matpc(tmp0, tmp1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	    tm_ndeg_matpc(out0, out1, gauge, tmp0, tmp1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	  } else {
	    tm_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		     inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	    tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		     inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	  }
	} else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	  if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	    errorQuda("Twisted mass solution type not supported");
	  tmc_matpc(spinorTmp, gauge, spinorOut, clover, clover_inv, inv_param.kappa, inv_param.mu,
		    inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  tmc_matpc(spinorCheck, gauge, spinorTmp, clover, clover_inv, inv_param.kappa, inv_param.mu,
		    inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_WILSON_DSLASH) {
	  wil_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
		    inv_param.cpu_prec, gauge_param);
	  wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
		    inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	  clover_matpc(spinorTmp, gauge, clover, clover_inv, spinorOut, inv_param.kappa,
		       inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa,
		       inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	} else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	  dw_matpc(spinorTmp, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
	  dw_matpc(spinorCheck, gauge, spinorTmp, kappa5, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param, inv_param.mass);
	} else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
	  dw_4d_matpc(spinorTmp, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
	  dw_4d_matpc(spinorCheck, gauge, spinorTmp, kappa5, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param, inv_param.mass);
	} else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
	  double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
	  double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
	  for(int xs = 0 ; xs < Lsdim ; xs++)
	    {
	      kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
	      kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
	    }
	  mdw_matpc(spinorTmp, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
	  mdw_matpc(spinorCheck, gauge, spinorTmp, kappa_b, kappa_c, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
	  free(kappa_b);
	  free(kappa_c);
	} else {
	  errorQuda("Unsupported dslash_type");
	}
	
	if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	  errorQuda("Mass normalization not implemented");
	}
	
	free(spinorTmp);
      }
      
      
      int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
      mxpy(spinorIn, spinorCheck, vol*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
      double nrm2 = norm_2(spinorCheck, vol*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
      double src2 = norm_2(spinorIn, vol*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
      double l2r = sqrt(nrm2 / src2);
      
      printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
		 inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);      
    }
}
  


size_t host_gauge_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
size_t host_spinor_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);



/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */

void get_gridsize_from_env(int *const dims)
{
  char *grid_size_env = getenv("QUDA_TEST_GRID_SIZE");
  if (grid_size_env) {
    std::stringstream grid_list(grid_size_env);

    int dim;
    int i = 0;
    while (grid_list >> dim) {
      if (i >= 4) errorQuda("Unexpected grid size array length");
      dims[i] = dim;
      if (grid_list.peek() == ',') grid_list.ignore();
      i++;
    }
  }
}

static int lex_rank_from_coords_t(const int *coords, void *fdata)
{
  int rank = coords[0];
  for (int i = 1; i < 4; i++) {
    rank = gridsize_from_cmdline[i] * rank + coords[i];
  }
  return rank;
}

static int lex_rank_from_coords_x(const int *coords, void *fdata)
{
  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = gridsize_from_cmdline[i] * rank + coords[i];
  }
  return rank;
}

void initComms(int argc, char **argv, std::array<int, 4> &commDims) { initComms(argc, argv, commDims.data()); }

void initComms(int argc, char **argv, int *const commDims)
{
  if (getenv("QUDA_TEST_GRID_SIZE")) get_gridsize_from_env(commDims);

#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  // make sure the QMP logical ordering matches QUDA's
  if (rank_order == 0) {
    int map[] = {3, 2, 1, 0};
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  } else {
    int map[] = { 0, 1, 2, 3 };
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  }
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  QudaCommsMap func = rank_order == 0 ? lex_rank_from_coords_t : lex_rank_from_coords_x;

  initCommsGridQuda(4, commDims, func, NULL);
  initRand();

  printfQuda("Rank order is %s major (%s running fastest)\n",
	     rank_order == 0 ? "column" : "row", rank_order == 0 ? "t" : "x");

}

bool last_node_in_t()
{
  // only apply T-boundary at edge nodes
#ifdef MULTI_GPU
  return commCoords(3) == commDim(3) - 1;
#else
  return true;
#endif
}

void finalizeComms()
{
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
}


void initRand()
{
  int rank = 0;

#if defined(QMP_COMMS)
  rank = QMP_get_node_number();
#elif defined(MPI_COMMS)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  srand(17*rank + 137);
}

void setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i=0; i<4; i++) {
      if (i==d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V/2;

  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];

  Vsh_x = Vs_x/2;
  Vsh_y = Vs_y/2;
  Vsh_z = Vs_z/2;
  Vsh_t = Vs_t/2;


  E1=X[0]+4; E2=X[1]+4; E3=X[2]+4; E4=X[3]+4;
  E1h=E1/2;
  E[0] = E1;
  E[1] = E2;
  E[2] = E3;
  E[3] = E4;
  V_ex = E1*E2*E3*E4;
  Vh_ex = V_ex/2;

}

void dw_setDims(int *X, const int L5)
{
  V = 1;
  for (int d = 0; d < 4; d++) {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i=0; i<4; i++) {
      if (i==d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V/2;

  Ls = L5;
  V5 = V*Ls;
  V5h = Vh*Ls;

  Vs_t = Z[0]*Z[1]*Z[2]*Ls;//?
  Vsh_t = Vs_t/2;  //?
}


void setSpinorSiteSize(int n)
{
  my_spinor_site_size = n;
}


template <typename Float>
static void printVector(Float *v) {
  printfQuda("{(%f %f) (%f %f) (%f %f)}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

// X indexes the lattice site
void printSpinorElement(void *spinor, int X, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION)
    for (int s=0; s<4; s++) printVector((double*)spinor+X*24+s*6);
  else
    for (int s=0; s<4; s++) printVector((float*)spinor+X*24+s*6);
}

// X indexes the full lattice
void printGaugeElement(void *gauge, int X, QudaPrecision precision) {
  if (getOddBit(X) == 0) {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m=0; m<3; m++) printVector((double*)gauge +(X/2)*gauge_site_size + m*3*2);
    else
      for (int m=0; m<3; m++) printVector((float*)gauge +(X/2)*gauge_site_size + m*3*2);

  } else {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m = 0; m < 3; m++) printVector((double*)gauge + (X/2+Vh)*gauge_site_size + m*3*2);
    else
      for (int m = 0; m < 3; m++) printVector((float*)gauge + (X/2+Vh)*gauge_site_size + m*3*2);
  }
}

// returns 0 or 1 if the full lattice index X is even or odd
int getOddBit(int Y) {
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  return (x4+x3+x2+x1) % 2;
}

// a+=b
template <typename Float>
inline void complexAddTo(Float *a, Float *b) {
  a[0] += b[0];
  a[1] += b[1];
}

// a = b*c
template <typename Float>
inline void complexProduct(Float *a, Float *b, Float *c) {
  a[0] = b[0]*c[0] - b[1]*c[1];
  a[1] = b[0]*c[1] + b[1]*c[0];
}

// a = conj(b)*conj(c)
template <typename Float>
inline void complexConjugateProduct(Float *a, Float *b, Float *c) {
  a[0] = b[0]*c[0] - b[1]*c[1];
  a[1] = -b[0]*c[1] - b[1]*c[0];
}

// a = conj(b)*c
template <typename Float>
inline void complexDotProduct(Float *a, Float *b, Float *c) {
  a[0] = b[0]*c[0] + b[1]*c[1];
  a[1] = b[0]*c[1] - b[1]*c[0];
}

// a += b*c
template <typename Float>
inline void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign) {
  a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
  a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
}

// a += conj(b)*c)
template <typename Float>
inline void accumulateComplexDotProduct(Float *a, Float *b, Float *c) {
  a[0] += b[0]*c[0] + b[1]*c[1];
  a[1] += b[0]*c[1] - b[1]*c[0];
}

template <typename Float>
inline void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign) {
  a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
  a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
}

template <typename Float>
inline void su3Construct12(Float *mat) {
  Float *w = mat+12;
  w[0] = 0.0;
  w[1] = 0.0;
  w[2] = 0.0;
  w[3] = 0.0;
  w[4] = 0.0;
  w[5] = 0.0;
}

// Stabilized Bunk and Sommer
template <typename Float>
inline void su3Construct8(Float *mat) {
  mat[0] = atan2(mat[1], mat[0]);
  mat[1] = atan2(mat[13], mat[12]);
  for (int i=8; i<18; i++) mat[i] = 0.0;
}

void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (precision == QUDA_DOUBLE_PRECISION) su3Construct12((double*)mat);
    else su3Construct12((float*)mat);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) su3Construct8((double*)mat);
    else su3Construct8((float*)mat);
  }
}

// given first two rows (u,v) of SU(3) matrix mat, reconstruct the third row
// as the cross product of the conjugate vectors: w = u* x v*
//
// 48 flops
template <typename Float>
static void su3Reconstruct12(Float *mat, int dir, int ga_idx, QudaGaugeParam *param) {
  Float *u = &mat[0*(3*2)];
  Float *v = &mat[1*(3*2)];
  Float *w = &mat[2*(3*2)];
  w[0] = 0.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0; w[5] = 0.0;
  accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
  accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
  accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
  accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
  accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
  accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
  Float u0 = (dir < 3 ? param->anisotropy :
	      (ga_idx >= (Z[3]-1)*Z[0]*Z[1]*Z[2]/2 ? param->t_boundary : 1));
  w[0]*=u0; w[1]*=u0; w[2]*=u0; w[3]*=u0; w[4]*=u0; w[5]*=u0;
}

template <typename Float>
static void su3Reconstruct8(Float *mat, int dir, int ga_idx, QudaGaugeParam *param) {
  // First reconstruct first row
  Float row_sum = 0.0;
  row_sum += mat[2]*mat[2];
  row_sum += mat[3]*mat[3];
  row_sum += mat[4]*mat[4];
  row_sum += mat[5]*mat[5];
  Float u0 = (dir < 3 ? param->anisotropy :
	      (ga_idx >= (Z[3]-1)*Z[0]*Z[1]*Z[2]/2 ? param->t_boundary : 1));
  Float U00_mag = sqrt(1.f/(u0*u0) - row_sum);

  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  Float column_sum = 0.0;
  for (int i=0; i<2; i++) column_sum += mat[i]*mat[i];
  for (int i=6; i<8; i++) column_sum += mat[i]*mat[i];
  Float U20_mag = sqrt(1.f/(u0*u0) - column_sum);

  mat[12] = U20_mag * cos(mat[15]);
  mat[13] = U20_mag * sin(mat[15]);

  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  Float r_inv2 = 1.0/(u0*row_sum);

  // U11
  Float A[2];
  complexDotProduct(A, mat+0, mat+6);
  complexConjugateProduct(mat+8, mat+12, mat+4);
  accumulateComplexProduct(mat+8, A, mat+2, u0);
  mat[8] *= -r_inv2;
  mat[9] *= -r_inv2;

  // U12
  complexConjugateProduct(mat+10, mat+12, mat+2);
  accumulateComplexProduct(mat+10, A, mat+4, -u0);
  mat[10] *= r_inv2;
  mat[11] *= r_inv2;

  // U21
  complexDotProduct(A, mat+0, mat+12);
  complexConjugateProduct(mat+14, mat+6, mat+4);
  accumulateComplexProduct(mat+14, A, mat+2, -u0);
  mat[14] *= r_inv2;
  mat[15] *= r_inv2;

  // U12
  complexConjugateProduct(mat+16, mat+6, mat+2);
  accumulateComplexProduct(mat+16, A, mat+4, u0);
  mat[16] *= -r_inv2;
  mat[17] *= -r_inv2;
}

void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision, QudaGaugeParam *param) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (precision == QUDA_DOUBLE_PRECISION) su3Reconstruct12((double*)mat, dir, ga_idx, param);
    else su3Reconstruct12((float*)mat, dir, ga_idx, param);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) su3Reconstruct8((double*)mat, dir, ga_idx, param);
    else su3Reconstruct8((float*)mat, dir, ga_idx, param);
  }
}

template <typename Float>
static int compareFloats(Float *a, Float *b, int len, double epsilon) {
  for (int i = 0; i < len; i++) {
    double diff = fabs(a[i] - b[i]);
    if (diff > epsilon) {
      printfQuda("ERROR: i=%d, a[%d]=%f, b[%d]=%f\n", i, i, a[i], i, b[i]);
      return 0;
    }
  }
  return 1;
}

int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision) {
  if  (precision == QUDA_DOUBLE_PRECISION) return compareFloats((double*)a, (double*)b, len, epsilon);
  else return compareFloats((float*)a, (float*)b, len, epsilon);
}

int fullLatticeIndex(int dim[4], int index, int oddBit){

  int za = index/(dim[0]>>1);
  int zb = za/dim[1];
  int x2 = za - zb*dim[1];
  int x4 = zb/dim[2];
  int x3 = zb - x4*dim[2];

  return  2*index + ((x2 + x3 + x4 + oddBit) & 1);
}

// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
int fullLatticeIndex(int i, int oddBit) {
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */

  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  //int X4 = Z[3];
  int X1h =X1/2;

  int sid =i;
  int za = sid/X1h;
  //int x1h = sid - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  //int x1 = 2*x1h + x1odd;
  int X = 2 * sid + x1odd;

  return X;
}

// i represents a "half index" into an even or odd "half lattice".
// when oddBit={0,1} the half lattice is {even,odd}.
//
// the displacements, such as dx, refer to the full lattice coordinates.
//
// neighborIndex() takes a "half index", displaces it, and returns the
// new "half index", which can be an index into either the even or odd lattices.
// displacements of magnitude one always interchange odd and even lattices.
//

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1) {
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];

  // assert (oddBit == (x+y+z+t)%2);

  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];

  return (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
}


int neighborIndex(int dim[4], int index, int oddBit, int dx[4]){

  const int fullIndex = fullLatticeIndex(dim, index, oddBit);

  int x[4];
  x[3] = fullIndex/(dim[2]*dim[1]*dim[0]);
  x[2] = (fullIndex/(dim[1]*dim[0])) % dim[2];
  x[1] = (fullIndex/dim[0]) % dim[1];
  x[0] = fullIndex % dim[0];

  for(int dir=0; dir<4; ++dir)
    x[dir] = (x[dir]+dx[dir]+dim[dir]) % dim[dir];

  return (((x[3]*dim[2] + x[2])*dim[1] + x[1])*dim[0] + x[0])/2;
}

int
neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1)
{
  int ret;

  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];

  int ghost_x4 = x4+ dx4;

  // assert (oddBit == (x+y+z+t)%2);

  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];

  if ( (ghost_x4 >= 0 && ghost_x4 < Z[3]) || !comm_dim_partitioned(3)){
    ret = (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
  }else{
    ret = (x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
  }

  return ret;
}

/*
 * This is a computation of neighbor using the full index and the displacement in each direction
 *
 */

int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1)
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }

  int nbr_half_idx = neighborIndex(half_idx, oddBit, dx4,dx3,dx2,dx1);
  int oddBitChanged = (dx4+dx3+dx2+dx1)%2;
  if (oddBitChanged){
    oddBit = 1 - oddBit;
  }
  int ret = nbr_half_idx;
  if (oddBit){
    ret = Vh + nbr_half_idx;
  }

  return ret;
}

int
neighborIndexFullLattice(int dim[4], int index, int dx[4])
{
  const int volume = dim[0]*dim[1]*dim[2]*dim[3];
  const int halfVolume = volume/2;
  int oddBit = 0;
  int halfIndex = index;

  if(index >= halfVolume){
    oddBit = 1;
    halfIndex = index - halfVolume;
  }

  int neighborHalfIndex = neighborIndex(dim, halfIndex, oddBit, dx);

  int oddBitChanged = (dx[0]+dx[1]+dx[2]+dx[3])%2;
  if(oddBitChanged){
    oddBit = 1 - oddBit;
  }

  return neighborHalfIndex + oddBit*halfVolume;
}

int neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1)
{
  int ret;
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }

  int Y = fullLatticeIndex(half_idx, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  int ghost_x4 = x4+ dx4;

  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];

  if ( ghost_x4 >= 0 && ghost_x4 < Z[3]){
    ret = (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
  }else{
    ret = (x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
    return ret;
  }

  int oddBitChanged = (dx4+dx3+dx2+dx1)%2;
  if (oddBitChanged){
    oddBit = 1 - oddBit;
  }

  if (oddBit){
    ret += Vh;
  }

  return ret;
}


// 4d checkerboard.
// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
// Cf. GPGPU code in dslash_core_ante.h.
// There, i is the thread index.
int fullLatticeIndex_4d(int i, int oddBit) {
  if (i >= Vh || i < 0) {printf("i out of range in fullLatticeIndex_4d"); exit(-1);}
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */

  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  //int X4 = Z[3];
  int X1h =X1/2;

  int sid =i;
  int za = sid/X1h;
  //int x1h = sid - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  //int x1 = 2*x1h + x1odd;
  int X = 2 * sid + x1odd;

  return X;
}

// 5d checkerboard.
// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
// Cf. GPGPU code in dslash_core_ante.h.
// There, i is the thread index sid.
// This function is used by neighborIndex_5d in dslash_reference.cpp.
//ok
int fullLatticeIndex_5d(int i, int oddBit) {
  int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2) + i/(Z[3]*Z[2]*Z[1]*Z[0]/2);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

int fullLatticeIndex_5d_4dpc(int i, int oddBit) {
  int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

int x4_from_full_index(int i)
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }

  int Y = fullLatticeIndex(half_idx, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);

  return x4;
}

template <typename Float>
static void applyGaugeFieldScaling(Float **gauge, int Vh, QudaGaugeParam *param) {
  // Apply spatial scaling factor (u0) to spatial links
  for (int d = 0; d < 3; d++) {
    for (int i = 0; i < gauge_site_size*Vh*2; i++) {
      gauge[d][i] /= param->anisotropy;
    }
  }

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t()) {
    for (int j = (Z[0]/2)*Z[1]*Z[2]*(Z[3]-1); j < Vh; j++) {
      for (int i = 0; i < gauge_site_size; i++) {
	gauge[3][j*gauge_site_size+i] *= -1.0;
	gauge[3][(Vh+j)*gauge_site_size+i] *= -1.0;
      }
    }
  }

  if (param->gauge_fix) {
    // set all gauge links (except for the last Z[0]*Z[1]*Z[2]/2) to the identity,
    // to simulate fixing to the temporal gauge.
    int iMax = (last_node_in_t() ? (Z[0] / 2) * Z[1] * Z[2] * (Z[3] - 1) : Vh);
    int dir = 3; // time direction only
    Float *even = gauge[dir];
    Float *odd  = gauge[dir]+Vh*gauge_site_size;
    for (int i = 0; i< iMax; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }
}

template <typename Float>
void applyGaugeFieldScaling_long(Float **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type)
{
  int X1h=param->X[0]/2;
  int X1 =param->X[0];
  int X2 =param->X[1];
  int X3 =param->X[2];
  int X4 =param->X[3];

  // rescale long links by the appropriate coefficient
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    for(int d=0; d<4; d++){
      for(int i=0; i < V*gauge_site_size; i++){
	gauge[d][i] /= (-24*param->tadpole_coeff*param->tadpole_coeff);
      }
    }
  }

  // apply the staggered phases
  for (int d = 0; d < 3; d++) {

    //even
    for (int i = 0; i < Vh; i++) {

      int index = fullLatticeIndex(i, 0);
      int i4 = index /(X3*X2*X1);
      int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
      int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
      int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
      int sign = 1;

      if (d == 0) {
	if (i4 % 2 == 1){
	  sign= -1;
	}
      }

      if (d == 1){
	if ((i4+i1) % 2 == 1){
	  sign= -1;
	}
      }
      if (d == 2){
	if ( (i4+i1+i2) % 2 == 1){
	  sign= -1;
	}
      }

      for (int j=0; j < 18; j++) {
	gauge[d][i*gauge_site_size + j] *= sign;
      }
    }
    //odd
    for (int i = 0; i < Vh; i++) {
      int index = fullLatticeIndex(i, 1);
      int i4 = index /(X3*X2*X1);
      int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
      int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
      int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
      int sign = 1;

      if (d == 0) {
	if (i4 % 2 == 1){
	  sign = -1;
	}
      }

      if (d == 1){
	if ((i4+i1) % 2 == 1){
	  sign = -1;
	}
      }
      if (d == 2){
	if ( (i4+i1+i2) % 2 == 1){
	  sign = -1;
	}
      }

      for (int j=0; j<18; j++){
	gauge[d][(Vh+i)*gauge_site_size + j] *= sign;
      }
    }

  }

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t()) {
    for (int j = 0; j < Vh; j++) {
      int sign = 1;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
	if (j >= (X4-3)*X1h*X2*X3 ){
	  sign = -1;
	}
      } else {
	if (j >= (X4-1)*X1h*X2*X3 ){
	  sign = -1;
	}
      }

      for (int i=0; i<18; i++) {
	gauge[3][j*gauge_site_size + i] *= sign;
	gauge[3][(Vh+j)*gauge_site_size + i] *= sign;
      }
    }
  }
}

void applyGaugeFieldScaling_long(void **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type, QudaPrecision local_prec) {
  if (local_prec == QUDA_DOUBLE_PRECISION) {
    applyGaugeFieldScaling_long((double**)gauge, Vh, param, dslash_type);
  } else if (local_prec == QUDA_SINGLE_PRECISION) {
    applyGaugeFieldScaling_long((float**)gauge, Vh, param, dslash_type);
  } else {
    errorQuda("Invalid type %d for applyGaugeFieldScaling_long\n", local_prec);
  }
}

template <typename Float>
static void constructUnitGaugeField(Float **res, QudaGaugeParam *param) {
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gauge_site_size;
  }

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }

  applyGaugeFieldScaling(res, Vh, param);
}

// normalize the vector a
template <typename Float>
static void normalize(complex<Float> *a, int len) {
  double sum = 0.0;
  for (int i=0; i<len; i++) sum += norm(a[i]);
  for (int i=0; i<len; i++) a[i] /= sqrt(sum);
}

// orthogonalize vector b to vector a
template <typename Float>
static void orthogonalize(complex<Float> *a, complex<Float> *b, int len) {
  complex<double> dot = 0.0;
  for (int i=0; i<len; i++) dot += conj(a[i])*b[i];
  for (int i=0; i<len; i++) b[i] -= (complex<Float>)dot*a[i];
}

template <typename Float>
static void constructGaugeField(Float **res, QudaGaugeParam *param, QudaDslashType dslash_type = QUDA_WILSON_DSLASH)
{
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gauge_site_size;
  }

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
          resOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / (Float)RAND_MAX;
        }
      }
      normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);

      normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	Float *w = resEven[dir]+(i*3+0)*3*2;
	Float *u = resEven[dir]+(i*3+1)*3*2;
	Float *v = resEven[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

      {
	Float *w = resOdd[dir]+(i*3+0)*3*2;
	Float *u = resOdd[dir]+(i*3+1)*3*2;
	Float *v = resOdd[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

    }
  }

  if (param->type == QUDA_WILSON_LINKS) {
    applyGaugeFieldScaling(res, Vh, param);
  } else if (param->type == QUDA_ASQTAD_LONG_LINKS) {
    applyGaugeFieldScaling_long(res, Vh, param, dslash_type);
  } else if (param->type == QUDA_ASQTAD_FAT_LINKS) {
    for (int dir = 0; dir < 4; dir++) {
      for (int i = 0; i < Vh; i++) {
	for (int m = 0; m < 3; m++) { // last 2 rows
	  for (int n = 0; n < 3; n++) { // 3 columns
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] =1.0* rand() / (Float)RAND_MAX;
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 2.0* rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = 3.0*rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 4.0*rand() / (Float)RAND_MAX;
	  }
	}
      }
    }
  }
}

template <typename Float> void constructUnitaryGaugeField(Float **res)
{
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gauge_site_size;
  }

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
          resOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / (Float)RAND_MAX;
        }
      }
      normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);

      normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	Float *w = resEven[dir]+(i*3+0)*3*2;
	Float *u = resEven[dir]+(i*3+1)*3*2;
	Float *v = resEven[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

      {
	Float *w = resOdd[dir]+(i*3+0)*3*2;
	Float *u = resOdd[dir]+(i*3+1)*3*2;
	Float *v = resOdd[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }
    }
  }
}

template <typename Float> static void applyStaggeredScaling(Float **res, QudaGaugeParam *param, int type)
{

  if(type == 3)  applyGaugeFieldScaling_long((Float**)res, Vh, param, QUDA_STAGGERED_DSLASH);

  return;
}

void construct_fat_long_gauge_field(void **fatlink, void **longlink, int type, QudaPrecision precision,
    QudaGaugeParam *param, QudaDslashType dslash_type)
{
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION) {
      constructUnitGaugeField((double**)fatlink, param);
      constructUnitGaugeField((double**)longlink, param);
    }else {
      constructUnitGaugeField((float**)fatlink, param);
      constructUnitGaugeField((float**)longlink, param);
    }
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) {
      // if doing naive staggered then set to long links so that the staggered phase is applied
      param->type = dslash_type == QUDA_ASQTAD_DSLASH ? QUDA_ASQTAD_FAT_LINKS : QUDA_ASQTAD_LONG_LINKS;
      if(type != 3) constructGaugeField((double**)fatlink, param, dslash_type);
      else applyStaggeredScaling((double**)fatlink, param, type);
      param->type = QUDA_ASQTAD_LONG_LINKS;
      if (dslash_type == QUDA_ASQTAD_DSLASH)
      {
        if(type != 3) constructGaugeField((double**)longlink, param, dslash_type);
        else applyStaggeredScaling((double**)longlink, param, type);
      }
    }else {
      param->type = dslash_type == QUDA_ASQTAD_DSLASH ? QUDA_ASQTAD_FAT_LINKS : QUDA_ASQTAD_LONG_LINKS;
      if(type != 3) constructGaugeField((float**)fatlink, param, dslash_type);
      else applyStaggeredScaling((float**)fatlink, param, type);

      param->type = QUDA_ASQTAD_LONG_LINKS;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
        if(type != 3) constructGaugeField((float**)longlink, param, dslash_type);
        else applyStaggeredScaling((float**)longlink, param, type);
      }
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      // incorporate non-trivial phase into long links
      const double phase = (M_PI * rand()) / RAND_MAX;
      const complex<double> z = polar(1.0, phase);
      for (int dir = 0; dir < 4; ++dir) {
        for (int i = 0; i < V; ++i) {
          for (int j = 0; j < gauge_site_size; j += 2) {
            if (precision == QUDA_DOUBLE_PRECISION) {
              complex<double> *l = (complex<double> *)(&(((double *)longlink[dir])[i * gauge_site_size + j]));
              *l *= z;
            } else {
              complex<float> *l = (complex<float> *)(&(((float *)longlink[dir])[i * gauge_site_size + j]));
              *l *= z;
            }
          }
        }
      }
    }

    if (type == 3) return;

    // set all links to zero to emulate the 1-link operator (needed for host comparison)
    if (dslash_type == QUDA_STAGGERED_DSLASH) {
      for (int dir = 0; dir < 4; ++dir) {
        for (int i = 0; i < V; ++i) {
          for (int j = 0; j < gauge_site_size; j += 2) {
            if (precision == QUDA_DOUBLE_PRECISION) {
              ((double *)longlink[dir])[i * gauge_site_size + j] = 0.0;
              ((double *)longlink[dir])[i * gauge_site_size + j + 1] = 0.0;
            } else {
              ((float *)longlink[dir])[i * gauge_site_size + j] = 0.0;
              ((float *)longlink[dir])[i * gauge_site_size + j + 1] = 0.0;
            }
          }
        }
      }
    }
  }
}

template <typename Float>
static void constructCloverField(Float *res, double norm, double diag) {

  Float c = 2.0 * norm / RAND_MAX;

  for(int i = 0; i < V; i++) {
    for (int j = 0; j < 72; j++) {
      res[i*72 + j] = c*rand() - norm;
    }

    //impose clover symmetry on each chiral block
    for (int ch=0; ch<2; ch++) {
      res[i*72 + 3 + 36*ch] = -res[i*72 + 0 + 36*ch];
      res[i*72 + 4 + 36*ch] = -res[i*72 + 1 + 36*ch];
      res[i*72 + 5 + 36*ch] = -res[i*72 + 2 + 36*ch];
      res[i*72 + 30 + 36*ch] = -res[i*72 + 6 + 36*ch];
      res[i*72 + 31 + 36*ch] = -res[i*72 + 7 + 36*ch];
      res[i*72 + 32 + 36*ch] = -res[i*72 + 8 + 36*ch];
      res[i*72 + 33 + 36*ch] = -res[i*72 + 9 + 36*ch];
      res[i*72 + 34 + 36*ch] = -res[i*72 + 16 + 36*ch];
      res[i*72 + 35 + 36*ch] = -res[i*72 + 17 + 36*ch];
    }

    for (int j = 0; j<6; j++) {
      res[i*72 + j] += diag;
      res[i*72 + j+36] += diag;
    }
  }
}

/*void strong_check(void *spinorRef, void *spinorGPU, int len, QudaPrecision prec) {
  printf("Reference:\n");
  printSpinorElement(spinorRef, 0, prec); printf("...\n");
  printSpinorElement(spinorRef, len-1, prec); printf("\n");

  printf("\nCUDA:\n");
  printSpinorElement(spinorGPU, 0, prec); printf("...\n");
  printSpinorElement(spinorGPU, len-1, prec); printf("\n");

  compare_spinor(spinorRef, spinorGPU, len, prec);
  }*/

template <typename Float>
static void checkGauge(Float **oldG, Float **newG, double epsilon) {

  const int fail_check = 17;
  int fail[4][fail_check];
  int iter[4][18];
  for (int d=0; d<4; d++) for (int i=0; i<fail_check; i++) fail[d][i] = 0;
  for (int d=0; d<4; d++) for (int i=0; i<18; i++) iter[d][i] = 0;

  for (int d=0; d<4; d++) {
    for (int eo=0; eo<2; eo++) {
      for (int i=0; i<Vh; i++) {
	int ga_idx = (eo*Vh+i);
	for (int j=0; j<18; j++) {
	  double diff = fabs(newG[d][ga_idx*18+j] - oldG[d][ga_idx*18+j]);/// fabs(oldG[d][ga_idx*18+j]);

	  for (int f=0; f<fail_check; f++) if (diff > pow(10.0,-(f+1))) fail[d][f]++;
	  if (diff > epsilon) iter[d][j]++;
	}
      }
    }
  }

  printf("Component fails (X, Y, Z, T)\n");
  for (int i=0; i<18; i++) printf("%d fails = (%8d, %8d, %8d, %8d)\n", i, iter[0][i], iter[1][i], iter[2][i], iter[3][i]);

  printf("\nDeviation Failures = (X, Y, Z, T)\n");
  for (int f=0; f<fail_check; f++) {
    printf("%e Failures = (%9d, %9d, %9d, %9d) = (%6.5f, %6.5f, %6.5f, %6.5f)\n", pow(10.0, -(f + 1)), fail[0][f],
           fail[1][f], fail[2][f], fail[3][f], fail[0][f] / (double)(V * 18), fail[1][f] / (double)(V * 18),
           fail[2][f] / (double)(V * 18), fail[3][f] / (double)(V * 18));
  }

}

void check_gauge(void **oldG, void **newG, double epsilon, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION)
    checkGauge((double**)oldG, (double**)newG, epsilon);
  else
    checkGauge((float**)oldG, (float**)newG, epsilon);
}

void createSiteLinkCPU(void **link, QudaPrecision precision, int phase)
{

  if (precision == QUDA_DOUBLE_PRECISION) {
    constructUnitaryGaugeField((double**)link);
  }else {
    constructUnitaryGaugeField((float**)link);
  }

  if(phase){

    for(int i=0;i < V;i++){
      for(int dir =XUP; dir <= TUP; dir++){
	int idx = i;
	int oddBit =0;
	if (i >= Vh) {
	  idx = i - Vh;
	  oddBit = 1;
	}

	int X1 = Z[0];
	int X2 = Z[1];
	int X3 = Z[2];
	int X4 = Z[3];

	int full_idx = fullLatticeIndex(idx, oddBit);
	int i4 = full_idx /(X3*X2*X1);
	int i3 = (full_idx - i4*(X3*X2*X1))/(X2*X1);
	int i2 = (full_idx - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
        int i1 = full_idx - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;

        double coeff= 1.0;
	switch(dir){
	case XUP:
	  if ( (i4 & 1) != 0){
	    coeff *= -1;
	  }
	  break;

	case YUP:
	  if ( ((i4+i1) & 1) != 0){
	    coeff *= -1;
	  }
	  break;

	case ZUP:
	  if ( ((i4+i1+i2) & 1) != 0){
	    coeff *= -1;
	  }
	  break;

        case TUP:
          if (last_node_in_t() && i4 == (X4 - 1)) { coeff *= -1; }
          break;

	default:
	  printf("ERROR: wrong dir(%d)\n", dir);
	  exit(1);
	}

        if (precision == QUDA_DOUBLE_PRECISION){
	  //double* mylink = (double*)link;
	  //mylink = mylink + (4*i + dir)*gauge_site_size;
	  double* mylink = (double*)link[dir];
	  mylink = mylink + i*gauge_site_size;

	  mylink[12] *= coeff;
	  mylink[13] *= coeff;
	  mylink[14] *= coeff;
	  mylink[15] *= coeff;
	  mylink[16] *= coeff;
	  mylink[17] *= coeff;

        }else{
	  //float* mylink = (float*)link;
	  //mylink = mylink + (4*<i + dir)*gauge_site_size;
	  float* mylink = (float*)link[dir];
	  mylink = mylink + i*gauge_site_size;

          mylink[12] *= coeff;
	  mylink[13] *= coeff;
	  mylink[14] *= coeff;
	  mylink[15] *= coeff;
	  mylink[16] *= coeff;
	  mylink[17] *= coeff;
        }
      }
    }
  }

#if 1
  for(int dir= 0;dir < 4;dir++){
    for(int i=0;i< V*gauge_site_size;i++){
      if (precision ==QUDA_SINGLE_PRECISION){
	float* f = (float*)link[dir];
	if (f[i] != f[i] || (fabsf(f[i]) > 1.e+3) ){
	  fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
	  exit(1);
	}
      }else{
	double* f = (double*)link[dir];
	if (f[i] != f[i] || (fabs(f[i]) > 1.e+3)){
	  fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
	  exit(1);
	}
      }
    }
  }
#endif

  return;
}

void constructRandomSpinorSource(void *v, int nSpin, int nColor, QudaPrecision precision, const int *const x, quda::RNG &rng)
{
  quda::ColorSpinorParam param;
  param.v = v;
  param.nColor = nColor;
  param.nSpin = nSpin;
  param.setPrecision(precision);
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param.nDim = 4;
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  for (int d = 0; d < 4; d++) param.x[d] = x[d];
  quda::cpuColorSpinorField spinor_in(param);
  quda::spinorNoise(spinor_in, rng, QUDA_NOISE_UNIFORM);
}

template <typename Float>
int compareLink(Float **linkA, Float **linkB, int len) {
  const int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[18];
  for (int i=0; i<18; i++) iter[i] = 0;

  for(int dir=0;dir < 4; dir++){
    for (int i=0; i<len; i++) {
      for (int j=0; j<18; j++) {
	int is = i*18+j;
	double diff = fabs(linkA[dir][is]-linkB[dir][is]);
	for (int f=0; f<fail_check; f++)
	  if (diff > pow(10.0,-(f+1))) fail[f]++;
	//if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
	if (diff > 1e-3) iter[j]++;
      }
    }
  }

  for (int i=0; i<18; i++) printfQuda("%d fails = %d\n", i, iter[i]);

  int accuracy_level = 0;
  for(int f =0; f < fail_check; f++){
    if(fail[f] == 0){
      accuracy_level =f;
    }
  }

  for (int f=0; f<fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], 4*len*18, fail[f] / (double)(4*len*18));
  }

  return accuracy_level;
}

static int
compare_link(void **linkA, void **linkB, int len, QudaPrecision precision)
{
  int ret;

  if (precision == QUDA_DOUBLE_PRECISION) {
    ret = compareLink((double**)linkA, (double**)linkB, len);
  } else {
    ret = compareLink((float**)linkA, (float**)linkB, len);
  }

  return ret;
}


// X indexes the lattice site
static void printLinkElement(void *link, int X, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION){
    for(int i=0; i < 3;i++){
      printVector((double*)link+ X*gauge_site_size + i*6);
    }

  }
  else{
    for(int i=0;i < 3;i++){
      printVector((float*)link+X*gauge_site_size + i*6);
    }
  }
}

int strong_check_link(void **linkA, const char *msgA, void **linkB, const char *msgB, int len, QudaPrecision prec)
{
  printfQuda("%s\n", msgA);
  printLinkElement(linkA[0], 0, prec);
  printfQuda("\n");
  printLinkElement(linkA[0], 1, prec);
  printfQuda("...\n");
  printLinkElement(linkA[3], len - 1, prec);
  printfQuda("\n");

  printfQuda("\n%s\n", msgB);
  printLinkElement(linkB[0], 0, prec);
  printfQuda("\n");
  printLinkElement(linkB[0], 1, prec);
  printfQuda("...\n");
  printLinkElement(linkB[3], len - 1, prec);
  printfQuda("\n");

  int ret = compare_link(linkA, linkB, len, prec);
  return ret;
}

void createMomCPU(void *mom, QudaPrecision precision)
{
  void* temp;

  size_t gauge_data_type_size = (precision == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  temp = malloc(4 * V * gauge_site_size * gauge_data_type_size);
  if (temp == NULL){
    fprintf(stderr, "Error: malloc failed for temp in function %s\n", __FUNCTION__);
    exit(1);
  }

  for(int i=0;i < V;i++){
    if (precision == QUDA_DOUBLE_PRECISION){
      for(int dir=0;dir < 4;dir++){
        double *thismom = (double *)mom;
        for(int k=0; k < mom_site_size; k++){
          thismom[(4 * i + dir) * mom_site_size + k] = 1.0 * rand() / RAND_MAX;
          if (k==mom_site_size-1) thismom[ (4*i+dir)*mom_site_size + k ]= 0.0;
        }
      }
    }else{
      for(int dir=0;dir < 4;dir++){
	float* thismom=(float*)mom;
	for(int k=0; k < mom_site_size; k++){
          thismom[(4 * i + dir) * mom_site_size + k] = 1.0 * rand() / RAND_MAX;
          if (k==mom_site_size-1) thismom[ (4*i+dir)*mom_site_size + k ]= 0.0;
        }
      }
    }
  }

  free(temp);
  return;
}

void
createHwCPU(void* hw,  QudaPrecision precision)
{
  for(int i=0;i < V;i++){
    if (precision == QUDA_DOUBLE_PRECISION){
      for(int dir=0;dir < 4;dir++){
	double* thishw = (double*)hw;
	for(int k=0; k < hw_site_size; k++){
	  thishw[ (4*i+dir)*hw_site_size + k ]= 1.0* rand() /RAND_MAX;
	}
      }
    }else{
      for(int dir=0;dir < 4;dir++){
	float* thishw=(float*)hw;
	for(int k=0; k < hw_site_size; k++){
	  thishw[ (4*i+dir)*hw_site_size + k ]= 1.0* rand() /RAND_MAX;
	}
      }
    }
  }

  return;
}


template <typename Float>
int compare_mom(Float *momA, Float *momB, int len) {
  const int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[mom_site_size];
  for (int i=0; i<mom_site_size; i++) iter[i] = 0;

  for (int i=0; i<len; i++) {
    for (int j=0; j<mom_site_size-1; j++) {
      int is = i*mom_site_size+j;
      double diff = fabs(momA[is]-momB[is]);
      for (int f=0; f<fail_check; f++)
	if (diff > pow(10.0,-(f+1))) fail[f]++;
      //if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
      if (diff > 1e-3) iter[j]++;
    }
  }

  int accuracy_level = 0;
  for(int f =0; f < fail_check; f++){
    if(fail[f] == 0){
      accuracy_level =f+1;
    }
  }

  for (int i=0; i<mom_site_size; i++) printfQuda("%d fails = %d\n", i, iter[i]);

  for (int f=0; f<fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], len*9, fail[f]/(double)(len*9));
  }

  return accuracy_level;
}

static void printMomElement(void *mom, int X, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION){
    double* thismom = ((double*)mom)+ X*mom_site_size;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
  }else{
    float* thismom = ((float*)mom)+ X*mom_site_size;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
  }
}
int strong_check_mom(void *momA, void *momB, int len, QudaPrecision prec)
{
  printfQuda("mom:\n");
  printMomElement(momA, 0, prec);
  printfQuda("\n");
  printMomElement(momA, 1, prec);
  printfQuda("\n");
  printMomElement(momA, 2, prec);
  printfQuda("\n");
  printMomElement(momA, 3, prec);
  printfQuda("...\n");

  printfQuda("\nreference mom:\n");
  printMomElement(momB, 0, prec);
  printfQuda("\n");
  printMomElement(momB, 1, prec);
  printfQuda("\n");
  printMomElement(momB, 2, prec);
  printfQuda("\n");
  printMomElement(momB, 3, prec);
  printfQuda("\n");

  int ret;
  if (prec == QUDA_DOUBLE_PRECISION){
    ret = compare_mom((double*)momA, (double*)momB, len);
  }else{
    ret = compare_mom((float*)momA, (float*)momB, len);
  }

  return ret;
}

static struct timeval startTime;

void stopwatchStart() { gettimeofday(&startTime, NULL); }

double stopwatchReadSeconds()
{
  struct timeval endTime;
  gettimeofday(&endTime, NULL);

  long ds = endTime.tv_sec - startTime.tv_sec;
  long dus = endTime.tv_usec - startTime.tv_usec;
  return ds + 0.000001*dus;
}

int dimPartitioned(int dim) { return ((gridsize_from_cmdline[dim] > 1) || dim_partitioned[dim]); }

void constructHostGaugeField(void **gauge, QudaGaugeParam &gauge_param, int argc, char **argv) {
  // Allocate space on the host
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gauge_site_size*host_gauge_data_type_size);
  }
  
  // 0 = random SU(3)
  // 1 = unit gauge
  // 2 = supplied field
  int construct_type = 0; 
  if (strcmp(latfile,"")) {  
    // load in the command line supplied gauge field using QIO and LIME
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_type = 2;
  } else { 
    if (unit_gauge) construct_type = 1;
    else construct_type = 0;
  }
  constructQudaGaugeField(gauge, construct_type, gauge_param.cpu_prec, &gauge_param);
}

void constructQudaGaugeField(void **gauge, int type, QudaPrecision precision, QudaGaugeParam *param) {
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION) constructUnitGaugeField((double**)gauge, param);
    else constructUnitGaugeField((float**)gauge, param);
  } else if (type == 1) {
    if (precision == QUDA_DOUBLE_PRECISION) constructGaugeField((double**)gauge, param);
    else constructGaugeField((float**)gauge, param);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) applyGaugeFieldScaling((double**)gauge, Vh, param);
    else
      applyGaugeFieldScaling((float **)gauge, Vh, param);
  }
}

void constructHostCloverField(void *clover, void *clover_inv, QudaInvertParam &inv_param) {
  double norm = 0.01; // clover components are random numbers in the range (-norm, norm)
  double diag = 1.0; // constant added to the diagonal
  
  size_t cSize = inv_param.clover_cpu_prec;
  clover = malloc(V*clover_site_size*cSize);
  clover_inv = malloc(V*clover_site_size*cSize);
  if (!compute_clover) constructQudaCloverField(clover, norm, diag, inv_param.clover_cpu_prec);
  
  inv_param.compute_clover = compute_clover;
  if (compute_clover) inv_param.return_clover = 1;
  inv_param.compute_clover_inverse = 1;
  inv_param.return_clover_inverse = 1;
}

void constructQudaCloverField(void *clover, double norm, double diag, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) constructCloverField((double *)clover, norm, diag);
  else constructCloverField((float *)clover, norm, diag);
}


void setWilsonGaugeParam(QudaGaugeParam &gauge_param) 
{
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.tadpole_coeff = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;

  gauge_param.reconstruct = link_recon;  
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0; 
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif  
}

void setInvertParam(QudaInvertParam &inv_param)
{
  // Set dslash type
  inv_param.dslash_type = dslash_type;

  // Use kappa or mass normalisation
  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass));
    if (dslash_type == QUDA_LAPLACE_DSLASH) inv_param.kappa = 1.0 / (8 + mass);
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5 / kappa - (1.0 + 3.0 / anisotropy);
    if (dslash_type == QUDA_LAPLACE_DSLASH) inv_param.mass = 1.0 / kappa - 8.0;
  }
  printfQuda("Kappa = %.8f Mass = %.8f\n", inv_param.kappa, inv_param.mass);

  // Use 3D or 4D laplace
  inv_param.laplace3D = laplace3D;

  // Some fermion specific parameters
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || 
	     dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || 
	     dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    inv_param.m5 = m5;
    kappa5 = 0.5 / (5 + inv_param.m5);
    inv_param.Ls = Lsdim;
    for (int k = 0; k < Lsdim; k++) { // for mobius only
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = b5;
      inv_param.c_5[k] = c5;
    }
  } else {
    inv_param.Ls = 1;
  }

  // Set clover specific parameters
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  // Offsets used only by multi-shift solver
  inv_param.num_offset = 12;
  double offset[12] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];
  
  // General parameter setup
  inv_param.inv_type = inv_type;
  inv_param.solution_type = solution_type;
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  inv_param.pipeline = pipeline;
  inv_param.Nsteps = 2;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.ca_basis = ca_basis;
  inv_param.ca_lambda_min = ca_lambda_min;
  inv_param.ca_lambda_max = ca_lambda_max;
  inv_param.tol = tol;
  inv_param.tol_restart = tol_restart; 
  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ? static_cast<QudaResidualType_s> ( inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ? static_cast<QudaResidualType_s> (inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : inv_param.residual_type;

  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;
  inv_param.use_alternative_reliable = alternative_reliable;
  inv_param.use_sloppy_partial_accumulator = 0;
  inv_param.solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param.max_res_increase = 1;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;
    
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = mg_verbosity[0];
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;
  //inv_param.cuda_prec_ritz = cuda_prec_ritz;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.sp_pad = 0; 
  inv_param.cl_pad = 0; 

  inv_param.verbosity = verbosity;

  inv_param.extlib_type = solver_ext_lib;
}

// Parameters defining the eigensolver
void setEigParam(QudaEigParam &eig_param)
{
  eig_param.eig_type = eig_type;
  eig_param.spectrum = eig_spectrum;
  if ((eig_type == QUDA_EIG_TR_LANCZOS || eig_type == QUDA_EIG_IR_LANCZOS)
      && !(eig_spectrum == QUDA_SPECTRUM_LR_EIG || eig_spectrum == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to Lanczos type solver");
  }

  // The solver will exit when nConv extremal eigenpairs have converged
  if (eig_nConv < 0) {
    eig_param.nConv = eig_nEv;
    eig_nConv = eig_nEv;
  } else {
    eig_param.nConv = eig_nConv;
  }

  eig_param.nEv = eig_nEv;
  eig_param.nKr = eig_nKr;
  eig_param.tol = eig_tol;
  eig_param.batched_rotate = eig_batched_rotate;
  eig_param.require_convergence = eig_require_convergence ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.check_interval = eig_check_interval;
  eig_param.max_restarts = eig_max_restarts;
  eig_param.cuda_prec_ritz = cuda_prec;

  eig_param.use_norm_op = eig_use_normop ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.use_dagger = eig_use_dagger ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.compute_svd = eig_compute_svd ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  if (eig_compute_svd) {
    eig_param.use_dagger = QUDA_BOOLEAN_FALSE;
    eig_param.use_norm_op = QUDA_BOOLEAN_TRUE;
  }

  eig_param.use_poly_acc = eig_use_poly_acc ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  eig_param.poly_deg = eig_poly_deg;
  eig_param.a_min = eig_amin;
  eig_param.a_max = eig_amax;

  eig_param.arpack_check = eig_arpack_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  strcpy(eig_param.arpack_logfile, eig_arpack_logfile);
  strcpy(eig_param.QUDA_logfile, eig_QUDA_logfile);

  strcpy(eig_param.vec_infile, eig_vec_infile);
  strcpy(eig_param.vec_outfile, eig_vec_outfile);
}

void setMultigridParam(QudaMultigridParam &mg_param)
{
  QudaInvertParam &inv_param = *mg_param.invert_param; // this will be used to setup SolverParam parent in MGParam class

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION; //DMH

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j = 0; j < 4; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    for (int j = 4; j < QUDA_MAX_DIM; j++) mg_param.geo_block_size[i][j] = 1;
    mg_param.use_eig_solver[i] = mg_eig[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.verbosity[i] = mg_verbosity[i];
    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];
    mg_param.setup_tol[i] = setup_tol[i];
    mg_param.setup_maxiter[i] = setup_maxiter[i];

    // Basis to use for CA-CGN(E/R) setup
    mg_param.setup_ca_basis[i] = setup_ca_basis[i];

    // Basis size for CACG setup
    mg_param.setup_ca_basis_size[i] = setup_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis setup
    mg_param.setup_ca_lambda_min[i] = setup_ca_lambda_min[i];
    mg_param.setup_ca_lambda_max[i] = setup_ca_lambda_max[i];

    mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i]; // default to 24 vectors if not set DMH
    mg_param.n_block_ortho[i] = n_block_ortho[i];    // number of times to Gram-Schmidt
    mg_param.precision_null[i] = prec_null; // precision to store the null-space basis
    mg_param.smoother_halo_precision[i] = smoother_halo_prec; // precision of the halo exchange in the smoother
    mg_param.nu_pre[i] = nu_pre[i];
    mg_param.nu_post[i] = nu_post[i];
    mg_param.mu_factor[i] = mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];

    // Basis to use for CA-CGN(E/R) coarse solver
    mg_param.coarse_solver_ca_basis[i] = coarse_solver_ca_basis[i];

    // Basis size for CACG coarse solver/
    mg_param.coarse_solver_ca_basis_size[i] = coarse_solver_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis
    mg_param.coarse_solver_ca_lambda_min[i] = coarse_solver_ca_lambda_min[i];
    mg_param.coarse_solver_ca_lambda_max[i] = coarse_solver_ca_lambda_max[i];

    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i];

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    mg_param.smoother_solve_type[i] = smoother_solve_type[i];

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = schwarz_cycle[i];

    // Set set coarse_grid_solution_type: this defines which linear
    // system we are solving on a given level
    // * QUDA_MAT_SOLUTION - we are solving the full system and inject
    //   a full field into coarse grid
    // * QUDA_MATPC_SOLUTION - we are solving the e/o-preconditioned
    //   system, and only inject single parity field into coarse grid
    //
    // Multiple possible scenarios here
    //
    // 1. **Direct outer solver and direct smoother**: here we use
    // full-field residual coarsening, and everything involves the
    // full system so coarse_grid_solution_type = QUDA_MAT_SOLUTION
    //
    // 2. **Direct outer solver and preconditioned smoother**: here,
    // only the smoothing uses e/o preconditioning, so
    // coarse_grid_solution_type = QUDA_MAT_SOLUTION_TYPE.
    // We reconstruct the full residual prior to coarsening after the
    // pre-smoother, and then need to project the solution for post
    // smoothing.
    //
    // 3. **Preconditioned outer solver and preconditioned smoother**:
    // here we use single-parity residual coarsening throughout, so
    // coarse_grid_solution_type = QUDA_MATPC_SOLUTION.  This is a bit
    // questionable from a theoretical point of view, since we don't
    // coarsen the preconditioned operator directly, rather we coarsen
    // the full operator and preconditioned that, but it just works.
    // This is the optimal combination in general for Wilson-type
    // operators: although there is an occasional increase in
    // iteration or two), by working completely in the preconditioned
    // space, we save the cost of reconstructing the full residual
    // from the preconditioned smoother, and re-projecting for the
    // subsequent smoother, as well as reducing the cost of the
    // ancillary blas operations in the coarse-grid solve.
    //
    // Note, we cannot use preconditioned outer solve with direct
    // smoother
    //
    // Finally, we have to treat the top level carefully: for all
    // other levels the entry into and out of the grid will be a
    // full-field, which we can then work in Schur complement space or
    // not (e.g., freedom to choose coarse_grid_solution_type).  For
    // the top level, if the outer solver is for the preconditioned
    // system, then we must use preconditoning, e.g., option 3.) above.

    if (i == 0) { // top-level treatment
      if (coarse_solve_type[0] != solve_type)
        errorQuda("Mismatch between top-level MG solve type %d and outer solve type %d", coarse_solve_type[0], solve_type);

      if (solve_type == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", solve_type);
      }

    } else {

      if (coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", coarse_solve_type[i]);
      }
    }

    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = solver_location[i];
    mg_param.setup_location[i] = setup_location[i];
  }

  // whether to run GPU setup but putting temporaries into mapped (slow CPU) memory
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_FALSE;

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2; //DMH

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_TRUE :  QUDA_BOOLEAN_FALSE;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_TRUE :  QUDA_BOOLEAN_FALSE;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES
    : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_TRUE :  QUDA_BOOLEAN_FALSE;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_low_mode_check = low_mode_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_oblique_proj_check = oblique_proj_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // Is NOT a staggered solve
  mg_param.is_staggered = QUDA_BOOLEAN_FALSE;

  // set file i/o parameters
  for (int i = 0; i < mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], mg_vec_infile[i]);
    strcpy(mg_param.vec_outfile[i], mg_vec_outfile[i]);
    if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_TRUE;
    if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_TRUE;
  }

  mg_param.coarse_guess = mg_eig_coarse_guess ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = reliable_delta;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;
}

void setMultigridInvertParam(QudaInvertParam &inv_param) {
  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = mg_verbosity[0];


  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.pipeline = pipeline;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.tol = tol;

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}

// Parameters defining the eigensolver
void setMultigridEigParam(QudaEigParam &mg_eig_param, int level)
{
  mg_eig_param.eig_type = mg_eig_type[level];
  mg_eig_param.spectrum = mg_eig_spectrum[level];
  if ((mg_eig_type[level] == QUDA_EIG_TR_LANCZOS || mg_eig_type[level] == QUDA_EIG_IR_LANCZOS)
      && !(mg_eig_spectrum[level] == QUDA_SPECTRUM_LR_EIG || mg_eig_spectrum[level] == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to the a Lanczos type solver");
  }

  mg_eig_param.nEv = mg_eig_nEv[level];
  mg_eig_param.nKr = mg_eig_nKr[level];
  mg_eig_param.nConv = nvec[level];
  mg_eig_param.batched_rotate = mg_eig_batched_rotate[level];
  mg_eig_param.require_convergence = mg_eig_require_convergence[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_eig_param.tol = mg_eig_tol[level];
  mg_eig_param.check_interval = mg_eig_check_interval[level];
  mg_eig_param.max_restarts = mg_eig_max_restarts[level];
  mg_eig_param.cuda_prec_ritz = cuda_prec;

  mg_eig_param.compute_svd = QUDA_BOOLEAN_FALSE;
  mg_eig_param.use_norm_op = mg_eig_use_normop[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_eig_param.use_dagger = mg_eig_use_dagger[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_eig_param.use_poly_acc = mg_eig_use_poly_acc[level] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_eig_param.poly_deg = mg_eig_poly_deg[level];
  mg_eig_param.a_min = mg_eig_amin[level];
  mg_eig_param.a_max = mg_eig_amax[level];

  // set file i/o parameters
  // Give empty strings, Multigrid will handle IO.
  strcpy(mg_eig_param.vec_infile, "");
  strcpy(mg_eig_param.vec_outfile, "");

  strcpy(mg_eig_param.QUDA_logfile, eig_QUDA_logfile);
}

void setContractInvertParam(QudaInvertParam &inv_param)
{
  inv_param.Ls = 1;
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;

  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  // Quda performs contractions in Degrand-Rossi gamma basis,
  // but the user may suppy vectors in any supported order.
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;
}

void setStaggeredGaugeParam(QudaGaugeParam &gauge_param)
{
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_sloppy = prec_sloppy;
  gauge_param.cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  gauge_param.cuda_prec_precondition = prec_precondition;

  gauge_param.anisotropy = 1.0;

  // For HISQ, this must always be set to 1.0, since the tadpole
  // correction is baked into the coefficients for the first fattening.
  // The tadpole doesn't mean anything for the second fattening
  // since the input fields are unitarized.
  gauge_param.tadpole_coeff = 1.0;

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.scale = -1.0 / 24.0;
    if (eps_naik != 0) { gauge_param.scale *= (1.0 + eps_naik); }
  } else {
    gauge_param.scale = 1.0;
  }
  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.type = QUDA_WILSON_LINKS;

  gauge_param.ga_pad = 0;

#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;
#endif
}

void setStaggeredInvertParam(QudaInvertParam &inv_param)
{
  // Solver params
  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.mass = mass;
  inv_param.kappa = kappa = 1.0 / (8.0 + mass); // for Laplace operator
  inv_param.laplace3D = laplace3D;              // for Laplace operator

  // outer solver parameters
  inv_param.inv_type = inv_type; 
  inv_param.tol = tol;
  inv_param.tol_restart = tol_restart;
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;
  inv_param.use_alternative_reliable = alternative_reliable;
  inv_param.use_sloppy_partial_accumulator = false;
  inv_param.solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param.pipeline = pipeline;

  inv_param.Ls = 1; // Nsrc

  if (tol_hq == 0 && tol == 0) {
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) :
    inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) :
    inv_param.residual_type;
  inv_param.heavy_quark_check = (inv_param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 5 : 0);

  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param.Nsteps = 2;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  inv_param.cuda_prec_precondition = prec_precondition;

  // Specify Krylov sub-size for GCR, BICGSTAB(L), basis size for CA-CG, CA-GCR
  inv_param.gcrNkrylov = gcrNkrylov;

  // Specify basis for CA-CG, lambda min/max for Chebyshev basis
  //   lambda_max < lambda_max . use power iters to generate
  inv_param.ca_basis = ca_basis;
  inv_param.ca_lambda_min = ca_lambda_min;
  inv_param.ca_lambda_max = ca_lambda_max;

  inv_param.solution_type = solution_type;
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = prec;
  inv_param.cuda_prec_sloppy = prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.dslash_type = dslash_type;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  int tmpint = MAX(ydim * zdim * tdim, xdim * zdim * tdim);
  tmpint = MAX(tmpint, xdim * ydim * tdim);
  tmpint = MAX(tmpint, xdim * ydim * zdim);

  inv_param.sp_pad = tmpint;
}

void setStaggeredMultigridParam(QudaMultigridParam &mg_param)
{
  QudaInvertParam &inv_param = *mg_param.invert_param; // this will be used to setup SolverParam parent in MGParam class

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION; //DMH

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.is_staggered = QUDA_BOOLEAN_TRUE; //DMH

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i = 0; i < mg_param.n_level; i++) {
    for (int j = 0; j < 4; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    for (int j = 4; j < QUDA_MAX_DIM; j++) mg_param.geo_block_size[i][j] = 1;
    mg_param.use_eig_solver[i] = mg_eig[i] ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    mg_param.verbosity[i] = mg_verbosity[i];
    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];
    mg_param.setup_tol[i] = setup_tol[i];
    mg_param.setup_maxiter[i] = setup_maxiter[i];

    // Basis to use for CA-CGN(E/R) setup
    mg_param.setup_ca_basis[i] = setup_ca_basis[i];

    // Basis size for CACG setup
    mg_param.setup_ca_basis_size[i] = setup_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis setup
    mg_param.setup_ca_lambda_min[i] = setup_ca_lambda_min[i];
    mg_param.setup_ca_lambda_max[i] = setup_ca_lambda_max[i];

    mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = (i == 0) ? 24 : nvec[i] == 0 ? 96 : nvec[i]; // default to 96 vectors if not set DMH
    mg_param.n_block_ortho[i] = n_block_ortho[i]; // number of times to Gram-Schmidt
    mg_param.precision_null[i] = prec_null; // precision to store the null-space basis
    mg_param.smoother_halo_precision[i] = smoother_halo_prec; // precision of the halo exchange in the smoother
    mg_param.nu_pre[i] = nu_pre[i];
    mg_param.nu_post[i] = nu_post[i];
    mg_param.mu_factor[i] = mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];

    // Basis to use for CA-CGN(E/R) coarse solver
    mg_param.coarse_solver_ca_basis[i] = coarse_solver_ca_basis[i];

    // Basis size for CACG coarse solver/
    mg_param.coarse_solver_ca_basis_size[i] = coarse_solver_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis
    mg_param.coarse_solver_ca_lambda_min[i] = coarse_solver_ca_lambda_min[i];
    mg_param.coarse_solver_ca_lambda_max[i] = coarse_solver_ca_lambda_max[i];

    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i];

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    mg_param.smoother_solve_type[i] = smoother_solve_type[i];

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = schwarz_cycle[i];

    // Set set coarse_grid_solution_type: this defines which linear
    // system we are solving on a given level
    // * QUDA_MAT_SOLUTION - we are solving the full system and inject
    //   a full field into coarse grid
    // * QUDA_MATPC_SOLUTION - we are solving the e/o-preconditioned
    //   system, and only inject single parity field into coarse grid
    //
    // Multiple possible scenarios here
    //
    // 1. **Direct outer solver and direct smoother**: here we use
    // full-field residual coarsening, and everything involves the
    // full system so coarse_grid_solution_type = QUDA_MAT_SOLUTION
    //
    // 2. **Direct outer solver and preconditioned smoother**: here,
    // only the smoothing uses e/o preconditioning, so
    // coarse_grid_solution_type = QUDA_MAT_SOLUTION_TYPE.
    // We reconstruct the full residual prior to coarsening after the
    // pre-smoother, and then need to project the solution for post
    // smoothing.
    //
    // 3. **Preconditioned outer solver and preconditioned smoother**:
    // here we use single-parity residual coarsening throughout, so
    // coarse_grid_solution_type = QUDA_MATPC_SOLUTION.  This is a bit
    // questionable from a theoretical point of view, since we don't
    // coarsen the preconditioned operator directly, rather we coarsen
    // the full operator and preconditioned that, but it just works.
    // This is the optimal combination in general for Wilson-type
    // operators: although there is an occasional increase in
    // iteration or two), by working completely in the preconditioned
    // space, we save the cost of reconstructing the full residual
    // from the preconditioned smoother, and re-projecting for the
    // subsequent smoother, as well as reducing the cost of the
    // ancillary blas operations in the coarse-grid solve.
    //
    // Note, we cannot use preconditioned outer solve with direct
    // smoother
    //
    // Finally, we have to treat the top level carefully: for all
    // other levels the entry into and out of the grid will be a
    // full-field, which we can then work in Schur complement space or
    // not (e.g., freedom to choose coarse_grid_solution_type).  For
    // the top level, if the outer solver is for the preconditioned
    // system, then we must use preconditoning, e.g., option 3.) above.

    if (i == 0) { // top-level treatment
      if (coarse_solve_type[0] != solve_type)
        errorQuda("Mismatch between top-level MG solve type %d and outer solve type %d", coarse_solve_type[0], solve_type);

      if (solve_type == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", solve_type);
      }
    } else {
      if (coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", coarse_solve_type[i]);
      }
    }

    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = solver_location[i];
    mg_param.setup_location[i] = setup_location[i];
    nu_pre[i] = 2; //DMH
    nu_post[i] = 2; //DMH
  }

  // whether to run GPU setup but putting temporaries into mapped (slow CPU) memory
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_FALSE;

  // coarsening the spin on the first restriction is undefined for staggered fields.
  mg_param.spin_block_size[0] = 0; //DMH

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_low_mode_check = low_mode_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg_param.run_oblique_proj_check = oblique_proj_check ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // set file i/o parameters
  for (int i = 0; i < mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], mg_vec_infile[i]);
    strcpy(mg_param.vec_outfile[i], mg_vec_outfile[i]);
    if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_TRUE;
    if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_TRUE;
  }

  mg_param.coarse_guess = mg_eig_coarse_guess ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = reliable_delta;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;
}

void setDeflatedInvertParam(QudaInvertParam &inv_param) {
  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

//  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  // inv_param.solution_type = QUDA_MATPC_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  if (inv_type != QUDA_EIGCG_INVERTER && inv_type != QUDA_INC_EIGCG_INVERTER && inv_type != QUDA_GMRESDR_INVERTER)
    errorQuda("Unknown deflated solver type %d.", inv_type);

  //! For deflated solvers only:
  inv_param.inv_type = inv_type;
  inv_param.tol      = tol;
  inv_param.tol_hq   = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param.rhs_idx  = 0;

  inv_param.nev = nev;
  inv_param.max_search_dim = max_search_dim;
  inv_param.deflation_grid = deflation_grid;
  inv_param.tol_restart = tol_restart;
  inv_param.eigcg_max_restarts = eigcg_max_restarts;
  inv_param.max_restart_num = max_restart_num;
  inv_param.inc_tol = inc_tol;
  inv_param.eigenval_tol = eigenval_tol;


  if(inv_param.inv_type == QUDA_EIGCG_INVERTER || inv_param.inv_type == QUDA_INC_EIGCG_INVERTER ){
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  }else if(inv_param.inv_type == QUDA_GMRESDR_INVERTER) {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    inv_param.tol_restart = 0.0;//restart is not requested...
  }

  inv_param.cuda_prec_ritz = cuda_prec_ritz;
  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;

  inv_param.inv_type_precondition = precon_type;
  inv_param.gcrNkrylov = 6;

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-1;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-2;
  inv_param.maxiter_precondition = 10;
  inv_param.omega = 1.0;

  inv_param.extlib_type = solver_ext_lib;
}

void setDeflationParam(QudaEigParam &df_param) {

  df_param.import_vectors = QUDA_BOOLEAN_FALSE;
  df_param.run_verify     = QUDA_BOOLEAN_FALSE;

  df_param.nk             = df_param.invert_param->nev;
  df_param.np             = df_param.invert_param->nev*df_param.invert_param->deflation_grid;
  df_param.extlib_type    = deflation_ext_lib;

  df_param.cuda_prec_ritz = prec_ritz;
  df_param.location       = location_ritz;
  df_param.mem_type_ritz  = mem_type_ritz;

  // set file i/o parameters
  strcpy(df_param.vec_infile, eig_vec_infile);
  strcpy(df_param.vec_outfile, eig_vec_outfile);
}

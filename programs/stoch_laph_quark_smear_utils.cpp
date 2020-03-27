#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <command_line_params.h>
#include <host_utils.h>
#include <misc.h>

#include <stoch_laph_quark_smear.h>

using namespace quda;

//!< Profiler for laphInvertSourcesQuda
static TimeProfile profileLaphInvert("laphInvertSourcesQuda");

void display_driver_info() {
  printfQuda("Running the stochastic laph quark smear driver:\n");
    
  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy solve_type S_dimension T_dimension Ls_dimension   dslash_type  normalization\n");
  printfQuda("%6s   %6s          %d     %12s     %2s     %2s         %10s %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
             get_prec_str(prec), get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type),
             get_recon_str(link_recon), get_recon_str(link_recon_sloppy),
             get_solve_str(solve_type), xdim, ydim, zdim, tdim, Lsdim,
             get_dslash_str(dslash_type), get_mass_normalization_str(normalization));

  if(inv_multigrid) {
    printfQuda("MG parameters\n");
    printfQuda(" - number of levels %d\n", mg_levels);
    for (int i=0; i<mg_levels-1; i++) {
      printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
      printfQuda(" - level %d number of pre-smoother applications %d\n", i+1, nu_pre[i]);
      printfQuda(" - level %d number of post-smoother applications %d\n", i+1, nu_post[i]);
    }

    printfQuda("MG Eigensolver parameters\n");
    for (int i = 0; i < mg_levels; i++) {
      if (low_mode_check || mg_eig[i]) {
	printfQuda(" - level %d solver mode %s\n", i + 1, get_eig_type_str(mg_eig_type[i]));
	printfQuda(" - level %d spectrum requested %s\n", i + 1, get_eig_spectrum_str(mg_eig_spectrum[i]));
	printfQuda(" - level %d number of eigenvectors requested nConv %d\n", i + 1, nvec[i]);
	printfQuda(" - level %d size of eigenvector search space %d\n", i + 1, mg_eig_nEv[i]);
	printfQuda(" - level %d size of Krylov space %d\n", i + 1, mg_eig_nKr[i]);
	printfQuda(" - level %d solver tolerance %e\n", i + 1, mg_eig_tol[i]);
	printfQuda(" - level %d convergence required (%s)\n", i + 1, mg_eig_require_convergence[i] ? "true" : "false");
	printfQuda(" - level %d Operator: daggered (%s) , norm-op (%s)\n", i + 1, mg_eig_use_dagger[i] ? "true" : "false",
		   mg_eig_use_normop[i] ? "true" : "false");
	if (mg_eig_use_poly_acc[i]) {
	  printfQuda(" - level %d Chebyshev polynomial degree %d\n", i + 1, mg_eig_poly_deg[i]);
	  printfQuda(" - level %d Chebyshev polynomial minumum %e\n", i + 1, mg_eig_amin[i]);
	  if (mg_eig_amax[i] <= 0)
	    printfQuda(" - level %d Chebyshev polynomial maximum will be computed\n", i + 1);
	  else
	    printfQuda(" - level %d Chebyshev polynomial maximum %e\n", i + 1, mg_eig_amax[i]);
	}
	printfQuda("\n");
      }
    }
  }
  
  if (inv_deflate) {
    printfQuda("\n   Eigensolver parameters\n");
    printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
    printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
    printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
    printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
    printfQuda(" - size of Krylov space %d\n", eig_nKr);
    printfQuda(" - solver tolerance %e\n", eig_tol);
    printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
    if (eig_compute_svd) {
      printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
      printfQuda(" - ***********************************************************\n");
      printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
      printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
      printfQuda(" - ***********************************************************\n");
    } else {
      printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
                 eig_use_normop ? "true" : "false");
    }
    if (eig_use_poly_acc) {
      printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
      printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
      if (eig_amax <= 0)
        printfQuda(" - Chebyshev polynomial maximum will be computed\n");
      else
        printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
    }
  }
  
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

void laphSourceConstruct(std::vector<quda::ColorSpinorField*> &quarks, std::vector<quda::ColorSpinorField*> &evecs,
			 const std::complex<double> *noise, const int dil_scheme)
{  
  int n_dil_vecs = evecs.size()/dil_scheme;
  printfQuda("evecs.size() = %d\n", (int)evecs.size());
  printfQuda("quarks.size() = %d\n", (int)quarks.size());
  printfQuda("dil_scheme = %d\n", dil_scheme);
  printfQuda("n_dil_vecs = %d\n", n_dil_vecs);
  // Construct 4 vectors to hold the 4 spin sources
  
  ColorSpinorParam csp_evecs(*evecs[0]);
  csp_evecs.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField*> sources;
  sources.reserve(4);
  for (int i = 0; i < 4; i++) {
    sources.push_back(ColorSpinorField::Create(csp_evecs));
  }

  // Construct 4 vectors to hold the 4 spin DILUTED sources
  ColorSpinorParam csp_quarks(*quarks[0]);
  //csp_quarks.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField*> dil_sources;
  dil_sources.reserve(4);
  for (int i = 0; i < 4; i++) {
    dil_sources.push_back(ColorSpinorField::Create(csp_quarks));
  }

  // Loop over dilutions
  for(int i = 0; i<dil_scheme; i++) {
    
    // Collect the relevant eigenvectors
    std::vector<ColorSpinorField *> dil_evecs_ptr;
    dil_evecs_ptr.reserve(n_dil_vecs);
    for (int j = 0; j < n_dil_vecs; j++) {
      dil_evecs_ptr.push_back(evecs[i + j * dil_scheme]);
    }

#ifdef DEBUG_LOCAL
    // Check evecs
    for (int ev = 0; ev < n_dil_vecs; ev++) {
      for(int j=0; j<V; j++) {	
	printfQuda("EVECS i = %d, ev = %d ", i, ev);
	dil_evecs_ptr[ev]->PrintVector(j);
      }
    }
    
    // Check noise
    if(i == 0) {
      for (int ev = 0; ev < n_dil_vecs; ev++) {
	printfQuda("SPIN ev = %d : (%e,%e) (%e,%e) (%e,%e) (%e,%e)\n",
		   ev,
		   noise[ev*4 + 0].real(),noise[ev*4 + 0].imag(),
		   noise[ev*4 + 1].real(),noise[ev*4 + 1].imag(),
		   noise[ev*4 + 2].real(),noise[ev*4 + 2].imag(),
		   noise[ev*4 + 3].real(),noise[ev*4 + 3].imag());
      }
    }
#endif
    
    // Construct source
    blas::caxpy(noise, dil_evecs_ptr, sources);

#ifdef DEBUG_LOCAL
    // Check sources
    for (int spin = 0; spin < 4; spin++) {
      for(int j=0; j<V; j++) {	
	printfQuda("SOURCE i = %d, spin = %d ", i, spin);
	sources[spin]->PrintVector(j);
      }
    }
#endif
    
    for (int spin = 0; spin < 4; spin++) {
      spinDiluteQuda(*dil_sources[spin], *sources[spin], spin);
      // Copy spin diluted sources into quark array
      *quarks[4 * i + spin] = *dil_sources[spin];
#ifdef DEBUG_LOCAL
      // Check quarks
      for(int j=0; j<V; j++) {
	printfQuda("QUARK i = %d, spin = %d\n", i, spin);
	quarks[4 * i + spin]->PrintVector(j);
      }
#endif
    }
  }
  printfQuda("All nSpin * dil_scheme sources constructed\n");
  for (int spin = 0; spin < 4; spin++) {
    delete dil_sources[spin];
    delete sources[spin];
  }
}

void laphSourceInvert(std::vector<ColorSpinorField*> &quarks,
		      QudaInvertParam *inv_param, const int *X)
{  
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMERR_PC_SOLVE);

  Dirac *d = nullptr;
  Dirac *dSloppy = nullptr;
  Dirac *dPre = nullptr;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  // `in` will point to the relevant quark[i] source
  // `out` will be copy back to quark[i]
  ColorSpinorField *in = nullptr;
  ColorSpinorField *out = nullptr;
  ColorSpinorField *x = nullptr;
  ColorSpinorField *b = nullptr;

  
  
  ColorSpinorParam cuda_param(*quarks[0]);
  b = ColorSpinorField::Create(cuda_param);  
  x = ColorSpinorField::Create(cuda_param);
  
  // Zero solver stats
  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;
  double secs = 0.0;
  double gflops = 0.0;
  int iter = 0;
  
  for(int i=0; i<(int)quarks.size(); i++) {
    *b = *quarks[i];
    dirac.prepare(in, out, *x, *b, inv_param->solution_type);  
    DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    SolverParam solverParam(*inv_param);    
    Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileLaphInvert);
    
    (*solve)(*out, *in);

    *quarks[i] = *x;
    solverParam.updateInvertParam(*inv_param);

    // Accumulate Solver stats
    secs += inv_param->secs;
    gflops += inv_param->gflops;
    iter += inv_param->iter;
    // Zero solver stats
    inv_param->secs = 0;
    inv_param->gflops = 0;
    inv_param->iter = 0;
    delete solve;
  }
  
  delete x;
  delete b;
  delete d;
  delete dSloppy;
  delete dPre;

  
  //for(int j=0; j<V; j++) quarks[0]->PrintVector(j);
}

// There will be nSpin * dil_scheme quark sinks, and n_evecs eigenvectors. We will produce
// n_evecs * dil_scheme smeared sinks, each of which has a spin index 
void laphSinkProject(std::vector<ColorSpinorField*> &quarks, std::vector<ColorSpinorField*> &evecs,
		     void *host_sinks, const int dil_scheme)
{
  int n_evecs = (int)(evecs.size());
  int n_quarks = (int)(quarks.size());
  
  for(int i=0; i<n_evecs; i++) {
    for(int k=0; k<n_quarks; k++) {
      //FIXME Indexing....
      //printfQuda("i=%d k=%d\n", i, k); 
      evecProjectQuda(*quarks[k], *evecs[i], (double*)host_sinks);
    }
  }
}

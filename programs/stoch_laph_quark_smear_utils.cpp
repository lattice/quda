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
/*
void laphSourceConstruct(std::vector<ColorSpinorField*> &quarks, std::vector<ColorSpinorField*> &evecs,
			 const Complex *noise, const int dil_scheme)
{  
  int n_dil_vecs = evecs.size()/dil_scheme;
  printfQuda("evecs.size() = %d\n", (int)evecs.size());
  // Construct 4 vectors to hold the 4 spin sources
  
  ColorSpinorParam csp_evecs(*evecs[0]);
  std::vector<ColorSpinorField*> sources;
  sources.reserve(4);
  for (int i = 0; i < 4; i++) {
    sources.push_back(ColorSpinorField::Create(csp_evecs));
  }

  // Construct 4 vectors to hold the 4 spin DILUTED sources
  ColorSpinorParam csp_quarks(*quarks[0]);
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
      dil_evecs_ptr.push_back(evecs[i + j*dil_scheme]);
    }
    
    // Construct source
    blas::caxpy(noise, dil_evecs_ptr, sources);
    
    for (int spin = 0; spin < 4; spin++) {
      spinDiluteQuda(*dil_sources[spin], *sources[spin], spin);
      // Copy spin diluted sources into quark array
      *quarks[4 * i + spin] = *dil_sources[spin];
    }
  }
  // All 4 * dil_scheme sources constructed  
}
*/

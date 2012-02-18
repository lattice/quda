#ifndef _HISQ_LINKS_QUDA_H
#define _HISQ_LINKS_QUDA_H

#include <gauge_field.h>


// ***************************************************
//  Declarations for unitarization functions used 
//  in the construction of the hisq-fattened links
// 
//  There are many algorithms for unitarizing 
//  fat7-smeared link variables. 
//  In practice, we use the method employed by 
//  MILC and QOPQDP, namely a combination of 
//  "analytic", or Cayley-Hamilton, unitarization, 
//  and SVD.  
//  Analytic unitarization is first attempted. 
//  The eigenvalues of the matrix Q = V^{dagger}V 
//  (V being the fat7 link) are computed, 
//  if the determinant of Q is less than a user-defined 
//  value (svd_abs_error), or the relative error on the 
//  determinant, estimated by comparing the product 
//  of the eigenvalues of Q to the determinant obtained 
//  from the standard formula, is greater than a 
//  user-specified tolerance (svd_rel_error), then 
//  SVD is used to perform the unitarization.
// ***************************************************  


namespace hisq {

void setUnitarizeLinksPadding(int input_padding, 
			     int output_padding);

void setUnitarizeLinksConstants(double unitarize_eps, double max_error, 
				bool allow_svd, bool svd_only,
				double svd_rel_error, double svd_abs_error,
				bool check_unitarization=true);


void unitarizeLinksCuda(const QudaGaugeParam& param,
			cudaGaugeField& infield,
			cudaGaugeField* outfield, 
			int* num_failures);

void unitarizeLinksCPU(const QudaGaugeParam& param,
		       cpuGaugeField& infield,
		       cpuGaugeField* outfield);

bool isUnitary(const QudaGaugeParam& param, cpuGaugeField& field, double max_error);

} // namespace hisq


#endif // _HISQ_LINKS_H

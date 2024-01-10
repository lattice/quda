#ifndef _UNITARIZATION_LINKS_QUDA_H
#define _UNITARIZATION_LINKS_QUDA_H

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


namespace quda {

  void setUnitarizeLinksConstants(double unitarize_eps, double max_error, 
				  bool allow_svd, bool svd_only,
				  double svd_rel_error, double svd_abs_error);

  void unitarizeLinksCPU(GaugeField &outfield, const GaugeField &infield);

  void unitarizeLinks(GaugeField &outfield, const GaugeField &infield, int *fails);
  void unitarizeLinks(GaugeField &outfield, int *fails);

  bool isUnitary(const GaugeField &field, double max_error);

} // namespace quda


#endif // _UNITARIZATION_LINKS_H

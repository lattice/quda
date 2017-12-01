/* C. Kallidonis: Internal header file for the qlua-interface
 * lib/interface_qlua.cpp
 */

#ifndef INTERFACE_QLUA_INT_H__
#define INTERFACE_QLUA_INT_H__

#include <interface_qlua.h>
#include <complex_quda.h>

#define CMUL(x,y)  ( (QLUA_COMPLEX) {(x.re) * (y.re) - (x.im) * (y.im), (x.re) * (y.im) + (x.im) * (y.re) } )
#define CADD(x,y)  ( (QLUA_COMPLEX) {(x.re)+(y.re), (x.im)+(y.im)})
#define CSUB(x,y)  ( (QLUA_COMPLEX) {(x.re)-(y.re), (x.im)-(y.im)})
#define FETCH(ptr,c,s) ( (QLUA_COMPLEX) { ptr[(c)+QUDA_Nc*(s)].x, ptr[(c)+QUDA_Nc*(s)].y } )

namespace quda {
  
  void cudaContractQQ(
  		      ColorSpinorField **propOut,
  		      ColorSpinorField **propIn1,
  		      ColorSpinorField **propIn2,
  		      int parity, int Nc, int Ns,
  		      contractParam cParam);

} //- namespace quda
  
#endif/*INTERFACE_QLUA_INT_H__*/

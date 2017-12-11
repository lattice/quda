/* C. Kallidonis: Internal header file for the qlua-interface
 * lib/interface_qlua.cpp
 */

#ifndef INTERFACE_QLUA_INT_H__
#define INTERFACE_QLUA_INT_H__

#include <interface_qlua.h>
#include <complex_quda.h>

namespace quda {
  
  void cudaContractQQ(
  		      ColorSpinorField **propOut,
  		      ColorSpinorField **propIn1,
  		      ColorSpinorField **propIn2,
  		      int parity, int Nc, int Ns,
  		      contractParam cParam);

} //- namespace quda
  
#endif/*INTERFACE_QLUA_INT_H__*/

#ifndef _QUDA_NEW_INTERFACE_H
#define _QUDA_NEW_INTERFACE_H

/**
 * @file quda_new_interface.h
 * @brief Experimental new interace.  This will eventually evolve into
 * the new quda.h, likely to coincide with release 0.5.0.
 *
 */


#ifdef __cplusplus
extern "C" {
#endif

  /**
   * Parameters relating to a DiracField
   */
  typedef struct QUDA_DiracFieldParam_s {
    int X[4];          /**< Lattice dimensions */ 
    int Ls;            /**< Extent of the 5th dimension (for domain wall) */
    int Nc;            /**< Number of colors */
    int Ns;            /**< Number of spins */
    QudaTwistFlavorType twist_flavor;  /**< Twisted mass flavor */
    int pad; /**< Pad used on the field */
  } QUDA_DiracFieldParam;  

  typedef struct QUDA_DiracField_s QUDA_DiracField;

  #ifdef __cplusplus
}
#endif

#endif /* _QUDA_NEW_INTERFACE_H */



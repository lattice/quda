#ifndef __DEBUG_UTILITIES_H__
#define __DEBUG_UTILITIES_H__

#include <cstdlib> 
/* *****************************************************************
 *  
 *   A set of host functions written to aid in the debugging 
 *   of QUDA routines. At the moment, most of these functions
 *   support single- and double-precision staggered fields only.
 *
 *   clearField(cudaColorSpinorField* const field):
 *     Sets all the elements of field to zero.
 *
 *   setField(cudaColorSpinorField* const field, double value):
 *     Sets all the field elements to value. Does not change the 
 *     ghost-zone values.
 *
 *   setElement(cudaColorSpinorField* const field, unsigned int index,
 *              double value):
 *     Assigns `value' to a single field component with location 
 *     given by index.
 *
 *   peek(const cudaColorSpinorField& field, unsigned int index):
 *     Copies the value of the field at array index "index" 
 *     back to the device.
 *
 *   getFieldData(const cudaColorSpinorField& field):
 *     Returns a host array containing the field data.
 *      Device layout preserved, except any padding is stripped away.
 *      Does not include ghost data.
 *
 * *****************************************************************/

class cudaColorSpinorField;

void clearField(cudaColorSpinorField* const field);
void setField(cudaColorSpinorField* const field, double value);
void setElement(cudaColorSpinorField* const field, unsigned int index, double value);
double peek(const cudaColorSpinorField& field, unsigned int index);
void *getFieldData(const cudaColorSpinorField& field);
#endif

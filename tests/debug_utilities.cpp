#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <quda.h>
#include <quda_internal.h>
#include <cuda.h>
#include <color_spinor_field.h>


using namespace std;
using namespace quda;
// a few utility methods to make debugging easier
// set all the field 
void clearField(cudaColorSpinorField* const field)
{
  const QudaPrecision precision = field->Precision();
  const unsigned int bytes = field->Bytes();

  cudaMemset(field->V(), 0, bytes);  

  if(precision == QUDA_HALF_PRECISION){
    cudaMemset(field->Norm(), 0, field->NormBytes());
  }
  return;
}

// sets all field variables to a particular value
void setField(cudaColorSpinorField* const field, double value)
{
  const QudaPrecision precision = field->Precision();
  const unsigned int length = field->Length();

  if(precision==QUDA_SINGLE_PRECISION){
    float val = value;
    for(unsigned int i=0; i<length; ++i)
      cudaMemcpy((char*)field->V() + i*precision, &val, sizeof(val), cudaMemcpyHostToDevice); 
  } else if(precision==QUDA_DOUBLE_PRECISION){
    for(unsigned int i=0; i<length; ++i)
      cudaMemcpy((char*)field->V() + i*precision, &value, sizeof(value), cudaMemcpyHostToDevice); 
  } else {
    errorQuda("Half-precision is not yet supported");
  }
  return;
}

void setElement(cudaColorSpinorField* const field, unsigned int index, double value)
{
  const unsigned int total_length = field->TotalLength();
  const unsigned int precision = field->Precision();

  if(index >= total_length){
    errorQuda("index %d exceeds max index %d", index, total_length-1);
  }

  if(precision == QUDA_DOUBLE_PRECISION){
    cudaMemcpy((char*)field->V() + index*precision, &value, sizeof(value), cudaMemcpyHostToDevice);
  }else if(precision == QUDA_SINGLE_PRECISION){
    float val = value;
    cudaMemcpy((char*)field->V() + index*precision, &val, sizeof(val), cudaMemcpyHostToDevice);
  }else{
    errorQuda("Half-precision is not yet supported");
  }
  return;
}


double peek(const cudaColorSpinorField& field, unsigned int index)
{
  const unsigned int total_length = field.TotalLength();
  const QudaPrecision precision = field.Precision();

  if(index >= total_length){
    errorQuda("index %d exceeds max index %d", index, total_length-1);
  }


  if(precision == QUDA_DOUBLE_PRECISION){
    double result;
    cudaMemcpy(&result, (char*)field.V() + index*precision, sizeof(result), cudaMemcpyDeviceToHost);
    return result;
  }else if(precision == QUDA_SINGLE_PRECISION){
    float result;
    cudaMemcpy(&result, (char*)field.V() + index*precision, sizeof(result), cudaMemcpyDeviceToHost);
    return result; 
  }else{
    errorQuda("Half-precision is not yet supported");
  }
  return -1;
}

// Returns the data stored in the field without the padding
void* getFieldData(const cudaColorSpinorField& field)
{
  if(field.Precision() == QUDA_HALF_PRECISION){
    errorQuda("Half-precision is not yet supported");
    return NULL;
  }
  const unsigned int real_length = field.RealLength();
  const QudaPrecision precision = field.Precision();
  const QudaSiteSubset subset = field.Subset();
  const unsigned int pad = field.Pad();
  const unsigned int stride = field.Stride();

  // Throw an exception is memory isn't properly allocated
  try {
    void* host_field = new char[real_length*precision];
  }

  // Now copy the field
  // Need to refactor so that this is done in another function
  if(subset == QUDA_PARITY_SITE_SUBSET){
    if(pad == 0){
      // can copy the field in one operation
      cudaMemcpy(host_field, field.V(), real_length*precision, cudaMemcpyDeviceToHost);
    }else{
      // define an active region
      unsigned int active_length = stride-pad;
      unsigned int num_strides = real_length/stride;
      assert(stride*num_strides == real_length); // check that real_length is an integer multiple of 
                                                 // of the stride size

      for(int i=0; i<num_strides; ++i){
        void* dst = (char*)host_field + i*active_length*precision;
        void* src = (char*)field.V() + i*stride*precision;
        cudaMemcpy(dst, src, active_length*precision, cudaMemcpyDeviceToHost);
      } // loop over the number of strides  
    }
  } else {
    errorQuda("QUDA_FULL_SITE_SUBSET is not yet supported");
  } 

  return host_field;
}

/* 
// In this file, we include a bunch of unit test for the various components needed for the 
// overlapping additive schwarz preconditioning
// Need to check the positioning of the ghost fields as well as the ghost ordering
// I guess, that I have pack ghost and unpack ghost functions.
// 
// First, let's look at the position of the ghost fields
// We want to test the ghost functions


// ColorSpinorFields tests
// ColorSpinor Field layout is 
//
// even_sites, even_ghost_sites, odd_sites, odd_ghost_sites
//
// ColorSpinorFields has variables 
// real_length = volume*nColor*nSpin*2, where volume would 
// be half the lattice volume in the case of a single-parity field
// real_length therefore gives the number of real numbers needed to store the field
// it does not include any padding of the fields or the ghost zones.
//
// The length variable, on the other hand, does include padding but 
// doesn't include the ghost zones needed for communication. 
// length = (volume + pad)*nColor*nSpin*2
// Then ghost_length is the number of real numbers needed to store the ghost 
// fields. 
// ghost_length = ghostVolume*nSpin*nColor*2
// if(siteSubset == QUDA_PARITY_SITE_SUBSET) then
// ghostFace[0] = x[1]*x[2]*x[3]/2
// ghostFace[1] = x[0]*x[2]*x[3]
// ghostFace[2] = x[0]*x[1]*x[3]
// ghostFace[3] = x[0]*x[1]*x[2]
// (We assume paritioning in all directions, if the i direction is not partitioned, then ghostFace[i]=0.)
// Note the difference of a factor of 2 between the definition of ghostFace[0] 
// and the other elements of ghostFace, which is due to the fact that x[0] already
// incorporates division by 2.
// Then ghostVolume = num_faces*(ghostFace[[0] + ghostFace[1] + ghostFace[2] + ghostFace[3])
//      ghostNormVolume = num_norm_faces*(ghostFace[0] + ghostFace[1] + ghostFace[2] + ghostFace[3])
// and
// ghostOffset[0] = 0
// ghostOffset[1] = ghostOffset[0] + num_faces*ghostFace[0]
// ghostOffset[2] = ghostOffset[1] + num_faces*ghostFace[1]
// ghostOffset[3] = ghostOffset[2] + num_faces*ghostFace[2]
//
// ghostNormOffset[0] = 0
// ghostNormOffset[1] = ghostNormOffset[0] + num_norm_faces*ghostNormFace[0]
// etc.
//
// Then ghost length = ghostVolume*nColor*nSpin*2
//      ghost_norm_length = ghostNormVolume
//
// Then total_length = length + 2*ghost_length, since there are two ghost zone in a full field.
// Note that for improved staggered fermions, num_faces = num_norm_faces = 3x2 = 6, where the 
// factor of 2 comes from communicating in the forward and backward directions.
//
// Then we should have that 
// ghost[i] = ((char*)v + (length + ghostOffset[i]*nColor*nSpin*2)*precision);
// ghostNorm[i] = ((char*)norm + (stride + ghostNormOffset[i])*precision





bool testGhostPointers(const cudaColorSpinorField& field)
{
// Use exceptions to check if the test is invalid
// Could have something like:
// the following tests pass ....
// the following tests fail ....
// the following tests do not apply ...
if(field.siteSubset() != QUDA_PARITY_SITE_SUBSET){
}
return false;
}


bool packUnpack(const cudaColorSpinorField& field)
{
  return false;
}


// Take a field, extend the field. Check if the border regions 
// match. Crop the field. Check that the field matches the original.
// Lot of things to do there.
bool resizeTest(const cudaColorSpinorField& field)
{
  return false;
}

  */
int main(int argc, char* argv[])
{

  return EXIT_SUCCESS;
}

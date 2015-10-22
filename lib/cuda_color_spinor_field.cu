#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>

#include <color_spinor_field.h>
#include <blas_quda.h>

#include <string.h>
#include <iostream>
#include <misc_helpers.h>
#include <face_quda.h>
#include <dslash_quda.h>

#ifdef DEVICE_PACK
#define REORDER_LOCATION QUDA_CUDA_FIELD_LOCATION
#else
#define REORDER_LOCATION QUDA_CPU_FIELD_LOCATION
#endif

int zeroCopy = 0;

namespace quda {

  int cudaColorSpinorField::bufferIndex = 0;
  int cudaColorSpinorField::initGhostFaceBuffer = 0;
  void* cudaColorSpinorField::ghostFaceBuffer[2]; //gpu memory
  void* cudaColorSpinorField::fwdGhostFaceBuffer[2][QUDA_MAX_DIM]; //pointers to ghostFaceBuffer
  void* cudaColorSpinorField::backGhostFaceBuffer[2][QUDA_MAX_DIM]; //pointers to ghostFaceBuffer
  size_t cudaColorSpinorField::ghostFaceBytes = 0;

  /*cudaColorSpinorField::cudaColorSpinorField() : 
    ColorSpinorField(), v(0), norm(0), alloc(false), init(false) {

    }*/

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
    ColorSpinorField(param), alloc(false), init(true), texInit(false), 
    initComms(false), bufferMessageHandler(0), nFaceComms(0) {

    // this must come before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      norm = param.norm;
    }

    create(param.create);

    if  (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // dp nothing
    } else if (param.create == QUDA_COPY_FIELD_CREATE){
      errorQuda("not implemented");
    }
    checkCudaError();
  }

  cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : 
    ColorSpinorField(src), alloc(false), init(true), texInit(false), 
    initComms(false), bufferMessageHandler(0), nFaceComms(0) {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, 
					     const ColorSpinorParam &param) :
    ColorSpinorField(src), alloc(false), init(true), texInit(false), 
    initComms(false), bufferMessageHandler(0), nFaceComms(0) {  

    // can only overide if we are not using a reference or parity special case
    if (param.create != QUDA_REFERENCE_FIELD_CREATE || 
	(param.create == QUDA_REFERENCE_FIELD_CREATE && 
	 src.SiteSubset() == QUDA_FULL_SITE_SUBSET && 
	 param.siteSubset == QUDA_PARITY_SITE_SUBSET && 
	 typeid(src) == typeid(cudaColorSpinorField) ) || 
         (param.create == QUDA_REFERENCE_FIELD_CREATE && param.eigv_dim > 0)) {
      reset(param);
    } else {
      errorQuda("Undefined behaviour"); // else silent bug possible?
    }

    // This must be set before create is called
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      if (typeid(src) == typeid(cudaColorSpinorField)) {
	v = (void*)src.V();
	norm = (void*)src.Norm();
      } else {
	errorQuda("Cannot reference a non-cuda field");
      }

      if (this->EigvDim() > 0) 
      {//setup eigenvector form the set
         if(eigv_dim != this->EigvDim()) errorQuda("\nEigenvector set does not match..\n") ;//for debug only.
         if(eigv_id > -1)
         {
           //printfQuda("\nSetting pointers for vector id %d\n", eigv_id); //for debug only.
           v    = (void*)((char*)v + eigv_id*bytes);         
           norm = (void*)((char*)norm + eigv_id*norm_bytes);         
         }
       //do nothing for the eigenvector subset...
      }
    }

    create(param.create);

    if (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      copySpinorField(src);
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else {
      errorQuda("CreateType %d not implemented", param.create);
    }

  }

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) 
    : ColorSpinorField(src), alloc(false), init(true), texInit(false), 
      initComms(false), bufferMessageHandler(0), nFaceComms(0) {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  ColorSpinorField& cudaColorSpinorField::operator=(const ColorSpinorField &src) {
    if (typeid(src) == typeid(cudaColorSpinorField)) {
      *this = (dynamic_cast<const cudaColorSpinorField&>(src));
    } else if (typeid(src) == typeid(cpuColorSpinorField)) {
      *this = (dynamic_cast<const cpuColorSpinorField&>(src));
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
    return *this;
  }

  cudaColorSpinorField& cudaColorSpinorField::operator=(const cudaColorSpinorField &src) {
    if (&src != this) {
      // keep current attributes unless unset
      if (!ColorSpinorField::init) { // note this will turn a reference field into a regular field
	destroy();
	destroyComms(); // not sure if this necessary
	ColorSpinorField::operator=(src);
	create(QUDA_COPY_FIELD_CREATE);
      }
      copySpinorField(src);
    }
    return *this;
  }

  cudaColorSpinorField& cudaColorSpinorField::operator=(const cpuColorSpinorField &src) {
    // keep current attributes unless unset
    if (!ColorSpinorField::init) { // note this will turn a reference field into a regular field
      destroy();
      ColorSpinorField::operator=(src);
      create(QUDA_COPY_FIELD_CREATE);
    }
    loadSpinorField(src);
    return *this;
  }

  cudaColorSpinorField::~cudaColorSpinorField() {
    destroyComms();
    destroy();
  }

  bool cudaColorSpinorField::isNative() const {

    if (precision == QUDA_DOUBLE_PRECISION) {
      if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) return true;
    } else if (precision == QUDA_SINGLE_PRECISION) {
      if (nSpin == 4) {
	if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) return true;
      } else if (nSpin == 1) {
	if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) return true;
      }
    } else if (precision == QUDA_HALF_PRECISION) {
      if (nSpin == 4) {
	if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) return true;
      } else if (nSpin == 1) {
	if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) return true;
      }
    }

    return false;
  }

  void cudaColorSpinorField::create(const QudaFieldCreate create) {

    if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER) {
      errorQuda("Subset not implemented");
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      v = device_malloc(bytes);
      if (precision == QUDA_HALF_PRECISION) {
	norm = device_malloc(norm_bytes);
      }
      alloc = true;
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      if(eigv_dim != 0) errorQuda("Eigenvectors must be parity fields!");
      // create the associated even and odd subsets
      ColorSpinorParam param;
      param.siteSubset = QUDA_PARITY_SITE_SUBSET;
      param.nDim = nDim;
      memcpy(param.x, x, nDim*sizeof(int));
      param.x[0] /= 2; // set single parity dimensions
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      param.v = v;
      param.norm = norm;
      even = new cudaColorSpinorField(*this, param);
      odd = new cudaColorSpinorField(*this, param);

      // need this hackery for the moment (need to locate the odd pointers half way into the full field)
      (dynamic_cast<cudaColorSpinorField*>(odd))->v = (void*)((char*)v + bytes/2);
      if (precision == QUDA_HALF_PRECISION) 
	(dynamic_cast<cudaColorSpinorField*>(odd))->norm = (void*)((char*)norm + norm_bytes/2);

      for(int i=0; i<nDim; ++i){
        if(commDimPartitioned(i)){
          (dynamic_cast<cudaColorSpinorField*>(odd))->ghost[i] =
	    static_cast<char*>((dynamic_cast<cudaColorSpinorField*>(odd))->ghost[i]) + bytes/2;
          if(precision == QUDA_HALF_PRECISION)
	    (dynamic_cast<cudaColorSpinorField*>(odd))->ghostNorm[i] =
	      static_cast<char*>((dynamic_cast<cudaColorSpinorField*>(odd))->ghostNorm[i]) + norm_bytes/2;
        }
      }

#ifdef USE_TEXTURE_OBJECTS
      dynamic_cast<cudaColorSpinorField*>(even)->destroyTexObject();
      dynamic_cast<cudaColorSpinorField*>(even)->createTexObject();
      dynamic_cast<cudaColorSpinorField*>(odd)->destroyTexObject();
      dynamic_cast<cudaColorSpinorField*>(odd)->createTexObject();
#endif
    }
    else{//siteSubset == QUDA_PARITY_SITE_SUBSET

      //! setup an object for selected eigenvector (the 1st one as a default):
      if ((eigv_dim > 0) && (create != QUDA_REFERENCE_FIELD_CREATE) && (eigv_id == -1)) 
      {
         //if(bytes > 1811939328) warningQuda("\nCUDA API probably won't be able to create texture object for the eigenvector set... Object size is : %u bytes\n", bytes);
         if (getVerbosity() == QUDA_DEBUG_VERBOSE) printfQuda("\nEigenvector set constructor...\n");
         // create the associated even and odd subsets
         ColorSpinorParam param;
         param.siteSubset = QUDA_PARITY_SITE_SUBSET;
         param.nDim = nDim;
         memcpy(param.x, x, nDim*sizeof(int));
         param.create = QUDA_REFERENCE_FIELD_CREATE;
         param.v = v;
         param.norm = norm;
         param.eigv_dim  = eigv_dim;
         //reserve eigvector set
         eigenvectors.reserve(eigv_dim);
         //setup volume, [real_]length and stride for a single eigenvector
         for(int id = 0; id < eigv_dim; id++)
         {
            param.eigv_id = id;
            eigenvectors.push_back(new cudaColorSpinorField(*this, param));

#ifdef USE_TEXTURE_OBJECTS //(a lot of texture objects...)
            dynamic_cast<cudaColorSpinorField*>(eigenvectors[id])->destroyTexObject();
            dynamic_cast<cudaColorSpinorField*>(eigenvectors[id])->createTexObject();
#endif
         }
      }
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (siteSubset != QUDA_FULL_SITE_SUBSET) {
	zeroPad();
      } else {
	(dynamic_cast<cudaColorSpinorField*>(even))->zeroPad();
	(dynamic_cast<cudaColorSpinorField*>(odd))->zeroPad();
      }
    }

#ifdef USE_TEXTURE_OBJECTS
    if((eigv_dim == 0) || (eigv_dim > 0 && eigv_id > -1))
       createTexObject();
#endif

    // initialize the ghost pointers 
    if(siteSubset == QUDA_PARITY_SITE_SUBSET) {
      for(int i=0; i<nDim; ++i){
        if(commDimPartitioned(i)){
          ghost[i] = (char*)v + (stride + ghostOffset[i])*nColor*nSpin*2*precision;
          if(precision == QUDA_HALF_PRECISION)
            ghostNorm[i] = (char*)norm + (stride + ghostNormOffset[i])*QUDA_SINGLE_PRECISION;
        }
      }
    }
    checkCudaError();
  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaColorSpinorField::createTexObject() {

    if (isNative()) {
      if (texInit) errorQuda("Already bound textures");
      
      // create the texture for the field components
      
      cudaChannelFormatDesc desc;
      memset(&desc, 0, sizeof(cudaChannelFormatDesc));
      if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
      else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2
      
      // staggered fields in half and single are always two component
      if (nSpin == 1 && (precision == QUDA_HALF_PRECISION || precision == QUDA_SINGLE_PRECISION)) {
	desc.x = 8*precision;
	desc.y = 8*precision;
	desc.z = 0;
	desc.w = 0;
      } else { // all others are four component
	desc.x = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	desc.y = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	desc.z = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	desc.w = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
      }
      
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = v;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = bytes;
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
      else texDesc.readMode = cudaReadModeElementType;
      
      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
      checkCudaError();
      
      // create the texture for the norm components
      if (precision == QUDA_HALF_PRECISION) {
	cudaChannelFormatDesc desc;
	memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	desc.f = cudaChannelFormatKindFloat;
	desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;
	
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = norm;
	resDesc.res.linear.desc = desc;
	resDesc.res.linear.sizeInBytes = norm_bytes;
	
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	
	cudaCreateTextureObject(&texNorm, &resDesc, &texDesc, NULL);
	checkCudaError();
      }
      
      texInit = true;
    }
  }

  void cudaColorSpinorField::destroyTexObject() {
    if (isNative() && texInit) {
      cudaDestroyTextureObject(tex);
      if (precision == QUDA_HALF_PRECISION) cudaDestroyTextureObject(texNorm);
      texInit = false;
      checkCudaError();
    }
  }
#endif

  void cudaColorSpinorField::destroy() {
    if (alloc) {
      device_free(v);
      if (precision == QUDA_HALF_PRECISION) device_free(norm);

      if (siteSubset != QUDA_FULL_SITE_SUBSET) {
        //! for deflated solvers:
        if (eigv_dim > 0) 
        {
          std::vector<ColorSpinorField*>::iterator vec;
          for(vec = eigenvectors.begin(); vec != eigenvectors.end(); vec++) delete *vec;
        } 
      }
      alloc = false;
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      delete even;
      delete odd;
    }

#ifdef USE_TEXTURE_OBJECTS
    if((eigv_dim == 0) || (eigv_dim > 0 && eigv_id > -1))
       destroyTexObject();
#endif

  }

  cudaColorSpinorField& cudaColorSpinorField::Even() const { 
    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      return *(dynamic_cast<cudaColorSpinorField*>(even)); 
    }

    errorQuda("Cannot return even subset of %d subset", siteSubset);
    exit(-1);
  }

  cudaColorSpinorField& cudaColorSpinorField::Odd() const {
    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      return *(dynamic_cast<cudaColorSpinorField*>(odd)); 
    }

    errorQuda("Cannot return odd subset of %d subset", siteSubset);
    exit(-1);
  }

  // cuda's floating point format, IEEE-754, represents the floating point
  // zero as 4 zero bytes
  void cudaColorSpinorField::zero() {
    cudaMemsetAsync(v, 0, bytes, streams[Nstream-1]);
    if (precision == QUDA_HALF_PRECISION) cudaMemsetAsync(norm, 0, norm_bytes, streams[Nstream-1]);
  }


  void cudaColorSpinorField::zeroPad() {
    size_t pad_bytes = (stride - volume) * precision * fieldOrder;
    int Npad = nColor * nSpin * 2 / fieldOrder;

    if (eigv_dim > 0 && eigv_id == -1){//we consider the whole eigenvector set:
      Npad      *= eigv_dim;
      pad_bytes /= eigv_dim;
    }

    size_t pitch = ((eigv_dim == 0 || eigv_id != -1) ? stride : eigv_stride)*fieldOrder*precision;
    char   *dst  = (char*)v + ((eigv_dim == 0 || eigv_id != -1) ? volume : eigv_volume)*fieldOrder*precision;
    if(pad_bytes) cudaMemset2D(dst, pitch, 0, pad_bytes, Npad);

    //for (int i=0; i<Npad; i++) {
    //  if (pad_bytes) cudaMemset((char*)v + (volume + i*stride)*fieldOrder*precision, 0, pad_bytes);
    //}
  }

  void cudaColorSpinorField::copy(const cudaColorSpinorField &src) {
    checkField(*this, src);
    copyCuda(*this, src);
  }

  void cudaColorSpinorField::copySpinorField(const ColorSpinorField &src) {
    
    // src is on the device and is native
    if (typeid(src) == typeid(cudaColorSpinorField) && 
	isNative() && dynamic_cast<const cudaColorSpinorField &>(src).isNative()) {
      copy(dynamic_cast<const cudaColorSpinorField&>(src));
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else if (typeid(src) == typeid(cpuColorSpinorField)) { // src is on the host
      loadSpinorField(src);
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
  } 

  void cudaColorSpinorField::loadSpinorField(const ColorSpinorField &src) {

    if (REORDER_LOCATION == QUDA_CPU_FIELD_LOCATION && 
	typeid(src) == typeid(cpuColorSpinorField)) {
      for(int b=0; b<2; ++b){
        resizeBufferPinned(bytes + norm_bytes, b);
        memset(bufferPinned[b], 0, bytes+norm_bytes); // FIXME (temporary?) bug fix for padding
      }
      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, 
			     bufferPinned[bufferIndex], 0, (char*)bufferPinned[bufferIndex]+bytes, 0);

      cudaMemcpy(v, bufferPinned[bufferIndex], bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(norm, (char*)bufferPinned[bufferIndex]+bytes, norm_bytes, cudaMemcpyHostToDevice);
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else {
      void *Src, *srcNorm;
      if (!zeroCopy) {
	resizeBufferDevice(src.Bytes()+src.NormBytes());
	Src = bufferDevice;
	srcNorm = (char*)bufferDevice + src.Bytes();	
	cudaMemcpy(Src, src.V(), src.Bytes(), cudaMemcpyHostToDevice);
	cudaMemcpy(srcNorm, src.Norm(), src.NormBytes(), cudaMemcpyHostToDevice);
      } else {
	for(int b=0; b<2; ++b){
	 resizeBufferPinned(src.Bytes()+src.NormBytes(), b);
	}
	memcpy(bufferPinned[bufferIndex], src.V(), src.Bytes());
	memcpy((char*)bufferPinned[bufferIndex]+src.Bytes(), src.Norm(), src.NormBytes());

	cudaHostGetDevicePointer(&Src, bufferPinned[bufferIndex], 0);
	srcNorm = (void*)((char*)Src + src.Bytes());
      }

      cudaMemset(v, 0, bytes); // FIXME (temporary?) bug fix for padding
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, 0, Src, 0, srcNorm);
    }

    checkCudaError();
    return;
  }


  void cudaColorSpinorField::saveSpinorField(ColorSpinorField &dest) const {

    if (REORDER_LOCATION == QUDA_CPU_FIELD_LOCATION && 
	typeid(dest) == typeid(cpuColorSpinorField)) {
      for(int b=0; b<2; ++b) resizeBufferPinned(bytes+norm_bytes,b);
      cudaMemcpy(bufferPinned[bufferIndex], v, bytes, cudaMemcpyDeviceToHost);
      cudaMemcpy((char*)bufferPinned[bufferIndex]+bytes, norm, norm_bytes, cudaMemcpyDeviceToHost);

      copyGenericColorSpinor(dest, *this, QUDA_CPU_FIELD_LOCATION, 
			     0, bufferPinned[bufferIndex], 0, (char*)bufferPinned[bufferIndex]+bytes);
    } else if (typeid(dest) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION);
    } else {
      void *dst, *dstNorm;
      if (!zeroCopy) {
	resizeBufferDevice(dest.Bytes()+dest.NormBytes());
	dst = bufferDevice;
	dstNorm = (char*)bufferDevice+dest.Bytes();
      } else {
	for(int b=0; b<2; ++b) resizeBufferPinned(dest.Bytes()+dest.NormBytes(),b);
	cudaHostGetDevicePointer(&dst, bufferPinned[bufferIndex], 0);
	dstNorm = (char*)dst+dest.Bytes();
      }
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dst, v, dstNorm, 0);

      if (!zeroCopy) {
	cudaMemcpy(dest.V(), dst, dest.Bytes(), cudaMemcpyDeviceToHost);
	cudaMemcpy(dest.Norm(), dstNorm, dest.NormBytes(), cudaMemcpyDeviceToHost);
      } else {
	memcpy(dest.V(), bufferPinned[bufferIndex], dest.Bytes());
	memcpy(dest.Norm(), (char*)bufferPinned[bufferIndex]+dest.Bytes(), dest.NormBytes());
      }
    }

    checkCudaError();
    return;
  }

  void cudaColorSpinorField::allocateGhostBuffer(int nFace) {
    int Nint = nColor * nSpin * 2; // number of internal degrees of freedom
    if (nSpin == 4) Nint /= 2; // spin projection for Wilson

    // compute size of buffer required
    size_t faceBytes = 0;
    for (int i=0; i<4; i++) {
      if(!commDimPartitioned(i)) continue;
      faceBytes += 2*nFace*ghostFace[i]*Nint*precision;
      // add extra space for the norms for half precision
      if (precision == QUDA_HALF_PRECISION) faceBytes += 2*nFace*ghostFace[i]*sizeof(float);
    }

    // only allocate if not already allocated or buffer required is bigger than previously
    if(initGhostFaceBuffer == 0 || faceBytes > ghostFaceBytes){    

      if (initGhostFaceBuffer){
        for(int b=0; b<2; ++b) device_free(ghostFaceBuffer[b]); 
      }

      if (faceBytes > 0) {
	for(int b=0; b<2; ++b) ghostFaceBuffer[b] = device_malloc(faceBytes);
	initGhostFaceBuffer = 1;
	ghostFaceBytes = faceBytes;
      }

    }

    size_t offset = 0;
    for (int i=0; i<4; i++) {
      if(!commDimPartitioned(i)) continue;
    
      for(int b=0; b<2; ++b) backGhostFaceBuffer[b][i] = (void*)(((char*)ghostFaceBuffer[b]) + offset);
      offset += nFace*ghostFace[i]*Nint*precision;
      if (precision == QUDA_HALF_PRECISION) offset += nFace*ghostFace[i]*sizeof(float);
      
      for(int b=0; b<2; ++b) fwdGhostFaceBuffer[b][i] = (void*)(((char*)ghostFaceBuffer[b]) + offset);
      offset += nFace*ghostFace[i]*Nint*precision;
      if (precision == QUDA_HALF_PRECISION) offset += nFace*ghostFace[i]*sizeof(float);
    }   
    
  }


  void cudaColorSpinorField::freeGhostBuffer(void)
  {
    if (!initGhostFaceBuffer) return;
  
    for(int b=0; b<2; ++b) device_free(ghostFaceBuffer[b]); 

    for(int i=0;i < 4; i++){
      if(!commDimPartitioned(i)) continue;
      for(int b=0; b<2; ++b){
        backGhostFaceBuffer[b][i] = NULL;
        fwdGhostFaceBuffer[b][i] = NULL;
      }
    }
    initGhostFaceBuffer = 0;  
  }

  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhost(const int nFace, const QudaParity parity, 
                                       const int dim, const QudaDirection dir,
				       const int dagger, cudaStream_t *stream, 
				       void *buffer, double a, double b) 
  {
#ifdef MULTI_GPU
    int face_num;
    if(dir == QUDA_BACKWARDS){
      face_num = 0;
    }else if(dir == QUDA_FORWARDS){
      face_num = 1;
    }else{
      face_num = 2;
    }
    void *packBuffer = buffer ? buffer : ghostFaceBuffer[bufferIndex];
    packFace(packBuffer, *this, nFace, dagger, parity, dim, face_num, *stream, a, b); 
#else
    errorQuda("packGhost not built on single-GPU build");
#endif

  }
 
  // send the ghost zone to the host
  void cudaColorSpinorField::sendGhost(void *ghost_spinor, const int nFace, const int dim, 
				       const QudaDirection dir, const int dagger, 
				       cudaStream_t *stream) {

#ifdef MULTI_GPU
    int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom
    
    if (dim !=3 || getKernelPackT() || getTwistPack()) { // use kernels to pack into contiguous buffers then a single cudaMemcpy

      size_t bytes = nFace*Nint*ghostFace[dim]*precision;
      if (precision == QUDA_HALF_PRECISION) bytes += nFace*ghostFace[dim]*sizeof(float);
      void* gpu_buf = 
	(dir == QUDA_BACKWARDS) ? this->backGhostFaceBuffer[bufferIndex][dim] : this->fwdGhostFaceBuffer[bufferIndex][dim];

      cudaMemcpyAsync(ghost_spinor, gpu_buf, bytes, cudaMemcpyDeviceToHost, *stream); 

    } else if(this->TwistFlavor() != QUDA_TWIST_NONDEG_DOUBLET){ // do multiple cudaMemcpys

      int Npad = Nint / Nvec; // number Nvec buffers we have
      int Nt_minus1_offset = (volume - nFace*ghostFace[3]); // N_t -1 = Vh-Vsh
      int offset = 0;
      if (nSpin == 1) {
	offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset;
      } else if (nSpin == 4) {    
	// !dagger: send lower components backwards, send upper components forwards
	// dagger: send upper components backwards, send lower components forwards
	bool upper = dagger ? true : false; // Fwd is !Back  
	if (dir == QUDA_FORWARDS) upper = !upper;
	int lower_spin_offset = Npad*stride;
	if (upper) offset = (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
	else offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
      }
    
      // QUDA Memcpy NPad's worth. 
      //  -- Dest will point to the right beginning PAD. 
      //  -- Each Pad has size Nvec*Vsh Floats. 
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

      void *dst = (char*)ghost_spinor;
      void *src = (char*)v + offset*Nvec*precision;
      size_t len = nFace*ghostFace[3]*Nvec*precision;     
      size_t spitch = stride*Nvec*precision;
      cudaMemcpy2DAsync(dst, len, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

      if (precision == QUDA_HALF_PRECISION) {
	int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
	void *dst = (char*)ghost_spinor + nFace*Nint*ghostFace[3]*precision;
	void *src = (char*)norm + norm_offset;
	cudaMemcpyAsync(dst, src, nFace*ghostFace[3]*sizeof(float), cudaMemcpyDeviceToHost, *stream); 
      }
    }else{
      int flavorVolume = volume / 2;
      int flavorTFace  = ghostFace[3] / 2;
      int Npad = Nint / Nvec; // number Nvec buffers we have
      int flavor1_Nt_minus1_offset = (flavorVolume - flavorTFace);
      int flavor2_Nt_minus1_offset = (volume - flavorTFace);
      int flavor1_offset = 0;
      int flavor2_offset = 0;
      // !dagger: send lower components backwards, send upper components forwards
      // dagger: send upper components backwards, send lower components forwards
      bool upper = dagger ? true : false; // Fwd is !Back
      if (dir == QUDA_FORWARDS) upper = !upper;
      int lower_spin_offset = Npad*stride;//ndeg tm: stride=2*flavor_volume+pad
      if (upper){
        flavor1_offset = (dir == QUDA_BACKWARDS ? 0 : flavor1_Nt_minus1_offset);
        flavor2_offset = (dir == QUDA_BACKWARDS ? flavorVolume : flavor2_Nt_minus1_offset);
      }else{
        flavor1_offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? 0 : flavor1_Nt_minus1_offset);
        flavor2_offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? flavorVolume : flavor2_Nt_minus1_offset);
      }

      // QUDA Memcpy NPad's worth.
      //  -- Dest will point to the right beginning PAD.
      //  -- Each Pad has size Nvec*Vsh Floats.
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

      void *dst = (char*)ghost_spinor;
      void *src = (char*)v + flavor1_offset*Nvec*precision;
      size_t len = flavorTFace*Nvec*precision;
      size_t spitch = stride*Nvec*precision;//ndeg tm: stride=2*flavor_volume+pad
      size_t dpitch = 2*len;
      cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);
      dst = (char*)ghost_spinor+len;
      src = (char*)v + flavor2_offset*Nvec*precision;
      cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

      if (precision == QUDA_HALF_PRECISION) {
        int Nt_minus1_offset = (flavorVolume - flavorTFace);
        int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
	void *dst = (char*)ghost_spinor + Nint*ghostFace[3]*precision;
	void *src = (char*)norm + norm_offset;
        size_t dpitch = flavorTFace*sizeof(float);
        size_t spitch = flavorVolume*sizeof(float);
	cudaMemcpy2DAsync(dst, dpitch, src, spitch, flavorTFace*sizeof(float), 2, cudaMemcpyDeviceToHost, *stream);
      }
    }
#else
    errorQuda("sendGhost not built on single-GPU build");
#endif

  }



  void cudaColorSpinorField::unpackGhost(const void* ghost_spinor, const int nFace, 
					 const int dim, const QudaDirection dir, 
					 const int dagger, cudaStream_t* stream) 
  {
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom

    int len = nFace*ghostFace[dim]*Nint;
    int offset = length + ghostOffset[dim]*nColor*nSpin*2;
    offset += (dir == QUDA_BACKWARDS) ? 0 : len;

    void *dst = (char*)v + precision*offset;
    const void *src = ghost_spinor;

    cudaMemcpyAsync(dst, src, len*precision, cudaMemcpyHostToDevice, *stream);
    
    if (precision == QUDA_HALF_PRECISION) {
      // norm region of host ghost zone is at the end of the ghost_spinor

      int normlen = nFace*ghostFace[dim];
      int norm_offset = stride + ghostNormOffset[dim];
      norm_offset += (dir == QUDA_BACKWARDS) ? 0 : normlen;

      void *dst = static_cast<char*>(norm) + norm_offset*sizeof(float);
      const void *src = static_cast<const char*>(ghost_spinor)+nFace*Nint*ghostFace[dim]*precision; 
      cudaMemcpyAsync(dst, src, normlen*sizeof(float), cudaMemcpyHostToDevice, *stream);
    }
  }




   // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhostExtended(const int nFace, const int R[], const QudaParity parity,
                                       const int dim, const QudaDirection dir,
                                       const int dagger, cudaStream_t *stream,
                                       void *buffer)
  {
#ifdef MULTI_GPU
    int face_num;
    if(dir == QUDA_BACKWARDS){
      face_num = 0;
    }else if(dir == QUDA_FORWARDS){
      face_num = 1;
    }else{
      face_num = 2;
    }
    void *packBuffer = buffer ? buffer : ghostFaceBuffer[bufferIndex];
    packFaceExtended(packBuffer, *this, nFace, R, dagger, parity, dim, face_num, *stream);
#else
    errorQuda("packGhostExtended not built on single-GPU build");
#endif

  }


  

  // copy data from host buffer into boundary region of device field
  void cudaColorSpinorField::unpackGhostExtended(const void* ghost_spinor, const int nFace, const QudaParity parity,
                                                 const int dim, const QudaDirection dir, 
                                                 const int dagger, cudaStream_t* stream)
  {

     
     
    // First call the regular unpackGhost routine to copy data into the `usual' ghost-zone region 
    // of the data array 
    unpackGhost(ghost_spinor, nFace, dim, dir, dagger, stream);

    // Next step is to copy data from the ghost zone back to the interior region
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom

    int len = nFace*ghostFace[dim]*Nint;
    int offset = length + ghostOffset[dim]*nColor*nSpin*2;
    offset += (dir == QUDA_BACKWARDS) ? 0 : len;

#ifdef MULTI_GPU
    const int face_num = 2;
    const bool unpack = true;
    const int R[4] = {0,0,0,0};
    packFaceExtended(ghostFaceBuffer[bufferIndex], *this, nFace, R, dagger, parity, dim, face_num, *stream, unpack); 
#else
    errorQuda("unpackGhostExtended not built on single-GPU build");
#endif
  }



  cudaStream_t *stream;

  void cudaColorSpinorField::createComms(int nFace) {

    if(bufferMessageHandler != bufferPinnedResizeCount) destroyComms();

    if (!initComms || nFaceComms != nFace) {

      // if we are requesting a new number of faces destroy and start over
      if(nFace != nFaceComms) destroyComms();

      if (siteSubset != QUDA_PARITY_SITE_SUBSET) 
	errorQuda("Only supports single parity fields");

#ifdef GPU_COMMS
      bool comms = false;
      for (int i=0; i<nDimComms; i++) if (commDimPartitioned(i)) comms = true;
#endif

      if (nFace > maxNface) 
	errorQuda("Requested number of faces %d in communicator is greater than supported %d",
		  nFace, maxNface);

      // faceBytes is the sum of all face sizes 
      size_t faceBytes = 0;
      
      // nbytes is the size in bytes of each face
      size_t nbytes[QUDA_MAX_DIM];
      
      // The number of degrees of freedom per site for the given
      // field.  Currently assumes spin projection of a Wilson-like
      // field (so half the number of degrees of freedom).
      int Ndof = (2 * nSpin * nColor) / (nSpin==4 ? 2 : 1);

      for (int i=0; i<nDimComms; i++) {
	nbytes[i] = maxNface*surfaceCB[i]*Ndof*precision;
	if (precision == QUDA_HALF_PRECISION) nbytes[i] += maxNface*surfaceCB[i]*sizeof(float);
	if (!commDimPartitioned(i)) continue;
	faceBytes += 2*nbytes[i];
      }
      
#ifndef GPU_COMMS
      // use static pinned memory for face buffers
      for(int b=0; b<2; ++b){
        resizeBufferPinned(2*faceBytes, b); // oversizes for GPU_COMMS case

        my_face[b] = bufferPinned[b];
        from_face[b] = static_cast<char*>(bufferPinned[b]) + faceBytes;
      }

      // assign pointers for each face - it's ok to alias for different Nface parameters
      size_t offset = 0;
#endif
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;
	
#ifdef GPU_COMMS
	for(int b=0; b<2; ++b){
	  my_back_face[b][i] = backGhostFaceBuffer[b][i];
	  from_back_face[b][i] = ghost[i];
	
	  if(precision == QUDA_HALF_PRECISION){
	    my_back_norm_face[b][i]  = static_cast<char*>(backGhostFaceBuffer[b][i]) + nFace*ghostFace[i]*Ndof*precision;
	    from_back_norm_face[b][i] = ghostNorm[i];
	  }
	} // loop over b

#else
        for(int b=0; b<2; ++b){
	  my_back_face[b][i] = static_cast<char*>(my_face[b]) + offset;
	  from_back_face[b][i] = static_cast<char*>(from_face[b]) + offset;
	}
	offset += nbytes[i];
#endif
	
#ifdef GPU_COMMS
	for(int b=0; b<2; ++b){
	  my_fwd_face[b][i] = fwdGhostFaceBuffer[b][i];	
	  from_fwd_face[b][i] = ghost[i] + nFace*ghostFace[i]*Ndof*precision;

	  if(precision == QUDA_HALF_PRECISION){
	    my_fwd_norm_face[b][i] = static_cast<char*>(fwdGhostFaceBuffer[b][i]) + nFace*ghostFace[i]*Ndof*precision;
	    from_fwd_norm_face[b][i] = static_cast<char*>(ghostNorm[i]) + nFace*ghostFace[i]*sizeof(float);
	  }
	} // loop over b
#else
	for(int b=0; b<2; ++b){
	  my_fwd_face[b][i] = static_cast<char*>(my_face[b]) + offset;
	  from_fwd_face[b][i] = static_cast<char*>(from_face[b]) + offset;
	}
	offset += nbytes[i];
#endif

      }

      // create a different message handler for each direction and Nface
      for(int b=0; b<2; ++b){
        mh_send_fwd[b] = new MsgHandle**[maxNface];
        mh_send_back[b] = new MsgHandle**[maxNface];
        mh_recv_fwd[b] = new MsgHandle**[maxNface];
        mh_recv_back[b] = new MsgHandle**[maxNface];
#ifdef GPU_COMMS
        if(precision == QUDA_HALF_PRECISION){
      	  mh_send_norm_fwd[b]  = new MsgHandle**[maxNface];
      	  mh_send_norm_back[b] = new MsgHandle**[maxNface];
     	  mh_recv_norm_fwd[b]  = new MsgHandle**[maxNface];
      	  mh_recv_norm_back[b] = new MsgHandle**[maxNface]; 
        }
#endif
      } // loop over b
      for (int j=0; j<maxNface; j++) {
	for(int b=0; b<2; ++b){
	  mh_send_fwd[b][j] = new MsgHandle*[2*nDimComms];
	  mh_send_back[b][j] = new MsgHandle*[2*nDimComms];
	  mh_recv_fwd[b][j] = new MsgHandle*[nDimComms];
	  mh_recv_back[b][j] = new MsgHandle*[nDimComms];
		
#ifdef GPU_COMMS
	  if(precision == QUDA_HALF_PRECISION){
	    mh_send_norm_fwd[b][j] = new MsgHandle*[2*nDimComms];
	    mh_send_norm_back[b][j] = new MsgHandle*[2*nDimComms];
	    mh_recv_norm_fwd[b][j] = new MsgHandle*[nDimComms];
	    mh_recv_norm_back[b][j] = new MsgHandle*[nDimComms];
	  }
#endif	
	} // loop over b


	for (int i=0; i<nDimComms; i++) {
	  if (!commDimPartitioned(i)) continue;
#ifdef GPU_COMMS
	  size_t nbytes_Nface = surfaceCB[i]*Ndof*precision*(j+1);
	  size_t nbytes_Nface_norm = surfaceCB[i]*(j+1)*sizeof(float);
	  if (i != 3 || getKernelPackT() || getTwistPack()) {
#else 
	    size_t nbytes_Nface = (nbytes[i] / maxNface) * (j+1);
#endif
	    for(int b=0; b<2; ++b){
	      mh_send_fwd[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(my_fwd_face[b][i], i, +1, nbytes_Nface) : NULL;
	      mh_send_back[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(my_back_face[b][i], i, -1, nbytes_Nface) : NULL;
	      mh_send_fwd[b][j][2*i+1] = mh_send_fwd[b][j][2*i]; // alias pointers
	      mh_send_back[b][j][2*i+1] = mh_send_back[b][j][2*i]; // alias pointers
	    }
#ifdef GPU_COMMS

	    if(precision == QUDA_HALF_PRECISION){
	      for(int b=0; b<2; ++b){
		mh_send_norm_fwd[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(my_fwd_norm_face[b][i], i, +1, nbytes_Nface_norm) : NULL;
		mh_send_norm_back[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(my_back_norm_face[b][i], i, -1, nbytes_Nface_norm) : NULL;
		mh_send_norm_fwd[b][j][2*i+1] = mh_send_norm_fwd[b][j][2*i];
		mh_send_norm_back[b][j][2*i+1] = mh_send_norm_back[b][j][2*i]; 	
	      }
	    }

	  } else if (this->TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET) {
	    errorQuda("GPU_COMMS for non-degenerate doublet only supported with time-dimension kernel packing enabled.");
	  } else {
	    /* 
	       use a strided communicator, here we can't really use
	       the previously declared my_fwd_face and my_back_face
	       pointers since they don't really map 1-to-1 so let's
	       just compute the required base pointers and pass these
	       directly into the communicator construction
	    */
	    
	    int Nblocks = Ndof / Nvec(); // number of Nvec buffers we have
	    // start of last time slice chunk we are sending forwards
	    int endOffset = (volume - (j+1)*ghostFace[i]); 

	    size_t offset[4];
	    void *base[4];
	    if (nSpin == 1) { // staggered is invariant with dagger
	      offset[2*0 + 0] = 0;
	      offset[2*1 + 0] = endOffset;
	      offset[2*0 + 1] = offset[2*0 + 0];
	      offset[2*1 + 1] = offset[2*1 + 0];
	    } else if (nSpin == 4) {    
	      // !dagger: send last components backwards, send first components forwards
	      offset[2*0 + 0] = Nblocks*stride;
	      offset[2*1 + 0] = endOffset;
	      //  dagger: send first components backwards, send last components forwards
	      offset[2*0 + 1] = 0;
	      offset[2*1 + 1] = Nblocks*stride + endOffset;
	    } else {
	      errorQuda("Unsupported number of spin components");
	    }

	    for (int k=0; k<4; k++) {
	      base[k] = static_cast<char*>(v) + offset[k]*Nvec()*precision; // total offset in bytes
	    }

	    size_t blksize  = (j+1)*ghostFace[i]*Nvec()*precision; // (j+1) is number of faces
	    size_t Stride = stride*Nvec()*precision;

	    if (blksize * Nblocks != nbytes_Nface) 
	      errorQuda("Total strided message size does not match expected size");

	    //printf("%d strided sends with Nface=%d Nblocks=%d blksize=%d Stride=%d\n", i, j+1, Nblocks, blksize, Stride);

            for(int b=0; b<2; ++b){
	      // only allocate a communicator for the present face (this needs cleaned up)
	      mh_send_fwd[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_strided_send_relative(base[2], i, +1, blksize, Nblocks, Stride) : NULL;
	      mh_send_back[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_strided_send_relative(base[0], i, -1, blksize, Nblocks, Stride) : NULL;
	      if (nSpin ==4) { // dagger communicators
	        mh_send_fwd[b][j][2*i+1] = (j+1 == nFace) ? comm_declare_strided_send_relative(base[3], i, +1, blksize, Nblocks, Stride) : NULL;
	        mh_send_back[b][j][2*i+1] = (j+1 == nFace) ? comm_declare_strided_send_relative(base[1], i, -1, blksize, Nblocks, Stride) : NULL;
	      } else {
	        mh_send_fwd[b][j][2*i+1] = mh_send_fwd[b][j][2*i+0];
	        mh_send_back[b][j][2*i+1] = mh_send_back[b][j][2*i+0];
	      }

            } // loop over b

          
	    if(precision == QUDA_HALF_PRECISION){
	      int Nt_minus1_offset = (volume - nFace*ghostFace[3]); // The space-time coordinate of the start of the last time slice
	      void *norm_fwd = static_cast<float*>(norm) + Nt_minus1_offset;
	      void *norm_back = norm; // the first time slice has zero offset
	      for(int b=0; b<2; ++b){
		mh_send_norm_fwd[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(norm_fwd, i, +1, surfaceCB[i]*(j+1)*sizeof(float)) : NULL;
		mh_send_norm_back[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(norm_back, i, -1, surfaceCB[i]*(j+1)*sizeof(float)) : NULL;
		mh_send_norm_fwd[b][j][2*i+1] = mh_send_norm_fwd[b][j][2*i];
		mh_send_norm_back[b][j][2*i+1] = mh_send_norm_back[b][j][2*i];  
	      }
	    }

	  }

	  if(precision == QUDA_HALF_PRECISION){
            for(int b=0; b<2; ++b){
	      mh_recv_norm_fwd[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_fwd_norm_face[b][i], i, +1, nbytes_Nface_norm) : NULL;
	      mh_recv_norm_back[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_back_norm_face[b][i], i, -1, nbytes_Nface_norm) : NULL;
            }
	  }
#endif // GPU_COMMS

	  for(int b=0; b<2; ++b){
	    mh_recv_fwd[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_fwd_face[b][i], i, +1, nbytes_Nface) : NULL;
	    mh_recv_back[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_back_face[b][i], i, -1, nbytes_Nface) : NULL;
	  }
	 


	} // loop over dimension
      }
     
      bufferMessageHandler = bufferPinnedResizeCount;
      initComms = true;
      nFaceComms = nFace;
    }
    checkCudaError();
  }
    
  void cudaColorSpinorField::destroyComms() {
    if (initComms) {
      for(int b=0; b<2; ++b){
      for (int j=0; j<maxNface; j++) {
	for (int i=0; i<nDimComms; i++) {
	  if (commDimPartitioned(i)) {
	    if (mh_recv_fwd[b][j][i]) comm_free(mh_recv_fwd[b][j][i]);
	    if (mh_recv_fwd[b][j][i]) comm_free(mh_recv_back[b][j][i]);
	    if (mh_send_fwd[b][j][2*i]) comm_free(mh_send_fwd[b][j][2*i]);
	    if (mh_send_back[b][j][2*i]) comm_free(mh_send_back[b][j][2*i]);
	    // only in a special case are these not aliasing pointers
#ifdef GPU_COMMS
	    if(precision == QUDA_HALF_PRECISION){
	      if (mh_recv_norm_fwd[b][j][i]) comm_free(mh_recv_norm_fwd[b][j][i]);
	      if (mh_recv_norm_back[b][j][i]) comm_free(mh_recv_norm_back[b][j][i]);
	      if (mh_send_norm_fwd[b][j][2*i]) comm_free(mh_send_norm_fwd[b][j][2*i]);
	      if (mh_send_norm_back[b][j][2*i]) comm_free(mh_send_norm_back[b][j][2*i]);
	    }

	    if (i == 3 && !getKernelPackT() && nSpin == 4) {
	      if (mh_send_fwd[b][j][2*i+1]) comm_free(mh_send_fwd[b][j][2*i+1]);
	      if (mh_send_back[b][j][2*i+1]) comm_free(mh_send_back[b][j][2*i+1]);
	    }
#endif // GPU_COMMS
	  }
	}
	delete []mh_recv_fwd[b][j];
	delete []mh_recv_back[b][j];
	delete []mh_send_fwd[b][j];
	delete []mh_send_back[b][j];
#ifdef GPU_COMMS
	if(precision == QUDA_HALF_PRECISION){
	  delete []mh_recv_norm_fwd[b][j];
	  delete []mh_recv_norm_back[b][j];
	  delete []mh_send_norm_fwd[b][j];
	  delete []mh_send_norm_back[b][j];
	}
#endif
      }    
      delete []mh_recv_fwd[b];
      delete []mh_recv_back[b];
      delete []mh_send_fwd[b];
      delete []mh_send_back[b];
      
      for (int i=0; i<nDimComms; i++) {
	my_fwd_face[b][i] = NULL;
	my_back_face[b][i] = NULL;
	from_fwd_face[b][i] = NULL;
	from_back_face[b][i] = NULL;      
      }
#ifdef GPU_COMMS
      if(precision == QUDA_HALF_PRECISION){
	delete []mh_recv_norm_fwd[b];
	delete []mh_recv_norm_back[b];
	delete []mh_send_norm_fwd[b];
	delete []mh_send_norm_back[b];
      }
	
      for(int i=0; i<nDimComms; i++){
	my_fwd_norm_face[b][i] = NULL;
	my_back_norm_face[b][i] = NULL;
	from_fwd_norm_face[b][i] = NULL;
	from_back_norm_face[b][i] = NULL;
      }
#endif      
      } // loop over b
      initComms = false;
      checkCudaError();
    }
  }

  void cudaColorSpinorField::streamInit(cudaStream_t *stream_p){
    stream = stream_p;
  }

  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, cudaStream_t *stream_p, 
				  bool zeroCopyPack, double a, double b) {
    allocateGhostBuffer(nFace);   // allocate the ghost buffer if not yet allocated  
    createComms(nFace); // must call this first

    stream = stream_p;
    
    const int dim=-1; // pack all partitioned dimensions
 
    if (zeroCopyPack) {
      void *my_face_d;
      cudaHostGetDevicePointer(&my_face_d, my_face[bufferIndex], 0); // set the matching device pointer
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d, a, b);
    } else {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger,  &stream[Nstream-1], 0, a, b);
    }
  }

  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, int stream_idx, 
				  bool zeroCopyPack, double a, double b) {
    allocateGhostBuffer(nFace);   // allocate the ghost buffer if not yet allocated  
    createComms(nFace); // must call this first

    const int dim=-1; // pack all partitioned dimensions
 
    if (zeroCopyPack) {
      void *my_face_d;
      cudaHostGetDevicePointer(&my_face_d, my_face[bufferIndex], 0); // set the matching device pointer
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[stream_idx], my_face_d, a, b);
    } else {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger,  &stream[stream_idx], 0, a, b);
    }
  }

  void cudaColorSpinorField::packExtended(const int nFace, const int R[], const int parity, 
                                          const int dagger, const int dim,
                                          cudaStream_t *stream_p, const bool zeroCopyPack){

    allocateGhostBuffer(nFace); // allocate the ghost buffer if not yet allocated
    createComms(nFace); // must call this first

    stream = stream_p;
 
    void *my_face_d = NULL;
    if(zeroCopyPack){ 
      cudaHostGetDevicePointer(&my_face_d, my_face[bufferIndex], 0);
      packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d);
    }else{
      packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[Nstream-1], my_face_d);
    }
  }
                                                      


  void cudaColorSpinorField::gather(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;

    // If stream_p != 0, use pack_stream, else use the stream array
    cudaStream_t *pack_stream = (stream_p) ? stream_p : stream+dir;

    if(dir%2 == 0){
      // backwards copy to host
      sendGhost(my_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, pack_stream);
    } else {
      // forwards copy to host
      sendGhost(my_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, pack_stream);
    }
  }


  void cudaColorSpinorField::recvStart(int nFace, int dir, int dagger) {
    int dim = dir/2;
    if(!commDimPartitioned(dim)) return;

    if (dir%2 == 0) { // sending backwards
      // Prepost receive
      comm_start(mh_recv_fwd[bufferIndex][nFace-1][dim]);
    } else { //sending forwards
      // Prepost receive
      comm_start(mh_recv_back[bufferIndex][nFace-1][dim]);
    }
#ifdef GPU_COMMS
    if(precision != QUDA_HALF_PRECISION) return;

    if (dir%2 == 0) { // sending backwards
      // Prepost receive
      comm_start(mh_recv_norm_fwd[bufferIndex][nFace-1][dim]);
    } else { //sending forwards
      // Prepost receive
      comm_start(mh_recv_norm_back[bufferIndex][nFace-1][dim]);
    }
#endif
  }

  void cudaColorSpinorField::sendStart(int nFace, int dir, int dagger) {
    int dim = dir / 2;
    if(!commDimPartitioned(dim)) return;

    if (dir%2 == 0) { // sending backwards
      comm_start(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]);
    } else { //sending forwards
      comm_start(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]);
    }
#ifdef GPU_COMMS
    if(precision != QUDA_HALF_PRECISION) return;
    if (dir%2 == 0) { // sending backwards
      comm_start(mh_send_norm_back[bufferIndex][nFace-1][2*dim+dagger]);
    } else { //sending forwards
      comm_start(mh_send_norm_fwd[bufferIndex][nFace-1][2*dim+dagger]);
    }
#endif
  }




 void cudaColorSpinorField::commsStart(int nFace, int dir, int dagger) {
    int dim = dir / 2;
    if(!commDimPartitioned(dim)) return;
    
    if (dir%2 == 0) { // sending backwards
      // Prepost receive
      comm_start(mh_recv_fwd[bufferIndex][nFace-1][dim]);
      comm_start(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]);
    } else { //sending forwards
      // Prepost receive
      comm_start(mh_recv_back[bufferIndex][nFace-1][dim]);
      // Begin forward send
      comm_start(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]);
    }
#ifdef GPU_COMMS
    if(precision != QUDA_HALF_PRECISION) return;		

    if (dir%2 == 0) { // sending backwards
      // Prepost receive
      comm_start(mh_recv_norm_fwd[bufferIndex][nFace-1][dim]);

      comm_start(mh_send_norm_back[bufferIndex][nFace-1][2*dim+dagger]);
    } else { //sending forwards
      // Prepost receive
      comm_start(mh_recv_norm_back[bufferIndex][nFace-1][dim]);
      // Begin forward send
      comm_start(mh_send_norm_fwd[bufferIndex][nFace-1][2*dim+dagger]);
    }
#endif
  }

  int cudaColorSpinorField::commsQuery(int nFace, int dir, int dagger) {
    int dim = dir / 2;
    if(!commDimPartitioned(dim)) return 0;

#ifdef GPU_COMMS
    if(precision != QUDA_HALF_PRECISION){
#endif
    if(dir%2==0) {
      if (comm_query(mh_recv_fwd[bufferIndex][nFace-1][dim]) && 
	  comm_query(mh_send_back[bufferIndex][nFace-1][2*dim+dagger])) return 1;
    } else {
      if (comm_query(mh_recv_back[bufferIndex][nFace-1][dim]) && 
	  comm_query(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger])) return 1;
    }
#ifdef GPU_COMMS
   }else{ // half precision
    if(dir%2==0) {
      if (comm_query(mh_recv_fwd[bufferIndex][nFace-1][dim]) && 
	  comm_query(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]) &&
	  comm_query(mh_recv_norm_fwd[bufferIndex][nFace-1][dim]) &&
	  comm_query(mh_send_norm_back[bufferIndex][nFace-1][2*dim+dagger])) return 1;
    } else {
      if (comm_query(mh_recv_back[bufferIndex][nFace-1][dim]) && 
	  comm_query(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]) &&
	  comm_query(mh_recv_norm_back[bufferIndex][nFace-1][dim]) && 
	  comm_query(mh_send_norm_fwd[bufferIndex][nFace-1][2*dim+dagger])) return 1;
    }
   } // half precision
#endif
    return 0;
  }

  void cudaColorSpinorField::commsWait(int nFace, int dir, int dagger) {
    int dim = dir / 2;
    if(!commDimPartitioned(dim)) return;

#ifdef GPU_COMMS
    if(precision != QUDA_HALF_PRECISION){
#endif
    if (dir%2==0) {
      comm_wait(mh_recv_fwd[bufferIndex][nFace-1][dim]);
      comm_wait(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]);
    } else {
      comm_wait(mh_recv_back[bufferIndex][nFace-1][dim]);
      comm_wait(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]);
    }
#ifdef GPU_COMMS
   } else { // half precision
      if (dir%2==0) {
	comm_wait(mh_recv_fwd[bufferIndex][nFace-1][dim]);
	comm_wait(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]);
	comm_wait(mh_recv_norm_fwd[bufferIndex][nFace-1][dim]);
	comm_wait(mh_send_norm_back[bufferIndex][nFace-1][2*dim+dagger]);
      } else {
	comm_wait(mh_recv_back[bufferIndex][nFace-1][dim]);
	comm_wait(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]);
	comm_wait(mh_recv_norm_back[bufferIndex][nFace-1][dim]);
	comm_wait(mh_send_norm_fwd[bufferIndex][nFace-1][2*dim+dagger]);
      }
    } // half precision
#endif

    return;
  }

  void cudaColorSpinorField::scatter(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;
    if(!commDimPartitioned(dim)) return;

    // both scattering occurances now go through the same stream
    if (dir%2==0) {// receive from forwards
      unpackGhost(from_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, stream_p);
    } else { // receive from backwards
      unpackGhost(from_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, stream_p);
    }
  }



  void cudaColorSpinorField::scatter(int nFace, int dagger, int dir)
  {
    int dim = dir/2;
    if(!commDimPartitioned(dim)) return;
    
    // both scattering occurances now go through the same stream
    if (dir%2==0) {// receive from forwards
      unpackGhost(from_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/]);
    } else { // receive from backwards
      unpackGhost(from_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+1*/]);
    }
  }

  
  void cudaColorSpinorField::scatterExtended(int nFace, int parity, int dagger, int dir)
  {
    int dim = dir/2;
    if(!commDimPartitioned(dim)) return;
    if (dir%2==0) {// receive from forwards
      unpackGhostExtended(from_fwd_face[bufferIndex][dim], nFace, static_cast<QudaParity>(parity), dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/]);
    } else { // receive from backwards
      unpackGhostExtended(from_back_face[bufferIndex][dim], nFace, static_cast<QudaParity>(parity),  dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+1*/]);
    }
  }
 

  // Return the location of the field
  QudaFieldLocation cudaColorSpinorField::Location() const { return QUDA_CUDA_FIELD_LOCATION; }

  std::ostream& operator<<(std::ostream &out, const cudaColorSpinorField &a) {
    out << (const ColorSpinorField&)a;
    out << "v = " << a.v << std::endl;
    out << "norm = " << a.norm << std::endl;
    out << "alloc = " << a.alloc << std::endl;
    out << "init = " << a.init << std::endl;
    return out;
  }

//! for deflated solvers:
  cudaColorSpinorField& cudaColorSpinorField::Eigenvec(const int idx) const {
    
    if (siteSubset == QUDA_PARITY_SITE_SUBSET && this->EigvId() == -1) {
      if (idx < this->EigvDim()) {//setup eigenvector form the set
        return *(dynamic_cast<cudaColorSpinorField*>(eigenvectors[idx])); 
      }
      else{
        errorQuda("Incorrect eigenvector index...");
      }
    }
    errorQuda("Eigenvector must be a parity spinor");
    exit(-1);
  }

//copyCuda currently cannot not work with set of spinor fields..
  void cudaColorSpinorField::CopyEigenvecSubset(cudaColorSpinorField &dst, const int range, const int first_element) const{
#if 0
    if(first_element < 0) errorQuda("\nError: trying to set negative first element.\n");
    if (siteSubset == QUDA_PARITY_SITE_SUBSET && this->EigvId() == -1) {
      if (first_element == 0 && range == this->EigvDim())
      {
        if(range != dst.EigvDim())errorQuda("\nError: eigenvector range to big.\n");
        checkField(dst, *this);
        copyCuda(dst, *this);
      }
      else if ((first_element+range) < this->EigvDim()) 
      {//setup eigenvector subset

        cudaColorSpinorField *eigv_subset;

        ColorSpinorParam param;

        param.nColor = nColor;
        param.nSpin = nSpin;
        param.twistFlavor = twistFlavor;
        param.precision = precision;
        param.nDim = nDim;
        param.pad = pad;
        param.siteSubset = siteSubset;
        param.siteOrder = siteOrder;
        param.fieldOrder = fieldOrder;
        param.gammaBasis = gammaBasis;
        memcpy(param.x, x, nDim*sizeof(int));
        param.create = QUDA_REFERENCE_FIELD_CREATE;
 
        param.eigv_dim  = range;
        param.eigv_id   = -1;
        param.v = (void*)((char*)v + first_element*eigv_bytes);
        param.norm = (void*)((char*)norm + first_element*eigv_norm_bytes);

        eigv_subset = new cudaColorSpinorField(param);

        //Not really needed:
        eigv_subset->eigenvectors.reserve(param.eigv_dim);
        for(int id = first_element; id < (first_element+range); id++)
        {
            param.eigv_id = id;
            eigv_subset->eigenvectors.push_back(new cudaColorSpinorField(*this, param));
        }
        checkField(dst, *eigv_subset);
        copyCuda(dst, *eigv_subset);

        delete eigv_subset;
      }
      else{
        errorQuda("Incorrect eigenvector dimension...");
      }
    }
    else{  
      errorQuda("Eigenvector must be a parity spinor");
      exit(-1);
    }
#endif
  }

  void cudaColorSpinorField::getTexObjectInfo() const
  {
#ifdef USE_TEXTURE_OBJECTS
    printfQuda("\nPrint texture info for the field:\n");
    std::cout << *this;
    cudaResourceDesc resDesc;
    //memset(&resDesc, 0, sizeof(resDesc));
    cudaGetTextureObjectResourceDesc(&resDesc, this->Tex());
    printfQuda("\nDevice pointer: %p\n", resDesc.res.linear.devPtr);
    printfQuda("\nVolume (in bytes): %d\n", resDesc.res.linear.sizeInBytes);
    if (resDesc.resType == cudaResourceTypeLinear) printfQuda("\nResource type: linear \n");
    checkCudaError();
#endif
  }

} // namespace quda

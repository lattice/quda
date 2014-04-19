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

  int cudaColorSpinorField::initGhostFaceBuffer = 0;
  void* cudaColorSpinorField::ghostFaceBuffer; //gpu memory
  void* cudaColorSpinorField::fwdGhostFaceBuffer[QUDA_MAX_DIM]; //pointers to ghostFaceBuffer
  void* cudaColorSpinorField::backGhostFaceBuffer[QUDA_MAX_DIM]; //pointers to ghostFaceBuffer
  size_t cudaColorSpinorField::ghostFaceBytes = 0;

  /*cudaColorSpinorField::cudaColorSpinorField() : 
    ColorSpinorField(), v(0), norm(0), alloc(false), init(false) {

    }*/

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
    ColorSpinorField(param), alloc(false), init(true), texInit(false), 
    initComms(false), nFaceComms(0) {

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
    initComms(false), nFaceComms(0) {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, 
					     const ColorSpinorParam &param) :
    ColorSpinorField(src), alloc(false), init(true), texInit(false), 
    initComms(false), nFaceComms(0) {  

    // can only overide if we are not using a reference or parity special case
    if (param.create != QUDA_REFERENCE_FIELD_CREATE || 
	(param.create == QUDA_REFERENCE_FIELD_CREATE && 
	 src.SiteSubset() == QUDA_FULL_SITE_SUBSET && 
	 param.siteSubset == QUDA_PARITY_SITE_SUBSET && 
	 typeid(src) == typeid(cudaColorSpinorField) ) ) {
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

    clearGhostPointers();
  }

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) 
    : ColorSpinorField(src), alloc(false), init(true), texInit(false), 
      initComms(false), nFaceComms(0) {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
    clearGhostPointers();
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

      // need this hackery for the moment (need to locate the odd pointer half way into the full field)
      (dynamic_cast<cudaColorSpinorField*>(odd))->v = (void*)((char*)v + bytes/2);
      if (precision == QUDA_HALF_PRECISION) 
	(dynamic_cast<cudaColorSpinorField*>(odd))->norm = (void*)((char*)norm + norm_bytes/2);

#ifdef USE_TEXTURE_OBJECTS
      dynamic_cast<cudaColorSpinorField*>(even)->destroyTexObject();
      dynamic_cast<cudaColorSpinorField*>(even)->createTexObject();
      dynamic_cast<cudaColorSpinorField*>(odd)->destroyTexObject();
      dynamic_cast<cudaColorSpinorField*>(odd)->createTexObject();
#endif
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
    createTexObject();
#endif

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
      if (siteSubset == QUDA_FULL_SITE_SUBSET) {
	delete even;
	delete odd;
      }
      alloc = false;
    }

#ifdef USE_TEXTURE_OBJECTS
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
    for (int i=0; i<Npad; i++) {
      if (pad_bytes) cudaMemset((char*)v + (volume + i*stride)*fieldOrder*precision, 0, pad_bytes);
    }
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
      resizeBufferPinned(bytes + norm_bytes);
      memset(bufferPinned, 0, bytes+norm_bytes); // FIXME (temporary?) bug fix for padding

      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, 
			     bufferPinned, 0, (char*)bufferPinned+bytes, 0);

      cudaMemcpy(v, bufferPinned, bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(norm, (char*)bufferPinned+bytes, norm_bytes, cudaMemcpyHostToDevice);
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
	resizeBufferPinned(src.Bytes()+src.NormBytes());
	memcpy(bufferPinned, src.V(), src.Bytes());
	memcpy((char*)bufferPinned+src.Bytes(), src.Norm(), src.NormBytes());

	cudaHostGetDevicePointer(&Src, bufferPinned, 0);
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
      resizeBufferPinned(bytes+norm_bytes);
      cudaMemcpy(bufferPinned, v, bytes, cudaMemcpyDeviceToHost);
      cudaMemcpy((char*)bufferPinned+bytes, norm, norm_bytes, cudaMemcpyDeviceToHost);

      copyGenericColorSpinor(dest, *this, QUDA_CPU_FIELD_LOCATION, 
			     0, bufferPinned, 0, (char*)bufferPinned+bytes);
    } else if (typeid(dest) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION);
    } else {
      void *dst, *dstNorm;
      if (!zeroCopy) {
	resizeBufferDevice(dest.Bytes()+dest.NormBytes());
	dst = bufferDevice;
	dstNorm = (char*)bufferDevice+dest.Bytes();
      } else {
	resizeBufferPinned(dest.Bytes()+dest.NormBytes());
	cudaHostGetDevicePointer(&dst, bufferPinned, 0);
	dstNorm = (char*)dst+dest.Bytes();
      }
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dst, v, dstNorm, 0);

      if (!zeroCopy) {
	cudaMemcpy(dest.V(), dst, dest.Bytes(), cudaMemcpyDeviceToHost);
	cudaMemcpy(dest.Norm(), dstNorm, dest.NormBytes(), cudaMemcpyDeviceToHost);
      } else {
	memcpy(dest.V(), bufferPinned, dest.Bytes());
	memcpy(dest.Norm(), (char*)bufferPinned+dest.Bytes(), dest.NormBytes());
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

      if (initGhostFaceBuffer) device_free(ghostFaceBuffer); 

      if (faceBytes > 0) {
	ghostFaceBuffer = device_malloc(faceBytes);
	initGhostFaceBuffer = 1;
	ghostFaceBytes = faceBytes;
      }

    }

    size_t offset = 0;
    for (int i=0; i<4; i++) {
      if(!commDimPartitioned(i)) continue;
      
      backGhostFaceBuffer[i] = (void*)(((char*)ghostFaceBuffer) + offset);
      offset += nFace*ghostFace[i]*Nint*precision;
      if (precision == QUDA_HALF_PRECISION) offset += nFace*ghostFace[i]*sizeof(float);
      
      fwdGhostFaceBuffer[i] = (void*)(((char*)ghostFaceBuffer) + offset);
      offset += nFace*ghostFace[i]*Nint*precision;
      if (precision == QUDA_HALF_PRECISION) offset += nFace*ghostFace[i]*sizeof(float);
    }   
    
  }


  void cudaColorSpinorField::freeGhostBuffer(void)
  {
    if (!initGhostFaceBuffer) return;
  
    device_free(ghostFaceBuffer); 

    for(int i=0;i < 4; i++){
      if(!commDimPartitioned(i)) continue;
      backGhostFaceBuffer[i] = NULL;
      fwdGhostFaceBuffer[i] = NULL;
    }
    initGhostFaceBuffer = 0;  
  }

  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhost(FullClover &clov, FullClover &clovInv,
				       const int nFace, const QudaParity parity, 
                                       const int dim, const QudaDirection dir,
				       const int dagger, cudaStream_t *stream, 
				       void *buffer, double a) 
  {
    int face_num;
    if(dir == QUDA_BACKWARDS){
      face_num = 0;
    }else if(dir == QUDA_FORWARDS){
      face_num = 1;
    }else{
      face_num = 2;
    }
#ifdef MULTI_GPU
    void *packBuffer = buffer ? buffer : ghostFaceBuffer;
    packFace(packBuffer, *this, clov, clovInv, nFace, dagger, parity, dim, face_num, *stream, a); 
#else
    errorQuda("packGhost not built on single-GPU build");
#endif

  }
 
  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhost(const int nFace, const QudaParity parity, 
                                       const int dim, const QudaDirection dir,
				       const int dagger, cudaStream_t *stream, 
				       void *buffer, double a, double b) 
  {
    int face_num;
    if(dir == QUDA_BACKWARDS){
      face_num = 0;
    }else if(dir == QUDA_FORWARDS){
      face_num = 1;
    }else{
      face_num = 2;
    }
#ifdef MULTI_GPU
    void *packBuffer = buffer ? buffer : ghostFaceBuffer;
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
	(dir == QUDA_BACKWARDS) ? this->backGhostFaceBuffer[dim] : this->fwdGhostFaceBuffer[dim];

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
    int face_num;
    if(dir == QUDA_BACKWARDS){
      face_num = 0;
    }else if(dir == QUDA_FORWARDS){
      face_num = 1;
    }else{
      face_num = 2;
    }
#ifdef MULTI_GPU
    void *packBuffer = buffer ? buffer : ghostFaceBuffer;
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
    packFaceExtended(ghostFaceBuffer, *this, nFace, R, dagger, parity, dim, face_num, *stream, unpack); 
#else
    errorQuda("unpackGhostExtended not built on single-GPU build");
#endif
  }



  cudaStream_t *stream;

  void cudaColorSpinorField::createComms(int nFace) {
    if (!initComms || nFaceComms != nFace) {

      // if we are requesting a new number of faces destroy and start over
      if(nFace != nFaceComms) destroyComms();

      if (siteSubset != QUDA_PARITY_SITE_SUBSET) 
	errorQuda("Only supports single parity fields");

#ifdef GPU_COMMS
      bool comms = false;
      for (int i=0; i<nDimComms; i++) if (commDimPartitioned(i)) comms = true;
      if (comms && precision == QUDA_HALF_PRECISION)
	errorQuda("GPU-aware communication not yet supported with half precision");
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
	if (siteSubset == QUDA_PARITY_SITE_SUBSET && i==0) nbytes[i] /= 2;
	if (!commDimPartitioned(i)) continue;
	faceBytes += 2*nbytes[i];
      }
      
      // use static pinned memory for face buffers
      resizeBufferPinned(2*faceBytes); // oversizes for GPU_COMMS case

      my_face = bufferPinned;
      from_face = static_cast<char*>(bufferPinned) + faceBytes;

      // assign pointers for each face - it's ok to alias for different Nface parameters
#ifndef GPU_COMMS
      size_t offset = 0;
#endif
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;
	
#ifdef GPU_COMMS
	size_t offset2 = precision*(length + ghostOffset[i]*nColor*nSpin*2);
	my_back_face[i] = backGhostFaceBuffer[i];
	from_back_face[i] = static_cast<char*>(v) + offset2;
#else
	my_back_face[i] = static_cast<char*>(my_face) + offset;
	from_back_face[i] = static_cast<char*>(from_face) + offset;
	offset += nbytes[i];
#endif
	
#ifdef GPU_COMMS
	offset2 += nFace*ghostFace[i]*Ndof*precision;
	my_fwd_face[i] = fwdGhostFaceBuffer[i];	
	from_fwd_face[i] = /*static_cast<char*>(ghost[i]) + nFace*ghostFace[i]*Ndof*precision; //*/static_cast<char*>(v) + offset2;
#else
	my_fwd_face[i] = static_cast<char*>(my_face) + offset;
	from_fwd_face[i] = static_cast<char*>(from_face) + offset;
	offset += nbytes[i];
#endif

      }

      // create a different message handler for each direction and Nface
      mh_send_fwd = new MsgHandle**[maxNface];
      mh_send_back = new MsgHandle**[maxNface];
      mh_recv_fwd = new MsgHandle**[maxNface];
      mh_recv_back = new MsgHandle**[maxNface];
      for (int j=0; j<maxNface; j++) {
	mh_send_fwd[j] = new MsgHandle*[2*nDimComms];
	mh_send_back[j] = new MsgHandle*[2*nDimComms];
	mh_recv_fwd[j] = new MsgHandle*[nDimComms];
	mh_recv_back[j] = new MsgHandle*[nDimComms];

	for (int i=0; i<nDimComms; i++) {
	  size_t nbytes_Nface = (nbytes[i] / maxNface) * (j+1);
	  if (!commDimPartitioned(i)) continue;
#ifdef GPU_COMMS
	  if (i != 3 || getKernelPackT()) {
#endif
	    mh_send_fwd[j][2*i+0] = comm_declare_send_relative(my_fwd_face[i], i, +1, nbytes_Nface);
	    mh_send_back[j][2*i+0] = comm_declare_send_relative(my_back_face[i], i, -1, nbytes_Nface);
	    mh_send_fwd[j][2*i+1] = mh_send_fwd[j][2*i]; // alias pointers
	    mh_send_back[j][2*i+1] = mh_send_back[j][2*i]; // alias pointers
#ifdef GPU_COMMS
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

	    //printf("\n%d strided sends with Nface=%d Nblocks=%d blksize=%d Stride=%d,
	    //	   i, j+1, Nblocks, blksize, Stride);

	    mh_send_fwd[j][2*i+0] = comm_declare_strided_send_relative(base[2], i, +1, 
								       blksize, Nblocks, Stride);
	    mh_send_back[j][2*i+0] = comm_declare_strided_send_relative(base[0], i, -1, 
									blksize, Nblocks, Stride);
	    if (nSpin ==4) { // dagger communicators
	      mh_send_fwd[j][2*i+1] = comm_declare_strided_send_relative(base[3], i, +1, 
									 blksize, Nblocks, Stride);
	      mh_send_back[j][2*i+1] = comm_declare_strided_send_relative(base[1], i, -1, 
									  blksize, Nblocks, Stride);
	    } else {
	      mh_send_fwd[j][2*i+1] = mh_send_fwd[j][2*i+0];
	      mh_send_back[j][2*i+1] = mh_send_back[j][2*i+0];
	    }
	  }
#endif // GPU_COMMS

	  mh_recv_fwd[j][i] = comm_declare_receive_relative(from_fwd_face[i], i, +1, nbytes_Nface);
	  mh_recv_back[j][i] = comm_declare_receive_relative(from_back_face[i], i, -1, nbytes_Nface);

	} // loop over dimension
      }
      
      initComms = true;
      nFaceComms = nFace;
    }
    checkCudaError();
  }
    
  void cudaColorSpinorField::destroyComms() {
    if (initComms) {
      for (int j=0; j<maxNface; j++) {
	for (int i=0; i<nDimComms; i++) {
	  if (commDimPartitioned(i)) {
	    comm_free(mh_recv_fwd[j][i]);
	    comm_free(mh_recv_back[j][i]);
	    comm_free(mh_send_fwd[j][2*i]);
	    comm_free(mh_send_back[j][2*i]);
	    // only in a special case are these not aliasing pointers
#ifdef GPU_COMMS
	    if (i == 3 && !getKernelPackT() && nSpin == 4) {
	      comm_free(mh_send_fwd[j][2*i+1]);
	      comm_free(mh_send_back[j][2*i+1]);
	    }
#endif // GPU_COMMS
	  }
	}
	delete []mh_recv_fwd[j];
	delete []mh_recv_back[j];
	delete []mh_send_fwd[j];
	delete []mh_send_back[j];
      }    
      delete []mh_recv_fwd;
      delete []mh_recv_back;
      delete []mh_send_fwd;
      delete []mh_send_back;
      
      for (int i=0; i<nDimComms; i++) {
	my_fwd_face[i] = NULL;
	my_back_face[i] = NULL;
	from_fwd_face[i] = NULL;
	from_back_face[i] = NULL;      
      }
      
      initComms = false;
      checkCudaError();
    }
  }

  void cudaColorSpinorField::pack(FullClover &clov, FullClover &clovInv, int nFace, int parity,
				  int dagger, cudaStream_t *stream_p, bool zeroCopyPack, double a) {
    allocateGhostBuffer(nFace);   // allocate the ghost buffer if not yet allocated  
    createComms(nFace); // must call this first

    stream = stream_p;
    
    const int dim=-1; // pack all partitioned dimensions
 
    if (zeroCopyPack) {
      void *my_face_d;
      cudaHostGetDevicePointer(&my_face_d, my_face, 0); // set the matching device pointer
      packGhost(clov, clovInv, nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d, a);
    } else {
      packGhost(clov, clovInv, nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger,  &stream[Nstream-1], 0, a);
    }
  }

  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, cudaStream_t *stream_p, 
				  bool zeroCopyPack, double a, double b) {
    allocateGhostBuffer(nFace);   // allocate the ghost buffer if not yet allocated  
    createComms(nFace); // must call this first

    stream = stream_p;
    
    const int dim=-1; // pack all partitioned dimensions
 
    if (zeroCopyPack) {
      void *my_face_d;
      cudaHostGetDevicePointer(&my_face_d, my_face, 0); // set the matching device pointer
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d, a, b);
    } else {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger,  &stream[Nstream-1], 0, a, b);
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
      cudaHostGetDevicePointer(&my_face_d, my_face, 0);
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
      sendGhost(my_back_face[dim], nFace, dim, QUDA_BACKWARDS, dagger, pack_stream);
    } else {
      // forwards copy to host
      sendGhost(my_fwd_face[dim], nFace, dim, QUDA_FORWARDS, dagger, pack_stream);
    }
  }



  void cudaColorSpinorField::recvStart(int nFace, int dir, int dagger) {
    int dim = dir/2;
    if(!commDimPartitioned(dim)) return;

    if (dir%2 == 0) { // sending backwards
      // Prepost receive
      comm_start(mh_recv_fwd[nFace-1][dim]);
    } else { //sending forwards
      // Prepost receive
      comm_start(mh_recv_back[nFace-1][dim]);
    }
  }

  void cudaColorSpinorField::sendStart(int nFace, int dir, int dagger) {
    int dim = dir / 2;
    if(!commDimPartitioned(dim)) return;

    if (dir%2 == 0) { // sending backwards
      comm_start(mh_send_back[nFace-1][2*dim+dagger]);
    } else { //sending forwards
      comm_start(mh_send_fwd[nFace-1][2*dim+dagger]);
    }
  }




  void cudaColorSpinorField::commsStart(int nFace, int dir, int dagger) {
    int dim = dir / 2;
    if(!commDimPartitioned(dim)) return;
    
    if (dir%2 == 0) { // sending backwards
      // Prepost receive
      comm_start(mh_recv_fwd[nFace-1][dim]);
      comm_start(mh_send_back[nFace-1][2*dim+dagger]);
    } else { //sending forwards
      // Prepost receive
      comm_start(mh_recv_back[nFace-1][dim]);
      // Begin forward send
      comm_start(mh_send_fwd[nFace-1][2*dim+dagger]);
    }
  }

  int cudaColorSpinorField::commsQuery(int nFace, int dir, int dagger) {
    int dim = dir / 2;
    if(!commDimPartitioned(dim)) return 0;
    
    if(dir%2==0) {
      if (comm_query(mh_recv_fwd[nFace-1][dim]) && 
	  comm_query(mh_send_back[nFace-1][2*dim+dagger])) return 1;
    } else {
      if (comm_query(mh_recv_back[nFace-1][dim]) && 
	  comm_query(mh_send_fwd[nFace-1][2*dim+dagger])) return 1;
    }
    
    return 0;
  }

  void cudaColorSpinorField::scatter(int nFace, int dagger, int dir)
  {
    int dim = dir/2;
    if(!commDimPartitioned(dim)) return;
    
    // both scattering occurances now go through the same stream
    if (dir%2==0) {// receive from forwards
      unpackGhost(from_fwd_face[dim], nFace, dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/]);
    } else { // receive from backwards
      unpackGhost(from_back_face[dim], nFace, dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+1*/]);
    }
  }

  
  void cudaColorSpinorField::scatterExtended(int nFace, int parity, int dagger, int dir)
  {
    int dim = dir/2;
    if(!commDimPartitioned(dim)) return;
    if (dir%2==0) {// receive from forwards
      unpackGhostExtended(from_fwd_face[dim], nFace, static_cast<QudaParity>(parity), dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/]);
    } else { // receive from backwards
      unpackGhostExtended(from_back_face[dim], nFace, static_cast<QudaParity>(parity),  dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+1*/]);
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

} // namespace quda

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
static const QudaFieldLocation reorder_location_ = QUDA_CUDA_FIELD_LOCATION;
#else
static const QudaFieldLocation reorder_location_ = QUDA_CPU_FIELD_LOCATION;
#endif

int zeroCopy = 0;

namespace quda {

  int cudaColorSpinorField::bufferIndex = 0;
  bool cudaColorSpinorField::initGhostFaceBuffer = false;
  void *cudaColorSpinorField::ghost_field[2] = {nullptr, nullptr};
  void* cudaColorSpinorField::ghostFaceBuffer[2] = {nullptr, nullptr}; //gpu memory
  void* cudaColorSpinorField::fwdGhostFaceBuffer[2][QUDA_MAX_DIM]; //pointers to ghostFaceBuffer
  void* cudaColorSpinorField::backGhostFaceBuffer[2][QUDA_MAX_DIM]; //pointers to ghostFaceBuffer
  size_t cudaColorSpinorField::ghostFaceBytes = 0;

  bool cudaColorSpinorField::initIPCComms = false;

  int cudaColorSpinorField::buffer_send_p2p_fwd[2][QUDA_MAX_DIM];
  int cudaColorSpinorField::buffer_recv_p2p_fwd[2][QUDA_MAX_DIM];
  int cudaColorSpinorField::buffer_send_p2p_back[2][QUDA_MAX_DIM];
  int cudaColorSpinorField::buffer_recv_p2p_back[2][QUDA_MAX_DIM];

  MsgHandle* cudaColorSpinorField::mh_send_p2p_fwd[2][QUDA_MAX_DIM];
  MsgHandle* cudaColorSpinorField::mh_send_p2p_back[2][QUDA_MAX_DIM];
  MsgHandle* cudaColorSpinorField::mh_recv_p2p_fwd[2][QUDA_MAX_DIM];
  MsgHandle* cudaColorSpinorField::mh_recv_p2p_back[2][QUDA_MAX_DIM];

  cudaEvent_t cudaColorSpinorField::ipcCopyEvent[2][2][QUDA_MAX_DIM];
  cudaEvent_t cudaColorSpinorField::ipcRemoteCopyEvent[2][2][QUDA_MAX_DIM];

  void* cudaColorSpinorField::fwdGhostSendDest[2][QUDA_MAX_DIM];
  void* cudaColorSpinorField::backGhostSendDest[2][QUDA_MAX_DIM];

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
    ColorSpinorField(param), alloc(false), init(true), texInit(false),
    ghostTexInit(false), initComms(false), bufferMessageHandler(0)
  {
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
      // do nothing
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      errorQuda("not implemented");
    }
  }

  cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : 
    ColorSpinorField(src), alloc(false), init(true), texInit(false),
    ghostTexInit(false), initComms(false), bufferMessageHandler(0)
  {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, 
					     const ColorSpinorParam &param) :
    ColorSpinorField(src), alloc(false), init(true), texInit(false),
    ghostTexInit(false), initComms(false), bufferMessageHandler(0)
  {
    // can only overide if we are not using a reference or parity special case
    if (param.create != QUDA_REFERENCE_FIELD_CREATE || 
	(param.create == QUDA_REFERENCE_FIELD_CREATE && 
	 src.SiteSubset() == QUDA_FULL_SITE_SUBSET && 
	 param.siteSubset == QUDA_PARITY_SITE_SUBSET && 
	 typeid(src) == typeid(cudaColorSpinorField) ) || 
         (param.create == QUDA_REFERENCE_FIELD_CREATE && (param.is_composite || param.is_component))) {
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

      if (composite_descr.is_component && !(src.SiteSubset() == QUDA_FULL_SITE_SUBSET && this->SiteSubset() == QUDA_PARITY_SITE_SUBSET)) 
      {//setup eigenvector form the set
        v    = (void*)((char*)v    + composite_descr.id*bytes);         
        norm = (void*)((char*)norm + composite_descr.id*norm_bytes);         
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
      ghostTexInit(false), initComms(false), bufferMessageHandler(0)
  {
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

  void cudaColorSpinorField::create(const QudaFieldCreate create) {

    if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER) {
      errorQuda("Subset not implemented");
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      v = device_malloc(bytes);
      if (precision == QUDA_HALF_PRECISION) norm = device_malloc(norm_bytes);
      alloc = true;
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      if(composite_descr.is_composite && (create != QUDA_REFERENCE_FIELD_CREATE)) {
	if(composite_descr.dim <= 0) errorQuda("\nComposite size is not defined\n");
	  
        ColorSpinorParam param;
        param.siteSubset = QUDA_FULL_SITE_SUBSET;
        param.nDim = nDim;
        memcpy(param.x, x, nDim*sizeof(int));
        param.create = QUDA_REFERENCE_FIELD_CREATE;
        param.v = v;
        param.norm = norm;
        param.is_composite   = false;
        param.composite_dim  = 0;
        param.is_component = true;

        components.reserve(composite_descr.dim);
        for(int cid = 0; cid < composite_descr.dim; cid++) {
	  param.component_id = cid;
	  components.push_back(new cudaColorSpinorField(*this, param));
        }
      } else {
        // create the associated even and odd subsets
        ColorSpinorParam param;
        param.siteSubset = QUDA_PARITY_SITE_SUBSET;
        param.nDim = nDim;
        memcpy(param.x, x, nDim*sizeof(int));
        param.x[0] /= 2; // set single parity dimensions
        param.create = QUDA_REFERENCE_FIELD_CREATE;
        param.v = v;
        param.norm = norm;
        param.is_composite  = false;
        param.composite_dim = 0;
        param.is_component  = composite_descr.is_component;
        param.component_id  = composite_descr.id;
        even = new cudaColorSpinorField(*this, param);
        odd = new cudaColorSpinorField(*this, param);

        // need this hackery for the moment (need to locate the odd pointers half way into the full field)
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
    } else { //siteSubset == QUDA_PARITY_SITE_SUBSET

      //! setup an object for selected eigenvector (the 1st one as a default):
      if (composite_descr.is_composite && (create != QUDA_REFERENCE_FIELD_CREATE)) 
      {
         if(composite_descr.dim <= 0) errorQuda("\nComposite size is not defined\n");
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
         param.is_composite   = false;
         param.composite_dim  = 0;
         param.is_component = true;
         //reserve eigvector set
         components.reserve(composite_descr.dim);
         //setup volume, [real_]length and stride for a single eigenvector
         for(int cid = 0; cid < composite_descr.dim; cid++)
         {
            param.component_id = cid;
            components.push_back(new cudaColorSpinorField(*this, param));

#ifdef USE_TEXTURE_OBJECTS //(a lot of texture objects...)
            dynamic_cast<cudaColorSpinorField*>(components[cid])->destroyTexObject();
            dynamic_cast<cudaColorSpinorField*>(components[cid])->createTexObject();
#endif
         }
      }
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (siteSubset != QUDA_FULL_SITE_SUBSET) {
	zeroPad();
      } else if(!composite_descr.is_composite) {
	(dynamic_cast<cudaColorSpinorField*>(even))->zeroPad();
	(dynamic_cast<cudaColorSpinorField*>(odd))->zeroPad();
      } else { //temporary hack for the full spinor field sets, manual zeroPad for each component:
	for(int cid = 0; cid < composite_descr.dim; cid++) {
	  (dynamic_cast<cudaColorSpinorField&>(components[cid]->Even())).zeroPad();
	  (dynamic_cast<cudaColorSpinorField&>(components[cid]->Odd())).zeroPad();
	}
      }
    }

#ifdef USE_TEXTURE_OBJECTS
    if (!composite_descr.is_composite || composite_descr.is_component)
      createTexObject();
#endif
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
      
      // staggered and coarse fields in half and single are always two component
      if ( (nSpin == 1 || nSpin == 2) && (precision == QUDA_HALF_PRECISION || precision == QUDA_SINGLE_PRECISION)) {
	desc.x = 8*precision;
	desc.y = 8*precision;
	desc.z = 0;
	desc.w = 0;
      } else { // all others are four component (double2 is spread across int4)
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
      }
      
      texInit = true;

      checkCudaError();
    }
  }

  void cudaColorSpinorField::createGhostTexObject() {
    // create the ghost texture object
    if (isNative() && ghost_bytes) {
      if (ghostTexInit) errorQuda("Already bound ghost texture");

      for (int b=0; b<2; b++) {
	cudaChannelFormatDesc desc;
	memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
	else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

	// staggered and coarse fields in half and single are always two component
	if ( (nSpin == 1 || nSpin == 2) && (precision == QUDA_HALF_PRECISION || precision == QUDA_SINGLE_PRECISION)) {
	  desc.x = 8*precision;
	  desc.y = 8*precision;
	  desc.z = 0;
	  desc.w = 0;
	} else { // all others are four component (double2 is spread across int4)
	  desc.x = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	  desc.y = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	  desc.z = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	  desc.w = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	}

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = ghost_field[b];
	resDesc.res.linear.desc = desc;
	resDesc.res.linear.sizeInBytes = ghost_bytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
	else texDesc.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&ghostTex[b], &resDesc, &texDesc, NULL);

	if (precision == QUDA_HALF_PRECISION) {
	  cudaChannelFormatDesc desc;
	  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	  desc.f = cudaChannelFormatKindFloat;
	  desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;

	  cudaResourceDesc resDesc;
	  memset(&resDesc, 0, sizeof(resDesc));
	  resDesc.resType = cudaResourceTypeLinear;
	  resDesc.res.linear.devPtr = ghost_field[b];
	  resDesc.res.linear.desc = desc;
	  resDesc.res.linear.sizeInBytes = ghost_bytes;

	  cudaTextureDesc texDesc;
	  memset(&texDesc, 0, sizeof(texDesc));
	  texDesc.readMode = cudaReadModeElementType;

	  cudaCreateTextureObject(&ghostTexNorm[b], &resDesc, &texDesc, NULL);
	}
      } // buffer index

      ghostTexInit = true;

      checkCudaError();
    }
  }

  void cudaColorSpinorField::destroyTexObject() {
    if (isNative() && texInit) {
      cudaDestroyTextureObject(tex);
      if (ghost_bytes) {
	cudaDestroyTextureObject(ghostTex[0]);
	cudaDestroyTextureObject(ghostTex[1]);
      }
      if (precision == QUDA_HALF_PRECISION) {
        cudaDestroyTextureObject(texNorm);
        if (ghost_bytes) {
	  cudaDestroyTextureObject(ghostTexNorm[0]);
	  cudaDestroyTextureObject(ghostTexNorm[1]);
	}
      }
      texInit = false;
    }
  }

  void cudaColorSpinorField::destroyGhostTexObject() {
    if (isNative() && ghostTexInit) {
      cudaDestroyTextureObject(ghostTex[0]);
      cudaDestroyTextureObject(ghostTex[1]);
      if (precision == QUDA_HALF_PRECISION) {
	cudaDestroyTextureObject(ghostTexNorm[0]);
	cudaDestroyTextureObject(ghostTexNorm[1]);
      }
      ghostTexInit = false;
    }
  }
#endif

  void cudaColorSpinorField::destroy() {
    if (alloc) {
      device_free(v);
      if (precision == QUDA_HALF_PRECISION) device_free(norm);
      alloc = false;
    }

    if (composite_descr.is_composite) 
    {
       CompositeColorSpinorField::iterator vec;
       for(vec = components.begin(); vec != components.end(); vec++) delete *vec;
    } 

    if ((siteSubset == QUDA_FULL_SITE_SUBSET && !composite_descr.is_composite) || (siteSubset == QUDA_FULL_SITE_SUBSET && composite_descr.is_component)) {
      delete even;
      delete odd;
    }

#ifdef USE_TEXTURE_OBJECTS
    if (!composite_descr.is_composite || composite_descr.is_component)
      destroyTexObject();
#endif

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

    if (composite_descr.is_composite && !composite_descr.is_component){//we consider the whole eigenvector set:
      Npad      *= composite_descr.dim;
      pad_bytes /= composite_descr.dim;
    }

    size_t pitch = ((!composite_descr.is_composite || composite_descr.is_component) ? stride : composite_descr.stride)*fieldOrder*precision;
    char   *dst  = (char*)v + ((!composite_descr.is_composite || composite_descr.is_component) ? volume : composite_descr.volume)*fieldOrder*precision;
    if (pad_bytes) cudaMemset2D(dst, pitch, 0, pad_bytes, Npad);

    //for (int i=0; i<Npad; i++) {
    //  if (pad_bytes) cudaMemset((char*)v + (volume + i*stride)*fieldOrder*precision, 0, pad_bytes);
    //}
  }

  void cudaColorSpinorField::copy(const cudaColorSpinorField &src) {
    checkField(*this, src);
    if (this->GammaBasis() != src.GammaBasis()) errorQuda("cannot call this copy with different basis");
    blas::copy(*this, src);
  }

  void cudaColorSpinorField::copySpinorField(const ColorSpinorField &src) {
    
    // src is on the device and is native
    if (typeid(src) == typeid(cudaColorSpinorField) && 
	isNative() && dynamic_cast<const cudaColorSpinorField &>(src).isNative() &&
	this->GammaBasis() == src.GammaBasis()) {
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

    if (reorder_location_ == QUDA_CPU_FIELD_LOCATION &&typeid(src) == typeid(cpuColorSpinorField)) {
      void *buffer = allocatePinned(bytes + norm_bytes);
      memset(buffer, 0, bytes+norm_bytes); // FIXME (temporary?) bug fix for padding

      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, 0, static_cast<char*>(buffer)+bytes, 0);

      qudaMemcpy(v, buffer, bytes, cudaMemcpyHostToDevice);
      qudaMemcpy(norm, static_cast<char*>(buffer)+bytes, norm_bytes, cudaMemcpyHostToDevice);

      freePinned(buffer);
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else {
      void *Src=nullptr, *srcNorm=nullptr, *buffer=nullptr;
      if (!zeroCopy) {
	resizeBufferDevice(src.Bytes()+src.NormBytes());
	Src = bufferDevice;
	srcNorm = (char*)bufferDevice + src.Bytes();
	qudaMemcpy(Src, src.V(), src.Bytes(), cudaMemcpyHostToDevice);
	qudaMemcpy(srcNorm, src.Norm(), src.NormBytes(), cudaMemcpyHostToDevice);
      } else {
	buffer = allocatePinned(src.Bytes()+src.NormBytes());
	memcpy(buffer, src.V(), src.Bytes());
	memcpy(static_cast<char*>(buffer)+src.Bytes(), src.Norm(), src.NormBytes());

	cudaHostGetDevicePointer(&Src, buffer, 0);
	srcNorm = (void*)((char*)Src + src.Bytes());
      }

      cudaMemset(v, 0, bytes); // FIXME (temporary?) bug fix for padding
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, 0, Src, 0, srcNorm);

      if (zeroCopy) freePinned(buffer);
    }

    return;
  }


  void cudaColorSpinorField::saveSpinorField(ColorSpinorField &dest) const {

    if (reorder_location_ == QUDA_CPU_FIELD_LOCATION &&	typeid(dest) == typeid(cpuColorSpinorField)) {
      void *buffer = allocatePinned(bytes+norm_bytes);
      qudaMemcpy(buffer, v, bytes, cudaMemcpyDeviceToHost);
      qudaMemcpy(static_cast<char*>(buffer)+bytes, norm, norm_bytes, cudaMemcpyDeviceToHost);

      copyGenericColorSpinor(dest, *this, QUDA_CPU_FIELD_LOCATION, 0, buffer, 0, static_cast<char*>(buffer)+bytes);
      freePinned(buffer);
    } else if (typeid(dest) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION);
    } else {
      void *dst=nullptr, *dstNorm=nullptr, *buffer=nullptr;
      if (!zeroCopy) {
	resizeBufferDevice(dest.Bytes()+dest.NormBytes());
	dst = bufferDevice;
	dstNorm = (char*)bufferDevice+dest.Bytes();
      } else {
	buffer = allocatePinned(dest.Bytes()+dest.NormBytes());
	cudaHostGetDevicePointer(&dst, buffer, 0);
	dstNorm = (char*)dst+dest.Bytes();
      }
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dst, v, dstNorm, 0);

      if (!zeroCopy) {
	qudaMemcpy(dest.V(), dst, dest.Bytes(), cudaMemcpyDeviceToHost);
	qudaMemcpy(dest.Norm(), dstNorm, dest.NormBytes(), cudaMemcpyDeviceToHost);
      } else {
	memcpy(dest.V(), buffer, dest.Bytes());
	memcpy(dest.Norm(), static_cast<char*>(buffer) + dest.Bytes(), dest.NormBytes());
      }

      if (zeroCopy) freePinned(buffer);
    }

    return;
  }

  static bool ghost_field_reset = false;

  void cudaColorSpinorField::allocateGhostBuffer(int nFace) {

    if (!comm_partitioned()) return;
    createGhostZone(nFace);

    // only allocate if not already allocated or buffer required is bigger than previously
    if ( !initGhostFaceBuffer || ghost_bytes > ghostFaceBytes ) {

      if (initGhostFaceBuffer) {
#ifdef USE_TEXTURE_OBJECTS
	destroyGhostTexObject();
#endif
	if (initGhostFaceBuffer && ghost_bytes) {
	  for (int b=0; b<2; b++) device_pinned_free(ghost_field[b]);
	}

        for (int b=0; b<2; ++b) device_free(ghostFaceBuffer[b]);
      }

      if (ghost_bytes > 0) {
	// GPU pinned allocator to avoid this being redirected, e.g., by QDPJIT
	if (ghost_bytes) {
	  for (int b=0; b<2; b++) ghost_field[b] = device_pinned_malloc(ghost_bytes);
	}
	ghost_field_reset = true;

	for (int b=0; b<2; ++b) ghostFaceBuffer[b] = device_malloc(ghost_bytes);
	initGhostFaceBuffer = true;
	ghostFaceBytes = ghost_bytes;
      }

    }

#ifdef USE_TEXTURE_OBJECTS
    // ghost texture is per object
    if (!ghostTexInit) createGhostTexObject();
#endif

    // always initialize the ghost receive pointers
    if (siteSubset == QUDA_PARITY_SITE_SUBSET) {
      for (int i=0; i<nDim; ++i) {
	if (commDimPartitioned(i)) {
	  for (int b=0; b<2; b++) {
	    ghost[b][i] = (char*)ghost_field[b] + ghostOffset[i][0]*precision;
	    if (precision == QUDA_HALF_PRECISION)
	      ghostNorm[b][i] = (char*)ghost_field[b] + ghostNormOffset[i][0]*QUDA_SINGLE_PRECISION;
	  }
	}
      }
    }

    // always initialize the ghost send pointers
    int Nint = nColor * nSpin * 2 / (nSpin == 4 ? 2 : 1); // number of internal degrees of freedom
    size_t offset = 0;
    for (int i=0; i<4; i++) {
      if (!commDimPartitioned(i)) continue;
    
      // compute size of buffer required
      ghost_face_bytes[i] = nFace*ghostFace[i]*Nint*precision;
      if (precision == QUDA_HALF_PRECISION) {
        ghost_face_bytes[i] += nFace*ghostFace[i]*sizeof(float);
      }

      for (int b=0; b<2; ++b) {
	backGhostFaceBuffer[b][i] = (void*)(((char*)ghostFaceBuffer[b]) + offset);
      }
      offset += nFace*ghostFace[i]*Nint*precision;
      if (precision == QUDA_HALF_PRECISION) offset += nFace*ghostFace[i]*sizeof(float);
      
      for (int b=0; b<2; ++b) {
	fwdGhostFaceBuffer[b][i] = (void*)(((char*)ghostFaceBuffer[b]) + offset);
      }
      offset += nFace*ghostFace[i]*Nint*precision;
      if (precision == QUDA_HALF_PRECISION) offset += nFace*ghostFace[i]*sizeof(float);
    }
  }

  void cudaColorSpinorField::allocateGhostBuffer(void *send_buf[], void *recv_buf[]) const
  {
    int num_faces = 1;
    if (nSpin == 1) num_faces = 3; // staggered

    int spinor_size = 2*nSpin*nColor*precision;

    // resize face only if requested size is larger than previously allocated one
    size_t faceBytes = 0;
    for (int i=0; i<nDimComms; i++) {
      if (comm_dim_partitioned(i)) faceBytes += 2*siteSubset*num_faces*surfaceCB[i]*spinor_size;
    }

    if (!initGhostFaceBuffer || faceBytes > ghostFaceBytes) {

      if (initGhostFaceBuffer) {
	for (int b=0; b<2; ++b) device_free(ghostFaceBuffer[b]);
      }

      if (faceBytes > 0) {
	for (int b=0; b<2; ++b) ghostFaceBuffer[b] = device_malloc(faceBytes);
	initGhostFaceBuffer = true;
	ghostFaceBytes = faceBytes;
      }

    }

    size_t offset = 0;
    for (int i=0; i<nDimComms; i++) {
      if (comm_dim_partitioned(i)) {
	// use first buffer for recv and second for send
	recv_buf[2*i+0] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[0]) + offset));
	send_buf[2*i+0] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[1]) + offset));
	offset += siteSubset*num_faces*surfaceCB[i]*spinor_size;

	recv_buf[2*i+1] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[0]) + offset));
	send_buf[2*i+1] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[1]) + offset));
	offset += siteSubset*num_faces*surfaceCB[i]*spinor_size;
      } else {
	recv_buf[2*i+0] = nullptr;
	recv_buf[2*i+1] = nullptr;
	send_buf[2*i+0] = nullptr;
	send_buf[2*i+1] = nullptr;
      }
    }

  }

  void cudaColorSpinorField::freeGhostBuffer(void)
  {
    destroyIPCComms();

    if (!initGhostFaceBuffer) return;
  
    for (int b=0; b<2; b++) {
      if (ghost_field[b]) device_pinned_free(ghost_field[b]);
      if (ghostFaceBuffer[b]) device_free(ghostFaceBuffer[b]);

      for (int i=0;i < 4; i++) {
	if (!commDimPartitioned(i)) continue;
        backGhostFaceBuffer[b][i] = NULL;
        fwdGhostFaceBuffer[b][i] = NULL;
      }
    }
    initGhostFaceBuffer = false;
  }

  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhost(const int nFace, const QudaParity parity, 
                                       const int dim, const QudaDirection dir,
				       const int dagger, cudaStream_t *stream, 
				       void *buffer, bool zero_copy, double a, double b)
  {
#ifdef MULTI_GPU
    int face_num;
    if (dir == QUDA_BACKWARDS) {
      face_num = 0;
    }else if (dir == QUDA_FORWARDS) {
      face_num = 1;
    }else{
      face_num = 2;
    }
    void *packBuffer = buffer ? buffer : ghostFaceBuffer[bufferIndex];
    packFace(packBuffer, *this, zero_copy, nFace, dagger, parity, dim, face_num, *stream, a, b);
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
    int Npad = Nint / Nvec; // number Nvec buffers we have
    
    if (dim !=3 || getKernelPackT() || getTwistPack()) { // use kernels to pack into contiguous buffers then a single cudaMemcpy

      size_t bytes = nFace*Nint*ghostFace[dim]*precision;

      if (precision == QUDA_HALF_PRECISION) bytes += nFace*ghostFace[dim]*sizeof(float);

      void* gpu_buf = 
	(dir == QUDA_BACKWARDS) ? this->backGhostFaceBuffer[bufferIndex][dim] : this->fwdGhostFaceBuffer[bufferIndex][dim];

      cudaMemcpyAsync(ghost_spinor, gpu_buf, bytes, cudaMemcpyDeviceToHost, *stream);

    } else if (this->TwistFlavor() != QUDA_TWIST_NONDEG_DOUBLET) { // do multiple cudaMemcpys

      const int x4 = nDim==5 ? x[4] : 1;
      const int Nt_minus1_offset = (volumeCB - nFace*ghostFace[3])/x4; // N_t -1 = Vh-Vsh

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
    
      size_t len = nFace*(ghostFace[3]/x4)*Nvec*precision;
      size_t dpitch = x4*len;
      size_t spitch = stride*Nvec*precision;

      // QUDA Memcpy NPad's worth. 
      //  -- Dest will point to the right beginning PAD. 
      //  -- Each Pad has size Nvec*Vsh Floats. 
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next
      for (int s=0; s<x4; s++) { // loop over multiple 4-d volumes (if they exist)
	void *dst = (char*)ghost_spinor + s*len;
	void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*precision;
	cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

	if (precision == QUDA_HALF_PRECISION) {
	  size_t len = nFace*(ghostFace[3]/x4)*sizeof(float);
	  int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
	  void *dst = (char*)ghost_spinor + nFace*Nint*ghostFace[3]*precision + s*len;
	  void *src = (char*)norm + norm_offset + s*(volumeCB/x4)*sizeof(float);
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream);
	}
      }
    }else{
      int flavorVolume = volume / 2;
      int flavorTFace  = ghostFace[3] / 2;
      int flavor1_Nt_minus1_offset = (flavorVolume - flavorTFace);
      int flavor2_Nt_minus1_offset = (volume - flavorTFace);
      int flavor1_offset = 0;
      int flavor2_offset = 0;
      // !dagger: send lower components backwards, send upper components forwards
      // dagger: send upper components backwards, send lower components forwards
      bool upper = dagger ? true : false; // Fwd is !Back
      if (dir == QUDA_FORWARDS) upper = !upper;
      int lower_spin_offset = Npad*stride;//ndeg tm: stride=2*flavor_volume+pad
      if (upper) {
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

    int len = nFace*ghostFace[dim]*Nint*precision;
    const void *src = ghost_spinor;
  
    int ghost_offset = (dir == QUDA_BACKWARDS) ? ghostOffset[dim][0] : ghostOffset[dim][1];
    void *ghost_dst = (char*)ghost_field[bufferIndex] + precision*ghost_offset;

    if (precision == QUDA_HALF_PRECISION) len += nFace*ghostFace[dim]*sizeof(float);

    cudaMemcpyAsync(ghost_dst, src, len, cudaMemcpyHostToDevice, *stream);
  }




   // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhostExtended(const int nFace, const int R[], const QudaParity parity,
					       const int dim, const QudaDirection dir,
					       const int dagger, cudaStream_t *stream,
					       void *buffer, bool zero_copy)
  {
#ifdef MULTI_GPU
    int face_num;
    if (dir == QUDA_BACKWARDS) {
      face_num = 0;
    }else if (dir == QUDA_FORWARDS) {
      face_num = 1;
    }else{
      face_num = 2;
    }
    void *packBuffer = buffer ? buffer : ghostFaceBuffer[bufferIndex];
    packFaceExtended(packBuffer, *this, zero_copy, nFace, R, dagger, parity, dim, face_num, *stream);
#else
    errorQuda("packGhostExtended not built on single-GPU build");
#endif

  }


  

  // copy data from host buffer into boundary region of device field
  void cudaColorSpinorField::unpackGhostExtended(const void* ghost_spinor, const int nFace, const QudaParity parity,
                                                 const int dim, const QudaDirection dir, 
                                                 const int dagger, cudaStream_t* stream, bool zero_copy)
  {

     
     
    // First call the regular unpackGhost routine to copy data into the `usual' ghost-zone region 
    // of the data array 
    unpackGhost(ghost_spinor, nFace, dim, dir, dagger, stream);

    // Next step is to copy data from the ghost zone back to the interior region
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom

    int len = nFace*ghostFace[dim]*Nint;
    int offset = length + ghostOffset[dim][0];
    offset += (dir == QUDA_BACKWARDS) ? 0 : len;

#ifdef MULTI_GPU
    const int face_num = 2;
    const bool unpack = true;
    const int R[4] = {0,0,0,0};
    packFaceExtended(ghostFaceBuffer[bufferIndex], *this, zero_copy, nFace, R, dagger, parity, dim, face_num, *stream, unpack);
#else
    errorQuda("unpackGhostExtended not built on single-GPU build");
#endif
  }


  cudaStream_t *stream;

  void cudaColorSpinorField::createComms(int nFace) {

    allocateGhostBuffer(nFace); // allocate the ghost buffer if not yet allocated

    if (bufferMessageHandler != bufferPinnedResizeCount) destroyComms();

    if (!initComms || nFaceComms != nFace) {

      // if we are requesting a new number of faces destroy and start over
      if (nFace != nFaceComms) destroyComms();

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
      for (int b=0; b<2; ++b) {
	if (faceBytes > 0) {
	  resizeBufferPinned(2*faceBytes, b); // oversizes for GPU_COMMS case

	  my_face[b] = bufferPinned[b];
	  cudaHostGetDevicePointer(&my_face_d[b], my_face[b], 0); // set the matching device pointer

	  from_face[b] = static_cast<char*>(my_face[b]) + faceBytes;
	  from_face_d[b] = static_cast<char*>(my_face_d[b]) + faceBytes;
	} else {
	  from_face[b] = nullptr;
	  from_face_d[b] = nullptr;
	  my_face[b] = nullptr;
	  my_face_d[b] = nullptr;
	}
      }

    checkCudaError();

      // assign pointers for each face - it's ok to alias for different Nface parameters
      size_t offset = 0;
#endif
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;
	
#ifdef GPU_COMMS
	for (int b=0; b<2; ++b) {
	  my_back_face[b][i] = backGhostFaceBuffer[b][i];
	  from_back_face[b][i] = ghost[b][i];
	
	  if (precision == QUDA_HALF_PRECISION) {
	    my_back_norm_face[b][i]  = static_cast<char*>(backGhostFaceBuffer[b][i]) + nFace*ghostFace[i]*Ndof*precision;
	    from_back_norm_face[b][i] = ghostNorm[b][i];
	  }
	} // loop over b

#else
        for (int b=0; b<2; ++b) {
	  my_back_face[b][i] = static_cast<char*>(my_face[b]) + offset;
	  from_back_face[b][i] = static_cast<char*>(from_face[b]) + offset;
	}
	offset += nbytes[i];
#endif
	
#ifdef GPU_COMMS
	for (int b=0; b<2; ++b) {
	  my_fwd_face[b][i] = fwdGhostFaceBuffer[b][i];
	  //from_fwd_face[b][i] = ghost[i] + nFace*ghostFace[i]*Ndof*precision;
	  from_fwd_face[b][i] = ghost_field[b] + ghostOffset[i][1]*precision;

	  if (precision == QUDA_HALF_PRECISION) {
	    my_fwd_norm_face[b][i] = static_cast<char*>(fwdGhostFaceBuffer[b][i]) + nFace*ghostFace[i]*Ndof*precision;
	   // from_fwd_norm_face[b][i] = static_cast<char*>(ghostNorm[i]) + nFace*ghostFace[i]*sizeof(float);
            from_fwd_norm_face[b][i] = static_cast<char*>(ghost_field[b]) + ghostNormOffset[i][1]*sizeof(float);
	  }
	} // loop over b
#else
	for (int b=0; b<2; ++b) {
	  my_fwd_face[b][i] = static_cast<char*>(my_face[b]) + offset;
	  from_fwd_face[b][i] = static_cast<char*>(from_face[b]) + offset;
	}
	offset += nbytes[i];
#endif

      }

      checkCudaError();

      // create a different message handler for each direction and Nface
      for (int b=0; b<2; ++b) {
        mh_send_fwd[b] = new MsgHandle**[maxNface];
        mh_send_back[b] = new MsgHandle**[maxNface];
        mh_recv_fwd[b] = new MsgHandle**[maxNface];
        mh_recv_back[b] = new MsgHandle**[maxNface];
#ifdef GPU_COMMS
        if (precision == QUDA_HALF_PRECISION) {
      	  mh_send_norm_fwd[b]  = new MsgHandle**[maxNface];
      	  mh_send_norm_back[b] = new MsgHandle**[maxNface];
     	  mh_recv_norm_fwd[b]  = new MsgHandle**[maxNface];
	  mh_recv_norm_back[b] = new MsgHandle**[maxNface];
        }
#endif
      } // loop over b
      for (int j=0; j<maxNface; j++) {
	for (int b=0; b<2; ++b) {
	  mh_send_fwd[b][j] = new MsgHandle*[2*nDimComms];
	  mh_send_back[b][j] = new MsgHandle*[2*nDimComms];
	  mh_recv_fwd[b][j] = new MsgHandle*[nDimComms];
	  mh_recv_back[b][j] = new MsgHandle*[nDimComms];
		
#ifdef GPU_COMMS
	  if (precision == QUDA_HALF_PRECISION) {
	    mh_send_norm_fwd[b][j] = new MsgHandle*[2*nDimComms];
	    mh_send_norm_back[b][j] = new MsgHandle*[2*nDimComms];
	    mh_recv_norm_fwd[b][j] = new MsgHandle*[nDimComms];
	    mh_recv_norm_back[b][j] = new MsgHandle*[nDimComms];
	  }
#endif	
	} // loop over b
	checkCudaError();

	for (int i=0; i<nDimComms; i++) {
	  if (!commDimPartitioned(i)) continue;
#ifdef GPU_COMMS
	  size_t nbytes_Nface = surfaceCB[i]*Ndof*precision*(j+1);
	  size_t nbytes_Nface_norm = surfaceCB[i]*(j+1)*sizeof(float);
	  if (i != 3 || getKernelPackT() || getTwistPack()) {
#else 
	    size_t nbytes_Nface = (nbytes[i] / maxNface) * (j+1);
#endif
	    for (int b=0; b<2; ++b) {
	      mh_send_fwd[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(my_fwd_face[b][i], i, +1, nbytes_Nface) : NULL;
	      mh_send_back[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(my_back_face[b][i], i, -1, nbytes_Nface) : NULL;
	      mh_send_fwd[b][j][2*i+1] = mh_send_fwd[b][j][2*i]; // alias pointers
	      mh_send_back[b][j][2*i+1] = mh_send_back[b][j][2*i]; // alias pointers
	    }
#ifdef GPU_COMMS

	    if (precision == QUDA_HALF_PRECISION) {
	      for (int b=0; b<2; ++b) {
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

            for (int b=0; b<2; ++b) {
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

          
	    if (precision == QUDA_HALF_PRECISION) {
	      int Nt_minus1_offset = (volume - nFace*ghostFace[3]); // The space-time coordinate of the start of the last time slice
	      void *norm_fwd = static_cast<float*>(norm) + Nt_minus1_offset;
	      void *norm_back = norm; // the first time slice has zero offset
	      for (int b=0; b<2; ++b) {
		mh_send_norm_fwd[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(norm_fwd, i, +1, surfaceCB[i]*(j+1)*sizeof(float)) : NULL;
		mh_send_norm_back[b][j][2*i+0] = (j+1 == nFace) ? comm_declare_send_relative(norm_back, i, -1, surfaceCB[i]*(j+1)*sizeof(float)) : NULL;
		mh_send_norm_fwd[b][j][2*i+1] = mh_send_norm_fwd[b][j][2*i];
		mh_send_norm_back[b][j][2*i+1] = mh_send_norm_back[b][j][2*i];
	      }
	    }

	  }
	  if (precision == QUDA_HALF_PRECISION) {
            for (int b=0; b<2; ++b) {
	      mh_recv_norm_fwd[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_fwd_norm_face[b][i], i, +1, nbytes_Nface_norm) : NULL;
	      mh_recv_norm_back[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_back_norm_face[b][i], i, -1, nbytes_Nface_norm) : NULL;
            }
	  }
#endif // GPU_COMMS

	  for (int b=0; b<2; ++b) {
	    mh_recv_fwd[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_fwd_face[b][i], i, +1, nbytes_Nface) : NULL;
	    mh_recv_back[b][j][i] = (j+1 == nFace) ? comm_declare_receive_relative(from_back_face[b][i], i, -1, nbytes_Nface) : NULL;
	  }

	} // loop over dimension
      }
     
      bufferMessageHandler = bufferPinnedResizeCount;
      initComms = true;
      nFaceComms = nFace;

      checkCudaError();
    }

    createIPCComms();
  }
   
  void cudaColorSpinorField::createIPCComms() {
    if ( (initIPCComms &&!ghost_field_reset) || comm_size() == 1 ) return;

    if (!initComms) errorQuda("Can only be called after create comms");
    if (!ghost_field[0] || !ghost_field[1]) errorQuda("ghost_field appears not to be allocated");

    // handles for obtained ghost pointers
    cudaIpcMemHandle_t ipcRemoteGhostDestHandle[2][2][QUDA_MAX_DIM];

    for (int b=0; b<2; b++) {
      for (int dim=0; dim<4; ++dim) {
	if (!commDimPartitioned(dim)) continue;
	for (int dir=0; dir<2; ++dir) {
	  MsgHandle* sendHandle = NULL;
	  MsgHandle* receiveHandle = NULL;
	  int disp = (dir == 1) ? +1 : -1;

	  // first set up receive
	  if (comm_peer2peer_enabled(1-dir,dim)) {
	    receiveHandle = comm_declare_receive_relative(&ipcRemoteGhostDestHandle[b][1-dir][dim],
							  dim,
							  -disp,
							  sizeof(ipcRemoteGhostDestHandle[b][1-dir][dim]));
	  }
	  // now send
	  if (comm_peer2peer_enabled(dir,dim)) {
	    cudaIpcMemHandle_t ipcLocalGhostDestHandle;
	    cudaIpcGetMemHandle(&ipcLocalGhostDestHandle, ghost_field[b]);
	    sendHandle = comm_declare_send_relative(&ipcLocalGhostDestHandle,
						    dim,
						    disp,
						    sizeof(ipcLocalGhostDestHandle));
	  }
	  if (receiveHandle) comm_start(receiveHandle);
	  if (sendHandle) comm_start(sendHandle);

	  if (receiveHandle) comm_wait(receiveHandle);
	  if (sendHandle) comm_wait(sendHandle);

	  if (sendHandle) comm_free(sendHandle);
	  if (receiveHandle) comm_free(receiveHandle);
	}
      }

      checkCudaError();

      // open the remote memory handles and set the send ghost pointers
      for (int dim=0; dim<4; ++dim) {
	if (!commDimPartitioned(dim)) continue;
	const int num_dir = (comm_dim(dim) == 2) ? 1 : 2;
	for (int dir=0; dir<num_dir; ++dir) {
	  if (!comm_peer2peer_enabled(dir,dim)) continue;
	  void **ghostDest = (dir==0) ? (&backGhostSendDest[b][dim]) : &(fwdGhostSendDest[b][dim]);
	  cudaIpcOpenMemHandle(ghostDest, ipcRemoteGhostDestHandle[b][dir][dim],
			       cudaIpcMemLazyEnablePeerAccess);
	}
	if (num_dir == 1) fwdGhostSendDest[b][dim] = backGhostSendDest[b][dim];
      }
    } // buffer index

    checkCudaError();

    // handles for obtained events
    cudaIpcEventHandle_t ipcRemoteEventHandle[2][2][QUDA_MAX_DIM];

    // Note that the events can and probably should be static.
    // We don't want a proliferation of events, I don't think
    // Also note that no b index is necessary here
    // Now communicate the event handles
    for (int dim=0; dim<4; ++dim) {
      if (!commDimPartitioned(dim)) continue;
      for (int dir=0; dir<2; ++dir) {
	for (int b=0; b<2; b++) {

	  MsgHandle* sendHandle = NULL;
	  MsgHandle* receiveHandle = NULL;
	  int disp = (dir == 1) ? +1 : -1;

	  // first set up receive
	  if (comm_peer2peer_enabled(1-dir,dim)) {
	    receiveHandle = comm_declare_receive_relative(&ipcRemoteEventHandle[b][1-dir][dim],
							  dim,
							  -disp,
							  sizeof(ipcRemoteEventHandle[b][1-dir][dim]));
	  }

	  // now send
	  if (comm_peer2peer_enabled(dir,dim)) {
	    cudaEventCreate(&ipcCopyEvent[b][dir][dim], cudaEventDisableTiming | cudaEventInterprocess);
	    cudaIpcEventHandle_t ipcLocalEventHandle;
	    cudaIpcGetEventHandle(&ipcLocalEventHandle, ipcCopyEvent[b][dir][dim]);

	    sendHandle = comm_declare_send_relative(&ipcLocalEventHandle,
						    dim,
						    disp,
						    sizeof(ipcLocalEventHandle));
	  }

	  if (receiveHandle) comm_start(receiveHandle);
	  if (sendHandle) comm_start(sendHandle);

	  if (receiveHandle) comm_wait(receiveHandle);
	  if (sendHandle) comm_wait(sendHandle);

	  if (sendHandle) comm_free(sendHandle);
	  if (receiveHandle) comm_free(receiveHandle);

	} // buffer index
      }
    }

    checkCudaError();

    for (int dim=0; dim<4; ++dim) {
      if (!commDimPartitioned(dim)) continue;
      for (int dir=0; dir<2; ++dir) {
	if (!comm_peer2peer_enabled(dir,dim)) continue;
	for (int b=0; b<2; b++) {
	  cudaIpcOpenEventHandle(&(ipcRemoteCopyEvent[b][dir][dim]), ipcRemoteEventHandle[b][dir][dim]);
	}
      }
    }

    // Create message handles for IPC synchronization
    for (int dim=0; dim<4; ++dim) {
      if (!commDimPartitioned(dim)) continue;
      if (comm_peer2peer_enabled(1,dim)) {
	for (int b=0; b<2; b++) {
	  // send to processor in forward direction
	  mh_send_p2p_fwd[b][dim] = comm_declare_send_relative(&buffer_send_p2p_fwd[b][dim], dim, +1, sizeof(int));
	  // receive from processor in forward direction
	  mh_recv_p2p_fwd[b][dim] = comm_declare_receive_relative(&buffer_recv_p2p_fwd[b][dim], dim, +1, sizeof(int));
	}
      }

      if (comm_peer2peer_enabled(0,dim)) {
	for (int b=0; b<2; b++) {
	  // send to processor in backward direction
	  mh_send_p2p_back[b][dim] = comm_declare_send_relative(&buffer_recv_p2p_back[b][dim], dim, -1, sizeof(int));
	  // receive from processor in backward direction
	  mh_recv_p2p_back[b][dim] = comm_declare_receive_relative(&buffer_recv_p2p_back[b][dim], dim, -1, sizeof(int));
	}
      }
    }
    checkCudaError();

    initIPCComms = true;
    ghost_field_reset = false;
  }

  void cudaColorSpinorField::destroyIPCComms() {

    if (!initIPCComms) return;
    checkCudaError();

    for (int dim=0; dim<4; ++dim) {

      if (!commDimPartitioned(dim)) continue;
      const int num_dir = (comm_dim(dim) == 2) ? 1 : 2;
    
      for (int b=0; b<2; b++) {
	if (comm_peer2peer_enabled(1,dim)) {
	  comm_free(mh_send_p2p_fwd[b][dim]);
	  comm_free(mh_recv_p2p_fwd[b][dim]);
	  cudaEventDestroy(ipcCopyEvent[b][1][dim]);

	  // only close this handle if it doesn't alias the back ghost
	  if (num_dir == 2) cudaIpcCloseMemHandle(fwdGhostSendDest[b][dim]);
	}
	
	if (comm_peer2peer_enabled(0,dim)) {
	  comm_free(mh_send_p2p_back[b][dim]);
	  comm_free(mh_recv_p2p_back[b][dim]);
	  cudaEventDestroy(ipcCopyEvent[b][0][dim]);

	  cudaIpcCloseMemHandle(backGhostSendDest[b][dim]);
	}
      } // buffer
    } // iterate over dim

    checkCudaError();
    initIPCComms = false;
  }

  void cudaColorSpinorField::destroyComms()
  {
    if (initComms) {

      for (int b=0; b<2; ++b) {
      for (int j=0; j<maxNface; j++) {
	for (int i=0; i<nDimComms; i++) {
	  if (commDimPartitioned(i)) {
	    if (mh_recv_fwd[b][j][i]) comm_free(mh_recv_fwd[b][j][i]);
	    if (mh_recv_fwd[b][j][i]) comm_free(mh_recv_back[b][j][i]);
	    if (mh_send_fwd[b][j][2*i]) comm_free(mh_send_fwd[b][j][2*i]);
	    if (mh_send_back[b][j][2*i]) comm_free(mh_send_back[b][j][2*i]);
	    // only in a special case are these not aliasing pointers
#ifdef GPU_COMMS
	    if (precision == QUDA_HALF_PRECISION) {
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
	if (precision == QUDA_HALF_PRECISION) {
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
      if (precision == QUDA_HALF_PRECISION) {
	delete []mh_recv_norm_fwd[b];
	delete []mh_recv_norm_back[b];
	delete []mh_send_norm_fwd[b];
	delete []mh_send_norm_back[b];
      }
	
      for (int i=0; i<nDimComms; i++) {
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

  void cudaColorSpinorField::streamInit(cudaStream_t *stream_p) {
    stream = stream_p;
  }


  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, cudaStream_t *stream_p, 
				  bool zero_copy, double a, double b) {

    createComms(nFace); // must call this first

    stream = stream_p;
    
    const int dim=-1; // pack all partitioned dimensions
 
    if (zero_copy) {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d[bufferIndex], true, a, b);
    } else {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[Nstream-1], 0, false, a, b);
    }
  }

  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, int stream_idx, 
				  bool zeroCopyPack, double a, double b) {

    createComms(nFace); // must call this first

    const int dim=-1; // pack all partitioned dimensions
 
    if (zeroCopyPack) {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[stream_idx], my_face_d[bufferIndex], true, a, b);
    } else {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[stream_idx], 0, false, a, b);
    }
  }

  void cudaColorSpinorField::packExtended(const int nFace, const int R[], const int parity, 
                                          const int dagger, const int dim,
                                          cudaStream_t *stream_p, const bool zero_copy) {

    createComms(nFace); // must call this first

    stream = stream_p;
 
    if (zero_copy) {
      packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d[bufferIndex], true);
    }else{
      packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[Nstream-1], 0, false);
    }
  }
                                                      


  void cudaColorSpinorField::gather(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;

    // If stream_p != 0, use pack_stream, else use the stream array
    cudaStream_t *pack_stream = (stream_p) ? stream_p : stream+dir;

    if (dir%2 == 0) {
      // backwards copy to host
      if (comm_peer2peer_enabled(0,dim)) return;

      sendGhost(my_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, pack_stream);
    } else {
      // forwards copy to host
      if (comm_peer2peer_enabled(1,dim)) return;

      sendGhost(my_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, pack_stream);
    }
  }


  void cudaColorSpinorField::recvStart(int nFace, int dir, int dagger, cudaStream_t* stream_p) {

    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;

    if (dir%2 == 0) { // sending backwards
      if (comm_peer2peer_enabled(1,dim)) {
	// receive from the processor in the +1 direction
	comm_start(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else {
        // Prepost receive
        comm_start(mh_recv_fwd[bufferIndex][nFace-1][dim]);
      }
    } else { //sending forwards
      // Prepost receive
      if (comm_peer2peer_enabled(0,dim)) {
	comm_start(mh_recv_p2p_back[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_back[bufferIndex][nFace-1][dim]);
      }
    }
  }


  void cudaColorSpinorField::sendStart(int nFace, int d, int dagger, cudaStream_t* stream_p) {

    int dim = d/2;
    int dir = d%2;
    if (!commDimPartitioned(dim)) return;

    int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2)/(nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom
    int Npad = Nint/Nvec;

    if (!comm_peer2peer_enabled(dir,dim)) {
      if (dir == 0) comm_start(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]);
      else comm_start(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]);
    } else { // doing peer-to-peer
      cudaStream_t *copy_stream = (stream_p) ? stream_p : stream + d;

      // all goes here
      void* ghost_dst = (dir == 0) ? (void*)((char*)(backGhostSendDest[bufferIndex][dim]) + precision*ghostOffset[dim][1]) :
	(void*)((char*)(fwdGhostSendDest[bufferIndex][dim]) + precision*ghostOffset[dim][0]);
      void *ghost_norm_dst = (dir == 0) ? static_cast<char*>(backGhostSendDest[bufferIndex][dim]) + QUDA_SINGLE_PRECISION*ghostNormOffset[dim][1] :
	static_cast<char*>(fwdGhostSendDest[bufferIndex][dim]) + QUDA_SINGLE_PRECISION*ghostNormOffset[dim][0];

      if (dim != 3 || getKernelPackT()) {

	cudaMemcpyAsync(ghost_dst,
			dir == 0 ? backGhostFaceBuffer[bufferIndex][dim] : fwdGhostFaceBuffer[bufferIndex][dim],
			ghost_face_bytes[dim],
			cudaMemcpyDeviceToDevice,
			*copy_stream); // copy to forward processor

      } else if (this->TwistFlavor() != QUDA_TWIST_NONDEG_DOUBLET) {

	const int x4 = nDim==5 ? x[4] : 1;
	const int Nt_minus_offset = (volumeCB - nFace*ghostFace[3])/x4;

	int offset = 0;
	if (nSpin == 1) {
	  offset = (dir == 0) ? 0 : Nt_minus_offset;
	} else if (nSpin == 4) {
	  // !dagger: send lower components backwards, send upper components forwards
	  // dagger: send upper components backwards, send lower components forwards
	  bool upper = dagger ? true : false;
	  if (dir == 1) upper = !upper;
	  int lower_spin_offset = Npad*stride;
	  if (dir == 0) {
	    offset = upper ? 0 : lower_spin_offset;
	  } else {
	    offset = (upper) ? Nt_minus_offset : lower_spin_offset + Nt_minus_offset;
	  }
	}

	size_t len = nFace*(ghostFace[3]/x4)*Nvec*precision;
	size_t dpitch = x4*len;
	size_t spitch = stride*Nvec*precision;

	for (int s=0; s<x4; s++) {
	  void *dst = (char*)ghost_dst + s*len;
	  void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*precision;
	  // start the copy
	  cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	  if (precision == QUDA_HALF_PRECISION) {
	    size_t len = nFace*(ghostFace[3]/x4)*sizeof(float);
	    int norm_offset = (dir == 0) ? 0 : Nt_minus_offset*sizeof(float);
	    void *dst = (char*)ghost_norm_dst + s*len;
	    void *src = static_cast<char*>(norm) + norm_offset + s*(volumeCB/x4)*sizeof(float);
	    cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, *copy_stream);
	  }
	}
      } else { // twisted doublet
	int flavorVolume = volume / 2;
	int flavorTFace  = ghostFace[3] / 2;
	int flavor1_Nt_minus1_offset = (flavorVolume - flavorTFace);
	int flavor2_Nt_minus1_offset = (volume - flavorTFace);
	int flavor1_offset = 0;
	int flavor2_offset = 0;
	// !dagger: send lower components backwards, send upper components forwards
	// dagger: send upper components backwards, send lower components forwards
	bool upper = dagger ? true : false; // Fwd is !Back
	if (dir == 1) upper = !upper;
	int lower_spin_offset = Npad*stride;//ndeg tm: stride=2*flavor_volume+pad
	if (upper) {
	  flavor1_offset = (dir == 0 ? 0 : flavor1_Nt_minus1_offset);
	  flavor2_offset = (dir == 0 ? flavorVolume : flavor2_Nt_minus1_offset);
	}else{
	  flavor1_offset = lower_spin_offset + (dir == 0 ? 0 : flavor1_Nt_minus1_offset);
	  flavor2_offset = lower_spin_offset + (dir == 0 ? flavorVolume : flavor2_Nt_minus1_offset);
	}

	// QUDA Memcpy NPad's worth.
	//  -- Dest will point to the right beginning PAD.
	//  -- Each Pad has size Nvec*Vsh Floats.
	//  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

	void *src = static_cast<char*>(v) + flavor1_offset*Nvec*precision;
	size_t len = flavorTFace*Nvec*precision;
	size_t spitch = stride*Nvec*precision;//ndeg tm: stride=2*flavor_volume+pad
	size_t dpitch = 2*len;
	cudaMemcpy2DAsync(ghost_dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	src = static_cast<char*>(v) + flavor2_offset*Nvec*precision;
	cudaMemcpy2DAsync(static_cast<char*>(ghost_dst)+len, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	if (precision == QUDA_HALF_PRECISION) {
	  int norm_offset = (dir == 0) ? 0 : flavor1_Nt_minus1_offset*sizeof(float);
	  void *src = static_cast<char*>(norm) + norm_offset;
	  size_t dpitch = flavorTFace*sizeof(float);
	  size_t spitch = flavorVolume*sizeof(float);
	  cudaMemcpy2DAsync(ghost_norm_dst, dpitch, src, spitch, flavorTFace*sizeof(float), 2, cudaMemcpyDeviceToDevice, *copy_stream);
	}
      }

      if (dir == 0) {
	// record the event
	cudaEventRecord(ipcCopyEvent[bufferIndex][0][dim], *copy_stream);
	// send to the propcessor in the -1 direction
	comm_start(mh_send_p2p_back[bufferIndex][dim]);
      } else {
	cudaEventRecord(ipcCopyEvent[bufferIndex][1][dim], *copy_stream);
	// send to the processor in the +1 direction
	comm_start(mh_send_p2p_fwd[bufferIndex][dim]);
      }
    }
  }

  void cudaColorSpinorField::commsStart(int nFace, int dir, int dagger, cudaStream_t* stream_p) {
    recvStart(nFace, dir, dagger, stream_p);
    sendStart(nFace, dir, dagger, stream_p);
  }


  bool cudaColorSpinorField::ipcCopyComplete(int dir, int dim)
  {
    return cudaSuccess == cudaEventQuery(ipcCopyEvent[bufferIndex][dir][dim]) ? true : false;
  }

  bool cudaColorSpinorField::ipcRemoteCopyComplete(int dir, int dim)
  {
    return cudaSuccess == cudaEventQuery(ipcRemoteCopyEvent[bufferIndex][dir][dim]) ? true : false;
  }

  static bool comm_query_mh_recv_p2p_fwd[QUDA_MAX_DIM] = { };
  static bool comm_query_mh_recv_p2p_back[QUDA_MAX_DIM] = { };
  static bool comm_query_mh_send_p2p_fwd[QUDA_MAX_DIM] = { };
  static bool comm_query_mh_send_p2p_back[QUDA_MAX_DIM] = { };

  static bool complete_recv_fwd[QUDA_MAX_DIM] = { };
  static bool complete_recv_back[QUDA_MAX_DIM] = { };
  static bool complete_send_fwd[QUDA_MAX_DIM] = { };
  static bool complete_send_back[QUDA_MAX_DIM] = { };

  int cudaColorSpinorField::commsQuery(int nFace, int dir, int dagger, cudaStream_t *stream_p) {

    int dim = dir/2;
    if (!commDimPartitioned(dim)) return 0;

    if (dir%2==0) {

      if (comm_peer2peer_enabled(1,dim)) {
	if (!comm_query_mh_recv_p2p_fwd[dim]) comm_query_mh_recv_p2p_fwd[dim] = comm_query(mh_recv_p2p_fwd[bufferIndex][dim]);
	//if (comm_query_mh_recv_p2p_fwd[dim] && !complete_recv_fwd[dim]) complete_recv_fwd[dim] = ipcRemoteCopyComplete(1,dim);
	if (comm_query_mh_recv_p2p_fwd[dim]) complete_recv_fwd[dim] = true;
      } else {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_fwd[bufferIndex][nFace-1][dim]);
      }

      if (comm_peer2peer_enabled(0,dim)) {
	if (!comm_query_mh_send_p2p_back[dim]) comm_query_mh_send_p2p_back[dim] = comm_query(mh_send_p2p_back[bufferIndex][dim]);
	if (comm_query_mh_send_p2p_back[dim]) complete_send_back[dim] = true; // no need to query send, fire and forget
      } else {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]);
      }

      if (complete_recv_fwd[dim] && complete_send_back[dim]) {
	comm_query_mh_recv_p2p_fwd[dim] = false;
	comm_query_mh_send_p2p_back[dim] = false;
	complete_recv_fwd[dim] = false;
	complete_send_back[dim] = false;
	return 1;
      }

    } else { // dir%2 == 1
      if (comm_peer2peer_enabled(0,dim)) {
	if (!comm_query_mh_recv_p2p_back[dim]) comm_query_mh_recv_p2p_back[dim] = comm_query(mh_recv_p2p_back[bufferIndex][dim]);
        //if (comm_query_mh_recv_p2p_back[dim] && !complete_recv_back[dim]) complete_recv_back[dim] = ipcRemoteCopyComplete(0,dim);
        if (comm_query_mh_recv_p2p_back[dim]) complete_recv_back[dim] = true;//ipcRemoteCopyComplete(0,dim);
      } else {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_back[bufferIndex][nFace-1][dim]);
      }

      if (comm_peer2peer_enabled(1,dim)) {
	if (!comm_query_mh_send_p2p_fwd[dim]) comm_query_mh_send_p2p_fwd[dim] = comm_query(mh_send_p2p_fwd[bufferIndex][dim]);
	if (comm_query_mh_send_p2p_fwd[dim]) complete_send_fwd[dim] = true; // no need to query send, fire and forget
      } else {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]);
      }

      if (complete_recv_back[dim] && complete_send_fwd[dim]) {
	comm_query_mh_recv_p2p_back[dim] = false;
	comm_query_mh_send_p2p_fwd[dim] = false;
	complete_recv_back[dim] = false;
	complete_send_fwd[dim] = false;
	return 1;
      }
    }

    return 0;
  }

  void cudaColorSpinorField::commsWait(int nFace, int dir, int dagger, cudaStream_t *stream_p) {
    int dim = dir / 2;
    if (!commDimPartitioned(dim)) return;

    if (dir%2==0) {

      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_recv_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][1][dim]);
      } else {
	comm_wait(mh_recv_fwd[bufferIndex][nFace-1][dim]);
#ifdef GPU_COMMS
	if (precision == QUDA_HALF_PRECISION) comm_wait(mh_recv_norm_fwd[bufferIndex][nFace-1][dim]);
#endif
      }

      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_send_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][0][dim]);
      } else {
	comm_wait(mh_send_back[bufferIndex][nFace-1][2*dim+dagger]);
#ifdef GPU_COMMS
	if (precision == QUDA_HALF_PRECISION) comm_wait(mh_send_norm_back[bufferIndex][nFace-1][2*dim+dagger]);
#endif
      }
    } else {
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_recv_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][0][dim]);
      } else {
	comm_wait(mh_recv_back[bufferIndex][nFace-1][dim]);
#ifdef GPU_COMMS
	comm_wait(mh_recv_norm_back[bufferIndex][nFace-1][dim]);
#endif
      }

      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_send_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][1][dim]);
      } else {
	comm_wait(mh_send_fwd[bufferIndex][nFace-1][2*dim+dagger]);
#ifdef GPU_COMMS
	if (precision == QUDA_HALF_PRECISION) comm_wait(mh_send_norm_fwd[bufferIndex][nFace-1][2*dim+dagger]);
#endif
      }
    }

    return;
  }

  const cudaEvent_t& cudaColorSpinorField::getIPCRemoteCopyEvent(int dir, int dim) const {
    return ipcRemoteCopyEvent[bufferIndex][dir][dim];
  }

  void cudaColorSpinorField::scatter(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;

    // both scattering occurances now go through the same stream
    if (dir%2==0) {// receive from forwards
      if (comm_peer2peer_enabled(1,dim)) return;

      unpackGhost(from_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, stream_p);
    } else { // receive from backwards
      if (comm_peer2peer_enabled(0,dim)) return;

      unpackGhost(from_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, stream_p);
    }
  }



  void cudaColorSpinorField::scatter(int nFace, int dagger, int dir)
  {
    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;

    // both scattering occurances now go through the same stream
    if (dir%2==0) {// receive from forwards
      if (comm_peer2peer_enabled(1,dim)) return;

      unpackGhost(from_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/]);
    } else { // receive from backwards
      if (comm_peer2peer_enabled(0,dim)) return;

      unpackGhost(from_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+1*/]);
    }
  }

  
  void cudaColorSpinorField::scatterExtended(int nFace, int parity, int dagger, int dir)
  {
    bool zero_copy = false;
    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;
    if (dir%2==0) {// receive from forwards
      unpackGhostExtended(from_fwd_face[bufferIndex][dim], nFace, static_cast<QudaParity>(parity), dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/], zero_copy);
    } else { // receive from backwards
      unpackGhostExtended(from_back_face[bufferIndex][dim], nFace, static_cast<QudaParity>(parity),  dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+1*/], zero_copy);
    }
  }
 
  void cudaColorSpinorField::exchangeGhost(QudaParity parity, int dagger) const {
    void **send = static_cast<void**>(safe_malloc(nDimComms * 2 * sizeof(void*)));

    // allocate ghost buffer if not yet allocated
    allocateGhostBuffer(send, ghost_fixme);

    genericPackGhost(send, *this, parity, dagger);

    int nFace = (nSpin == 1) ? 3 : 1;
    exchange(ghost_fixme, send, nFace);

    host_free(send);
  }

  std::ostream& operator<<(std::ostream &out, const cudaColorSpinorField &a) {
    out << (const ColorSpinorField&)a;
    out << "v = " << a.v << std::endl;
    out << "norm = " << a.norm << std::endl;
    out << "alloc = " << a.alloc << std::endl;
    out << "init = " << a.init << std::endl;
    return out;
  }

//! for composite fields:
  cudaColorSpinorField& cudaColorSpinorField::Component(const int idx) const {
    
    if (this->IsComposite()) {
      if (idx < this->CompositeDim()) {//setup eigenvector form the set
        return *(dynamic_cast<cudaColorSpinorField*>(components[idx])); 
      }
      else{
        errorQuda("Incorrect component index...");
      }
    }
    errorQuda("Cannot get requested component");
    exit(-1);
  }

//copyCuda currently cannot not work with set of spinor fields..
  void cudaColorSpinorField::CopySubset(cudaColorSpinorField &dst, const int range, const int first_element) const{
#if 0
    if (first_element < 0) errorQuda("\nError: trying to set negative first element.\n");
    if (siteSubset == QUDA_PARITY_SITE_SUBSET && this->EigvId() == -1) {
      if (first_element == 0 && range == this->EigvDim())
      {
        if (range != dst.EigvDim())errorQuda("\nError: eigenvector range to big.\n");
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
        for (int id = first_element; id < (first_element+range); id++)
        {
            param.eigv_id = id;
            eigv_subset->eigenvectors.push_back(new cudaColorSpinorField(*this, param));
        }
        checkField(dst, *eigv_subset);
        copyCuda(dst, *eigv_subset);

        delete eigv_subset;
      } else {
        errorQuda("Incorrect eigenvector dimension...");
      }
    } else{
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
    printfQuda("\nVolume (in bytes): %lu\n", resDesc.res.linear.sizeInBytes);
    if (resDesc.resType == cudaResourceTypeLinear) printfQuda("\nResource type: linear \n");
#endif
  }

  void cudaColorSpinorField::Source(const QudaSourceType sourceType, const int st, const int s, const int c) {
    ColorSpinorParam param(*this);
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_NULL_FIELD_CREATE;

    cpuColorSpinorField tmp(param);
    tmp.Source(sourceType, st, s, c);
    *this = tmp;
  }


} // namespace quda

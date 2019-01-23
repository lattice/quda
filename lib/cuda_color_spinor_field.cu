#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <string.h>
#include <iostream>

#include <color_spinor_field.h>
#include <color_spinor_field_order.h> // Evan hack
#include <blas_quda.h>
#include <dslash_quda.h>

int zeroCopy = 0;

namespace quda {

  static bool ghost_precision_reset = false;
  static QudaPrecision ghost_precision_old = QUDA_INVALID_PRECISION;

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
    ColorSpinorField(param), alloc(false), init(true), texInit(false),
    ghostTexInit(false), ghost_field_tex{nullptr,nullptr,nullptr,nullptr}, bufferMessageHandler(0)
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
    ghostTexInit(false), ghost_field_tex{nullptr,nullptr,nullptr,nullptr}, bufferMessageHandler(0)
  {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, 
					     const ColorSpinorParam &param) :
    ColorSpinorField(src), alloc(false), init(true), texInit(false),
    ghostTexInit(false), ghost_field_tex{nullptr,nullptr,nullptr,nullptr}, bufferMessageHandler(0)
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
      ghostTexInit(false), ghost_field_tex{nullptr,nullptr,nullptr,nullptr}, bufferMessageHandler(0)
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
      switch(mem_type) {
      case QUDA_MEMORY_DEVICE:
	v = pool_device_malloc(bytes);
	if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) norm = pool_device_malloc(norm_bytes);
	break;
      case QUDA_MEMORY_MAPPED:
	v_h = mapped_malloc(bytes);
	cudaHostGetDevicePointer(&v, v_h, 0); // set the matching device pointer
	if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
	  norm_h = mapped_malloc(norm_bytes);
	  cudaHostGetDevicePointer(&norm, norm_h, 0); // set the matching device pointer
	}
	break;
      default:
	errorQuda("Unsupported memory type %d", mem_type);
      }
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
	param.mem_type = mem_type;

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
	param.mem_type = mem_type;

        even = new cudaColorSpinorField(*this, param);
        odd = new cudaColorSpinorField(*this, param);

        // need this hackery for the moment (need to locate the odd pointers half way into the full field)
        (dynamic_cast<cudaColorSpinorField*>(odd))->v = (void*)((char*)v + bytes/2);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) 
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
	 param.mem_type = mem_type;

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

    if ( (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) && nVec == 1 ) {
      if (texInit) errorQuda("Already bound textures");
      
      // create the texture for the field components
      
      cudaChannelFormatDesc desc;
      memset(&desc, 0, sizeof(cudaChannelFormatDesc));
      if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
      else desc.f = cudaChannelFormatKindSigned; // quarter is char, half is short, double is int2
      
      // staggered and coarse fields in half and single are always two component
      int texel_size = 1;
      // all FLOAT2-ordred fields that are not double precision
      if (precision != QUDA_DOUBLE_PRECISION && fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {
        desc.x = 8*precision;
        desc.y = 8*precision;
        desc.z = 0;
        desc.w = 0;
        texel_size = 2*precision;
      } else { // all others are four component (double2 is spread across int4)
        desc.x = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
        desc.y = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
        desc.z = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
        desc.w = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
        texel_size = 4 * (precision == QUDA_DOUBLE_PRECISION ? sizeof(int) : precision);
      }
      
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = v;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = bytes;
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
      else texDesc.readMode = cudaReadModeElementType;

      if (resDesc.res.linear.sizeInBytes % deviceProp.textureAlignment != 0) {
        errorQuda("Allocation size %lu does not have correct alignment for textures (%lu)",
          resDesc.res.linear.sizeInBytes, deviceProp.textureAlignment);
      }

      unsigned long texels = resDesc.res.linear.sizeInBytes / texel_size;
      if (texels > (unsigned)deviceProp.maxTexture1DLinear) {
        errorQuda("Attempting to bind too large a texture %lu > %d", texels, deviceProp.maxTexture1DLinear);
      }

      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

      checkCudaError();

      // create the texture for the norm components
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
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

        if (resDesc.res.linear.sizeInBytes % deviceProp.textureAlignment != 0) {
          errorQuda("Allocation size %lu does not have correct alignment for textures (%lu)",
        	    resDesc.res.linear.sizeInBytes, deviceProp.textureAlignment);
        }

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&texNorm, &resDesc, &texDesc, NULL);

        checkCudaError();
      }
      
      texInit = true;

      checkCudaError();
    }
  }

  void cudaColorSpinorField::createGhostTexObject() const {
    // create the ghost texture object
    if ( (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) && nVec == 1 && ghost_bytes) {
      if (ghostTexInit) errorQuda("Already bound ghost texture");

      for (int b=0; b<2; b++) {
	cudaChannelFormatDesc desc;
	memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	if (ghost_precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
	else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

	// all FLOAT2-ordred fields that are not double precision
	if (ghost_precision != QUDA_DOUBLE_PRECISION && fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {
	  desc.x = 8*ghost_precision;
	  desc.y = 8*ghost_precision;
	  desc.z = 0;
	  desc.w = 0;
	} else { // all others are four component (double2 is spread across int4)
	  desc.x = (ghost_precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*ghost_precision;
	  desc.y = (ghost_precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*ghost_precision;
	  desc.z = (ghost_precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*ghost_precision;
	  desc.w = (ghost_precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*ghost_precision;
	}

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = ghost_recv_buffer_d[b];
	resDesc.res.linear.desc = desc;
	resDesc.res.linear.sizeInBytes = ghost_bytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
	else texDesc.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&ghostTex[b], &resDesc, &texDesc, NULL);

	// second set of ghost texture map to the host-mapped pinned receive buffers
	resDesc.res.linear.devPtr = ghost_pinned_recv_buffer_hd[b];
	cudaCreateTextureObject(&ghostTex[2+b], &resDesc, &texDesc, NULL);

	if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) {
	  cudaChannelFormatDesc desc;
	  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	  desc.f = cudaChannelFormatKindFloat;
	  desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;

	  cudaResourceDesc resDesc;
	  memset(&resDesc, 0, sizeof(resDesc));
	  resDesc.resType = cudaResourceTypeLinear;
	  resDesc.res.linear.devPtr = ghost_recv_buffer_d[b];
	  resDesc.res.linear.desc = desc;
	  resDesc.res.linear.sizeInBytes = ghost_bytes;

	  cudaTextureDesc texDesc;
	  memset(&texDesc, 0, sizeof(texDesc));
	  texDesc.readMode = cudaReadModeElementType;

	  cudaCreateTextureObject(&ghostTexNorm[b], &resDesc, &texDesc, NULL);

	  resDesc.res.linear.devPtr = ghost_pinned_recv_buffer_hd[b];
	  cudaCreateTextureObject(&ghostTexNorm[2+b], &resDesc, &texDesc, NULL);
	}

	ghost_field_tex[b] = ghost_recv_buffer_d[b];
	ghost_field_tex[2+b] = ghost_pinned_recv_buffer_hd[b];
      } // buffer index

      ghostTexInit = true;

      checkCudaError();
    }

  }

  void cudaColorSpinorField::destroyTexObject() {
    if ( (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) && nVec == 1 && texInit) {
      cudaDestroyTextureObject(tex);
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) cudaDestroyTextureObject(texNorm);
      texInit = false;
    }
  }

  void cudaColorSpinorField::destroyGhostTexObject() const {
    if ( (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) && nVec == 1 && ghostTexInit) {
      for (int i=0; i<4; i++) cudaDestroyTextureObject(ghostTex[i]);
      if ( (ghost_precision_reset && (ghost_precision_old == QUDA_HALF_PRECISION || ghost_precision_old == QUDA_QUARTER_PRECISION) ||
        (!ghost_precision_reset && (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) ) ) ) {
        for (int i=0; i<4; i++) cudaDestroyTextureObject(ghostTexNorm[i]);
      }
      ghostTexInit = false;
    }
  }
#endif

  void cudaColorSpinorField::destroy() {

    if (alloc) {
      switch(mem_type) {
      case QUDA_MEMORY_DEVICE:
        pool_device_free(v);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) pool_device_free(norm);
        break;
      case QUDA_MEMORY_MAPPED:
        host_free(v_h);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) host_free(norm_h);
        break;
      default:
        errorQuda("Unsupported memory type %d", mem_type);
      }
    }


    if (composite_descr.is_composite) 
    {
       CompositeColorSpinorField::iterator vec;
       for (vec = components.begin(); vec != components.end(); vec++) delete *vec;
    } 

    if ( siteSubset == QUDA_FULL_SITE_SUBSET && (!composite_descr.is_composite || composite_descr.is_component) ) {
      delete even;
      delete odd;
    }

#ifdef USE_TEXTURE_OBJECTS
    if (!composite_descr.is_composite || composite_descr.is_component) {
      destroyTexObject();
      destroyGhostTexObject();
    }
#endif

  }

  void cudaColorSpinorField::backup() const {
    if (backed_up) errorQuda("ColorSpinorField already backed up");
    backup_h = new char[bytes];
    cudaMemcpy(backup_h, v, bytes, cudaMemcpyDeviceToHost);
    if (norm_bytes) {
      backup_norm_h = new char[norm_bytes];
      cudaMemcpy(backup_norm_h, norm, norm_bytes, cudaMemcpyDeviceToHost);
    }
    checkCudaError();
    backed_up = true;
  }

  void cudaColorSpinorField::restore() const {
    if (!backed_up) errorQuda("Cannot restore since not backed up");
    cudaMemcpy(v, backup_h, bytes, cudaMemcpyHostToDevice);
    delete []backup_h;
    if (norm_bytes) {
      cudaMemcpy(v, backup_norm_h, norm_bytes, cudaMemcpyHostToDevice);
      delete []backup_norm_h;
    }
    checkCudaError();
    backed_up = false;
  }

  // cuda's floating point format, IEEE-754, represents the floating point
  // zero as 4 zero bytes
  void cudaColorSpinorField::zero() {
    cudaMemsetAsync(v, 0, bytes, streams[Nstream-1]);
    if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) cudaMemsetAsync(norm, 0, norm_bytes, streams[Nstream-1]);
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

    if ( reorder_location() == QUDA_CPU_FIELD_LOCATION && typeid(src) == typeid(cpuColorSpinorField)) {
      void *buffer = pool_pinned_malloc(bytes + norm_bytes);
      memset(buffer, 0, bytes+norm_bytes); // FIXME (temporary?) bug fix for padding

      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, 0, static_cast<char*>(buffer)+bytes, 0);

      qudaMemcpy(v, buffer, bytes, cudaMemcpyHostToDevice);
      qudaMemcpy(norm, static_cast<char*>(buffer)+bytes, norm_bytes, cudaMemcpyHostToDevice);

      pool_pinned_free(buffer);
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else {

      if (src.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
        // special case where we use mapped memory to read/write directly from application's array
        void *src_d;
        cudaError_t error = cudaHostGetDevicePointer(&src_d, const_cast<void*>(src.V()), 0);
        if (error != cudaSuccess) errorQuda("Failed to get device pointer for ColorSpinorField field");
        copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, v, src_d);
      } else {
        void *Src=nullptr, *srcNorm=nullptr, *buffer=nullptr;
        if (!zeroCopy) {
          buffer = pool_device_malloc(src.Bytes()+src.NormBytes());
          Src = buffer;
          srcNorm = static_cast<char*>(Src) + src.Bytes();
          qudaMemcpy(Src, src.V(), src.Bytes(), cudaMemcpyHostToDevice);
          qudaMemcpy(srcNorm, src.Norm(), src.NormBytes(), cudaMemcpyHostToDevice);
        } else {
          buffer = pool_pinned_malloc(src.Bytes()+src.NormBytes());
          memcpy(buffer, src.V(), src.Bytes());
          memcpy(static_cast<char*>(buffer)+src.Bytes(), src.Norm(), src.NormBytes());
          cudaError_t error = cudaHostGetDevicePointer(&Src, buffer, 0);
          if (error != cudaSuccess) errorQuda("Failed to get device pointer for ColorSpinorField field");
          srcNorm = static_cast<char*>(Src) + src.Bytes();
        }

        cudaMemset(v, 0, bytes); // FIXME (temporary?) bug fix for padding
        copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, 0, Src, 0, srcNorm);

        if (zeroCopy) pool_pinned_free(buffer);
        else pool_device_free(buffer);
      }
    }

    qudaDeviceSynchronize(); // include sync here for accurate host-device profiling
    checkCudaError();
  }


  void cudaColorSpinorField::saveSpinorField(ColorSpinorField &dest) const {

    if ( reorder_location() == QUDA_CPU_FIELD_LOCATION && typeid(dest) == typeid(cpuColorSpinorField)) {
      void *buffer = pool_pinned_malloc(bytes+norm_bytes);
      qudaMemcpy(buffer, v, bytes, cudaMemcpyDeviceToHost);
      qudaMemcpy(static_cast<char*>(buffer)+bytes, norm, norm_bytes, cudaMemcpyDeviceToHost);

      copyGenericColorSpinor(dest, *this, QUDA_CPU_FIELD_LOCATION, 0, buffer, 0, static_cast<char*>(buffer)+bytes);
      pool_pinned_free(buffer);
    } else if (typeid(dest) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION);
    } else {

      if (dest.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
	// special case where we use zero-copy memory to read/write directly from application's array
	void *dest_d;
	cudaError_t error = cudaHostGetDevicePointer(&dest_d, const_cast<void*>(dest.V()), 0);
        if (error != cudaSuccess) errorQuda("Failed to get device pointer for ColorSpinorField field");
        copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dest_d, v);
      } else {
        void *dst=nullptr, *dstNorm=nullptr, *buffer=nullptr;
        if (!zeroCopy) {
          buffer = pool_device_malloc(dest.Bytes()+dest.NormBytes());
          dst = buffer;
          dstNorm = static_cast<char*>(dst) + dest.Bytes();
        } else {
          buffer = pool_pinned_malloc(dest.Bytes()+dest.NormBytes());
          cudaError_t error = cudaHostGetDevicePointer(&dst, buffer, 0);
          if (error != cudaSuccess) errorQuda("Failed to get device pointer for ColorSpinorField");
          dstNorm = static_cast<char*>(dst)+dest.Bytes();
        }

        copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dst, 0, dstNorm, 0);

        if (!zeroCopy) {
          qudaMemcpy(dest.V(), dst, dest.Bytes(), cudaMemcpyDeviceToHost);
          qudaMemcpy(dest.Norm(), dstNorm, dest.NormBytes(), cudaMemcpyDeviceToHost);
        } else {
          memcpy(dest.V(), buffer, dest.Bytes());
          memcpy(dest.Norm(), static_cast<char*>(buffer) + dest.Bytes(), dest.NormBytes());
        }

        if (zeroCopy) pool_pinned_free(buffer);
        else pool_device_free(buffer);
      }
    }

    qudaDeviceSynchronize(); // need to sync before data can be used on CPU
    checkCudaError();
  }

  void cudaColorSpinorField::allocateGhostBuffer(int nFace, bool spin_project) const {

    createGhostZone(nFace, spin_project);
    LatticeField::allocateGhostBuffer(ghost_bytes);

#ifdef USE_TEXTURE_OBJECTS
    // ghost texture is per object
    if (ghost_field_tex[0] != ghost_recv_buffer_d[0] || ghost_field_tex[1] != ghost_recv_buffer_d[1] || ghost_precision_reset)
      destroyGhostTexObject();
    if (!ghostTexInit) createGhostTexObject();
#endif
  }

  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhost(const int nFace, const QudaParity parity, 
                                       const int dim, const QudaDirection dir,
				       const int dagger, cudaStream_t *stream, 
				       MemoryLocation location [2*QUDA_MAX_DIM],
                                       MemoryLocation location_label, double a, double b, double c)
  {
#ifdef MULTI_GPU
    int face_num = (dir == QUDA_BACKWARDS) ? 0 : (dir == QUDA_FORWARDS) ? 1 : 2;
    void *packBuffer[2*QUDA_MAX_DIM];

    for (int dim=0; dim<4; dim++) {
      for (int dir=0; dir<2; dir++) {
	switch(location[2*dim+dir]) {
	case Device: // pack to local device buffer
	  packBuffer[2*dim+dir] = my_face_dim_dir_d[bufferIndex][dim][dir];
          break;
	case Host:   // pack to zero-copy memory
	  packBuffer[2*dim+dir] = my_face_dim_dir_hd[bufferIndex][dim][dir];
          break;
	case Remote:   // pack to remote peer memory
	  packBuffer[2*dim+dir] = static_cast<char*>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + precision*ghostOffset[dim][1-dir];
          break;
	default: errorQuda("Undefined location %d", location[2*dim+dir]);
	}
      }
    }

    // new packing kernel is only for Wilson/Clover/twisted at the moment
    if (nSpin == 1 || (nDim == 5 && (twistFlavor == QUDA_TWIST_NO || twistFlavor == QUDA_TWIST_INVALID) ) ) {
      packFace(packBuffer, *this, location_label, nFace, dagger, parity, dim, face_num, *stream, a, b);
    } else {
      PackGhost(packBuffer, *this, location_label, nFace, dagger, parity, a, b, c, *stream);
    }

#else
    errorQuda("packGhost not built on single-GPU build");
#endif
  }
 
  // send the ghost zone to the host
  void cudaColorSpinorField::sendGhost(void *ghost_spinor, const int nFace, const int dim, 
				       const QudaDirection dir, const int dagger, 
				       cudaStream_t *stream) {

#ifdef MULTI_GPU
    int Nvec = (nSpin == 1 || ghost_precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom
    int Npad = Nint / Nvec; // number Nvec buffers we have

    if (precision != ghost_precision) { pushKernelPackT(true); }
    
    if (dim !=3 || getKernelPackT()) { // use kernels to pack into contiguous buffers then a single cudaMemcpy

      size_t bytes = nFace*Nint*ghostFace[dim]*ghost_precision;

      if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) bytes += nFace*ghostFace[dim]*sizeof(float);

      void* gpu_buf = (dir == QUDA_BACKWARDS) ? my_face_dim_dir_d[bufferIndex][dim][0] : my_face_dim_dir_d[bufferIndex][dim][1];

      qudaMemcpyAsync(ghost_spinor, gpu_buf, bytes, cudaMemcpyDeviceToHost, *stream);

    } else {

      const int nParity = siteSubset;
      const int x4 = nDim==5 ? x[4] : 1;
      const int Nt_minus1_offset = (volumeCB - nFace*ghostFaceCB[3])/x4; // N_t-1 = Vh-Vsh

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
    
      size_t len = nFace*(ghostFaceCB[3]/x4)*Nvec*ghost_precision;
      size_t dpitch = x4*len;
      size_t spitch = stride*Nvec*ghost_precision;

      // QUDA Memcpy NPad's worth. 
      //  -- Dest will point to the right beginning PAD. 
      //  -- Each Pad has size Nvec*Vsh Floats. 
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

      for (int parity = 0; parity<nParity; parity++) {
        for (int s=0; s<x4; s++) { // loop over multiple 4-d volumes (if they exist)
          void *dst = (char*)ghost_spinor + s*len + parity*nFace*Nint*ghostFaceCB[3]*ghost_precision;
          void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*ghost_precision + parity*bytes/2;
          qudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

          // we can probably issue this as a single cudaMemcpy2d along the fifth dimension
          if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) {
            size_t len = nFace*(ghostFaceCB[3]/x4)*sizeof(float);
            int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
            void *dst = (char*)ghost_spinor + nParity*nFace*Nint*ghostFaceCB[3]*ghost_precision
              + s*len + parity*nFace*ghostFaceCB[3]*sizeof(float);
            void *src = (char*)norm + norm_offset + s*(volumeCB/x4)*sizeof(float) + parity*norm_bytes/2;
            qudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream);
          }
        } // fifth dimension
      } // parity

    }

    if (precision != ghost_precision) { popKernelPackT(); }

#else
    errorQuda("sendGhost not built on single-GPU build");
#endif

  }


  void cudaColorSpinorField::unpackGhost(const void* ghost_spinor, const int nFace, 
					 const int dim, const QudaDirection dir, 
					 const int dagger, cudaStream_t* stream) 
  {
    const void *src = ghost_spinor;
  
    int ghost_offset = (dir == QUDA_BACKWARDS) ? ghostOffset[dim][0] : ghostOffset[dim][1];
    void *ghost_dst = (char*)ghost_recv_buffer_d[bufferIndex] + ghost_precision*ghost_offset;

    qudaMemcpyAsync(ghost_dst, src, ghost_face_bytes[dim], cudaMemcpyHostToDevice, *stream);
  }


  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhostExtended(const int nFace, const int R[], const QudaParity parity,
					       const int dim, const QudaDirection dir,
					       const int dagger, cudaStream_t *stream, bool zero_copy)
  {
#ifdef MULTI_GPU
    int face_num = (dir == QUDA_BACKWARDS) ? 0 : (dir == QUDA_FORWARDS) ? 1 : 2;
    void *packBuffer[2*QUDA_MAX_DIM];
    MemoryLocation location[2*QUDA_MAX_DIM];

    if (zero_copy) {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_hd[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_hd[bufferIndex][d][1];
	location[2*d+0] = Host;
	location[2*d+1] = Host;
      }
    } else {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_d[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_d[bufferIndex][d][1];
	location[2*d+0] = Device;
	location[2*d+1] = Device;
      }
    }

    packFaceExtended(packBuffer, *this, location[0], nFace, R, dagger, parity, dim, face_num, *stream);
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
    void *packBuffer[2*QUDA_MAX_DIM];
    MemoryLocation location[2*QUDA_MAX_DIM];

    if (zero_copy) {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_hd[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_hd[bufferIndex][d][1];
	location[2*d+0] = Host;
	location[2*d+1] = Host;
      }
    } else {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_d[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_d[bufferIndex][d][1];
	location[2*d+0] = Device;
	location[2*d+1] = Device;
      }
    }

    packFaceExtended(packBuffer, *this, location[0], nFace, R, dagger, parity, dim, face_num, *stream, unpack);
#else
    errorQuda("unpackGhostExtended not built on single-GPU build");
#endif
  }


  cudaStream_t *stream;

  void cudaColorSpinorField::createComms(int nFace, bool spin_project) {

    allocateGhostBuffer(nFace,spin_project); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs its comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face_h[0] != ghost_pinned_send_buffer_h[0]) || (my_face_h[1] != ghost_pinned_send_buffer_h[1]) ||
      (from_face_h[0] != ghost_pinned_recv_buffer_h[0]) || (from_face_h[1] != ghost_pinned_recv_buffer_h[1]) ||
      (ghost_field_tex[0] != ghost_recv_buffer_d[0]) || (ghost_field_tex[1] != ghost_recv_buffer_d[1]) || // receive buffers
      ghost_precision_reset; // ghost_precision has changed

    if (!initComms || comms_reset) {

      LatticeField::createComms();

      // reinitialize the ghost receive pointers
      for (int i=0; i<nDimComms; ++i) {
	if (commDimPartitioned(i)) {
	  for (int b=0; b<2; b++) {
	    ghost[b][i] = static_cast<char*>(ghost_recv_buffer_d[b]) + ghostOffset[i][0]*ghost_precision;
	    if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION)
	      ghostNorm[b][i] = static_cast<char*>(ghost_recv_buffer_d[b]) + ghostNormOffset[i][0]*QUDA_SINGLE_PRECISION;
	  }
	}
      }

      ghost_precision_reset = false;
    }

    if (ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  void cudaColorSpinorField::streamInit(cudaStream_t *stream_p) {
    stream = stream_p;
  }

  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, int stream_idx,
				  MemoryLocation location[2*QUDA_MAX_DIM], MemoryLocation location_label,
                                  double a, double b, double c)
  {
    createComms(nFace); // must call this first

    const int dim=-1; // pack all partitioned dimensions

    packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[stream_idx], location, location_label, a, b, c);
  }

  void cudaColorSpinorField::packExtended(const int nFace, const int R[], const int parity, 
                                          const int dagger, const int dim,
                                          cudaStream_t *stream_p, const bool zero_copy)
  {
    createComms(nFace); // must call this first

    stream = stream_p;
 
    packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[zero_copy ? 0 : (Nstream-1)], zero_copy);
  }

  void cudaColorSpinorField::gather(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;

    // If stream_p != 0, use pack_stream, else use the stream array
    cudaStream_t *pack_stream = (stream_p) ? stream_p : stream+dir;

    if (dir%2 == 0) {
      // backwards copy to host
      if (comm_peer2peer_enabled(0,dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][0], nFace, dim, QUDA_BACKWARDS, dagger, pack_stream);
    } else {
      // forwards copy to host
      if (comm_peer2peer_enabled(1,dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][1], nFace, dim, QUDA_FORWARDS, dagger, pack_stream);
    }
  }


  void cudaColorSpinorField::recvStart(int nFace, int d, int dagger, cudaStream_t* stream_p, bool gdr) {

    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (dir == 0) { // receive from forwards
      // receive from the processor in the +1 direction
      if (comm_peer2peer_enabled(1,dim)) {
	comm_start(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
        comm_start(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_fwd[bufferIndex][dim]);
      }
    } else { // receive from backwards
      // receive from the processor in the -1 direction
      if (comm_peer2peer_enabled(0,dim)) {
	comm_start(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
        comm_start(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_back[bufferIndex][dim]);
      }
    }
  }


  void cudaColorSpinorField::sendStart(int nFace, int d, int dagger, cudaStream_t* stream_p, bool gdr, bool remote_write) {

    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    int Nvec = (nSpin == 1 || ghost_precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2)/(nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom
    int Npad = Nint/Nvec;

    if (!comm_peer2peer_enabled(dir,dim)) {
      if (dir == 0)
	if (gdr) comm_start(mh_send_rdma_back[bufferIndex][dim]);
	else comm_start(mh_send_back[bufferIndex][dim]);
      else
	if (gdr) comm_start(mh_send_rdma_fwd[bufferIndex][dim]);
	else comm_start(mh_send_fwd[bufferIndex][dim]);
    } else { // doing peer-to-peer
      cudaStream_t *copy_stream = (stream_p) ? stream_p : stream + d;

      // if not using copy engine then the packing kernel will remotely write the halos
      if (!remote_write) {
        // all goes here
        void* ghost_dst = static_cast<char*>(ghost_remote_send_buffer_d[bufferIndex][dim][dir])
          + ghost_precision*ghostOffset[dim][(dir+1)%2];
        void *ghost_norm_dst = static_cast<char*>(ghost_remote_send_buffer_d[bufferIndex][dim][dir])
          + QUDA_SINGLE_PRECISION*ghostNormOffset[dim][(d+1)%2];

        if (ghost_precision != precision) pushKernelPackT(true);

        if (dim != 3 || getKernelPackT()) {

          void* ghost_dst = static_cast<char*>(ghost_remote_send_buffer_d[bufferIndex][dim][dir])
            + ghost_precision*ghostOffset[dim][(dir+1)%2];
          cudaMemcpyAsync(ghost_dst,
                          my_face_dim_dir_d[bufferIndex][dim][dir],
                          ghost_face_bytes[dim],
                          cudaMemcpyDeviceToDevice,
                          *copy_stream); // copy to forward processor

        } else {

          const int nParity = siteSubset;
          const int x4 = nDim==5 ? x[4] : 1;
          const int Nt_minus_offset = (volumeCB - nFace*ghostFaceCB[3])/x4;

          int offset = 0;
          if (nSpin == 1) {
            offset = (dir == 0) ? 0 : Nt_minus_offset;
          } else if (nSpin == 4) {
            // !dagger: send lower components backwards, send upper components forwards
            // dagger: send upper components backwards, send lower components forwards
            bool upper = dagger ? true : false;
            if (dir == 1) upper = !upper;
            int lower_spin_offset = Npad*stride;
            if (upper) offset = (dir == 0 ? 0 : Nt_minus_offset);
            else offset = lower_spin_offset + (dir == 0 ? 0 : Nt_minus_offset);
          }

          size_t len = nFace*(ghostFaceCB[3]/x4)*Nvec*ghost_precision;
          size_t dpitch = x4*len;
          size_t spitch = stride*Nvec*ghost_precision;

          for (int parity = 0; parity<nParity; parity++) {
            for (int s=0; s<x4; s++) {
              void *dst = (char*)ghost_dst + s*len + parity*nFace*Nint*ghostFaceCB[3]*ghost_precision;
              void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*ghost_precision + parity*bytes/2;
              // start the copy
              cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

              // we can probably issue this as a single cudaMemcpy2d along the fifth dimension
              if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) {
                size_t len = nFace*(ghostFace[3]/x4)*sizeof(float);
                int norm_offset = (dir == 0) ? 0 : Nt_minus_offset*sizeof(float);
                void *dst = (char*)ghost_norm_dst + s*len + parity*nFace*ghostFaceCB[3]*sizeof(float);
                void *src = (char*)norm + norm_offset + s*(volumeCB/x4)*sizeof(float) + parity*norm_bytes/2;
                cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, *copy_stream);
              }
            }
          } // fifth dimension
        } // parity
      } // remote_write

      if (ghost_precision != precision) popKernelPackT();

      if (dir == 0) {
	// record the event
	qudaEventRecord(ipcCopyEvent[bufferIndex][0][dim], *copy_stream);
	// send to the processor in the -1 direction
	comm_start(mh_send_p2p_back[bufferIndex][dim]);
      } else {
	qudaEventRecord(ipcCopyEvent[bufferIndex][1][dim], *copy_stream);
	// send to the processor in the +1 direction
	comm_start(mh_send_p2p_fwd[bufferIndex][dim]);
      }
    }
  }

  void cudaColorSpinorField::commsStart(int nFace, int dir, int dagger, cudaStream_t* stream_p, bool gdr_send, bool gdr_recv) {
    recvStart(nFace, dir, dagger, stream_p, gdr_recv);
    sendStart(nFace, dir, dagger, stream_p, gdr_send);
  }


  static bool complete_recv_fwd[QUDA_MAX_DIM] = { };
  static bool complete_recv_back[QUDA_MAX_DIM] = { };
  static bool complete_send_fwd[QUDA_MAX_DIM] = { };
  static bool complete_send_back[QUDA_MAX_DIM] = { };

  int cudaColorSpinorField::commsQuery(int nFace, int d, int dagger, cudaStream_t *stream_p, bool gdr_send, bool gdr_recv) {

    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;

    if (!commDimPartitioned(dim)) return 1;
    if ((gdr_send || gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (dir==0) {

      // first query send to backwards
      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_p2p_back[bufferIndex][dim]);
      } else if (gdr_send) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_back[bufferIndex][dim]);
      }

      // second query receive from forwards
      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr_recv) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_fwd[bufferIndex][dim]);
      }

      if (complete_recv_fwd[dim] && complete_send_back[dim]) {
	complete_send_back[dim] = false;
	complete_recv_fwd[dim] = false;
	return 1;
      }

    } else { // dir == 1

      // first query send to forwards
      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_p2p_fwd[bufferIndex][dim]);
      } else if (gdr_send) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_fwd[bufferIndex][dim]);
      }

      // second query receive from backwards
      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr_recv) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_back[bufferIndex][dim]);
      }

      if (complete_recv_back[dim] && complete_send_fwd[dim]) {
	complete_send_fwd[dim] = false;
	complete_recv_back[dim] = false;
	return 1;
      }

    }

    return 0;
  }

  void cudaColorSpinorField::commsWait(int nFace, int d, int dagger, cudaStream_t *stream_p, bool gdr_send, bool gdr_recv) {

    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;

    if (!commDimPartitioned(dim)) return;
    if ( (gdr_send && gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (dir==0) {

      // first wait on send to backwards
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_send_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][0][dim]);
      } else if (gdr_send) {
	comm_wait(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_back[bufferIndex][dim]);
      }

      // second wait on receive from forwards
      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_recv_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][1][dim]);
      } else if (gdr_recv) {
	comm_wait(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_fwd[bufferIndex][dim]);
      }

    } else {

      // first wait on send to forwards
      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_send_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][1][dim]);
      } else if (gdr_send) {
	comm_wait(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_fwd[bufferIndex][dim]);
      }

      // second wait on receive from backwards
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_recv_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][0][dim]);
      } else if (gdr_recv) {
	comm_wait(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_back[bufferIndex][dim]);
      }

    }

    return;
  }

  void cudaColorSpinorField::scatter(int nFace, int dagger, int dim_dir, cudaStream_t* stream_p)
  {
    // note this is scatter centric, so input expects dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards), so here we need flip to receive centric

    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 recive from backwards
    if (!commDimPartitioned(dim)) return;

    if (comm_peer2peer_enabled(dir,dim)) return;
    unpackGhost(from_face_dim_dir_h[bufferIndex][dim][dir], nFace, dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, dagger, stream_p);
  }

  void cudaColorSpinorField::scatter(int nFace, int dagger, int dim_dir)
  {
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards), so here we need flip to receive centric

    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 receive from backwards
    if (!commDimPartitioned(dim)) return;

    if (comm_peer2peer_enabled(dir,dim)) return;
    unpackGhost(from_face_dim_dir_h[bufferIndex][dim][dir], nFace, dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, dagger, &stream[dim_dir]);
  }

  void cudaColorSpinorField::scatterExtended(int nFace, int parity, int dagger, int dim_dir)
  {
    bool zero_copy = false;
    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 receive from backwards
    if (!commDimPartitioned(dim)) return;
    unpackGhostExtended(from_face_dim_dir_h[bufferIndex][dim][dir], nFace, static_cast<QudaParity>(parity), dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/], zero_copy);
  }
 
  void cudaColorSpinorField::exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination_,
					   const MemoryLocation *halo_location_, bool gdr_send, bool gdr_recv,
					   QudaPrecision ghost_precision_)  const {

    if (ghost_precision_ != QUDA_INVALID_PRECISION && ghost_precision != ghost_precision_) {
      ghost_precision_reset = true;
      ghost_precision_old = ghost_precision;
      ghost_precision = ghost_precision_;
    }

    // not overriding the ghost precision, but we did previously so need to update
    if (ghost_precision == QUDA_INVALID_PRECISION && ghost_precision != precision) {
      ghost_precision_reset = true;
      ghost_precision_old = ghost_precision;
      ghost_precision = precision;
    }

    if ((gdr_send || gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");
    pushKernelPackT(true); // ensure kernel packing is enabled for all dimensions
    const_cast<cudaColorSpinorField&>(*this).streamInit(streams); // ensures streams are set (needed for p2p)
    const_cast<cudaColorSpinorField&>(*this).createComms(nFace, false);

    // first set default values to device if needed
    MemoryLocation pack_destination[2*QUDA_MAX_DIM], halo_location[2*QUDA_MAX_DIM];
    for (int i=0; i<2*nDimComms; i++) {
      pack_destination[i] = pack_destination_ ? pack_destination_[i] : Device;
      halo_location[i] = halo_location_ ? halo_location_[i] : Device;
    }

    // Contiguous send buffers and we aggregate copies to reduce
    // latency.  Only if all locations are "Device" and no p2p
    bool fused_pack_memcpy = true;

    // Contiguous recv buffers and we aggregate copies to reduce
    // latency.  Only if all locations are "Device" and no p2p
    bool fused_halo_memcpy = true;

    bool pack_host = false; // set to true if any of the ghost packing is being done to Host memory
    bool halo_host = false; // set to true if the final halos will be left in Host memory

    void *send[2*QUDA_MAX_DIM];
    for (int d=0; d<nDimComms; d++) {
      for (int dir=0; dir<2; dir++) {
	send[2*d+dir] = pack_destination[2*d+dir] == Host ? my_face_dim_dir_hd[bufferIndex][d][dir] : my_face_dim_dir_d[bufferIndex][d][dir];
	ghost_buf[2*d+dir] = halo_location[2*d+dir] == Host ? from_face_dim_dir_hd[bufferIndex][d][dir] : from_face_dim_dir_d[bufferIndex][d][dir];
      }

      // if doing p2p, then we must pack to and load the halo from device memory
      for (int dir=0; dir<2; dir++) {
	if (comm_peer2peer_enabled(dir,d)) { pack_destination[2*d+dir] = Device; halo_location[2*d+1-dir] = Device; }
      }

      // if zero-copy packing or p2p is enabled then we cannot do fused memcpy
      if (pack_destination[2*d+0] != Device || pack_destination[2*d+1] != Device || comm_peer2peer_enabled_global()) fused_pack_memcpy = false;
      // if zero-copy halo read or p2p is enabled then we cannot do fused memcpy
      if (halo_location[2*d+0] != Device || halo_location[2*d+1] != Device || comm_peer2peer_enabled_global()) fused_halo_memcpy = false;

      if (pack_destination[2*d+0] == Host || pack_destination[2*d+1] == Host) pack_host = true;
      if (halo_location[2*d+0] == Host || halo_location[2*d+1] == Host) halo_host = true;
    }

    // Error if zero-copy and p2p for now
    if ( (pack_host || halo_host) && comm_peer2peer_enabled_global()) errorQuda("Cannot use zero-copy memory with peer-to-peer comms yet");

    genericPackGhost(send, *this, parity, nFace, dagger, pack_destination); // FIXME - need support for asymmetric topologies

    size_t total_bytes = 0;
    for (int i=0; i<nDimComms; i++) if (comm_dim_partitioned(i)) total_bytes += 2*ghost_face_bytes[i]; // 2 for fwd/bwd

    if (!gdr_send)  {
      if (!fused_pack_memcpy) {
	for (int i=0; i<nDimComms; i++) {
	  if (comm_dim_partitioned(i)) {
	    if (pack_destination[2*i+0] == Device && !comm_peer2peer_enabled(0,i) && // fuse forwards and backwards if possible
		pack_destination[2*i+1] == Device && !comm_peer2peer_enabled(1,i)) {
	      cudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][i][0], my_face_dim_dir_d[bufferIndex][i][0], 2*ghost_face_bytes[i], cudaMemcpyDeviceToHost, 0);
	    } else {
	      if (pack_destination[2*i+0] == Device && !comm_peer2peer_enabled(0,i))
		cudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][i][0], my_face_dim_dir_d[bufferIndex][i][0], ghost_face_bytes[i], cudaMemcpyDeviceToHost, 0);
	      if (pack_destination[2*i+1] == Device && !comm_peer2peer_enabled(1,i))
		cudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][i][1], my_face_dim_dir_d[bufferIndex][i][1], ghost_face_bytes[i], cudaMemcpyDeviceToHost, 0);
	    }
	  }
	}
      } else if (total_bytes && !pack_host) {
	cudaMemcpyAsync(my_face_h[bufferIndex], ghost_send_buffer_d[bufferIndex], total_bytes, cudaMemcpyDeviceToHost, 0);
      }
    }

    // prepost receive
    for (int i=0; i<2*nDimComms; i++) const_cast<cudaColorSpinorField*>(this)->recvStart(nFace, i, dagger, 0, gdr_recv);

    bool sync = pack_host ? true : false; // no p2p if pack_host so we need to synchronize
    // if not p2p in any direction then need to synchronize before MPI
    for (int i=0; i<nDimComms; i++) if (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i)) sync = true;
    if (sync) qudaDeviceSynchronize(); // need to make sure packing and/or memcpy has finished before kicking off MPI

    for (int p2p=0; p2p<2; p2p++) {
      for (int dim=0; dim<nDimComms; dim++) {
	for (int dir=0; dir<2; dir++) {
	  if ( (comm_peer2peer_enabled(dir,dim) + p2p) % 2 == 0 ) { // issue non-p2p transfers first
	    const_cast<cudaColorSpinorField*>(this)->sendStart(nFace, 2*dim+dir, dagger, 0, gdr_send);
	  }
	}
      }
    }

    bool comms_complete[2*QUDA_MAX_DIM] = { };
    int comms_done = 0;
    while (comms_done < 2*nDimComms) { // non-blocking query of each exchange and exit once all have completed
      for (int dim=0; dim<nDimComms; dim++) {
	for (int dir=0; dir<2; dir++) {
	  if (!comms_complete[dim*2+dir]) {
	    comms_complete[2*dim+dir] = const_cast<cudaColorSpinorField*>(this)->commsQuery(nFace, 2*dim+dir, dagger, 0, gdr_send, gdr_recv);
	    if (comms_complete[2*dim+dir]) {
	      comms_done++;
	      if (comm_peer2peer_enabled(1-dir,dim)) qudaStreamWaitEvent(0, ipcRemoteCopyEvent[bufferIndex][1-dir][dim], 0);
	    }
	  }
	}
      }
    }

    if (!gdr_recv) {
      if (!fused_halo_memcpy) {
	for (int i=0; i<nDimComms; i++) {
	  if (comm_dim_partitioned(i)) {
	    if (halo_location[2*i+0] == Device && !comm_peer2peer_enabled(0,i) && // fuse forwards and backwards if possible
		halo_location[2*i+1] == Device && !comm_peer2peer_enabled(1,i)) {
	      cudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][i][0], from_face_dim_dir_h[bufferIndex][i][0], 2*ghost_face_bytes[i], cudaMemcpyHostToDevice, 0);
	    } else {
	      if (halo_location[2*i+0] == Device && !comm_peer2peer_enabled(0,i))
		cudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][i][0], from_face_dim_dir_h[bufferIndex][i][0], ghost_face_bytes[i], cudaMemcpyHostToDevice, 0);
	      if (halo_location[2*i+1] == Device && !comm_peer2peer_enabled(1,i))
		cudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][i][1], from_face_dim_dir_h[bufferIndex][i][1], ghost_face_bytes[i], cudaMemcpyHostToDevice, 0);
	    }
	  }
	}
      } else if (total_bytes && !halo_host) {
	cudaMemcpyAsync(ghost_recv_buffer_d[bufferIndex], from_face_h[bufferIndex], total_bytes, cudaMemcpyHostToDevice, 0);
      }
    }

    popKernelPackT(); // restore kernel packing
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
    param.create = (sourceType == QUDA_POINT_SOURCE ? QUDA_ZERO_FIELD_CREATE : QUDA_NULL_FIELD_CREATE);

    cpuColorSpinorField tmp(param);
    tmp.Source(sourceType, st, s, c);
    *this = tmp;
  }


  // This existed before but was never called. It kept giving me segfaults,
  // but it would be ideal if I could've gotten it to work.
  /*
  template<typename Float,int Ns, int Nc>
  void genericCudaPrintVector(const cudaColorSpinorField& field, unsigned int i)
  {
    typedef typename colorspinor_mapper<Float,Ns,Nc>::type ColorSpinor;
    ColorSpinor csmap(field);
    typedef typename mapper<Float>::type TrueFloat;
    TrueFloat* data_gpu=0;
    cudaMalloc((void**)&data_gpu, 2*Ns*Nc*sizeof(TrueFloat));
    csmap.load(data_gpu,i);
    TrueFloat* data_cpu = new TrueFloat[2*Ns*Nc];
    cudaMemcpy(data_cpu, data_gpu, 2*Ns*Nc*sizeof(TrueFloat), cudaMemcpyDeviceToHost);
    cudaFree(data_gpu);
    for (int s = 0; s < Ns; s++) {
      printfQuda("x = %u, s = %d, { ", i, s);
      for (int c = 0; c < Nc; c++) {
        printfQuda("(%f,%f)", data_cpu[(s*Nc+c)*2], data_cpu[(s*Nc+c)*2+1]);
        printfQuda("%s", c == Nc-1 ? " " : " , ");
      }
      printfQuda("}\n");
    }
    delete[] data_cpu;
  }*/

  template <typename StoreType, int Ns, int Nc, QudaFieldOrder FieldOrder>
  void genericCudaPrintVector(const cudaColorSpinorField& field, unsigned int i)
  {

    typedef colorspinor::AccessorCB<StoreType,Ns,Nc,1,FieldOrder> AccessorType;

    AccessorType A(field);

    // Register type
    typedef typename scalar<typename mapper<StoreType>::type>::type Float;

    // Allocate a real+imag component for the storage type.
    StoreType indiv_num[2];

    // Allocate space for the full site.
    Float* data_cpu = new Float[2*Ns*Nc];

    // Grab the pointer to the field.
    complex<StoreType>* field_ptr = (complex<StoreType>*)field.V();

    // Grab the pointer to the norm field. Might be ignored as appropriate.
    float* norm_ptr = (float*)field.Norm();
    float scale = 1.0;

    if (isFixed<StoreType>::value)
    {
      cudaMemcpy(&scale, &norm_ptr[i], sizeof(float), cudaMemcpyDeviceToHost);
      scale *= fixedInvMaxValue<StoreType>::value;
    }

    for (int s = 0; s < Ns; s++)
    {
      for (int c = 0; c < Nc; c++)
      {
        cudaMemcpy(indiv_num, &field_ptr[A.index(i%2,i/2,s,c,0)], 2*sizeof(StoreType), cudaMemcpyDeviceToHost);
        data_cpu[2*(c+Nc*s)] = scale*static_cast<Float>(indiv_num[0]);
        data_cpu[2*(c+Nc*s)+1] = scale*static_cast<Float>(indiv_num[1]);
      }
    }
    // print
    for (int s = 0; s < Ns; s++) {
      printfQuda("x = %u, s = %d, { ", i, s);
      for (int c = 0; c < Nc; c++) {
        printfQuda("(%f,%f)", data_cpu[(s*Nc+c)*2], data_cpu[(s*Nc+c)*2+1]);
        printfQuda("%s", c == Nc-1 ? " " : " , ");
      }
      printfQuda("}\n");
    }
    //for (int j = 0; j < 2*Ns*Nc; j++)
    //  printfQuda("%f\n", data_cpu[j]);
    
    delete[] data_cpu;

  }

  template<typename Float,int Ns, int Nc>
  void genericCudaPrintVector(const cudaColorSpinorField& field, unsigned int i)
  {
    switch (field.FieldOrder()) {
    case QUDA_FLOAT_FIELD_ORDER:
      genericCudaPrintVector<Float,Ns,Nc,QUDA_FLOAT_FIELD_ORDER>(field,i);
      break;
    case QUDA_FLOAT2_FIELD_ORDER:
      genericCudaPrintVector<Float,Ns,Nc,QUDA_FLOAT2_FIELD_ORDER>(field,i);
      break;
    case QUDA_FLOAT4_FIELD_ORDER:
      genericCudaPrintVector<Float,Ns,Nc,QUDA_FLOAT4_FIELD_ORDER>(field,i);
      break;
    case QUDA_SPACE_SPIN_COLOR_FIELD_ORDER:
      genericCudaPrintVector<Float,Ns,Nc,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(field,i);
      break;
    case QUDA_SPACE_COLOR_SPIN_FIELD_ORDER:
      genericCudaPrintVector<Float,Ns,Nc,QUDA_SPACE_COLOR_SPIN_FIELD_ORDER>(field,i);
      break;
    default:
      errorQuda("Unsupported field order %d", field.FieldOrder());
    }
  }

  template<typename Float>
  void genericCudaPrintVector(const cudaColorSpinorField& field, unsigned int i) {
    if (field.Ncolor() == 3 && field.Nspin() == 4)  {
      genericCudaPrintVector<Float,4,3>(field,i);
    }
    else if (field.Ncolor() == 3 && field.Nspin() == 1)  {
      genericCudaPrintVector<Float,1,3>(field,i);
    }
    else if (field.Ncolor() == 6 && field.Nspin() == 2) { // wilson free field MG
      genericCudaPrintVector<Float,2,6>(field,i);
    }
    else if (field.Ncolor() == 24 && field.Nspin() == 2) { // common value for Wilson, also staggered free field
      genericCudaPrintVector<Float,2,24>(field,i);
    }
    else if (field.Ncolor() == 32 && field.Nspin() == 2) {
      genericCudaPrintVector<Float,2,32>(field,i);
    } else {
      errorQuda("Not supported Ncolor = %d, Nspin = %d", field.Ncolor(), field.Nspin());
    }
  }

  void cudaColorSpinorField::PrintVector(unsigned int i) {
    
    switch (this->precision)
    {
      case QUDA_QUARTER_PRECISION:
        genericCudaPrintVector<char>(*this,i);
        break;
      case QUDA_HALF_PRECISION:
        genericCudaPrintVector<short>(*this,i);
        break;
      case QUDA_SINGLE_PRECISION:
        genericCudaPrintVector<float>(*this,i);
        break;
      case QUDA_DOUBLE_PRECISION:
        genericCudaPrintVector<double>(*this, i);
        break;
      default:
        errorQuda("Unsupported precision = %d\n", this->precision);
    }

    /*
    ColorSpinorParam param(*this);
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_NULL_FIELD_CREATE;

    cpuColorSpinorField tmp(param);
    tmp = *this;
    tmp.PrintVector(i);*/
  }


} // namespace quda

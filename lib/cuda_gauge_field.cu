#include <string.h>
#include <gauge_field.h>
#include <face_quda.h>
#include <typeinfo>
#include <misc_helpers.h>
#include <blas_quda.h>

namespace quda {

  cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) :
    GaugeField(param), gauge(0), even(0), odd(0), backed_up(false)
  {
    if ((order == QUDA_QDP_GAUGE_ORDER || order == QUDA_QDPJIT_GAUGE_ORDER) && 
        create != QUDA_REFERENCE_FIELD_CREATE) {
      errorQuda("QDP ordering only supported for reference fields");
    }

    if (order == QUDA_QDP_GAUGE_ORDER || order == QUDA_MILC_GAUGE_ORDER ||
	order == QUDA_TIFR_GAUGE_ORDER || order == QUDA_BQCD_GAUGE_ORDER ||
	order == QUDA_CPS_WILSON_GAUGE_ORDER) 
      errorQuda("Field ordering %d presently disabled for this type", order);

#ifdef MULTI_GPU
    if (link_type != QUDA_ASQTAD_MOM_LINKS &&
	ghostExchange == QUDA_GHOST_EXCHANGE_PAD) {
      bool pad_check = true;
      for (int i=0; i<nDim; i++)
	if (pad < nFace*surfaceCB[i]) pad_check = false;
      if (!pad_check)
	errorQuda("cudaGaugeField being constructed with insufficient padding\n");
    }
#endif

    if(create != QUDA_NULL_FIELD_CREATE &&  
        create != QUDA_ZERO_FIELD_CREATE && 
        create != QUDA_REFERENCE_FIELD_CREATE){
      errorQuda("ERROR: create type(%d) not supported yet\n", create);
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      gauge = device_malloc(bytes);  
      if (create == QUDA_ZERO_FIELD_CREATE) cudaMemset(gauge, 0, bytes);
    } else { 
      gauge = param.gauge;
    }

    if ( !isNative() ) {
      for (int i=0; i<nDim; i++) {
        size_t nbytes = nFace * surface[i] * reconstruct * precision;
        ghost[i] = nbytes ? device_malloc(nbytes) : NULL;
      }        
    }

    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD) {
      if (create == QUDA_REFERENCE_FIELD_CREATE) exchangeGhost(); 
    }

    even = gauge;
    odd = (char*)gauge + bytes/2; 

#ifdef USE_TEXTURE_OBJECTS
    createTexObject(evenTex, even);
    createTexObject(oddTex, odd);
    if(reconstruct == QUDA_RECONSTRUCT_13 || reconstruct == QUDA_RECONSTRUCT_9)
    {  // Create texture objects for the phases
      const int isPhase = 1;
      createTexObject(evenPhaseTex, (char*)even + phase_offset, isPhase);
      createTexObject(oddPhaseTex, (char*)odd + phase_offset, isPhase);
    }
#endif

  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaGaugeField::createTexObject(cudaTextureObject_t &tex, void *field, int isPhase) {

    if( isNative() ){
      // create the texture for the field components
      cudaChannelFormatDesc desc;
      memset(&desc, 0, sizeof(cudaChannelFormatDesc));
      if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
      else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

      if(isPhase){
        if(precision == QUDA_DOUBLE_PRECISION){
          desc.x = 8*sizeof(int);
          desc.y = 8*sizeof(int);
          desc.z = 0;
          desc.w = 0;
        }else{
          desc.x = 8*precision;
          desc.y = desc.z = desc.w = 0;
        }
      }else{
        // always four components regardless of precision
        if (precision == QUDA_DOUBLE_PRECISION) {
          desc.x = 8*sizeof(int);
          desc.y = 8*sizeof(int);
          desc.z = 8*sizeof(int);
          desc.w = 8*sizeof(int);
        } else {
          desc.x = 8*precision;
          desc.y = 8*precision;
          desc.z = (reconstruct == 18) ? 0 : 8*precision; // float2 or short2 for 18 reconstruct
          desc.w = (reconstruct == 18) ? 0 : 8*precision;
        }
      }

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = field;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = isPhase ? phase_bytes/2 : (bytes-phase_bytes)/2;

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
      else texDesc.readMode = cudaReadModeElementType;

      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
      checkCudaError();
    }
  }

  void cudaGaugeField::destroyTexObject() {
    if( isNative() ){
      cudaDestroyTextureObject(evenTex);
      cudaDestroyTextureObject(oddTex);
      if(reconstruct == QUDA_RECONSTRUCT_9 || reconstruct == QUDA_RECONSTRUCT_13){
        cudaDestroyTextureObject(evenPhaseTex);
        cudaDestroyTextureObject(oddPhaseTex);
      }
      checkCudaError();
    }
  }
#endif

  cudaGaugeField::~cudaGaugeField()
  {
#ifdef USE_TEXTURE_OBJECTS
    destroyTexObject();
#endif

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (gauge) device_free(gauge);
    }

    if ( !isNative() ) {
      for (int i=0; i<nDim; i++) {
        if (ghost[i]) device_free(ghost[i]);
      }
    }

  }

  // This does the exchange of the gauge field ghost zone and places it
  // into the ghost array.
  void cudaGaugeField::exchangeGhost() {
    if (geometry != QUDA_VECTOR_GEOMETRY) 
      errorQuda("Cannot exchange for %d geometry gauge field", geometry);

    void *ghost_[QUDA_MAX_DIM];
    void *send[QUDA_MAX_DIM];
    for (int d=0; d<nDim; d++) {
      ghost_[d] = isNative() ? device_malloc(nFace*surface[d]*reconstruct*precision) : ghost[d];
      send[d] = device_malloc(nFace*surface[d]*reconstruct*precision);
    }

    // get the links into contiguous buffers
    extractGaugeGhost(*this, send);

    // communicate between nodes
    FaceBuffer faceBuf(x, nDim, reconstruct, nFace, precision);
    faceBuf.exchangeLink(ghost_, send, QUDA_CUDA_FIELD_LOCATION);

    for (int d=0; d<nDim; d++) device_free(send[d]);

    if (isNative()) {
      // copy from ghost into the padded region in gauge
      copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, 0, ghost_, 1);
      for (int d=0; d<nDim; d++) device_free(ghost_[d]);
    }
  }

  void cudaGaugeField::exchangeExtendedGhost(const int *R, bool no_comms_fill) {
    
    void *send[QUDA_MAX_DIM];
    void *recv[QUDA_MAX_DIM];
    void *send_d[QUDA_MAX_DIM];
    void *recv_d[QUDA_MAX_DIM];
    size_t bytes[QUDA_MAX_DIM];

    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d) && !no_comms_fill) continue;
      // store both parities and directions in each
      bytes[d] = surface[d] * R[d] * geometry * reconstruct * precision;
      send_d[d] = device_malloc(2 * bytes[d]);
      recv_d[d] = device_malloc(2 * bytes[d]);
    }

#ifndef GPU_COMMS
    void *send_h[QUDA_MAX_DIM];
    void *recv_h[QUDA_MAX_DIM];
    size_t total_bytes = 0;
    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d)) continue;
      total_bytes += 4*bytes[d];
    }
    resizeBufferPinned(total_bytes);

    size_t offset = 0;
    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d)) continue;

      recv_h[d] = static_cast<char*>(bufferPinned) + offset;
      send_h[d] = static_cast<char*>(recv_h[d]) + 2*bytes[d];
      offset += 4*bytes[d];
    }
#endif

    // do the exchange
    MsgHandle *mh_recv_back[QUDA_MAX_DIM];
    MsgHandle *mh_recv_fwd[QUDA_MAX_DIM];
    MsgHandle *mh_send_fwd[QUDA_MAX_DIM];
    MsgHandle *mh_send_back[QUDA_MAX_DIM];

    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d)) continue;
#ifdef GPU_COMMS
      recv[d] = recv_d[d];
      send[d] = send_d[d];
#else
      recv[d] = recv_h[d];
      send[d] = send_h[d];
#endif

      // look into storing these for later
      mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
      mh_recv_fwd[d]  = comm_declare_receive_relative(static_cast<char*>(recv[d])+bytes[d], 
						      d, +1, bytes[d]);
      mh_send_back[d] = comm_declare_send_relative(send[d], d, -1, bytes[d]);
      mh_send_fwd[d]  = comm_declare_send_relative(static_cast<char*>(send[d])+bytes[d], 
						   d, +1, bytes[d]);
    }

    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d) && !no_comms_fill) continue;

      // FIXME why does this break if the order is switched?
      // prepost the receives
      if (commDimPartitioned(d)) {
	comm_start(mh_recv_fwd[d]);
	comm_start(mh_recv_back[d]);
      }

      //extract into a contiguous buffer
      extractExtendedGaugeGhost(*this, d, R, send_d, true);

      if (commDimPartitioned(d)) {
	
	// pipeline the forwards and backwards sending
#ifndef GPU_COMMS
	cudaMemcpyAsync(send_h[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost, streams[0]);
	cudaMemcpyAsync(static_cast<char*>(send_h[d])+bytes[d], 
			static_cast<char*>(send_d[d])+bytes[d], bytes[d], cudaMemcpyDeviceToHost, streams[1]);
#endif      
	
#ifndef GPU_COMMS
	cudaStreamSynchronize(streams[0]);
#endif
	comm_start(mh_send_back[d]);
	
#ifndef GPU_COMMS
	cudaStreamSynchronize(streams[1]);
#endif
	comm_start(mh_send_fwd[d]);
	
	// forwards recv
	comm_wait(mh_send_back[d]);
	comm_wait(mh_recv_fwd[d]);
#ifndef GPU_COMMS
	cudaMemcpyAsync(static_cast<char*>(recv_d[d])+bytes[d], 
			static_cast<char*>(recv_h[d])+bytes[d], bytes[d], cudaMemcpyHostToDevice, streams[0]);
#endif      
	
	// backwards recv
	comm_wait(mh_send_fwd[d]);
	comm_wait(mh_recv_back[d]);
#ifndef GPU_COMMS
	cudaMemcpyAsync(recv_d[d], recv_h[d], bytes[d], cudaMemcpyHostToDevice, streams[1]);
#endif      
      } else { // if just doing a local exchange to fill halo then need to swap faces
	cudaMemcpy(static_cast<char*>(recv_d[d])+bytes[d], send_d[d], bytes[d], cudaMemcpyDeviceToDevice);
	cudaMemcpy(recv_d[d], static_cast<char*>(send_d[d])+bytes[d], bytes[d], cudaMemcpyDeviceToDevice);
      }

      // inject back into the gauge field
      extractExtendedGaugeGhost(*this, d, R, recv_d, false);
    }

    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d) && !no_comms_fill) continue;

      if (commDimPartitioned(d)) {
	comm_free(mh_send_fwd[d]);
	comm_free(mh_send_back[d]);
	comm_free(mh_recv_back[d]);
	comm_free(mh_recv_fwd[d]);
      }

      device_free(send_d[d]);
      device_free(recv_d[d]);
    }

  }

  void cudaGaugeField::setGauge(void *gauge_)
  {
    if(create != QUDA_REFERENCE_FIELD_CREATE) {
      errorQuda("Setting gauge pointer is only allowed when create="
          "QUDA_REFERENCE_FIELD_CREATE type\n");
    }
    gauge = gauge_;
  }

  void cudaGaugeField::copy(const GaugeField &src) {
    if (this == &src) return;

    checkField(src);

    if (link_type == QUDA_ASQTAD_FAT_LINKS) {
      fat_link_max = src.LinkMax();
      if (precision == QUDA_HALF_PRECISION && fat_link_max == 0.0) 
        errorQuda("fat_link_max has not been computed");
    } else {
      fat_link_max = 1.0;
    }

    //if (src.Order() == QUDA_TIFR_GAUGE_ORDER) fat_link_max = src.Scale();

    if (typeid(src) == typeid(cudaGaugeField)) {
      // copy field and ghost zone into this field
      copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, 
          static_cast<const cudaGaugeField&>(src).gauge);
    } else if (typeid(src) == typeid(cpuGaugeField)) {
      LatticeField::resizeBufferPinned(bytes);

      // copy field and ghost zone into bufferPinned
      copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, bufferPinned, 
		       static_cast<const cpuGaugeField&>(src).gauge); 

      // this copies over both even and odd
      cudaMemcpy(gauge, bufferPinned, bytes, cudaMemcpyHostToDevice);
    } else {
      errorQuda("Invalid gauge field type");
    }

    // if we have copied from a source without a pad then we need to exchange
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD &&
	src.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD) {
      exchangeGhost(); 
    }

    checkCudaError();
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, const QudaFieldLocation &pack_location)
  {
    if (pack_location == QUDA_CUDA_FIELD_LOCATION) {
      if (cpu.Order() == QUDA_MILC_GAUGE_ORDER ||
	  cpu.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
	resizeBufferPinned(cpu.Bytes());
	memcpy(bufferPinned, cpu.Gauge_p(), cpu.Bytes());

	// run kernel directly using host-mapped input data
	void *bufferPinnedMapped;
	cudaHostGetDevicePointer(&bufferPinnedMapped, bufferPinned, 0);
	copyGenericGauge(*this, cpu, QUDA_CUDA_FIELD_LOCATION, gauge, bufferPinnedMapped);
      } else {
	errorQuda("Not implemented for order %d", cpu.Order());
      }
    } else if (pack_location == QUDA_CPU_FIELD_LOCATION) {
      copy(cpu);
    } else {
      errorQuda("Invalid pack location %d", pack_location);
    }

  }

  /*
     Copies the device gauge field to the host.
     - no reconstruction support
     - device data is always Float2 ordered
     - host data is a 1-dimensional array (MILC ordered)
     - no support for half precision
     - input and output precisions must match
   */
  template<typename FloatN, typename Float>
    static void storeGaugeField(Float* cpuGauge, FloatN *gauge, int bytes, int volumeCB, 
        int stride, QudaPrecision prec) 
    {  
      cudaStream_t streams[2];
      for (int i=0; i<2; i++) cudaStreamCreate(&streams[i]);

      FloatN *even = gauge;
      FloatN *odd = (FloatN*)((char*)gauge + bytes/2);

      size_t datalen = 4*2*volumeCB*gaugeSiteSize*sizeof(Float); // both parities
      void *unpacked = device_malloc(datalen);
      void *unpackedEven = unpacked;
      void *unpackedOdd = (char*)unpacked + datalen/2;

      //unpack even data kernel
      link_format_gpu_to_cpu((void*)unpackedEven, (void*)even, volumeCB, stride, prec, streams[0]);
#ifdef GPU_DIRECT
      cudaMemcpyAsync(cpuGauge, unpackedEven, datalen/2, cudaMemcpyDeviceToHost, streams[0]);
#else
      cudaMemcpy(cpuGauge, unpackedEven, datalen/2, cudaMemcpyDeviceToHost);
#endif

      //unpack odd data kernel
      link_format_gpu_to_cpu((void*)unpackedOdd, (void*)odd, volumeCB, stride, prec, streams[1]);
#ifdef GPU_DIRECT
      cudaMemcpyAsync(cpuGauge + 4*volumeCB*gaugeSiteSize, unpackedOdd, datalen/2, cudaMemcpyDeviceToHost, streams[1]);  
      for(int i=0; i<2; i++) cudaStreamSynchronize(streams[i]);
#else
      cudaMemcpy(cpuGauge + 4*volumeCB*gaugeSiteSize, unpackedOdd, datalen/2, cudaMemcpyDeviceToHost);  
#endif

      device_free(unpacked);
      for(int i=0; i<2; i++) cudaStreamDestroy(streams[i]);
    }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, const QudaFieldLocation &pack_location) const
  {
    // FIXME use the generic copying for the below copying
    // do device-side reordering then copy
    if (pack_location == QUDA_CUDA_FIELD_LOCATION) {
      // check parameters are suitable for device-side packing
      if (precision != cpu.Precision())
        errorQuda("cpu precision %d and cuda precision %d must be the same", 
            cpu.Precision(), precision);

      if (reconstruct != QUDA_RECONSTRUCT_NO) errorQuda("Only no reconstruction supported");
      if (order != QUDA_FLOAT2_GAUGE_ORDER) errorQuda("Only QUDA_FLOAT2_GAUGE_ORDER supported");
      if (cpu.Order() != QUDA_MILC_GAUGE_ORDER) errorQuda("Only QUDA_MILC_GAUGE_ORDER supported");

      if (precision == QUDA_DOUBLE_PRECISION){
        storeGaugeField((double*)cpu.gauge, (double2*)gauge, bytes, volumeCB, stride, precision);
      } else if (precision == QUDA_SINGLE_PRECISION){
        storeGaugeField((float*)cpu.gauge, (float2*)gauge, bytes, volumeCB, stride, precision);
      } else {
        errorQuda("Half precision not supported");
      }

    } else if (pack_location == QUDA_CPU_FIELD_LOCATION) { // do copy then host-side reorder
      resizeBufferPinned(bytes);

      // this copies over both even and odd
      cudaMemcpy(bufferPinned, gauge, bytes, cudaMemcpyDeviceToHost);
      checkCudaError();

      copyGenericGauge(cpu, *this, QUDA_CPU_FIELD_LOCATION, cpu.gauge, bufferPinned);
    } else {
      errorQuda("Invalid pack location %d", pack_location);
    }

  }

  void cudaGaugeField::backup() const {
    if (backed_up) errorQuda("Gauge field already backed up");
    backup_h = new char[bytes];
    cudaMemcpy(backup_h, gauge, bytes, cudaMemcpyDeviceToHost);
    checkCudaError();
    backed_up = true;
  }

  void cudaGaugeField::restore() {
    if (!backed_up) errorQuda("Cannot restore since not backed up");
    cudaMemcpy(gauge, backup_h, bytes, cudaMemcpyHostToDevice);
    delete []backup_h;
    checkCudaError();
    backed_up = false;
  }

  void setGhostSpinor(bool value);

  // Return the L2 norm squared of the gauge field
  double norm2(const cudaGaugeField &a) {

    if (a.FieldOrder() == QUDA_QDP_GAUGE_ORDER || 
        a.FieldOrder() == QUDA_QDPJIT_GAUGE_ORDER)
      errorQuda("Not implemented");

    int spin = 0;
    switch (a.Geometry()) {
      case QUDA_SCALAR_GEOMETRY:
        spin = 1;
        break;
      case QUDA_VECTOR_GEOMETRY:
        spin = a.Ndim();
        break;
      case QUDA_TENSOR_GEOMETRY:
        spin = a.Ndim() * (a.Ndim()-1) / 2;
        break;
      default:
        errorQuda("Unsupported field geometry %d", a.Geometry());
    }

    if (a.Precision() == QUDA_HALF_PRECISION) 
      errorQuda("Casting a cudaGaugeField into cudaColorSpinorField not possible in half precision");

    if (a.Reconstruct() == QUDA_RECONSTRUCT_13 || a.Reconstruct() == QUDA_RECONSTRUCT_9)
      errorQuda("Unsupported field reconstruct %d", a.Reconstruct());

    ColorSpinorParam spinor_param;
    spinor_param.nColor = a.Reconstruct()/2;
    spinor_param.nSpin = a.Ndim();
    spinor_param.nDim = spin;
    for (int d=0; d<a.Ndim(); d++) spinor_param.x[d] = a.X()[d];
    spinor_param.precision = a.Precision();
    spinor_param.pad = a.Pad();
    spinor_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    spinor_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    spinor_param.fieldOrder = (a.Precision() == QUDA_DOUBLE_PRECISION || spinor_param.nSpin == 1) ? 
    QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER; 
    spinor_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    spinor_param.create = QUDA_REFERENCE_FIELD_CREATE;
    spinor_param.v = (void*)a.Gauge_p();

    // quick hack to disable ghost zone creation which otherwise breaks this mapping on multi-gpu
    setGhostSpinor(false);
    cudaColorSpinorField b(spinor_param);
    setGhostSpinor(true);

    return norm2(b);
  }

} // namespace quda

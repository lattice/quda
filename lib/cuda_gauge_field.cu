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
    if (order == QUDA_QDP_GAUGE_ORDER) errorQuda("QDP ordering not supported");
    
    if(create != QUDA_NULL_FIELD_CREATE &&  
       create != QUDA_ZERO_FIELD_CREATE && 
       create != QUDA_REFERENCE_FIELD_CREATE){
      errorQuda("ERROR: create type(%d) not supported yet\n", create);
    }
  
    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      gauge = device_malloc(bytes);  
      if (create == QUDA_ZERO_FIELD_CREATE) cudaMemset(gauge, 0, bytes);
    } else { // for reference fields (e.g., external fields) we need to do the ghost exchange
      gauge = param.gauge;
      exchangeGhost();
    }

    even = gauge;
    odd = (char*)gauge + bytes/2; 

#ifdef USE_TEXTURE_OBJECTS
    createTexObject(evenTex, even);
    createTexObject(oddTex, odd);
#endif
  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaGaugeField::createTexObject(cudaTextureObject_t &tex, void *field) {
    // create the texture for the field components
    cudaChannelFormatDesc desc;
    memset(&desc, 0, sizeof(cudaChannelFormatDesc));
    if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
    else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

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

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = field;
    resDesc.res.linear.desc = desc;
    resDesc.res.linear.sizeInBytes = bytes/2;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
    else texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    checkCudaError();
  }

  void cudaGaugeField::destroyTexObject() {
    cudaDestroyTextureObject(evenTex);
    cudaDestroyTextureObject(oddTex);
    checkCudaError();
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
  }

  // This does the exchange of the gauge field ghost zone and places it
  // into the ghost array.
  void cudaGaugeField::exchangeGhost() {
    if (ghostExchange) return;

    void *ghost[QUDA_MAX_DIM];
    void *send[QUDA_MAX_DIM];
    for (int d=0; d<nDim; d++) {
      ghost[d] = device_malloc(nFace*surface[d]*reconstruct*precision);
      send[d] = device_malloc(nFace*surface[d]*reconstruct*precision);
    }

    // get the links into contiguous buffers
    extractGaugeGhost(*this, send);

    // communicate between nodes
    FaceBuffer faceBuf(x, nDim, reconstruct, nFace, precision);
    faceBuf.exchangeLink(ghost, send, QUDA_CUDA_FIELD_LOCATION);

    // copy from ghost into the padded region in gauge
    copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, 0, ghost);

    for (int d=0; d<nDim; d++) {
      device_free(send[d]);
      device_free(ghost[d]);
    }

    ghostExchange = true;
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

    if (geometry != QUDA_VECTOR_GEOMETRY) errorQuda("Only vector geometry is supported");
    checkField(src);

    if (link_type == QUDA_ASQTAD_FAT_LINKS) {
      fat_link_max = src.LinkMax();
      if (precision == QUDA_HALF_PRECISION && fat_link_max == 0.0) 
	errorQuda("fat_link_max has not been computed");
    }
    
    if (typeid(src) == typeid(cudaGaugeField)) {
      // copy field and ghost zone into this field
      copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, 
		       static_cast<const cudaGaugeField&>(src).gauge);

    } else if (typeid(src) == typeid(cpuGaugeField)) {
      LatticeField::resizeBuffer(bytes);

      // copy field and ghost zone into bufferPinned
      copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, bufferPinned, 
		       static_cast<const cpuGaugeField&>(src).gauge); 

      // this copies over both even and odd
      cudaMemcpy(gauge, bufferPinned, bytes, cudaMemcpyHostToDevice);
    } else {
      errorQuda("Invalid gauge field type");
    }

    checkCudaError();
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, const QudaFieldLocation &pack_location)
  {
    if (geometry != QUDA_VECTOR_GEOMETRY) errorQuda("Only vector geometry is supported");

    if (pack_location == QUDA_CUDA_FIELD_LOCATION) {
      errorQuda("Not implemented");
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
    if (geometry != QUDA_VECTOR_GEOMETRY) errorQuda("Only vector geometry is supported");

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
    
      resizeBuffer(bytes);

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

  // Return the L2 norm squared of the gauge field
  double norm2(const cudaGaugeField &a) {
  
    int spin = 0;
    switch (a.Geometry()) {
    case QUDA_SCALAR_GEOMETRY:
      spin = 1;
      break;
    case QUDA_VECTOR_GEOMETRY:
      spin = a.Ndim();
      break;
    case QUDA_TENSOR_GEOMETRY:
      spin = a.Ndim() * (a.Ndim()-1);
      break;
    default:
      errorQuda("Unsupported field geometry %d", a.Geometry());
    }

    if (a.Precision() == QUDA_HALF_PRECISION) 
      errorQuda("Casting a cudaGaugeField into cudaColorSpinorField not possible in half precision");

    ColorSpinorParam spinor_param;
    spinor_param.nColor = a.Reconstruct()/2;
    spinor_param.nSpin = a.Ndim();
    spinor_param.nDim = spin;
    for (int d=0; d<a.Ndim(); d++) spinor_param.x[d] = a.X()[d];
    spinor_param.precision = a.Precision();
    spinor_param.pad = a.Pad();
    spinor_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    spinor_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    spinor_param.fieldOrder = (QudaFieldOrder)a.FieldOrder();
    spinor_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    spinor_param.create = QUDA_REFERENCE_FIELD_CREATE;
    spinor_param.v = (void*)a.Gauge_p();
    cudaColorSpinorField b(spinor_param);
    return norm2(b);
  }

} // namespace quda

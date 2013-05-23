#include <string.h>
#include <gauge_field.h>
#include <face_quda.h>
#include <typeinfo>
#include <misc_helpers.h>
#include <blas_quda.h>

#include <pack_gauge.h>

namespace quda {

  cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) :
    GaugeField(param), gauge(0), even(0), odd(0), backed_up(false)
  {
    if(create != QUDA_NULL_FIELD_CREATE &&  create != QUDA_ZERO_FIELD_CREATE){
      errorQuda("ERROR: create type(%d) not supported yet\n", create);
    }
  
    gauge = device_malloc(bytes);  
    if(create == QUDA_ZERO_FIELD_CREATE) cudaMemset(gauge, 0, bytes);
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

    if (gauge) device_free(gauge);
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, const QudaFieldLocation &pack_location)
  {
    if (geometry != QUDA_VECTOR_GEOMETRY) errorQuda("Only vector geometry is supported");

    checkField(cpu);

    if (pack_location == QUDA_CUDA_FIELD_LOCATION) {
      errorQuda("Not implemented"); // awaiting Guochun's new gauge packing
    } else if (pack_location == QUDA_CPU_FIELD_LOCATION) {

      if (precision == QUDA_HALF_PRECISION && link_type == QUDA_ASQTAD_FAT_LINKS) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  fat_link_max = maxGauge<double>(cpu);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  fat_link_max = maxGauge<float>(cpu);
	}
      }

      LatticeField::resizeBuffer(bytes);

      if (precision == QUDA_DOUBLE_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  packGauge((double*)LatticeField::bufferPinned, (double*)cpu.gauge, *this, cpu, 0);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((double*)LatticeField::bufferPinned, (float*)cpu.gauge, *this, cpu, 0);
	}
      } else if (precision == QUDA_SINGLE_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  packGauge((float*)LatticeField::bufferPinned, (double*)cpu.gauge, *this, cpu, 0);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((float*)LatticeField::bufferPinned, (float*)cpu.gauge, *this, cpu, 0);
	}
      } else if (precision == QUDA_HALF_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION){
	  packGauge((short*)LatticeField::bufferPinned, (double*)cpu.gauge, *this, cpu, 0);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((short*)LatticeField::bufferPinned, (float*)cpu.gauge, *this, cpu, 0);
	}
      } 

#ifdef MULTI_GPU
      //FIXME: if this is MOM field, we don't need exchange data
      if(link_type != QUDA_ASQTAD_MOM_LINKS) cpu.exchangeGhost();
      if (precision == QUDA_DOUBLE_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  packGauge((double*)LatticeField::bufferPinned, (double*)cpu.gauge, *this, cpu, 1);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((double*)LatticeField::bufferPinned, (float*)cpu.gauge, *this, cpu, 1);
	}
      } else if (precision == QUDA_SINGLE_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  packGauge((float*)LatticeField::bufferPinned, (double*)cpu.gauge, *this, cpu, 1);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((float*)LatticeField::bufferPinned, (float*)cpu.gauge, *this, cpu, 1);
	}
      } else if (precision == QUDA_HALF_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION){
	  packGauge((short*)LatticeField::bufferPinned, (double*)cpu.gauge, *this, cpu, 1);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((short*)LatticeField::bufferPinned, (float*)cpu.gauge, *this, cpu, 1);
	}
      } 
#endif

      // this copies over both even and odd
      cudaMemcpy(gauge, LatticeField::bufferPinned, bytes, cudaMemcpyHostToDevice);
      checkCudaError();
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

      if (precision == QUDA_DOUBLE_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  packGauge((double*)cpu.gauge, (double*)bufferPinned, cpu, *this, 0);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((float*)cpu.gauge, (double*)bufferPinned, cpu, *this, 0);
	}
      } else if (precision == QUDA_SINGLE_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  packGauge((double*)cpu.gauge, (float*)bufferPinned, cpu, *this, 0);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((float*)cpu.gauge, (float*)bufferPinned, cpu, *this, 0);
	}
      } else if (precision == QUDA_HALF_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION){
	  packGauge((double*)cpu.gauge, (short*)bufferPinned, cpu, *this, 0);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  packGauge((float*)cpu.gauge, (short*)bufferPinned, cpu, *this, 0);
	}
      }
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
    if (!backed_up) errorQuda("Cannot retore since not backed up");
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

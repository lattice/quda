#include <gauge_field.h>
#include <face_quda.h>
#include <typeinfo>
#include <misc_helpers.h>
#include <blas_quda.h>

cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) :
  GaugeField(param), gauge(0), even(0), odd(0), 
  backed_up(false)
{
  if(create != QUDA_NULL_FIELD_CREATE &&  create != QUDA_ZERO_FIELD_CREATE){
    errorQuda("ERROR: create type(%d) not supported yet\n", create);
  }
  
  if (cudaMalloc((void **)&gauge, bytes) == cudaErrorMemoryAllocation) {
    errorQuda("Error allocating gauge field");
  }
  
  if(create == QUDA_ZERO_FIELD_CREATE){
    cudaMemset(gauge, 0, bytes);
  }
  
  even = gauge;
  odd = (char*)gauge + bytes/2;
 
}

cudaGaugeField::~cudaGaugeField() {
  if (gauge) cudaFree(gauge);
  checkCudaError();
}

// FIXME temporary hack 
static double anisotropy_;
static double fat_link_max_;
static const int *X_;
static QudaTboundary t_boundary_; 
#include <pack_gauge.h>

template <typename Float, typename FloatN>
static void loadGaugeField(FloatN *even, FloatN *odd, Float *cpuGauge, Float **cpuGhost,
			   GaugeFieldOrder cpu_order, QudaReconstructType reconstruct, 
			   size_t bytes, int volumeCB, int *surfaceCB, int pad, int nFace,
			   QudaLinkType type, double &fat_link_max) {
  // Use pinned memory
  FloatN *packed, *packedEven, *packedOdd;
  if (cudaMallocHost(&packed, bytes) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for packed\n");
  }
  packedEven = packed;
  packedOdd = (FloatN*)((char*)packed + bytes/2);

  if( ! packedEven ) errorQuda( "packedEven is borked\n");
  if( ! packedOdd ) errorQuda( "packedOdd is borked\n");
  if( ! even ) errorQuda( "even is borked\n");
  if( ! odd ) errorQuda( "odd is borked\n");
  if( ! cpuGauge ) errorQuda( "cpuGauge is borked\n");

#ifdef MULTI_GPU
  if (cpu_order != QUDA_QDP_GAUGE_ORDER)
    errorQuda("Only QUDA_QDP_GAUGE_ORDER is supported for multi-gpu\n");
#endif

  //for QUDA_ASQTAD_FAT_LINKS, need to find out the max value
  //fat_link_max will be used in encoding half precision fat link
  if(type == QUDA_ASQTAD_FAT_LINKS){
    fat_link_max = 0.0;
    for(int dir=0; dir < 4; dir++){
      for(int i=0;i < 2*volumeCB*gaugeSiteSize; i++){
	Float** tmp = (Float**)cpuGauge;
	if( tmp[dir][i] > fat_link_max ){
	  fat_link_max = tmp[dir][i];
	}
      }
    }
  }
  
  double fat_link_max_double = fat_link_max;
#ifdef MULTI_GPU
  reduceMaxDouble(fat_link_max_double);
#endif
  fat_link_max = fat_link_max_double;

  int voxels[] = {volumeCB, volumeCB, volumeCB, volumeCB};

  // FIXME - hack for the moment
  fat_link_max_ = fat_link_max;

  int nFaceLocal = 1;
  if (cpu_order == QUDA_QDP_GAUGE_ORDER) {
    packQDPGaugeField(packedEven, (Float**)cpuGauge, 0, reconstruct, volumeCB, 
		      voxels, pad, 0, nFaceLocal, type);
    packQDPGaugeField(packedOdd,  (Float**)cpuGauge, 1, reconstruct, volumeCB, 
		      voxels, pad, 0, nFaceLocal, type);
  } else if (cpu_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    packCPSGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, volumeCB, pad);
    packCPSGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, volumeCB, pad);    
  } else if (cpu_order == QUDA_MILC_GAUGE_ORDER) {
    packMILCGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, volumeCB, pad);
    packMILCGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, volumeCB, pad);    
  } else {
    errorQuda("Invalid gauge_order %d", cpu_order);
  }

#ifdef MULTI_GPU
  if(type != QUDA_ASQTAD_MOM_LINKS){
    // pack into the padded regions
    packQDPGaugeField(packedEven, (Float**)cpuGhost, 0, reconstruct, volumeCB,
		      surfaceCB, pad, volumeCB, nFace, type);
    packQDPGaugeField(packedOdd,  (Float**)cpuGhost, 1, reconstruct, volumeCB, 
		      surfaceCB, pad, volumeCB, nFace, type);
  }
#endif

  cudaMemcpy(even, packed, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
    
  cudaFreeHost(packed);
}

template <typename Float, typename Float2>
void loadMomField(Float2 *even, Float2 *odd, Float *mom, int bytes, int Vh, int pad) 
{  
  Float2 *packedEven, *packedOdd;
  if(cudaMallocHost(&packedEven, bytes/2) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for packedEven\n");
  }
  if (cudaMallocHost(&packedOdd, bytes/2) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for packedOdd\n");
  }
    
  packMomField(packedEven, (Float*)mom, 0, Vh, pad);
  packMomField(packedOdd,  (Float*)mom, 1, Vh, pad);
    
  cudaMemcpy(even, packedEven, bytes/2, cudaMemcpyHostToDevice);
  cudaMemcpy(odd,  packedOdd, bytes/2, cudaMemcpyHostToDevice); 
  
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
}

void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, const QudaFieldLocation &pack_location)
{
  if (geometry != QUDA_VECTOR_GEOMETRY) errorQuda("Only vector geometry is supported");

  checkField(cpu);

  if (pack_location == QUDA_CUDA_FIELD_LOCATION) {
    errorQuda("Not implemented"); // awaiting Guochun's new gauge packing
  } else if (pack_location == QUDA_CPU_FIELD_LOCATION) {
    // FIXME
    anisotropy_ = anisotropy;
    X_ = x;
    t_boundary_ = t_boundary;
    
#ifdef MULTI_GPU
    //FIXME: if this is MOM field, we don't need exchange data
    if(link_type != QUDA_ASQTAD_MOM_LINKS){ 
      cpu.exchangeGhost();
    }
#endif
    
    if (reconstruct != QUDA_RECONSTRUCT_10) { // gauge field
      if (precision == QUDA_DOUBLE_PRECISION) {
	
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  loadGaugeField((double2*)(even), (double2*)(odd), (double*)cpu.gauge, (double**)cpu.ghost,
			 cpu.Order(), reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  loadGaugeField((double2*)(even), (double2*)(odd), (float*)cpu.gauge, (float**)cpu.ghost,
			 cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
	}
	
      } else if (precision == QUDA_SINGLE_PRECISION) {
	
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    loadGaugeField((float2*)(even), (float2*)(odd), (double*)cpu.gauge, (double**)cpu.ghost, 
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);	      
	  } else {
	    loadGaugeField((float4*)(even), (float4*)(odd), (double*)cpu.gauge, (double**)cpu.ghost,
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
	  }
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    loadGaugeField((float2*)(even), (float2*)(odd), (float*)cpu.gauge, (float**)cpu.ghost,
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
	  } else {
	    loadGaugeField((float4*)(even), (float4*)(odd), (float*)cpu.gauge, (float**)cpu.ghost,
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
	  }
	}
	
      } else if (precision == QUDA_HALF_PRECISION) {
	
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION){
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    loadGaugeField((short2*)(even), (short2*)(odd), (double*)cpu.gauge, (double**)cpu.ghost,
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
	  } else {
	    loadGaugeField((short4*)(even), (short4*)(odd), (double*)cpu.gauge, (double**)cpu.ghost,
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);	      
	  }
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    loadGaugeField((short2*)(even), (short2*)(odd), (float*)cpu.gauge, (float**)cpu.ghost,
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
	  } else {
	    loadGaugeField((short4*)(even), (short4*)(odd), (float*)cpu.gauge, (float**)(cpu.ghost),
			   cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);	      
	  }
	}
      }
    } else { // momentum field
      if  (precision == QUDA_DOUBLE_PRECISION) {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  loadMomField((double2*)(even), (double2*)(odd), (double*)cpu.gauge, bytes, volumeCB, pad);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  loadMomField((double2*)(even), (double2*)(odd), (float*)cpu.gauge, bytes, volumeCB, pad);
	} 
      } else {
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  loadMomField((float2*)(even), (float2*)(odd), (double*)cpu.gauge, bytes, volumeCB, pad);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  loadMomField((float2*)(even), (float2*)(odd), (float*)cpu.gauge, bytes, volumeCB, pad);
	} 
      }      
    } // gauge or momentum
  } else {
    errorQuda("Invalid pack location %d", pack_location);
  }
    
}
  
/* 
   Generic gauge field retrieval.  Copies the cuda gauge field into a
   cpu gauge field.  The reordering takes place on the host and
   supports most gauge field options.
 */
template <typename Float, typename FloatN>
static void storeGaugeField(Float *cpuGauge, FloatN *gauge, GaugeFieldOrder cpu_order,
			    QudaReconstructType reconstruct, int bytes, int volumeCB, int pad) {

  // Use pinned memory
  FloatN *packed;
  if (cudaMallocHost(&packed, bytes) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for packed\n");
  }
  cudaMemcpy(packed, gauge, bytes, cudaMemcpyDeviceToHost);
    
  FloatN *packedEven = packed;
  FloatN *packedOdd = (FloatN*)((char*)packed + bytes/2);
    
  if (cpu_order == QUDA_QDP_GAUGE_ORDER) {
    unpackQDPGaugeField((Float**)cpuGauge, packedEven, 0, reconstruct, volumeCB, pad);
    unpackQDPGaugeField((Float**)cpuGauge, packedOdd, 1, reconstruct, volumeCB, pad);
  } else if (cpu_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    unpackCPSGaugeField((Float*)cpuGauge, packedEven, 0, reconstruct, volumeCB, pad);
    unpackCPSGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, volumeCB, pad);
  } else if (cpu_order == QUDA_MILC_GAUGE_ORDER) {
    unpackMILCGaugeField((Float*)cpuGauge, packedEven, 0, reconstruct, volumeCB, pad);
    unpackMILCGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, volumeCB, pad);
  } else {
    errorQuda("Invalid gauge_order");
  }
    
  cudaFreeHost(packed);
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

  int datalen = 4*2*volumeCB*gaugeSiteSize*sizeof(Float); // both parities
  void *unpacked;  
  if(cudaMalloc(&unpacked, datalen) != cudaSuccess){
    errorQuda("cudaMalloc() failed for unpacked\n");
  }
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
  
  cudaFree(unpacked);
  for(int i=0; i<2; i++) cudaStreamDestroy(streams[i]);
}

template <typename Float, typename Float2>
void 
storeMomToCPUArray(Float* mom, Float2 *even, Float2 *odd, 
		   int bytes, int V, int pad) 
{    
  Float2 *packedEven, *packedOdd;   
  if (cudaMallocHost(&packedEven, bytes/2) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for packedEven\n");
  }
  if (cudaMallocHost(&packedOdd, bytes/2) == cudaErrorMemoryAllocation) {
    errorQuda("ERROR: cudaMallocHost failed for packedEven\n");
  }
  cudaMemcpy(packedEven, even, bytes/2, cudaMemcpyDeviceToHost); 
  cudaMemcpy(packedOdd, odd, bytes/2, cudaMemcpyDeviceToHost);  
  
  unpackMomField((Float*)mom, packedEven,0, V/2, pad);
  unpackMomField((Float*)mom, packedOdd, 1, V/2, pad);
  
  cudaFreeHost(packedEven); 
  cudaFreeHost(packedOdd); 
}

void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, const QudaFieldLocation &pack_location) const
{
  if (geometry != QUDA_VECTOR_GEOMETRY) errorQuda("Only vector geometry is supported");

  // do device-side reordering then copy
  if (pack_location == QUDA_CUDA_FIELD_LOCATION) {
    // check parameters are suitable for device-side packing
    if (precision != cpu.Precision())
      errorQuda("cpu precision %d and cuda precision %d must be the same", 
		cpu.Precision(), precision );

    if (reconstruct != QUDA_RECONSTRUCT_NO)
      errorQuda("Only no reconstruction supported");

    if (order != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Only QUDA_FLOAT2_GAUGE_ORDER supported");

    if (cpu.Order() != QUDA_MILC_GAUGE_ORDER)
      errorQuda("Only QUDA_MILC_GAUGE_ORDER supported");

    if (precision == QUDA_DOUBLE_PRECISION){
      storeGaugeField((double*)cpu.gauge, (double2*)gauge, bytes, volumeCB, stride, precision);
    } else if (precision == QUDA_SINGLE_PRECISION){
      storeGaugeField((float*)cpu.gauge, (float2*)gauge, bytes, volumeCB, stride, precision);
    } else {
      errorQuda("Half precision not supported");
    }

  } else if (pack_location == QUDA_CPU_FIELD_LOCATION) { // do copy then host-side reorder
    
    // FIXME - nasty globals
    anisotropy_ = anisotropy;
    fat_link_max_ = fat_link_max;
    X_ = x;
    t_boundary_ = t_boundary;
    
    if (reconstruct != QUDA_RECONSTRUCT_10) {
      if (precision == QUDA_DOUBLE_PRECISION) {
	
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  storeGaugeField((double*)cpu.gauge, (double2*)(gauge),
			  cpu.order, reconstruct, bytes, volumeCB, pad);
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  storeGaugeField((float*)cpu.gauge, (double2*)(gauge),
			  cpu.order, reconstruct, bytes, volumeCB, pad);
	}
	
      } else if (precision == QUDA_SINGLE_PRECISION) {
	
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    storeGaugeField((double*)cpu.gauge, (float2*)(gauge),
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  } else {
	    storeGaugeField((double*)cpu.gauge, (float4*)(gauge),
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  }
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    storeGaugeField((float*)cpu.gauge, (float2*)(gauge),
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  } else {
	    storeGaugeField((float*)cpu.gauge, (float4*)(gauge),
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  }
	}
	
      } else if (precision == QUDA_HALF_PRECISION) {
	
	if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    storeGaugeField((double*)cpu.gauge, (short2*)(gauge),
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  } else {
	    storeGaugeField((double*)cpu.gauge, (short4*)(gauge),
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  }
	} else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
	  if (reconstruct == QUDA_RECONSTRUCT_NO) {
	    storeGaugeField((float*)cpu.gauge, (short2*)(gauge),
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  } else {
	    storeGaugeField((float*)cpu.gauge, (short4*)(gauge), 
			    cpu.order, reconstruct, bytes, volumeCB, pad);
	  }
	}
      }
    } else {

      if (cpu.Precision() != precision)
	errorQuda("cpu and gpu precison has to be the same at this moment");
    
      if (precision == QUDA_HALF_PRECISION)
	errorQuda("half precision is not supported at this moment");
    
      if (cpu.order != QUDA_MILC_GAUGE_ORDER)
	errorQuda("Only MILC gauge order supported in momentum unpack, not %d", cpu.order);

      if (precision == QUDA_DOUBLE_PRECISION) {
	storeMomToCPUArray( (double*)cpu.gauge, (double2*)even, (double2*)odd, bytes, volume, pad);	
      }else { //SINGLE PRECISIONS
	storeMomToCPUArray( (float*)cpu.gauge, (float2*)even, (float2*)odd, bytes, volume, pad);	
      }
    } // reconstruct 10
  } else {
    errorQuda("Invalid pack location %d", pack_location);
  }

}

void cudaGaugeField::backup() const {
  if (backed_up) errorQuda("Gauge field already backedup");
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

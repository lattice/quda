#include <gauge_field.h>
#include <face_quda.h>
#include <typeinfo>

cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) :
  GaugeField(param, QUDA_CUDA_FIELD_LOCATION), gauge(0), even(0), odd(0)
{
  if (cudaMalloc((void **)&gauge, bytes) == cudaErrorMemoryAllocation) {
    errorQuda("Error allocating gauge field");
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
static void loadGaugeField(FloatN *even, FloatN *odd, Float *cpuGauge, Float *cpuGhost,
			   GaugeFieldOrder cpu_order, QudaReconstructType reconstruct, 
			   size_t bytes, int volumeCB, int *surfaceCB, int pad, int nFace,
			   QudaLinkType type, double &fat_link_max) {
  // Use pinned memory
  FloatN *packed, *packedEven, *packedOdd;
  cudaMallocHost((void**)&packed, bytes);
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
  } else {
    errorQuda("Invalid gauge_order %d", cpu_order);
  }

#ifdef MULTI_GPU
  // pack into the padded regions
  packQDPGaugeField(packedEven, (Float**)cpuGhost, 0, reconstruct, volumeCB,
		    surfaceCB, pad, volumeCB, nFace, type);
  packQDPGaugeField(packedOdd,  (Float**)cpuGhost, 1, reconstruct, volumeCB, 
  		    surfaceCB, pad, volumeCB, nFace, type);
#endif

  cudaMemcpy(even, packed, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
    
  cudaFreeHost(packed);
}

void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu) {

  checkField(cpu);

  // FIXME
  anisotropy_ = anisotropy;
  X_ = x;
  t_boundary_ = t_boundary;

#ifdef MULTI_GPU
  cpu.exchangeGhost();
#endif

  if (precision == QUDA_DOUBLE_PRECISION) {

    if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
      loadGaugeField((double2*)(even), (double2*)(odd), (double*)cpu.gauge, (double*)cpu.ghost,
		     cpu.Order(), reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
    } else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
      loadGaugeField((double2*)(even), (double2*)(odd), (float*)cpu.gauge, (float*)cpu.ghost,
		     cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
    }

  } else if (precision == QUDA_SINGLE_PRECISION) {

    if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(even), (float2*)(odd), (double*)cpu.gauge, (double*)cpu.ghost, 
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);	      
      } else {
	loadGaugeField((float4*)(even), (float4*)(odd), (double*)cpu.gauge, (double*)cpu.ghost,
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
      }
    } else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(even), (float2*)(odd), (float*)cpu.gauge, (float*)cpu.ghost,
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
      } else {
	loadGaugeField((float4*)(even), (float4*)(odd), (float*)cpu.gauge, (float*)cpu.ghost,
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
      }
    }

  } else if (precision == QUDA_HALF_PRECISION) {

    if (cpu.Precision() == QUDA_DOUBLE_PRECISION){
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(even), (short2*)(odd), (double*)cpu.gauge, (double*)cpu.ghost,
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
      } else {
	loadGaugeField((short4*)(even), (short4*)(odd), (double*)cpu.gauge, (double*)cpu.ghost,
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);	      
      }
    } else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(even), (short2*)(odd), (float*)cpu.gauge, (float*)cpu.ghost,
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);
      } else {
	loadGaugeField((short4*)(even), (short4*)(odd), (float*)cpu.gauge, (float*)cpu.ghost,
		       cpu.order, reconstruct, bytes, volumeCB, surfaceCB, pad, nFace, link_type, fat_link_max);	      
      }
    }

  }

}

template <typename Float, typename FloatN>
static void retrieveGaugeField(Float *cpuGauge, FloatN *gauge, GaugeFieldOrder cpu_order,
			       QudaReconstructType reconstruct, int bytes, int volumeCB, int pad) {

  // Use pinned memory
  FloatN *packed;
  FloatN *packedEven = packed;
  FloatN *packedOdd = (FloatN*)((char*)packed + bytes/2);
    
  cudaMallocHost((void**)&packed, bytes);
    
  cudaMemcpy(packed, gauge, bytes, cudaMemcpyDeviceToHost);
    
  if (cpu_order == QUDA_QDP_GAUGE_ORDER) {
    unpackQDPGaugeField((Float**)cpuGauge, packedEven, 0, reconstruct, volumeCB, pad);
    unpackQDPGaugeField((Float**)cpuGauge, packedOdd, 1, reconstruct, volumeCB, pad);
  } else if (cpu_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    unpackCPSGaugeField((Float*)cpuGauge, packedEven, 0, reconstruct, volumeCB, pad);
    unpackCPSGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, volumeCB, pad);
  } else {
    errorQuda("Invalid gauge_order");
  }
    
  cudaFreeHost(packed);
}

void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const
{

  anisotropy_ = anisotropy;
  fat_link_max_ = fat_link_max;
  X_ = x;
  t_boundary_ = t_boundary;

  if (precision == QUDA_DOUBLE_PRECISION) {

    if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
      retrieveGaugeField((double*)cpu.gauge, (double2*)(gauge),
			 cpu.order, reconstruct, bytes, volumeCB, pad);
    } else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
      retrieveGaugeField((float*)cpu.gauge, (double2*)(gauge),
			 cpu.order, reconstruct, bytes, volumeCB, pad);
    }

  } else if (precision == QUDA_SINGLE_PRECISION) {

    if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpu.gauge, (float2*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      } else {
	retrieveGaugeField((double*)cpu.gauge, (float4*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      }
    } else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpu.gauge, (float2*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      } else {
	retrieveGaugeField((float*)cpu.gauge, (float4*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      }
    }

  } else if (precision == QUDA_HALF_PRECISION) {

    if (cpu.Precision() == QUDA_DOUBLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpu.gauge, (short2*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      } else {
	retrieveGaugeField((double*)cpu.gauge, (short4*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      }
    } else if (cpu.Precision() == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpu.gauge, (short2*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      } else {
	retrieveGaugeField((float*)cpu.gauge, (short4*)(gauge),
			   cpu.order, reconstruct, bytes, volumeCB, pad);
      }
    }

  }

}


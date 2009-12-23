#include <stdlib.h>
#include <stdio.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

void zeroField(cudaColorSpinorField &a);
void copyField(cudaColorSpinorField &a, const cudaColorSpinorField &b);

bool cudaColorSpinorField::bufferInit = false;

cudaColorSpinorField::cudaColorSpinorField() : 
  ColorSpinorField(), init(false) {

}

cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
  ColorSpinorField(param), init(false) {
  create();
  if (param.create == CREATE_ZERO) {
    zeroField(*this);
  } else {
    errorQuda("not implemented");
  }

}

cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create();
  copyField(*this, src);
}

cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) 
  : ColorSpinorField(src), init(false) {
  create();
  type = CUDA_FIELD;
  if (src.fieldType() == CUDA_FIELD) {
    copyField(*this, dynamic_cast<const cudaColorSpinorField&>(src));
  } else if (src.fieldType() == CPU_FIELD) {
    loadCPUSpinorField(src);
  }
}

cudaColorSpinorField::~cudaColorSpinorField() {
  destroy();
}

void cudaColorSpinorField::create() {
  if (basis != UKQCD_BASIS) {
    errorQuda("Basis not implemented");
  }

  if (subset == FULL_FIELD_SUBSET && subset_order != EVEN_ODD_SUBSET_ORDER) {
    errorQuda("Subset not implemented");
  }
									     

  if (cudaMalloc((void**)&v, bytes) == cudaErrorMemoryAllocation) {
    errorQuda("Error allocating spinor");
  }

  if (prec == QUDA_HALF_PRECISION) {
    if (cudaMalloc((void**)&norm, bytes/12) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating norm");
    }
  }

  // Check if buffer isn't big enough
  if (bytes > bufferBytes && bufferInit) {
    cudaFree(buffer);
    bufferInit = false;
  }

  if (!bufferInit) {
    bufferBytes = bytes;
    cudaMallocHost(&buffer, bufferBytes);    
    bufferInit = true;
  }

  init = true;
}

void cudaColorSpinorField::destroy() {
  if (init) {
    cudaFree(v);
    if (prec == QUDA_HALF_PRECISION) cudaFree(norm);
  }
  if (bufferInit) cudaFree(buffer);
}

/*
  On my mark, unleash template hell

  Here we are templating on the following
  - input precision
  - output precision
  - number of colors
  - number of spins
  - short vector lengh (float, float2, float4 etc.)
  
*/

// SPACE_SPIN_COLOR -> FLOAT1, FLOAT2 or FLOAT4 (no "internal re-ordering")
template <int Nc, int Ns, int N, typename Float, typename FloatN>
inline void packSpinorField(FloatN* a, Float *b, int V) {
  for (int sc=0; sc<(Ns*Nc*2)/N; sc++) {
    for (int zc=0; zc<N; zc++) {
      (a+N*V*sc)[zc] = b[sc*4+zc];
    }
  }
}

// SPACE_COLOR_SPIN -> FLOATN (maintain Dirac basis)
template <int Nc, int Ns, int N, typename Float, typename FloatN>
inline void packQLASpinorField(FloatN* a, Float *b, int V) {
  for (int s=0; s<Ns; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	(a+(((s*Nc+c)*2+z)/N)*V*N) [((s*Nc+c)*2+z) % N] = b[(c*Ns+s)*2+z];
      }
    }
  }

}

// SPACE_SPIN_COLOR -> FLOATN (Chiral -> non-relativistic)
template <int Nc, int N, typename Float, typename FloatN>
inline void packNonRelSpinorField(FloatN* a, Float *b, int V) {
  int s1[4] = {1, 2, 3, 0};
  int s2[4] = {3, 0, 1, 2};
  Float K1[4] = {0.5, -0.5, -0.5, -0.5};
  Float K2[4] = {0.5, -0.5, 0.5, 0.5};

  for (int s=0; s<4; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	(a+(((s*Nc+c)*2+z)/N)*V*N )[((s*Nc+c)*2+z) % N] =
	  K1[s]*b[(s1[s]*Nc+c)*2+z] + K2[s]*b[(s2[s]*Nc+c)*2+z];
      }
    }
  }

}

// SPACE_COLOR_SPIN -> FLOATN (Chiral -> non-relativistic)
template <int Nc, int N, typename Float, typename FloatN>
inline void packNonRelQLASpinorField(FloatN* a, Float *b, int V) {
  int s1[4] = {1, 2, 3, 0};
  int s2[4] = {3, 0, 1, 2};
  Float K1[4] = {0.5, -0.5, -0.5, -0.5};
  Float K2[4] = {0.5, -0.5, 0.5, 0.5};

  for (int s=0; s<4; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	(a+(((s*Nc+c)*2+z)/N)*V*N )[((s*Nc+c)*2+z) % N] =
	  K1[s]*b[(c*4+s1[s])*2+z] + K2[s]*b[(c*4+s2[s])*2+z];
      }
    }
  }

}

// Standard spinor packing, colour inside spin
template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packParitySpinor(FloatN *dest, Float *src, int Vh, int pad, 
		      GammaBasis destBasis, GammaBasis srcBasis) {
  if (destBasis==srcBasis) {
    for (int i = 0; i < Vh; i++) {
      packSpinorField<Nc, Ns, N>(dest+N*i, src+24*i, Vh+pad);
    }
  } else if (destBasis == UKQCD_BASIS && srcBasis == DEGRAND_ROSSI_BASIS) {
    for (int i = 0; i < Vh; i++) {
      packNonRelSpinorField<Nc, N>(dest+N*i, src+24*i, Vh+pad);
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

// QLA spinor packing, spin inside colour
template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packQLAParitySpinor(FloatN *dest, Float *src, int Vh, int pad,
			 GammaBasis destBasis, GammaBasis srcBasis) {
  if (destBasis==srcBasis) {
    for (int i = 0; i < Vh; i++) {
      packQLASpinorField<Nc, Ns, N>(dest+N*i, src+i*24, Vh+pad);
    }
  } else if (destBasis == UKQCD_BASIS && srcBasis == DEGRAND_ROSSI_BASIS) {
    for (int i = 0; i < Vh; i++) {
      packNonRelQLASpinorField<Nc, N>(dest+N*i, src+i*24, Vh+pad);
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packFullSpinor(FloatN *dest, Float *src, int V, int pad, int *x, int length,
		    GammaBasis destBasis, GammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packSpinorField<Nc,Ns,N>(dest+N*i, src+24*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packSpinorField<Nc,Ns,N>(dest+length/2+N*i, src+24*k, Vh+pad);
      }
    }
  } else if (destBasis == UKQCD_BASIS && srcBasis == DEGRAND_ROSSI_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packNonRelSpinorField<Nc,N>(dest+N*i, src+24*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packNonRelSpinorField<Nc,N>(dest+length/2+N*i, src+24*k, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packQLAFullSpinor(FloatN *dest, Float *src, int V, int pad, int *x, int length,
		    GammaBasis destBasis, GammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packQLASpinorField<Nc,Ns,N>(dest+N*i, src+24*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packQLASpinorField<Nc,Ns,N>(dest+length/2+N*i, src+24*k, Vh+pad);
      }
    }
  } else if (destBasis == UKQCD_BASIS && srcBasis == DEGRAND_ROSSI_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packNonRelQLASpinorField<Nc,N>(dest+N*i, src+24*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packNonRelQLASpinorField<Nc,N>(dest+length/2+N*i, src+24*k, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packSpinor(FloatN *dest, Float *src, int V, int pad, int *x, int length,
		FieldSubset srcSubset, SubsetOrder subsetOrder, 
		GammaBasis destBasis, GammaBasis srcBasis,
		FieldOrder srcOrder) {

  if (srcSubset == FULL_FIELD_SUBSET) {
    if (subsetOrder == LEXICOGRAPHIC_SUBSET_ORDER) {
      // We are copying from a full spinor field that is not parity ordered
      if (srcOrder == SPACE_SPIN_COLOR_ORDER) {
	packFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, length, destBasis, srcBasis);
      } else if (srcOrder == SPACE_COLOR_SPIN_ORDER) {
	packQLAFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, length, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
      }
    } else {
      // We are copying a parity ordered field
      
      // check what src parity ordering is
      uint evenOff, oddOff;
      if (subsetOrder == EVEN_ODD_SUBSET_ORDER) {
	evenOff = 0;
	oddOff = length/2;
      } else {
	oddOff = 0;
	evenOff = length/2;
      }

      if (srcOrder == SPACE_SPIN_COLOR_ORDER) {
	packParitySpinor<Nc,Ns,N>(dest, src+evenOff, V/2, pad, destBasis, srcBasis);
	packParitySpinor<Nc,Ns,N>(dest + length/(2*N), src+oddOff, V/2, pad, destBasis, srcBasis);
      } else if (srcOrder == SPACE_COLOR_SPIN_ORDER) {
	packQLAParitySpinor<Nc,Ns,N>(dest, src+evenOff, V/2, pad, destBasis, srcBasis);
	packQLAParitySpinor<Nc,Ns,N>(dest + length/(2*N), src+oddOff, V/2, pad, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
      }
    }
  } else {
    // src is defined on a single parity only
    if (srcOrder == SPACE_SPIN_COLOR_ORDER) {
      packParitySpinor<Nc,Ns,N>(dest, src, V, pad, destBasis, srcBasis);
    } else if (srcOrder == SPACE_COLOR_SPIN_ORDER) {
      packQLAParitySpinor<Nc,Ns,N>(dest, src, V, pad, destBasis, srcBasis);
    } else {
      errorQuda("Source field order not supported");
    }
  }

}

void cudaColorSpinorField::loadCPUSpinorField(const cpuColorSpinorField &src) {

  if (volume != src.volume) {
    errorQuda("Volumes don't match");
  }

  if (nColor != 3) {
    errorQuda("Nc != 3 not yet supported");
  }

  if (nSpin != 4) {
    errorQuda("Ns != 4 not yet supported");
  }

  if (prec == QUDA_HALF_PRECISION) {
    cudaColorSpinorField tmp(src);
    copyField(*this, tmp);
    return;
  }
    
  if (prec == QUDA_DOUBLE_PRECISION) {
    if (src.precision() == QUDA_DOUBLE_PRECISION) {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((double*)buffer, (double*)src.v, volume, pad, x, length, 
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((double*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    } else {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((double*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((double*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    }
  } else {
    if (src.precision() == QUDA_DOUBLE_PRECISION) {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	packSpinor<3,4,4>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      }
    } else {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	packSpinor<3,4,4>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      }
    }
  }
    
  cudaMemcpy(v, buffer, bytes, cudaMemcpyHostToDevice);
  return;
}


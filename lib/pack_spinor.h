/*
  MAC:

  On my mark, unleash template hell

  Here we are templating on the following
  - input precision
  - output precision
  - number of colors
  - number of spins
  - short vector lengh (float, float2, float4 etc.)
  
  This is still quite a mess.  Options to reduce to the amount of code
  bloat here include:

  1. Using functors to define arbitrary ordering
  2. Use class inheritance to the same effect
  3. Abuse the C preprocessor to define arbitrary mappings
  4. Something else

  I probably prefer option 2., the problem with 1.) and and 2.) as far
  as I can see it is that the fields are defined using void*, and cast
  at runtime as appropriate.  This doesn't mesh well with defining
  return types based on inheritance which is decided at compile time.
  I could hack it returing void* and casting as appropriate (see
  functor examples in color_spinor_field.h, but this is going to be
  S..L..O..W.

  While I think about this, I shall leave in the verbose state it is
  in, and come back to this in the future.

*/


/*

  Packing routines

*/

#define PRESERVE_SPINOR_NORM 1

#ifdef PRESERVE_SPINOR_NORM // Preserve the norm regardless of basis
double kP = (1.0/sqrt(2.0));
double kU = (1.0/sqrt(2.0));
#else // More numerically accurate not to preserve the norm between basis
double kP = 0.5;
double kU = 1.0;
#endif

// SPACE_SPIN_COLOR -> FLOAT1, FLOAT2 or FLOAT4 (no "internal re-ordering")
template <int Nc, int Ns, int N, typename Float, typename FloatN>
inline void packSpinorField(FloatN* a, Float *b, int V) {
  for (int sc=0; sc<(Ns*Nc*2)/N; sc++) {
    for (int zc=0; zc<N; zc++) {
      (a+N*V*sc)[zc] = b[sc*N+zc];
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
  Float K1[4] = {kP, -kP, -kP, -kP};
  Float K2[4] = {kP, -kP, kP, kP};

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
  Float K1[4] = {kP, -kP, -kP, -kP};
  Float K2[4] = {kP, -kP, kP, kP};

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
      packSpinorField<Nc, Ns, N>(dest+N*i, src+2*Nc*Ns*i, Vh+pad);
    }
  } else if (destBasis == QUDA_UKQCD_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i = 0; i < Vh; i++) {
      packNonRelSpinorField<Nc, N>(dest+N*i, src+2*Nc*Ns*i, Vh+pad);
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
      packQLASpinorField<Nc, Ns, N>(dest+N*i, src+2*Nc*Ns*i, Vh+pad);
    }
  } else if (destBasis == QUDA_UKQCD_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i = 0; i < Vh; i++) {
      packNonRelQLASpinorField<Nc, N>(dest+N*i, src+2*Nc*Ns*i, Vh+pad);
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packFullSpinor(FloatN *dest, Float *src, int V, int pad, const int x[], int destLength,
		    GammaBasis destBasis, GammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packSpinorField<Nc,Ns,N>(dest+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packSpinorField<Nc,Ns,N>(dest+destLength/2+N*i, src+ 2*Nc*Ns*k, Vh+pad);
      }
    }
  } else if (destBasis == QUDA_UKQCD_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packNonRelSpinorField<Nc,N>(dest+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packNonRelSpinorField<Nc,N>(dest+destLength/2+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packQLAFullSpinor(FloatN *dest, Float *src, int V, int pad, const int x[], 
		       int destLength, GammaBasis destBasis, GammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packQLASpinorField<Nc,Ns,N>(dest+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packQLASpinorField<Nc,Ns,N>(dest+destLength/2+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
    }
  } else if (destBasis == QUDA_UKQCD_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packNonRelQLASpinorField<Nc,N>(dest+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packNonRelQLASpinorField<Nc,N>(dest+destLength/2+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}


/*

  Unpacking routines

*/

// SPACE_SPIN_COLOR -> FLOAT1, FLOAT2 or FLOAT4 (no "internal re-ordering")
template <int Nc, int Ns, int N, typename Float, typename FloatN>
inline void unpackSpinorField(Float* a, FloatN *b, int V) {
  for (int sc=0; sc<(Ns*Nc*2)/N; sc++) {
    for (int zc=0; zc<N; zc++) {
      a[sc*N+zc] = (b+N*V*sc)[zc];
    }
  }
}

// SPACE_COLOR_SPIN -> FLOATN (maintain Dirac basis)
template <int Nc, int Ns, int N, typename Float, typename FloatN>
inline void unpackQLASpinorField(Float* a, FloatN *b, int V) {
  for (int s=0; s<Ns; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	a[(c*Ns+s)*2+z] = (b+(((s*Nc+c)*2+z)/N)*V*N) [((s*Nc+c)*2+z) % N];
      }
    }
  }

}

// SPACE_SPIN_COLOR -> FLOATN (Chiral -> non-relativistic)
template <int Nc, int N, typename Float, typename FloatN>
inline void unpackNonRelSpinorField(Float* a, FloatN *b, int V) {
  int s1[4] = {1, 2, 3, 0};
  int s2[4] = {3, 0, 1, 2};
  Float K1[4] = {-kU, kU,  kU,  kU};
  Float K2[4] = {-kU, kU, -kU, -kU};

  for (int s=0; s<4; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	  a[(s*Nc+c)*2+z] =
	    K1[s]*(b+(((s1[s]*Nc+c)*2+z)/N)*V*N )[((s1[s]*Nc+c)*2+z) % N] +
	    K2[s]*(b+(((s2[s]*Nc+c)*2+z)/N)*V*N )[((s2[s]*Nc+c)*2+z) % N];
      }
    }
  }

}

// SPACE_COLOR_SPIN -> FLOATN (Chiral -> non-relativistic)
template <int Nc, int N, typename Float, typename FloatN>
inline void unpackNonRelQLASpinorField(Float* a, FloatN *b, int V) {
  int s1[4] = {1, 2, 3, 0};
  int s2[4] = {3, 0, 1, 2};
  Float K1[4] = {-kU, kU,  kU,  kU};
  Float K2[4] = {-kU, kU, -kU, -kU};

  for (int s=0; s<4; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	  a[(c*4+s)*2+z] =
	    K1[s]*(b+(((s1[s]*Nc+c)*2+z)/N)*V*N )[((s1[s]*Nc+c)*2+z) % N] +
	    K2[s]*(b+(((s2[s]*Nc+c)*2+z)/N)*V*N )[((s2[s]*Nc+c)*2+z) % N];
      }
    }
  }

}

// Standard spinor unpacking, colour inside spin
template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackParitySpinor(Float *dest, FloatN *src, int Vh, int pad, 
		      GammaBasis destBasis, GammaBasis srcBasis) {
  if (destBasis==srcBasis) {
    for (int i = 0; i < Vh; i++) {
      unpackSpinorField<Nc, Ns, N>(dest+2*Nc*Ns*i, src+N*i, Vh+pad);
    }
  } else if (srcBasis == QUDA_UKQCD_BASIS && destBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i = 0; i < Vh; i++) {
      unpackNonRelSpinorField<Nc, N>(dest+2*Nc*Ns*i, src+N*i, Vh+pad);
    }
  } else {
    errorQuda("Basis change from %d to %d not supported", srcBasis, destBasis);
  }
}

// QLA spinor unpacking, spin inside colour
template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackQLAParitySpinor(Float *dest, FloatN *src, int Vh, int pad,
			 GammaBasis destBasis, GammaBasis srcBasis) {
  if (destBasis==srcBasis) {
    for (int i = 0; i < Vh; i++) {
      unpackQLASpinorField<Nc, Ns, N>(dest+2*Nc*Ns*i, src+N*i, Vh+pad);
    }
  } else if (srcBasis == QUDA_UKQCD_BASIS && destBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i = 0; i < Vh; i++) {
      unpackNonRelQLASpinorField<Nc, N>(dest+2*Nc*Ns*i, src+N*i, Vh+pad);
    }
  } else {
    errorQuda("Basis change from %d to %d not supported", srcBasis, destBasis);
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackFullSpinor(Float *dest, FloatN *src, int V, int pad, const int x[], 
		      int srcLength, GammaBasis destBasis, GammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackSpinorField<Nc,Ns,N>(dest+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackSpinorField<Nc,Ns,N>(dest+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else if (srcBasis == QUDA_UKQCD_BASIS && destBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackNonRelSpinorField<Nc,N>(dest+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackNonRelSpinorField<Nc,N>(dest+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackQLAFullSpinor(Float *dest, FloatN *src, int V, int pad, const int x[], 
			 int srcLength, GammaBasis destBasis, GammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackQLASpinorField<Nc,Ns,N>(dest+2*Nc*Ns*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackQLASpinorField<Nc,Ns,N>(dest+2*Nc*Ns*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else if (srcBasis == QUDA_UKQCD_BASIS && destBasis == QUDA_DEGRAND_ROSSI_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackNonRelQLASpinorField<Nc,N>(dest+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackNonRelQLASpinorField<Nc,N>(dest+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packSpinor(FloatN *dest, Float *src, int V, int pad, const int x[], int destLength, 
		int srcLength, FieldSubset srcSubset, SubsetOrder subsetOrder, 
		GammaBasis destBasis, GammaBasis srcBasis, QudaColorSpinorOrder srcOrder) {

  //  printf("%d %d %d %d %d %d %d %d %d %d %d\n", Nc, Ns, N, V, pad, length, srcSubset, subsetOrder, destBasis, srcBasis, srcOrder);

  if (srcSubset == QUDA_FULL_FIELD_SUBSET) {
    if (subsetOrder == QUDA_LEXICOGRAPHIC_SUBSET_ORDER) {
      // We are copying from a full spinor field that is not parity ordered
      if (srcOrder == QUDA_SPACE_SPIN_COLOR_ORDER) {
	  packFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, destLength, destBasis, srcBasis);
      } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_ORDER) {
	packQLAFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, destLength, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
      }
    } else {
      // We are copying a parity ordered field
      
      // check what src parity ordering is
      uint evenOff, oddOff;
      if (subsetOrder == QUDA_EVEN_ODD_SUBSET_ORDER) {
	evenOff = 0;
	oddOff = srcLength/2;
      } else {
	oddOff = 0;
	evenOff = srcLength/2;
      }

      if (srcOrder == QUDA_SPACE_SPIN_COLOR_ORDER) {
	packParitySpinor<Nc,Ns,N>(dest, src+evenOff, V/2, pad, destBasis, srcBasis);
	packParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff, V/2, pad, destBasis, srcBasis);
      } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_ORDER) {
	packQLAParitySpinor<Nc,Ns,N>(dest, src+evenOff, V/2, pad, destBasis, srcBasis);
	packQLAParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff, V/2, pad, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
      }
    }
  } else {
    // src is defined on a single parity only
    if (srcOrder == QUDA_SPACE_SPIN_COLOR_ORDER) {
      packParitySpinor<Nc,Ns,N>(dest, src, V, pad, destBasis, srcBasis);
    } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_ORDER) {
      packQLAParitySpinor<Nc,Ns,N>(dest, src, V, pad, destBasis, srcBasis);
    } else {
      errorQuda("Source field order not supported");
    }
  }

}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackSpinor(Float *dest, FloatN *src, int V, int pad, const int x[], int destLength, 
		  int srcLength, FieldSubset destSubset, SubsetOrder subsetOrder,  
		  GammaBasis destBasis, GammaBasis srcBasis, QudaColorSpinorOrder destOrder) {

  if (destSubset == QUDA_FULL_FIELD_SUBSET) {
    if (subsetOrder == QUDA_LEXICOGRAPHIC_SUBSET_ORDER) {
      // We are copying from a full spinor field that is not parity ordered
      if (destOrder == QUDA_SPACE_SPIN_COLOR_ORDER) {
	unpackFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, srcLength, destBasis, srcBasis);
      } else if (destOrder == QUDA_SPACE_COLOR_SPIN_ORDER) {
	unpackQLAFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, srcLength, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
      }
    } else {
      // We are copying a parity ordered field
      
      // check what src parity ordering is
      uint evenOff, oddOff;
      if (subsetOrder == QUDA_EVEN_ODD_SUBSET_ORDER) {
	evenOff = 0;
	oddOff = srcLength/2;
      } else {
	oddOff = 0;
	evenOff = srcLength/2;
      }

      if (destOrder == QUDA_SPACE_SPIN_COLOR_ORDER) {
	unpackParitySpinor<Nc,Ns,N>(dest,            src+evenOff, V/2, pad, destBasis, srcBasis);
	unpackParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff,  V/2, pad, destBasis, srcBasis);
      } else if (destOrder == QUDA_SPACE_COLOR_SPIN_ORDER) {
	unpackQLAParitySpinor<Nc,Ns,N>(dest,            src+evenOff, V/2, pad, destBasis, srcBasis);
	unpackQLAParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff,  V/2, pad, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
      }
    }
  } else {
    // src is defined on a single parity only
    if (destOrder == QUDA_SPACE_SPIN_COLOR_ORDER) {
      unpackParitySpinor<Nc,Ns,N>(dest, src, V, pad, destBasis, srcBasis);
    } else if (destOrder == QUDA_SPACE_COLOR_SPIN_ORDER) {
      unpackQLAParitySpinor<Nc,Ns,N>(dest, src, V, pad, destBasis, srcBasis);
    } else {
      errorQuda("Source field order not supported");
    }
  }

}


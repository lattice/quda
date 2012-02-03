#if (__COMPUTE_CAPABILITY__ < 130)
// Computes c = a + b in "double single" precision.
__device__ void dsadd(volatile QudaSumFloat &c0, volatile QudaSumFloat &c1, const volatile QudaSumFloat &a0, 
		      const volatile QudaSumFloat &a1, const float b0, const float b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0 + b0;
  QudaSumFloat e = t1 - a0;
  QudaSumFloat t2 = ((b0 - e) + (a0 - (t1 - e))) + a1 + b1;
  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (complex version)
__device__ void zcadd(volatile QudaSumComplex &c0, volatile QudaSumComplex &c1, const volatile QudaSumComplex &a0, 
		      const volatile QudaSumComplex &a1, const volatile QudaSumComplex &b0, const volatile QudaSumComplex &b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (float3 version)
__device__ void dsadd3(volatile QudaSumFloat3 &c0, volatile QudaSumFloat3 &c1, const volatile QudaSumFloat3 &a0, 
		       const volatile QudaSumFloat3 &a1, const volatile QudaSumFloat3 &b0, const volatile QudaSumFloat3 &b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.z + b0.z;
  e = t1 - a0.z;
  t2 = ((b0.z - e) + (a0.z - (t1 - e))) + a1.z + b1.z;
  // The result is t1 + t2, after normalization.
  c0.z = e = t1 + t2;
  c1.z = t2 - (e - t1);
}
#endif

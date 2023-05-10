#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>
#include <atomic_helper.h>
#include <shared_memory_cache_helper.h>

namespace quda {

  /**
   * Retrieve the SU(N) indices for the current block number
   * @param[in] block, current block number, from 0 to (Nc * (Nc - 1) / 2)
   * @param[out] p, row index pointing to the SU(N) matrix 
   * @param[out] q, column index pointing to the SU(N) matrix
  */
  template <int nColor>
  __host__ __device__ inline void IndexBlock(int block, int &p, int &q)
  {
    if ( nColor == 3 ) {
      if ( block == 0 ) { p = 0; q = 1; }
      else if ( block == 1 ) { p = 1; q = 2; }
      else{ p = 0; q = 2; }
    } else {
      int i1;
      int found = 0;
      int del_i = 0;
      int index = -1;
      while ( del_i < (nColor - 1) && found == 0 ) {
        del_i++;
        for ( i1 = 0; i1 < (nColor - del_i); i1++ ) {
          index++;
          if ( index == block ) {
            found = 1;
            break;
          }
        }
      }
      q = i1 + del_i;
      p = i1;
    }
  }

  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation needs 8x more shared memory than the implementation using atomicadd 
   */
  template <typename Float, int gauge_dir, int nColor>
  inline __device__ void GaugeFixHit_AtomicAdd(Matrix<complex<Float>,nColor> &link, const Float relax_boost, int mu)
  {
    auto blockSize = target::block_dim().x;
    auto tid = target::thread_idx().x;

    //Container for the four real parameters of SU(2) subgroup in shared memory
    SharedMemoryCache<Float> cache;
    auto elems = cache.data();

    //initialize shared memory
    if (mu < 4) elems[mu * blockSize + tid] = 0.0;
    cache.sync();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (nColor * (nColor - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<nColor>(block, p, q);
      Float asq = 1.0;
      if (mu < 4) asq = -1.0;
      //FOR COULOMB AND LANDAU!!!!!!!!
      //if(nu0<gauge_dir){
      //In terms of thread index
      if (mu < gauge_dir || (mu >= 4 && mu < (gauge_dir + 4))) {
        //Retrieve the four SU(2) parameters...
        // a0
        atomic_fetch_add(elems + tid, (link(p,p)).x + (link(q,q)).x); //a0
        // a1
        atomic_fetch_add(elems + tid + blockSize, (link(p,q).y + link(q,p).y) * asq); //a1
        // a2
        atomic_fetch_add(elems + tid + blockSize * 2, (link(p,q).x - link(q,p).x) * asq); //a2
        // a3
        atomic_fetch_add(elems + tid + blockSize * 3, (link(p,p).y - link(q,q).y) * asq); //a3
      } //FLOP per lattice site = gauge_dir * 2 * (4 + 7) = gauge_dir * 22

      cache.sync();

      if (mu==0) {
        //Over-relaxation boost
        asq =  elems[tid + blockSize] * elems[tid + blockSize];
        asq += elems[tid + blockSize * 2] * elems[tid + blockSize * 2];
        asq += elems[tid + blockSize * 3] * elems[tid + blockSize * 3];
        Float a0sq = elems[tid] * elems[tid];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = quda::rsqrt((a0sq + x * x * asq));
        elems[tid + blockSize * 0] *= r;
        elems[tid + blockSize * 1] *= x * r;
        elems[tid + blockSize * 2] *= x * r;
        elems[tid + blockSize * 3] *= x * r;
      } //FLOP per lattice site = 22CUB: "Collective" Software Primitives for CUDA Kernel Development

      cache.sync();

      //_____________
      if (mu < 4) {
        complex<Float> m0;
        //Do SU(2) hit on all upward links
        //left multiply an su3_matrix by an su2 matrix
        //link <- u * link
        //#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
          m0 = link(p,j);
          link(p,j) = complex<Float>( elems[tid], elems[tid + blockSize * 3] ) * m0 + complex<Float>( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
          link(q,j) = complex<Float>(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + complex<Float>( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
        }
      }
      else{
        complex<Float> m0;
        //Do SU(2) hit on all downward links
        //right multiply an su3_matrix by an su2 matrix
        //link <- link * u_adj
        //#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
          m0 = link(j,p);
          link(j,p) = complex<Float>( elems[tid], -elems[tid + blockSize * 3] ) * m0 + complex<Float>( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link(j,q);
          link(j,q) = complex<Float>(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + complex<Float>( elems[tid],elems[tid + blockSize * 3] ) * link(j,q);
        }
      }
      //_____________ //FLOP per lattice site = 8 * Nc * 2 * (2*6+2) = Nc * 224
      if (block < (nColor * (nColor - 1) / 2) - 1) {
        cache.sync();
        //reset shared memory SU(2) elements
        if (mu < 4) elems[mu * blockSize + tid] = 0.0;
        cache.sync();
      }
    } //FLOP per lattice site = (block < Nc * ( Nc - 1) / 2) * (22 + 28 gauge_dir + 224 Nc)
     //write updated link to global memory
  }

  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 4 threads per lattice site, the reduction is performed by shared memory using atomicadd.
   */
  template <typename Float, int gauge_dir, int nColor>
  inline __device__ void GaugeFixHit_NoAtomicAdd(Matrix<complex<Float>,nColor> &link, const Float relax_boost, int mu)
  {
    auto blockSize = target::block_dim().x;
    auto tid = target::thread_idx().x;

    //Container for the four real parameters of SU(2) subgroup in shared memory
    SharedMemoryCache<Float> cache;
    auto elems = cache.data();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (nColor * (nColor - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<nColor>(block, p, q);
      //FLOP per lattice site = gauge_dir * 2 * 7 = gauge_dir * 14
      if (mu < gauge_dir) {
        elems[mu * blockSize + tid] = link(p,p).x + link(q,q).x;
        elems[mu * blockSize + tid + blockSize * 8] = -(link(p,q).y + link(q,p).y);
        elems[mu * blockSize + tid + blockSize * 8 * 2] = -(link(p,q).x - link(q,p).x);
        elems[mu * blockSize + tid + blockSize * 8 * 3] = -(link(p,p).y - link(q,q).y);
      }
      if ((mu >= 4 && mu < (gauge_dir + 4))) {
        elems[mu * blockSize + tid] = link(p,p).x + link(q,q).x;
        elems[mu * blockSize + tid + blockSize * 8] = (link(p,q).y + link(q,p).y);
        elems[mu * blockSize + tid + blockSize * 8 * 2] = (link(p,q).x - link(q,p).x);
        elems[mu * blockSize + tid + blockSize * 8 * 3] = (link(p,p).y - link(q,q).y);
      }
      //FLOP per lattice site = gauge_dir * 2 * 7 = gauge_dir * 14
      cache.sync();

      if (mu == 0) {
        Float a0, a1, a2, a3;
        a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0;
      #pragma unroll
        for (int i = 0; i < gauge_dir; i++) {
          a0 += elems[tid + i * blockSize] + elems[tid + (i + 4) * blockSize];
          a1 += elems[tid + i * blockSize + blockSize * 8] + elems[tid + (i + 4) * blockSize + blockSize * 8];
          a2 += elems[tid + i * blockSize + blockSize * 8 * 2] + elems[tid + (i + 4) * blockSize + blockSize * 8 * 2];
          a3 += elems[tid + i * blockSize + blockSize * 8 * 3] + elems[tid + (i + 4) * blockSize + blockSize * 8 * 3];
        }
        //Over-relaxation boost
        Float asq =  a1 * a1 + a2 * a2 + a3 * a3;
        Float a0sq = a0 * a0;
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = quda::rsqrt((a0sq + x * x * asq));
        elems[tid] = a0 * r;
        elems[tid + blockSize] = a1 * x * r;
        elems[tid + blockSize * 2] = a2 * x * r;
        elems[tid + blockSize * 3] = a3 * x * r;
      } //FLOP per lattice site = 22 + 8 * 4
      cache.sync();
      //_____________
      if (mu < 4) {
        complex<Float> m0;
        //Do SU(2) hit on all upward links
        //left multiply an su3_matrix by an su2 matrix
        //link <- u * link
        //#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
          m0 = link(p,j);
          link(p,j) = complex<Float>( elems[tid], elems[tid + blockSize * 3] ) * m0 + complex<Float>( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
          link(q,j) = complex<Float>(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + complex<Float>( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
        }
      } else {
        complex<Float> m0;
        //Do SU(2) hit on all downward links
        //right multiply an su3_matrix by an su2 matrix
        //link <- link * u_adj
        //#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
          m0 = link(j,p);
          link(j,p) = complex<Float>( elems[tid], -elems[tid + blockSize * 3] ) * m0 + complex<Float>( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link(j,q);
          link(j,q) = complex<Float>(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + complex<Float>(elems[tid],elems[tid + blockSize * 3] ) * link(j,q);
        }
      }
      //_____________ //FLOP per lattice site = 8 * Nc * 2 * (2*6+2) = Nc * 224
      if ( block < (nColor * (nColor - 1) / 2) - 1 ) { cache.sync(); }
    } //FLOP per lattice site = (Nc * ( Nc - 1) / 2) * (22 + 28 gauge_dir + 224 Nc)
     //write updated link to global memory
  }

  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
   */
  template <typename Float, int gauge_dir, int nColor>
  inline __device__ void GaugeFixHit_NoAtomicAdd_LessSM(Matrix<complex<Float>,nColor> &link, const Float relax_boost, int mu)
  {
    auto blockSize = target::block_dim().x;
    auto tid = target::thread_idx().x;

    //Container for the four real parameters of SU(2) subgroup in shared memory
    SharedMemoryCache<Float> cache;
    auto elems = cache.data();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (nColor * (nColor - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<nColor>(block, p, q);

      if (mu == 0) {
        elems[tid] = link(p,p).x + link(q,q).x;
        elems[tid + blockSize] = -(link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] = -(link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] = -(link(p,p).y - link(q,q).y);
      }

#pragma unroll
      for (int mu_ = 1; mu_ < gauge_dir; mu_++) {
        cache.sync();
        if (mu_ == mu) {
          elems[tid] += link(p,p).x + link(q,q).x;
          elems[tid + blockSize] -= (link(p,q).y + link(q,p).y);
          elems[tid + blockSize * 2] -= (link(p,q).x - link(q,p).x);
          elems[tid + blockSize * 3] -= (link(p,p).y - link(q,q).y);
        }
      }

#pragma unroll
      for (int mu_ = 4; mu_ < 4 + gauge_dir; mu_++) {
        cache.sync();
        if (mu_ == mu) {
          elems[tid] += link(p,p).x + link(q,q).x;
          elems[tid + blockSize] += (link(p,q).y + link(q,p).y);
          elems[tid + blockSize * 2] += (link(p,q).x - link(q,p).x);
          elems[tid + blockSize * 3] += (link(p,p).y - link(q,q).y);
        }
      }

      //FLOP per lattice site = gauge_dir * 2 * 7 = gauge_dir * 14
      cache.sync();
      if (mu == 0) {
        Float asq =  elems[tid + blockSize] * elems[tid + blockSize];
        asq += elems[tid + blockSize * 2] * elems[tid + blockSize * 2];
        asq += elems[tid + blockSize * 3] * elems[tid + blockSize * 3];
        Float a0sq = elems[tid] * elems[tid];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = quda::rsqrt((a0sq + x * x * asq));
        elems[tid] *= r;
        elems[tid + blockSize] *= x * r;
        elems[tid + blockSize * 2] *= x * r;
        elems[tid + blockSize * 3] *= x * r;
      } //FLOP per lattice site = 22 + 8 * 4
      cache.sync();
      //_____________
      if (mu < 4) {
        complex<Float> m0;
        //Do SU(2) hit on all upward links
        //left multiply an su3_matrix by an su2 matrix
        //link <- u * link
        //#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
          m0 = link(p,j);
          link(p,j) = complex<Float>( elems[tid], elems[tid + blockSize * 3] ) * m0 + complex<Float>( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
          link(q,j) = complex<Float>(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + complex<Float>( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
        }
      }
      else{
        complex<Float> m0;
        //Do SU(2) hit on all downward links
        //right multiply an su3_matrix by an su2 matrix
        //link <- link * u_adj
        //#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
          m0 = link(j,p);
          link(j,p) = complex<Float>( elems[tid], -elems[tid + blockSize * 3] ) * m0 + complex<Float>( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link(j,q);
          link(j,q) = complex<Float>(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + complex<Float>( elems[tid],elems[tid + blockSize * 3] ) * link(j,q);
        }
      }
      //_____________ //FLOP per lattice site = 8 * Nc * 2 * (2*6+2) = nColor * 224
      if ( block < (nColor * (nColor - 1) / 2) - 1 ) { cache.sync(); }
    } //FLOP per lattice site = (Nc * ( Nc - 1) / 2) * (22 + 28 gauge_dir + 224 Nc)
     //write updated link to global memory
  }

  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 8 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation needs 8x more shared memory than the implementation using atomicadd 
   */
  template <typename Float, int gauge_dir, int nColor>
  inline __device__ void GaugeFixHit_AtomicAdd(Matrix<complex<Float>,nColor> &link, Matrix<complex<Float>,nColor> &link1,
							const Float relax_boost, int mu)
  {
    auto blockSize = target::block_dim().x;
    auto tid = target::thread_idx().x;

    //Container for the four real parameters of SU(2) subgroup in shared memory
    SharedMemoryCache<Float> cache;
    auto elems = cache.data();

    //initialize shared memory
    if (mu < 4) elems[mu * blockSize + tid] = 0.0;
    cache.sync();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (nColor * (nColor - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<nColor>(block, p, q);
      if (mu < gauge_dir) {
        //Retrieve the four SU(2) parameters...
        // a0
        atomic_fetch_add(elems + tid, (link1(p,p)).x + (link1(q,q)).x + (link(p,p)).x + (link(q,q)).x); //a0
        // a1
        atomic_fetch_add(elems + tid + blockSize, (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y)); //a1
        // a2
        atomic_fetch_add(elems + tid + blockSize * 2, (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x)); //a2
        // a3
        atomic_fetch_add(elems + tid + blockSize * 3, (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y)); //a3
      }
      cache.sync();
      if (mu == 0) {
        //Over-relaxation boost
        Float asq =  elems[tid + blockSize] * elems[tid + blockSize];
        asq += elems[tid + blockSize * 2] * elems[tid + blockSize * 2];
        asq += elems[tid + blockSize * 3] * elems[tid + blockSize * 3];
        Float a0sq = elems[tid] * elems[tid];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = quda::rsqrt((a0sq + x * x * asq));
        elems[tid] *= r;
        elems[tid + blockSize] *= x * r;
        elems[tid + blockSize * 2] *= x * r;
        elems[tid + blockSize * 3] *= x * r;
      } //FLOP per lattice site = 22CUB: "Collective" Software Primitives for CUDA Kernel Development
      cache.sync();
      complex<Float> m0;
      //Do SU(2) hit on all upward links
      //left multiply an su3_matrix by an su2 matrix
      //link <- u * link
      //#pragma unroll
      for (int j = 0; j < nColor; j++) {
        m0 = link(p,j);
        link(p,j) = complex<Float>( elems[tid], elems[tid + blockSize * 3] ) * m0 +
	  complex<Float>( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
        link(q,j) = complex<Float>(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 +
	  complex<Float>( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
      }
      //Do SU(2) hit on all downward links
      //right multiply an su3_matrix by an su2 matrix
      //link <- link * u_adj
      //#pragma unroll
      for ( int j = 0; j < nColor; j++ ) {
        m0 = link1(j,p);
        link1(j,p) = complex<Float>( elems[tid], -elems[tid + blockSize * 3] ) * m0 +
	  complex<Float>( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link1(j,q);
        link1(j,q) = complex<Float>(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 +
	  complex<Float>( elems[tid],elems[tid + blockSize * 3] ) * link1(j,q);
      }
      if ( block < (nColor * (nColor - 1) / 2) - 1 ) {
        cache.sync();
        //reset shared memory SU(2) elements
        if (mu < 4) elems[mu * blockSize + tid] = 0.0;
        cache.sync();
      }
    }
  }

  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 4 threads per lattice site, the reduction is performed by shared memory using atomicadd.
   */
  template <typename Float, int gauge_dir, int nColor>
  inline __device__ void GaugeFixHit_NoAtomicAdd(Matrix<complex<Float>,nColor> &link, Matrix<complex<Float>,nColor> &link1,
                                                 const Float relax_boost, int mu)
  {
    auto blockSize = target::block_dim().x;
    auto tid = target::thread_idx().x;

    //Container for the four real parameters of SU(2) subgroup in shared memory
    SharedMemoryCache<Float> cache;
    auto elems = cache.data();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for (int block = 0; block < (nColor * (nColor - 1) / 2); block++) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<nColor>(block, p, q);
      if (mu < gauge_dir) {
        elems[mu * blockSize + tid] = link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
        elems[mu * blockSize + tid + blockSize * 4] = (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
        elems[mu * blockSize + tid + blockSize * 4 * 2] = (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
        elems[mu * blockSize + tid + blockSize * 4 * 3] = (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
      }
      cache.sync();
      if (mu == 0) {
        Float a0, a1, a2, a3;
        a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0;
      #pragma unroll
        for ( int i = 0; i < gauge_dir; i++ ) {
          a0 += elems[tid + i * blockSize];
          a1 += elems[tid + i * blockSize + blockSize * 4];
          a2 += elems[tid + i * blockSize + blockSize * 4 * 2];
          a3 += elems[tid + i * blockSize + blockSize * 4 * 3];
        }
        //Over-relaxation boost
        Float asq =  a1 * a1 + a2 * a2 + a3 * a3;
        Float a0sq = a0 * a0;
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = quda::rsqrt((a0sq + x * x * asq));
        elems[tid] = a0 * r;
        elems[tid + blockSize] = a1 * x * r;
        elems[tid + blockSize * 2] = a2 * x * r;
        elems[tid + blockSize * 3] = a3 * x * r;
      } //FLOP per lattice site = 22 + 8 * 4
      cache.sync();
      complex<Float> m0;
      //Do SU(2) hit on all upward links
      //left multiply an su3_matrix by an su2 matrix
      //link <- u * link
      //#pragma unroll
      for ( int j = 0; j < nColor; j++ ) {
        m0 = link(p,j);
        link(p,j) = complex<Float>( elems[tid], elems[tid + blockSize * 3] ) * m0 +
	  complex<Float>( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
        link(q,j) = complex<Float>(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 +
	  complex<Float>( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
      }
      //Do SU(2) hit on all downward links
      //right multiply an su3_matrix by an su2 matrix
      //link <- link * u_adj
      //#pragma unroll
      for ( int j = 0; j < nColor; j++ ) {
        m0 = link1(j,p);
        link1(j,p) = complex<Float>( elems[tid], -elems[tid + blockSize * 3] ) * m0 +
	  complex<Float>( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link1(j,q);
        link1(j,q) = complex<Float>(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 +
	  complex<Float>( elems[tid],elems[tid + blockSize * 3] ) * link1(j,q);
      }
      if ( block < (nColor * (nColor - 1) / 2) - 1 ) { cache.sync(); }
    }
  }

  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 4 threads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
   */
  template <typename Float, int gauge_dir, int nColor>
  inline __device__ void GaugeFixHit_NoAtomicAdd_LessSM(Matrix<complex<Float>,nColor> &link, Matrix<complex<Float>,nColor> &link1, const Float relax_boost, int mu)
  {
    auto blockSize = target::block_dim().x;
    auto tid = target::thread_idx().x;

    //Container for the four real parameters of SU(2) subgroup in shared memory
    SharedMemoryCache<Float> cache;
    auto elems = cache.data();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (nColor * (nColor - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<nColor>(block, p, q);
      if (mu == 0) {
        elems[tid] = link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
        elems[tid + blockSize] = (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] = (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] = (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
      }

#pragma unroll
      for (int mu_ = 1; mu_ < gauge_dir; mu_++) {
        cache.sync();
        if (mu_ == mu) {
          elems[tid] += link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
          elems[tid + blockSize] += (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
          elems[tid + blockSize * 2] += (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
          elems[tid + blockSize * 3] += (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
        }
      }

      cache.sync();
      if (mu == 0) {
        Float asq =  elems[tid + blockSize] * elems[tid + blockSize];
        asq += elems[tid + blockSize * 2] * elems[tid + blockSize * 2];
        asq += elems[tid + blockSize * 3] * elems[tid + blockSize * 3];
        Float a0sq = elems[tid] * elems[tid];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = quda::rsqrt((a0sq + x * x * asq));
        elems[tid] *= r;
        elems[tid + blockSize] *= x * r;
        elems[tid + blockSize * 2] *= x * r;
        elems[tid + blockSize * 3] *= x * r;
      }
      cache.sync();
      complex<Float> m0;
      //Do SU(2) hit on all upward links
      //left multiply an su3_matrix by an su2 matrix
      //link <- u * link
      //#pragma unroll
      for ( int j = 0; j < nColor; j++ ) {
        m0 = link(p,j);
        link(p,j) = complex<Float>( elems[tid], elems[tid + blockSize * 3] ) * m0 +
	  complex<Float>( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
        link(q,j) = complex<Float>(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 +
	  complex<Float>( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
      }
      //Do SU(2) hit on all downward links
      //right multiply an su3_matrix by an su2 matrix
      //link <- link * u_adj
      //#pragma unroll
      for ( int j = 0; j < nColor; j++ ) {
        m0 = link1(j,p);
        link1(j,p) = complex<Float>( elems[tid], -elems[tid + blockSize * 3] ) * m0 +
	  complex<Float>( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link1(j,q);
        link1(j,q) = complex<Float>(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 +
	  complex<Float>( elems[tid],elems[tid + blockSize * 3] ) * link1(j,q);
      }
      if ( block < (nColor * (nColor - 1) / 2) - 1 ) { cache.sync(); }
    }
  }

}

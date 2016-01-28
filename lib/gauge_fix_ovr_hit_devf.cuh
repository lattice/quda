#ifndef _GAUGE_FIX_OVR_HIT_DEVF_H
#define _GAUGE_FIX_OVR_HIT_DEVF_H


#include <quda_internal.h>
#include <quda_matrix.h>
#include <atomic.cuh>

namespace quda {



  template<class T>
  struct SharedMemory
  {
    __device__ inline operator T*()
    {
      extern __shared__ int __smem[];
      return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
      extern __shared__ int __smem[];
      return (T*)__smem;
    }
  };

  /**
   * Retrieve the SU(N) indices for the current block number
   * @param[in] block, current block number, from 0 to (NCOLORS * (NCOLORS - 1) / 2)
   * @param[out] p, row index pointing to the SU(N) matrix 
   * @param[out] q, column index pointing to the SU(N) matrix
  */
  template<int NCOLORS>
  static __host__ __device__ inline void   IndexBlock(int block, int &p, int &q){
    if ( NCOLORS == 3 ) {
      if ( block == 0 ) { p = 0; q = 1; }
      else if ( block == 1 ) { p = 1; q = 2; }
      else{ p = 0; q = 2; }
    }
    else{
      int i1;
      int found = 0;
      int del_i = 0;
      int index = -1;
      while ( del_i < (NCOLORS - 1) && found == 0 ) {
        del_i++;
        for ( i1 = 0; i1 < (NCOLORS - del_i); i1++ ) {
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
   * Uses 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation needs 8x more shared memory than the implementation using atomicadd 
   */
  template<int blockSize, typename Float2, typename Float, int gauge_dir, int NCOLORS>
  __forceinline__ __device__ void GaugeFixHit_AtomicAdd(Matrix<Float2,NCOLORS> &link, const Float relax_boost, const int tid){

    //Container for the four real parameters of SU(2) subgroup in shared memory
    //__shared__ Float elems[blockSize * 4];
    Float *elems = SharedMemory<Float>();
    //initialize shared memory
    if ( threadIdx.x < blockSize * 4 ) elems[threadIdx.x] = 0.0;
    __syncthreads();


    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (NCOLORS * (NCOLORS - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<NCOLORS>(block, p, q);
      Float asq = 1.0;
      if ( threadIdx.x < blockSize * 4 ) asq = -1.0;
      //FOR COULOMB AND LANDAU!!!!!!!!
      //if(nu0<gauge_dir){
      //In terms of thread index
      if ( threadIdx.x < blockSize * gauge_dir || (threadIdx.x >= blockSize * 4 && threadIdx.x < blockSize * (gauge_dir + 4))) {
        //Retrieve the four SU(2) parameters...
        // a0
        atomicAdd(elems + tid, (link(p,p)).x + (link(q,q)).x); //a0
        // a1
        atomicAdd(elems + tid + blockSize, (link(p,q).y + link(q,p).y) * asq); //a1
        // a2
        atomicAdd(elems + tid + blockSize * 2, (link(p,q).x - link(q,p).x) * asq); //a2
        // a3
        atomicAdd(elems + tid + blockSize * 3, (link(p,p).y - link(q,q).y) * asq); //a3
      } //FLOP per lattice site = gauge_dir * 2 * (4 + 7) = gauge_dir * 22
      __syncthreads();
      if ( threadIdx.x < blockSize ) {
        //Over-relaxation boost
        asq =  elems[threadIdx.x + blockSize] * elems[threadIdx.x + blockSize];
        asq += elems[threadIdx.x + blockSize * 2] * elems[threadIdx.x + blockSize * 2];
        asq += elems[threadIdx.x + blockSize * 3] * elems[threadIdx.x + blockSize * 3];
        Float a0sq = elems[threadIdx.x] * elems[threadIdx.x];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = rsqrt((a0sq + x * x * asq));
        elems[threadIdx.x] *= r;
        elems[threadIdx.x + blockSize] *= x * r;
        elems[threadIdx.x + blockSize * 2] *= x * r;
        elems[threadIdx.x + blockSize * 3] *= x * r;
      } //FLOP per lattice site = 22CUB: "Collective" Software Primitives for CUDA Kernel Development
      __syncthreads();
      //_____________
      if ( threadIdx.x < blockSize * 4 ) {
        Float2 m0;
        //Do SU(2) hit on all upward links
        //left multiply an su3_matrix by an su2 matrix
        //link <- u * link
        //#pragma unroll
        for ( int j = 0; j < NCOLORS; j++ ) {
          m0 = link(p,j);
          link(p,j) = makeComplex( elems[tid], elems[tid + blockSize * 3] ) * m0 + makeComplex( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
          link(q,j) = makeComplex(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + makeComplex( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
        }
      }
      else{
        Float2 m0;
        //Do SU(2) hit on all downward links
        //right multiply an su3_matrix by an su2 matrix
        //link <- link * u_adj
        //#pragma unroll
        for ( int j = 0; j < NCOLORS; j++ ) {
          m0 = link(j,p);
          link(j,p) = makeComplex( elems[tid], -elems[tid + blockSize * 3] ) * m0 + makeComplex( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link(j,q);
          link(j,q) = makeComplex(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + makeComplex( elems[tid],elems[tid + blockSize * 3] ) * link(j,q);
        }
      }
      //_____________ //FLOP per lattice site = 8 * NCOLORS * 2 * (2*6+2) = NCOLORS * 224
      if ( block < (NCOLORS * (NCOLORS - 1) / 2) - 1 ) {
        __syncthreads();
        //reset shared memory SU(2) elements
        if ( threadIdx.x < blockSize * 4 ) elems[threadIdx.x] = 0.0;
        __syncthreads();
      }
    } //FLOP per lattice site = (block < NCOLORS * ( NCOLORS - 1) / 2) * (22 + 28 gauge_dir + 224 NCOLORS)
     //write updated link to global memory
  }



  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 4 treads per lattice site, the reduction is performed by shared memory using atomicadd.
   */
  template<int blockSize, typename Float2, typename Float, int gauge_dir, int NCOLORS>
  __forceinline__ __device__ void GaugeFixHit_NoAtomicAdd(Matrix<Float2,NCOLORS> &link, const Float relax_boost, const int tid){

    //Container for the four real parameters of SU(2) subgroup in shared memory
    //__shared__ Float elems[blockSize * 4 * 8];
    Float *elems = SharedMemory<Float>();


    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (NCOLORS * (NCOLORS - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<NCOLORS>(block, p, q);
      /*Float asq = 1.0;
         if(threadIdx.x < blockSize * 4) asq = -1.0;
         if(threadIdx.x < blockSize * gauge_dir || (threadIdx.x >= blockSize * 4 && threadIdx.x < blockSize * (gauge_dir + 4))){
         elems[threadIdx.x] = link(p,p).x + link(q,q).x;
         elems[threadIdx.x + blockSize * 8] = (link(p,q).y + link(q,p).y) * asq;
         elems[threadIdx.x + blockSize * 8 * 2] = (link(p,q).x - link(q,p).x) * asq;
         elems[threadIdx.x + blockSize * 8 * 3] = (link(p,p).y - link(q,q).y) * asq;
         }*/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         //FLOP per lattice site = gauge_dir * 2 * 7 = gauge_dir * 14
      if ( threadIdx.x < blockSize * gauge_dir ) {
        elems[threadIdx.x] = link(p,p).x + link(q,q).x;
        elems[threadIdx.x + blockSize * 8] = -(link(p,q).y + link(q,p).y);
        elems[threadIdx.x + blockSize * 8 * 2] = -(link(p,q).x - link(q,p).x);
        elems[threadIdx.x + blockSize * 8 * 3] = -(link(p,p).y - link(q,q).y);
      }
      if ((threadIdx.x >= blockSize * 4 && threadIdx.x < blockSize * (gauge_dir + 4))) {
        elems[threadIdx.x] = link(p,p).x + link(q,q).x;
        elems[threadIdx.x + blockSize * 8] = (link(p,q).y + link(q,p).y);
        elems[threadIdx.x + blockSize * 8 * 2] = (link(p,q).x - link(q,p).x);
        elems[threadIdx.x + blockSize * 8 * 3] = (link(p,p).y - link(q,q).y);
      }
      //FLOP per lattice site = gauge_dir * 2 * 7 = gauge_dir * 14
      __syncthreads();
      if ( threadIdx.x < blockSize ) {
        Float a0, a1, a2, a3;
        a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0;
      #pragma unroll
        for ( int i = 0; i < gauge_dir; i++ ) {
          a0 += elems[tid + i * blockSize] + elems[tid + (i + 4) * blockSize];
          a1 += elems[tid + i * blockSize + blockSize * 8] + elems[tid + (i + 4) * blockSize + blockSize * 8];
          a2 += elems[tid + i * blockSize + blockSize * 8 * 2] + elems[tid + (i + 4) * blockSize + blockSize * 8 * 2];
          a3 += elems[tid + i * blockSize + blockSize * 8 * 3] + elems[tid + (i + 4) * blockSize + blockSize * 8 * 3];
        }
        //Over-relaxation boost
        Float asq =  a1 * a1 + a2 * a2 + a3 * a3;
        Float a0sq = a0 * a0;
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = rsqrt((a0sq + x * x * asq));
        elems[threadIdx.x] = a0 * r;
        elems[threadIdx.x + blockSize] = a1 * x * r;
        elems[threadIdx.x + blockSize * 2] = a2 * x * r;
        elems[threadIdx.x + blockSize * 3] = a3 * x * r;
      } //FLOP per lattice site = 22 + 8 * 4
      __syncthreads();
      //_____________
      if ( threadIdx.x < blockSize * 4 ) {
        Float2 m0;
        //Do SU(2) hit on all upward links
        //left multiply an su3_matrix by an su2 matrix
        //link <- u * link
        //#pragma unroll
        for ( int j = 0; j < NCOLORS; j++ ) {
          m0 = link(p,j);
          link(p,j) = makeComplex( elems[tid], elems[tid + blockSize * 3] ) * m0 + makeComplex( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
          link(q,j) = makeComplex(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + makeComplex( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
        }
      }
      else{
        Float2 m0;
        //Do SU(2) hit on all downward links
        //right multiply an su3_matrix by an su2 matrix
        //link <- link * u_adj
        //#pragma unroll
        for ( int j = 0; j < NCOLORS; j++ ) {
          m0 = link(j,p);
          link(j,p) = makeComplex( elems[tid], -elems[tid + blockSize * 3] ) * m0 + makeComplex( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link(j,q);
          link(j,q) = makeComplex(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + makeComplex( elems[tid],elems[tid + blockSize * 3] ) * link(j,q);
        }
      }
      //_____________ //FLOP per lattice site = 8 * NCOLORS * 2 * (2*6+2) = NCOLORS * 224
      if ( block < (NCOLORS * (NCOLORS - 1) / 2) - 1 ) { __syncthreads(); }
    } //FLOP per lattice site = (NCOLORS * ( NCOLORS - 1) / 2) * (22 + 28 gauge_dir + 224 NCOLORS)
     //write updated link to global memory
  }



  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
   */
  template<int blockSize, typename Float2, typename Float, int gauge_dir, int NCOLORS>
  __forceinline__ __device__ void GaugeFixHit_NoAtomicAdd_LessSM(Matrix<Float2,NCOLORS> &link, const Float relax_boost, const int tid){

    //Container for the four real parameters of SU(2) subgroup in shared memory
    //__shared__ Float elems[blockSize * 4 * 8];
    Float *elems = SharedMemory<Float>();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (NCOLORS * (NCOLORS - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<NCOLORS>(block, p, q);

      if ( threadIdx.x < blockSize ) {
        elems[tid] = link(p,p).x + link(q,q).x;
        elems[tid + blockSize] = -(link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] = -(link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] = -(link(p,p).y - link(q,q).y);
      }
      __syncthreads();
      if ( threadIdx.x < blockSize * 2 && threadIdx.x >= blockSize ) {
        elems[tid] += link(p,p).x + link(q,q).x;
        elems[tid + blockSize] -= (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] -= (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] -= (link(p,p).y - link(q,q).y);
      }
      __syncthreads();
      if ( threadIdx.x < blockSize * 3 && threadIdx.x >= blockSize * 2 ) {
        elems[tid] += link(p,p).x + link(q,q).x;
        elems[tid + blockSize] -= (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] -= (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] -= (link(p,p).y - link(q,q).y);
      }
      if ( gauge_dir == 4 ) {
        __syncthreads();
        if ( threadIdx.x < blockSize * 4 && threadIdx.x >= blockSize * 3 ) {
          elems[tid] += link(p,p).x + link(q,q).x;
          elems[tid + blockSize] -= (link(p,q).y + link(q,p).y);
          elems[tid + blockSize * 2] -= (link(p,q).x - link(q,p).x);
          elems[tid + blockSize * 3] -= (link(p,p).y - link(q,q).y);
        }
      }
      __syncthreads();
      if ( threadIdx.x < blockSize * 5 && threadIdx.x >= blockSize * 4 ) {
        elems[tid] += link(p,p).x + link(q,q).x;
        elems[tid + blockSize] += (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] += (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] += (link(p,p).y - link(q,q).y);
      }
      __syncthreads();
      if ( threadIdx.x < blockSize * 6 && threadIdx.x >= blockSize * 5 ) {
        elems[tid] += link(p,p).x + link(q,q).x;
        elems[tid + blockSize] += (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] += (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] += (link(p,p).y - link(q,q).y);
      }
      __syncthreads();
      if ( threadIdx.x < blockSize * 7 && threadIdx.x >= blockSize * 6 ) {
        elems[tid] += link(p,p).x + link(q,q).x;
        elems[tid + blockSize] += (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] += (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] += (link(p,p).y - link(q,q).y);
      }
      if ( gauge_dir == 4 ) {
        __syncthreads();
        if ( threadIdx.x < blockSize * 8 && threadIdx.x >= blockSize * 7 ) {
          elems[tid] += link(p,p).x + link(q,q).x;
          elems[tid + blockSize] += (link(p,q).y + link(q,p).y);
          elems[tid + blockSize * 2] += (link(p,q).x - link(q,p).x);
          elems[tid + blockSize * 3] += (link(p,p).y - link(q,q).y);
        }
      }
      //FLOP per lattice site = gauge_dir * 2 * 7 = gauge_dir * 14
      __syncthreads();
      if ( threadIdx.x < blockSize ) {
        Float asq =  elems[tid + blockSize] * elems[tid + blockSize];
        asq += elems[tid + blockSize * 2] * elems[tid + blockSize * 2];
        asq += elems[tid + blockSize * 3] * elems[tid + blockSize * 3];
        Float a0sq = elems[tid] * elems[tid];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = rsqrt((a0sq + x * x * asq));
        elems[tid] *= r;
        elems[tid + blockSize] *= x * r;
        elems[tid + blockSize * 2] *= x * r;
        elems[tid + blockSize * 3] *= x * r;
      } //FLOP per lattice site = 22 + 8 * 4
      __syncthreads();
      //_____________
      if ( threadIdx.x < blockSize * 4 ) {
        Float2 m0;
        //Do SU(2) hit on all upward links
        //left multiply an su3_matrix by an su2 matrix
        //link <- u * link
        //#pragma unroll
        for ( int j = 0; j < NCOLORS; j++ ) {
          m0 = link(p,j);
          link(p,j) = makeComplex( elems[tid], elems[tid + blockSize * 3] ) * m0 + makeComplex( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
          link(q,j) = makeComplex(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + makeComplex( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
        }
      }
      else{
        Float2 m0;
        //Do SU(2) hit on all downward links
        //right multiply an su3_matrix by an su2 matrix
        //link <- link * u_adj
        //#pragma unroll
        for ( int j = 0; j < NCOLORS; j++ ) {
          m0 = link(j,p);
          link(j,p) = makeComplex( elems[tid], -elems[tid + blockSize * 3] ) * m0 + makeComplex( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link(j,q);
          link(j,q) = makeComplex(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + makeComplex( elems[tid],elems[tid + blockSize * 3] ) * link(j,q);
        }
      }
      //_____________ //FLOP per lattice site = 8 * NCOLORS * 2 * (2*6+2) = NCOLORS * 224
      if ( block < (NCOLORS * (NCOLORS - 1) / 2) - 1 ) { __syncthreads(); }
    } //FLOP per lattice site = (NCOLORS * ( NCOLORS - 1) / 2) * (22 + 28 gauge_dir + 224 NCOLORS)
     //write updated link to global memory
  }















  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation needs 8x more shared memory than the implementation using atomicadd 
   */
  template<int blockSize, typename Float2, typename Float, int gauge_dir, int NCOLORS>
  __forceinline__ __device__ void GaugeFixHit_AtomicAdd(Matrix<Float2,NCOLORS> &link, Matrix<Float2,NCOLORS> &link1, const Float relax_boost, const int tid){

    //Container for the four real parameters of SU(2) subgroup in shared memory
    //__shared__ Float elems[blockSize * 4];
    Float *elems = SharedMemory<Float>();
    //initialize shared memory
    if ( threadIdx.x < blockSize * 4 ) elems[threadIdx.x] = 0.0;
    __syncthreads();


    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (NCOLORS * (NCOLORS - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<NCOLORS>(block, p, q);
      if ( threadIdx.x < blockSize * gauge_dir ) {
        //Retrieve the four SU(2) parameters...
        // a0
        atomicAdd(elems + tid, (link1(p,p)).x + (link1(q,q)).x + (link(p,p)).x + (link(q,q)).x); //a0
        // a1
        atomicAdd(elems + tid + blockSize, (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y)); //a1
        // a2
        atomicAdd(elems + tid + blockSize * 2, (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x)); //a2
        // a3
        atomicAdd(elems + tid + blockSize * 3, (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y)); //a3
      }
      __syncthreads();
      if ( threadIdx.x < blockSize ) {
        //Over-relaxation boost
        Float asq =  elems[threadIdx.x + blockSize] * elems[threadIdx.x + blockSize];
        asq += elems[threadIdx.x + blockSize * 2] * elems[threadIdx.x + blockSize * 2];
        asq += elems[threadIdx.x + blockSize * 3] * elems[threadIdx.x + blockSize * 3];
        Float a0sq = elems[threadIdx.x] * elems[threadIdx.x];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = rsqrt((a0sq + x * x * asq));
        elems[threadIdx.x] *= r;
        elems[threadIdx.x + blockSize] *= x * r;
        elems[threadIdx.x + blockSize * 2] *= x * r;
        elems[threadIdx.x + blockSize * 3] *= x * r;
      } //FLOP per lattice site = 22CUB: "Collective" Software Primitives for CUDA Kernel Development
      __syncthreads();
      Float2 m0;
      //Do SU(2) hit on all upward links
      //left multiply an su3_matrix by an su2 matrix
      //link <- u * link
      //#pragma unroll
      for ( int j = 0; j < NCOLORS; j++ ) {
        m0 = link(p,j);
        link(p,j) = makeComplex( elems[tid], elems[tid + blockSize * 3] ) * m0 + \
                    makeComplex( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
        link(q,j) = makeComplex(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + \
                    makeComplex( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
      }
      //Do SU(2) hit on all downward links
      //right multiply an su3_matrix by an su2 matrix
      //link <- link * u_adj
      //#pragma unroll
      for ( int j = 0; j < NCOLORS; j++ ) {
        m0 = link1(j,p);
        link1(j,p) = makeComplex( elems[tid], -elems[tid + blockSize * 3] ) * m0 + \
                     makeComplex( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link1(j,q);
        link1(j,q) = makeComplex(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + \
                     makeComplex( elems[tid],elems[tid + blockSize * 3] ) * link1(j,q);
      }
      if ( block < (NCOLORS * (NCOLORS - 1) / 2) - 1 ) {
        __syncthreads();
        //reset shared memory SU(2) elements
        if ( threadIdx.x < blockSize * 4 ) elems[threadIdx.x] = 0.0;
        __syncthreads();
      }
    }
  }















  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 4 treads per lattice site, the reduction is performed by shared memory using atomicadd.
   */
  template<int blockSize, typename Float2, typename Float, int gauge_dir, int NCOLORS>
  __forceinline__ __device__ void GaugeFixHit_NoAtomicAdd(Matrix<Float2,NCOLORS> &link, Matrix<Float2,NCOLORS> &link1, const Float relax_boost, const int tid){

    //Container for the four real parameters of SU(2) subgroup in shared memory
    //__shared__ Float elems[blockSize * 4 * 8];
    Float *elems = SharedMemory<Float>();
    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (NCOLORS * (NCOLORS - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<NCOLORS>(block, p, q);
      if ( threadIdx.x < blockSize * gauge_dir ) {
        elems[threadIdx.x] = link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
        elems[threadIdx.x + blockSize * 4] = (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
        elems[threadIdx.x + blockSize * 4 * 2] = (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
        elems[threadIdx.x + blockSize * 4 * 3] = (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
      }
      __syncthreads();
      if ( threadIdx.x < blockSize ) {
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
        Float r = rsqrt((a0sq + x * x * asq));
        elems[threadIdx.x] = a0 * r;
        elems[threadIdx.x + blockSize] = a1 * x * r;
        elems[threadIdx.x + blockSize * 2] = a2 * x * r;
        elems[threadIdx.x + blockSize * 3] = a3 * x * r;
      } //FLOP per lattice site = 22 + 8 * 4
      __syncthreads();
      Float2 m0;
      //Do SU(2) hit on all upward links
      //left multiply an su3_matrix by an su2 matrix
      //link <- u * link
      //#pragma unroll
      for ( int j = 0; j < NCOLORS; j++ ) {
        m0 = link(p,j);
        link(p,j) = makeComplex( elems[tid], elems[tid + blockSize * 3] ) * m0 + \
                    makeComplex( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
        link(q,j) = makeComplex(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + \
                    makeComplex( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
      }
      //Do SU(2) hit on all downward links
      //right multiply an su3_matrix by an su2 matrix
      //link <- link * u_adj
      //#pragma unroll
      for ( int j = 0; j < NCOLORS; j++ ) {
        m0 = link1(j,p);
        link1(j,p) = makeComplex( elems[tid], -elems[tid + blockSize * 3] ) * m0 + \
                     makeComplex( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link1(j,q);
        link1(j,q) = makeComplex(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + \
                     makeComplex( elems[tid],elems[tid + blockSize * 3] ) * link1(j,q);
      }
      if ( block < (NCOLORS * (NCOLORS - 1) / 2) - 1 ) { __syncthreads(); }
    }
  }





  /**
   * Device function to perform gauge fixing with overrelxation.
   * Uses 4 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
   * This implementation uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
   */
  template<int blockSize, typename Float2, typename Float, int gauge_dir, int NCOLORS>
  __forceinline__ __device__ void GaugeFixHit_NoAtomicAdd_LessSM(Matrix<Float2,NCOLORS> &link, Matrix<Float2,NCOLORS> &link1, const Float relax_boost, const int tid){

    //Container for the four real parameters of SU(2) subgroup in shared memory
    //__shared__ Float elems[blockSize * 4 * 8];
    Float *elems = SharedMemory<Float>();

    //Loop over all SU(2) subroups of SU(N)
    //#pragma unroll
    for ( int block = 0; block < (NCOLORS * (NCOLORS - 1) / 2); block++ ) {
      int p, q;
      //Get the two indices for the SU(N) matrix
      IndexBlock<NCOLORS>(block, p, q);
      if ( threadIdx.x < blockSize ) {
        elems[tid] = link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
        elems[tid + blockSize] = (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] = (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] = (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
      }
      __syncthreads();
      if ( threadIdx.x < blockSize * 2 && threadIdx.x >= blockSize ) {
        elems[tid] += link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
        elems[tid + blockSize] += (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] += (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] += (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
      }
      __syncthreads();
      if ( threadIdx.x < blockSize * 3 && threadIdx.x >= blockSize * 2 ) {
        elems[tid] += link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
        elems[tid + blockSize] += (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
        elems[tid + blockSize * 2] += (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
        elems[tid + blockSize * 3] += (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
      }
      if ( gauge_dir == 4 ) {
        __syncthreads();
        if ( threadIdx.x < blockSize * 4 && threadIdx.x >= blockSize * 3 ) {
          elems[tid] += link1(p,p).x + link1(q,q).x + link(p,p).x + link(q,q).x;
          elems[tid + blockSize] += (link1(p,q).y + link1(q,p).y) - (link(p,q).y + link(q,p).y);
          elems[tid + blockSize * 2] += (link1(p,q).x - link1(q,p).x) - (link(p,q).x - link(q,p).x);
          elems[tid + blockSize * 3] += (link1(p,p).y - link1(q,q).y) - (link(p,p).y - link(q,q).y);
        }
      }
      __syncthreads();
      if ( threadIdx.x < blockSize ) {
        Float asq =  elems[tid + blockSize] * elems[tid + blockSize];
        asq += elems[tid + blockSize * 2] * elems[tid + blockSize * 2];
        asq += elems[tid + blockSize * 3] * elems[tid + blockSize * 3];
        Float a0sq = elems[tid] * elems[tid];
        Float x = (relax_boost * a0sq + asq) / (a0sq + asq);
        Float r = rsqrt((a0sq + x * x * asq));
        elems[tid] *= r;
        elems[tid + blockSize] *= x * r;
        elems[tid + blockSize * 2] *= x * r;
        elems[tid + blockSize * 3] *= x * r;
      }
      __syncthreads();
      Float2 m0;
      //Do SU(2) hit on all upward links
      //left multiply an su3_matrix by an su2 matrix
      //link <- u * link
      //#pragma unroll
      for ( int j = 0; j < NCOLORS; j++ ) {
        m0 = link(p,j);
        link(p,j) = makeComplex( elems[tid], elems[tid + blockSize * 3] ) * m0 + \
                    makeComplex( elems[tid + blockSize * 2], elems[tid + blockSize] ) * link(q,j);
        link(q,j) = makeComplex(-elems[tid + blockSize * 2], elems[tid + blockSize]) * m0 + \
                    makeComplex( elems[tid],-elems[tid + blockSize * 3] ) * link(q,j);
      }
      //Do SU(2) hit on all downward links
      //right multiply an su3_matrix by an su2 matrix
      //link <- link * u_adj
      //#pragma unroll
      for ( int j = 0; j < NCOLORS; j++ ) {
        m0 = link1(j,p);
        link1(j,p) = makeComplex( elems[tid], -elems[tid + blockSize * 3] ) * m0 + \
                     makeComplex( elems[tid + blockSize * 2], -elems[tid + blockSize] ) * link1(j,q);
        link1(j,q) = makeComplex(-elems[tid + blockSize * 2], -elems[tid + blockSize]) * m0 + \
                     makeComplex( elems[tid],elems[tid + blockSize * 3] ) * link1(j,q);
      }
      if ( block < (NCOLORS * (NCOLORS - 1) / 2) - 1 ) { __syncthreads(); }
    }
  }

}
#endif

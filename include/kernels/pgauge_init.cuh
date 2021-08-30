#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <random_helper.h>
#include <index_helper.cuh>
#include <kernel.h>

#ifndef PI
#define PI    3.1415926535897932384626433832795    // pi
#endif
#ifndef PII
#define PII   6.2831853071795864769252867665590    // 2 * pi
#endif

namespace quda {

  template <typename Float, int nColor_, QudaReconstructType recon_>
  struct InitGaugeColdArg : kernel_param<> {
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    using real = typename mapper<Float>::type;
    using Gauge = typename gauge_mapper<real, recon>::type;
    int X[4]; // grid dimensions
    Gauge U;
    InitGaugeColdArg(const GaugeField &U) :
      kernel_param(dim3(U.VolumeCB(), 2, 1)),
      U(U)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = U.X()[dir];
    }
  };

  template <typename Arg> struct ColdStart {
    const Arg &arg;
    constexpr ColdStart(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      Matrix<complex<typename Arg::real>, Arg::nColor> U;
      setIdentity(&U);
      for ( int d = 0; d < 4; d++ ) arg.U(d, x_cb, parity) = U;
    }
  };

  template <typename Float, int nColor_, QudaReconstructType recon_>
  struct InitGaugeHotArg : kernel_param<> {
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    using real = typename mapper<Float>::type;
    using Gauge = typename gauge_mapper<real, recon>::type;
    int X[4]; // grid dimensions
    Gauge U;
    RNGState *rng;
    int border[4];
    InitGaugeHotArg(const GaugeField &U, RNGState *rng) :
      //the optimal number of RNG states in rngstate array must be equal to half the lattice volume
      //this number is the same used in heatbath...
      kernel_param(dim3(U.LocalVolumeCB(), 1, 1)),
      U(U),
      rng(rng)
    {
      for (int dir = 0; dir < 4; dir++) {
        border[dir] = U.R()[dir];
        X[dir] = U.X()[dir] - border[dir] * 2;
      } 
    }
  };

  template <typename Float>
  __host__ __device__ static inline void reunit_link( Matrix<complex<Float>,3> &U )
  {
    complex<Float> t2((Float)0.0, (Float)0.0);
    Float t1 = 0.0;
    //first normalize first row
    //sum of squares of row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t1 += norm(U(0,c));
    t1 = (Float)1.0 / sqrt(t1);
    //14
    //used to normalize row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(0,c) *= t1;
    //6
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t2 += conj(U(0,c)) * U(1,c);
    //24
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(1,c) -= t2 * U(0,c);
    //24
    //normalize second row
    //sum of squares of row
    t1 = 0.0;
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t1 += norm(U(1,c));
    t1 = (Float)1.0 / sqrt(t1);
    //14
    //used to normalize row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(1, c) *= t1;
    //6
    //Reconstruct lat row
    U(2,0) = conj(U(0,1) * U(1,2) - U(0,2) * U(1,1));
    U(2,1) = conj(U(0,2) * U(1,0) - U(0,0) * U(1,2));
    U(2,2) = conj(U(0,0) * U(1,1) - U(0,1) * U(1,0));
    //42
    //T=130
  }

  /**
     @brief Generate the four random real elements of the SU(2) matrix
     @param localstate CURAND rng state
     @return four real numbers of the SU(2) matrix
  */
  template <class T>
  __device__ static inline Matrix<T,2> randomSU2(RNGState& localState){
    Matrix<T,2> a;
    T aabs, ctheta, stheta, phi;
    a(0,0) = uniform<T>::rand(localState, (T)-1.0, (T)1.0);
    aabs = sqrt( 1.0 - a(0,0) * a(0,0));
    ctheta = uniform<T>::rand(localState, (T)-1.0, (T)1.0);
    phi = PII * uniform<T>::rand(localState);

    // Was   xurand(*localState>& 1 ? 1 : -1
    // which presumably just selects when the lowest bit is 1 or 0 with 50% probability each
    // so this should do the same, without an appeal to the bit swizzle, but may end up being slower.
    stheta = ( uniform<T>::rand(localState) < static_cast<T>(0.5) ? 1 : -1 ) * sqrt( (T)1.0 - ctheta * ctheta );
    a(0,1) = aabs * stheta * cos( phi );
    a(1,0) = aabs * stheta * sin( phi );
    a(1,1) = aabs * ctheta;
    return a;
  }

  /**
     @brief Update the SU(Nc) link with the new SU(2) matrix, link <- u * link
     @param u SU(2) matrix represented by four real numbers
     @param link SU(Nc) matrix
     @param id indices
  */
  template <class T, int nColor>
  __host__ __device__ static inline void mul_block_sun( Matrix<T,2> u, Matrix<complex<T>,nColor> &link, int2 id ){
    for ( int j = 0; j < nColor; j++ ) {
      complex<T> tmp = complex<T>( u(0,0), u(1,1) ) * link(id.x, j) + complex<T>( u(1,0), u(0,1) ) * link(id.y, j);
      link(id.y, j) = complex<T>(-u(1,0), u(0,1) ) * link(id.x, j) + complex<T>( u(0,0),-u(1,1) ) * link(id.y, j);
      link(id.x, j) = tmp;
    }
  }

  /**
     @brief Calculate the SU(2) index block in the SU(Nc) matrix
     @param block number to calculate the index's, the total number of blocks is Nc * ( Nc - 1) / 2.
     @return Returns two index's in int2 type, accessed by .x and .y.
  */
  template<int nColor>
  __host__ __device__ static inline int2 IndexBlock(int block){
    int2 id;
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
    id.y = i1 + del_i;
    id.x = i1;
    return id;
  }

  /**
     @brief Generate a SU(Nc) random matrix
     @param localstate CURAND rng state
     @return SU(Nc) matrix
  */
  template <class Float, int nColor>
  __device__ inline Matrix<complex<Float>,nColor> randomize( RNGState& localState )
  {
    Matrix<complex<Float>,nColor> U;

    for ( int i = 0; i < nColor; i++ )
      for ( int j = 0; j < nColor; j++ )
        U(i,j) = complex<Float>( (Float)(uniform<Float>::rand(localState) - 0.5), (Float)(uniform<Float>::rand(localState) - 0.5) );
    reunit_link<Float>(U);
    return U;

    /*setIdentity(&U);
       for( int block = 0; block < nColor * ( nColor - 1) / 2; block++ ) {
       Matrix<Float,2> rr = randomSU2<Float>(localState);
       int2 id = IndexBlock<nColor>( block );
       mul_block_sun<Float, nColor>(rr, U, id);
       //U = block_su2_to_su3<Float>( U, a00, a01, a10, a11, block );
       }
       return U;*/
  }

  template<typename Arg> struct HotStart {
    const Arg &arg;
    constexpr HotStart(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb)
    {
      int X[4], x[4];
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
      for ( int dr = 0; dr < 4; ++dr ) X[dr] += 2 * arg.border[dr];
      RNGState localState = arg.rng[x_cb];
      for (int parity = 0; parity < 2; parity++) {
        getCoords(x, x_cb, arg.X, parity);
        for (int dr = 0; dr < 4; dr++) x[dr] += arg.border[dr];
        int e_cb = linkIndex(x, X);
        for (int d = 0; d < 4; d++) {
          Matrix<complex<typename Arg::real>, Arg::nColor> U;
          U = randomize<typename Arg::real, Arg::nColor>(localState);
          arg.U(d, e_cb, parity) = U;
        }
      }
      arg.rng[x_cb] = localState;
    }
  };

}

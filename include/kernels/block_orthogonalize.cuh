#include <multigrid_helper.cuh>
#include <fast_intdiv.h>

// this removes ghost accessor reducing the parameter space needed
#define DISABLE_GHOST true // do not rename this (it is both a template parameter and a macro)

#include <color_spinor_field_order.h>
#include <cub_helper.cuh>

// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
//#define SWIZZLE

namespace quda {

#define MAX_MATRIX_SIZE 4096
  __constant__ signed char B_array_d[MAX_MATRIX_SIZE];

  // to avoid overflowing the parameter space we put the B array into a separate constant memory buffer
  static signed char B_array_h[MAX_MATRIX_SIZE];

  /**
      Kernel argument struct
  */
  template <typename Rotator, typename Vector, int fineSpin, int spinBlockSize, int coarseSpin, int nVec>
  struct BlockOrthoArg {
    Rotator V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    const int nBlockOrtho; // number of times we Gram-Schmidt
    int coarseVolume;
    int fineVolumeCB;
    int geoBlockSizeCB; // number of geometric elements in each checkerboarded block
    int_fastdiv swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate
    const Vector *B;
    template <typename... T>
    BlockOrthoArg(ColorSpinorField &V, const int *fine_to_coarse, const int *coarse_to_fine, int parity,
                  const int *geo_bs, const int n_block_ortho, const ColorSpinorField &meta, T... B) :
      V(V),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      spin_map(),
      parity(parity),
      nParity(meta.SiteSubset()),
      nBlockOrtho(n_block_ortho),
      B(V.Location() == QUDA_CPU_FIELD_LOCATION ? reinterpret_cast<Vector *>(B_array_h) : nullptr)
    {
      const Vector Btmp[nVec]{*B...};
      if (sizeof(Btmp) > MAX_MATRIX_SIZE) errorQuda("B array size (%lu) is larger than maximum allowed (%d)\n", sizeof(Btmp), MAX_MATRIX_SIZE);
      memcpy(B_array_h, (void *)Btmp, sizeof(Btmp));
      int geoBlockSize = 1;
      for (int d = 0; d < V.Ndim(); d++) geoBlockSize *= geo_bs[d];
      geoBlockSizeCB = geoBlockSize/2;
      coarseVolume = meta.Volume() / geoBlockSize;
      fineVolumeCB = meta.VolumeCB();
      if (nParity != 2) errorQuda("BlockOrtho only presently supports full fields");
    }
  };

  template <int nColor, typename sumType, typename real>
  inline __device__ __host__ void colorInnerProduct(complex<sumType> &dot, int i, complex<real> v[nColor],
                                                    complex<real> w[nColor])
  {
#pragma unroll
    for (int c = 0; c < nColor; c++) {
      dot.x += w[c].real() * v[c].real();
      dot.x += w[c].imag() * v[c].imag();
      dot.y += w[c].real() * v[c].imag();
      dot.y -= w[c].imag() * v[c].real();
    }
  }

  template <int nColor, typename sumType, typename real>
  inline __device__ __host__ void colorNorm(sumType &nrm, complex<real> v[nColor])
  {
#pragma unroll
    for (int c = 0; c < nColor; c++) {
      nrm += v[c].real() * v[c].real();
      nrm += v[c].imag() * v[c].imag();
    }
  }

  template <typename real, int nColor>
  inline __device__ __host__ void colorScaleSubtract(complex<real> v[nColor], complex<real> a, complex<real> w[nColor])
  {
#pragma unroll
    for (int c = 0; c < nColor; c++) {
      v[c].x -= a.real() * w[c].real();
      v[c].x += a.imag() * w[c].imag();
      v[c].y -= a.real() * w[c].imag();
      v[c].y -= a.imag() * w[c].real();
    }
  }

  template <typename real, int nColor> inline __device__ __host__ void colorScale(complex<real> v[nColor], real a)
  {
#pragma unroll
    for (int c=0; c<nColor; c++) v[c] *= a;
  }

#ifndef __CUDACC_RTC___
  template <typename sumFloat, typename Float, int nSpin, int spinBlockSize, int nColor, int coarseSpin, int nVec, typename Arg>
  void blockOrthoCPU(Arg &arg)
  {
    // loop over geometric blocks
#pragma omp parallel for
    for (int x_coarse=0; x_coarse<arg.coarseVolume; x_coarse++) {

      // first copy over raw components into the container
      for (int j = 0; j < nVec; j++) {
        for (int parity = 0; parity < arg.nParity; parity++) {
          parity = (arg.nParity == 2) ? parity : arg.parity;
          for (int b = 0; b < arg.geoBlockSizeCB; b++) {
            int x = arg.coarse_to_fine[(x_coarse * 2 + parity) * arg.geoBlockSizeCB + b];
            int x_cb = x - parity * arg.fineVolumeCB;
            for (int s = 0; s < nSpin; s++) {
              for (int c = 0; c < nColor; c++) {
                arg.V(parity, x_cb, s, c, j) = arg.B[j](parity, x_cb, s, c);
              }
            }
          }
        }
      }

      // loop over number of block orthos
      for (int n = 0; n < arg.nBlockOrtho; n++) {
        for (int j = 0; j < nVec; j++) {

          for (int i = 0; i < j; i++) {

            // compute (j,i) block inner products

            complex<sumFloat> dot[coarseSpin];
            for (int s = 0; s < coarseSpin; s++) dot[s] = 0.0;
            for (int parity = 0; parity < arg.nParity; parity++) {
              parity = (arg.nParity == 2) ? parity : arg.parity;

              for (int b = 0; b < arg.geoBlockSizeCB; b++) {

                int x = arg.coarse_to_fine[(x_coarse * 2 + parity) * arg.geoBlockSizeCB + b];
                int x_cb = x - parity * arg.fineVolumeCB;

                complex<Float> v[nSpin][nColor];
                for (int s = 0; s < nSpin; s++)
                  for (int c = 0; c < nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

                for (int s = 0; s < nSpin; s++) {
                  complex<Float> vis[nColor];
                  for (int c = 0; c < nColor; c++) vis[c] = arg.V(parity, x_cb, s, c, i);
                  colorInnerProduct<nColor>(dot[arg.spin_map(s, parity)], i, v[s], vis);
                }
              }
            }

            // subtract the i blocks to orthogonalise
            for (int parity = 0; parity < arg.nParity; parity++) {
              parity = (arg.nParity == 2) ? parity : arg.parity;

              for (int b = 0; b < arg.geoBlockSizeCB; b++) {

                int x = arg.coarse_to_fine[(x_coarse * 2 + parity) * arg.geoBlockSizeCB + b];
                int x_cb = x - parity * arg.fineVolumeCB;

                complex<Float> v[nSpin][nColor];
                for (int s = 0; s < nSpin; s++)
                  for (int c = 0; c < nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

                for (int s = 0; s < nSpin; s++) {
                  complex<Float> vis[nColor];
                  for (int c = 0; c < nColor; c++) vis[c] = arg.V(parity, x_cb, s, c, i);
                  colorScaleSubtract<Float, nColor>(v[s], static_cast<complex<Float>>(dot[arg.spin_map(s, parity)]), vis);
                }

                for (int s = 0; s < nSpin; s++)
                  for (int c = 0; c < nColor; c++) arg.V(parity, x_cb, s, c, j) = v[s][c];
              }
            }

          } // i

          sumFloat nrm[coarseSpin] = {};
          for (int parity = 0; parity < arg.nParity; parity++) {
            parity = (arg.nParity == 2) ? parity : arg.parity;

            for (int b = 0; b < arg.geoBlockSizeCB; b++) {

              int x = arg.coarse_to_fine[(x_coarse * 2 + parity) * arg.geoBlockSizeCB + b];
              int x_cb = x - parity * arg.fineVolumeCB;

              complex<Float> v[nSpin][nColor];
              for (int s = 0; s < nSpin; s++)
                for (int c = 0; c < nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);
              for (int s = 0; s < nSpin; s++) { colorNorm<nColor>(nrm[arg.spin_map(s, parity)], v[s]); }
            }
          }

          for (int s = 0; s < coarseSpin; s++) nrm[s] = nrm[s] > 0.0 ? rsqrt(nrm[s]) : 0.0;

          for (int parity = 0; parity < arg.nParity; parity++) {
            parity = (arg.nParity == 2) ? parity : arg.parity;

            for (int b = 0; b < arg.geoBlockSizeCB; b++) {

              int x = arg.coarse_to_fine[(x_coarse * 2 + parity) * arg.geoBlockSizeCB + b];
              int x_cb = x - parity * arg.fineVolumeCB;

              complex<Float> v[nSpin][nColor];
              for (int s = 0; s < nSpin; s++)
                for (int c = 0; c < nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

              for (int s = 0; s < nSpin; s++) { colorScale<Float, nColor>(v[s], nrm[arg.spin_map(s, parity)]); }

              for (int s = 0; s < nSpin; s++)
                for (int c = 0; c < nColor; c++) arg.V(parity, x_cb, s, c, j) = v[s][c];
            }
          }

        } // j

      } // n

    } // x_coarse
  }
#endif

  template <int block_size, typename sumFloat, typename Float, int nSpin, int spinBlockSize, int nColor, int coarseSpin,
            int nVec, typename Arg>
  __launch_bounds__(2 * block_size) __global__ void blockOrthoGPU(Arg arg)
  {
    int x_coarse = blockIdx.x;
#ifdef SWIZZLE
    // the portion of the grid that is exactly divisible by the number of SMs
    const int gridp = gridDim.x - gridDim.x % arg.swizzle;

    if (blockIdx.x < gridp) {
      // this is the portion of the block that we are going to transpose
      const int i = blockIdx.x % arg.swizzle;
      const int j = blockIdx.x / arg.swizzle;

      // tranpose the coordinates
      x_coarse = i * (gridp / arg.swizzle) + j;
    }
#endif
    int parity = (arg.nParity == 2) ? threadIdx.y + blockIdx.y*blockDim.y : arg.parity;
    int x = arg.coarse_to_fine[ (x_coarse*2 + parity) * blockDim.x + threadIdx.x];
    int x_cb = x - parity*arg.fineVolumeCB;
    if (x_cb >= arg.fineVolumeCB) return;
    int chirality = blockIdx.z; // which chiral block we're working on (if chirality is present)

    constexpr int spinBlock = nSpin / coarseSpin; // size of spin block
    typedef cub::BlockReduce<complex<sumFloat>, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> dotReduce;
    typedef cub::BlockReduce<sumFloat, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> normReduce;

    __shared__ typename dotReduce::TempStorage dot_storage;
    typename normReduce::TempStorage *norm_storage = (typename normReduce::TempStorage *)&dot_storage;
    complex<sumFloat> *dot_ = (complex<sumFloat> *)&dot_storage;
    sumFloat *nrm_ = (sumFloat *)&dot_storage;

    // cast the constant memory buffer to a Vector array
    const auto *B = reinterpret_cast<decltype(arg.B)>(B_array_d);

    // loop over number of block orthos
    for (int n = 0; n < arg.nBlockOrtho; n++) {
      for (int j = 0; j < nVec; j++) {

        complex<Float> v[spinBlock][nColor];
        if (n == 0) { // load from B on first Gram-Schmidt, otherwise V.
#pragma unroll
          for (int s = 0; s < spinBlock; s++)
#pragma unroll
            for (int c = 0; c < nColor; c++) v[s][c] = B[j](parity, x_cb, chirality * spinBlock + s, c);
        } else {
#pragma unroll
          for (int s = 0; s < spinBlock; s++)
#pragma unroll
            for (int c = 0; c < nColor; c++) v[s][c] = arg.V(parity, x_cb, chirality * spinBlock + s, c, j);
        }

        for (int i = 0; i < j; i++) {

          complex<Float> dot = 0.0;

          // compute (j,i) block inner products
          complex<Float> vi[spinBlock][nColor];
#pragma unroll
          for (int s = 0; s < spinBlock; s++)
#pragma unroll
            for (int c = 0; c < nColor; c++) vi[s][c] = arg.V(parity, x_cb, chirality * spinBlock + s, c, i);

#pragma unroll
          for (int s = 0; s < spinBlock; s++) { colorInnerProduct<nColor>(dot, i, v[s], vi[s]); }

          __syncthreads();
          dot = dotReduce(dot_storage).Sum(dot);
          if (threadIdx.x == 0 && threadIdx.y == 0) *dot_ = dot;
          __syncthreads();
          dot = *dot_;

          // subtract the blocks to orthogonalise
#pragma unroll
          for (int s = 0; s < spinBlock; s++) { colorScaleSubtract<Float, nColor>(v[s], dot, vi[s]); }

        } // i

        // normalize the block
        sumFloat nrm = static_cast<sumFloat>(0.0);

#pragma unroll
        for (int s = 0; s < spinBlock; s++) { colorNorm<nColor>(nrm, v[s]); }

        __syncthreads();
        nrm = normReduce(*norm_storage).Sum(nrm);
        if (threadIdx.x == 0 && threadIdx.y == 0) *nrm_ = nrm;
        __syncthreads();
        nrm = *nrm_;

        nrm = nrm > 0.0 ? rsqrt(nrm) : 0.0;

#pragma unroll
        for (int s = 0; s < spinBlock; s++) { colorScale<Float, nColor>(v[s], nrm); }

#pragma unroll
        for (int s = 0; s < spinBlock; s++)
#pragma unroll
          for (int c = 0; c < nColor; c++) arg.V(parity, x_cb, chirality * spinBlock + s, c, j) = v[s][c];

      } // j
    }   // n
  }

} // namespace quda

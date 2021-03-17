#pragma once

#include <clover_field_order.h>
#include <kernels/dslash_wilson.cuh>
#include <shared_memory_cache_helper.cuh>
#include <linalg.cuh>

namespace quda
{
  
  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_, bool dynamic_clover_, bool asymmetric_>
    struct NdegTwistedCloverPreconditionedArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    using WilsonArg<Float, nColor, nDim, reconstruct_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool dynamic_clover = dynamic_clover_;
    static constexpr bool asymmetric = asymmetric_;
    
    typedef typename mapper<Float>::type real;
    typedef typename clover_mapper<Float, length>::type C;
    const C A;
    const C A2inv; // A^{-2}
    real a;          /** this is the Wilson-dslash scale factor */
    real b;          /** this is the chiral twist factor */
    real c;          /** this is the flavor twist factor */
    
  NdegTwistedCloverPreconditionedArg(ColorSpinorField &out, const ColorSpinorField &in,
                                     const GaugeField &U, const CloverField &A,
                                     double a, double b, double c,
                                     const ColorSpinorField &x, int parity, bool dagger,
                                     const int *comm_override) :
    WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, a, x, parity, dagger, comm_override),
      A(A, false),
      A2inv(A, dynamic_clover ? false : true), // if dynamic clover we don't want the inverse field
      a(a),
      b(dagger ? -0.5 * b : 0.5 * b), // if dagger flip the chiral twist
      c(0.5*c)
      {
      }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
    struct nDegTwistedCloverPreconditioned : dslash_default {
    
    Arg &arg;
    constexpr nDegTwistedCloverPreconditioned(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

  
  /**
     @brief Apply the twisted-clover dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1)*x
     Note this routine only exists in xpay form.
  */
    __device__ __host__ inline void operator()(int idx, int flavor, int parity)
    {
      using namespace linalg; // for Cholesky
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
      constexpr int n = Arg::nColor * Arg::nSpin / 2;
      typedef HMatrix<real, n> HMat;

      bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                          // which dimension is thread working on (fused kernel only)
      auto clover_coord = getCoords<QUDA_4D_PC, kernel_type>(arg, idx, 0, parity, thread_dim);
      auto coord = getCoords<QUDA_4D_PC, kernel_type>(arg, idx, flavor, parity, thread_dim);


      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      applyWilson<Float, nDim, nColor, nParity, dagger, kernel_type>(out, arg, coord, x_cb, flavor, parity, idx, thread_dim, active);

      int my_flavor_idx = coord.x_cb + flavor * arg.dc.volume_4d_cb;

      if (kernel_type != INTERIOR_KERNEL && active) {
        // if we're not the interior kernel, then we must sum the partial
        Vector x = arg.out(my_flavor_idx, my_spinor_parity);
        out += x;
      }

      if (isComplete<kernel_type>(arg, coord) && active) {

        VectorCache<real, HalfVector> cache; // used for flavor twist
        Vector tmp;

#pragma unroll
        for (int chirality = 0; chirality < 2; chirality++) {
          HMat A = arg.A(clover_coord.x_cb, parity, chirality);
          
          HalfVector chi = out.chiral_project(chirality);
          cache.save(chi); // put "chi" in shared memory so the other flavor can access it
          
          HalfVector A_chi = A * chi;
          const complex<real> b(0.0, (chirality^flavor) == 0 ? arg.b : -arg.b);
          A_chi += b * chi;
          
          cache.sync(); // safe to sync in here since other threads will exit
          A_chi += arg.c * cache.load(threadIdx.x, 1-flavor, threadIdx.z);
          
          if (arg.dynamic_clover) {
            HMat A2 = A.square();
            A2 += b.imag() * b.imag();
            A2 += (-arg.c * arg.c);
            Cholesky<HMatrix, real, nColor * Arg::nSpin / 2> cholesky(A2);
            chi = cholesky.backward(cholesky.forward(A_chi));
            tmp += static_cast<real>(0.25) * chi.chiral_reconstruct(chirality);
          }
          else {
            HMat A2inv = arg.A2inv(clover_coord.x_cb, parity, chirality);
            chi = A2inv * A_chi;
            tmp += static_cast<real>(2.0) * chi.chiral_reconstruct(chirality);
          }
        }

        tmp.toNonRel(); // switch back to non-chiral basis

        if (xpay) {
          Vector x = arg.x(coord.x_cb, my_spinor_parity);
          out = x + arg.a * tmp;
        } else {
          // multiplication with a needed here?
          out = arg.a * tmp;
        }
      }

      if (x_cb == 0) {
        out.print();
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(my_flavor_idx, my_spinor_parity) = out;
    }
  };
} // namespace quda

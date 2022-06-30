#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_domain_wall_4d.cuh>

/**
   This is the gauged domain-wall 4-d preconditioned operator.

   Note, for now, this just applies a batched 4-d dslash across the fifth
   dimension.
*/

namespace quda
{
  static constexpr int num_buckets = 4;
  using array_t = std::array<int, num_buckets>;

  static int powi(int base, int power) {
    int prod = 1;
    for (int p = 0; p < power; p++) {
      prod *= base;
    }
    return prod;
  }

  static int encode(const std::vector<bool> &v) {
    int s = 0;
    for (size_t i = 0; i < v.size(); i++) {
      if (v[i]) {
        // 1
        s = s * 2 + 1;
      } else {
        s = s * 2;
      }
    }
    return s;
  }

  static auto decode(int code, int num_two) {
    std::vector<bool> v(num_buckets + num_two - 1);
    for (int i = v.size() - 1; i >= 0; i--) {
      v[i] = code % 2;
      code /= 2;
    }
    return v;
  }

  static auto initialize_dist(int num_two) {
    int num_divider = num_buckets - 1;
    std::vector<bool> v(num_two + num_divider);

    // So we want num_divider of 1's, and num_two of 0's
    for (size_t i = 0; i < num_two; i++) {
      v[i] = 0;
    }
    for (size_t i = num_two; i < v.size(); i++) {
      v[i] = 1;
    }
    return v;
  }

  static auto get_dist(const std::vector<bool> &v, int num_two) {
    std::vector<int> p(num_buckets);
    for (int d = 0; d < num_buckets; d++) {
      p[d] = 1;
    }
    int d = 0;
    for (size_t i = 0; i < v.size(); i++) {
      if (v[i]) {
        // 1
        d++;
      } else {
        // 0
        p[d] *= 2;
      }
    }
    return p;
  }

  static int count_two(int in) {
    int count = 0;
    while (in % 2 == 0 && in > 0) {
      count++;
      in /= 2;
    }
    return count;
  }


  template <typename Arg> class DomainWall4D : public Dslash<domainWall4D, Arg>
  {
    using Dslash = Dslash<domainWall4D, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    DomainWall4D(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      TunableKernel3D::resizeVector(in.X(4), arg.nParity);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.kernel_type == INTERIOR_KERNEL) {
        // offset = (4, 2, 2, 2)
        int num_two = count_two(tp.block.x * 2 / 2);
        auto p = get_dist(decode(tp.aux.z, num_two), num_two);
        p[0] *= 2;
        p[1] *= 1;
        p[2] *= 1;
        p[3] *= 1;

        printf("Launching p: %d, %d, %d, %d\n", p[0], p[1], p[2], p[3]);
        arg.tb.X0h = p[0] / 2;
        for (int d = 0; d < 4; d++) {
          arg.tb.dim[d] = p[d];
        }
        arg.tb.X1 = p[0];
        arg.tb.X2X1 = p[1] * p[0];
        arg.tb.X3X2X1 = p[2] * p[1] * p[0];

        arg.tb.X2X1mX1 = (p[1] - 1) * p[0];
        arg.tb.X3X2X1mX2X1 = (p[2] - 1) * p[1] * p[0];
        arg.tb.X4X3X2X1mX3X2X1 = (p[3] - 1) * p[2] * p[1] * p[0];
      }
      Dslash::template instantiate<packShmem>(tp, stream);
    }

    virtual unsigned int sharedBytesPerThread() const
    {
      if (arg.kernel_type == INTERIOR_KERNEL) {
        return 4 * 4 * 3 * 2;
      } else {
        return 0;
      }
    }

    bool if_valid_thread_block(std::vector<int> p) const {
      p[0] *= 2;
      p[1] *= 1;
      p[2] *= 1;
      p[3] *= 1;
      return arg.dim[0] % p[0] == 0 && arg.dim[1] % p[1] == 0 && arg.dim[2] % p[2] == 0 && arg.dim[3] % p[3] == 0;
    }

    bool find_valid_z(std::vector<bool> &v, int num_two) const {
      bool found_valid_z = false;
      auto p = get_dist(v, num_two);
      if (if_valid_thread_block(p)) {
        found_valid_z = true;
      } else {
        while (std::next_permutation(v.begin(), v.end())) {
          auto p = get_dist(v, num_two);
          if (if_valid_thread_block(p)) {
            // Valid config
            found_valid_z = true;
            break;
          }
        }
      }
      return found_valid_z;
    }

    int _aux;

    virtual bool moveBlockDimStep(TuneParam &param) const {
      param.block.x *= 2;

      int num_two = count_two(param.block.x * 2 / 2);
      auto v = initialize_dist(num_two);
      auto p = get_dist(v, num_two);

      bool found_valid_z = find_valid_z(v, num_two);
      return found_valid_z;
    }

    virtual void moveAux(TuneParam &param) const {
      int num_two = count_two(param.block.x * 2 / 2);
      auto v = initialize_dist(num_two);
      auto p = get_dist(v, num_two);

      find_valid_z(v, num_two);
      param.aux.z = encode(v);
    }

    virtual bool advanceAux(TuneParam & tp) const {
      if (arg.kernel_type != INTERIOR_KERNEL) {
        return Dslash::advanceAux(tp);
      }
      if (Dslash::advanceAux(tp)) {
        return true;
      } else {
        int num_two = count_two(tp.block.x * 2 / 2);
        auto v = decode(tp.aux.z, num_two);
        while (std::next_permutation(v.begin(), v.end())) {
          auto p = get_dist(v, num_two);
          if (if_valid_thread_block(p)) {
            // Valid config
            tp.aux.z = encode(v);
            return true;
          }
        }
        // No valid config available
        find_valid_z(v, num_two);
        tp.aux.z = encode(v);
        return false;
      }
    }

    virtual void initTuneParam(TuneParam &param) const {
      Dslash::initTuneParam(param);
      if (arg.kernel_type == INTERIOR_KERNEL) {
        int num_two = count_two(param.block.x * 2 / 2);
        auto v = initialize_dist(num_two);
        bool found_valid_z = find_valid_z(v, num_two);
        if (found_valid_z) {
          param.aux.z = encode(v);
        } else {
          errorQuda("No valid configuration available ...");
        }
      }
    }

    virtual void defaultTuneParam(TuneParam &param) const {
      initTuneParam(param);
    }

    virtual int blockStep() const {
      if (arg.kernel_type == INTERIOR_KERNEL) {
        return 16;
      } else {
        return Dslash::blockStep();
      }
    }

    virtual int blockMin() const {
      if (arg.kernel_type == INTERIOR_KERNEL) {
        return 16;
      } else {
        return Dslash::blockMin();
      }
    }

    unsigned int maxBlockSize(const TuneParam &tp) const {
      if (arg.kernel_type == INTERIOR_KERNEL) {
        return 1024;
      } else {
        return Dslash::maxBlockSize(tp);
      }
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApply {

    inline DomainWall4DApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                             double m_5, const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity,
                             bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      DomainWall4DArg<Float, nColor, nDim, recon> arg(out, in, U, a, m_5, b_5, c_5, a != 0.0, x, parity, dagger,
                                                      comm_override);
      DomainWall4D<decltype(arg)> dwf(arg, out, in);

      dslash::DslashPolicyTune<decltype(dwf)> policy(dwf, in, in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
#if defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_NDEG_TWISTED_CLOVER_DIRAC)
  void ApplyDomainWall4D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                         const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    instantiate<DomainWall4DApply>(out, in, U, a, m_5, b_5, c_5, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyDomainWall4D(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double, double,
                         const Complex *, const Complex *, const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Domain-wall dslash has not been built");
  }
#endif // GPU_DOMAIN_WALL_DIRAC

} // namespace quda

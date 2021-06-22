#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <linalg.cuh>
#include <kernels/gauge_utils.cuh>
#include <kernel.h>

namespace quda
{
  template <typename Float, int nColor_, QudaReconstructType recon_>
  struct GaugeTransformArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    //static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;
    
    const Gauge transformation;
    Gauge gauge;
    
    int X[4];    // grid dimensions
    int border[4];
    
    GaugeTransformArg(GaugeField &gauge, const GaugeField &transformation) :
      kernel_param(dim3(gauge.VolumeCB(), 2, 4)),
      transformation(transformation),
      gauge(gauge)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = gauge.R()[dir];
        X[dir] = gauge.X()[dir] - border[dir] * 2;
        //this->threads.x *= X[dir];
      }
      //this->threads.x /= 2;
    }
  };

  template <typename Arg> struct Transform
  {
    const Arg &arg;
    constexpr Transform(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, Arg::nColor>;

      // Compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      int dx[4] = {0,0,0,0};

      // Get link
      Link U = arg.gauge(dir, linkIndexShift(x, dx, X), parity);
      
      // Get transformation matrices
      Link G  = arg.transformation(0, linkIndexShift(x, dx, X), parity);
      dx[dir]++;
      Link Gp = arg.transformation(0, linkIndexShift(x, dx, X), 1 - parity);
      dx[dir]--;
      
      // Apply transform
      U = G * U * conj(Gp);
      
      // Save result
      arg.gauge(dir, linkIndexShift(x, dx, X), parity) = U;      
    }
  };

} // namespace quda

#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <fast_intdiv.h>
#include <kernel.h>

namespace quda {

  template <typename Float, QudaReconstructType recon_>
  struct GaugeFixUnPackArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr QudaReconstructType recon = recon_;
    int_fastdiv X[4]; // grid dimensions
    int_fastdiv Xh[4]; // grid dimensions
    using Gauge = typename gauge_mapper<Float, recon>::type;
    Gauge U;
    complex<real> *array;
    int parity;
    int face;
    int dir;
    int borderid;
    bool pack;
    GaugeFixUnPackArg(GaugeField &U) :
      U(U)
    {
      for (int dir = 0; dir < 4; dir++) {
        X[dir] = U.X()[dir];
        Xh[dir] = X[dir] / 2;
      }
    }
  };

  template <typename Arg> struct unpacker {
    const Arg &arg;
    constexpr unpacker(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int idx)
    {
      auto &X = arg.X;
      auto &Xh = arg.Xh;
      int x[4];
      int za, xodd;
      switch ( arg.face ) {
      case 0: //X FACE
        za = idx / Xh[1];
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[0] = arg.borderid;
        xodd = (arg.borderid + x[2] + x[3] + arg.parity) & 1;
        x[1] = (2 * idx + xodd)  - za * X[1];
        break;
      case 1: //Y FACE
        za = idx / Xh[0];
        x[3] = za / X[2];
        x[2] = za - x[3] * X[2];
        x[1] = arg.borderid;
        xodd = (arg.borderid + x[2] + x[3] + arg.parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      case 2: //Z FACE
        za = idx / Xh[0];
        x[3] = za / X[1];
        x[1] = za - x[3] * X[1];
        x[2] = arg.borderid;
        xodd = (arg.borderid + x[1] + x[3] + arg.parity) & 1;
        x[0] = (2 * idx + xodd)  - za * X[0];
        break;
      case 3: //T FACE
        za = idx / Xh[0];
        x[2] = za / X[1];
        x[1] = za - x[2] * X[1];
        x[3] = arg.borderid;
        xodd = (arg.borderid + x[1] + x[2] + arg.parity) & 1;
        x[0] = (2 * idx + xodd) - za * X[0];
        break;
      }

      int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      using real = typename mapper<typename Arg::real>::type;
      real tmp[Arg::recon];
      complex<real> data[9];

      if (arg.pack) {
        arg.U.load(data, id, arg.dir, arg.parity);
        arg.U.reconstruct.Pack(tmp, data);
        for (int i = 0; i < Arg::recon / 2; i++)
          arg.array[idx + arg.threads.x * i] = complex<real>(tmp[2*i+0], tmp[2*i+1]);
      } else {
        for (int i = 0; i < Arg::recon / 2; i++) {
          tmp[2*i+0] = arg.array[idx + arg.threads.x * i].real();
          tmp[2*i+1] = arg.array[idx + arg.threads.x * i].imag();
        }
        arg.U.reconstruct.Unpack(data, tmp, id, arg.dir, 0, arg.U.X, arg.U.R);
        arg.U.save(data, id, arg.dir, arg.parity);
      }
    }
  };

}

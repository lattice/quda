#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/spinor_chiral_project.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int Nc> class SpinorChiralReconstruct : TunableKernel2D
  {
    ColorSpinorField &out;
    const ColorSpinorField &in_left;
    const ColorSpinorField &in_right;
    const QudaChirality chirality;
    unsigned int minThreads() const { return out.VolumeCB(); }

  public:
    SpinorChiralReconstruct(ColorSpinorField &out, const ColorSpinorField &in_left, const ColorSpinorField &in_right,
                            QudaChirality chirality) :
      TunableKernel2D(out, out.SiteSubset()), out(out), in_left(in_left), in_right(in_right), chirality(chirality)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (chirality == QUDA_INVALID_CHIRALITY) {
        ChiralReconstructSpinorArg<Float, Nc, QUDA_INVALID_CHIRALITY> arg(out, in_left, in_right);
        launch<ChiralReconstructSpinor>(tp, stream, arg);
      } else if (chirality == QUDA_LEFT_CHIRALITY) {
        ChiralReconstructSpinorArg<Float, Nc, QUDA_LEFT_CHIRALITY> arg(out, in_left, in_right);
        launch<ChiralReconstructSpinor>(tp, stream, arg);
      } else if (chirality == QUDA_RIGHT_CHIRALITY) {
        ChiralReconstructSpinorArg<Float, Nc, QUDA_RIGHT_CHIRALITY> arg(out, in_left, in_right);
        launch<ChiralReconstructSpinor>(tp, stream, arg);
      } else {
        errorQuda("Unsupported chirality %d", chirality);
      }
    }

    long long bytes() const
    {
      return ((chirality != QUDA_RIGHT_CHIRALITY) ? in_left.Bytes() : 0)
        + ((chirality != QUDA_LEFT_CHIRALITY) ? in_right.Bytes() : 0) + out.Bytes();
    }
  };

  void spinorChiralReconstruct(ColorSpinorField &dst, const ColorSpinorField &src_left,
                               const ColorSpinorField &src_right, QudaChirality chirality)
  {
    checkPrecision(dst, src_left, src_right);
    checkColor(dst, src_left, src_right);

    if (dst.Nspin() != 4 || src_left.Nspin() != 2 || src_right.Nspin() != 2) {
      errorQuda("Unsupported nspin combination: dst=%d, src_left=%d, src_right=%d\n", dst.Nspin(), src_left.Nspin(),
                src_right.Nspin());
    }
    if (dst.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS || src_left.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS
        || src_right.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
      errorQuda("Unsupported gamma basis combination: dst_left %d, dst_right %d, src %d\n", dst.GammaBasis(),
                src_left.GammaBasis(), src_right.GammaBasis());
    }

    if (dst.Ncolor() == 3) {
      if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
        SpinorChiralReconstruct<double, 3>(dst, src_left, src_right, chirality);
      } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
        SpinorChiralReconstruct<float, 3>(dst, src_left, src_right, chirality);
      } else {
        errorQuda("Precision %d not implemented", dst.Precision());
      }
    } else {
      errorQuda("nColor=%d not implemented", dst.Ncolor());
    }
  }

  void spinorChiralReconstruct(ColorSpinorField &dst, const ColorSpinorField &src, QudaChirality chirality)
  {
    spinorChiralReconstruct(dst, src, src, chirality);
  }

  void spinorChiralReconstruct(ColorSpinorField &dst, const ColorSpinorField &src_left, const ColorSpinorField &src_right)
  {
    spinorChiralReconstruct(dst, src_left, src_right, QUDA_INVALID_CHIRALITY);
  }

  template <typename Float, int Nc> class SpinorChiralProject : TunableKernel2D
  {
    ColorSpinorField &out_left;
    ColorSpinorField &out_right;
    const ColorSpinorField &in;
    const QudaChirality chirality;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    SpinorChiralProject(ColorSpinorField &out_left, ColorSpinorField &out_right, const ColorSpinorField &in,
                        QudaChirality chirality) :
      TunableKernel2D(in, in.SiteSubset()), out_left(out_left), out_right(out_right), in(in), chirality(chirality)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (chirality == QUDA_INVALID_CHIRALITY) {
        ChiralProjectSpinorArg<Float, Nc, QUDA_INVALID_CHIRALITY> arg(out_left, out_right, in);
        launch<ChiralProjectSpinor>(tp, stream, arg);
      } else if (chirality == QUDA_LEFT_CHIRALITY) {
        ChiralProjectSpinorArg<Float, Nc, QUDA_LEFT_CHIRALITY> arg(out_left, out_right, in);
        launch<ChiralProjectSpinor>(tp, stream, arg);
      } else if (chirality == QUDA_RIGHT_CHIRALITY) {
        ChiralProjectSpinorArg<Float, Nc, QUDA_RIGHT_CHIRALITY> arg(out_left, out_right, in);
        launch<ChiralProjectSpinor>(tp, stream, arg);
      } else {
        errorQuda("Unsupported chirality %d", chirality);
      }
    }

    long long bytes() const
    {
      return in.Bytes() + ((chirality != QUDA_RIGHT_CHIRALITY) ? out_left.Bytes() : 0)
        + ((chirality != QUDA_LEFT_CHIRALITY) ? out_right.Bytes() : 0);
    }
  };

  void spinorChiralProject(ColorSpinorField &dst_left, ColorSpinorField &dst_right, const ColorSpinorField &src,
                           QudaChirality chirality)
  {
    checkPrecision(dst_left, dst_right, src);
    checkColor(dst_left, dst_right, src);

    if (dst_left.Nspin() != 2 || dst_right.Nspin() != 2 || src.Nspin() != 4) {
      errorQuda("Unsupported nspin combination: dst_left=%d, dst_right=%d, src=%d\n", dst_left.Nspin(),
                dst_right.Nspin(), src.Nspin());
    }
    if (dst_left.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS
        || dst_right.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || src.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS) {
      errorQuda("Unsupported gamma basis combination: dst_left %d, dst_right %d, src %d\n", dst_left.GammaBasis(),
                dst_right.GammaBasis(), src.GammaBasis());
    }

    if (src.Ncolor() == 3) {
      if (src.Precision() == QUDA_DOUBLE_PRECISION) {
        SpinorChiralProject<double, 3>(dst_left, dst_right, src, chirality);
      } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
        SpinorChiralProject<float, 3>(dst_left, dst_right, src, chirality);
      } else {
        errorQuda("Precision %d not implemented", src.Precision());
      }
    } else {
      errorQuda("nColor=%d not implemented", src.Ncolor());
    }
  }

  void spinorChiralProject(ColorSpinorField &dst, const ColorSpinorField &src, QudaChirality chirality)
  {
    spinorChiralProject(dst, dst, src, chirality);
  }

  void spinorChiralProject(ColorSpinorField &dst_left, ColorSpinorField &dst_right, const ColorSpinorField &src)
  {
    spinorChiralProject(dst_left, dst_right, src, QUDA_INVALID_CHIRALITY);
  }

} // namespace quda

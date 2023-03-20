#include <tunable_nd.h>
#include <kernels/copy_gauge.cuh>

namespace quda
{

  using namespace gauge;

  template <typename Arg> class CopyGauge : TunableKernel3D
  {
    Arg &arg;
    int size;
    QudaFieldLocation location;
    bool is_ghost;
    GaugeField &out;
    const GaugeField &in;

    unsigned int minThreads() const override { return size; }

  public:
    CopyGauge(Arg &arg, GaugeField &out, const GaugeField &in, QudaFieldLocation location) :
      TunableKernel3D(in, Arg::fine_grain ? in.Ncolor() : 1, in.Geometry() * 2, location),
      arg(arg),
      location(location),
      is_ghost(false),
      out(out),
      in(in)
    {
      set_ghost(is_ghost); // initial state is not ghost
      strcat(aux, ",");
      strcat(aux, out.AuxString().c_str());
      if (Arg::fine_grain) strcat(aux, ",fine-grained");
    }

    void set_ghost(int is_ghost_)
    {
      is_ghost = is_ghost_;
      if (is_ghost_ == 2) arg.out_offset = in.Ndim(); // forward links

      int face_max = 0;
      for (int d = 0; d < in.Ndim(); d++) face_max = std::max(in.SurfaceCB(d), face_max);
      size = is_ghost ? in.Nface() * face_max : in.VolumeCB();
      if (size == 0 && is_ghost) {
        errorQuda("Cannot copy zero-sized ghost zone.  Check nFace parameter is non-zero for both input and output "
                  "gauge fields");
      }

      resizeVector(vector_length_y, (is_ghost ? in.Ndim() : in.Geometry()) * 2); // only resizing z component
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      arg.threads.x = size;
      constexpr bool enable_host = true;
      if (!is_ghost)
        launch<CopyGauge_, enable_host>(tp, stream, arg);
      else
        launch<CopyGhost_, enable_host>(tp, stream, arg);
    }

    TuneKey tuneKey() const override
    {
      char aux_[TuneKey::aux_n];
      strcpy(aux_, aux);
      if (is_ghost) strcat(aux_, ",ghost");
      return TuneKey(in.VolString().c_str(), typeid(*this).name(), aux_);
    }

    long long bytes() const override
    {
      auto sites = 4 * in.VolumeCB();
      if (is_ghost) {
        sites = 0;
        for (int d = 0; d < 4; d++) sites += in.SurfaceCB(d) * in.Nface();
      }
      return sites * (out.Bytes() + in.Bytes()) / (4 * in.VolumeCB());
    }
  };

  template <typename FloatOut, typename FloatIn, int length, bool fine_grain, typename OutOrder, typename InOrder>
  void copyGauge(OutOrder &&outOrder, const InOrder &inOrder, GaugeField &out, const GaugeField &in,
                 QudaFieldLocation location, int type)
  {
    CopyGaugeArg<FloatOut, FloatIn, length, fine_grain, OutOrder, InOrder> arg(outOrder, inOrder, in);
    CopyGauge<decltype(arg)> gaugeCopier(arg, out, in, location);

#ifdef HOST_DEBUG
    if (location == QUDA_CPU_FIELD_LOCATION) checkNan(arg);
#endif

    // first copy body
    if (type == 0 || type == 2) {
      gaugeCopier.set_ghost(0);
      gaugeCopier.apply(device::get_default_stream());
    }

#ifdef MULTI_GPU
    if (type == 0 || type == 1) {
      if (in.Geometry() == QUDA_VECTOR_GEOMETRY || in.Geometry() == QUDA_COARSE_GEOMETRY) {
        // now copy ghost
        gaugeCopier.set_ghost(1);
        gaugeCopier.apply(device::get_default_stream());
      } else {
        warningQuda("Cannot copy for %d geometry gauge field", in.Geometry());
      }
    }

    // special copy that only copies the second set of links in the
    // ghost zone for bi-directional link fields - at present this is
    // only used in cudaGaugefield::exchangeGhost where we copy from
    // the buffer into the field's ghost zone (padded
    // region), so we only have the offset on the receiver
    if (type == 3) {
      if (in.Geometry() != QUDA_COARSE_GEOMETRY)
        errorQuda("Cannot request copy type %d on non-coarse link fields", in.Geometry());
      gaugeCopier.set_ghost(2);
      gaugeCopier.apply(device::get_default_stream());
    }
#endif
  }

} // namespace quda

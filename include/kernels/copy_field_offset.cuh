#include <lattice_field.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <comm_key.h>
#include <kernel.h>

namespace quda
{

  template <class Field_, class Element_, class Accessor_, QudaPCType pc_type_ = QUDA_4D_PC>
  struct CopyFieldOffsetArg : kernel_param<> {

    static constexpr int nDim = 4; // No matter what the underlying field is, the dimension is 4
    static constexpr QudaPCType pc_type = pc_type_;

    using Field = Field_;
    using Element = Element_;
    using Accessor = Accessor_;

    Accessor out;      // output field
    const Accessor in; // input field

    int_fastdiv X0h_in;
    int_fastdiv X0h_out;
    int_fastdiv dim_in[nDim];  // full lattice dimensions
    int_fastdiv dim_out[nDim]; // full lattice dimensions

    CommKey offset;

    int nParity;

    int volume_cb_in;
    int volume_4d_cb_in;

    int volume_cb_out;
    int volume_4d_cb_out;

    int volume_cb;
    int volume_4d_cb;

    int Ls; // The fifth dimension size

    QudaOffsetCopyMode mode;

    CopyFieldOffsetArg(Accessor &out_accessor, Field &out_field, const Accessor &in_accessor, const Field &in_field,
                       CommKey offset) :
      out(out_accessor),
      in(in_accessor),
      offset(offset),
      nParity(in_field.SiteSubset())
    {
      auto X_in = in_field.X();
      auto X_out = out_field.X();

      Ls = in_field.Ndim() == 4 ? 1 : X_in[4];

      if (Ls > 1 && X_out[4] != Ls) { errorQuda("Ls mismatch: in: %d, out: %d", X_out[4], Ls); }

      if ((offset[0] + offset[1] + offset[2] + offset[3]) % 2 == 1) {
        // This offset would change the parity between the input and output fields.
        errorQuda("Offset (%d,%d,%d,%d) not supported", offset[0], offset[1], offset[2], offset[3]);
      }

      for (int d = 0; d < nDim; d++) {
        dim_out[d] = out_field.full_dim(d);
        dim_in[d] = in_field.full_dim(d);
      }

      X0h_out = dim_out[0] / 2;
      X0h_in = dim_in[0] / 2;

      volume_cb_in = in_field.VolumeCB();
      volume_cb_out = out_field.VolumeCB();

      if (volume_cb_out > volume_cb_in) {
        volume_cb = volume_cb_in;
        mode = QudaOffsetCopyMode::COLLECT;
      } else {
        volume_cb = volume_cb_out;
        mode = QudaOffsetCopyMode::DISPERSE;
      }

      volume_4d_cb_in = volume_cb_in / Ls;
      volume_4d_cb_out = volume_cb_out / Ls;
      volume_4d_cb = volume_cb / Ls;

      this->threads = dim3(volume_4d_cb, Ls, nParity);
    }
  };

  template <class Arg>
  __device__ __host__ inline
    typename std::enable_if<std::is_same<typename Arg::Field, ColorSpinorField>::value, void>::type
    copy_field(int out, int in, int parity, const Arg &arg)
  {
    using Element = typename Arg::Element;

    Element element = arg.in(in, parity);
    arg.out(out, parity) = element;
  }

  template <class Arg>
  __device__ __host__ inline typename std::enable_if<std::is_same<typename Arg::Field, CloverField>::value, void>::type
  copy_field(int out, int in, int parity, const Arg &arg)
  {
    using Element = typename Arg::Element;
    constexpr int length = 72;
    Element reg_in[length];
    Element reg_out[length];
    arg.in.load(reg_in, in, parity);
    for (int i = 0; i < length; i++) { reg_out[i] = reg_in[i]; }
    arg.out.save(reg_out, out, parity);
  }

  template <class Arg>
  __device__ __host__ inline typename std::enable_if<std::is_same<typename Arg::Field, GaugeField>::value, void>::type
  copy_field(int out, int in, int parity, const Arg &arg)
  {
    using Element = typename Arg::Element;
#pragma unroll
    for (int d = 0; d < 4; d++) {
      Element element = arg.in(d, in, parity);
      arg.out(d, out, parity) = element;
    }
  }

  template <typename Arg> struct copy_field_offset
  {
    const Arg &arg;
    constexpr copy_field_offset(const Arg &arg) : arg(arg) { }
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int s, int parity)
    {
      // XXX: This code assumes parity if NOT changed when offset is added.
      static_assert(Arg::pc_type == QUDA_4D_PC || Arg::pc_type == QUDA_5D_PC,
                    "pc_type should either be QUDA_4D_PC or QUDA_5D_PC.");
      if (Arg::pc_type == QUDA_4D_PC) { // 4d even-odd preconditioning, works for most fermions
        int coordinate[4];
        int idx_in;
        int idx_out;
        if (arg.mode == QudaOffsetCopyMode::COLLECT) {
          // we are collecting so x_cb is the index for the input.
          idx_in = x_cb;
          getCoordsExtended(coordinate, x_cb, arg.dim_in, parity, arg.offset.array);
          idx_out = linkIndex(coordinate, arg.dim_out);
        } else {
          // we are dispersing so x_cb is the index for the output.
          idx_out = x_cb;
          getCoordsExtended(coordinate, x_cb, arg.dim_out, parity, arg.offset.array);
          idx_in = linkIndex(coordinate, arg.dim_in);
        }
        copy_field(s * arg.volume_4d_cb_out + idx_out, s * arg.volume_4d_cb_in + idx_in, parity, arg);
      } else { // 5d even-odd preconditioning, works for 5d DWF
        int coordinate[5];
        int idx_in;
        int idx_out;
        if (arg.mode == QudaOffsetCopyMode::COLLECT) {
          // we are collecting so x_cb is the index for the input.
          idx_in = x_cb + arg.volume_4d_cb_in * s;
          getCoords5CB(coordinate, idx_in, arg.dim_in, arg.X0h_in, parity, QUDA_5D_PC);
#pragma unroll
          for (int d = 0; d < 4; d++) { coordinate[d] += arg.offset[d]; }
          idx_out = linkIndex(coordinate, arg.dim_out) + arg.volume_4d_cb_out * coordinate[4];
        } else {
          // we are dispersing so x_cb is the index for the output.
          idx_out = x_cb + arg.volume_4d_cb_out * s;
          getCoords5CB(coordinate, idx_out, arg.dim_out, arg.X0h_out, parity, QUDA_5D_PC);
#pragma unroll
          for (int d = 0; d < 4; d++) { coordinate[d] += arg.offset[d]; }
          idx_in = linkIndex(coordinate, arg.dim_in) + arg.volume_4d_cb_in * coordinate[4];
        }
        copy_field(idx_out, idx_in, parity, arg);
      }
    }
  };
} // namespace quda

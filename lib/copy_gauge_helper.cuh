#include <tune_quda.h>

#include <jitify_helper.cuh>
#include <kernels/copy_gauge.cuh>

namespace quda {

  using namespace gauge;

  template <typename FloatOut, typename FloatIn, int length, typename Arg>
  class CopyGauge : TunableVectorYZ {
    Arg arg;
    int size;
    const GaugeField &meta;
    QudaFieldLocation location;
    bool is_ghost;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return size; }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we are doing the copy on the GPU
      return location == QUDA_CUDA_FIELD_LOCATION ? TunableVectorYZ::advanceTuneParam(param) : false;
    }

public:
    CopyGauge(Arg &arg, const GaugeField &out, const GaugeField &in, QudaFieldLocation location)
#ifndef FINE_GRAINED_ACCESS
      : TunableVectorYZ(1, in.Geometry()* 2),
#else
      : TunableVectorYZ(Ncolor(length), in.Geometry() * 2),
#endif
        arg(arg), meta(in), location(location), is_ghost(false) {

      set_ghost(is_ghost); // initial state is not ghost

      strcpy(aux, compile_type_str(in,location));
      strcat(aux, "out:");
      strcat(aux, out.AuxString());
      strcat(aux, ",in:");
      strcat(aux, in.AuxString());

#ifdef FINE_GRAINED_ACCESS
      strcat(aux,",fine-grained");
#endif

#ifdef JITIFY
#ifdef FINE_GRAINED_ACCESS
      std::vector<std::string> macro = { "-DFINE_GRAINED_ACCESS" }; // need to pass macro to jitify
      create_jitify_program("kernels/copy_gauge.cuh", macro);
#else
      create_jitify_program("kernels/copy_gauge.cuh");
#endif
#endif
    }

    void set_ghost(int is_ghost_) {
      is_ghost = is_ghost_;
      if (is_ghost_ == 2) arg.out_offset = meta.Ndim(); // forward links

      int faceMax = 0;
      for (int d=0; d<arg.nDim; d++) {
        faceMax = (arg.faceVolumeCB[d] > faceMax ) ? arg.faceVolumeCB[d] : faceMax;
      }
      size = is_ghost ? faceMax : arg.volume/2;
      if (size == 0 && is_ghost) {
	errorQuda("Cannot copy zero-sized ghost zone.  Check nFace parameter is non-zero for both input and output gauge fields");
      }
#ifndef FINE_GRAINED_ACCESS
      resizeVector(1, (is_ghost ? arg.nDim : meta.Geometry()) * 2);
#else
      resizeVector(Ncolor(length), (is_ghost ? arg.nDim : meta.Geometry()) * 2);
#endif
    }

    virtual ~CopyGauge() { ; }
  
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (!is_ghost) {
          copyGauge<FloatOut, FloatIn, length>(arg);
        } else {
          copyGhost<FloatOut, FloatIn, length>(arg);
        }
      } else if (location == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel(!is_ghost ? "quda::copyGaugeKernel" : "quda::copyGhostKernel")
          .instantiate(Type<FloatOut>(),Type<FloatIn>(),length,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
        if (!is_ghost) {
          copyGaugeKernel<FloatOut, FloatIn, length, Arg>
            <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        } else {
          copyGhostKernel<FloatOut, FloatIn, length, Arg>
            <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        }
#endif
      } else {
        errorQuda("Invalid field location %d\n", location);
      }
    }

    TuneKey tuneKey() const {
      char aux_[TuneKey::aux_n];
      strcpy(aux_,aux);
      if (is_ghost) strcat(aux_, ",ghost");
      return TuneKey(meta.VolString(), typeid(*this).name(), aux_);
    }

    long long flops() const { return 0; } 
    long long bytes() const {
      int sites = 4*arg.volume/2;
      if (is_ghost) {
	sites = 0;
	for (int d=0; d<4; d++) sites += arg.faceVolumeCB[d];
      }
#ifndef FINE_GRAINED_ACCESS
      return 2 * sites * (  arg.in.Bytes() + arg.in.hasPhase*sizeof(FloatIn) 
			    + arg.out.Bytes() + arg.out.hasPhase*sizeof(FloatOut) ); 
#else      
      return 2 * sites * (  arg.in.Bytes() + arg.out.Bytes() );
#endif
    } 
  };


  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void copyGauge(OutOrder &&outOrder, const InOrder &inOrder, const GaugeField &out, const GaugeField &in,
		   QudaFieldLocation location, int type) {

    CopyGaugeArg<OutOrder,InOrder> arg(outOrder, inOrder, in);
    CopyGauge<FloatOut, FloatIn, length, CopyGaugeArg<OutOrder,InOrder> > gaugeCopier(arg, out, in, location);

#ifdef HOST_DEBUG
    if (location == QUDA_CPU_FIELD_LOCATION) checkNan<FloatIn, length>(arg);
#endif

    // first copy body
    if (type == 0 || type == 2) {
      gaugeCopier.set_ghost(0);
      gaugeCopier.apply(0);
    }

#ifdef MULTI_GPU
    if (type == 0 || type == 1) {
      if (in.Geometry() == QUDA_VECTOR_GEOMETRY || in.Geometry() == QUDA_COARSE_GEOMETRY) {
        // now copy ghost
        gaugeCopier.set_ghost(1);
        gaugeCopier.apply(0);
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
      if (in.Geometry() != QUDA_COARSE_GEOMETRY) errorQuda("Cannot request copy type %d on non-coarse link fields", in.Geometry());
      gaugeCopier.set_ghost(2);
      gaugeCopier.apply(0);
    }
#endif

  }

} // namespace quda

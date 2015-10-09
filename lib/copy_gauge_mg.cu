#include <gauge_field_order.h>
#include <copy_gauge_helper.cuh>

namespace quda {
  
  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGauge(const InOrder &inOrder, GaugeField &out, QudaFieldLocation location, 
		 FloatOut *Out, FloatOut **outGhost, int type) {
    if (out.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Reconstruct type %d not supported", out.Reconstruct());

    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface(); 
    if (out.isNative()) {
      typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_NO,length>::type G;
      copyGauge<FloatOut,FloatIn,length>
	(G(out,Out,outGhost), inOrder, out.Volume(), faceVolumeCB,
	 out.Ndim(), out.Geometry(), out, location, type);
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(QDPOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(), 
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
    void copyGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, 
		   FloatOut *Out, FloatIn *In, FloatOut **outGhost, FloatIn **inGhost, int type) {
    if (in.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Reconstruct type %d not supported", in.Reconstruct());

    // reconstruction only supported on FloatN fields currently
    if (in.isNative()) {      
      typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_NO>::type G;
      copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost), out, location, Out, outGhost, type);
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(QDPOrder<FloatIn,length>(in, In, inGhost), 
					 out, location, Out, outGhost, type);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }

  }

  template <typename FloatOut, typename FloatIn>
  void copyGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location, FloatOut *Out, 
		   FloatIn *In, FloatOut **outGhost, FloatIn **inGhost, int type) {

#ifdef GPU_MULTIGRID
    if (in.Ncolor() == 4) {
      // we are doing gauge field packing
      const int Nc = 4;
      copyGauge<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else 
#endif // GPU_MULTIGRID
    {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }
  }

  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
			  void *Out, void *In, void **ghostOut, void **ghostIn, int type) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGaugeMG(out, in, location, (double*)Out, (double*)In, (double**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeMG(out, in, location, (double*)Out, (float*)In, (double**)ghostOut, (float**)ghostIn, type);
      } else {
	errorQuda("Precision %d not supported", in.Precision());
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGaugeMG(out, in, location, (float*)Out, (double*)In, (float**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeMG(out, in, location, (float*)Out, (float*)In, (float**)ghostOut, (float**)ghostIn, type);
      } else {
	errorQuda("Precision %d not supported", in.Precision());
      }
    } else {
      errorQuda("Precision %d not supported", out.Precision());
    } 
  } 



} // namespace quda

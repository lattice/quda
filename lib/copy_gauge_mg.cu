#include <gauge_field_order.h>

#define FINE_GRAINED_ACCESS

#include <copy_gauge_helper.cuh>

namespace quda {
  
  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGaugeMG(const InOrder &inOrder, GaugeField &out, const GaugeField &in,
		   QudaFieldLocation location, FloatOut *Out, FloatOut **outGhost, int type) {
    if (out.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Reconstruct type %d not supported", out.Reconstruct());

    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface(); 
    if (out.isNative()) {

#ifdef FINE_GRAINED_ACCESS
      if (outGhost) {
	typedef typename gauge::FieldOrder<FloatOut,Ncolor(length),1,QUDA_FLOAT2_GAUGE_ORDER,false> G;
	copyGauge<FloatOut,FloatIn,length>(G(out,(void*)Out,(void**)outGhost), inOrder, out.Volume(), faceVolumeCB,
					   out.Ndim(), out.Geometry(), out, in, location, type);
      } else {
	typedef typename gauge::FieldOrder<FloatOut,Ncolor(length),1,QUDA_FLOAT2_GAUGE_ORDER,true> G;
	copyGauge<FloatOut,FloatIn,length>(G(out,(void*)Out,(void**)outGhost), inOrder, out.Volume(), faceVolumeCB,
					   out.Ndim(), out.Geometry(), out, in, location, type);
      }
#else
      typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_NO,length>::type G;
      copyGauge<FloatOut,FloatIn,length>
	(G(out,Out,outGhost), inOrder, out.Volume(), faceVolumeCB,
	 out.Ndim(), out.Geometry(), out, in, location, type);
#endif

    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef FINE_GRAINED_ACCESS
      typedef typename gauge::FieldOrder<FloatOut,Ncolor(length),1,QUDA_QDP_GAUGE_ORDER> G;
      copyGauge<FloatOut,FloatIn,length>(G(out,(void*)Out,(void**)outGhost), inOrder, out.Volume(),
					 faceVolumeCB, out.Ndim(), out.Geometry(), out, in, location, type);
#else
      typedef typename QDPOrder<FloatOut,length> G;
      copyGauge<FloatOut,FloatIn,length>(G(out, Out, outGhost), inOrder, out.Volume(),
					 faceVolumeCB, out.Ndim(), out.Geometry(), out, in, location, type);
#endif

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef FINE_GRAINED_ACCESS
      typedef typename gauge::FieldOrder<FloatOut,Ncolor(length),1,QUDA_MILC_GAUGE_ORDER> G;
      copyGauge<FloatOut,FloatIn,length>(G(out,(void*)Out,(void**)outGhost), inOrder, out.Volume(),
					 faceVolumeCB, out.Ndim(), out.Geometry(), out, in, location, type);
#else
      typedef typename MILCOrder<FloatOut,length> G;
      copyGauge<FloatOut,FloatIn,length>(G(out, Out, outGhost), inOrder, out.Volume(),
					 faceVolumeCB, out.Ndim(), out.Geometry(), out, in, location, type);
#endif

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
    void copyGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
		     FloatOut *Out, FloatIn *In, FloatOut **outGhost, FloatIn **inGhost, int type) {

    if (in.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Reconstruct type %d not supported", in.Reconstruct());

    // reconstruction only supported on FloatN fields currently
    if (in.isNative()) {      
#ifdef FINE_GRAINED_ACCESS
      if (inGhost) {
	typedef typename gauge::FieldOrder<FloatIn,Ncolor(length),1,QUDA_FLOAT2_GAUGE_ORDER,false> G;
	copyGaugeMG<FloatOut,FloatIn,length> (G(const_cast<GaugeField&>(in),(void*)In,(void**)inGhost), out, in, location, Out, outGhost, type);
      } else {
	typedef typename gauge::FieldOrder<FloatIn,Ncolor(length),1,QUDA_FLOAT2_GAUGE_ORDER,true> G;
	copyGaugeMG<FloatOut,FloatIn,length> (G(const_cast<GaugeField&>(in),(void*)In,(void**)inGhost), out, in, location, Out, outGhost, type);
      }
#else
      typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_NO,length>::type G;
      copyGaugeMG<FloatOut,FloatIn,length> (G(in, In,inGhost), out, in, location, Out, outGhost, type);
#endif
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef FINE_GRAINED_ACCESS
      typedef typename gauge::FieldOrder<FloatIn,Ncolor(length),1,QUDA_QDP_GAUGE_ORDER> G;
      copyGaugeMG<FloatOut,FloatIn,length>(G(const_cast<GaugeField&>(in),(void*)In,(void**)inGhost), out, in, location, Out, outGhost, type);
#else
      typedef typename QDPOrder<FloatIn,length> G;
      copyGaugeMG<FloatOut,FloatIn,length>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
#endif

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef FINE_GRAINED_ACCESS
      typedef typename gauge::FieldOrder<FloatIn,Ncolor(length),1,QUDA_MILC_GAUGE_ORDER> G;
      copyGaugeMG<FloatOut,FloatIn,length>(G(const_cast<GaugeField&>(in),(void*)In,(void**)inGhost), out, in, location, Out, outGhost, type);
#else
      typedef typename MILCOrder<FloatIn,length> G;
      copyGaugeMG<FloatOut,FloatIn,length>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
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
      const int Nc = 4;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 8) {
      const int Nc = 8;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 16) {
      const int Nc = 16;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 24) {
      const int Nc = 24;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 32) {
      const int Nc = 32;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 40) {
      const int Nc = 40;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 48) {
      const int Nc = 48;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 56) {
      const int Nc = 56;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
    } else  if (in.Ncolor() == 64) {
      const int Nc = 64;
      copyGaugeMG<FloatOut,FloatIn,2*Nc*Nc>(out, in, location, Out, In, outGhost, inGhost, type);
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
#ifdef GPU_MULTIGRID_DOUBLE
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGaugeMG(out, in, location, (double*)Out, (double*)In, (double**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeMG(out, in, location, (double*)Out, (float*)In, (double**)ghostOut, (float**)ghostIn, type);
      } else {
	errorQuda("Precision %d not supported", in.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
	copyGaugeMG(out, in, location, (float*)Out, (double*)In, (float**)ghostOut, (double**)ghostIn, type);
#else
	errorQuda("Double precision multigrid has not been enabled");
#endif
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

#include <gauge_field.h>
#include <typeinfo>

namespace quda {

  GaugeField::GaugeField(const GaugeFieldParam &param) :
    LatticeField(param), bytes(0), phase_offset(0), phase_bytes(0), nColor(param.nColor), nFace(param.nFace),
    geometry(param.geometry), reconstruct(param.reconstruct), order(param.order), 
    fixed(param.fixed), link_type(param.link_type), t_boundary(param.t_boundary), 
    anisotropy(param.anisotropy), tadpole(param.tadpole), fat_link_max(0.0), scale(param.scale),  
    create(param.create), ghostExchange(param.ghostExchange), 
    staggeredPhaseType(param.staggeredPhaseType), staggeredPhaseApplied(param.staggeredPhaseApplied)
  {
    if (nColor != 3) errorQuda("nColor must be 3, not %d\n", nColor);
    if (nDim != 4) errorQuda("Number of dimensions must be 4 not %d", nDim);
    if (link_type != QUDA_WILSON_LINKS && anisotropy != 1.0) errorQuda("Anisotropy only supported for Wilson links");
    if (link_type != QUDA_WILSON_LINKS && fixed == QUDA_GAUGE_FIXED_YES)
      errorQuda("Temporal gauge fixing only supported for Wilson links");

    if(link_type != QUDA_ASQTAD_LONG_LINKS && (reconstruct ==  QUDA_RECONSTRUCT_13 || reconstruct == QUDA_RECONSTRUCT_9))
      errorQuda("reconstruct %d only supported for staggered long links\n", reconstruct);
       
    if (link_type == QUDA_ASQTAD_MOM_LINKS) scale = 1.0;

    if(geometry == QUDA_SCALAR_GEOMETRY) {
      real_length = volume*reconstruct;
      length = 2*stride*reconstruct; // two comes from being full lattice
    } else if (geometry == QUDA_VECTOR_GEOMETRY) {
      real_length = nDim*volume*reconstruct;
      length = 2*nDim*stride*reconstruct; // two comes from being full lattice
    } else if(geometry == QUDA_TENSOR_GEOMETRY){
      real_length = (nDim*(nDim-1)/2)*volume*reconstruct;
      length = 2*(nDim*(nDim-1)/2)*stride*reconstruct; // two comes from being full lattice
    }


    if(reconstruct == QUDA_RECONSTRUCT_9 || reconstruct == QUDA_RECONSTRUCT_13)
    {
      // Need to adjust the phase alignment as well.  
      int half_phase_bytes = (length/(2*reconstruct))*precision; // number of bytes needed to store phases for a single parity
      int half_gauge_bytes = (length/2)*precision - half_phase_bytes; // number of bytes needed to store the gauge field for a single parity excluding the phases
      // Adjust the alignments for the gauge and phase separately
      half_phase_bytes = ((half_phase_bytes + (512-1))/512)*512;
      half_gauge_bytes = ((half_gauge_bytes + (512-1))/512)*512;
    
      phase_offset = half_gauge_bytes;
      phase_bytes = half_phase_bytes*2;
      bytes = (half_gauge_bytes + half_phase_bytes)*2;      
    }else{
      bytes = length*precision;
      bytes = 2*ALIGNMENT_ADJUST(bytes/2);
    }
    total_bytes = bytes;
  }

  GaugeField::~GaugeField() {

  }

  void GaugeField::applyStaggeredPhase() {
    if (staggeredPhaseApplied) errorQuda("Staggered phases already applied");
    applyGaugePhase(*this);
    if (ghostExchange==QUDA_GHOST_EXCHANGE_PAD) {
      if (typeid(*this)==typeid(cudaGaugeField)) {
	static_cast<cudaGaugeField&>(*this).exchangeGhost();
      } else {
	static_cast<cpuGaugeField&>(*this).exchangeGhost();
      }
    }
    staggeredPhaseApplied = true;
  }

  void GaugeField::removeStaggeredPhase() {
    if (!staggeredPhaseApplied) errorQuda("No staggered phases to remove");
    applyGaugePhase(*this);
    if (ghostExchange==QUDA_GHOST_EXCHANGE_PAD) {
      if (typeid(*this)==typeid(cudaGaugeField)) {
	static_cast<cudaGaugeField&>(*this).exchangeGhost();
      } else {
	static_cast<cpuGaugeField&>(*this).exchangeGhost();
      }
    }
    staggeredPhaseApplied = false;
  }

  bool GaugeField::isNative() const {
    if (precision == QUDA_DOUBLE_PRECISION) {
      if (order  == QUDA_FLOAT2_GAUGE_ORDER) return true;
    } else if (precision == QUDA_SINGLE_PRECISION || 
	       precision == QUDA_HALF_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	if (order == QUDA_FLOAT2_GAUGE_ORDER) return true;
      } else if (reconstruct == QUDA_RECONSTRUCT_12 || reconstruct == QUDA_RECONSTRUCT_13) {
	if (order == QUDA_FLOAT4_GAUGE_ORDER) return true;
      } else if (reconstruct == QUDA_RECONSTRUCT_8 || reconstruct == QUDA_RECONSTRUCT_9) {
	if (order == QUDA_FLOAT4_GAUGE_ORDER) return true;
      } else if (reconstruct == QUDA_RECONSTRUCT_10) {
	if (order == QUDA_FLOAT2_GAUGE_ORDER) return true;
      }
    }
    return false;
  }

  void GaugeField::checkField(const GaugeField &a) {
    LatticeField::checkField(a);
    if (a.link_type != link_type) errorQuda("link_type does not match %d %d", link_type, a.link_type);
    if (a.nColor != nColor) errorQuda("nColor does not match %d %d", nColor, a.nColor);
    if (a.nFace != nFace) errorQuda("nFace does not match %d %d", nFace, a.nFace);
    if (a.fixed != fixed) errorQuda("fixed does not match %d %d", fixed, a.fixed);
    if (a.t_boundary != t_boundary) errorQuda("t_boundary does not match %d %d", t_boundary, a.t_boundary);
    if (a.anisotropy != anisotropy) errorQuda("anisotropy does not match %e %e", anisotropy, a.anisotropy);
    if (a.tadpole != tadpole) errorQuda("tadpole does not match %e %e", tadpole, a.tadpole);
    //if (a.scale != scale) errorQuda("scale does not match %e %e", scale, a.scale); 
  }

  std::ostream& operator<<(std::ostream& output, const GaugeFieldParam& param) {
    output << static_cast<const LatticeFieldParam &>(param);
    output << "nColor = " << param.nColor << std::endl;
    output << "nFace = " << param.nFace << std::endl;
    output << "reconstruct = " << param.reconstruct << std::endl;
    output << "order = " << param.order << std::endl;
    output << "fixed = " << param.fixed << std::endl;
    output << "link_type = " << param.link_type << std::endl;
    output << "t_boundary = " << param.t_boundary << std::endl;
    output << "anisotropy = " << param.anisotropy << std::endl;
    output << "tadpole = " << param.tadpole << std::endl;
    output << "scale = " << param.scale << std::endl;
    output << "create = " << param.create << std::endl;
    output << "geometry = " << param.geometry << std::endl;
    output << "ghostExchange = " << param.ghostExchange << std::endl;
    output << "staggeredPhaseType = " << param.staggeredPhaseType << std::endl;
    output << "staggeredPhaseApplied = " << param.staggeredPhaseApplied << std::endl;

    return output;  // for multiple << operators.
  }

} // namespace quda

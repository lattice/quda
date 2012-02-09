#include <gauge_field.h>

GaugeField::GaugeField(const GaugeFieldParam &param, const QudaFieldLocation &location) :
  LatticeField(param, location), bytes(0), nColor(param.nColor), nFace(param.nFace),
  reconstruct(param.reconstruct), order(param.order), fixed(param.fixed), 
  link_type(param.link_type), t_boundary(param.t_boundary), anisotropy(param.anisotropy),
  tadpole(param.tadpole), create(param.create), is_staple(param.is_staple)
{
  if (nColor != 3) errorQuda("nColor must be 3, not %d\n", nColor);
  if (nDim != 4 && nDim != 1) errorQuda("Number of dimensions must be 4 or 1, not %d", nDim);
  if (link_type != QUDA_WILSON_LINKS && anisotropy != 1.0) errorQuda("Anisotropy only supported for Wilson links");
  if (link_type != QUDA_WILSON_LINKS && fixed == QUDA_GAUGE_FIXED_YES)
    errorQuda("Temporal gauge fixing only supported for Wilson links");

  if(is_staple){
    real_length = volume*reconstruct;
    length = 2*stride*reconstruct; // two comes from being full lattice
  }else{
    real_length = 4*volume*reconstruct;
    length = 2*4*stride*reconstruct; // two comes from being full lattice
  }

  bytes = length*precision;
  bytes = ALIGNMENT_ADJUST(bytes);
  total_bytes = bytes;
}

GaugeField::~GaugeField() {

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
  output << "create = " << param.create << std::endl;

  return output;  // for multiple << operators.
}

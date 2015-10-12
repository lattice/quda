#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>

namespace quda {

  template <typename Field>
  struct PackGhostArg {

    Field field;
    void **ghost;
    const void *v;
    int X[QUDA_MAX_DIM];
    const int volumeCB;
    const int nDim;
    const int nFace;
    const int parity;
    const int dagger;
    const QudaDWFPCType pc_type;

    PackGhostArg(Field field, void **ghost, const ColorSpinorField &a, int parity, int dagger)
      : field(field),
	ghost(ghost),
	v(a.V()),
	volumeCB(a.VolumeCB()),
	nDim(a.Ndim()),
	nFace(a.Nspin() == 1 ? 3 : 1),
	parity(parity), dagger(dagger),
	pc_type(a.DWFPCtype())
    {
      for (int d=0; d<nDim; d++) X[d] = a.X(d);
      X[0] *= 2; // set to full lattice size
      X[4] = (nDim == 5) ? a.X(4) : 1; // set fifth dimension correctly
    }

  };

  template <int dir, typename Arg>
  __device__ __host__ inline int ghostFaceIndex(const int x[], int dim, Arg arg) {
    const int *X = arg.X;
    int index;
    switch(dim) {
    case 0:
      switch(dir) {
      case 0:
	index = (x[0]*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] + x[1])>>1;
	break;
      case 1:
	index = ((x[0]-X[0]+arg.nFace)*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1]) + x[2]*X[1] + x[1])>>1;
	break;
      }
      break;
    case 1:
      switch(dir) {
      case 0:
	index = (x[1]*X[4]*X[3]*X[2]*X[0] + x[4]*X[3]*X[2]*X[0] + x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1;
	break;
      case 1:
	index = ((x[1]-X[1]+arg.nFace)*X[4]*X[3]*X[2]*X[0] +x[4]*X[3]*X[2]*X[0]+ x[3]*X[2]*X[0] + x[2]*X[0] + x[0])>>1;
	break;
      }
      break;
    case 2:
      switch(dir) {
      case 0:
	index = (x[2]*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
	break;
      case 1:
	index = ((x[2]-X[2]+arg.nFace)*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0] + x[1]*X[0] + x[0])>>1;
	break;
      }
      break;
    case 3:
      switch(dir) {
      case 0:
	index = (x[3]*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
	break;
      case 1:
	index  = ((x[3]-X[3]+arg.nFace)*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0] + x[0])>>1;
	break;
      }
      break;
    }
    return index;
  }

  template <typename Float, int Ns, int Nc, typename Arg>
  __host__ void packGhost(Arg &arg, int cb_idx) {
    typedef typename mapper<Float>::type RegType;
    const int spinor_size = 2*Ns*Nc*sizeof(Float);

    const int *X = arg.X;
    int x[5] = { };
    if (arg.nDim == 5)  getCoords5(x, cb_idx, X, arg.parity, arg.pc_type);
    else getCoords(x, cb_idx, X, arg.parity);

    const void *v = arg.v;
    void **ghost = arg.ghost;
    RegType tmp[2*Ns*Nc];

#pragma unroll
    for (int dim; dim<4; dim++) {
      if (x[dim] < arg.nFace){
	arg.field.load(tmp, cb_idx);
	arg.field.saveGhost(tmp, ghostFaceIndex<0>(x,dim,arg), dim, 0);
      }
      
      if (x[dim] >= X[dim] - arg.nFace){
	arg.field.load(tmp, cb_idx);
	arg.field.saveGhost(tmp, ghostFaceIndex<1>(x,dim,arg), dim, 1);
      }
    }
  }

  template <typename Float, int Ns, int Nc, typename Arg>
  void GenericPackGhost(Arg &arg) {
    for (int i=0; i<arg.volumeCB; i++) packGhost<Float,Ns,Nc>(arg, i);
  }

  template <typename Float, QudaFieldOrder order, int Ns, int Nc>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    typedef typename colorspinor_order_mapper<Float,order,Ns,Nc>::type Q;
    Q field(a, (Float*)0, (float*)0, (Float**)ghost);
    PackGhostArg<Q> arg(field, ghost, a, parity, dagger);
    
    if (a.Ncolor() == 3) {
      GenericPackGhost<Float,Ns,3>(arg);
    } else {
      errorQuda("Unsupported nColor = %d", a.Ncolor());
    }

  }

  template <typename Float, QudaFieldOrder order, int Ns>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {
    
    if (a.Ncolor() == 3) {
      genericPackGhost<Float,order,Ns,3>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported nColor = %d", a.Ncolor());
    }

  }

  template <typename Float, QudaFieldOrder order>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.Nspin() == 4) {
      genericPackGhost<Float,order,4>(ghost, a, parity, dagger);
    } else if (a.Nspin() == 2) {
      genericPackGhost<Float,order,2>(ghost, a, parity, dagger);
    } else if (a.Nspin() == 1) {
      genericPackGhost<Float,order,1>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported nSpin = %d", a.Nspin());
    }

  }

  template <typename Float>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      genericPackGhost<Float,QUDA_FLOAT2_FIELD_ORDER>(ghost, a, parity, dagger);
    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericPackGhost<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported field order = %d", a.FieldOrder());
    }

  }

  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.SiteSubset() == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in packGhost for cpu");
    }
    if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      errorQuda("Field order %d not supported", a.FieldOrder());
    }

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      genericPackGhost<double>(ghost, a, parity, dagger);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      genericPackGhost<float>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported precision %d", a.Precision());
    }

  }

} // namespace quda

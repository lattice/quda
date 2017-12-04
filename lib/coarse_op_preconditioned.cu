#include <gauge_field.h>
#include <gauge_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <blas_cublas.h>

namespace quda {

  template <typename PreconditionedGauge, typename Gauge, int n>
  struct CalculateYhatArg {
    PreconditionedGauge Yhat;
    const Gauge Y;
    const Gauge Xinv;
    int dim[QUDA_MAX_DIM];
    int comm_dim[QUDA_MAX_DIM];
    int nFace;
    const int coarseVolumeCB;   /** Coarse grid volume */

    CalculateYhatArg(const PreconditionedGauge &Yhat, const Gauge Y, const Gauge Xinv, const int *dim, const int *comm_dim, int nFace)
      : Yhat(Yhat), Y(Y), Xinv(Xinv), nFace(nFace), coarseVolumeCB(Y.VolumeCB()) {
      for (int i=0; i<4; i++) {
	this->comm_dim[i] = comm_dim[i];
	this->dim[i] = dim[i];
      }
    }
  };

  template<typename Float, int n, typename Arg>
  __device__ __host__ void computeYhat(Arg &arg, int d, int x_cb, int parity, int i) {

    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
    coord[4] = 0;

    const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

    // first do the backwards links Y^{+\mu} * X^{-\dagger}
    if ( arg.comm_dim[d] && (coord[d] - arg.nFace < 0) ) {

      for(int j = 0; j<n; j++) {
	complex<Float> yHat = 0.0;
	for(int k = 0; k<n; k++) {
	  yHat += arg.Y.Ghost(d,1-parity,ghost_idx,i,k) * conj(arg.Xinv(0,parity,x_cb,j,k));
	}
	arg.Yhat.Ghost(d,1-parity,ghost_idx,i,j) = yHat;
      }

    } else {
      const int back_idx = linkIndexM1(coord, arg.dim, d);

      for(int j = 0; j<n; j++) {
	complex<Float> yHat = 0.0;
	for(int k = 0; k<n; k++) {
	  yHat += arg.Y(d,1-parity,back_idx,i,k) * conj(arg.Xinv(0,parity,x_cb,j,k));
	}
	arg.Yhat(d,1-parity,back_idx,i,j) = yHat;
      }

    }

    // now do the forwards links X^{-1} * Y^{-\mu}
    for(int j = 0; j<n; j++) {
      complex<Float> yHat = 0.0;
      for(int k = 0; k<n; k++) {
	yHat += arg.Xinv(0,parity,x_cb,i,k) * arg.Y(d+4,parity,x_cb,k,j);
      }
      arg.Yhat(d+4,parity,x_cb,i,j) = yHat;
    }

  }

  template<typename Float, int n, typename Arg>
  void CalculateYhatCPU(Arg &arg) {

    for (int d=0; d<4; d++) {
      for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
	for (int x_cb=0; x_cb<arg.Y.VolumeCB(); x_cb++) {
	  for (int i=0; i<n; i++) computeYhat<Float,n>(arg, d, x_cb, parity, i);
	} // x_cb
      } //parity
    } // dimension
  }

  template<typename Float, int n, typename Arg>
  __global__ void CalculateYhatGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int i_parity = blockDim.y*blockIdx.y + threadIdx.y;
    if (i_parity >= 2*n) return;
    int d = blockDim.z*blockIdx.z + threadIdx.z;
    if (d >= 4) return;

    int i = i_parity % n;
    int parity = i_parity / n;
    computeYhat<Float,n>(arg, d, x_cb, parity, i);
  }

  template <typename Float, int n, typename Arg>
  class CalculateYhat : public TunableVectorYZ {

  protected:
    Arg &arg;
    const LatticeField &meta;

    long long flops() const { return 2l * arg.coarseVolumeCB * 8 * n * n * (8*n-2); } // 8 from dir, 8 from complexity,
    long long bytes() const { return 2l * (arg.Xinv.Bytes() + 8*arg.Y.Bytes() + 8*arg.Yhat.Bytes()); }

    unsigned int minThreads() const { return arg.coarseVolumeCB; }

    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    CalculateYhat(Arg &arg, const LatticeField &meta) : TunableVectorYZ(2*n,4), arg(arg), meta(meta)
    {
      strcpy(aux,comm_dim_partitioned_string());
    }
    virtual ~CalculateYhat() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	CalculateYhatCPU<Float,n,Arg>(arg);
      } else {
	CalculateYhatGPU<Float,n,Arg> <<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      }
    }

    // no locality in this kernel so no point in shared-memory tuning
    bool advanceSharedBytes(TuneParam &param) const { return false; }

    bool advanceTuneParam(TuneParam &param) const {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);
      strcat(Aux,meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU" : ",CPU");
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(Aux, getOmpThreadStr());
      return TuneKey(meta.VolString(), typeid(*this).name(), Aux);
    }
  };

  /**
     @brief Calculate the preconditioned coarse-link field and the clover inverse.

     @param Yhat[out] Preconditioned coarse link field
     @param Xinv[out] Coarse clover inverse field
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
   */
  template<typename storeFloat, typename Float, int N, QudaGaugeFieldOrder gOrder>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X)
  {
    // invert the clover matrix field
    const int n = X.Ncolor();
    if (X.Location() == QUDA_CUDA_FIELD_LOCATION && X.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      GaugeFieldParam param(X);
      // need to copy into AoS format for CUBLAS
      param.order = QUDA_MILC_GAUGE_ORDER;
      cudaGaugeField X_(param);
      cudaGaugeField Xinv_(param);
      X_.copy(X);
      blas::flops += cublas::BatchInvertMatrix((void*)Xinv_.Gauge_p(), (void*)X_.Gauge_p(), n, X_.Volume(), X_.Precision(), X.Location());
      Xinv.copy(Xinv_);
    } else if (X.Location() == QUDA_CPU_FIELD_LOCATION && X.Order() == QUDA_QDP_GAUGE_ORDER) {
      const cpuGaugeField *X_h = static_cast<const cpuGaugeField*>(&X);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv);
      blas::flops += cublas::BatchInvertMatrix(((void**)Xinv_h->Gauge_p())[0], ((void**)X_h->Gauge_p())[0], n, X_h->Volume(), X.Precision(), QUDA_CPU_FIELD_LOCATION);
    } else {
      errorQuda("Unsupported location=%d and order=%d", X.Location(), X.Order());
    }

    // now exchange Y halos of both forwards and backwards links for multi-process dslash
    const_cast<GaugeField&>(Y).exchangeGhost(QUDA_LINK_BIDIRECTIONAL);

    // compute the preconditioned links
    // Yhat_back(x-\mu) = Y_back(x-\mu) * Xinv^dagger(x) (positive projector)
    // Yhat_fwd(x) = Xinv(x) * Y_fwd(x)                  (negative projector)
    {
      int xc_size[5];
      for (int i=0; i<4; i++) xc_size[i] = X.X()[i];
      xc_size[4] = 1;

      // use spin-ignorant accessor to make multiplication simpler
      typedef typename gauge::FieldOrder<Float,N,1,gOrder> gCoarse;
      typedef typename gauge::FieldOrder<Float,N,1,gOrder,true,storeFloat> gPreconditionedCoarse;
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gPreconditionedCoarse yHatAccessor(const_cast<GaugeField&>(Yhat));
      gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));
      printfQuda("Xinv = %e\n", xInvAccessor.norm2(0));

      int comm_dim[4];
      for (int i=0; i<4; i++) comm_dim[i] = comm_dim_partitioned(i);
      typedef CalculateYhatArg<gPreconditionedCoarse,gCoarse,N> yHatArg;
      yHatArg arg(yHatAccessor, yAccessor, xInvAccessor, xc_size, comm_dim, 1);

      if (Yhat.Precision() == QUDA_HALF_PRECISION) {
	double max = 3.0 * arg.Y.abs_max() * arg.Xinv.abs_max();
	Yhat.Scale(max);
	arg.Yhat.resetScale(max);
      }

      CalculateYhat<Float, N, yHatArg> yHat(arg, Y);
      yHat.apply(0);

#if 0
      for (int d=0; d<8; d++) printfQuda("Yhat[%d] = %e (%e %e = %e x %e)\n", d, arg.Yhat.norm2(d),
					 arg.Yhat.abs_max(d), arg.Y.abs_max(d) * arg.Xinv.abs_max(0),
					 arg.Y.abs_max(d), arg.Xinv.abs_max(0));
#endif

    }

    // fill back in the bulk of Yhat so that the backward link is updated on the previous node
    // need to put this in the bulk of the previous node - but only send backwards the backwards
    // links to and not overwrite the forwards bulk
    Yhat.injectGhost(QUDA_LINK_BACKWARDS);

    // exchange forwards links for multi-process dslash dagger
    // need to put this in the ghost zone of the next node - but only send forwards the forwards
    // links and not overwrite the backwards ghost
    Yhat.exchangeGhost(QUDA_LINK_FORWARDS);
  }

  template <typename storeFloat, typename Float, int N>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X)
  {
    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;
      if (Y.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", Y.FieldOrder());
      calculateYhat<storeFloat,Float,N,gOrder>(Yhat, Xinv, Y, X);
    } else {
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;
      if (Y.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", Y.FieldOrder());
      calculateYhat<storeFloat,Float,N,gOrder>(Yhat, Xinv, Y, X);
    }
  }

  // template on the number of coarse degrees of freedom
  template <typename storeFloat, typename Float>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X) {
    switch (Y.Ncolor()) {
    case  2: calculateYhat<storeFloat,Float, 2>(Yhat, Xinv, Y, X); break;
    case  4: calculateYhat<storeFloat,Float, 4>(Yhat, Xinv, Y, X); break;
    case  8: calculateYhat<storeFloat,Float, 8>(Yhat, Xinv, Y, X); break;
    case 12: calculateYhat<storeFloat,Float,12>(Yhat, Xinv, Y, X); break;
    case 16: calculateYhat<storeFloat,Float,16>(Yhat, Xinv, Y, X); break;
    case 20: calculateYhat<storeFloat,Float,20>(Yhat, Xinv, Y, X); break;
    case 24: calculateYhat<storeFloat,Float,24>(Yhat, Xinv, Y, X); break;
    case 32: calculateYhat<storeFloat,Float,32>(Yhat, Xinv, Y, X); break;
    case 48: calculateYhat<storeFloat,Float,48>(Yhat, Xinv, Y, X); break;
    case 64: calculateYhat<storeFloat,Float,64>(Yhat, Xinv, Y, X); break;
    default: errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor()); break;
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X) {
    QudaPrecision precision = checkPrecision(Xinv, Y, X);
    printfQuda("Computing Yhat field......\n");

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (Yhat.Precision() != QUDA_DOUBLE_PRECISION) errorQuda("Unsupported precision %d\n", Yhat.Precision());
      calculateYhat<double>(Yhat, Xinv, Y, X);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      if (Yhat.Precision() == QUDA_SINGLE_PRECISION) {
	calculateYhat<float,float>(Yhat, Xinv, Y, X);
      } else if (Yhat.Precision() == QUDA_HALF_PRECISION) {
	calculateYhat<short,float>(Yhat, Xinv, Y, X);
      } else {
	errorQuda("Unsupported precision %d\n", precision);
      }
    } else {
      errorQuda("Unsupported precision %d\n", precision);
    }

    printfQuda("....done computing Yhat field\n");
  }

} //namespace quda


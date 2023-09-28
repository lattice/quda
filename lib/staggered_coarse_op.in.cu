#include <memory>

#include <transfer.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>

// For naive Kahler-Dirac coarsening
#include <kernels/staggered_coarse_op_kernel.cuh>

// This define controls which kernels get compiled in `coarse_op.cuh`.
// This ensures only kernels relevant for coarsening the staggered
// operator get built, saving compile time.
#define STAGGEREDCOARSE
#include <coarse_op.cuh>
#include "multigrid.h"

namespace quda {

  template <typename Float, int fineColor>
  class CalculateStaggeredY : public TunableKernel3D {

    GaugeField& Y;
    GaugeField& X;
    const GaugeField& g;
    double mass;

    const int nDim = 4;

    long long flops() const {
      // no work, just a permutation
      return 0ll;
    }

    long long bytes() const
    {
      // 2 from forwards / backwards contributions, Y and X are sparse - only needs to write non-zero elements, 2nd term is mass term
      return g.Bytes() + (2 * g.Bytes() * Y.Precision()) / g.Precision() + X.Volume() * X.Ncolor() * X.Precision();
    }

    unsigned int minThreads() const { return g.VolumeCB(); }

  public:
    CalculateStaggeredY(GaugeField &Y, GaugeField &X, const GaugeField &g, const double mass) :
      TunableKernel3D(g, fineColor*fineColor, 2),
      Y(Y),
      X(X),
      g(g),
      mass(mass)
    {
      checkPrecision(Y, X);
      checkLocation(Y, X, g);
      if (g.Geometry() != QUDA_VECTOR_GEOMETRY)
        errorQuda("Unsupported geometry %d", g.Geometry());
      if (g.Ndim() != nDim)
        errorQuda("Number of dimensions %d is not supported", g.Ndim());
      if (X.Geometry() != QUDA_SCALAR_GEOMETRY)
        errorQuda("Unsupported geometry %d", X.Geometry());
      if (Y.Geometry() != QUDA_COARSE_GEOMETRY)
        errorQuda("Unsupported geometry %d", Y.Geometry());

      strcat(aux,comm_dim_partitioned_string());
      strcat(aux,",computeStaggeredVUV");

      // X and Y are relatively sparse; the kernel assumes the rest of X and Y
      // are already zero
      X.zero();
      Y.zero();

      // reset scales as appropriate
      if constexpr (sizeof(Float) < QUDA_SINGLE_PRECISION) {
        double max_scale = g.abs_max();
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global U_max = %e\n", max_scale);
        X.Scale(max_scale > 2.0*mass ? max_scale : 2.0*mass);
        Y.Scale(max_scale);
      }

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing VUV by permutation\n");

      apply(device::get_default_stream());

        if (getVerbosity() >= QUDA_VERBOSE) {
        for (int d = 0; d < nDim; d++) printfQuda("Y2[%d] = %e\n", 4+d, Y.norm2( 4+d ));
        for (int d = 0; d < nDim; d++) printfQuda("Y2[%d] = %e\n", d, Y.norm2( d ));
      }

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X.norm2(0));
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      // We're building both X and Y
      constexpr bool kd_build_x = false;
      if (X.Location() == QUDA_CPU_FIELD_LOCATION) {
        constexpr QudaGaugeFieldOrder order = QUDA_QDP_GAUGE_ORDER;
        CalculateStaggeredYArg<Float,fineColor,order,kd_build_x> arg(Y, X, g, mass);
        launch_host<ComputeStaggeredVUV>(tp, stream, arg);
      } else if (X.Location() == QUDA_CUDA_FIELD_LOCATION) {
        constexpr QudaGaugeFieldOrder order = QUDA_FLOAT2_GAUGE_ORDER;
        CalculateStaggeredYArg<Float,fineColor,order,kd_build_x> arg(Y, X, g, mass);
        launch_device<ComputeStaggeredVUV>(tp, stream, arg);
      }
    }

  };

  template <typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin, int uvSpin>
  void aggregateStaggeredY(GaugeField &Y, GaugeField &X,
                        const Transfer &T, const GaugeField &g, const GaugeField &l, const GaugeField &XinvKD,
                        double mass, bool allow_truncation, QudaDiracType dirac, QudaMatPCType matpc)
  {
    // Actually create the temporaries like UV, etc.
    auto location = Y.Location();

    // Create a field UV which holds U*V.  Has roughly the same structure as V,
    // though we need to double the spin for the KD operator to keep track of from even vs from odd.
    ColorSpinorParam UVparam(T.Vectors(location));
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.nSpin = uvSpin;
    UVparam.setPrecision(T.Vectors(location).Precision());
    UVparam.mem_type = Y.MemType(); // allocate temporaries to match coarse-grid link field

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    ColorSpinorField *av = (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) ? ColorSpinorField::Create(UVparam) : &const_cast<ColorSpinorField&>(T.Vectors(location));

    GaugeField *Yatomic = &Y;
    GaugeField *Xatomic = &X;

    if (Y.Precision() < QUDA_SINGLE_PRECISION) {
      // we need to coarsen into single precision fields (float or int), so we allocate temporaries for this purpose
      // else we can just coarsen directly into the original fields
      GaugeFieldParam param(X); // use X since we want scalar geometry
      param.location = location;
      param.setPrecision(QUDA_SINGLE_PRECISION, location == QUDA_CUDA_FIELD_LOCATION ? true : false);

      Yatomic = GaugeField::Create(param);
      Xatomic = GaugeField::Create(param);
    }

    // Moving along to the build

    const double kappa = -1.; // cancels a minus sign factor for kappa w/in the dslash application
    const double mu_dummy = 0.; 
    const double mu_factor_dummy = 0.;
    constexpr bool use_mma = false;
    
    bool need_bidirectional = false;
    if (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) need_bidirectional = true;

    // need to exchange with depth 3 b/c of long links
    const int nFace = (dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) ? 3 : 1;

    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using V = typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat>;
      using F = typename colorspinor::FieldOrderCB<Float,uvSpin,fineColor,coarseColor,csOrder,vFloat>;
      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder>;
      using gCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;
      using gCoarseAtomic = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,storeType>;

      const ColorSpinorField &v = T.Vectors(Y.Location());

      V vAccessor(const_cast<ColorSpinorField&>(v), nFace);
      F uvAccessor(*uv, nFace);
      F avAccessor(*av, nFace);
      gFine gAccessor(const_cast<GaugeField&>(g));
      gFine lAccessor(const_cast<GaugeField&>(l));
      gFine xinvAccessor(const_cast<GaugeField&>(XinvKD));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarseAtomic yAccessorAtomic(*Yatomic);
      gCoarseAtomic xAccessorAtomic(*Xatomic);
      
      // the repeated xinvAccessor is intentional
      calculateY<use_mma, QUDA_CPU_FIELD_LOCATION, false, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, avAccessor, vAccessor, gAccessor,
        lAccessor, xinvAccessor, xinvAccessor, xinvAccessor, Y, X, *Yatomic, *Xatomic, *uv, *av, v,
        kappa, mass, mu_dummy, mu_factor_dummy, allow_truncation, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
        T.coarseToFine(Y.Location()));
    } else {

      constexpr QudaFieldOrder csOrder = colorspinor::getNative<vFloat>(fineSpin);
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using V = typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat, vFloat, false, false>;
      using F = typename colorspinor::FieldOrderCB<Float, uvSpin, fineColor, coarseColor, csOrder, vFloat, vFloat, false, false>;
      using gFine =  typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float>;
      using gCoarse = typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, vFloat>;
      using gCoarseAtomic = typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, storeType>;

      const ColorSpinorField &v = T.Vectors(Y.Location());

      V vAccessor(const_cast<ColorSpinorField &>(v), nFace);
      F uvAccessor(*uv, nFace);
      F avAccessor(*av, nFace);
      gFine gAccessor(const_cast<GaugeField &>(g));
      gFine lAccessor(const_cast<GaugeField &>(l));
      gFine xinvAccessor(const_cast<GaugeField&>(XinvKD));
      gCoarse yAccessor(const_cast<GaugeField &>(Y));
      gCoarse xAccessor(const_cast<GaugeField &>(X));
      gCoarseAtomic yAccessorAtomic(*Yatomic);
      gCoarseAtomic xAccessorAtomic(*Xatomic);

      // create a dummy clover field to allow us to call the external clover reduction routines elsewhere
      calculateY<use_mma, QUDA_CUDA_FIELD_LOCATION, false, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, avAccessor, vAccessor, gAccessor,
        lAccessor, xinvAccessor, xinvAccessor, xinvAccessor, Y, X, *Yatomic, *Xatomic, *uv, *av, v,
        kappa, mass, mu_dummy, mu_factor_dummy, allow_truncation, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
        T.coarseToFine(Y.Location()));
    }

    // Clean up
    if (Yatomic != &Y) delete Yatomic;
    if (Xatomic != &X) delete Xatomic;

    if (av != nullptr && &T.Vectors(location) != av) delete av;
    if (uv != nullptr) delete uv;
  }

  // template on UV spin, which can be 1 for the non-KD ops but needs to be 2 for the KD op
  template <typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void aggregateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g, const GaugeField &l,
                           const GaugeField &XinvKD, double mass, bool allow_truncation, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if (dirac == QUDA_STAGGERED_DIRAC || dirac == QUDA_STAGGEREDPC_DIRAC || dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC) {
      // uvSpin == 1
      aggregateStaggeredY<Float, vFloat, fineColor, fineSpin, coarseColor, coarseSpin, 1>(Y, X, T, g, l, XinvKD, mass, allow_truncation, dirac, matpc);
    } else if (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) {
      // uvSpin == 2
      aggregateStaggeredY<Float, vFloat, fineColor, fineSpin, coarseColor, coarseSpin, 2>(Y, X, T, g, l, XinvKD, mass, allow_truncation, dirac, matpc);
    } else {
      errorQuda("Unexpected dirac type %d", dirac);
    }
  }

  // template on the number of coarse degrees of freedom, branch between naive K-D 
  // and actual aggregation
  template <typename Float, typename vFloat, int fineColor, int coarseColor>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g, const GaugeField &l,
                           const GaugeField &XinvKD, double mass, bool allow_truncation, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if (T.Vectors(X.Location()).Nspin() != 1) errorQuda("Unsupported number of spins %d", T.Vectors(X.Location()).Nspin());
    constexpr int fineSpin = 1;
    constexpr int coarseSpin = 2;

    if constexpr (coarseColor == 24) {
      if (T.getTransferType() == QUDA_TRANSFER_COARSE_KD)
        // The permutation routines need minimal machinery, inputs, trim down appropriately
        // see staggered_kd_build_xinv.cu for reference
        CalculateStaggeredY<vFloat,fineColor>(Y, X, g, mass);
      else {
        // free field aggregation
        aggregateStaggeredY<Float, vFloat, fineColor, fineSpin, coarseColor, coarseSpin>(Y, X, T, g, l, XinvKD, mass, allow_truncation, dirac, matpc);
      }
    } else {
      aggregateStaggeredY<Float, vFloat, fineColor, fineSpin, coarseColor, coarseSpin>(Y, X, T, g, l, XinvKD, mass, allow_truncation, dirac, matpc);
    }
  }

  template <int fineColor, int coarseColor>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g, const GaugeField &l,
                           const GaugeField &XinvKD, double mass, bool allow_truncation, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if constexpr (is_enabled_multigrid() && is_enabled_spin(1)) {
      logQuda(QUDA_SUMMARIZE, "Computing Y field......\n");
      auto precision = Y.Precision();
      if (!is_enabled(precision)) errorQuda("Precision %d not enabled by QUDA_PRECISION = %d", precision, QUDA_PRECISION);

      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          calculateStaggeredY<double, double, fineColor, coarseColor>(Y, X, T, g, l, XinvKD, mass, allow_truncation, dirac, matpc);
        else
          errorQuda("Double precision multigrid has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) {
          calculateStaggeredY<float, float, fineColor, coarseColor>(Y, X, T, g, l, XinvKD, mass, allow_truncation, dirac, matpc);
        }
      } else if (precision == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
          calculateStaggeredY<float, short, fineColor, coarseColor>(Y, X, T, g, l, XinvKD, mass, allow_truncation, dirac, matpc);
        }
      } else {
        errorQuda("Unsupported precision %d", precision);
      }
      logQuda(QUDA_SUMMARIZE, "....done computing Y field\n");
    } else {
      errorQuda("Staggered multigrid has not been built");
    }
  }

  constexpr int fineColor = 3;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC@;

  template <>
  void StaggeredCoarseOp<fineColor, coarseColor>(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge,
                                                 const cudaGaugeField &longGauge, const GaugeField &XinvKD, double mass,
                                                 bool allow_truncation, QudaDiracType dirac, QudaMatPCType matpc)
  {
    QudaPrecision precision = checkPrecision(T.Vectors(X.Location()), X, Y);
    QudaFieldLocation location = checkLocation(Y, X);

    // sanity check long link coarsening
    if ((dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) && &gauge == &longGauge)
      errorQuda("Dirac type is %d but fat and long gauge links alias", dirac);

    // sanity check KD op coarsening
    if ((dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) && &gauge == &XinvKD)
      errorQuda("Dirac type is %d but fat links and KD inverse fields alias", dirac);

    if (dirac == QUDA_ASQTADKD_DIRAC && &longGauge == &XinvKD)
      errorQuda("Dirac type is %d but long links and KD inverse fields alias", dirac);

    if ((dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) && XinvKD.Reconstruct() != QUDA_RECONSTRUCT_NO)
      errorQuda("Invalid reconstruct %d for KD inverse field", XinvKD.Reconstruct());

    std::unique_ptr<GaugeField> tmp_U(nullptr);
    std::unique_ptr<GaugeField> tmp_L(nullptr);
    std::unique_ptr<GaugeField> tmp_Xinv(nullptr);

    bool need_tmp_U = false;
    bool need_tmp_L = false;
    bool need_tmp_Xinv = false;

    if (location == QUDA_CPU_FIELD_LOCATION) {
      //First make a cpu gauge field from the cuda gauge field
      int pad = 0;
      GaugeFieldParam gf_param(gauge.X(), precision, QUDA_RECONSTRUCT_NO, pad, gauge.Geometry());
      gf_param.location = location;
      gf_param.order = QUDA_QDP_GAUGE_ORDER;
      gf_param.fixed = gauge.GaugeFixed();
      gf_param.link_type = gauge.LinkType();
      gf_param.t_boundary = gauge.TBoundary();
      gf_param.anisotropy = gauge.Anisotropy();
      gf_param.gauge = nullptr;
      gf_param.create = QUDA_NULL_FIELD_CREATE;
      gf_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      gf_param.nFace = 1;
      gf_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

      tmp_U = std::make_unique<cpuGaugeField>(gf_param);
      need_tmp_U = true;

      //Copy the cuda gauge field to the cpu
      gauge.saveCPUField(reinterpret_cast<cpuGaugeField&>(*tmp_U));

            // Create either a real or a dummy L field
      GaugeFieldParam lgf_param(longGauge.X(), precision, QUDA_RECONSTRUCT_NO, pad, longGauge.Geometry());
      if (!(dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC))
        for (int i = 0; i < lgf_param.nDim; i++) lgf_param.x[i] = 0;
      lgf_param.location = location;
      lgf_param.order = QUDA_QDP_GAUGE_ORDER;
      lgf_param.fixed = longGauge.GaugeFixed();
      lgf_param.link_type = longGauge.LinkType();
      lgf_param.t_boundary = longGauge.TBoundary();
      lgf_param.anisotropy = longGauge.Anisotropy();
      lgf_param.gauge = nullptr;
      lgf_param.create = QUDA_NULL_FIELD_CREATE;
      lgf_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      lgf_param.nFace = 3;
      lgf_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

      tmp_L = std::make_unique<cpuGaugeField>(lgf_param);
      need_tmp_L = true;

      //Copy the cuda gauge field to the cpu
      if (dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC)
        longGauge.saveCPUField(reinterpret_cast<cpuGaugeField&>(*tmp_L));

      // Create either a real or a dummy Xinv field
      GaugeFieldParam xgf_param(XinvKD.X(), precision, QUDA_RECONSTRUCT_NO, pad, XinvKD.Geometry());
      if (!(dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC))
        for (int i = 0; i < xgf_param.nDim; i++) xgf_param.x[i] = 0;
      xgf_param.location = location;
      xgf_param.order = QUDA_QDP_GAUGE_ORDER;
      xgf_param.fixed = XinvKD.GaugeFixed();
      xgf_param.link_type = XinvKD.LinkType();
      xgf_param.t_boundary = XinvKD.TBoundary();
      xgf_param.anisotropy = XinvKD.Anisotropy();
      if (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) {
        xgf_param.create = QUDA_COPY_FIELD_CREATE;
      } else {
        xgf_param.gauge = nullptr;
        xgf_param.create = QUDA_NULL_FIELD_CREATE;
      }
      xgf_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      xgf_param.nFace = 0;
      xgf_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

      tmp_Xinv = std::make_unique<cpuGaugeField>(xgf_param);
      need_tmp_Xinv = true;

      //Copy the cuda gauge field to the cpu
      //if (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC)
      //  XinvKD.saveCPUField(*static_cast<cpuGaugeField*>(Xinv));
    } else if (location == QUDA_CUDA_FIELD_LOCATION) {

      int pad = 0;

      // create some dummy fields if need be
      if (!(dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC)) {
        // create a dummy field
        GaugeFieldParam lgf_param(longGauge);
        for (int i = 0; i < lgf_param.nDim; i++) lgf_param.x[i] = 0;
        lgf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        lgf_param.order = QUDA_FLOAT2_GAUGE_ORDER;
        lgf_param.setPrecision(lgf_param.Precision());
        lgf_param.create = QUDA_NULL_FIELD_CREATE;
        tmp_L = std::make_unique<cudaGaugeField>(lgf_param);
        need_tmp_L = true;
      } else if ((dirac == QUDA_ASQTAD_DIRAC || dirac == QUDA_ASQTADPC_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) && longGauge.Reconstruct() != QUDA_RECONSTRUCT_NO) {
        // create a copy of the gauge field with no reconstruction
        GaugeFieldParam lgf_param(longGauge);
        lgf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        lgf_param.order = QUDA_FLOAT2_GAUGE_ORDER;
        lgf_param.setPrecision(lgf_param.Precision());
        tmp_L = std::make_unique<cudaGaugeField>(lgf_param);

        tmp_L->copy(longGauge);
        tmp_L->exchangeGhost();
        need_tmp_L = true;
      }

      if (!(dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC)) {
        // Create a dummy field
        GaugeFieldParam xgf_param(XinvKD.X(), precision, QUDA_RECONSTRUCT_NO, pad, XinvKD.Geometry());
        for (int i = 0; i < xgf_param.nDim; i++) xgf_param.x[i] = 0;
        xgf_param.location = location;
        xgf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        xgf_param.order = QUDA_FLOAT2_GAUGE_ORDER;
        xgf_param.setPrecision(xgf_param.Precision());
        xgf_param.create = QUDA_NULL_FIELD_CREATE;
        tmp_Xinv = std::make_unique<cudaGaugeField>(xgf_param);
        need_tmp_Xinv = true;
      }
      // no need to worry about XinvKD's reconstruct

      if (gauge.Reconstruct() != QUDA_RECONSTRUCT_NO) {
        //Create a copy of the gauge field with no reconstruction, required for fine-grained access
        GaugeFieldParam gf_param(gauge);
        gf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        gf_param.order = QUDA_FLOAT2_GAUGE_ORDER;
        gf_param.setPrecision(gf_param.Precision());
        tmp_U = std::make_unique<cudaGaugeField>(gf_param);
        need_tmp_U = true;

        tmp_U->copy(gauge);
        tmp_U->exchangeGhost();
      }
    }

    const GaugeField& U = need_tmp_U ? *tmp_U : reinterpret_cast<const GaugeField&>(gauge);
    const GaugeField& L = need_tmp_L ? *tmp_L : reinterpret_cast<const GaugeField&>(longGauge);
    const GaugeField& Xinv = need_tmp_Xinv ? *tmp_Xinv : reinterpret_cast<const GaugeField&>(XinvKD);

    calculateStaggeredY<fineColor, coarseColor>(Y, X, T, U, L, Xinv, mass, allow_truncation, dirac, matpc);
  }

} //namespace quda

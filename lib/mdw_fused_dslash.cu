#include <mdw_dslash5_tensor_core.cuh>

namespace quda {
namespace mdw_tensor_core {

#if defined (GPU_DOMAIN_WALL_DIRAC) && (__COMPUTE_CAPABILITY__ >= 700)
  /**
    @brief Structure containing zMobius / Zolotarev coefficients

//    FIXME
//    - fix flops counters
//    - use kappa notation and not b/c for consistency with other codes and sanity
//  */
//  template <typename real>
//    struct coeff_5 {
//      complex<real> a[QUDA_MAX_DWF_LS]; // xpay coefficients
//      complex<real> b[QUDA_MAX_DWF_LS];
//      complex<real> c[QUDA_MAX_DWF_LS];
//    };
//
//  constexpr int size = 4096;
//  static __constant__ char mobius_d[size]; // constant buffer used for Mobius coefficients for GPU kernel
//  static char mobius_h[size];              // constant buffer used for Mobius coefficients for CPU kernel
//
//  /**
//    @brief Helper function for grabbing the constant struct, whether
//    we are on the GPU or CPU.
//   */
//  template <typename real>
//    inline __device__ __host__ const coeff_5<real>* coeff() {
//#ifdef __CUDA_ARCH__
//      return reinterpret_cast<const coeff_5<real>*>(mobius_d);
//#else
//      return reinterpret_cast<const coeff_5<real>*>(mobius_h);
//#endif
//    }
//
//  template <typename real, Dslash5Type, typename Arg> struct coeff_type {
//    typedef real type;
//    const Arg &arg;
//    __device__ __host__ coeff_type(const Arg &arg) : arg(arg) { }
//    __device__ __host__ real a(int s) { return arg.a; }
//    __device__ __host__ real b(int s) { return arg.b; }
//    __device__ __host__ real c(int s) { return arg.c; }
//  };
//
//  template <typename real, typename Arg> struct coeff_type<real,M5_INV_ZMOBIUS,Arg> {
//    typedef complex<real> type;
//    __device__ __host__ coeff_type(const Arg &arg) { }
//    __device__ __host__ complex<real> a(int s) { return coeff<real>()->a[s]; }
//    __device__ __host__ complex<real> b(int s) { return coeff<real>()->b[s]; }
//    __device__ __host__ complex<real> c(int s) { return coeff<real>()->c[s]; }
//  };
  
  /**
    @brief Parameter structure for applying the Dslash
  */
  template<int Ls_>
  struct FusedDslashArg {
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<short, 4, 3, spin_project, spinor_direct_load>::type F;
    typedef typename mapper<short>::type real;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<short, QUDA_RECONSTRUCT_NO, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    F out;                  // output vector field
    const F in;             // input vector field
    F y;                    // auxiliary output vector field
    const F x;              // auxiliary input vector field
    
    const G U;
    
    const int nParity;      // number of parities we're working on
    const int volume_cb;    // checkerboarded volume
    const int volume_4d_cb; // 4-d checkerboarded volume
    
    const int dim[4]; 
    
    const int_fastdiv Ls;   // length of 5th dimension

    const int shift[4] // sites where we actually calculate.
    const int halo_shift[4]; // halo means zero. When we are expanding we have halo of cs-field where values are zero.
    
    const int_fastdiv shrinked_dim[4]; // dimension after shifts are considered. 
    // const int_fastdiv halo_shrinked_dim[4]; // dimension after halo shifts are considered.

    // partial kernel and expansion parameters
    const int volume_4d_cb_shift; // number of 4d sites we need calculate 
    // const int volume_4d_cb_expansive; // 

    const real m_f;         // fermion mass parameter
    const real m_5;         // Wilson mass shift

    const bool dagger;      // dagger
    const bool xpay;        // whether we are doing xpay or not

    real b;                 // real constant Mobius coefficient
    real c;                 // real constant Mobius coefficient
    real a;                 // real xpay coefficient

    real kappa;
    real fac_inv;

    real alpha = 1.;
    real beta = 0.;

    MdwfFusedDslashType type;
    FusedDslashArg(ColorSpinorField& out, const ColorSpinorField& in, const GaugeField& U,
                          ColorSpinorField& y, const ColorSpinorField& x, double m_f_, double m_5_, 
                          const Complex* b_5, const Complex* c_5, MdwfFusedDslashType type)
      : out(out), in(in), U(U), y(y), x(x), nParity(in.SiteSubset()), volume_cb(in.VolumeCB()), 
      volume_4d_cb(volume_cb/Ls_), Ls(Ls_), m_f(m_f_), m_5(m_5_), 
      shift{ ... }, halo_shift{ ... }, dim{ (3-nParity) * in.X(0), in.X(1), in.X(2), in.X(3)}, 
      shrinked_dim{ (dim[0]-2*shift[0], dim[1]-2*shift[1], dim[2]-2*shift[2], dim[3]-2*shift[3]}, type(type)
//      halo_shrinked_dim{ (dim[0]-2*halo_shift[0], dim[1]-2*halo_shift[1], dim[2]-2*halo_shift[2], dim[3]-2*halo_shift[3]}, type(type)
    {
      if(in.Nspin() != 4){
        errorQuda("nSpin = %d not support", in.Nspin());
      }
      
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

//      if (sizeof(coeff_5<real>) > size) errorQuda("Coefficient buffer too large at %lu bytes\n", sizeof(coeff_5<real>));
//      coeff_5<real> *coeff = reinterpret_cast<coeff_5<real>*>(&mobius_h);
//      auto *a_5 =  coeff->a;
//      auto *b_5 =  coeff->b;
//      auto *c_5 =  coeff->c;

      b = b_5_[0].real();
      c = c_5_[0].real();
      kappa = -(c*(4.+m_5)-1.) / (b*(4.+m_5)+1.);
      fac_inv = 0.5/(1.+std::pow(kappa,(int)Ls)*m_f); // 0.5 to normalize the (1 +/- gamma5) in the chiral projector.

      switch(type){
        case dslash4_dslash5pre_dslash5inv:
          // a *= pow(0.5 / (b_5_[0].real() * (m_5 + 4.0) + 1.0), 2);
          alpha = b-c/kappa;
          beta = c/kappa;
          break;
        default:
          errorQuda("Unknown MdwfFusedDslashType %d", type);
      }

//      cudaMemcpyToSymbolAsync(mobius_d, mobius_h, sizeof(coeff_5<real>), 0, cudaMemcpyHostToDevice, streams[Nstream-1]);

    }
  };

  __device__ inline void index_4d_cb_from_coordinate_4d(const int coordinate[], const int dim[]){
    return ( ((coordinate[3]*dim[2]+coordinate[2])*dim[1]+coordinate[1])*dim[0]+coordinate[0] ) >> 1;
  }

  __device__ inline bool is_halo_4d(const int coordinate[], const int dim[4], const int halo_shift){
    bool ret = false;
    #pragma unroll
    for(int d = 0; d < 4; d++){
      ret = ret or (coordinate[d] >= halo_dim[d]-halo_shift or coordinate[d] < halo_shift) ;
    }
    return ret;
  }

  /**
  -> Everything should be understood in a 4d checkboarding sense.
  */
  template<class Float, bool dagger, bool halo, class Vector, class Arg>
  __device__ __host__ inline void apply_wilson_5d(Vector& out, const int coordinate_4d[], Arg& arg, int s, int parity, int nParity) {
    typedef typename mapper<Float>::type real;
//    typedef ColorSpinor<real,nColor,2> HalfVector;
    typedef Matrix<complex<real>, 3> Link;
    const int their_spinor_parity = nParity == 2 ? 1-parity : 0;

    const int index_4d_cb = index_4d_cb_from_coordinate_4d(coordinate_4d, arg.dim);
//    int coord[nDim];
//    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

    int x[4] = { coordinate_4d[0], coordinate_4d[1], coordinate_4d[2], coordinate_4d[3] };

    #pragma unroll
    for (int d = 0; d < 4; d++) // loop over dimension
    {
      x[d]++;
      if(halo and is_halo_4d(x, arg.dim, arg.halo_shift)){
      
      }else{ // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = 2*arg.volume_4d_cb+index_4d_cb_from_coordinate_4d(x, arg.dim);
        constexpr int proj_dir = dagger ? +1 : -1;

        if ( false ) {
//          const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
//
//          const Link U = arg.U(d, x_cb, parity);
//          const HalfVector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
//
//          out += (U * in).reconstruct(d, proj_dir);
        } else {
          const Link U = arg.U(d, index_4d_cb, parity);
          const Vector in = arg.in(fwd_idx, their_spinor_parity);
          out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }
      x[d] -= 2;
      if(halo and  is_halo_4d(x, arg.dim, arg.halo_shift)){
      
      }else{ //Backward gather - compute back offset for spinor and gauge fetch
        const int gauge_idx = index_4d_cb_from_coordinate_4d(x, arg.dim);;
        const int back_idx = 2*arg.volume_4d_cb+gauge_idx;
        constexpr int proj_dir = dagger ? -1 : +1;

        if ( false ) {
//          const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
//
//          const Link U = arg.U.Ghost(d, ghost_idx, 1-parity);
//          const HalfVector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
//
//          out += (conj(U) * in).reconstruct(d, proj_dir);
        } else {
          const Link U = arg.U(d, gauge_idx, 1-parity);
          const Vector in = arg.in(back_idx, their_spinor_parity);
          out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }
      x[d]++;
    } //nDim

  }

  /**
  -> Everything should be understood in a 4d checkboarding sense.
  */
  template<class T>
  __device__ inline void coordinate_from_shrinked_index(int coordinate[], int shrinked_index, 
                          int dim[], T shrinked_dim[], const int shift[], int parity) // s is the 5d stuff, 
  {
    int aux[4];
    aux[0] = shrinked_index*2;
    
    #pragma unroll
    for(int i = 0; i < 3; i++){
      aux[i+1] = aux[i] / shrinked_dim[i]; 
    }

    coordinate[0] = aux[0] - aux[1] * shrinked_index[0];
    coordinate[1] = aux[1] - aux[2] * shrinked_index[1];
    coordinate[2] = aux[2] - aux[3] * shrinked_index[2];
    coordinate[3] = aux[3];

    // Find the full coordinate in the shrinked volume. 
    coordinate[0] += (parity+coordinate[3]+coordinate[2]+coordinate[1])&1;

    // Now go back to the extended volume.
    #pragma unroll
    for(int d = 0; d < 4; d++){
      coordinate[d] += shift[d];
    }
  }
  
  /**
    @brief Tensor core kernel for applying the beta + alpha*M5inv operator
  */
  template<int block_dim_x, int Ls_, bool reload, class Arg>
  __global__ void fused_tensor_core(Arg arg)
  {
    float scale;

    TensorCoreSharedMemory<half2> shared_memory_data;
    
    constexpr int M = 4*Ls_;
    constexpr int N = 6*block_dim_x;
    
    constexpr int sm_m_pad_size = 0;
    constexpr int sm_n_pad_size = 16;
    
    constexpr int N_sm = N + sm_n_pad_size;
    constexpr int M_sm = M + sm_m_pad_size;
    
    half2* sm_b = shared_memory_data;
    half*  sm_c = (half*)sm_b;
    half*  sm_a = sm_c+M*N_sm;

    construct_matrix_a_m5inv<block_dim_x, Ls_, M_sm, dagger, Arg>(arg, sm_a);
    
    __syncthreads();
   
    bool idle = false;
    int s4_base = blockIdx.x*blockDim.x; // base.
    int s4, sid;
 
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    constexpr int tm_dim = M/WMMA_M;
    constexpr int tn_dim = N/WMMA_N;
    
    constexpr int total_warp = block_dim_x*Ls_/32;
    const int this_warp = (threadIdx.y*block_dim_x+threadIdx.x)/32;
    
    constexpr int total_tile = tm_dim*tn_dim;
    
    constexpr int warp_cycle = total_tile/total_warp;
    const int warp_m = this_warp*warp_cycle/tn_dim;
     
    typedef typename nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_type;
    
    a_type a_frag[reload?1:tm_dim];
    if(!reload){ // in the preload case we preload ... 
      #pragma unroll
      for( int k = 0; k < tm_dim; k++ ){
        const int a_row = warp_m*WMMA_M;
        const int a_col = k*WMMA_K;
        // Load Matrix
        nvcuda::wmma::load_matrix_sync(a_frag[k], sm_a+a_row+a_col*M_sm, M_sm);
      } 
    }

    while(s4_base < arg.volume_4d_cb){
      
      s4 = s4_base + threadIdx.x;
      sid = threadIdx.y*arg.volume_4d_cb + s4;
      
      if (s4 >= arg.volume_4d_cb){
        idle = true;
      }
    
      if(!idle){
        scale = load_matrix_b_tex<N_sm, Arg>(arg, sm_b, sid);
      }
      
      __syncthreads();
    
      wmma_gemm<block_dim_x, Ls_, M, N, M_sm, N_sm, reload>(a_frag, sm_a, sm_c, sm_c);        
      
      __syncthreads();
    
      if(!idle){
        store_matrix_c<N_sm, Arg>(arg, sm_b, sid, scale);
      }
    
      s4_base += gridDim.x*blockDim.x;
    
    } // while
  }
 
  template<int Ls_, class Arg>
  class FusedDslash : public TunableVectorYZ {

    protected:
      Arg &arg;
      const ColorSpinorField &meta;
      static constexpr bool shared = true; // whether to use shared memory cache blocking for M5inv

      /** Whether to use variable or fixed coefficient algorithm.  Must be true if using ZMOBIUS */
      static constexpr bool var_inverse = true;

      long long flops() const {
        long long Ls = Ls_;
        long long bulk = (Ls-2)*(meta.Volume()/Ls);
        long long wall = 2*meta.Volume()/Ls;
        long long n = meta.Ncolor() * meta.Nspin();

        long long flops_ = 0;
        switch (arg.type) {
          case M5_INV_MOBIUS: // FIXME flops
            //flops_ = ((2 + 8 * n) * Ls + (arg.xpay ? 4ll : 0)) * meta.Volume();
            flops_ = (144 * Ls + (arg.xpay ? 4ll : 0)) * meta.Volume();
            break;
          default:
            errorQuda("Unknown MdwfFusedDslashType %d", arg.type);
        }

        return flops_;
      }

      long long bytes() const {
        // long long Ls = meta.X(4);
        switch (arg.type) {
          case M5_INV_MOBIUS:
            return arg.out.Bytes() + arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
          default: 
            errorQuda("Unknown MdwfFusedDslashType %d", arg.type);
        }
        return 0ll;
      }

      virtual bool tuneGridDim() const { return true; }
      virtual bool tuneAuxDim() const { return true; }
      virtual bool tuneSharedBytes() const { return true; }
      unsigned int minThreads() const { return arg.volume_4d_cb; }
  
      unsigned int shared_bytes_per_block(int x, int y) const { 
        // (Ls*4) by (Ls*4), (Ls*4) by (volume_4d*6 + 16)
        return ( (y*4)*(y*4+0)+(y*4)*(x*6+16) )*2;
      }
   
      virtual bool advanceBlockDim(TuneParam &param) const
      {
        if( param.block.x < max_block_size() ){
          param.block.x += step_block_size();
          param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y); 
          return true;
        }else{
          return false;
        }
      }
      
      virtual bool advanceGridDim(TuneParam &param) const
      {
        const unsigned int max_blocks = maxGridSize();
        const int step = deviceProp.multiProcessorCount;
        param.grid.x += step;
        if (param.grid.x > max_blocks) {
          return false;
        } else {
          param.block.x = min_block_size();
          param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y); 
          return true;
        }
      }
      
      virtual bool advanceAux(TuneParam &param) const
      {
        if (param.aux.x < 1) {
          param.aux.x++;
          // We have updated the "aux" so reset all other parameters. 
          param.grid.x = minGridSize();
          param.block.x = min_block_size();
          param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y); 
          return true;
        } else {
          param.aux.x = 0;
          return false;
        }
      }

      virtual unsigned int maxGridSize() const { return 32*deviceProp.multiProcessorCount; }
      virtual unsigned int minGridSize() const { return deviceProp.multiProcessorCount; }
      unsigned int min_block_size() const { return  8; }
      unsigned int max_block_size() const { return 64; }
      unsigned int step_block_size() const { return  8; }

      // overloaded to return max dynamic shared memory if doing shared-memory inverse
      unsigned int maxSharedBytesPerBlock() const {
        if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS) ) {
          return maxDynamicSharedBytesPerBlock();
        } else {
          return TunableVectorYZ::maxSharedBytesPerBlock();
        }
      }

    public:
      FusedDslash(Arg &arg, const ColorSpinorField &meta)
        : TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta)
      {
        strcpy(aux, meta.AuxString());
        if (arg.dagger) strcat(aux, ",Dagger");
        if (arg.xpay) strcat(aux,",xpay");
        switch (arg.type) {
          case M5_INV_MOBIUS:
            strcat(aux, ",dslash4_dslash5pre_dslash5inv");
            break;
          default: 
            errorQuda("Unknown MdwfFusedDslashType %d", arg.type);
        }
      }
      virtual ~FusedDslash() { }

      template<typename T>
      inline void launch(T *f, const TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
        // static bool init = false;
        if ( shared ) {
          // if inverse kernel uses shared memory then maximize total shared memory pool
          setMaxDynamicSharedBytesPerBlock(f);
          // set_shared_memory_on_volta((const void*)f, "Some Function");
          // init = true;
        }
        void *args[] = { &arg };
        qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
      }

      void apply(const cudaStream_t &stream) {
        // By its name we ONLY have a GPU version
        // TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_DEBUG_VERBOSE);
          switch(arg.type){
            case dslash4_dslash5pre_dslash5inv:
              switch(tp.block.x){
                case  8:
                  tp.aux.x?
                  launch(fused_tensor_core< 8, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core< 8, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                case 16:
                  tp.aux.x?
                  launch(fused_tensor_core<16, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core<16, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                case 24:
                  tp.aux.x?
                  launch(fused_tensor_core<24, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core<24, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                case 32:
                  tp.aux.x?
                  launch(fused_tensor_core<32, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core<32, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                case 40:
                  tp.aux.x?
                  launch(fused_tensor_core<40, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core<40, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                case 48:
                  tp.aux.x?
                  launch(fused_tensor_core<48, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core<48, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                case 56:
                  tp.aux.x?
                  launch(fused_tensor_core<56, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core<56, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                case 64:
                  tp.aux.x?
                  launch(fused_tensor_core<64, Ls_, true, Arg>, tp, arg, stream) :
                  launch(fused_tensor_core<64, Ls_,false, Arg>, tp, arg, stream) ;
                  break;
                default:
                  errorQuda("NOT valid blockDim.x(=%d)\n", tp.block.x);
              }
              break;
            default: 
              errorQuda("Unknown MdwfFusedDslashType %d", arg.type);
          }
      }

      void initTuneParam(TuneParam &param) const {
        TunableVectorYZ::initTuneParam(param);
        param.block = dim3(min_block_size(), arg.Ls, 1); // Ls must be contained in the block
        param.grid = dim3(minGridSize(), 1, 1);
        param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y); 
        param.aux.x = 0;
      }

      void defaultTuneParam(TuneParam &param) const {
        initTuneParam(param);
      }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

#endif // defined (GPU_DOMAIN_WALL_DIRAC) && (__COMPUTE_CAPABILITY__ >= 700)
  
  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5 * in
  
  void apply_fused_dslash(ColorSpinorField& out, const ColorSpinorField& in, const GaugeField& U,
                          ColorSpinorField& y, const ColorSpinorField& x, double m_f, double m_5, 
                          const Complex* b_5, const Complex* c_5, MdwfFusedDslashType type)
  {
#if defined (GPU_DOMAIN_WALL_DIRAC) && (__COMPUTE_CAPABILITY__ >= 700)
    if (in.DWFPCtype() != QUDA_4D_PC) errorQuda("ONLY 4D preconditioned fields are supported");
    checkLocation(out, in);     // check all locations match
  
    if( checkPrecision(out, in) == QUDA_HALF_PRECISION && in.Ncolor() == 3){
      // switch for Ls
      switch(in.X(4)){
        case  8:
          {
            FusedDslashArg< 8> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
            FusedDslash< 8, FusedDslashArg< 8> > dslash(arg, in);
            dslash.apply(streams[Nstream-1]);
          }
        break;
        case 12:
          {
            FusedDslashArg<12> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
            FusedDslash<12, FusedDslashArg<12> > dslash(arg, in);
            dslash.apply(streams[Nstream-1]);
          }
        break;
        case 16:
          {
            FusedDslashArg<16> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
            FusedDslash<16, FusedDslashArg<16> > dslash(arg, in);
            dslash.apply(streams[Nstream-1]);
          }
        break;
        case 20:
          {
            FusedDslashArg<20> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
            FusedDslash<20, FusedDslashArg<20> > dslash(arg, in);
            dslash.apply(streams[Nstream-1]);
          }
        break;
        case 24:
          {
            FusedDslashArg<24> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
            FusedDslash<24, FusedDslashArg<24> > dslash(arg, in);
            dslash.apply(streams[Nstream-1]);
          }
        break;
        default: 
          errorQuda("Ls = %d is NOT supported.\n", in.X(4));
      }
    }else{
      errorQuda("Tensor core implemtation ONLY supports HALF precision and n_color = 3.\n");
    }
#else
    errorQuda("Domain wall dslash WITH tensor cores has not been built");
#endif
  }

} // namespace mdw_tensor_core
} // namespace quda


#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_quda.h>
#include <index_helper.cuh>
#include <inline_ptx.h>
#include <math_helper.cuh>
#include <shared_memory_cache_helper.cuh>

#include <cub/cub.cuh>

#include <madwf_ml.h>

namespace quda {
namespace madwf_ml {

  template<class real>
  using WilsonVector = ColorSpinor<real, 3, 4>;

  template<class real>
  class WilsonMatrix {
   
    static constexpr int size = wm_dim;
    complex<real> data[size];
    
  public:
    
    __device__ __host__ inline WilsonMatrix<real>() {
      #pragma unroll
      for (int i = 0; i < size; i++) { data[i] = 0; }
    }

    __device__ __host__ inline WilsonMatrix<real>(const WilsonMatrix<real>& a) {
      #pragma unroll
      for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
    }

    __device__ __host__ inline WilsonMatrix<real>& operator=(const WilsonMatrix<real>& a) {
      if (this != &a) {
        #pragma unroll
        for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
      }
      return *this;
    }
    
    __device__ __host__ inline complex<real>& operator()(int index) {
      return data[index];
    }
    
    __device__ __host__ inline const complex<real>& operator()(int index) const {
      return data[index];
    }

    // Wilson Matrix is row major
    __device__ __host__ inline complex<real>& operator()(int row, int column) {
      return data[row * color_spin_dim + column];
    }
    
    __device__ __host__ inline const complex<real>& operator()(int row, int column) const {
      return data[row * color_spin_dim + column];
    }
  
  };
  
  template<bool dagger, class real>
  __device__ __host__ inline WilsonVector<real> matrix_vector_multiply(const WilsonMatrix<real>& m, const WilsonVector<real>& v){
    WilsonVector<real> out; // out is initialized to zero
    #pragma unroll
    for(int column = 0; column < color_spin_dim; column++){
      auto v_col = v(column);
      #pragma unroll
      for(int row = 0; row < color_spin_dim; row++){
        if(dagger){
          out(row) += conj(m(column, row)) * v_col;
        }else{
          out(row) += m(row, column) * v_col;
        }
      }
    }
    return out;
  }

  template<class real>
  __device__ __host__ inline WilsonMatrix<real> vector_tensor_matrix(const WilsonVector<real>& v, const WilsonVector<real>& w){
    WilsonMatrix<real> m;
    #pragma unroll
    for(int row = 0; row < color_spin_dim; row++){
      #pragma unroll
      for(int column = 0; column < color_spin_dim; column++){
        m(row, column) = conj(conj(v(row)) * w(column));
        // m(row, column) = conj(v(row)) * w(column);
      }
    }
    return m;
  }

#ifdef GPU_DOMAIN_WALL_DIRAC

  template <typename storage_type> struct MadwfMlArg {

    typedef typename colorspinor_mapper<storage_type, 4, 3>::type F;
    typedef typename mapper<storage_type>::type real;

    F out; // output vector field
    const F in; // input vector field
    
    const int Ls_out; // length of 5th dimension
    const int Ls_in; // length of 5th dimension

    const int volume_4d_cb;
    
    WilsonMatrix<real>* tensor_out_p;
    const WilsonMatrix<real>* wm_p;
    
    const bool dagger; // dagger

    const int nParity;

    const bool transfer;

    MadwfMlArg(ColorSpinorField& out, const ColorSpinorField& in, const WilsonMatrix<real>* wm_p, bool dagger)
      : out(out)
        , in(in)
        , volume_4d_cb(in.VolumeCB() / in.X(4))
        , Ls_in(in.X(4))
        , Ls_out(out.X(4))
        , wm_p(wm_p)
        , dagger(dagger)
        , transfer(true)
        , nParity(in.SiteSubset()) {
          if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
          if (in.Ncolor() != 3) errorQuda("nColor = %d not support", in.Ncolor());
          if (out.Nspin() != 4) errorQuda("nSpin = %d not support", out.Nspin());
          if (out.Ncolor() != 3) errorQuda("nColor = %d not support", out.Ncolor());

          if (!in.isNative() || !out.isNative())
            errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
        }
     
     MadwfMlArg(ColorSpinorField& out, const ColorSpinorField& in, WilsonMatrix<real>* wm_p)
      : out(out)
        , in(in)
        , volume_4d_cb(in.VolumeCB() / in.X(4))
        , Ls_in(in.X(4))
        , Ls_out(out.X(4))
        , tensor_out_p(wm_p)
        , dagger(false)
        , transfer(false)
        , nParity(in.SiteSubset()) {
          if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
          if (in.Ncolor() != 3) errorQuda("nColor = %d not support", in.Ncolor());
          if (out.Nspin() != 4) errorQuda("nSpin = %d not support", out.Nspin());
          if (out.Ncolor() != 3) errorQuda("nColor = %d not support", out.Ncolor());

          if (!in.isNative() || !out.isNative())
            errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
        }

  };
  
  template<class storage_type, class Arg>
    __global__ void tensor_5d_kernel(Arg arg) {

      typedef typename mapper<storage_type>::type real;
      typedef ColorSpinor<real, 3, 4> Vector;

      const int Ls_in = arg.Ls_in;
      const int Ls_out = arg.Ls_out;
      const int volume_4d_cb = arg.volume_4d_cb;
      WilsonMatrix<real>* wm_p = arg.tensor_out_p;

      int index_4d_cb = blockIdx.x * blockDim.x + threadIdx.x;
      int s = blockIdx.y * blockDim.y + threadIdx.y;
      int parity = blockIdx.z * blockDim.z + threadIdx.z;

      if (index_4d_cb >= volume_4d_cb) return;
      if (s >= Ls_out) return;
      if (parity >= arg.nParity) return;
   
      VectorCache<real, Vector> cache;
      
      int ld = Ls_in * blockDim.x;
      int t = s;
      while(t < Ls_in){
        int index = t * blockDim.x + threadIdx.x;
        cache.save(index, ld, arg.in(t * volume_4d_cb + index_4d_cb, parity)); 
        t += blockDim.y;
      }
      cache.sync();

      typedef cub::WarpReduce<complex<real>> WarpReduce;
      __shared__ typename WarpReduce::TempStorage temp_storage;

      // t -> s_in, s-> s_out
      const Vector v = arg.out(s * volume_4d_cb + index_4d_cb, parity); 
      for(t = 0; t < Ls_in; t++){
        const Vector w = cache.load(t * blockDim.x + threadIdx.x, ld); 
        int wm_index = s * Ls_in + t;
        real* p = reinterpret_cast<real*>(&wm_p[wm_index]);
        for(int a = 0; a < color_spin_dim; a++){
        for(int b = 0; b < color_spin_dim; b++){
          int cs = a * color_spin_dim + b;
          auto z = conj(conj(v(a)) * w(b));
          complex<real> aggregate = WarpReduce(temp_storage).Sum(z);
          if(threadIdx.x == 0) {
            atomicAdd(&p[cs*2+0], aggregate.real());
            atomicAdd(&p[cs*2+1], aggregate.imag());
          }
        }}
      }

    }

  template<class storage_type, bool dagger, class Arg>
    __global__ void transfer_5d_kernel(Arg arg) {

      typedef typename mapper<storage_type>::type real;
      typedef ColorSpinor<real, 3, 4> Vector;

      const int Ls_in = arg.Ls_in;
      const int Ls_out = arg.Ls_out;
      const int volume_4d_cb = arg.volume_4d_cb;
      const WilsonMatrix<real>* wm_p = arg.wm_p;

      int index_4d_cb = blockIdx.x * blockDim.x + threadIdx.x;
      int s = blockIdx.y * blockDim.y + threadIdx.y;
      int parity = blockIdx.z * blockDim.z + threadIdx.z;

      if (index_4d_cb >= volume_4d_cb) return;
      if (s >= Ls_out) return;
      if (parity >= arg.nParity) return;
      
      Vector out;
      // t -> s_in, s-> s_out
      for(int t = 0; t < Ls_in; t++){
        Vector in = arg.in(t * volume_4d_cb + index_4d_cb, parity); 
        int wm_index;
        if(dagger){
          wm_index = t * Ls_out + s;
        }else{
          wm_index = s * Ls_in  + t;
        }
        out += matrix_vector_multiply<dagger>(wm_p[wm_index], in);
      }
      arg.out(s * volume_4d_cb + index_4d_cb, parity) = out;
    }

  template <class storage_type, class Arg>
    class Transfer5d : public TunableVectorYZ {

      typedef typename mapper<storage_type>::type real;

      Arg& arg;
      const ColorSpinorField& meta; // this reference is for meta data only

      private:
      unsigned int sharedBytesPerThread() const { 
        if(arg.transfer){
          return 0;
        }else{
          return (arg.Ls_in) * color_spin_dim * 2 * sizeof(typename mapper<storage_type>::type) / arg.Ls_out; 
        }
      }
      
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
      // bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.volume_4d_cb; }

      public:
      Transfer5d(const ColorSpinorField &meta, Arg& arg)
        : TunableVectorYZ(arg.Ls_out, arg.nParity), arg(arg), meta(meta) {
        strcpy(aux, meta.AuxString());
        strcat(aux, arg.transfer?",transfer_5d":",tensor_5d");
        if (arg.dagger) strcat(aux, ",Dagger");
      }

      virtual ~Transfer5d() { ; }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if(arg.transfer){
          if(arg.dagger){
            transfer_5d_kernel<storage_type, true><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
          }else{
            transfer_5d_kernel<storage_type, false><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
          }
        }else{
          tensor_5d_kernel<storage_type><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); 
        }
      }

      void initTuneParam(TuneParam &param) const
      {
        TunableVectorYZ::initTuneParam(param);
        param.block.y = arg.Ls_out; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }

      void defaultTuneParam(TuneParam &param) const
      {
        TunableVectorYZ::defaultTuneParam(param);
        param.block.y = arg.Ls_out; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

      long long flops() const { return 0; }
      long long bytes() const { return arg.in.Bytes() + arg.out.Bytes(); }
    };

#endif
  void transfer_5d_hh(ColorSpinorField& out, const ColorSpinorField& in, const TrainingParameter<float>& tp, bool dagger) {
#ifdef GPU_DOMAIN_WALL_DIRAC

    checkLocation(out, in); // check all locations match

    switch (checkPrecision(out, in)) {
      case QUDA_HALF_PRECISION:
        {
          size_t m_size = in.X(4)*out.X(4)*sizeof(WilsonMatrix<float>);
          if(tp.get_size()*sizeof(float) != m_size){
            errorQuda("Training Parameter size mismatch %lu neq %lu.\n", tp.get_size()*sizeof(complex<float>), m_size);
          }
          MadwfMlArg<short> arg(out, in, (const WilsonMatrix<float>*)tp.data(), dagger);
          Transfer5d<short, MadwfMlArg<short>> dslash(in, arg);
          dslash.apply(streams[Nstream - 1]);
        }
        break;
      default: errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Mobius dslash has not been built");
#endif
  }
  
  void tensor_5d_hh(ColorSpinorField& out, const ColorSpinorField& in, TrainingParameter<float>& tp) {
#ifdef GPU_DOMAIN_WALL_DIRAC

    checkLocation(out, in); // check all locations match

    switch (checkPrecision(out, in)) {
      case QUDA_HALF_PRECISION:
        {
          size_t m_size = in.X(4)*out.X(4)*sizeof(WilsonMatrix<float>);
          if(tp.get_size()*sizeof(float) != m_size){
            errorQuda("Training Parameter size mismatch %lu neq %lu.\n", tp.get_size()*sizeof(complex<float>), m_size);
          }
          cudaMemsetAsync(tp.data(), 0, m_size, streams[Nstream - 1]);
          MadwfMlArg<short> arg(out, in, (WilsonMatrix<float>*)tp.data());
          Transfer5d<short, MadwfMlArg<short>> dslash(in, arg);
          dslash.apply(streams[Nstream - 1]);
        }
        break;
      default: errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Mobius dslash has not been built");
#endif
  }

  __global__ void axpby_kernel(complex<float>* out_p, int size, 
      complex<float> a, const complex<float>* x_p, 
      complex<float> b, const complex<float>* y_p){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while(index < size){
      out_p[index] += a * x_p[index] + b * y_p[index];
      index += blockDim.x * gridDim.x;
    }
  }

  void axpby(TrainingParameter<float>& out,
      complex<float> a, const TrainingParameter<float>& x,
      complex<float> b, const TrainingParameter<float>& y){
    
    int p_size = out.get_size() / 2; // complex
    constexpr int block_size = 256;
    int grid_size = (p_size + block_size - 1) / block_size;

    axpby_kernel<<<grid_size, block_size, 0, streams[Nstream - 1]>>>((complex<float>*)out.data(), p_size, a, (complex<float>*)x.data(), b, (complex<float>*)y.data());

  }

}
} // namespace quda

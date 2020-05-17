#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <launch_kernel.cuh>
#include <kernels/project_reduce.cuh>
#include <jitify_helper.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, typename Arg> class ProjectReduceCompute : TunableLocalParity
  {
    Arg &arg;
    const complex<Float> *contractions;
    
  private:
    bool tuneSharedBytes() const { return true; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }
    unsigned int tuneBlockDimMultiple() const { return x.X()[0]*x.X()[1]*x.X()[2]/2; } // Only tune multiples 32 that are factors of this number
    
  public:
    ProjectReduceCompute(Arg &arg, const complex<Float> *contractions) :
      TunableLocalParity(),
      arg(arg),
      contractions(contractions),
    {
      strcat(aux, "project_reduce,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/project_reduce.cuh");
#endif
    }
    
    void apply(const cudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
	for (int i=0; i< 2*4*Arg::t; i++) ((double*)arg.result_h)[i] = 0.0; 
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	
#ifdef JITIFY
	std::string function_name = "quda::spin";
	
        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	LAUNCH_KERNEL_LOCAL_PARITY(computeProjectReduce, (*this), tp, stream, arg, Arg);
#endif
      } else {
        errorQuda("CPU not supported yet\n");
      }
    }

    virtual ~ProjectReduceCompute() {}
    
    TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(*this).name(), aux); }

    void preTune() {}
    void postTune() {}
    
    long long flops() const
    {
      // 4 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 4 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };
  
  template <typename Float, int t>
  void projectReduce(Float *result, complex<Float> *contractions)
  {
    double timer = -clock();
    ProjectReduceArg<Float, t> arg(result, contractions);
    ProjectReduceCompute<Float, ProjectReduceArg<Float, t>> project_reduce(contractions);
    timer += clock();
    //printfQuda("time1 = %e\n", timer/CLOCKS_PER_SEC);

    timer = -clock();
    project_reduce.apply(0);
    qudaDeviceSynchronize();
    timer += clock();
    //printfQuda("time2 = %e\n", timer/CLOCKS_PER_SEC);

    timer = -clock();
    //comm_allreduce_array((double*)arg.result_h, 8*t);
    timer += clock();
    //printfQuda("time3 = %e\n", timer/CLOCKS_PER_SEC);

    timer = -clock();
    for(int i=0; i<2*4*t; i++) result[i] = ((double*)arg.result_h)[i];
    timer += clock();
    //printfQuda("time4 = %e\n", timer/CLOCKS_PER_SEC);

  }
  
  void projectReduceQuda(void *result, void *contractions, const int t_dim_size, const QudaPrecision prec)
  {

    switch(t_dim_size) {

    case 24 :
      if (prec == QUDA_SINGLE_PRECISION) {
	projectReduce<float, 24>((float*)result, (complex<float>*)contractions);
      } else if (prec == QUDA_DOUBLE_PRECISION) {
	projectReduce<float, 24>((double*)result, (complex<double>*)contractions);
      } else {
	errorQuda("Precision %d not supported", prec);
      }
      break;

      /*
    case 48 :
      if (prec == QUDA_SINGLE_PRECISION) {
	projectReduce<float, 48>((float*)result, (complex<float>*)contractions);
      } else if (prec == QUDA_DOUBLE_PRECISION) {
	projectReduce<float, 48>((double*)result, (complex<double>*)contractions);
      } else {
	errorQuda("Precision %d not supported", prec);
      }
      break;


    case 96 :
      if (prec == QUDA_SINGLE_PRECISION) {
	projectReduce<float, 96>((float*)result, (complex<float>*)contractions);
      } else if (prec == QUDA_DOUBLE_PRECISION) {
	projectReduce<float, 96>((double*)result, (complex<double>*)contractions);
      } else {
	errorQuda("Precision %d not supported", prec);
      }
      break;
      */
    default :
      errorQuda("Uninstantiated T dim %d", t_dim_size);
      
    }
  }
} // namespace quda

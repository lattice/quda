#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <launch_kernel.cuh>
#include <contract_quda.h>
#include <jitify_helper.cuh>
#include <kernels/contraction.cuh>

namespace quda {

#ifdef GPU_CONTRACT
  template <typename real, typename Arg> class Contraction : TunableVectorY
  {
protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;

private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    Contraction(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, const QudaContractType cType) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      y(y),
      cType(cType)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN: strcat(aux, "open,"); break;
      case QUDA_CONTRACT_TYPE_DR: strcat(aux, "degrand-rossi,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/contraction.cuh");
#endif
    }
    virtual ~Contraction() {}

    void apply(const qudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        std::string function_name;
        switch (cType) {
        case QUDA_CONTRACT_TYPE_OPEN: function_name = "quda::computeColorContraction"; break;
        case QUDA_CONTRACT_TYPE_DR: function_name = "quda::computeDegrandRossiContraction"; break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }

        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
                         .instantiate(Type<real>(), Type<Arg>())
                         .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                         .launch(arg);
#else
        switch (cType) {
        case QUDA_CONTRACT_TYPE_OPEN: computeColorContraction<real><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        case QUDA_CONTRACT_TYPE_DR:
          computeDegrandRossiContraction<real><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
          break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }
#endif
      } else {
        errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(*this).name(), aux); }

    void preTune() {}
    void postTune() {}

    long long flops() const
    {
      if (cType == QUDA_CONTRACT_TYPE_OPEN)
        return 16 * 3 * 6ll * x.Volume();
      else
        return ((16 * 3 * 6ll) + (16 * (4 + 12))) * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<real>);
    }
  };

  template <typename Arg> class ContractionSumCompute : TunableLocalParity
  {
    //- The protected members of the class are the argument structure we have already seen, 
    //- the memory addresses of the two fermion fields to be contracted, and the type of 
    //- contraction to compute.
  protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;

    //- The private members are explicit definitions of helper functions defined virtually 
    //- in the Tunable class. We must define them explicitly whenever we create a class 
    //- that inherits from Tunable. 
  private:
    //- In this particular case, we instruct the autotuner to not tune the grid dimensions because
    //- we are going to use the grid dimension to denote which timeslice we are on.
    bool staticGridDim() const { return true; } // Maintain grid dims set in this class.

    //- We also request that there be a minimum number of threads, given by arg.threads. 
    //- This value was set when we created the ContractionArg structure. It was inferred 
    //- from the checkerboard volume of one of the lattice fermions. Make sure you can 
    //- identify in the body of the ContractionSumArg code where this happened. 
    unsigned int minThreads() const { return arg.threads; }

    //- We no not tune shared bytes in the kernel as there is no locality to exploit
    bool tuneSharedBytes() const { return false; }

    //- We initialise the tuning parameter object by explitly declaring that there will be
    //- two parities, denoted by the block index in the .y dim. NB, this is not referencing
    //- the `y` vector of our data, rather this is a CUDA specific way of splitting up the 
    //- the block, threads, and grids in a 3D (x,y,z) fashion. We use the .z dim of the grids
    //- to denote which timeslice we are working on. This is accessed by a handy member
    //- function of the ColorSpinorField object x, namely `.X(3)`.
    void initTuneParam(TuneParam &param) const {
      TunableLocalParity::initTuneParam(param);
      param.block.y = 2;
      param.grid.z = x.X(3); // T dimension is mapped to different blocks in the Z dimension
    }

    //- This is just making sure the Tunable class defaults to the same params.
    void defaultTuneParam(TuneParam &param) const {
      TunableLocalParity::defaultTuneParam(param);
      param.block.y = 2;
      param.grid.z = x.X(3); // T dimension is mapped to different blocks in the Z dimension
    }

    //- Next is the all important public interface of the class. The first method is the 
    //- constructor. This is what was called when we created the `ContractionSumCompute` 
    //- object named `contract_with_sum`.
    //- It is fairly standard for a constructor. We initialise the class from which we 
    //- inherit `TunableLocalParity()` which just means that all the data the thread uses 
    //- is local (no collecting data from other lattice points) and that we may also split the
    //- threads by parity. That particular class of `Tunable` will then tune for that (and only
    //- that) type of calculation. We then initialise the arg, teh x and y vectors, and 
    //- contraction type. 
  public:
    ContractionSumCompute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, const QudaContractType cType) :
      TunableLocalParity(),
      arg(arg),
      x(x),
      y(y),
      cType(cType)
    {
      //- The object aux is a member of the Tunable class. It is simply a string that defines 
      //- the kernel name and attributes for the autotuner. Once the kernel has tuned, 
      //- the autotuner will save the optimal set up to a file and, if the same kernel 
      //- (as defined by the aux string) is called again, the autotuner will use the 
      //- optimal setup immediately. 
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN_SUM: strcat(aux, "open-summed,"); break;
      case QUDA_CONTRACT_TYPE_DR_SUM: strcat(aux, "degrand-rossi-summed,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      strcat(aux, x.AuxString());
      //- The part that is encapsulated by the JITIFY define is an instruction to the compiler 
      //- that the kernel is to be compiled at run time, or Just In Time.
#ifdef JITIFY
      create_jitify_program("kernels/contraction.cuh");
#endif
    }
    //- The destructor is about as boiler plate as it gets.
    virtual ~ContractionSumCompute() {}
    //- Please go back to the creation of the `contract_with_sum` oject below to follow the rest
    //- of the workflow.
    
    //- Ok, here we go, nearly there. We will now apply the kernel computation we have 
    //- meticulously and assiduously constrcuted. You'll remember that the argument we passed
    //- to `apply` was `0`. That just means `please use the 0 stream in this function. This 
    //- is the stream QUDA reserves for main program workflow. Other streams are availible,
    //- but that is an advanced topic for later exploration.
    //- Before we go any further we must make note of something very important. `apply` is NOT
    //- a member function of this class. It is a pure virtual member function of the Tunable class.
    //- By calling this member function, we have passed control of workflow to the parent class
    //- namely TunableLocalParity.
    void apply(const qudaStream_t &stream)
    {
      //- QUDA is evolving so that the C++ in the kernels can be compiled by both NVCC and 
      //- possibly a host compiler such as gcc with parallel stl compatibility. Therefore, 
      //- we place the CUDA specific code under a boolean x.Location() == QUDA_CUDA_FIELD_LOCATION 
      //- so we know we are using a GPU.
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
	//- Ensure the reduce array is zeroed out. This is the array that comes baked in to the
	//- `ReduceArg` class from which the `arg` structure inherited.
	for (int i=0; i<2*x.Nspin()*x.Nspin()*x.X(3); i++) ((double*)arg.result_h)[i] = 0.0;
	//- We next define a TuneParam object that is used by the Tunable class. It takes 
	//- `this` class (see now what *this means?) and some other dat
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	//- This, again, is for when Just In Time compilation is used. You can skip over this for 
	//- now.
#ifdef JITIFY
        std::string function_name;
        switch (cType) {
	case QUDA_CONTRACT_TYPE_OPEN_SUM: function_name = "quda::computeColorContractionSum"; break;
        case QUDA_CONTRACT_TYPE_DR_SUM: function_name = "quda::computeDegrandRossiContractionSumSingle"; break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }

        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<real>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	//- Here are are at the calling of teh compute kernels, albeit wrapped in Macros! computeColorContractionSum 
	//- will contract only the colour indices, leaving the 16 complex numbers per open spin 
	//- index per lattice site (one for each \mu, and \nu combination) and then sum each one
	//- of those 16 elements with its counterparts on the same timeslice.
	//- computeDegrandRossiContractionSumSingle will insert all 16 Gamma matrices into the
	//- contraction, also giving 16 complex numbers, one for each unique gamma matrix 
	//- combination. It will then do the same timeslice sum. Here we also see how the 
	//- autotuner works. The standard CUDA kernel arguments of blocks, threads, and 
	//- shared memory size are controlled by the TuneParam tp object. The macro 
	//- `LAUNCH_KERNEL_LOCAL_PARITY` will use the tp object to define a set of possible 
	//- configurations, then save the best one.
        switch (cType) {
        case QUDA_CONTRACT_TYPE_OPEN_SUM: 
	  LAUNCH_KERNEL_LOCAL_PARITY(computeColorContractionSum, (*this), tp, stream, arg, Arg); 
	  break;
        case QUDA_CONTRACT_TYPE_DR_SUM:
	  //- This is the kernel we are going to follow. We will NOT go into the Caves of Moria
	  //- hunting the veritable Balrog that is QUDA's tuning class! Rather we will cheat
	  //- slighly and look only at the compute code. Please navigate to 
	  //- quda/include/kernels/contraction.cuh and the function
	  //- `computeDegrandRossiContractionSumSingle` then return here.
	  LAUNCH_KERNEL_LOCAL_PARITY(computeDegrandRossiContractionSumSingle, (*this), tp, stream, arg, Arg);
	  //- Welcome back! We're almost done! Back you go to `contraction_with_sum.apply(0);`
          break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }
#endif
      } else {
        errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(*this).name(), aux); }

    void preTune() {}
    void postTune() {}

    long long flops() const
    {
      if (cType == QUDA_CONTRACT_TYPE_OPEN_SUM)
        return x.Nspin() * x.Nspin() * 3 * 6ll * x.Volume();
      else
        return ((x.Nspin() * x.Nspin() * 3 * 6ll) + (x.Nspin() * x.Nspin() * (4 + 12))) * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes();
    }
  };
  template <typename Arg> class ContractionSumSpatialCompute : TunableLocalParity
  {
  protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;

   private:
    bool staticGridDim() const { return true; } // Maintain grid dims set in this class.
    unsigned int minThreads() const { return arg.threads; }
    bool tuneSharedBytes() const { return false; }

    void initTuneParam(TuneParam &param) const {
      TunableLocalParity::initTuneParam(param);
      param.block.y = 2;
      param.grid.z = x.X(2); // T dimension is mapped to different blocks in the Z dimension
    }

    void defaultTuneParam(TuneParam &param) const {
      TunableLocalParity::defaultTuneParam(param);
      param.block.y = 2;
      param.grid.z = x.X(2); // T dimension is mapped to different blocks in the Z dimension
    }

  public:
    ContractionSumSpatialCompute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, const QudaContractType cType) :
      TunableLocalParity(),
      arg(arg),
      x(x),
      y(y),
      cType(cType)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN_SUM: strcat(aux, "open-summed,"); break;
      case QUDA_CONTRACT_TYPE_DR_SUM: strcat(aux, "degrand-rossi-summed,"); break;
      case QUDA_CONTRACT_TYPE_DR_SUM_SPATIAL: strcat(aux, "degrand-rossi-summed-spacial,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/contraction.cuh");
#endif
    }
    virtual ~ContractionSumSpatialCompute() {}
    void apply(const qudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
        for (int i=0; i<2*x.Nspin()*x.Nspin()*x.X(2); i++) ((double*)arg.result_h)[i] = 0.0;
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        std::string function_name;
        switch (cType) {
	case QUDA_CONTRACT_TYPE_OPEN_SUM: function_name = "quda::computeColorContractionSum"; break;
        case QUDA_CONTRACT_TYPE_DR_SUM: function_name = "quda::computeDegrandRossiContractionSumSingle"; break;
        case QUDA_CONTRACT_TYPE_DR_SUM_SPATIAL: function_name = "quda::computeDegrandRossiContractionSumSpatial"; break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }

        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<real>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
        switch (cType) {
        case QUDA_CONTRACT_TYPE_OPEN_SUM:
          LAUNCH_KERNEL_LOCAL_PARITY(computeColorContractionSum, (*this), tp, stream, arg, Arg);
          break;
        case QUDA_CONTRACT_TYPE_DR_SUM:
          LAUNCH_KERNEL_LOCAL_PARITY(computeDegrandRossiContractionSum, (*this), tp, stream, arg, Arg);
          break;
        case QUDA_CONTRACT_TYPE_DR_SUM_SPATIAL:
          LAUNCH_KERNEL_LOCAL_PARITY(computeDegrandRossiContractionSum, (*this), tp, stream, arg, Arg);
          break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }
#endif
      } else {
        errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(*this).name(), aux); }

    void preTune() {}
    void postTune() {}

    long long flops() const
    {
      if (cType == QUDA_CONTRACT_TYPE_OPEN_SUM || cType == QUDA_CONTRACT_TYPE_DR_SUM_SPATIAL)
        return x.Nspin() * x.Nspin() * 3 * 6ll * x.Volume();
      else
        return ((x.Nspin() * x.Nspin() * 3 * 6ll) + (x.Nspin() * x.Nspin() * (4 + 12))) * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes();
    }
  };

  template <typename real>
  void contract_quda(const ColorSpinorField &x, const ColorSpinorField &y, const size_t s1, const size_t c1,
                     const size_t s2, const size_t c2, complex<real> *result, const QudaContractType cType)
  {
    if(cType == QUDA_CONTRACT_TYPE_OPEN_SUM || cType == QUDA_CONTRACT_TYPE_DR_SUM) {
      ContractionSumArg<real> arg(x, y, s1, s2, c1, c2);
      ContractionSumCompute<decltype(arg)> contraction_with_sum(arg, x, y, cType);
      contraction_with_sum.apply(0);
      qudaDeviceSynchronize();

      // Copy timeslice sums back to device
      auto *res = (double*)arg.result_h;
      for (int i=0; i<x.Nspin()*x.Nspin()*x.X(3); i++) result[i] = complex<real>(res[2*i], res[2*i+1]);
      // Head on back to your place in contractQuda to finish up.

    } else if (cType == QUDA_CONTRACT_TYPE_DR_SUM_SPATIAL){
      ContractionSumSpatialArg<real> arg(x, y);
      ContractionSumSpatialCompute<decltype(arg)> contraction_with_sum_spatial(arg, x, y, cType);
      contraction_with_sum_spatial.apply(0);
      qudaDeviceSynchronize();
      auto *res = (double*)arg.result_h;
      for (int i=0; i<x.Nspin()*x.Nspin()*x.X(2); i++) result[i] = complex<real>(res[2*i], res[2*i+1]);
    } else {
      ContractionArg<real> arg(x, y, result);
      Contraction<real, ContractionArg<real>> contraction(arg, x, y, cType);
      contraction.apply(0);
      qudaDeviceSynchronize();
    }
  }

#endif

  //- The interface function has called the GPU function contractQuda. The first thing to
  //- notice is that the body of the code is encapsulated in a preprocessor definition,
  //- which one can adjust at CMake configure time. If the QUDA_CONTRACT CMake option
  //- is not set, none of the following code will be compiled. This is an important
  //- feature to include as it keeps compile time to a minimum.
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, const size_t s1, const size_t c1, const size_t s2, const size_t c2, void *result, const QudaContractType cType)
  {
#ifdef GPU_CONTRACT
    //- After some checks, 
    checkPrecision(x, y);

    if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    //- we see that the contract_quda function is instantiated on the desired precision. Go to
    //- the `contract_quda` and then return here.
    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      contract_quda<float>(x, y, s1, c1, s2, c2, (complex<float> *)result, cType);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      contract_quda<double>(x, y, s1, c1, s2, c2, (complex<double> *)result, cType);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
    //- All done, head on back to quda/lib/interface_quda.cpp to finish up.
#else
    errorQuda("Contraction code has not been built");
#endif

  }
} // namespace quda

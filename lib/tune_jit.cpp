#include <fstream>
#include <cxxabi.h>
#include <nvrtc.h>

namespace quda {

#define NVRTC_SAFE_CALL(Name, x)					\
  do {									\
  nvrtcResult result = x;						\
  if (result != NVRTC_SUCCESS) {					\
    errorQuda("%s failed with error ", nvrtcGetErrorString(result));	\
  }									\
  } while(0)

#define TYPE_STR_LENGTH 128

  /**
     This derived class is for algorithms that utilize run-time compilation
   */
  class TunableJIT : public Tunable {

  protected:
    void getTypeString(char *type_string, const std::type_info &T) {

      std::string mangled_name = T.name();
      
      int status = -1;
      char *result = abi::__cxa_demangle(mangled_name.c_str(), 0, 0, &status);
      if (status != 0) errorQuda("Name demangling failed with error %d", status);
      
      strcpy(type_string, result);
      free(result);
    }

    void compileToPTX(const char *fname, const char *filename, 
		      char **ptxResult, size_t *ptxResultSize, int n_options, char **options)
    {
      std::ifstream inputFile(filename, std::ios::in | std::ios::binary | std::ios::ate);

      if (!inputFile.is_open()) {
	std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
	exit(1);
      }

      std::streampos pos = inputFile.tellg();
      size_t inputSize = (size_t)pos;
      char * memBlock = new char [inputSize + 1];

      inputFile.seekg (0, std::ios::beg);
      inputFile.read (memBlock, inputSize);
      inputFile.close();
      memBlock[inputSize] = '\x0';

      nvrtcProgram prog;
      NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, memBlock,
							       fname, 0, NULL, NULL));
      nvrtcResult res = nvrtcCompileProgram(prog, n_options, (const char**)options);

      // dump log if compilation fails                                                                                                                                                                                                                                            
      if (res != NVRTC_SUCCESS) {
	size_t logSize;
	NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &logSize));
	char *log = (char *) malloc(sizeof(char) * logSize + 1);
	nvrtcResult compileResult = nvrtcGetProgramLog(prog, log);

	log[logSize] = '\x0';
	std::cerr << "Compilation failed with...\n";
	std::cerr << log;
	free(log);

	exit(0);
      }

      // fetch PTX                                                                                                                                                                                                                                                                
      size_t ptxSize;
      NVRTC_SAFE_CALL("nvrtcGetPTXSize", nvrtcGetPTXSize(prog, &ptxSize));
      char *ptx = (char *) malloc(sizeof(char) * ptxSize);
      NVRTC_SAFE_CALL("nvrtcGetPTX", nvrtcGetPTX(prog, ptx));
      NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));
      *ptxResult = ptx;
      *ptxResultSize = ptxSize;
    }

  };

} // namespace quda

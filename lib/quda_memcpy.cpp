#include <tune_quda.h>

namespace quda {

  class QudaMemCopy : public Tunable {

    void *dst;
    const void *src;
    const size_t count;
    const cudaMemcpyKind kind;
    const char *name;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    QudaMemCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
		const char *func, const char *file, int line)
      : dst(dst), src(src), count(count), kind(kind) {

      switch(kind) {
      case cudaMemcpyDeviceToHost:
	name = "cudaMemcpyDeviceToHost";
	break;
      case cudaMemcpyHostToDevice:
	name = "cudaMemcpyHostToDevice";
	break;
      case cudaMemcpyHostToHost:
	name = "cudaMemcpyHostToHost";
	break;
      case cudaMemcpyDeviceToDevice:
	name = "cudaMemcpyDeviceToDevice";
	break;
      default:
	errorQuda("Unsupported cudaMemcpyType %d", kind);
      }
      sprintf(aux, "%s,%s:%d", func, file, line);
    }

    virtual ~QudaMemCopy() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      cudaMemcpy(dst, src, count, kind);
    }

    bool advanceTuneParam(TuneParam &param) const { return false; }

    TuneKey tuneKey() const {
      char vol[128];
      sprintf(vol, "bytes=%llu", (long long unsigned int)count);
      return TuneKey(vol, name, aux);
    }

    long long flops() const { return 0; }
    long long bytes() const { return kind == cudaMemcpyDeviceToDevice ? 2*count : count; }

  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
		  const char *func, const char *file, int line) {
    if (getVerbosity() == QUDA_DEBUG_VERBOSE)
      printfQuda("%s bytes = %llu\n", __func__, (long long unsigned int)count);

    if (count == 0) return;
    QudaMemCopy copy(dst, src, count, kind, func, file, line);
    copy.apply(0);
    checkCudaError();
  }

} // namespace quda

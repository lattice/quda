#include <tune_quda.h>

namespace quda {

  class QudaMemCopy : public Tunable {

    void *dst;
    const void *src;
    const size_t count;
    const cudaMemcpyKind kind;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    QudaMemCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
		const char *func, const char *file, int line)
      : dst(dst), src(src), count(count), kind(kind) {

      switch(kind) {
      case cudaMemcpyDeviceToHost:
	sprintf(aux, "cudaMemcpyDeviceToHost");
	break;
      case cudaMemcpyHostToDevice:
	sprintf(aux, "cudaMemcpyHostToDevice");
	break;
      case cudaMemcpyHostToHost:
	sprintf(aux, "cudaMemcpyHostToHost");
	break;
      case cudaMemcpyDeviceToDevice:
	sprintf(aux, "cudaMemcpyDeviceToDevice");
	break;
      default:
	errorQuda("Unsupported cudaMemcpyType %d", kind);
      }
      sprintf(aux, "%s,%s,%s:%d", aux, func, file, line);
    }

    virtual ~QudaMemCopy() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      cudaMemcpy(dst, src, count, kind);
    }

    bool advanceTuneParam(TuneParam &param) const { return false; }

    TuneKey tuneKey() const {
      char vol[128];
      sprintf(vol, "bytes=%u", (unsigned int)count);
      return TuneKey(vol, "cudaMemcpy", aux);
    }

    long long flops() const { return 0; }
    long long bytes() const { return kind == cudaMemcpyDeviceToDevice ? 2*count : count; }

  };

  void qudaMemcpy_(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
		  const char *func, const char *file, int line) {
    if (count == 0) return;
    QudaMemCopy copy(dst, src, count, kind, func, file, line);
    copy.apply(0);
  }

} // namespace quda

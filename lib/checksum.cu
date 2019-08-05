#include <gauge_field_order.h>
#include <cub_helper.cuh>

namespace quda {

  template <typename T, QudaGaugeFieldOrder order, int Nc>
  struct ChecksumArg {
    static constexpr int nColor = Nc;
    typedef typename mapper<T>::type real;
    typedef typename gauge_order_mapper<T,order,Nc>::type G;
    const G U;
    const int volumeCB;
    ChecksumArg(const GaugeField &U, bool mini) : U(U), volumeCB(mini ? 1 : U.VolumeCB()) { }
  };

  template <typename Arg>
  __device__ __host__ inline uint64_t siteChecksum(const Arg &arg, int d, int parity, int x_cb) {
    const Matrix<complex<typename Arg::real>,Arg::nColor> u = arg.U(d, x_cb, parity);
    return u.checksum(); 
  }

  template <typename Arg>
  uint64_t ChecksumCPU(const Arg &arg)
  {
    uint64_t checksum_ = 0;
    for (int parity=0; parity<2; parity++)
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++)
	for (int d=0; d<arg.U.geometry; d++)
	  checksum_ ^= siteChecksum(arg, d, parity, x_cb);
    return checksum_;
  }

  template <typename T, int Nc>
  uint64_t Checksum(const GaugeField &u, bool mini)
  {
    uint64_t checksum = 0;
    if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      ChecksumArg<T,QUDA_QDP_GAUGE_ORDER,Nc> arg(u,mini);
      checksum = ChecksumCPU(arg);
    } else if (u.Order() == QUDA_QDPJIT_GAUGE_ORDER) {
      ChecksumArg<T,QUDA_QDPJIT_GAUGE_ORDER,Nc> arg(u,mini);
      checksum = ChecksumCPU(arg);
    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      ChecksumArg<T,QUDA_MILC_GAUGE_ORDER,Nc> arg(u,mini);
      checksum = ChecksumCPU(arg);
    } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {
      ChecksumArg<T,QUDA_BQCD_GAUGE_ORDER,Nc> arg(u,mini);
      checksum = ChecksumCPU(arg);
    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {
      ChecksumArg<T,QUDA_TIFR_GAUGE_ORDER,Nc> arg(u,mini);
      checksum = ChecksumCPU(arg);
    } else if (u.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
      ChecksumArg<T,QUDA_TIFR_PADDED_GAUGE_ORDER,Nc> arg(u,mini);
      checksum = ChecksumCPU(arg);
    } else {
      errorQuda("Checksum not implemented");
    }    

    return checksum;
  }

  template <typename T>
  uint64_t Checksum(const GaugeField &u, bool mini)
  {
    uint64_t checksum = 0;
    switch (u.Ncolor()) {
    case 3: checksum = Checksum<T,3>(u,mini); break;
    default: errorQuda("Unsupported nColor = %d", u.Ncolor());
    }
    return checksum;
  }

  uint64_t Checksum(const GaugeField &u, bool mini)
  {
    uint64_t checksum = 0;
    switch (u.Precision()) {
    case QUDA_DOUBLE_PRECISION: checksum = Checksum<double>(u,mini); break;
    case QUDA_SINGLE_PRECISION: checksum = Checksum<float>(u,mini); break;
    default: errorQuda("Unsupported precision = %d", u.Precision());
    }

    comm_allreduce_xor(&checksum);

    return checksum;
  }

}

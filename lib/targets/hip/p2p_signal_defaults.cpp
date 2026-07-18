// lib/targets/hip/p2p_signal_defaults.cpp
//
// HIP backend implementation of the QudaP2PSignal allow-list and default.
// HIP currently lacks a confirmed equivalent of cuStreamWriteValue64 /
// cuStreamWaitValue64 (hipStreamWriteValue64 exists in newer ROCm but the
// stream-mem-op-gated halo path is not yet wired in lib/targets/hip/).
// So STREAM_GATED is unsupported and the default is REMOTE_EVENT (event-based,
// cudaIPC analogue via hipIpc).  Once a HIP impl of the stream-mem-op path
// lands in this directory, supported() can flip true for STREAM_GATED and
// p2p_signal_default() can return it.

#include <comm_quda.h>

namespace quda::comm
{
  bool p2p_signal_supported(QudaP2PSignal kind)
  {
    switch (kind) {
    case QudaP2PSignal::REMOTE_EVENT: return true;
    case QudaP2PSignal::STREAM_GATED: return false; // not yet wired on HIP
    }
    return false;
  }

  QudaP2PSignal p2p_signal_default()
  {
    // Same clamping rule as the CUDA backend: prefer events, fall back to
    // stream-gated only where events are unsupported.  On HIP events are
    // always supported, so this is REMOTE_EVENT.
    return p2p_signal_supported(QudaP2PSignal::REMOTE_EVENT) ? QudaP2PSignal::REMOTE_EVENT : QudaP2PSignal::STREAM_GATED;
  }
} // namespace quda::comm

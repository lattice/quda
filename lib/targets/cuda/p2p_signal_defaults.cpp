// lib/targets/cuda/p2p_signal_defaults.cpp
//
// Backend implementation of the QudaP2PSignal allow-list and default.
//
// CUDA backend rules:
//   - MNNVL build (QUDA_MNNVL):  STREAM_GATED only.  cudaIPC event handles
//     cannot traverse the MNNVL fabric; supported() returns false for
//     REMOTE_EVENT, and default() returns STREAM_GATED.
//   - Non-MNNVL build:           both supported, and default() returns
//     REMOTE_EVENT (events) -- the established develop behaviour, least surprise.
//     Stream-gating is available opt-in via QUDA_P2P_TRANSPORT=stream_gated.

#include <comm_quda.h>

namespace quda::comm
{
  bool p2p_signal_supported(QudaP2PSignal kind)
  {
    switch (kind) {
    case QudaP2PSignal::REMOTE_EVENT:
#ifdef QUDA_MNNVL
      // cudaIPC events do not cross the MNNVL NVLink fabric.  Trying to
      // exchange ipcEventHandles via cudaIpcGetEventHandle / OpenEventHandle
      // across nodes in a fabric clique returns an error at handle-open time.
      return false;
#else
      return true;
#endif
    case QudaP2PSignal::STREAM_GATED: return true;
    }
    return false;
  }

  QudaP2PSignal p2p_signal_default()
  {
#ifdef QUDA_MNNVL
    // REMOTE_EVENT (events) is unsupported across the MNNVL fabric, so STREAM_GATED
    // is the only choice.
    return QudaP2PSignal::STREAM_GATED;
#else
    // Default to REMOTE_EVENT (events) to match develop behaviour; STREAM_GATED is
    // opt-in via QUDA_P2P_TRANSPORT=stream_gated (validated by the resolver against
    // p2p_signal_supported()).
    return QudaP2PSignal::REMOTE_EVENT;
#endif
  }
} // namespace quda::comm

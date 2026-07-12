#include <memory>
#include <tune_quda.h>
#include <index_helper.cuh>
#include <timer.h>
#include <dslash_quda.h>
#include <dslash_shmem.h>

namespace quda
{

  namespace dslash
  {

    extern int it;

    extern qudaEvent_t packEnd[]; // double buffered
    extern qudaEvent_t gatherEnd[];
    extern qudaEvent_t scatterEnd[];
    extern qudaEvent_t dslashStart[]; // double buffered

    // FIX this is a hack from hell
    // Auxiliary work that can be done while waiting on comms to finish
    extern Worker *aux_worker;

    // these variables are used for benchmarking the dslash components in isolation
    extern bool dslash_pack_compute;
    extern bool dslash_interior_compute;
    extern bool dslash_exterior_compute;
    extern bool dslash_comms;
    extern bool dslash_copy;
    static ColorSpinorField *inSpinor;

    /**
     * Arrays used for the dynamic scheduling.
     */
    struct DslashCommsPattern {
      static constexpr int nDim = 4;
      static constexpr int nDir = 2;
      // nStream here is not tied to the number underlying CUDA
      // streams (although historically it was), rather it signifies
      // the number of logical parallel streams we have when
      // overlapping comms in 4 dimensions * 2 directions and compute
      static constexpr int nStream = nDim * nDir + 1;

      std::array<int, nStream> gatherCompleted;
      std::array<int, nStream> previousDir;
      std::array<int, nStream> commsCompleted;
      std::array<int, nStream> dslashCompleted;
      int commDimTotal;
      int completeSum;

      inline DslashCommsPattern(const int commDim[], bool gdr_send = false) :
        commsCompleted {},
        dslashCompleted {},
        completeSum(0)
      {

        for (int i = 0; i < nStream - 1; i++) gatherCompleted[i] = gdr_send ? 1 : 0;
        gatherCompleted[nStream - 1] = 1;
        commsCompleted[nStream - 1] = 1;
        dslashCompleted[nStream - 1] = 1;

        //   We need to know which was the previous direction in which
        //   communication was issued, since we only query a given event /
        //   comms call after the previous the one has successfully
        //   completed.
        for (int i = 3; i >= 0; i--) {
          if (commDim[i]) {
            int prev = nStream - 1;
            for (int j = 3; j > i; j--)
              if (commDim[j]) prev = 2 * j;
            previousDir[2 * i + 1] = prev;
            previousDir[2 * i + 0] = 2 * i + 1; // always valid
          }
        }

        // this tells us how many events / comms occurances there are in
        // total.  Used for exiting the while loop
        commDimTotal = 0;
        for (int i = 3; i >= 0; i--) { commDimTotal += commDim[i]; }
        commDimTotal *= gdr_send ? 2 : 4; // 2 from pipe length, 2 from direction
      }
    };

    template <typename Arg, typename Dslash>
    inline void setFusedParam(Arg &param, Dslash &dslash, const ColorSpinorField &halo)
    {
      int prev = -1;

      param.threads = 0;
      for (int i = 0; i < 4; ++i) {
        param.threadDimMapLower[i] = 0;
        param.threadDimMapUpper[i] = 0;
        if (!dslash.dslashParam.commDim[i]) continue;
        param.threadDimMapLower[i] = (prev >= 0 ? param.threadDimMapUpper[prev] : 0);
        param.threadDimMapUpper[i]
          = param.threadDimMapLower[i] + dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i];
        param.threads = param.threadDimMapUpper[i];
        prev = i;
      }

      param.exterior_threads = param.threads;
      param.kernel_type = EXTERIOR_KERNEL_ALL;
    }

#undef DSLASH_PROFILE
#ifdef DSLASH_PROFILE
#define PROFILE(f, profile, idx)		\
  profile.TPSTART(idx);				\
  f;						\
  profile.TPSTOP(idx);
#define PROFILE_START(idx) profile.TPSTART(idx);
#define PROFILE_STOP(idx) profile.TPSTOP(idx);
#else
#define PROFILE(f, profile, idx) f;
#define PROFILE_START(profile, idx) (void)profile;
#define PROFILE_STOP(profile, idx)
#endif


  /**
     @brief This helper function simply posts all receives in all directions
     @param[out] input Field that we are doing halo exchange
     @param[in] dslash The dslash object
     @param[in] stream Stream were the receive is being posted (effectively ignored)
     @param[in] gdr Whether we are using GPU Direct RDMA or not
  */
    template <typename Dslash>
    inline void issueRecv(const ColorSpinorField &halo, const Dslash &dslash, bool gdr, bool skip_p2p_recv = false)
    {
      for (int i = 3; i >= 0; i--) {
        if (!dslash.dslashParam.commDim[i]) continue;
        for (int dir = 1; dir >= 0; dir--) {
          // When the caller (DslashBasic w/ stream-gated P2P) handles the P2P-able
          // directions via stream-mem-op signalling, skip the MPI recv doorbell post
          // here.  Posting it without a matching MPI_Wait would leave the request
          // active and trigger MPI_ERR_REQUEST on the next iteration.
          if (skip_p2p_recv && comm_peer2peer_enabled(1 - dir, i)) continue;
          PROFILE(if (dslash_comms) halo.recvStart(2 * i + dir, device::get_stream(2 * i + dir), gdr), profile,
                  QUDA_PROFILE_COMMS_START);
        }
      }
  }

  /**
     @brief This helper function simply posts the packing kernel needed for halo exchange
     @param[out] in Field that we are packing
     @param[in] dslash The dslash object
     @param[in] parity Field parity
     @param[in] location Memory location where we are packing to
     - if Host is requested, the non-p2p halos will be sent to host
     - if Remote is requested, the p2p halos will be written directly
     @param[in] packIndex Stream index where the packing kernel will run
  */
  template <typename Dslash>
  inline void issuePack(const ColorSpinorField &halo, cvector_ref<const ColorSpinorField> &in, const Dslash &dslash,
                        int parity, MemoryLocation location, int packIndex, int shmem = 0)
  {
    auto &arg = dslash.dslashParam;
    if ( (location & Device) & Host) errorQuda("MemoryLocation cannot be both Device and Host");

    bool pack = false;
    for (int i=3; i>=0; i--)
      if (arg.commDim[i]) {
        pack = true;
        break;
      }

    MemoryLocation pack_dest[2*QUDA_MAX_DIM];
    for (int dim=0; dim<4; dim++) {
      for (int dir=0; dir<2; dir++) {
        if ((location & Shmem)) {
          pack_dest[2 * dim + dir] = Shmem; // pack to p2p remote
        } else if ((location & Remote) && comm_peer2peer_enabled(dir, dim)) {
          pack_dest[2*dim+dir] = Remote; // pack to p2p remote
        } else if (location & Host && !comm_peer2peer_enabled(dir, dim)) {
          pack_dest[2*dim+dir] = Host;   // pack to cpu memory
        } else {
          pack_dest[2*dim+dir] = Device; // pack to local gpu memory
        }
      }
    }
    if (pack) {
      PROFILE(if (dslash_pack_compute)
                halo.pack(dslash.Nface() / 2, parity, dslash.Dagger(), device::get_stream(packIndex), pack_dest,
                          location, arg.spin_project, arg.twist_a, arg.twist_b, arg.twist_c, shmem, in),
              profile, QUDA_PROFILE_PACK_KERNEL);

      // Record the end of the packing
      PROFILE(if (location != Host) qudaEventRecord(packEnd[halo.bufferIndex], device::get_stream(packIndex)), profile,
              QUDA_PROFILE_EVENT_RECORD);
    }
  }

  /**
     @brief This helper function simply posts the device-host memory
     copies of all halos in all dimensions and directions
     @param[out] in Field that whose halos we are communicating
     @param[in] dslash The dslash object
  */
  template <typename Dslash> inline void issueGather(const ColorSpinorField &halo, const Dslash &dslash)
  {

    for (int i = 3; i >=0; i--) {
      if (!dslash.dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) { // forwards gather
        auto &event = packEnd[halo.bufferIndex];

        PROFILE(qudaStreamWaitEvent(device::get_stream(2*i+dir), event, 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

        // Initialize host transfer from source spinor
        PROFILE(if (dslash_copy) halo.gather(2 * i + dir, device::get_stream(2 * i + dir)), profile, QUDA_PROFILE_GATHER);

        // Record the end of the gathering if not peer-to-peer
	if (!comm_peer2peer_enabled(dir,i)) {
	  PROFILE(qudaEventRecord(gatherEnd[2*i+dir], device::get_stream(2*i+dir)), profile, QUDA_PROFILE_EVENT_RECORD);
	}
      }
    }
  }

  /**
     @brief Returns a stream index for posting the pack/scatters to.
     We desire a stream index that is not being used for peer-to-peer
     communication.  This is used by the fused halo dslash kernels
     where we post all scatters to the same stream so we only have a
     single event to wait on before the exterior kernel is applied,
     and by the zero-copy dslash kernels where we want to post the
     packing kernel to an unused stream.
     @return stream index
  */
  template <typename T>
  inline int getStreamIndex(const T &dslashParam) {
    // set index to a stream index not being used for p2p
    int index = -1;
    for (int i = 3; i >=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      if (!comm_peer2peer_enabled(0,i)) index = 2*i+0;
      else if (!comm_peer2peer_enabled(1,i)) index = 2*i+1;
    }
    // make sure we pick a valid index, in case we are fully p2p connected
    if (index == -1) index = 0;
    return index;
  }

  /**
     @brief Wrapper for querying if communication is finished in the
     dslash, and if it is take the appropriate action:

     - if peer-to-peer then we now know that the peer-to-peer copy is
       now in flight and we are safe to post the cudaStreamWaitEvent
       in our GPU context

     - if gdr or zero-copy for the receive buffer then we have nothing
       else to do, it is now safe to post halo kernel

     - if basic staging, we post the scatter (host to device memory copy)

     @param[in,out] in Field being commicated
     @param[in] dslash The dslash object (unused)
     @param[in] dim Dimension we are working on
     @param[in] dir Direction we are working on
     @param[in] gdr_send Whether GPU Direct RDMA is being used for sending
     @param[in] gdr_recv Whether GPU Direct RDMA is being used for receiving
     @param[in] zero_copy_recv Whether we are using zero-copy on the
     receive end (and hence do not need to do CPU->GPU copy)
     @param[in] scatterIndex The stream index used for posting the host-to-device memory copy in
   */
  template <typename Dslash>
  inline bool commsComplete(const ColorSpinorField &halo, const Dslash &, int dim, int dir, bool gdr_send,
                            bool gdr_recv, bool zero_copy_recv, int scatterIndex = -1)
  {
    PROFILE(int comms_test
            = dslash_comms ? halo.commsQuery(2 * dim + dir, device::get_stream(2 * dim + dir), gdr_send, gdr_recv) : 1,
            profile, QUDA_PROFILE_COMMS_QUERY);
    if (comms_test) {
      // now we are receive centric
      int dir2 = 1-dir;

      // if peer-2-peer in a given direction then we need to insert a wait on that copy event
      if (comm_peer2peer_enabled(dir2,dim)) {
        PROFILE(qudaStreamWaitEvent(device::get_default_stream(), halo.getIPCRemoteCopyEvent(dir2, dim), 0), profile,
                QUDA_PROFILE_STREAM_WAIT_EVENT);
      } else {

        if (scatterIndex == -1) scatterIndex = 2 * dim + dir;

        if (!gdr_recv && !zero_copy_recv) { // Issue CPU->GPU copy if not GDR
          // note the ColorSpinorField::scatter transforms from
          // scatter centric to gather centric (e.g., flips
          // direction) so here just use dir not dir2
          PROFILE(if (dslash_copy) halo.scatter(2 * dim + dir, device::get_stream(scatterIndex)), profile,
                  QUDA_PROFILE_SCATTER);
        }
      }

    }
    return comms_test;
  }

  /**
     @brief Ensure that the dslash is complete.  By construction, the
     dslash will have completed (or is in flight) on this process,
     however, we must also ensure that no local work begins until any
     communication in flight from this process to another has
     completed.  This prevents a race condition where we could start
     updating the local buffers on a subsequent computation before we
     have finished sending.
  */
  template <typename T> inline void completeDslash(const ColorSpinorField &halo, const T &dslashParam)
  {
    // this ensures that the p2p sending is completed before any
    // subsequent work is done on the compute stream
    for (int dim=3; dim>=0; dim--) {
      if (!dslashParam.commDim[dim]) continue;
      for (int dir=0; dir<2; dir++) {
	if (comm_peer2peer_enabled(dir,dim)) {
          PROFILE(qudaStreamWaitEvent(device::get_default_stream(), halo.getIPCCopyEvent(dir, dim), 0), profile,
                  QUDA_PROFILE_STREAM_WAIT_EVENT);
        }
      }
    }
  }

  /**
     Stream-gated analogue of completeDslash.  The stream-gated transport has no
     ipcCopyEvent to wait on (the doorbell+remote-event was replaced by
     WriteValue64/WaitValue64), so completion is a local DRAIN: make the compute
     (default) stream wait for every per-direction P2P stream -- which carries this
     apply's send (memcpy + WriteValue64) and the receive gate (WaitValue64) plus
     the exterior kernels queued after that gate -- to finish.  Without this, the
     next apply can reuse/overwrite the double-buffered ghost while the previous
     apply's P2P traffic is still in flight (monotonic-counter + GEQ wait then reads
     the overwritten halo).  Purely local stream ordering: no IPC event, no host block.
  */
  template <typename T> inline void completeDslashStreamGated(const ColorSpinorField &halo, const T &dslashParam)
  {
    (void)halo;
    // HARD drain: host-wait each per-direction P2P stream so this apply's send
    // (memcpy + WriteValue64) AND its receive gate (WaitValue64, which holds the
    // stream until the peer's write lands) have completed before the next apply is
    // enqueued.  This bounds cross-rank run-ahead to within the double-buffer depth,
    // exactly as the event path's commsComplete host-synced the recv ipcRemoteCopyEvent.
    // An async stream-merge is NOT enough: it orders our own compute stream but lets
    // the peer race >=2 applies ahead and overwrite the ghost mid-read (monotonic
    // counter + GEQ then passes on the overwritten halo).  Sends are already enqueued
    // before this point on both ranks, so the mutual recv waits cannot deadlock.
    bool any_p2p = false;
    for (int dim = 3; dim >= 0; dim--) {
      if (!dslashParam.commDim[dim]) continue;
      for (int dir = 0; dir < 2; dir++) {
        if (comm_peer2peer_enabled(dir, dim)) {
          qudaStreamSynchronize(device::get_stream(2 * dim + dir));
          any_p2p = true;
        }
      }
    }
    // The FUSED policies queue the recv WaitValue64 on the pack/scatter stream (not the
    // per-dir stream), merged into the compute stream via scatterEnd before the fused
    // exterior kernel.  Drain the compute stream too so their receive gate -- the
    // cross-rank pacing point -- is bounded, not just the per-dir sends.
    if (any_p2p) qudaStreamSynchronize(device::get_default_stream());
  }

  /**
     @brief Set the ghosts to the mapped CPU ghost buffer, or unsets
     if already set.  Note this must not be called until after the
     interior dslash has been called, since it sets the peer-to-peer
     ghost pointers, and this need to be done without the mapped ghost
     enabled.

     @param[in,out] dslash The dslash object
     @param[in,out] in The ColorSpinorField source
     @param[in] to_mapped Whether we are switching to mapped ghosts or not
   */
  template <typename Dslash> inline void setMappedGhost(Dslash &dslash, const ColorSpinorField &halo, bool to_mapped)
  {
    static char aux_copy[TuneKey::aux_n];
    static bool set_mapped = false;

    if (to_mapped) {
      if (set_mapped) errorQuda("set_mapped already set");
      // in the below we switch to the mapped ghost buffer and update the tuneKey to reflect this
      halo.bufferIndex += 2;
      strcpy(aux_copy,dslash.getAux(dslash.dslashParam.kernel_type));
      if (comm_peer2peer_enabled_global())
        dslash.augmentAux(dslash.dslashParam.kernel_type, ",zero_copy,p2p=1");
      else
        dslash.augmentAux(dslash.dslashParam.kernel_type, ",zero_copy,p2p=0");
      set_mapped = true;
    } else {
      if (!set_mapped) errorQuda("set_mapped not set");
      // reset to default
      dslash.setAux(dslash.dslashParam.kernel_type, aux_copy);
      halo.bufferIndex -= 2;
      set_mapped = false;
    }
  }

  template <typename Dslash> struct DslashPolicyImp {

    virtual void operator()(Dslash &, cvector_ref<const ColorSpinorField> &, const ColorSpinorField &, TimeProfile &) { }

    virtual ~DslashPolicyImp() = default;
  };

  /**
     Standard dslash parallelization with host staging for send and receive
  */
  template <typename Dslash> struct DslashBasic : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);
      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = device::get_default_stream_idx();
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Device | (Remote * dslashParam.remote_write)),
                packIndex);

      issueGather(halo, dslash);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      DslashCommsPattern pattern(dslashParam.commDim);
      // Track which stream-gated receive waits have already been queued, so the
      // poll loop below enqueues each cuStreamWaitValue64 exactly once while it
      // continues to poll the matching (possibly non-P2P/MPI) send to completion.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            // Query if gather has completed
            if (!pattern.gatherCompleted[2 * i + dir] && pattern.gatherCompleted[pattern.previousDir[2 * i + dir]]) {
              bool event_test = comm_peer2peer_enabled(dir, i);
              if (!event_test)
                PROFILE(event_test = qudaEventQuery(gatherEnd[2 * i + dir]), profile, QUDA_PROFILE_EVENT_QUERY);

              if (event_test) {
                pattern.gatherCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
                if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                  // Stream-mem-op signalling: queues qudaMemcpyP2PAsync + cuStreamWriteValue64
                  // on the per-direction stream.  No cudaIPC event, no MPI doorbell.
                  PROFILE(if (dslash_comms)
                            halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                } else {
                  PROFILE(if (dslash_comms) halo.sendStart(
                            2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                            dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                }
              }
            }

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir] && pattern.gatherCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op receive: queue cuStreamWaitValue64 on the per-direction stream
                // ONCE (it is a fire-and-forget GPU-side wait, captured by the scatterEnd event
                // below; there is no host completion to poll).  NB the receive neighbour is
                // (1 - dir): a partitioned dim may be P2P on one side and non-P2P (cross-node/IB)
                // on the other (e.g. 1x1x1x8 over 2 nodes), so non-P2P receive dirs fall through
                // to commsComplete (MPI) below -- stream-gating only ever applies to genuinely
                // P2P-enabled directions.
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                // The matching send to "dir" may itself be non-P2P (MPI): poll it to completion
                // exactly as the message-passing path does, so its persistent request is drained
                // before the next iteration reuses it (else MPI_Start -> MPI_ERR_REQUEST).
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }

          } // dir=0,1

          if (!pattern.dslashCompleted[2 * i] && pattern.dslashCompleted[pattern.previousDir[2 * i + 1]]
              && pattern.commsCompleted[2 * i] && pattern.commsCompleted[2 * i + 1]) {

            for (int dir = 1; dir >= 0; dir--) {
              const bool peer_dir = comm_peer2peer_enabled(1 - dir, i);
              // Merge per-direction stream into the default stream for:
              //   - non-P2P dirs (scatter copy must finish before exterior dslash)
              //   - stream-gated P2P dirs (cuStreamWaitValue64 sits on the per-dir
              //     stream and must drain before the exterior dslash reads halo)
              const bool need_merge = !peer_dir || ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && peer_dir);
              if (need_merge) {
                PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile, QUDA_PROFILE_EVENT_RECORD);
                PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                    QUDA_PROFILE_STREAM_WAIT_EVENT);
              }
            }

            dslashParam.kernel_type = static_cast<KernelType>(i);
            dslashParam.threads = dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i]; // updating 2 or 6 faces

            // all faces use this stream
            PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

            pattern.dslashCompleted[2 * i] = 1;
          }
        }
      }

      // completeDslash syncs ipcCopyEvent (cudaIPC event) for COPY_ENGINE / REMOTE_WRITE.
      // Stream-gated P2P sends don't record that event — the per-dir stream merge above
      // captures their completion instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

  /**
     Generic shmem dslash
      // shmem bitfield encodes
      // 0 - no shmem
      // 1 - pack P2P (merged in interior)
      // 2 - pack IB (merged in interior)
      // 3 - pack P2P + IB (merged in interior)
      // 8 - barrier part I (packing) (merged in interior, only useful if packing) -- currently required
      // 16 - barrier part II (spin exterior) (merged in exterior) -- currently required
      // 32 - use packstream -- not used
      // 64 - use uber kernel (merge exterior)
  */
  template <typename Dslash, int shmem> struct DslashShmemGeneric : DslashPolicyImp<Dslash> {

#ifdef NVSHMEM_COMMS
    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      setFusedParam(dslashParam, dslash, halo);

      DslashCommsPattern pattern(dslashParam.commDim);
      dslashParam.kernel_type = (shmem & 64) ? UBER_KERNEL : INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(shmem);
      dslashParam.setExteriorDims(shmem & 64);

      // record start of the dslash
      const int packIndex = device::get_default_stream_idx();
      constexpr MemoryLocation location = static_cast<MemoryLocation>(Shmem);

      if (!((shmem & 2) and (shmem & 1))) {
        issuePack(halo, in, dslash, 1 - dslashParam.parity, location, packIndex, shmem);
      }

      dslash.setPack(((shmem & 2) or (shmem & 1)), location); // enable fused kernel packing

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

      dslash.setPack(false, location); // disable fused kernel packing
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for fused exterior kernel
        if (!(shmem & 64)) {
          PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
        }
      }

      dslash::inc_dslash_shmem_sync_counter();
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
#else
    void operator()(Dslash &, cvector_ref<const ColorSpinorField> &, const ColorSpinorField &, TimeProfile &)
    {
      errorQuda("NVSHMEM Dslash policies not built");
    }
#endif
  };

  template <typename Dslash> using DslashShmemUberPackIntra = DslashShmemGeneric<Dslash, 64 + 16 + 8 + 1>;
  template <typename Dslash> using DslashShmemUberPackFull = DslashShmemGeneric<Dslash, 64 + 16 + 8 + 2 + 1>;
  template <typename Dslash> using DslashShmemPackIntra = DslashShmemGeneric<Dslash, 16 + 8 + 1>;
  template <typename Dslash> using DslashShmemPackFull = DslashShmemGeneric<Dslash, 16 + 8 + 2 + 1>;

  /**
   Standard dslash parallelization with host staging for send and receive, and fused halo update kernel
 */
  template <typename Dslash> struct DslashFusedExterior : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = device::get_default_stream_idx();
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Device | (Remote * dslashParam.remote_write)),
                packIndex);

      issueGather(halo, dslash);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      const int scatterIndex = getStreamIndex(dslashParam);
      DslashCommsPattern pattern(dslashParam.commDim);
      // Enqueue each stream-gated receive wait exactly once; keep polling the
      // matching (possibly non-P2P/MPI) send to completion.  See DslashBasic.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            // Query if gather has completed
            if (!pattern.gatherCompleted[2 * i + dir] && pattern.gatherCompleted[pattern.previousDir[2 * i + dir]]) {
              bool event_test = comm_peer2peer_enabled(dir, i);
              if (!event_test)
                PROFILE(event_test = qudaEventQuery(gatherEnd[2 * i + dir]), profile, QUDA_PROFILE_EVENT_QUERY);

              if (event_test) {
                pattern.gatherCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
                if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                  // Stream-mem-op signal send.  Queue qudaMemcpyP2PAsync + cuStreamWriteValue64
                  // on the per-direction stream 2*i+dir.  issueGather already inserted a
                  // qudaStreamWaitEvent(packEnd) on that stream, so the P2P copy serializes
                  // with the pack output but runs in parallel with the other directions.
                  PROFILE(if (dslash_comms)
                            halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                } else {
                  PROFILE(if (dslash_comms) halo.sendStart(
                            2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                            dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                }
              }
            }

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir] && pattern.gatherCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op wait queues cuStreamWaitValue64 on scatterIndex ONCE so the
                // single scatterEnd[0] event below captures it together with any non-P2P
                // scatter copies.  NB receive neighbour is (1 - dir); non-P2P receive dirs
                // fall through to commsComplete (MPI).
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(scatterIndex)),
                          profile, QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                // The matching send to "dir" may be non-P2P (MPI): poll it to completion as the
                // message-passing path does, so its persistent request is drained before reuse.
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, false, scatterIndex)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          } // dir=0,1
        }   // i
      }     // while(pattern.completeSum < commDimTotal)

      // Merge scatterIndex stream into default for: any non-P2P dir (scatter copy
      // must finish) OR any stream-gated P2P dir (cuStreamWaitValue64 must drain).
      for (int i = 3; i >= 0; i--) {
        if (!dslashParam.commDim[i]) continue;
        const bool has_non_p2p = !comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i);
        const bool has_stream_gated = (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
          && (comm_peer2peer_enabled(0, i) || comm_peer2peer_enabled(1, i));
        if (has_non_p2p || has_stream_gated) {
          PROFILE(qudaEventRecord(scatterEnd[0], device::get_stream(scatterIndex)), profile, QUDA_PROFILE_EVENT_RECORD);
          PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
          break;
        }
      }

      // Launch exterior kernel
      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for exterior kernel
        PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      }

      // completeDslash syncs ipcCopyEvent (cudaIPC event) for COPY_ENGINE / REMOTE_WRITE.
      // Stream-gated P2P sends don't record that event — the scatterEnd merge above
      // captures their completion instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Dslash parallelization with GDR for send and receive
 */
  template <typename Dslash> struct DslashGDR : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      issueRecv(halo, dslash, true,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = device::get_default_stream_idx();
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Device | (Remote * dslashParam.remote_write)),
                packIndex);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      bool pack_event = false;
      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          if (!pack_event) {
            qudaEventSynchronize(packEnd[halo.bufferIndex]);
            pack_event = true;
          }

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op P2P send on the per-direction stream (packEnd synced above).
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), true,
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Enqueue each stream-gated receive wait (and its scatterEnd gate) exactly once; keep
      // polling the matching (possibly non-P2P GDR/MPI) send.  See DslashFusedGDR.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op P2P receive on the per-dir stream; record a per-dir scatterEnd
                // and gate default_stream on it (same shape as commsComplete's ipcRemoteCopyEvent
                // wait).  Queue ONCE (GPU-side).  Non-P2P receive dirs use device GDR and fall
                // through to commsComplete; the matching send to "dir" may be non-P2P -> poll it.
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_EVENT_RECORD);
                  PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                          QUDA_PROFILE_STREAM_WAIT_EVENT);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, true)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, true, true, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }

          } // dir=0,1

          if (!pattern.dslashCompleted[2 * i] && pattern.dslashCompleted[pattern.previousDir[2 * i + 1]]
              && pattern.commsCompleted[2 * i] && pattern.commsCompleted[2 * i + 1]) {
            dslashParam.kernel_type = static_cast<KernelType>(i);
            dslashParam.threads = dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i]; // updating 2 or 6 faces

            // all faces use this stream
            PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

            pattern.dslashCompleted[2 * i] = 1;
          }
        }
      }

      // completeDslash syncs ipcCopyEvent for non-stream-gated P2P; stream-gated P2P uses the
      // per-dir scatterEnd merge above instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Dslash parallelization with GDR for send and receive with fused halo update kernel
 */
  template <typename Dslash> struct DslashFusedGDR : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      issueRecv(halo, dslash, true,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = device::get_default_stream_idx();
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Device | (Remote * dslashParam.remote_write)),
                packIndex);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      bool pack_event = false;
      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          if (!pack_event) {
            qudaEventSynchronize(packEnd[halo.bufferIndex]);
            pack_event = true;
          }

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op P2P send on per-direction stream so memcpyP2Ps
                // across directions run in parallel.  Host already synced packEnd
                // above so the per-dir stream's buffer is ready.
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), true,
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }
        }
      } // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Enqueue each stream-gated receive wait (and its scatterEnd gate) exactly
      // once; keep polling the matching (possibly non-P2P GDR/MPI) send.  See DslashBasic.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op wait on the per-direction stream 2*i+dir.  Different
                // directions / transports progress on their own streams in parallel;
                // we then record a per-direction scatterEnd event on that stream and
                // gate default_stream on it (same shape as commsComplete's
                // qudaStreamWaitEvent(default_stream, ipcRemoteCopyEvent) for the
                // non-stream-gated P2P case).  Queue this ONCE (it is GPU-side); NB the
                // receive neighbour is (1 - dir) so non-P2P receive dirs fall through to
                // commsComplete (MPI) below.
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_EVENT_RECORD);
                  PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                          QUDA_PROFILE_STREAM_WAIT_EVENT);
                  stream_recv_queued[2 * i + dir] = true;
                }
                // The matching send to "dir" may be non-P2P (GDR/MPI): poll it to completion as
                // the message-passing path does, so its persistent request is drained before reuse.
                if (halo.commsQuerySend(2 * i + dir, true)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, true, true, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          } // dir=0,1
        }   // i
      }     // pattern.completeSum < pattern.CommDimTotal

      // Launch exterior kernel
      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for fused exterior kernel
        PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      }

      // completeDslash syncs ipcCopyEvent for non-stream-gated P2P.  Stream-gated
      // P2P uses the scatterIndex merge above instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Dslash parallelization with host staging for send and GDR for receive
 */
  template <typename Dslash> struct DslashGDRRecv : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      issueRecv(halo, dslash, true,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = device::get_default_stream_idx();
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Device | (Remote * dslashParam.remote_write)),
                packIndex);

      issueGather(halo, dslash);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      DslashCommsPattern pattern(dslashParam.commDim);
      // Enqueue each stream-gated receive wait (and its scatterEnd gate) exactly once; keep
      // polling the matching (possibly non-P2P) send.  See DslashBasic / DslashFusedGDR.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            // Query if gather has completed
            if (!pattern.gatherCompleted[2 * i + dir] && pattern.gatherCompleted[pattern.previousDir[2 * i + dir]]) {
              bool event_test = comm_peer2peer_enabled(dir, i);
              if (!event_test)
                PROFILE(event_test = qudaEventQuery(gatherEnd[2 * i + dir]), profile, QUDA_PROFILE_EVENT_QUERY);

              if (event_test) {
                pattern.gatherCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
                if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                  // Stream-mem-op P2P send on the per-direction stream (no cudaIPC event / doorbell).
                  PROFILE(if (dslash_comms)
                            halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                } else {
                  PROFILE(if (dslash_comms) halo.sendStart(
                            2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                            dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                }
              }
            }

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir] && pattern.gatherCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op P2P receive: wait on the per-dir stream, record a per-dir
                // scatterEnd and gate default_stream on it (same shape as commsComplete's
                // ipcRemoteCopyEvent wait).  Queue ONCE.  Non-P2P recv dirs use device GDR and
                // fall through to commsComplete; the matching non-P2P send is polled (gdr=false).
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_EVENT_RECORD);
                  PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                          QUDA_PROFILE_STREAM_WAIT_EVENT);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, true, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }

          } // dir=0,1

          if (!pattern.dslashCompleted[2 * i] && pattern.dslashCompleted[pattern.previousDir[2 * i + 1]]
              && pattern.commsCompleted[2 * i] && pattern.commsCompleted[2 * i + 1]) {
            dslashParam.kernel_type = static_cast<KernelType>(i);
            dslashParam.threads = dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i]; // updating 2 or 6 faces

            // all faces use this stream
            PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

            pattern.dslashCompleted[2 * i] = 1;
          }
        }
      }

      // completeDslash syncs ipcCopyEvent for non-stream-gated P2P; stream-gated P2P uses the
      // per-dir scatterEnd merge above instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Dslash parallelization with host staging for send and GDR for receive, with fused halo update kernel
 */
  template <typename Dslash> struct DslashFusedGDRRecv : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      issueRecv(halo, dslash, true,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = device::get_default_stream_idx();
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Device | (Remote * dslashParam.remote_write)),
                packIndex);

      issueGather(halo, dslash);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      DslashCommsPattern pattern(dslashParam.commDim);
      // Enqueue each stream-gated receive wait (and its scatterEnd gate) exactly once; keep
      // polling the matching (possibly non-P2P) send.  See DslashBasic / DslashFusedGDR.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            // Query if gather has completed
            if (!pattern.gatherCompleted[2 * i + dir] && pattern.gatherCompleted[pattern.previousDir[2 * i + dir]]) {
              bool event_test = comm_peer2peer_enabled(dir, i);
              if (!event_test)
                PROFILE(event_test = qudaEventQuery(gatherEnd[2 * i + dir]), profile, QUDA_PROFILE_EVENT_QUERY);

              if (event_test) {
                pattern.gatherCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
                if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                  // Stream-mem-op P2P send on the per-direction stream (no cudaIPC event / doorbell).
                  PROFILE(if (dslash_comms)
                            halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                } else {
                  PROFILE(if (dslash_comms) halo.sendStart(2 * i + dir,
                                                           dslashParam.remote_write ? device::get_default_stream() :
                                                                                      device::get_stream(2 * i + dir),
                                                           false, dslashParam.remote_write),
                          profile, QUDA_PROFILE_COMMS_START);
                }
              }
            }

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir] && pattern.gatherCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op P2P receive: wait on the per-dir stream, record a per-dir
                // scatterEnd and gate default_stream on it so the fused exterior sees the halo.
                // Queue ONCE.  Non-P2P recv dirs use device GDR and fall through to commsComplete;
                // the matching non-P2P send is polled (gdr=false).
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_EVENT_RECORD);
                  PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                          QUDA_PROFILE_STREAM_WAIT_EVENT);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, true, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          } // dir=0,1
        }   // i
      }     // while(pattern.completeSum < commDimTotal)

      // Launch exterior kernel
      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for fused exterior kernel
        PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      }

      // completeDslash syncs ipcCopyEvent for non-stream-gated P2P; stream-gated P2P uses the
      // per-dir scatterEnd merge above instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

  /**
     Variation of multi-gpu dslash where the packing kernel writes
     buffers directly to host memory
  */
  template <typename Dslash> struct DslashZeroCopyPack : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = getStreamIndex(dslashParam);
      PROFILE(qudaStreamWaitEvent(device::get_stream(packIndex), dslashStart[halo.bufferIndex], 0), profile,
              QUDA_PROFILE_STREAM_WAIT_EVENT);
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write)),
                packIndex);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_stream(packIndex));
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op signalling on the per-direction stream (no cudaIPC event / MPI doorbell).
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Track stream-gated receive waits so each cuStreamWaitValue64 is queued exactly once
      // while the matching (possibly non-P2P/MPI) send is polled to completion.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms have finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op receive: queue cuStreamWaitValue64 ONCE on the per-dir stream
                // (fire-and-forget GPU-side wait, captured by scatterEnd below).  The matching
                // send to "dir" may be non-P2P (MPI) on a partitioned dim, so poll it to drain
                // its persistent request.
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          }

          if (!pattern.dslashCompleted[2 * i] && pattern.dslashCompleted[pattern.previousDir[2 * i + 1]]
              && pattern.commsCompleted[2 * i] && pattern.commsCompleted[2 * i + 1]) {
            for (int dir = 1; dir >= 0; dir--) {
              const bool peer_dir = comm_peer2peer_enabled(1 - dir, i);
              // Merge per-dir stream into the default stream for non-P2P dirs (scatter copy) and
              // for stream-gated P2P dirs (the cuStreamWaitValue64 must drain before the exterior
              // dslash reads the halo).
              const bool need_merge = !peer_dir || ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && peer_dir);
              if (need_merge) {
                // Record the end of the scattering
                PROFILE(
                        qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile, QUDA_PROFILE_EVENT_RECORD);
                // wait for scattering to finish and then launch dslash
                PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                    QUDA_PROFILE_STREAM_WAIT_EVENT);
              }
            }

            dslashParam.kernel_type = static_cast<KernelType>(i);
            dslashParam.threads = dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i]; // updating 2 or 6 faces

            // all faces use this stream
            PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

            pattern.dslashCompleted[2 * i] = 1;
          }
        }
      }

      // Stream-gated P2P sends don't record ipcCopyEvent; the per-dir stream merge captures them.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory with fused halo update kernel
*/
  template <typename Dslash> struct DslashFusedZeroCopyPack : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      const int packScatterIndex = getStreamIndex(dslashParam);
      PROFILE(qudaStreamWaitEvent(device::get_stream(packScatterIndex), dslashStart[halo.bufferIndex], 0), profile,
              QUDA_PROFILE_STREAM_WAIT_EVENT);
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write)),
                packScatterIndex);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_stream(packScatterIndex));
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op signalling on the per-direction stream (no cudaIPC event / MPI doorbell).
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packScatterIndex : 2 * i + dir),
                          false, dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Enqueue each stream-gated receive wait exactly once on packScatterIndex; keep polling the
      // matching (possibly non-P2P/MPI) send to completion.  See DslashFusedExterior.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op wait queues cuStreamWaitValue64 on packScatterIndex ONCE so the
                // single scatterEnd[0] event below captures it with any non-P2P scatter copies.
                // Non-P2P receive dirs fall through to commsComplete (MPI).
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(packScatterIndex)),
                          profile, QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, false, packScatterIndex)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }

          } // dir=0,1
        }   // i
      }     // pattern.completeSum

      // Merge packScatterIndex into default for: any non-P2P dir (scatter copy must finish) OR
      // any stream-gated P2P dir (cuStreamWaitValue64 must drain before the fused exterior dslash).
      for (int i = 3; i >= 0; i--) {
        if (!dslashParam.commDim[i]) continue;
        const bool has_non_p2p = !comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i);
        const bool has_stream_gated = (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
          && (comm_peer2peer_enabled(0, i) || comm_peer2peer_enabled(1, i));
        if (has_non_p2p || has_stream_gated) {
          // post an event in the scatter stream and wait on that
          PROFILE(qudaEventRecord(scatterEnd[0], device::get_stream(packScatterIndex)), profile, QUDA_PROFILE_EVENT_RECORD);
          PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[0], 0), profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
          break;
        }
      }

      // Launch exterior kernel
      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for fused exterior kernel
        PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      }

      // Stream-gated P2P sends don't record ipcCopyEvent; the scatterEnd merge above captures them.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Multi-GPU Dslash zero-copy for the send and GDR for the receive
 */
  template <typename Dslash> struct DslashZeroCopyPackGDRRecv : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      issueRecv(halo, dslash, true,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = getStreamIndex(dslashParam);
      PROFILE(qudaStreamWaitEvent(device::get_stream(packIndex), dslashStart[halo.bufferIndex], 0), profile,
              QUDA_PROFILE_STREAM_WAIT_EVENT);
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write)),
                packIndex);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_stream(packIndex));
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op P2P send on the per-direction stream (no cudaIPC event / doorbell).
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Enqueue each stream-gated receive wait (and its scatterEnd gate) exactly once; keep
      // polling the matching (possibly non-P2P) send.  See DslashFusedGDR.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir] && pattern.gatherCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op P2P receive: wait on the per-dir stream, record a per-dir
                // scatterEnd and gate default_stream on it (same shape as commsComplete's
                // ipcRemoteCopyEvent wait).  Queue ONCE.  Non-P2P recv dirs use device GDR and
                // fall through to commsComplete; the matching non-P2P send is polled (gdr=false).
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_EVENT_RECORD);
                  PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                          QUDA_PROFILE_STREAM_WAIT_EVENT);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, true, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }

          } // dir=0,1

          if (!pattern.dslashCompleted[2 * i] && pattern.dslashCompleted[pattern.previousDir[2 * i + 1]]
              && pattern.commsCompleted[2 * i] && pattern.commsCompleted[2 * i + 1]) {
            dslashParam.kernel_type = static_cast<KernelType>(i);
            dslashParam.threads = dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i]; // updating 2 or 6 faces

            // all faces use this stream
            PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

            pattern.dslashCompleted[2 * i] = 1;
          }
        }
      }

      // completeDslash syncs ipcCopyEvent for non-stream-gated P2P; stream-gated P2P uses the
      // per-dir scatterEnd merge above instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Multi-GPU Dslash zero-copy for the send and GDR for the receive,
   with fused halo update kernel
 */
  template <typename Dslash> struct DslashFusedZeroCopyPackGDRRecv : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      const int packIndex = getStreamIndex(dslashParam);
      PROFILE(qudaStreamWaitEvent(device::get_stream(packIndex), dslashStart[halo.bufferIndex], 0), profile,
              QUDA_PROFILE_STREAM_WAIT_EVENT);
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write)),
                packIndex);

      issueRecv(halo, dslash, true,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_stream(packIndex));
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op P2P send on the per-direction stream (no cudaIPC event / doorbell).
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Enqueue each stream-gated receive wait (and its scatterEnd gate) exactly once; keep
      // polling the matching (possibly non-P2P) send.  See DslashFusedGDR.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms has finished
            if (!pattern.commsCompleted[2 * i + dir] && pattern.gatherCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op P2P receive: wait on the per-dir stream, record a per-dir
                // scatterEnd and gate default_stream on it so the fused exterior sees the halo.
                // Queue ONCE.  Non-P2P recv dirs use device GDR and fall through to commsComplete;
                // the matching non-P2P send is polled (gdr=false).
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_EVENT_RECORD);
                  PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                          QUDA_PROFILE_STREAM_WAIT_EVENT);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, true, false)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          } // dir=0,1
        }   // i
      }     // while(pattern.completeSum < commDimTotal)

      // Launch exterior kernel
      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for fused exterior kernel
        PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      }

      // completeDslash syncs ipcCopyEvent for non-stream-gated P2P; stream-gated P2P uses the
      // per-dir scatterEnd merge above instead.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory
*/
  template <typename Dslash> struct DslashZeroCopy : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = getStreamIndex(dslashParam);
      PROFILE(qudaStreamWaitEvent(device::get_stream(packIndex), dslashStart[halo.bufferIndex], 0), profile,
              QUDA_PROFILE_STREAM_WAIT_EVENT);
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write)),
                packIndex);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_stream(packIndex));
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op signalling on the per-direction stream (no cudaIPC event / MPI doorbell).
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Track stream-gated receive waits so each cuStreamWaitValue64 is queued exactly once
      // while the matching (possibly non-P2P/MPI) send is polled to completion.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms have finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op receive: queue cuStreamWaitValue64 ONCE on the per-dir stream
                // (fire-and-forget GPU-side wait, captured by scatterEnd below).  The matching
                // send to "dir" may be non-P2P (MPI) on a partitioned dim, so poll it to drain
                // its persistent request.  For P2P dirs the recv data lands in the DEVICE ghost
                // (pack() forces device for P2P dirs); only non-P2P dirs read the mapped host ghost.
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, true)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          }

          // enqueue the boundary dslash kernel as soon as the scatters have been enqueued
          if (!pattern.dslashCompleted[2 * i] && pattern.dslashCompleted[pattern.previousDir[2 * i + 1]]
              && pattern.commsCompleted[2 * i] && pattern.commsCompleted[2 * i + 1]) {
            // Merge the per-dir stream into the default stream for stream-gated P2P dirs: the
            // cuStreamWaitValue64 must drain before the exterior dslash reads the (device) halo.
            // Non-P2P dirs need no merge -- zero-copy receive reads mapped host memory directly
            // (no scatter), as in the un-converted policy.
            for (int dir = 1; dir >= 0; dir--) {
              const bool peer_dir = comm_peer2peer_enabled(1 - dir, i);
              const bool need_merge = (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && peer_dir;
              if (need_merge) {
                PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                        QUDA_PROFILE_EVENT_RECORD);
                PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                        QUDA_PROFILE_STREAM_WAIT_EVENT);
              }
            }

            dslashParam.kernel_type = static_cast<KernelType>(i);
            dslashParam.threads = dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i]; // updating 2 or 6 faces

            setMappedGhost(dslash, halo, true);
            PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
            setMappedGhost(dslash, halo, false);

            pattern.dslashCompleted[2 * i] = 1;
          }
        }
      }

      // Stream-gated P2P has no ipcCopyEvent; drain the per-dir P2P streams into the
      // compute stream so buffer reuse next apply can't race this apply's traffic.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) completeDslashStreamGated(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

/**
   Variation of multi-gpu dslash where the packing kernel writes
   buffers directly to host memory with fused halo update kernel
*/
  template <typename Dslash> struct DslashFusedZeroCopy : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      const int packIndex = getStreamIndex(dslashParam);
      PROFILE(qudaStreamWaitEvent(device::get_stream(packIndex), dslashStart[halo.bufferIndex], 0), profile,
              QUDA_PROFILE_STREAM_WAIT_EVENT);
      const int parity_src = (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 1 - dslashParam.parity : 0);
      issuePack(halo, in, dslash, parity_src, static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write)),
                packIndex);

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_stream(packIndex));
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op signalling on the per-direction stream (no cudaIPC event / MPI doorbell).
                PROFILE(if (dslash_comms)
                          halo.sendStartStreamGated(2 * i + dir, device::get_stream(2 * i + dir), dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(
                          2 * i + dir, device::get_stream(dslashParam.remote_write ? packIndex : 2 * i + dir), false,
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Enqueue each stream-gated receive wait exactly once on packIndex; keep polling the
      // matching (possibly non-P2P/MPI) send to completion.  See DslashFusedZeroCopyPack.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms have finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op wait queues cuStreamWaitValue64 on packIndex ONCE so the single
                // scatterEnd[0] event below captures it before the fused exterior dslash.  For P2P
                // dirs the recv data lands in the DEVICE ghost (pack() forces device for P2P dirs);
                // non-P2P receive dirs read the mapped host ghost and fall through to commsComplete.
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(packIndex)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, true)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          }
        }
      }

      // Merge packIndex into the default stream when any stream-gated P2P dir is present: the
      // cuStreamWaitValue64(s) must drain before the fused exterior dslash reads the (device)
      // halo.  Non-P2P dirs need no merge (zero-copy receive reads mapped host memory directly).
      for (int i = 3; i >= 0; i--) {
        if (!dslashParam.commDim[i]) continue;
        const bool has_stream_gated = (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
          && (comm_peer2peer_enabled(0, i) || comm_peer2peer_enabled(1, i));
        if (has_stream_gated) {
          PROFILE(qudaEventRecord(scatterEnd[0], device::get_stream(packIndex)), profile, QUDA_PROFILE_EVENT_RECORD);
          PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[0], 0), profile,
                  QUDA_PROFILE_STREAM_WAIT_EVENT);
          break;
        }
      }

      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for fused exterior kernel
        setMappedGhost(dslash, halo, true);
        PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
        setMappedGhost(dslash, halo, false);
      }

      // Stream-gated P2P sends don't record ipcCopyEvent; the scatterEnd merge above captures them.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

  /**
     Variation of multi-gpu dslash where the packing kernel is fused
     into the interior dslash kernel.  Only really makes sense on
     systems that are fully peer connected.
  */
  template <typename Dslash> struct DslashFusedPack : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      MemoryLocation location = static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write));
      dslash.setPack(true, location); // enable fused kernel packing

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

      dslash.setPack(false, location); // disable fused kernel packing
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_default_stream());
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op signalling.  The fused pack writes on the default stream, so for
                // remote_write the signal goes on the default stream (ordered after the write);
                // for copy-engine it goes on the per-dir stream (the default stream was synced
                // above so the memcpy source is ready).  No cudaIPC event / MPI doorbell.
                PROFILE(if (dslash_comms) halo.sendStartStreamGated(
                          2 * i + dir,
                          dslashParam.remote_write ? device::get_default_stream() : device::get_stream(2 * i + dir),
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(2 * i + dir,
                                                         dslashParam.remote_write ? device::get_default_stream() :
                                                                                    device::get_stream(2 * i + dir),
                                                         false, dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Track stream-gated receive waits so each cuStreamWaitValue64 is queued exactly once
      // while the matching (possibly non-P2P/MPI) send is polled to completion.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms have finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op receive: queue cuStreamWaitValue64 ONCE on the per-dir stream
                // (fire-and-forget GPU-side wait, captured by scatterEnd below).  For P2P dirs the
                // recv data lands in the DEVICE ghost (pack() forces device for P2P dirs); only
                // non-P2P dirs read the mapped host ghost (and fall through to commsComplete).
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, true)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          }

          // enqueue the boundary dslash kernel as soon as the scatters have been
          // enqueued
          if (!pattern.dslashCompleted[2 * i] && pattern.dslashCompleted[pattern.previousDir[2 * i + 1]]
              && pattern.commsCompleted[2 * i] && pattern.commsCompleted[2 * i + 1]) {
            // Merge the per-dir stream into the default stream for stream-gated P2P dirs (the
            // cuStreamWaitValue64 must drain before the exterior dslash reads the device halo).
            // Non-P2P dirs need no merge -- zero-copy receive reads mapped host memory directly.
            for (int dir = 1; dir >= 0; dir--) {
              const bool peer_dir = comm_peer2peer_enabled(1 - dir, i);
              const bool need_merge = (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && peer_dir;
              if (need_merge) {
                PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                        QUDA_PROFILE_EVENT_RECORD);
                PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                        QUDA_PROFILE_STREAM_WAIT_EVENT);
              }
            }

            dslashParam.kernel_type = static_cast<KernelType>(i);
            dslashParam.threads = dslash.Nface() * halo.getDslashConstant().ghostFaceCB[i]; // updating 2 or 6 faces

            setMappedGhost(dslash, halo, true);
            PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
            setMappedGhost(dslash, halo, false);

            pattern.dslashCompleted[2 * i] = 1;
          }
        }
      }

      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

  /**
     Variation of multi-gpu dslash where the packing kernel is fused
     into the interior dslash kernel, and the halo update kernels are
     all fused.  Only really makes sense on systems that are fully peer
     connected.
  */
  template <typename Dslash> struct DslashFusedPackFusedHalo : DslashPolicyImp<Dslash> {

    void operator()(Dslash &dslash, cvector_ref<const ColorSpinorField> &, const ColorSpinorField &halo,
                    TimeProfile &profile)
    {
      PROFILE_START(profile, QUDA_PROFILE_TOTAL);

      auto &dslashParam = dslash.dslashParam;
      dslashParam.kernel_type = INTERIOR_KERNEL;
      dslashParam.threads = halo.getDslashConstant().volume_4d_cb;
      dslash.setShmem(0);

      // record start of the dslash
      PROFILE(qudaEventRecord(dslashStart[halo.bufferIndex], device::get_default_stream()), profile,
              QUDA_PROFILE_EVENT_RECORD);

      issueRecv(halo, dslash, false,
                (dslashParam.p2p_signal
                 == QudaP2PSignal::STREAM_GATED)); // Prepost receives (skip P2P doorbell when stream-gating)

      MemoryLocation location = static_cast<MemoryLocation>(Host | (Remote * dslashParam.remote_write));
      dslash.setPack(true, location); // enable fused kernel packing

      PROFILE(if (dslash_interior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);

      dslash.setPack(false, location); // disable fused kernel packing
      if (aux_worker) aux_worker->apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) { // only synchronize if we need to
        if (!dslashParam.remote_write
            || (dslashParam.commDim[i] && (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)))) {
          qudaStreamSynchronize(device::get_default_stream());
          break;
        }
      }

      for (int p2p = 0; p2p < 2; p2p++) { // schedule non-p2p traffic first, then do p2p
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {
            if ((comm_peer2peer_enabled(dir, i) + p2p) % 2 == 0) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(dir, i)) {
                // Stream-mem-op signalling.  The fused pack writes on the default stream, so for
                // remote_write the signal goes on the default stream (ordered after the write);
                // for copy-engine it goes on the per-dir stream (the default stream was synced
                // above so the memcpy source is ready).  No cudaIPC event / MPI doorbell.
                PROFILE(if (dslash_comms) halo.sendStartStreamGated(
                          2 * i + dir,
                          dslashParam.remote_write ? device::get_default_stream() : device::get_stream(2 * i + dir),
                          dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              } else {
                PROFILE(if (dslash_comms) halo.sendStart(2 * i + dir,
                                                         dslashParam.remote_write ? device::get_default_stream() :
                                                                                    device::get_stream(2 * i + dir),
                                                         false, dslashParam.remote_write),
                        profile, QUDA_PROFILE_COMMS_START);
              }
            } // is p2p?
          }   // dir
        }     // i
      }       // p2p

      DslashCommsPattern pattern(dslashParam.commDim, true);
      // Enqueue each stream-gated receive wait exactly once on its per-dir stream; keep polling
      // the matching (possibly non-P2P/MPI) send to completion.
      bool stream_recv_queued[2 * QUDA_MAX_DIM] = {};
      while (pattern.completeSum < pattern.commDimTotal) {

        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;

          for (int dir = 1; dir >= 0; dir--) {

            // Query if comms have finished
            if (!pattern.commsCompleted[2 * i + dir]) {
              if ((dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) && comm_peer2peer_enabled(1 - dir, i)) {
                // Stream-mem-op receive: queue cuStreamWaitValue64 ONCE on the per-dir stream
                // (captured by the per-dir scatterEnd merge below before the fused exterior).  For
                // P2P dirs the recv data lands in the DEVICE ghost (pack() forces device for P2P
                // dirs); non-P2P dirs read the mapped host ghost and fall through to commsComplete.
                if (!stream_recv_queued[2 * i + dir]) {
                  PROFILE(if (dslash_comms) halo.commsWaitStream(2 * i + dir, device::get_stream(2 * i + dir)), profile,
                          QUDA_PROFILE_COMMS_QUERY);
                  stream_recv_queued[2 * i + dir] = true;
                }
                if (halo.commsQuerySend(2 * i + dir, false)) {
                  pattern.commsCompleted[2 * i + dir] = 1;
                  pattern.completeSum++;
                }
              } else if (commsComplete(halo, dslash, i, dir, false, false, true)) {
                pattern.commsCompleted[2 * i + dir] = 1;
                pattern.completeSum++;
              }
            }
          }
        }
      }

      // Merge each stream-gated P2P dir's per-dir stream into the default stream so every
      // cuStreamWaitValue64 drains before the single fused exterior dslash reads the device halo.
      // Non-P2P dirs need no merge (zero-copy receive reads mapped host memory directly).
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED) {
        for (int i = 3; i >= 0; i--) {
          if (!dslashParam.commDim[i]) continue;
          for (int dir = 1; dir >= 0; dir--) {
            if (comm_peer2peer_enabled(1 - dir, i)) {
              PROFILE(qudaEventRecord(scatterEnd[2 * i + dir], device::get_stream(2 * i + dir)), profile,
                      QUDA_PROFILE_EVENT_RECORD);
              PROFILE(qudaStreamWaitEvent(device::get_default_stream(), scatterEnd[2 * i + dir], 0), profile,
                      QUDA_PROFILE_STREAM_WAIT_EVENT);
            }
          }
        }
      }

      if (pattern.commDimTotal) {
        setFusedParam(dslashParam, dslash, halo); // setup for fused exterior kernel
        setMappedGhost(dslash, halo, true);
        PROFILE(if (dslash_exterior_compute) dslash.apply(device::get_default_stream()), profile, QUDA_PROFILE_DSLASH_KERNEL);
        setMappedGhost(dslash, halo, false);
      }

      // Stream-gated P2P sends don't record ipcCopyEvent; the per-dir scatterEnd merge captures them.
      if (dslashParam.p2p_signal == QudaP2PSignal::STREAM_GATED)
        completeDslashStreamGated(halo, dslashParam);
      else
        completeDslash(halo, dslashParam);
      halo.bufferIndex = (1 - halo.bufferIndex);
      PROFILE_STOP(profile, QUDA_PROFILE_TOTAL);
    }
  };

  // whether we have initialized the dslash policy tuner
  extern bool dslash_policy_init;

  // used to keep track of which policy to start the autotuning
  extern int first_active_policy;
  extern int first_active_p2p_policy;

  enum class QudaDslashPolicy {
    QUDA_DSLASH,
    QUDA_FUSED_DSLASH,
    QUDA_GDR_DSLASH,
    QUDA_FUSED_GDR_DSLASH,
    QUDA_GDR_RECV_DSLASH,
    QUDA_FUSED_GDR_RECV_DSLASH,
    QUDA_ZERO_COPY_PACK_DSLASH,
    QUDA_FUSED_ZERO_COPY_PACK_DSLASH,
    QUDA_ZERO_COPY_DSLASH,
    QUDA_FUSED_ZERO_COPY_DSLASH,
    QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH,
    QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH,
    QUDA_DSLASH_FUSED_PACK,
    QUDA_DSLASH_FUSED_PACK_FUSED_HALO,
    QUDA_SHMEM_UBER_PACKINTRA_DSLASH,
    QUDA_SHMEM_UBER_PACKFULL_DSLASH,
    QUDA_SHMEM_PACKINTRA_DSLASH,
    QUDA_SHMEM_PACKFULL_DSLASH,
    QUDA_DSLASH_POLICY_DISABLED // this MUST be the last element
  };

  // list of dslash policies that are enabled
  extern std::vector<QudaDslashPolicy> policies;

  // string used as a tunekey to ensure we retune if the dslash policy env changes
  extern char policy_string[TuneKey::aux_n];

  // P2P DATA-MOVEMENT policies (tuned on aux.y).  The completion-SIGNALLING
  // mechanism (event vs stream-mem-op) is orthogonal and is NOT a policy here:
  // it is resolved globally by comm::p2p_signal() and applied via
  // dslashParam.p2p_signal.  (Historically STREAM_GATED was a 4th policy that
  // conflated copy-engine movement with stream-gated signalling; that has been
  // split out -- see QUDA_P2P_TRANSPORT.)
  enum class QudaP2PPolicy {
    QUDA_P2P_DEFAULT,         // no special hanlding for p2p
    QUDA_P2P_COPY_ENGINE,     // use copy engine for p2p traffic
    QUDA_P2P_REMOTE_WRITE,    // write packed halos directly to peers
    QUDA_P2P_POLICY_DISABLED, // this must be the last element
  };

  // list of p2p policies that are enabled
  extern std::vector<QudaP2PPolicy> p2p_policies;

  template <typename Dslash> struct DslashFactory {

    // map of GDR policies to their non-GDR equivalents
    static auto blacklist_map(const QudaDslashPolicy &policy)
    {
      switch (policy) {
      case QudaDslashPolicy::QUDA_GDR_DSLASH:
      case QudaDslashPolicy::QUDA_GDR_RECV_DSLASH:
        return QudaDslashPolicy::QUDA_DSLASH;
      case QudaDslashPolicy::QUDA_FUSED_GDR_DSLASH:
      case QudaDslashPolicy::QUDA_FUSED_GDR_RECV_DSLASH:
        return QudaDslashPolicy::QUDA_FUSED_DSLASH;
      case QudaDslashPolicy::QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH:
        return QudaDslashPolicy::QUDA_ZERO_COPY_PACK_DSLASH;
      case QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH:
        return QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_DSLASH;
      default: return policy;
      }
    }

    static std::unique_ptr<DslashPolicyImp<Dslash>> create(const QudaDslashPolicy &policy_)
    {
      // if GDR policy and blacklist enabled, create the non-GDR equivalent
      QudaDslashPolicy policy = comm_gdr_blacklist() ? blacklist_map(policy_) : policy_;

      switch (policy) {
      case QudaDslashPolicy::QUDA_DSLASH: return std::make_unique<DslashBasic<Dslash>>();
      case QudaDslashPolicy::QUDA_FUSED_DSLASH: return std::make_unique<DslashFusedExterior<Dslash>>();
      case QudaDslashPolicy::QUDA_GDR_DSLASH: return std::make_unique<DslashGDR<Dslash>>();
      case QudaDslashPolicy::QUDA_FUSED_GDR_DSLASH: return std::make_unique<DslashFusedGDR<Dslash>>();
      case QudaDslashPolicy::QUDA_GDR_RECV_DSLASH: return std::make_unique<DslashGDRRecv<Dslash>>();
      case QudaDslashPolicy::QUDA_FUSED_GDR_RECV_DSLASH: return std::make_unique<DslashFusedGDRRecv<Dslash>>();
      case QudaDslashPolicy::QUDA_ZERO_COPY_PACK_DSLASH: return std::make_unique<DslashZeroCopyPack<Dslash>>();
      case QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_DSLASH: return std::make_unique<DslashFusedZeroCopyPack<Dslash>>();
      case QudaDslashPolicy::QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH: return std::make_unique<DslashZeroCopyPackGDRRecv<Dslash>>();
      case QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH: return std::make_unique<DslashFusedZeroCopyPackGDRRecv<Dslash>>();
      case QudaDslashPolicy::QUDA_ZERO_COPY_DSLASH: return std::make_unique<DslashZeroCopy<Dslash>>();
      case QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_DSLASH: return std::make_unique<DslashFusedZeroCopy<Dslash>>();
      case QudaDslashPolicy::QUDA_DSLASH_FUSED_PACK: return std::make_unique<DslashFusedPack<Dslash>>();
      case QudaDslashPolicy::QUDA_DSLASH_FUSED_PACK_FUSED_HALO: return std::make_unique<DslashFusedPackFusedHalo<Dslash>>();
      case QudaDslashPolicy::QUDA_SHMEM_UBER_PACKINTRA_DSLASH: return std::make_unique<DslashShmemUberPackIntra<Dslash>>();
      case QudaDslashPolicy::QUDA_SHMEM_UBER_PACKFULL_DSLASH: return std::make_unique<DslashShmemUberPackFull<Dslash>>();
      case QudaDslashPolicy::QUDA_SHMEM_PACKINTRA_DSLASH: return std::make_unique<DslashShmemPackIntra<Dslash>>();
      case QudaDslashPolicy::QUDA_SHMEM_PACKFULL_DSLASH: return std::make_unique<DslashShmemPackFull<Dslash>>();
      default: errorQuda("Dslash policy %d not recognized", static_cast<int>(policy));
      }

      return std::make_unique<DslashPolicyImp<Dslash>>();
    }
  };

  inline void enable_policy(QudaDslashPolicy p)
  {
    size_t p_idx = static_cast<std::size_t>(p);
    if (p >= QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED) errorQuda("Invalid policy %lu", p_idx);
    policies[p_idx] = p;
  }

  inline void disable_policy(QudaDslashPolicy p)
  {
    policies[static_cast<std::size_t>(p)] = QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED;
  }

  template <typename Dslash> class DslashPolicyTune : public Tunable
  {
    Dslash &dslash;
    using Arg = std::remove_reference_t<decltype(dslash.dslashParam)>;
    Arg &dslashParam;
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const ColorSpinorField &halo;
    TimeProfile &profile;

    bool tuneGridDim() const override { return false; } // Don't tune the grid dimensions.
    bool tuneAuxDim() const override { return true; }   // Do tune the aux dimensions.
    unsigned int sharedBytesPerThread() const override { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const override { return 0; }

  public:
    DslashPolicyTune(Dslash &dslash, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                     const ColorSpinorField &halo, TimeProfile &profile) :
      dslash(dslash), dslashParam(dslash.dslashParam), out(out), in(in), halo(halo), profile(profile)
    {
      if (!dslash_policy_init) {

        first_active_policy = static_cast<int>(QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED);
        first_active_p2p_policy = static_cast<int>(QudaP2PPolicy::QUDA_P2P_POLICY_DISABLED);

        // P2P data-movement policy candidates, keyed on the QUDA_ENABLE_P2P bitmap.
        // Valid in both non-MNNVL and MNNVL builds: completion signalling is resolved
        // separately (comm::p2p_signal() / dslashParam.p2p_signal), so COPY_ENGINE and
        // REMOTE_WRITE no longer depend on cudaIPC events; under MNNVL the resolver
        // forces STREAM_GATED signalling, which works cross-clique.  REMOTE_WRITE (the
        // pack kernel stores the halo directly into the peer's buffer) is offered only
        // when bit 1 is set and the transport can make it safe: a remote-written halo
        // may be observed at the receiver before its multi-transaction data commits, and
        // the receiver-side flush guard (CU_STREAM_WAIT_VALUE_FLUSH) is not available on
        // all hardware, so comm_p2p_remote_write_supported() drops REMOTE_WRITE where it
        // would be unsafe and the tuner falls back to the copy engine.
        if ((comm_peer2peer_enabled_global() & 2)
            && comm_p2p_remote_write_supported()) { // bit 1: remote-write (direct store) policy tuning
          p2p_policies[static_cast<std::size_t>(QudaP2PPolicy::QUDA_P2P_REMOTE_WRITE)]
              = QudaP2PPolicy::QUDA_P2P_REMOTE_WRITE;
          first_active_p2p_policy = static_cast<int>(QudaP2PPolicy::QUDA_P2P_REMOTE_WRITE);
        }

        if (comm_peer2peer_enabled_global() & 1) { // bit 0: copy-engine policy tuning
          p2p_policies[static_cast<std::size_t>(QudaP2PPolicy::QUDA_P2P_COPY_ENGINE)]
              = QudaP2PPolicy::QUDA_P2P_COPY_ENGINE;
          first_active_p2p_policy = static_cast<int>(QudaP2PPolicy::QUDA_P2P_COPY_ENGINE);
        }

        if (!(comm_peer2peer_enabled_global() & 4)) { // bit 2: enable non-p2p (DEFAULT) policy tuning
          p2p_policies[static_cast<std::size_t>(QudaP2PPolicy::QUDA_P2P_DEFAULT)] = QudaP2PPolicy::QUDA_P2P_DEFAULT;
          first_active_p2p_policy = static_cast<int>(QudaP2PPolicy::QUDA_P2P_DEFAULT);
        }

        static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_POLICY");
        if (dslash_policy_env) { // set the policies to tune for explicitly
          std::stringstream policy_list(dslash_policy_env);

          int policy_;
          while (policy_list >> policy_) {
            QudaDslashPolicy dslash_policy = static_cast<QudaDslashPolicy>(policy_);

            // check this is a valid policy choice
            if ((dslash_policy == QudaDslashPolicy::QUDA_GDR_DSLASH
                    || dslash_policy == QudaDslashPolicy::QUDA_FUSED_GDR_DSLASH
                    || dslash_policy == QudaDslashPolicy::QUDA_GDR_RECV_DSLASH
                    || dslash_policy == QudaDslashPolicy::QUDA_FUSED_GDR_RECV_DSLASH)
                && !comm_gdr_enabled()) {
              errorQuda("Cannot select a GDR policy %d unless QUDA_ENABLE_GDR is set", static_cast<int>(dslash_policy));
            }

            // check valid policy for nvshmem
            if (dslash_policy == QudaDslashPolicy::QUDA_SHMEM_UBER_PACKINTRA_DSLASH
                || dslash_policy == QudaDslashPolicy::QUDA_SHMEM_UBER_PACKFULL_DSLASH
                || dslash_policy == QudaDslashPolicy::QUDA_SHMEM_PACKINTRA_DSLASH
                || dslash_policy == QudaDslashPolicy::QUDA_SHMEM_PACKFULL_DSLASH) {
#ifndef NVSHMEM_COMMS
              errorQuda("Cannot select a NVSHMEM policy %d when QUDA is not build with QUDA_NVSHMEM enabled.",
                        static_cast<int>(dslash_policy));
#endif
            }

            enable_policy(static_cast<QudaDslashPolicy>(policy_));
            first_active_policy = policy_ < first_active_policy ? policy_ : first_active_policy;
            if (policy_list.peek() == ',') policy_list.ignore();
          }
          if (first_active_policy == static_cast<int>(QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED))
            errorQuda("No valid policy found in QUDA_ENABLE_DSLASH_POLICY");
        } else {
          // In MNNVL builds, every P2P-using dslash policy honours
          // dslashParam.p2p_signal == STREAM_GATED (the legacy P2P branch records
          // ipcCopyEvent, no longer created cross-clique under MNNVL, and would crash
          // on cuMemImport / event record at runtime).  All non-NVSHMEM policies are
          // now converted + enabled: DslashBasic, DslashFusedExterior, the GDR family
          // (GDR, FUSED_GDR, GDR_RECV, FUSED_GDR_RECV), the *-ZERO_COPY_PACK send
          // policies, the *-ZERO_COPY_PACK_GDR_RECV policies, and the zero-copy-RECEIVE
          // policies (ZERO_COPY, FUSED_ZERO_COPY, FUSED_PACK, FUSED_PACK_FUSED_HALO).
          enable_policy(QudaDslashPolicy::QUDA_DSLASH);
          first_active_policy = 0;
          enable_policy(QudaDslashPolicy::QUDA_FUSED_DSLASH);

          // if we have gdr then enable tuning these policies (all stream-gated-capable)
          if (comm_gdr_enabled()) {
            enable_policy(QudaDslashPolicy::QUDA_GDR_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_FUSED_GDR_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_GDR_RECV_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_FUSED_GDR_RECV_DSLASH);
          }

          // ZeroCopy-pack SEND policies are now MNNVL-capable: their bodies honour
          // dslashParam.p2p_signal == STREAM_GATED (wired in DslashZeroCopyPack /
          // DslashFusedZeroCopyPack), so no longer gated off under QUDA_MNNVL.
          if (comm_zero_copy_enabled()) {
            enable_policy(QudaDslashPolicy::QUDA_ZERO_COPY_PACK_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_DSLASH);
          }

          // Zero-copy-pack send + GDR receive: GDR recv lands in the DEVICE ghost and the
          // P2P send leg is stream-gated, so these are MNNVL-capable (converted, mirroring
          // the GDR transform).
          if (comm_gdr_enabled()) {
            enable_policy(QudaDslashPolicy::QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH);
          }

          // The zero-copy-RECEIVE policies (ZERO_COPY, FUSED_ZERO_COPY, FUSED_PACK,
          // FUSED_PACK_FUSED_HALO) are MNNVL-capable: these are MIXED-direction kernels.
          // For a P2P direction pack() forces both the pack output and the halo read to
          // DEVICE memory (color_spinor_field.cpp: "if doing p2p ... pack to and load the
          // halo from device memory"), so the P2P leg is device<->device exactly like the
          // *-ZERO_COPY_PACK send policies and policies 6/7.  setMappedGhost only redirects
          // the NON-P2P directions to the mapped pinned HOST ghost, which MPI fills (no P2P
          // writer ever targets host memory).  So no HOST_NUMA/FABRIC host buffer is needed
          // -- only the stream-gated signalling transform (honoured via dslashParam.p2p_signal
          // == STREAM_GATED in the policy bodies above), mirroring the 6/7 conversion.
          if (comm_zero_copy_enabled()) {
            enable_policy(QudaDslashPolicy::QUDA_ZERO_COPY_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_DSLASH_FUSED_PACK);
            enable_policy(QudaDslashPolicy::QUDA_DSLASH_FUSED_PACK_FUSED_HALO);
          }

          if (comm_nvshmem_enabled()) {
            if (in.size() <= 16) {
              // FIXME until uber dslash gets fine-grained
              // synchronization, we cannot use it with large RHS
              enable_policy(QudaDslashPolicy::QUDA_SHMEM_UBER_PACKINTRA_DSLASH);
              enable_policy(QudaDslashPolicy::QUDA_SHMEM_UBER_PACKFULL_DSLASH);
            }
            enable_policy(QudaDslashPolicy::QUDA_SHMEM_PACKINTRA_DSLASH);
            enable_policy(QudaDslashPolicy::QUDA_SHMEM_PACKFULL_DSLASH);
          }
        }
        // construct string specifying which policies have been enabled
        for (int i = 0; i < (int)QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED; i++) {
          strcat(policy_string, (int)policies[i] == i ? "1" : "0");
        }

       static char *dslash_pack_env = getenv("QUDA_ENABLE_DSLASH_PACK");
       if (dslash_pack_env && strcmp(dslash_pack_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash halo packing");
	 dslash_pack_compute = false;
       }

       static char *dslash_interior_env = getenv("QUDA_ENABLE_DSLASH_INTERIOR");
       if (dslash_interior_env && strcmp(dslash_interior_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash interior computation");
	 dslash_interior_compute = false;
       }

       static char *dslash_exterior_env = getenv("QUDA_ENABLE_DSLASH_EXTERIOR");
       if (dslash_exterior_env && strcmp(dslash_exterior_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash exterior computation");
	 dslash_exterior_compute = false;
       }

       static char *dslash_copy_env = getenv("QUDA_ENABLE_DSLASH_COPY");
       if (dslash_copy_env && strcmp(dslash_copy_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash host-device copying");
	 dslash_copy = false;
       }

       static char *dslash_comms_env = getenv("QUDA_ENABLE_DSLASH_COMMS");
       if (dslash_comms_env && strcmp(dslash_comms_env, "0") == 0) {
	 if (getVerbosity() > QUDA_SILENT) warningQuda("Disabling Dslash communication");
	 dslash_comms = false;
       }
      }

      // before we do policy tuning we must ensure the kernel
      // constituents have been tuned since we can't do nested tuning
      if (!tuned()) {
        disableProfileCount();
        preTune();

        for (auto &p2p : p2p_policies) {

          if (p2p == QudaP2PPolicy::QUDA_P2P_POLICY_DISABLED) continue;

          bool p2p_enabled = comm_peer2peer_enabled_global();
          if (p2p == QudaP2PPolicy::QUDA_P2P_DEFAULT)
            comm_enable_peer2peer(false); // disable p2p if using default policy
          dslashParam.remote_write = (p2p == QudaP2PPolicy::QUDA_P2P_REMOTE_WRITE ? 1 : 0);
          // Signalling mechanism is orthogonal to the data-movement policy:
          // resolved once globally (QUDA_P2P_TRANSPORT / backend default).
          dslashParam.p2p_signal = comm::p2p_signal();

          for (auto &i : policies) {

            if (i == QudaDslashPolicy::QUDA_DSLASH ||
                i == QudaDslashPolicy::QUDA_FUSED_DSLASH ||
                i == QudaDslashPolicy::QUDA_ZERO_COPY_PACK_DSLASH ||
                i == QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_DSLASH ||
                i == QudaDslashPolicy::QUDA_ZERO_COPY_DSLASH ||
                i == QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_DSLASH ||
                i == QudaDslashPolicy::QUDA_DSLASH_FUSED_PACK ||
                i == QudaDslashPolicy::QUDA_DSLASH_FUSED_PACK_FUSED_HALO ||
                i == QudaDslashPolicy::QUDA_SHMEM_UBER_PACKINTRA_DSLASH ||
                i == QudaDslashPolicy::QUDA_SHMEM_UBER_PACKFULL_DSLASH ||
                i == QudaDslashPolicy::QUDA_SHMEM_PACKINTRA_DSLASH ||
                i == QudaDslashPolicy::QUDA_SHMEM_PACKFULL_DSLASH) {

              auto dslashImp = DslashFactory<Dslash>::create(i);
              (*dslashImp)(dslash, in, halo, profile);

          } else if (i == QudaDslashPolicy::QUDA_GDR_DSLASH ||
                     i == QudaDslashPolicy::QUDA_FUSED_GDR_DSLASH ||
                     i == QudaDslashPolicy::QUDA_GDR_RECV_DSLASH ||
                     i == QudaDslashPolicy::QUDA_FUSED_GDR_RECV_DSLASH ||
                     i == QudaDslashPolicy::QUDA_ZERO_COPY_PACK_GDR_RECV_DSLASH ||
                     i == QudaDslashPolicy::QUDA_FUSED_ZERO_COPY_PACK_GDR_RECV_DSLASH) {
              // if we are using GDR policies then we must tune the
              // non-GDR equivalent as well - this ensures that all GPUs
              // will have the required tunecache entries prior to
              // potential process divergence regardless of which GPUs
              // are blacklisted.
              {
                QudaDslashPolicy policy = DslashFactory<Dslash>::blacklist_map(i);
                auto dslashImp = DslashFactory<Dslash>::create(policy);
                (*dslashImp)(dslash, in, halo, profile);
              }

              auto dslashImp = DslashFactory<Dslash>::create(i);
              (*dslashImp)(dslash, in, halo, profile);

            } else if (i != QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED) {
              errorQuda("Unsupported dslash policy %d\n", static_cast<int>(i));
            }

          }

          comm_enable_peer2peer(p2p_enabled); // restore p2p state
        }                                     // p2p policies

        postTune();
        enableProfileCount();
        setPolicyTuning(true);
      }
      dslash_policy_init = true;

      apply(device::get_default_stream());
    }

   virtual ~DslashPolicyTune() { setPolicyTuning(false); }

  private:
   void apply(const qudaStream_t &) override
   {
     TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

     if (tp.aux.x >= static_cast<int>(policies.size())) errorQuda("Requested policy that is outside of range");
     if (static_cast<QudaDslashPolicy>(tp.aux.x) == QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED)  errorQuda("Requested policy is disabled");

     bool p2p_enabled = comm_peer2peer_enabled_global();
     if (p2p_policies[tp.aux.y] == QudaP2PPolicy::QUDA_P2P_DEFAULT) comm_enable_peer2peer(false); // disable p2p if using default policy
     dslashParam.remote_write = (p2p_policies[tp.aux.y] == QudaP2PPolicy::QUDA_P2P_REMOTE_WRITE ? 1 : 0); // set whether we are using remote packing writes or copy engines
     // Signalling mechanism is orthogonal to the data-movement policy: resolved
     // once globally via comm::p2p_signal() (QUDA_P2P_TRANSPORT / backend default).
     dslashParam.p2p_signal = comm::p2p_signal();

     auto dslashImp = DslashFactory<Dslash>::create(static_cast<QudaDslashPolicy>(tp.aux.x));
     (*dslashImp)(dslash, in, halo, profile);

     // restore p2p state
     comm_enable_peer2peer(p2p_enabled);
   }

   // Find the best dslash policy
   bool advanceAux(TuneParam &param) const override
   {
     while ((unsigned)param.aux.x < policies.size()-1) {
       param.aux.x++;
       if (policies[param.aux.x] != QudaDslashPolicy::QUDA_DSLASH_POLICY_DISABLED) return true;
     }
     param.aux.x = first_active_policy;

     while ((unsigned)param.aux.y < p2p_policies.size()-1) {
       param.aux.y++;
       if (p2p_policies[param.aux.y] != QudaP2PPolicy::QUDA_P2P_POLICY_DISABLED) return true;
     }
     param.aux.y = first_active_p2p_policy;

     return false;
   }

   bool advanceTuneParam(TuneParam &param) const override { return advanceAux(param); }

   void initTuneParam(TuneParam &param) const override {
     Tunable::initTuneParam(param);
     param.aux.x = first_active_policy;
     param.aux.y = first_active_p2p_policy;
     param.aux.z = 0;
   }

   void defaultTuneParam(TuneParam &param) const override {
     Tunable::defaultTuneParam(param);
     param.aux.x = first_active_policy;
     param.aux.y = first_active_p2p_policy;
     param.aux.z = 0;
   }

   TuneKey tuneKey() const override {
     KernelType kernel_type = dslashParam.kernel_type;
     dslashParam.kernel_type = KERNEL_POLICY;
     TuneKey key = dslash.tuneKey();
     strcat(key.aux, comm_dim_topology_string());
     strcat(key.aux, comm_config_string()); // any change in P2P/GDR will be stored as a separate tunecache entry
     strcat(key.aux, policy_string);        // any change in policies enabled will be stored as a separate entry
     dslashParam.kernel_type = kernel_type;
     return key;
   }

   long long flops() const override {
     KernelType kernel_type = dslashParam.kernel_type;
     dslashParam.kernel_type = KERNEL_POLICY;
     long long flops_ = dslash.flops();
     dslashParam.kernel_type = kernel_type;
     return flops_;
   }

   long long bytes() const override {
     KernelType kernel_type = dslashParam.kernel_type;
     dslashParam.kernel_type = KERNEL_POLICY;
     long long bytes_ = dslash.bytes();
     dslashParam.kernel_type = kernel_type;
     return bytes_;
   }

   void preTune() override { out.backup(); }

   void postTune() override { out.restore(); }

   int32_t getTuneRank() const override { return dslash.getTuneRank(); }
  };

  } // namespace dslash

} // namespace quda

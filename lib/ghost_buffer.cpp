#include <ghost_buffer.h>

namespace quda
{
  void GhostBuffer::allocate(size_t ghost_bytes) const
  {
    // only allocate if not already allocated or buffer required is bigger than previously
    if (!initGhostFaceBuffer || ghost_bytes > ghostFaceBytes) {

      if (initGhostFaceBuffer) {
        if (ghostFaceBytes) {
          // remove potential for inter-process race conditions
          // ensures that all outstanding communication is complete
          // before we free any comms buffers
          qudaDeviceSynchronize();
          comm_barrier();
          for (int b = 0; b < 2; b++) {
            device_comms_pinned_free(ghost_recv_buffer_d[b]);
            device_comms_pinned_free(ghost_send_buffer_d[b]);
            host_free(ghost_pinned_send_buffer_h[b]);
            host_free(ghost_pinned_recv_buffer_h[b]);
          }
        }
      }

      if (ghost_bytes > 0) {
        for (int b = 0; b < 2; ++b) {
          // gpu receive buffer (use pinned allocator to avoid this being redirected, e.g., by QDPJIT)
          ghost_recv_buffer_d[b] = device_comms_pinned_malloc(ghost_bytes);
          // silence any false cuda-memcheck initcheck errors
          qudaMemset(ghost_recv_buffer_d[b], 0, ghost_bytes);

          // gpu send buffer (use pinned allocator to avoid this being redirected, e.g., by QDPJIT)
          ghost_send_buffer_d[b] = device_comms_pinned_malloc(ghost_bytes);
          // silence any false cuda-memcheck initcheck errors
          qudaMemset(ghost_send_buffer_d[b], 0, ghost_bytes);

          // pinned buffer used for sending
          ghost_pinned_send_buffer_h[b] = mapped_malloc(ghost_bytes);

          // set the matching device-mapped pointer
          ghost_pinned_send_buffer_hd[b] = get_mapped_device_pointer(ghost_pinned_send_buffer_h[b]);

          // pinned buffer used for receiving
          ghost_pinned_recv_buffer_h[b] = mapped_malloc(ghost_bytes);

          // set the matching device-mapped pointer
          ghost_pinned_recv_buffer_hd[b] = get_mapped_device_pointer(ghost_pinned_recv_buffer_h[b]);
        }

        initGhostFaceBuffer = true;
        ghostFaceBytes = ghost_bytes;
      }

      ghost_field_reset = true; // this signals that we must reset the IPC comms
    }
  }

  void GhostBuffer::free(void)
  {
    destroyIPCComms();
    if (!initGhostFaceBuffer) return;

    for (int b = 0; b < 2; b++) {
      // free receive buffer
      if (ghost_recv_buffer_d[b]) device_comms_pinned_free(ghost_recv_buffer_d[b]);
      ghost_recv_buffer_d[b] = nullptr;

      // free send buffer
      if (ghost_send_buffer_d[b]) device_comms_pinned_free(ghost_send_buffer_d[b]);
      ghost_send_buffer_d[b] = nullptr;

      // free pinned send memory buffer
      if (ghost_pinned_recv_buffer_h[b]) host_free(ghost_pinned_recv_buffer_h[b]);

      // free pinned send memory buffer
      if (ghost_pinned_send_buffer_h[b]) host_free(ghost_pinned_send_buffer_h[b]);

      ghost_pinned_recv_buffer_h[b] = nullptr;
      ghost_pinned_recv_buffer_hd[b] = nullptr;
      ghost_pinned_send_buffer_h[b] = nullptr;
      ghost_pinned_send_buffer_hd[b] = nullptr;
    }
    initGhostFaceBuffer = false;
  }

  void GhostBuffer::createIPCComms()
  {
    if (initIPCComms) return;
    createIPCBuffers();
    createIPCHandles();
    initIPCComms = true;
  }

  void GhostBuffer::createIPCBuffers()
  {
    // if (!initComms) errorQuda("Can only be called after create comms");
    if ((!ghost_recv_buffer_d[0] || !ghost_recv_buffer_d[1]) && comm_size() > 1)
      errorQuda("ghost_field appears not to be allocated");

    for (int b = 0; b < 2; b++) {
      // set remote send buffer to ghost receive buffers on neighboring processes
      comm_create_neighbor_memory(ghost_remote_send_buffer_d[b], ghost_recv_buffer_d[b]);
      // get remote events
      // comm_create_neighbor_event(ipcRemoteCopyEvent[b], ipcCopyEvent[b]);
    }
    ghost_field_reset = false;
  }

  void GhostBuffer::createIPCHandles()
  {
    for (int b = 0; b < 2; b++) {
      // set remote send buffer to ghost receive buffers on neighboring processes
      // comm_create_neighbor_memory(ghost_remote_send_buffer_d[b], ghost_recv_buffer_d[b]);
      // get remote events
      comm_create_neighbor_event(ipcRemoteCopyEvent[b], ipcCopyEvent[b]);
    }

    // Create message handles for IPC synchronization
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;

      for (int dir = 0; dir < 2; dir++) {
        int hop = dir == 0 ? -1 : +1;
        if (comm_peer2peer_enabled(dir, dim)) {
          for (int b = 0; b < 2; b++) {
            // send to processor in forward direction
            mh_send_p2p[b][dim][dir] = comm_declare_send_relative(&buffer_send_p2p[b][dim][dir], dim, hop, sizeof(int));
            // receive from processor in forward direction
            mh_recv_p2p[b][dim][dir]
              = comm_declare_receive_relative(&buffer_recv_p2p[b][dim][dir], dim, hop, sizeof(int));
          }
        }
      }
    }
  }

  void GhostBuffer::destroyIPCComms()
  {
    if (!initIPCComms) return;
    destroyIPCBuffers();
    destroyIPCHandles();
    initIPCComms = false;
  }

  void GhostBuffer::destroyIPCHandles()
  {

    // ensure that all processes bring down their communicators
    // synchronously so that we don't end up in an undefined state
    qudaDeviceSynchronize();
    comm_barrier();

    for (int b = 0; b < 2; b++) {
      // comm_destroy_neighbor_memory(ghost_remote_send_buffer_d[b]);
      comm_destroy_neighbor_event(ipcRemoteCopyEvent[b], ipcCopyEvent[b]);
    }
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;

      for (int b = 0; b < 2; b++) {
        for (int dir = 0; dir < 2; dir++) {
          if (comm_peer2peer_enabled(dir, dim)) {
            if (mh_send_p2p[b][dim][dir]) comm_free(mh_send_p2p[b][dim][dir]);
            if (mh_recv_p2p[b][dim][dir]) comm_free(mh_recv_p2p[b][dim][dir]);
          }
        }
      } // buffer
    }   // iterate over dim

    // local take down complete - now synchronize to ensure globally complete
    qudaDeviceSynchronize();
    comm_barrier();

    // initIPCComms = false;
  }

  void GhostBuffer::destroyIPCBuffers()
  {
    // if (!initIPCComms) return;

    // ensure that all processes bring down their communicators
    // synchronously so that we don't end up in an undefined state
    qudaDeviceSynchronize();
    comm_barrier();

    for (int b = 0; b < 2; b++) {
      comm_destroy_neighbor_memory(ghost_remote_send_buffer_d[b]);
      // comm_destroy_neighbor_event(ipcRemoteCopyEvent[b], ipcCopyEvent[b]);
    }

    // local take down complete - now synchronize to ensure globally complete
    qudaDeviceSynchronize();
    comm_barrier();

    // initIPCComms = false;
  }

} // namespace quda
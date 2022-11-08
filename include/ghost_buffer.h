#pragma once

#include <iostream>
#include <quda_internal.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <object.h>
#include <quda_api.h>

/**
 * @
 *
 */

namespace quda
{

  class GhostBuffer
  {
  public:
    void allocate(size_t ghost_bytes) const;
    static void free(void);
    void createIPCComms();
    void createIPCBuffers();
    void createIPCHandles();
    static void destroyIPCComms();
    static void destroyIPCHandles();
    static void destroyIPCBuffers();
    /**
           Double buffered static GPU halo send buffer
        */
    inline static array<void *, 2> ghost_send_buffer_d = {};

    /**
       Double buffered static GPU halo receive buffer
     */
    inline static array<void *, 2> ghost_recv_buffer_d = {};

    /**
       Double buffered static pinned send buffers
    */
    inline static array<void *, 2> ghost_pinned_send_buffer_h = {};

    /**
       Double buffered static pinned recv buffers
    */
    inline static array<void *, 2> ghost_pinned_recv_buffer_h = {};

    /**
       Mapped version of pinned send buffers
    */
    inline static array<void *, 2> ghost_pinned_send_buffer_hd = {};

    /**
       Mapped version of pinned recv buffers
    */
    inline static array<void *, 2> ghost_pinned_recv_buffer_hd = {};

    /**
       Remote ghost pointer for sending to
    */
    inline static array_3d<void *, 2, QUDA_MAX_DIM, 2> ghost_remote_send_buffer_d;

    /**
       The current size of the static ghost allocation
    */
    inline static size_t ghostFaceBytes = 0;

    /**
       Message handles for receiving
    */
    inline static array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_recv_p2p = {};

    /**
       Message handles for sending
    */
    inline static array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_send_p2p = {};

    /**
       Buffer used by peer-to-peer message handler
    */
    inline static array_3d<int, 2, QUDA_MAX_DIM, 2> buffer_send_p2p = {};

    /**
       Buffer used by peer-to-peer message handler
    */
    inline static array_3d<int, 2, QUDA_MAX_DIM, 2> buffer_recv_p2p = {};

    /**
       Local copy of event used for peer-to-peer synchronization
    */
    inline static array_3d<qudaEvent_t, 2, QUDA_MAX_DIM, 2> ipcCopyEvent = {};

    /**
       Remote copy of event used for peer-to-peer synchronization
    */
    inline static array_3d<qudaEvent_t, 2, QUDA_MAX_DIM, 2> ipcRemoteCopyEvent = {};

    /**
       Whether we have initialized peer-to-peer communication
    */
    inline static bool initIPCComms = false;

    /**
       Whether the ghost buffers have been initialized
    */
    inline static bool initGhostFaceBuffer = false;

    /**
     Bool which is triggered if the ghost field is reset
  */
    inline static bool ghost_field_reset = false;
  };
} // namespace quda
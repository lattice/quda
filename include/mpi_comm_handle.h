#ifndef _COMM_HANDLE_H
#define _COMM_HANDLE_H

#if defined(QMP_COMMS) || defined(MPI_COMMS)
#include <mpi.h>
namespace quda {
  MPI_Comm get_mpi_handle();
}
#endif

#ifdef QMP_COMMS
#include <qmp.h>

#ifdef __cplusplus
extern "C" {
#endif

QMP_status_t QMP_get_mpi_comm(QMP_comm_t comm, void **mpicomm);

#ifdef __cplusplus
}
#endif

#endif

#endif /* _COMM_HANDLE_H */

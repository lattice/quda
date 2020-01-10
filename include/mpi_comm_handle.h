#ifndef _COMM_HANDLE_H
#define _COMM_HANDLE_H

#if defined(QMP_COMMS) || defined(MPI_COMMS)
#include <mpi.h>
extern MPI_Comm MPI_COMM_HANDLE;
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

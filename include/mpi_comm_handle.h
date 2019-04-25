#ifndef _COMM_HANDLE_H
#define _COMM_HANDLE_H

#if defined(QMP_COMMS) || defined(MPI_COMMS)
#include <mpi.h>
extern MPI_Comm MPI_COMM_HANDLE;
#endif

#ifdef QMP_COMMS
extern "C" {
QMP_status_t QMP_get_mpi_comm(QMP_comm_t comm, void **mpicomm);
}
#endif

#endif /* _COMM_HANDLE_H */

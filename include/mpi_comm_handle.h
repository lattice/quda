#ifndef _COMM_HANDLE_H
#define _COMM_HANDLE_H

#include <mpi.h>
extern MPI_Comm MPI_COMM_HANDLE;

extern "C"{
QMP_status_t QMP_get_mpi_comm(QMP_comm_t comm, void** mpicomm);
}

#endif /* _COMM_HANDLE_H */

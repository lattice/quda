#ifndef _COMM_QUDA_H
#define _COMM_QUDA_H

#define BACK_NBR 1
#define FWD_NBR 2

#define X_BACK_NBR 1
#define Y_BACK_NBR 2
#define Z_BACK_NBR 3
#define T_BACK_NBR 4
#define X_FWD_NBR  5
#define Y_FWD_NBR  6
#define Z_FWD_NBR  7
#define T_FWD_NBR  8

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MsgHandle_s MsgHandle;

/* The following routines are implemented over MPI only. */

MsgHandle *	comm_send_to_rank(void*, int, int);
MsgHandle *	comm_recv_from_rank(void*, int, int);
int		comm_get_neighbor_rank(int dx, int dy, int dz, int dt);

/* implemented over MPI, QMP, and single */

int		comm_gpuid();
void            comm_create(int argc, char **argv);
void		comm_init(void);
void		comm_cleanup(void);
void		comm_exit(int);
char *          comm_hostname(void);
void            comm_set_gridsize(const int *X, int nDim);
void		comm_barrier(void);
void            comm_broadcast(void *data, size_t nbytes);
int		comm_rank(void);
int		comm_size(void);
MsgHandle *     comm_declare_send_relative(void *buffer, int i, int dir, size_t bytes);
MsgHandle *     comm_declare_receive_relative(void *buffer, int i, int dir, size_t bytes);
void            comm_free(MsgHandle *handle);
void            comm_start(MsgHandle *handle);
void		comm_wait(MsgHandle *handle);
int             comm_query(MsgHandle *handle);
int             comm_dim_partitioned(int dir);
void            comm_dim_partitioned_set(int dir);
int             comm_dim(int);
int             comm_coords(int);
void		comm_allreduce(double* data);
void            comm_allreduce_int(int* data);
void		comm_allreduce_array(double* data, size_t size);
void		comm_allreduce_max(double* data);
  
#ifdef __cplusplus
}
#endif

#endif /* _COMM_QUDA_H */

#ifndef __MPICOMM_H__
#define __MPICOMM_H_

#define BACK_NBR 1
#define FWD_NBR 2

#ifdef __cplusplus
extern "C" {
#endif

#define X_BACK_NBR 1
#define Y_BACK_NBR 2
#define Z_BACK_NBR 3
#define T_BACK_NBR 4
#define X_FWD_NBR  5
#define Y_FWD_NBR  6
#define Z_FWD_NBR  7
#define T_FWD_NBR  8

void            comm_set_gridsize(int x, int y, int z, int t);
void		comm_init(void);
int		comm_rank(void);
int		comm_size(void);
unsigned long	comm_send(void*, int, int);
unsigned long   comm_send_with_tag(void*, int, int, int);
unsigned long	comm_recv(void*, int, int);
unsigned long   comm_recv_with_tag(void*, int, int, int);
void		comm_wait(unsigned long);
void		comm_allreduce(double* data);
void		comm_allreduce_array(double* data, size_t size);
void		comm_allreduce_max(double* data);
void		comm_barrier(void);
void		comm_exit(int);
void		comm_cleanup(void);
int		comm_gpuid();

#ifdef __cplusplus
}
#endif



#endif
   


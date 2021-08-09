#pragma once

// for QIO_HAS_EXTENDED_LAYOUT, QIO_Index
#include <qio.h>

#ifdef __cplusplus
extern "C" {
#endif
/* These routines get a quda_* prefix to avoid
   potential linker conflicts, with MILC */
int quda_setup_layout(int len[], int nd, int numnodes, int single_parity);
extern int quda_this_node;

#ifdef QIO_HAS_EXTENDED_LAYOUT
int quda_node_number_ext(const int x[], void *arg);
QIO_Index quda_node_index_ext(const int x[], void *arg);
void quda_get_coords_ext(int x[], int node, QIO_Index index, void *arg);
QIO_Index quda_num_sites_ext(int node, void *arg);
#else
int quda_node_number(const int x[]);
int quda_node_index(const int x[]);
void quda_get_coords(int x[], int node, int index);
int quda_num_sites(int node);
#endif

#ifdef __cplusplus
}
#endif

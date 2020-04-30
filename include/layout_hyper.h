#ifndef LAYOUT_HYPER_H
#define LAYOUT_HYPER_H

#ifdef __cplusplus
extern "C" {
#endif
  /* layout_hyper */
  /* These routines get a quda_* prefix to avoid
     linker conflicts with MILC */
  int quda_setup_layout(int len[], int nd, int numnodes);
  int quda_node_number(const int x[]);
  int quda_node_index(const int x[]);
  void quda_get_coords(int x[], int node, int index);
  int quda_num_sites(int node);
  extern int quda_this_node;

#ifdef __cplusplus
}
#endif

#endif //  LAYOUT_HYPER_H

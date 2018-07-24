#ifndef LAYOUT_HYPER_H
#define LAYOUT_HYPER_H

#ifdef __cplusplus
extern "C" {
#endif
  /* layout_hyper */
  int setup_layout(int len[], int nd, int numnodes);
  int node_number(const int x[]);
  int node_index(const int x[]);
  void get_coords(int x[], int node, int index);
  int num_sites(int node);
  extern int this_node;

#ifdef __cplusplus
}
#endif

#endif //  LAYOUT_HYPER_H

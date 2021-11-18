/*
   setup_layout()  sets up layout
   node_number()   returns the node number on which a site lives
   node_index()    returns the index of the site on the node
   get_coords()    gives lattice coords from node & index
*/

// for int max
#include <limits>

#include <stdlib.h>
#include <stdio.h>
#include <qmp.h>
#include <layout_hyper.h>
#include <util_quda.h>

/* The following globals are required:
   QMP_get_logical_topology()
   QMP_logical_topology_is_declared()
   this_node
   QMP_abort()
*/

static int *squaresize = nullptr; /* dimensions of hypercubes */
static int *nsquares = nullptr;   /* number of hypercubes in each direction */
static int ndim;
static size_t *size1[2] = {nullptr, nullptr}, *size2 = nullptr;
static size_t sites_on_node;
static int *mcoord = nullptr;
static bool single_parity = false;

int quda_setup_layout(int len[], int nd, int, int single_parity_)
{
  ndim = nd;
  single_parity = single_parity_;

  if (squaresize) free(squaresize);
  squaresize = (int *)malloc(ndim * sizeof(int));

  if (nsquares) free(nsquares);
  nsquares = (int *)malloc(ndim * sizeof(int));

  if (mcoord) free(mcoord);
  mcoord = (int *)malloc(ndim * sizeof(int));

  /* setup QMP logical topology */
  if (!QMP_logical_topology_is_declared()) {
    if (QMP_declare_logical_topology(nsquares, ndim) != 0) return 1;
  }

  // use the predetermined geometry
  for (int i = 0; i < ndim; i++) {
    nsquares[i] = QMP_get_logical_dimensions()[i];
    squaresize[i] = len[i] / nsquares[i];
  }

  sites_on_node = 1;
  for (int i = 0; i < ndim; ++i) { sites_on_node *= squaresize[i]; }

  if (size1[0]) free(size1[0]);
  size1[0] = (size_t *)malloc(2 * (ndim + 1) * sizeof(size_t));
  size1[1] = size1[0] + ndim + 1;

  if (size2) free(size2);
  size2 = (size_t *)malloc((ndim + 1) * sizeof(size_t));

  size1[0][0] = 1;
  size1[1][0] = 0;
  size2[0] = 1;
  for (int i = 1; i <= ndim; i++) {
    size1[0][i] = size2[i - 1] * (squaresize[i - 1] / 2) + size1[0][i - 1] * (squaresize[i - 1] % 2);
    size1[1][i] = size2[i - 1] * (squaresize[i - 1] / 2) + size1[1][i - 1] * (squaresize[i - 1] % 2);
    size2[i] = size1[0][i] + size1[1][i];
    // printf("%s %i\t%i\t%i\n", __func__, size1[0][i], size1[1][i], size2[i]);
  }
  return 0;
}

int quda_node_number(const int x[])
{
  for (int i = 0; i < ndim; i++) { mcoord[i] = x[i] / squaresize[i]; }
  return QMP_get_node_number_from(mcoord);
}

#ifdef QIO_HAS_EXTENDED_LAYOUT
int quda_node_number_ext(const int x[], void *arg)
{
  (void)arg;
  return quda_node_number(x);
}
#endif // QIO_HAS_EXTENDED_LAYOUT

size_t quda_node_index_helper(const int x[])
{
  size_t r = 0, p = 0;

  for (int i = ndim - 1; i >= 0; --i) {
    r = r * squaresize[i] + (x[i] % squaresize[i]);
    p += x[i];
  }

  if (!single_parity) {
    if (p % 2 == 0) { /* even site */
      r /= 2;
    } else {
      r = (r + sites_on_node) / 2;
    }
  }

  return r;
}

#ifdef QIO_HAS_EXTENDED_LAYOUT
QIO_Index quda_node_index_ext(const int x[], void *arg)
{
  (void)arg;
  return static_cast<QIO_Index>(quda_node_index_helper(x));
}
#else
int quda_node_index(const int x[])
{
  size_t node_idx = quda_node_index_helper(x);

  if (node_idx > static_cast<size_t>(std::numeric_limits<int>::max())) errorQuda("Invalid node_idx %lu", node_idx);

  return static_cast<int>(node_idx);
}
#endif // QIO_HAS_EXTENDED_LAYOUT

void quda_get_coords_helper(int x[], int node, size_t index)
{
  size_t si = index;
  int *m = QMP_get_logical_coordinates_from(node);

  size_t s = 0;
  for (int i = 0; i < ndim; ++i) {
    x[i] = m[i] * squaresize[i];
    s += x[i];
  }

  if (!single_parity) {
    s &= 1;

    if (index >= size1[s][ndim]) {
      index -= size1[s][ndim];
      s ^= 1;
    }

    for (int i = ndim - 1; i > 0; i--) {
      x[i] += 2 * (index / size2[i]);
      index %= size2[i];
      if (index >= size1[s][i]) {
        index -= size1[s][i];
        s ^= 1;
        x[i]++;
      }
    }
    x[0] += 2 * index + s;
  } else {
    // ((t*Z + z) * Y + y) * X + x
    for (int i = ndim - 1; i > 0; i--) {
      x[i] += index / size2[i];
      index %= size2[i];
    }
    x[0] += index;
  }

  free(m);

  /* Check the result */
#ifdef QIO_HAS_EXTENDED_LAYOUT
  size_t node_index = quda_node_index_ext(x, NULL);
#else
  size_t node_index = static_cast<size_t>(quda_node_index(x));
#endif
  if (node_index != si) {
    if (quda_this_node == 0) {
      fprintf(stderr, "get_coords: error in layout!\n");
      for (int i = 0; i < ndim; i++) {
        fprintf(stderr, "%lu\t%lu\t%lu\n", (size_t)size1[0][i], (size_t)size1[1][i], (size_t)size2[i]);
      }
      fprintf(stderr, "%i\tindex=%lu\tx=(", node, (size_t)si);
      for (int i = 0; i < ndim; i++) fprintf(stderr, i < ndim - 1 ? "%i, " : "%i)\n", x[i]);
    }
    QMP_abort(1);
    exit(1);
  }
}

#ifdef QIO_HAS_EXTENDED_LAYOUT
void quda_get_coords_ext(int x[], int node, QIO_Index index, void *arg)
{
  (void)arg;

  if (index > static_cast<QIO_Index>(std::numeric_limits<int>::max()))
    errorQuda("Invalid index %lu", index);
  quda_get_coords_helper(x, node, static_cast<size_t>(index));
}
#else
void quda_get_coords(int x[], int node, int index) { quda_get_coords_helper(x, node, static_cast<size_t>(index)); }
#endif // QIO_HAS_EXTENDED_LAYOUT

/* The number of sites on the specified node */
#ifdef QIO_HAS_EXTENDED_LAYOUT
QIO_Index quda_num_sites_ext(int, void *) { return sites_on_node; }
#else
int quda_num_sites(int)
{
  if (sites_on_node > static_cast<size_t>(std::numeric_limits<int>::max()))
    errorQuda("Invalid sites_on_node %lu", sites_on_node);

  return static_cast<int>(sites_on_node);
}
#endif // QIO_HAS_EXTENDED_LAYOUT

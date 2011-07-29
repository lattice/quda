/******** layout_hyper.c *********/
/* adapted from SciDAC QDP Data Parallel API */
/* Includes new entry "get_sites_on_node" */
/* adapted from MIMD version 6 */

/* ROUTINES WHICH DETERMINE THE DISTRIBUTION OF SITES ON NODES */

/* This version divides the lattice by factors of prime numbers in any of the
   four directions.  It prefers to divide the longest dimensions,
   which mimimizes the area of the surfaces.  Similarly, it prefers
   to divide dimensions which have already been divided, thus not
   introducing more off-node directions.

        S. Gottlieb, May 18, 1999
        The code will start trying to divide with the largest prime factor
        and then work its way down to 2.  The current maximum prime is 53.
        The array of primes "prime[]" may be extended if necessary.

   This requires that the lattice volume be divisible by the number
   of nodes.  Each dimension must be divisible by a suitable factor
   such that the product of the four factors is the number of nodes.

   3/29/00 EVENFIRST is the rule now. CD.
   12/10/00 Fixed so k = MAXPRIMES-1 DT
*/

/*
   setup_layout()  sets up layout
   node_number()   returns the node number on which a site lives
   node_index()    returns the index of the site on the node
   get_coords()    gives lattice coords from node & index
*/


#include <stdlib.h>
#include <stdio.h>
#include <qmp.h>
#include <layout_hyper.h>

//#include <qio_util.h>

/* The following globals are required:

   QMP_get_logical_topology()
   QMP_logical_topology_is_declared()
   this_node
   QMP_abort()

*/

static int *squaresize;   /* dimensions of hypercubes */
static int *nsquares;     /* number of hypercubes in each direction */
static int ndim;
static int *size1[2], *size2;
static int prime[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
static int sites_on_node;
static int *mcoord;

#define MAXPRIMES (sizeof(prime)/sizeof(int))

// MAC this function assumes that the QMP geometry has been predetermined
static void setup_qmp_fixed(int len[], int nd, int numnodes) {
  int i;

  for (i=0; i<ndim; i++) {
    nsquares[i] = QMP_get_logical_dimensions()[i];
    squaresize[i] = len[i]/nsquares[i];
  }

}

static void setup_qmp_grid(int len[], int nd, int numnodes){
  int ndim2, i;
  const int *nsquares2;

  ndim2 = QMP_get_allocated_number_of_dimensions();
  nsquares2 = QMP_get_allocated_dimensions();
  for(i=0; i<ndim; i++) {
    if(i<ndim2) nsquares[i] = nsquares2[i];
    else nsquares[i] = 1;
  }

  for(i=0; i<ndim; i++) {
    if(len[i]%nsquares[i] != 0) {
      printf("LATTICE SIZE DOESN'T FIT GRID\n");
      QMP_abort(0);
    }
    squaresize[i] = len[i]/nsquares[i];
  }
}

static void setup_hyper_prime(int len[], int nd, int numnodes)
{
  int i, j, k, n;

  /* Figure out dimensions of rectangle */
  for(i=0; i<ndim; ++i) {
    squaresize[i] = len[i];
    nsquares[i] = 1;
  }

  n = numnodes; /* remaining number of nodes to be factored */
  k = MAXPRIMES-1;
  while(n>1) {
    /* figure out which prime to divide by starting with largest */
    while( (n%prime[k]!=0) && (k>0) ) --k;

    /* figure out which direction to divide */
    /* find largest divisible dimension of h-cubes */
    /* if one direction with largest dimension has already been
       divided, divide it again.  Otherwise divide first direction
       with largest dimension. */
    j = -1;
    for(i=0; i<ndim; i++) {
      if(squaresize[i]%prime[k]==0) {
        if( (j<0) || (squaresize[i]>squaresize[j]) ) {
          j = i;
        } else if(squaresize[i]==squaresize[j]) {
          if((nsquares[j]==1)&&(nsquares[i]!=1)) j = i;
        }
      }
    }

    /* This can fail if we run out of prime factors in the dimensions */
    if(j<0) {
      if(this_node==0) {
	fprintf(stderr, "LAYOUT: Not enough prime factors in lattice dimensions\n");
      }
      QMP_abort(1);
      exit(1);
    }

    /* do the surgery */
    n /= prime[k];
    squaresize[j] /= prime[k];
    nsquares[j] *= prime[k];
  }
}

int setup_layout(int len[], int nd, int numnodes){
  int i;

  ndim = nd;
  squaresize = (int *) malloc(ndim*sizeof(int));
  nsquares = (int *) malloc(ndim*sizeof(int));
  mcoord = (int *) malloc(ndim*sizeof(int));

  /*
   MAC: The miniminum surface area partitioning is disabled and QUDA
   expects it to be determined by the user or calling application, but
   this functionality is included for possible future use.
  */

#if 0
  if(QMP_get_msg_passing_type()==QMP_GRID) {
    printf("grid\n");
    setup_qmp_grid(len, ndim, numnodes);
  }  else {
    printf("prime\n");    setup_hyper_prime(len, ndim, numnodes);
  }
#else
  setup_qmp_fixed(len, ndim, numnodes); // use the predetermined geometry
#endif

  /* setup QMP logical topology */
  if(!QMP_logical_topology_is_declared()) {
    if(QMP_declare_logical_topology(nsquares, ndim)!=0)
      return 1;
  }

  sites_on_node = 1;
  for(i=0; i<ndim; ++i) {
    sites_on_node *= squaresize[i];
  }

  size1[0] = (int*)malloc(2*(ndim+1)*sizeof(int));
  size1[1] = size1[0] + ndim + 1;
  size2 = (int*)malloc((ndim+1)*sizeof(int));

  size1[0][0] = 1;
  size1[1][0] = 0;
  size2[0] = 1;
  for(i=1; i<=ndim; i++) {
    size1[0][i] = size2[i-1]*(squaresize[i-1]/2)
                + size1[0][i-1]*(squaresize[i-1]%2);
    size1[1][i] = size2[i-1]*(squaresize[i-1]/2)
                + size1[1][i-1]*(squaresize[i-1]%2);
    size2[i] = size1[0][i] + size1[1][i];
    //printf("%i\t%i\t%i\n", size1[0][i], size1[1][i], size2[i]);
  }
  return 0;
}

int node_number(const int x[])
{
  int i;

  for(i=0; i<ndim; i++) {
    mcoord[i] = x[i]/squaresize[i];
  }
  return QMP_get_node_number_from(mcoord);
}

int node_index(const int x[])
{
  int i, r=0, p=0;

  for(i=ndim-1; i>=0; --i) {
    r = r*squaresize[i] + (x[i]%squaresize[i]);
    p += x[i];
  }

  if( p%2==0 ) { /* even site */
    r /= 2;
  } else {
    r = (r+sites_on_node)/2;
  }
  return r;
}

void get_coords(int x[], int node, int index)
{
  int i, s, si;
  int *m;

  si = index;

  m = QMP_get_logical_coordinates_from(node);

  s = 0;
  for(i=0; i<ndim; ++i) {
    x[i] = m[i] * squaresize[i];
    s += x[i];
  }
  s &= 1;

  if(index>=size1[s][ndim]) {
    index -= size1[s][ndim];
    s ^= 1;
  }

  for(i=ndim-1; i>0; i--) {
    x[i] += 2*(index/size2[i]);
    index %= size2[i];
    if(index>=size1[s][i]) {
      index -= size1[s][i];
      s ^= 1;
      x[i]++;
    }
  }
  x[0] += 2*index + s;

  free(m);

  /* Check the result */
  if(node_index(x)!=si) {
    if(this_node==0) {
      fprintf(stderr,"get_coords: error in layout!\n");
      for(i=0; i<ndim; i++) {
	fprintf(stderr,"%i\t%i\t%i\n", size1[0][i], size1[1][i], size2[i]);
      }
      fprintf(stderr,"%i\t%i", node, si);
      for(i=0; i<ndim; i++) fprintf(stderr,"\t%i", x[i]);
      fprintf(stderr,"\n");
    }
    QMP_abort(1);
    exit(1);
  }
}

/* The number of sites on the specified node */
int num_sites(int node){
  return sites_on_node;
}

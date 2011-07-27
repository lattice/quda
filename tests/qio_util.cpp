/* Utilities for testing QIO */
#include <stdio.h>
#include <string.h>

#include <qio_util.h>

void print_m(suN_matrix *a)
{
  int i;
  
  for ( i=0; i< NCLR; i++)
    {
      printf("%f %f %f %f %f %f\n",a->e[i][0].re,a->e[i][0].im,a->e[i][1].re,a->e[i][1].im,a->e[i][2].re,a->e[i][2].im);
    }
  
  return;
}


void vfill_m(suN_matrix *a, int coords[], int rank)
{
  int i,j;
  
  for ( j=0; j< NCLR; j++)
    for ( i=0; i< NCLR; i++)
      {
	a->e[j][i].re = 0.0;
	a->e[j][i].im = 0.0;
      }
  
  for ( j=0; j< NCLR; j++)
    a->e[j][j].re = 100*rank + coords[0] + 
      lattice_size[0]*(coords[1] + lattice_size[1]*
		       (coords[2] + lattice_size[2]*coords[3]));
  return;
}


void vset_M(suN_matrix *field[], int count)
{
  int x[4];
  int index,i;

  for(i = 0; i < count; i++)
    for(x[3] = 0; x[3] < lattice_size[3]; x[3]++)
      for(x[2] = 0; x[2] < lattice_size[2]; x[2]++)
	for(x[1] = 0; x[1] < lattice_size[1]; x[1]++)
	  for(x[0] = 0; x[0] < lattice_size[0]; x[0]++)
	    {
	      if(node_number(x) == this_node){
		index = node_index(x);
		vfill_m(field[i] + index, x, i);
	      }
	    }
}

int vcreate_M(suN_matrix *field[], int count)
{
  int i;
  /* Create an output field */
  for(i = 0; i < count; i++){
    field[i] = (suN_matrix *)malloc(sizeof(suN_matrix)*num_sites(this_node));
    if(field[i] == NULL){
      printf("vcreate_M(%d): Can't malloc field\n",this_node);
      return 1;
    }
  }

  return 0;
}

/* destroy array of fields */
void vdestroy_M (suN_matrix *field[], int count)
{
  int i;
  for(i = 0; i < count; i++)
    free(field[i]);
}


float vcompare_M (suN_matrix *fielda[], suN_matrix *fieldb[], int count)
{
  int i,j,k,m;
  float diff;
  float sum2 = 0;
  
  for(k = 0; k < count; k++)for(m = 0; m < num_sites(this_node); m++)
    {
      for ( j=0; j< NCLR; j++)
	for ( i=0; i< NCLR; i++)
	  {
	    diff = fielda[k][m].e[j][i].re - fieldb[k][m].e[j][i].re;
	    sum2 += diff*diff;
	    diff = fielda[k][m].e[j][i].im - fieldb[k][m].e[j][i].im;
	    sum2 += diff*diff;
	  }
    }

  /* Global sum */
  QMP_sum_float(&sum2);
  return sum2;
}

void vput_M(char *s1, size_t index, int count, void *s2)
{
  suN_matrix **field = (suN_matrix **)s2;
  suN_matrix *dest;
  suN_matrix *src = (suN_matrix *)s1;
  int i;
  
/* For the site specified by "index", move an array of "count" data
   from the read buffer to an array of fields */

  for (i=0;i<count;i++)
    {
      dest = field[i] + index;
      *dest = *(src + i);
    }
}

void vget_M(char *s1, size_t index, int count, void *s2)
{
  suN_matrix **field = (suN_matrix **)s2;
  suN_matrix *src;
  suN_matrix *dest = (suN_matrix *)s1;
  int i;

/* For the site specified by "index", move an array of "count" data
   from the array of fields to the write buffer */
  for (i=0; i<count; i++, dest++)
    {
      src = field[i] + index;
      *dest = *src;
    }
}

/* Copy a subset */
int inside_subset(int x[], int lower[], int upper[])
{
  int i;
  int status = 1;

  for(i = 0; i < 4; i++)
    if(lower[i] > x[i] || upper[i] < x[i]){
      status = 0;
      break;
    }
  
  return status;
}


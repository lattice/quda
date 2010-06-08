#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <test_util.h>
#include "llfat_reference.h"
#include "misc.h"
#include <string.h>

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

typedef struct {   
  float real;	   
  float imag; 
} fcomplex;  

/* specific for double complex */
typedef struct {
  double real;
  double imag;
} dcomplex;

typedef struct { fcomplex e[3][3]; } fsu3_matrix;
typedef struct { fcomplex c[3]; } fsu3_vector;
typedef struct { dcomplex e[3][3]; } dsu3_matrix;
typedef struct { dcomplex c[3]; } dsu3_vector;


#define CADD(a,b,c) { (c).real = (a).real + (b).real;	\
    (c).imag = (a).imag + (b).imag; }
#define CMUL(a,b,c) { (c).real = (a).real*(b).real - (a).imag*(b).imag; \
    (c).imag = (a).real*(b).imag + (a).imag*(b).real; }
#define CSUM(a,b) { (a).real += (b).real; (a).imag += (b).imag; }

/* c = a* * b */
#define CMULJ_(a,b,c) { (c).real = (a).real*(b).real + (a).imag*(b).imag; \
    (c).imag = (a).real*(b).imag - (a).imag*(b).real; }

/* c = a * b* */
#define CMUL_J(a,b,c) { (c).real = (a).real*(b).real + (a).imag*(b).imag; \
    (c).imag = (a).imag*(b).real - (a).real*(b).imag; }

extern int Z[4];
extern int V;
extern int Vh;

template<typename su3_matrix, typename Real>
void 
llfat_scalar_mult_su3_matrix( su3_matrix *a, Real s, su3_matrix *b )
{
    
  int i,j;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      b->e[i][j].real = s*a->e[i][j].real;
      b->e[i][j].imag = s*a->e[i][j].imag;
    }
    
  return;
}

template<typename su3_matrix, typename Real>
void
llfat_scalar_mult_add_su3_matrix(su3_matrix *a,su3_matrix *b, Real s, su3_matrix *c)
{    
  int i,j;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      c->e[i][j].real = a->e[i][j].real + s*b->e[i][j].real;
      c->e[i][j].imag = a->e[i][j].imag + s*b->e[i][j].imag;
    }
    
}

template <typename su3_matrix>
void 
llfat_mult_su3_na(  su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
  int i,j,k;
  typeof(a->e[0][0]) x,y;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      x.real=x.imag=0.0;
      for(k=0;k<3;k++){
	CMUL_J( a->e[i][k] , b->e[j][k] , y );
	CSUM( x , y );
      }
      c->e[i][j] = x;
    }
}

template <typename su3_matrix>
void
llfat_mult_su3_nn( su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
  int i,j,k;
  typeof(a->e[0][0]) x,y;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      x.real=x.imag=0.0;
      for(k=0;k<3;k++){
	CMUL( a->e[i][k] , b->e[k][j] , y );
	CSUM( x , y );
      }
      c->e[i][j] = x;
    }
}

template<typename su3_matrix>
void
llfat_mult_su3_an( su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
  int i,j,k;
  typeof(a->e[0][0]) x,y;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      x.real=x.imag=0.0;
      for(k=0;k<3;k++){
	CMULJ_( a->e[k][i] , b->e[k][j], y );
	CSUM( x , y );
      }
      c->e[i][j] = x;
    }
}





template<typename su3_matrix>
void 
llfat_add_su3_matrix( su3_matrix *a, su3_matrix *b, su3_matrix *c ) 
{
  int i,j;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      CADD( a->e[i][j], b->e[i][j], c->e[i][j] );
    }
}



template<typename su3_matrix, typename Real>
void 
llfat_compute_gen_staple_field(su3_matrix *staple, int mu, int nu, 
			       su3_matrix* mulink, su3_matrix* sitelink, su3_matrix* fatlink, Real coef,
			       int use_staple) 
{
  su3_matrix tmat1,tmat2;
  int i ;
  su3_matrix *fat1;
    
  /* Upper staple */
  /* Computes the staple :
   *                mu (B)
   *               +-------+
   *       nu	   |	   | 
   *	     (A)   |	   |(C)
   *		   X	   X
   *
   * Where the mu link can be any su3_matrix. The result is saved in staple.
   * if staple==NULL then the result is not saved.
   * It also adds the computed staple to the fatlink[mu] with weight coef.
   */
    
  int dx[4];

  /* upper staple */
    
  for(i=0;i < V;i++){	    
	
    fat1 = fatlink + 4*i + mu;
    su3_matrix* A = sitelink + 4*i + nu;
	
    memset(dx, 0, sizeof(dx));
    dx[nu] =1;
    int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
    su3_matrix* B;
    if (use_staple){
      B = mulink + nbr_idx;
    }else{
      B = mulink + 4*nbr_idx + mu;
    }
	
    memset(dx, 0, sizeof(dx));
    dx[mu] =1;
    nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2],dx[1],dx[0]);
    su3_matrix* C = sitelink + 4*nbr_idx + nu;
	
    llfat_mult_su3_nn( A, B,&tmat1);
	
    if(staple!=NULL){/* Save the staple */
      llfat_mult_su3_na( &tmat1, C, &staple[i]); 	    
    } else{ /* No need to save the staple. Add it to the fatlinks */
      llfat_mult_su3_na( &tmat1, C, &tmat2); 	    
      llfat_scalar_mult_add_su3_matrix(fat1, &tmat2, coef, fat1);	    
    }
  }    
  /***************lower staple****************
   *
   *               X       X
   *       nu	   |	   | 
   *	     (A)   |       |(C)
   *		   +-------+
   *                mu (B)
   *
   *********************************************/

  for(i=0;i < V;i++){	    
	
    fat1 = fatlink + 4*i + mu;
    memset(dx, 0, sizeof(dx));
    dx[nu] = -1;
    int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);	
    if (nbr_idx >= V || nbr_idx <0){
      fprintf(stderr, "ERROR: invliad nbr_idx(%d), line=%d\n", nbr_idx, __LINE__);
      exit(1);
    }
    su3_matrix* A = sitelink + 4*nbr_idx + nu;
        
    su3_matrix* B;
    if (use_staple){
      B = mulink + nbr_idx;
    }else{
      B = mulink + 4*nbr_idx + mu;
    }
	
    memset(dx, 0, sizeof(dx));
    dx[mu] = 1;
    nbr_idx = neighborIndexFullLattice(nbr_idx, dx[3], dx[2],dx[1],dx[0]);
    su3_matrix* C = sitelink + 4*nbr_idx + nu;

    llfat_mult_su3_an( A, B,&tmat1);	
    llfat_mult_su3_nn( &tmat1, C,&tmat2);
	
    if(staple!=NULL){/* Save the staple */
      llfat_add_su3_matrix(&staple[i], &tmat2, &staple[i]);
      llfat_scalar_mult_add_su3_matrix(fat1, &staple[i], coef, fat1);
	    
    } else{ /* No need to save the staple. Add it to the fatlinks */
      llfat_scalar_mult_add_su3_matrix(fat1, &tmat2, coef, fat1);	    
    }
  } 
    
} /* compute_gen_staple_site */



/*  Optimized fattening code for the Asq and Asqtad actions.           
 *  I assume that: 
 *  path 0 is the one link
 *  path 2 the 3-staple
 *  path 3 the 5-staple 
 *  path 4 the 7-staple
 *  path 5 the Lapage term.
 *  Path 1 is the Naik term
 *
 */
template <typename su3_matrix, typename Float>
void llfat_cpu(su3_matrix* fatlink, su3_matrix* sitelink, Float* act_path_coeff)
{

  su3_matrix* staple = (su3_matrix *)malloc(V*sizeof(su3_matrix));
  if(staple == NULL){
    fprintf(stderr, "Error: malloc failed for staple in function %s\n", __FUNCTION__);
    exit(1);
  }
    
  su3_matrix* tempmat1 = (su3_matrix *)malloc(V*sizeof(su3_matrix));
  if(tempmat1 == NULL){
    fprintf(stderr, "ERROR:  malloc failed for tempmat1 in function %s\n", __FUNCTION__);
    exit(1);
  }
    
  /* to fix up the Lepage term, included by a trick below */
  Float one_link = (act_path_coeff[0] - 6.0*act_path_coeff[5]);
    

  for (int dir=XUP; dir<=TUP; dir++){

    /* Intialize fat links with c_1*U_\mu(x) */
    for(int i=0;i < V;i ++){
      su3_matrix* fat1 = fatlink +  4*i + dir;
      llfat_scalar_mult_su3_matrix(sitelink+ 4*i + dir, one_link, fat1 );
    }
  }
  for (int dir=XUP; dir<=TUP; dir++){
    for(int nu=XUP; nu<=TUP; nu++){
      if(nu!=dir){
	llfat_compute_gen_staple_field(staple,dir,nu,sitelink, sitelink,fatlink, act_path_coeff[2], 0);

	/* The Lepage term */
	/* Note this also involves modifying c_1 (above) */

		
	llfat_compute_gen_staple_field((su3_matrix*)NULL,dir,nu,staple,sitelink, fatlink, act_path_coeff[5],1);
		
	for(int rho=XUP; rho<=TUP; rho++) {
	  if((rho!=dir)&&(rho!=nu)){
	    llfat_compute_gen_staple_field( tempmat1, dir, rho, staple,sitelink,fatlink, act_path_coeff[3], 1);
	    
	    for(int sig=XUP; sig<=TUP; sig++){
	      if((sig!=dir)&&(sig!=nu)&&(sig!=rho)){
		llfat_compute_gen_staple_field((su3_matrix*)NULL,dir,sig,tempmat1,sitelink,fatlink, act_path_coeff[4], 1);
	      } 
	    }/* sig */

	  } 

	}/* rho */


      } 

    }/* nu */
	
  }/* dir */      
    

  free(staple);
  free(tempmat1);

}


void
llfat_reference(void* fatlink, void* sitelink, QudaPrecision prec, void* act_path_coeff)
{
  switch(prec){
  case QUDA_DOUBLE_PRECISION:
    llfat_cpu((dsu3_matrix*)fatlink, (dsu3_matrix*)sitelink, (double*) act_path_coeff);
    break;
  case QUDA_SINGLE_PRECISION:
    llfat_cpu((fsu3_matrix*)fatlink, (fsu3_matrix*)sitelink, (float*) act_path_coeff);
    break;
  default:
    fprintf(stderr, "ERROR: unsupported precision\n");
    exit(1);
    break;
	
  }

  return;

}

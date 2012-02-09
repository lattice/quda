#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <test_util.h>
#include "llfat_reference.h"
#include "misc.h"
#include <string.h>

#include <quda_internal.h>
#include "face_quda.h"

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
extern int Vs[];
extern int Vsh[];
extern int Vs_x, Vs_y, Vs_z, Vs_t;
extern int Vsh_x, Vsh_y, Vsh_z, Vsh_t;


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
			       su3_matrix* mulink, su3_matrix** sitelink, void** fatlink, Real coef,
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
	
    fat1 = ((su3_matrix*)fatlink[mu]) + i;
    su3_matrix* A = sitelink[nu] + i;
	
    memset(dx, 0, sizeof(dx));
    dx[nu] =1;
    int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
    su3_matrix* B;
    if (use_staple){
      B = mulink + nbr_idx;
    }else{
      B = mulink + nbr_idx;
    }
	
    memset(dx, 0, sizeof(dx));
    dx[mu] =1;
    nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2],dx[1],dx[0]);
    su3_matrix* C = sitelink[nu] + nbr_idx;
	
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
	
    fat1 = ((su3_matrix*)fatlink[mu]) + i;
    memset(dx, 0, sizeof(dx));
    dx[nu] = -1;
    int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);	
    if (nbr_idx >= V || nbr_idx <0){
      fprintf(stderr, "ERROR: invliad nbr_idx(%d), line=%d\n", nbr_idx, __LINE__);
      exit(1);
    }
    su3_matrix* A = sitelink[nu] + nbr_idx;
        
    su3_matrix* B;
    if (use_staple){
      B = mulink + nbr_idx;
    }else{
      B = mulink + nbr_idx;
    }
	
    memset(dx, 0, sizeof(dx));
    dx[mu] = 1;
    nbr_idx = neighborIndexFullLattice(nbr_idx, dx[3], dx[2],dx[1],dx[0]);
    su3_matrix* C = sitelink[nu] + nbr_idx;

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
void llfat_cpu(void** fatlink, su3_matrix** sitelink, Float* act_path_coeff)
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
      su3_matrix* fat1 = ((su3_matrix*)fatlink[dir]) +  i;
      llfat_scalar_mult_su3_matrix(sitelink[dir] + i, one_link, fat1 );
    }
  }




  for (int dir=XUP; dir<=TUP; dir++){
    for(int nu=XUP; nu<=TUP; nu++){
      if(nu!=dir){
	llfat_compute_gen_staple_field(staple,dir,nu,sitelink[dir], sitelink,fatlink, act_path_coeff[2], 0);

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
llfat_reference(void** fatlink, void** sitelink, QudaPrecision prec, void* act_path_coeff)
{
  switch(prec){
  case QUDA_DOUBLE_PRECISION:{
    llfat_cpu((void**)fatlink, (dsu3_matrix**)sitelink, (double*) act_path_coeff);
    break;
  }
  case QUDA_SINGLE_PRECISION:{
    llfat_cpu((void**)fatlink, (fsu3_matrix**)sitelink, (float*) act_path_coeff);
    break;
  }
  default:
    fprintf(stderr, "ERROR: unsupported precision(%d)\n", prec);
    exit(1);
    break;
	
  }

  return;

}

#ifdef MULTI_GPU

template<typename su3_matrix, typename Real>
void 
llfat_compute_gen_staple_field_mg(su3_matrix *staple, int mu, int nu, 
				  su3_matrix* mulink, su3_matrix** ghost_mulink, 
				  su3_matrix** sitelink, su3_matrix** ghost_sitelink, su3_matrix** ghost_sitelink_diag, 
				  void** fatlink, Real coef,
				  int use_staple) 
{
  su3_matrix tmat1,tmat2;
  int i ;
  su3_matrix *fat1;
  

  int X1 = Z[0];  
  int X2 = Z[1];
  int X3 = Z[2];
  int X4 = Z[3];
  int X1h =X1/2;
  
  int X2X1 = X1*X2;
  int X3X2 = X3*X2;
  int X3X1 = X3*X1;  

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
	
    int half_index = i;
    int oddBit =0;
    if (i >= Vh){
      oddBit = 1;
      half_index = i -Vh;
    }
    //int x4 = x4_from_full_index(i);


    
    int sid =half_index;
    int za = sid/X1h;
    int x1h = sid - za*X1h;
    int zb = za/X2;
    int x2 = za - zb*X2;
    int x4 = zb/X3;
    int x3 = zb - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    int x[4] = {x1,x2,x3,x4};
    int space_con[4]={
      (x4*X3X2+x3*X2+x2)/2,
      (x4*X3X1+x3*X1+x1)/2,
      (x4*X2X1+x2*X1+x1)/2,
      (x3*X2X1+x2*X1+x1)/2
    };

    fat1 = ((su3_matrix*)fatlink[mu]) + i;
    su3_matrix* A = sitelink[nu] + i;
	
    memset(dx, 0, sizeof(dx));
    dx[nu] =1;
    int nbr_idx;
    
    su3_matrix* B;  
    if (use_staple){
      if (x[nu] + dx[nu]  >= Z[nu]){
	B =  ghost_mulink[nu] + Vs[nu] + (1-oddBit)*Vsh[nu] + space_con[nu];
      }else{
	nbr_idx = neighborIndexFullLattice_mg(i, dx[3], dx[2], dx[1], dx[0]);     
	B = mulink + nbr_idx;
      }
    }else{      
      if(x[nu]+dx[nu] >= Z[nu]){ //out of boundary, use ghost data
	B = ghost_sitelink[nu] + 4*Vs[nu] + mu*Vs[nu] + (1-oddBit)*Vsh[nu] + space_con[nu];
      }else{
	nbr_idx = neighborIndexFullLattice_mg(i, dx[3], dx[2], dx[1], dx[0]);	
	B = sitelink[mu] + nbr_idx;
      }
    }
	

    //we could be in the ghost link area if mu is T and we are at high T boundary
    su3_matrix* C;
    memset(dx, 0, sizeof(dx));
    dx[mu] =1;    
    if(x[mu] + dx[mu] >= Z[mu]){ //out of boundary, use ghost data
      C = ghost_sitelink[mu] + 4*Vs[mu] + nu*Vs[mu] + (1-oddBit)*Vsh[mu] + space_con[mu];
    }else{
      nbr_idx = neighborIndexFullLattice_mg(i, dx[3], dx[2],dx[1],dx[0]);    
      C = sitelink[nu] + nbr_idx;
    }

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
	    
    int half_index = i;
    int oddBit =0;
    if (i >= Vh){
      oddBit = 1;
      half_index = i -Vh;
    }

    int sid =half_index;
    int za = sid/X1h;
    int x1h = sid - za*X1h;
    int zb = za/X2;
    int x2 = za - zb*X2;
    int x4 = zb/X3;
    int x3 = zb - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    int x[4] = {x1,x2,x3,x4};
    int space_con[4]={
      (x4*X3X2+x3*X2+x2)/2,
      (x4*X3X1+x3*X1+x1)/2,
      (x4*X2X1+x2*X1+x1)/2,
      (x3*X2X1+x2*X1+x1)/2
    };

    //int x4 = x4_from_full_index(i);

    fat1 = ((su3_matrix*)fatlink[mu]) + i;

    //we could be in the ghost link area if nu is T and we are at low T boundary    
    su3_matrix* A;
    memset(dx, 0, sizeof(dx));
    dx[nu] = -1;

    int nbr_idx;
    if(x[nu] + dx[nu] < 0){ //out of boundary, use ghost data
      A = ghost_sitelink[nu] + nu*Vs[nu] + (1-oddBit)*Vsh[nu] + space_con[nu];
    }else{
      nbr_idx = neighborIndexFullLattice_mg(i, dx[3], dx[2], dx[1], dx[0]);	
      A = sitelink[nu] + nbr_idx;
    }
    
    su3_matrix* B;
    if (use_staple){
      nbr_idx = neighborIndexFullLattice_mg(i, dx[3], dx[2], dx[1], dx[0]);     
      if (x[nu] + dx[nu]  < 0){
	B =  ghost_mulink[nu] + (1-oddBit)*Vsh[nu] + space_con[nu];
      }else{
	B = mulink + nbr_idx;
      }
    }else{      
      if(x[nu] + dx[nu] < 0){ //out of boundary, use ghost data
	B = ghost_sitelink[nu] + mu*Vs[nu] + (1-oddBit)*Vsh[nu] + space_con[nu];	
      }else{
	nbr_idx = neighborIndexFullLattice_mg(i, dx[3], dx[2], dx[1], dx[0]);
	B = sitelink[mu] + nbr_idx;
      }
    }

    //we could be in the ghost link area if nu is T and we are at low T boundary
    // or mu is T and we are on high T boundary
    su3_matrix* C;
    memset(dx, 0, sizeof(dx));
    dx[nu] = -1;
    dx[mu] = 1;
    nbr_idx = neighborIndexFullLattice_mg(i, dx[3], dx[2],dx[1],dx[0]);
    
    //space con must be recomputed because we have coodinates change in 2 directions
    int new_x1, new_x2, new_x3, new_x4;
    new_x1 = (x[0] + dx[0] + Z[0])%Z[0];
    new_x2 = (x[1] + dx[1] + Z[1])%Z[1];
    new_x3 = (x[2] + dx[2] + Z[2])%Z[2];
    new_x4 = (x[3] + dx[3] + Z[3])%Z[3];
    int new_x[4] = {new_x1, new_x2, new_x3, new_x4};
    space_con[0] = (new_x4*X3X2 + new_x3*X2 + new_x2)/2;
    space_con[1] = (new_x4*X3X1 + new_x3*X1 + new_x1)/2;
    space_con[2] = (new_x4*X2X1 + new_x2*X1 + new_x1)/2;
    space_con[3] = (new_x3*X2X1 + new_x2*X1 + new_x1)/2;

    if( (x[nu] + dx[nu]) < 0  && (x[mu] + dx[mu] >= Z[mu])){
      //find the other 2 directions, dir1, dir2
      //with dir2 the slowest changing direction
      int dir1, dir2; //other two dimensions
      for(dir1=0; dir1 < 4; dir1 ++){
	if(dir1 != nu && dir1 != mu){
	  break;
	}
      }
      for(dir2=0; dir2 < 4; dir2 ++){
	if(dir2 != nu && dir2 != mu && dir2 != dir1){
	  break;
	}
      }  
      C = ghost_sitelink_diag[nu*4+mu] +  oddBit*Z[dir1]*Z[dir2]/2 + (new_x[dir2]*Z[dir1]+new_x[dir1])/2;	
    }else if (x[nu] + dx[nu] < 0){
      C = ghost_sitelink[nu] + nu*Vs[nu] + oddBit*Vsh[nu]+ space_con[nu];
    }else if (x[mu] + dx[mu] >= Z[mu]){
      C = ghost_sitelink[mu] + 4*Vs[mu] + nu*Vs[mu] + oddBit*Vsh[mu]+space_con[mu];
    }else{
      C = sitelink[nu] + nbr_idx;
    }
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


template <typename su3_matrix, typename Float>
void llfat_cpu_mg(void** fatlink, su3_matrix** sitelink, su3_matrix** ghost_sitelink,
		  su3_matrix** ghost_sitelink_diag, Float* act_path_coeff)
{
  QudaPrecision prec;
  if (sizeof(Float) == 4){
    prec = QUDA_SINGLE_PRECISION;
  }else{
    prec = QUDA_DOUBLE_PRECISION;
  }
  
  su3_matrix* staple = (su3_matrix *)malloc(V*sizeof(su3_matrix));
  if(staple == NULL){
    fprintf(stderr, "Error: malloc failed for staple in function %s\n", __FUNCTION__);
    exit(1);
  }
  

  su3_matrix* ghost_staple[4];
  su3_matrix* ghost_staple1[4];

  for(int i=0;i < 4;i++){
    ghost_staple[i] = (su3_matrix*)malloc(2*Vs[i]*sizeof(su3_matrix));
    if (ghost_staple[i] == NULL){
      fprintf(stderr, "Error: malloc failed for ghost staple in function %s\n", __FUNCTION__);
      exit(1);
    }
    
    ghost_staple1[i] = (su3_matrix*)malloc(2*Vs[i]*sizeof(su3_matrix));
    if (ghost_staple1[i] == NULL){ 
      fprintf(stderr, "Error: malloc failed for ghost staple1 in function %s\n", __FUNCTION__);
      exit(1);
    }     
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
      su3_matrix* fat1 = ((su3_matrix*)fatlink[dir]) +  i;
      llfat_scalar_mult_su3_matrix(sitelink[dir] + i, one_link, fat1 );
    }
  }

  for (int dir=XUP; dir<=TUP; dir++){
    for(int nu=XUP; nu<=TUP; nu++){
      if(nu!=dir){
	llfat_compute_gen_staple_field_mg(staple,dir,nu,
					  sitelink[dir], (su3_matrix**)NULL, 
					  sitelink, ghost_sitelink, ghost_sitelink_diag, 
					  fatlink, act_path_coeff[2], 0);	
	/* The Lepage term */
	/* Note this also involves modifying c_1 (above) */

	exchange_cpu_staple(Z, staple, (void**)ghost_staple, prec);
	
	llfat_compute_gen_staple_field_mg((su3_matrix*)NULL,dir,nu,
					  staple,ghost_staple, 
					  sitelink, ghost_sitelink, ghost_sitelink_diag, 
					  fatlink, act_path_coeff[5],1);

	for(int rho=XUP; rho<=TUP; rho++) {
	  if((rho!=dir)&&(rho!=nu)){
	    llfat_compute_gen_staple_field_mg( tempmat1, dir, rho, 
					       staple,ghost_staple, 
					       sitelink, ghost_sitelink, ghost_sitelink_diag, 
					       fatlink, act_path_coeff[3], 1);


	    exchange_cpu_staple(Z, tempmat1, (void**)ghost_staple1, prec);
	    
	    for(int sig=XUP; sig<=TUP; sig++){
	      if((sig!=dir)&&(sig!=nu)&&(sig!=rho)){

		llfat_compute_gen_staple_field_mg((su3_matrix*)NULL,dir,sig,
						  tempmat1, ghost_staple1,
						  sitelink, ghost_sitelink, ghost_sitelink_diag, 
						  fatlink, act_path_coeff[4], 1);
		//FIXME
		//return;

	      } 
	    }/* sig */		
	  } 
	}/* rho */
      } 

    }/* nu */
	
  }/* dir */      
  
  free(staple);
  for(int i=0;i < 4;i++){
    free(ghost_staple[i]);
    free(ghost_staple1[i]);
  }
  free(tempmat1);

}



void
llfat_reference_mg(void** fatlink, void** sitelink, void** ghost_sitelink,
		   void** ghost_sitelink_diag, QudaPrecision prec, void* act_path_coeff)
{

  switch(prec){
  case QUDA_DOUBLE_PRECISION:{
    llfat_cpu_mg((void**)fatlink, (dsu3_matrix**)sitelink, (dsu3_matrix**)ghost_sitelink, 
		 (dsu3_matrix**)ghost_sitelink_diag, (double*) act_path_coeff);
    break;
  }
  case QUDA_SINGLE_PRECISION:{
    llfat_cpu_mg((void**)fatlink, (fsu3_matrix**)sitelink, (fsu3_matrix**)ghost_sitelink, 
		 (fsu3_matrix**)ghost_sitelink_diag, (float*) act_path_coeff);
    break;
  }
  default:
    fprintf(stderr, "ERROR: unsupported precision(%d)\n", prec);
    exit(1);
    break;
	
  }

  return;

}


#endif

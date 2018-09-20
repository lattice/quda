#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include "gauge_field.h"
#include <test_util.h>
#include <unitarization_links.h>
#include "misc.h"
#include <string.h>

#include <llfat_reference.h>

#include <quda_internal.h>
#include <complex>

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

using namespace quda;


template<typename real> struct su3_matrix { std::complex<real> e[3][3]; };
template<typename real> struct su3_vector { std::complex<real> e[3]; };

static int Vs[4];
static int Vsh[4];

template<typename su3_matrix, typename Real>
  void 
llfat_scalar_mult_su3_matrix( su3_matrix *a, Real s, su3_matrix *b )
{

  int i,j;
  for(i=0;i<3;i++) for(j=0;j<3;j++){
    b->e[i][j] = s*a->e[i][j];
  }

  return;
}

template<typename su3_matrix, typename Real>
  void
llfat_scalar_mult_add_su3_matrix(su3_matrix *a,su3_matrix *b, Real s, su3_matrix *c)
{    
  int i,j;
  for(i=0;i<3;i++) for(j=0;j<3;j++){
    c->e[i][j] = a->e[i][j] + s*b->e[i][j];
  }

}

template <typename su3_matrix>
  void 
llfat_mult_su3_na(  su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
  int i,j,k;
  typename std::remove_reference<decltype(a->e[0][0])>::type x,y;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
    x=0.0;
    for(k=0;k<3;k++){
      y = a->e[i][k] * conj(b->e[j][k]);
      x += y;
    }
    c->e[i][j] = x;
  }
}

template <typename su3_matrix>
  void
llfat_mult_su3_nn( su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
  int i,j,k;
  typename std::remove_reference<decltype(a->e[0][0])>::type x,y;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
    x=0.0;
    for(k=0;k<3;k++){
      y = a->e[i][k] * b->e[k][j];
      x += y;
    }
    c->e[i][j] = x;
  }
}

template<typename su3_matrix>
  void
llfat_mult_su3_an( su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
  int i,j,k;
  typename std::remove_reference<decltype(a->e[0][0])>::type x,y;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
    x=0.0;
    for(k=0;k<3;k++){
      y = conj(a->e[k][i]) * b->e[k][j];
      x += y;
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
    c->e[i][j] = a->e[i][j] + b->e[i][j];
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

#ifndef MULTI_GPU 
  template<typename su3_matrix, typename Float> 
void computeLongLinkCPU(void** longlink, su3_matrix** sitelink, 
    Float* act_path_coeff)
{

  su3_matrix temp;
  for(int dir=XUP; dir<=TUP; ++dir){
    int dx[4] = {0,0,0,0}; 
    for(int i=0; i<V; ++i){
      // Initialize the longlinks
      su3_matrix* llink = ((su3_matrix*)longlink[dir]) + i;
      llfat_scalar_mult_su3_matrix(sitelink[dir]+i, act_path_coeff[1], llink);
      dx[dir] = 1;
      int nbr_idx = neighborIndexFullLattice(Z, i, dx);
      llfat_mult_su3_nn(llink, sitelink[dir]+nbr_idx, &temp);
      dx[dir] = 2;  
      nbr_idx = neighborIndexFullLattice(Z, i, dx);
      llfat_mult_su3_nn(&temp, sitelink[dir]+nbr_idx, llink);
    }
  }
  return;
}
#else
  template<typename su3_matrix, typename Float>
void computeLongLinkCPU(void** longlink, su3_matrix** sitelinkEx, Float* act_path_coeff)
{
  int E[4];
  for(int dir=0; dir<4; ++dir) E[dir] = Z[dir]+4;


 const int extended_volume = E[3]*E[2]*E[1]*E[0];

  su3_matrix temp;
  for(int t=0; t<Z[3]; ++t){
    for(int z=0; z<Z[2]; ++z){
      for(int y=0; y<Z[1]; ++y){
        for(int x=0; x<Z[0]; ++x){
          const int oddBit = (x+y+z+t)&1;
          int little_index = ((((t*Z[2] + z)*Z[1] + y)*Z[0] + x)/2) + oddBit*Vh;
          int large_index  = (((((t+2)*E[2] + (z+2))*E[1] + (y+2))*E[0] + x+2)/2) + oddBit*(extended_volume/2);
        
      
          for(int dir=XUP; dir<=TUP; ++dir){
            int dx[4] = {0,0,0,0};
            su3_matrix* llink = ((su3_matrix*)longlink[dir]) + little_index;
            llfat_scalar_mult_su3_matrix(sitelinkEx[dir]+large_index, act_path_coeff[1], llink);
            dx[dir] = 1;
            int nbr_index = neighborIndexFullLattice(E, large_index, dx);
            llfat_mult_su3_nn(llink, sitelinkEx[dir]+nbr_index, &temp);
            dx[dir] = 2;
            nbr_index = neighborIndexFullLattice(E, large_index, dx);  
            llfat_mult_su3_nn(&temp, sitelinkEx[dir]+nbr_index, llink);
          }
        } // x
      } // y
    }  // z
  } // t
  return;
}
#endif


void computeLongLinkCPU(void** longlink, void** sitelink, QudaPrecision prec, void* act_path_coeff)
{
  if(longlink){
    switch(prec){
      case QUDA_DOUBLE_PRECISION: 
        computeLongLinkCPU((void**)longlink, (su3_matrix<double>**)sitelink, (double*)act_path_coeff);
        break;

      case QUDA_SINGLE_PRECISION: 
        computeLongLinkCPU((void**)longlink, (su3_matrix<float>**)sitelink, (float*)act_path_coeff);
        break;
      default:
        fprintf(stderr, "ERROR: unsupported precision(%d)\n", prec);
        exit(1);
        break;
    }
  } // if(longlink)

}


  void
llfat_reference(void** fatlink, void** sitelink, QudaPrecision prec, void* act_path_coeff)
{
  Vs[0] = Vs_x;
  Vs[1] = Vs_y;
  Vs[2] = Vs_z;
  Vs[3] = Vs_t;

  Vsh[0] = Vsh_x;
  Vsh[1] = Vsh_y;
  Vsh[2] = Vsh_z;
  Vsh[3] = Vsh_t;


  switch(prec){
    case QUDA_DOUBLE_PRECISION:
      llfat_cpu((void**)fatlink, (su3_matrix<double>**)sitelink, (double*) act_path_coeff);
      break;

    case QUDA_SINGLE_PRECISION:
      llfat_cpu((void**)fatlink, (su3_matrix<float>**)sitelink, (float*) act_path_coeff);
      break;

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
  //int X4 = Z[3];
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

  Vs[0] = Vs_x;
  Vs[1] = Vs_y;
  Vs[2] = Vs_z;
  Vs[3] = Vs_t;

  Vsh[0] = Vsh_x;
  Vsh[1] = Vsh_y;
  Vsh[2] = Vsh_z;
  Vsh[3] = Vsh_t;

  switch(prec){
    case QUDA_DOUBLE_PRECISION:{
      llfat_cpu_mg((void**)fatlink, (su3_matrix<double>**)sitelink, (su3_matrix<double>**)ghost_sitelink,
		   (su3_matrix<double>**)ghost_sitelink_diag, (double*) act_path_coeff);
                                 break;
                               }
    case QUDA_SINGLE_PRECISION:{
      llfat_cpu_mg((void**)fatlink, (su3_matrix<float>**)sitelink, (su3_matrix<float>**)ghost_sitelink,
		   (su3_matrix<float>**)ghost_sitelink_diag, (float*) act_path_coeff);
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

// CPU-style BLAS routines
void cpu_axy(QudaPrecision prec, double a, void* x, void* y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double* dst = (double*)y;
    double* src = (double*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] = a*src[i];
    }
  } else { // QUDA_SINGLE_PRECISION
    float* dst = (float*)y;
    float* src = (float*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] = a*src[i];
    }
  }
}

void cpu_xpy(QudaPrecision prec, void* x, void* y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double* dst = (double*)y;
    double* src = (double*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] += src[i];
    }
  } else { // QUDA_SINGLE_PRECISION
    float* dst = (float*)y;
    float* src = (float*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] += src[i];
    }
  }
}

  // data reordering routines
  template <typename Out, typename In>
  void reorderQDPtoMILC(Out* milc_out, In** qdp_in, int V, int siteSize) {
    for (int i = 0; i < V; i++) {
      for (int dir = 0; dir < 4; dir++) {
        for (int j = 0; j < siteSize; j++) {
          milc_out[(i*4+dir)*siteSize+j] = static_cast<Out>(qdp_in[dir][i*siteSize+j]);
        }
      }
    }
  }

  void reorderQDPtoMILC(void* milc_out, void** qdp_in, int V, int siteSize, QudaPrecision out_precision, QudaPrecision in_precision) {
    if (out_precision == QUDA_SINGLE_PRECISION) {
      if (in_precision == QUDA_SINGLE_PRECISION) {
        reorderQDPtoMILC<float,float>((float*)milc_out, (float**)qdp_in, V, siteSize);
      } else if (in_precision == QUDA_DOUBLE_PRECISION) {
        reorderQDPtoMILC<float,double>((float*)milc_out, (double**)qdp_in, V, siteSize);
      }
    } else if (out_precision == QUDA_DOUBLE_PRECISION) {
      if (in_precision == QUDA_SINGLE_PRECISION) {
        reorderQDPtoMILC<double,float>((double*)milc_out, (float**)qdp_in, V, siteSize);
      } else if (in_precision == QUDA_DOUBLE_PRECISION) {
        reorderQDPtoMILC<double,double>((double*)milc_out, (double**)qdp_in, V, siteSize);
      }
    }
  }

  template <typename Out, typename In>
  void reorderMILCtoQDP(Out** qdp_out, In* milc_in, int V, int siteSize) {
    for (int i = 0; i < V; i++) {
      for (int dir = 0; dir < 4; dir++) {
        for (int j = 0; j < siteSize; j++) {
           qdp_out[dir][i*siteSize+j] = static_cast<Out>(milc_in[(i*4+dir)*siteSize+j]);
        }
      }
    }
  }

  void reorderMILCtoQDP(void** qdp_out, void* milc_in, int V, int siteSize, QudaPrecision out_precision, QudaPrecision in_precision) {
    if (out_precision == QUDA_SINGLE_PRECISION) {
      if (in_precision == QUDA_SINGLE_PRECISION) {
        reorderMILCtoQDP<float,float>((float**)qdp_out, (float*)milc_in, V, siteSize);
      } else if (in_precision == QUDA_DOUBLE_PRECISION) {
        reorderMILCtoQDP<float,double>((float**)qdp_out, (double*)milc_in, V, siteSize);
      }
    } else if (out_precision == QUDA_DOUBLE_PRECISION) {
      if (in_precision == QUDA_SINGLE_PRECISION) {
        reorderMILCtoQDP<double,float>((double**)qdp_out, (float*)milc_in, V, siteSize);
      } else if (in_precision == QUDA_DOUBLE_PRECISION) {
        reorderMILCtoQDP<double,double>((double**)qdp_out, (double*)milc_in, V, siteSize);
      }
    }
  }

// Compute the full HISQ stencil on the CPU. 
// If "eps_naik" is 0, there's no naik correction,
// and this routine skips building the paths in "act_path_coeffs[2]"
void computeHISQLinksCPU(void** fatlink, void** longlink,
        void** fatlink_eps, void** longlink_eps,
        void** sitelink, void* qudaGaugeParamPtr,
        double** act_path_coeffs, double eps_naik)
{
  // Prepare various things
  QudaGaugeParam& qudaGaugeParam = *((QudaGaugeParam*)qudaGaugeParamPtr);
  // Needed for unitarization, following "unitarize_link_test.cpp"
  GaugeFieldParam gParam(0, qudaGaugeParam);
  gParam.pad = 0;
  gParam.link_type   = QUDA_GENERAL_LINKS;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.order = QUDA_MILC_GAUGE_ORDER; // must be true!

  const QudaPrecision prec = qudaGaugeParam.cpu_prec;
  const size_t gSize = prec;

  // Compute n_naiks
  const int n_naiks = (eps_naik == 0.0 ? 1 : 2);

  ///////////////////////////////
  // Create extended CPU field //
  ///////////////////////////////

  void* sitelink_ex[4];
  for(int i=0;i < 4;i++) sitelink_ex[i] = pinned_malloc(V_ex*gaugeSiteSize*gSize);


#ifdef MULTI_GPU
  void* ghost_sitelink[4];
  void* ghost_sitelink_diag[16];
#endif

  int X1=Z[0];
  int X2=Z[1];
  int X3=Z[2];
  int X4=Z[3];

  for(int i=0; i < V_ex; i++){
    int sid = i;
    int oddBit=0;
    if(i >= Vh_ex){
      sid = i - Vh_ex;
      oddBit = 1;
    }

    int za = sid/E1h;
    int x1h = sid - za*E1h;
    int zb = za/E2;
    int x2 = za - zb*E2;
    int x4 = zb/E3;
    int x3 = zb - x4*E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;


    if( x1< 2 || x1 >= X1 +2
        || x2< 2 || x2 >= X2 +2
        || x3< 2 || x3 >= X3 +2
        || x4< 2 || x4 >= X4 +2){
#ifdef MULTI_GPU
      continue;
#endif
    }



    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;

    int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
    if(oddBit){
      idx += Vh;
    }
    for(int dir= 0; dir < 4; dir++){
      char* src = (char*)sitelink[dir];
      char* dst = (char*)sitelink_ex[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
    }//dir
  }//i

  /////////////////////////////////////
  // Allocate all CPU intermediaries //
  /////////////////////////////////////

  void* v_reflink[4];         // V link -- fat7 smeared link
  void* w_reflink[4];         // unitarized V link
  void* w_reflink_ex[4];      // extended W link
  for(int i=0;i < 4;i++){
    v_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    w_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    w_reflink_ex[i] = safe_malloc(V_ex*gaugeSiteSize*gSize);
  }

#ifdef MULTI_GPU
  void* ghost_wlink[4];
  void* ghost_wlink_diag[16];
#endif

  // Copy of V link needed for CPU unitarization routines
  void* v_sitelink = pinned_malloc(4*V*gaugeSiteSize*gSize);


  //FIXME: we have this complication because references takes coeff as float/double
  //        depending on the precision while the GPU code aways take coeff as double
  void* coeff;
  double coeff_dp[6];
  float  coeff_sp[6];

  /////////////////////////////////////////////////////
  // Create V links (fat7 links), 1st path table set //
  /////////////////////////////////////////////////////

  for (int i=0; i < 6;i++) coeff_sp[i] = coeff_dp[i] = act_path_coeffs[0][i];
  coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void*)coeff_dp : (void*)coeff_sp;

  // Only need fat links.
#ifdef MULTI_GPU
  int optflag = 0;
  //we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs_t
  int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
  for (int i=0; i < 4; i++) ghost_sitelink[i] = safe_malloc(8*Vs[i]*gaugeSiteSize*gSize);

  /*
     nu |     |
        |_____|
          mu
     */

  for(int nu=0;nu < 4;nu++){
    for(int mu=0; mu < 4;mu++){
      if(nu == mu){
        ghost_sitelink_diag[nu*4+mu] = NULL;
      }else{
        //the other directions
        int dir1, dir2;
        for(dir1= 0; dir1 < 4; dir1++){
          if(dir1 !=nu && dir1 != mu){
            break;
          }
        }
        for(dir2=0; dir2 < 4; dir2++){
          if(dir2 != nu && dir2 != mu && dir2 != dir1){
            break;
          }
        }
        ghost_sitelink_diag[nu*4+mu] = safe_malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
        memset(ghost_sitelink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
      }

    }
  }
  exchange_cpu_sitelink(gParam.x, sitelink, ghost_sitelink, ghost_sitelink_diag, prec, &qudaGaugeParam, optflag);
  llfat_reference_mg(v_reflink, sitelink, ghost_sitelink, ghost_sitelink_diag, prec, coeff);
#else
  llfat_reference(v_reflink, sitelink, prec, coeff);
#endif

  /////////////////////////////////////////
  // Create W links (unitarized V links) //
  /////////////////////////////////////////

  // This is based on "unitarize_link_test.cpp"

  // Format change
  reorderQDPtoMILC(v_sitelink,v_reflink,V,gaugeSiteSize,prec,prec);
  /*if (prec == QUDA_DOUBLE_PRECISION){
    double* link = reinterpret_cast<double*>(v_sitelink);
    for(int dir=0; dir<4; ++dir){
      double* slink = reinterpret_cast<double*>(v_reflink[dir]);
      for(int i=0; i<V; ++i){
        for(int j=0; j<gaugeSiteSize; j++){
          link[(i*4 + dir)*gaugeSiteSize + j] = slink[i*gaugeSiteSize + j];
        }
      }
    }
  } else if(prec == QUDA_SINGLE_PRECISION){
    float* link = reinterpret_cast<float*>(v_sitelink);
    for(int dir=0; dir<4; ++dir){
      float* slink = reinterpret_cast<float*>(v_reflink[dir]);
      for(int i=0; i<V; ++i){
        for(int j=0; j<gaugeSiteSize; j++){
          link[(i*4 + dir)*gaugeSiteSize + j] = slink[i*gaugeSiteSize + j];
        }
      }
    }
  }*/

  // Prepare cpuGaugeFields for unitarization
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.gauge = v_sitelink;
  cpuGaugeField* cpuVLink = new cpuGaugeField(gParam);

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  cpuGaugeField* cpuWLink = new cpuGaugeField(gParam);

  // unitarize
  unitarizeLinksCPU(*cpuWLink, *cpuVLink);

  // Copy back into "w_reflink"
  reorderMILCtoQDP(w_reflink,cpuWLink->Gauge_p(),V,gaugeSiteSize,prec,prec);
  /*if (prec == QUDA_DOUBLE_PRECISION){
    double* link = reinterpret_cast<double*>(cpuWLink->Gauge_p());
    for(int dir=0; dir<4; ++dir){
      double* slink = reinterpret_cast<double*>(w_reflink[dir]);
      for(int i=0; i<V; ++i){
        for(int j=0; j<gaugeSiteSize; j++){
          slink[i*gaugeSiteSize + j] = link[(i*4 + dir)*gaugeSiteSize + j];
        }
      }
    }
  } else if(prec == QUDA_SINGLE_PRECISION){
    float* link = reinterpret_cast<float*>(cpuWLink->Gauge_p());
    for(int dir=0; dir<4; ++dir){
      float* slink = reinterpret_cast<float*>(w_reflink[dir]);
      for(int i=0; i<V; ++i){
        for(int j=0; j<gaugeSiteSize; j++){
          slink[i*gaugeSiteSize + j] = link[(i*4 + dir)*gaugeSiteSize + j];
        }
      }
    }
  }*/


  // Clean up cpuGaugeFields, we don't need them anymore.

  delete cpuVLink;
  delete cpuWLink;

  ///////////////////////////////////
  // Prepare for extended W fields //
  ///////////////////////////////////

  for(int i=0; i < V_ex; i++) {
    int sid = i;
    int oddBit=0;
    if(i >= Vh_ex){
      sid = i - Vh_ex;
      oddBit = 1;
    }

    int za = sid/E1h;
    int x1h = sid - za*E1h;
    int zb = za/E2;
    int x2 = za - zb*E2;
    int x4 = zb/E3;
    int x3 = zb - x4*E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;


    if( x1< 2 || x1 >= X1 +2
        || x2< 2 || x2 >= X2 +2
        || x3< 2 || x3 >= X3 +2
        || x4< 2 || x4 >= X4 +2){
#ifdef MULTI_GPU
      continue;
#endif
    }



    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;

    int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
    if(oddBit){
      idx += Vh;
    }
    for(int dir= 0; dir < 4; dir++){
      char* src = (char*)w_reflink[dir];
      char* dst = (char*)w_reflink_ex[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
    }//dir
  }//i

  //////////////////////////////
  // Create extended W fields //
  //////////////////////////////

#ifdef MULTI_GPU
  optflag = 0;
  //we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs_t
  for (int i=0; i < 4; i++) ghost_wlink[i] = safe_malloc(8*Vs[i]*gaugeSiteSize*gSize);

  /*
     nu |     |
        |_____|
          mu
     */

  for(int nu=0;nu < 4;nu++){
    for(int mu=0; mu < 4;mu++){
      if(nu == mu){
        ghost_wlink_diag[nu*4+mu] = NULL;
      }else{
        //the other directions
        int dir1, dir2;
        for(dir1= 0; dir1 < 4; dir1++){
          if(dir1 !=nu && dir1 != mu){
            break;
          }
        }
        for(dir2=0; dir2 < 4; dir2++){
          if(dir2 != nu && dir2 != mu && dir2 != dir1){
            break;
          }
        }
        ghost_wlink_diag[nu*4+mu] = safe_malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
        memset(ghost_wlink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
      }

    }
  }
#endif

  ////////////////////////////////////////////
  // Prepare to create Naiks, 3rd table set //
  ////////////////////////////////////////////

  if (n_naiks > 1) {

    for (int i=0; i < 6;i++) coeff_sp[i] = coeff_dp[i] = act_path_coeffs[2][i];
    coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void*)coeff_dp : (void*)coeff_sp;

#ifdef MULTI_GPU

    exchange_cpu_sitelink(qudaGaugeParam.X, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, &qudaGaugeParam, optflag);
    llfat_reference_mg(fatlink, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, coeff);
  
    {
      int R[4] = {2,2,2,2};
      exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, w_reflink_ex, QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0, 4);
      computeLongLinkCPU(longlink, w_reflink_ex, qudaGaugeParam.cpu_prec, coeff);
    }
#else
    llfat_reference(fatlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
    computeLongLinkCPU(longlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
#endif

    // Rescale fat and long links into eps links
    for (int i = 0; i < 4; i++) {
      cpu_axy(prec, eps_naik, fatlink[i], fatlink_eps[i], V*gaugeSiteSize);
      cpu_axy(prec, eps_naik, longlink[i], longlink_eps[i], V*gaugeSiteSize);
    }
  }

  /////////////////////////////////////////////////////////////
  // Prepare to create X links and long links, 2nd table set //
  /////////////////////////////////////////////////////////////

  for (int i=0; i < 6;i++) coeff_sp[i] = coeff_dp[i] = act_path_coeffs[1][i];
  coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void*)coeff_dp : (void*)coeff_sp;

#ifdef MULTI_GPU
  optflag = 0;

  // We've already built the extended W fields.

  exchange_cpu_sitelink(qudaGaugeParam.X, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, &qudaGaugeParam, optflag);
  llfat_reference_mg(fatlink, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, coeff);

  {
    int R[4] = {2,2,2,2};
    exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, w_reflink_ex, QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0, 4);
    computeLongLinkCPU(longlink, w_reflink_ex, qudaGaugeParam.cpu_prec, coeff);
  }
#else
  llfat_reference(fatlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
  computeLongLinkCPU(longlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
#endif

  if (n_naiks > 1) {
    // Accumulate into eps links.
    for (int i = 0; i < 4; i++) {
      cpu_xpy(prec, fatlink[i], fatlink_eps[i], V*gaugeSiteSize);
      cpu_xpy(prec, longlink[i], longlink_eps[i], V*gaugeSiteSize);
    }
  }

  //////////////
  // Clean up //
  //////////////

  for(int i=0; i < 4; i++){
    host_free(sitelink_ex[i]);
    host_free(v_reflink[i]);
    host_free(w_reflink[i]);
    host_free(w_reflink_ex[i]);
  }
  host_free(v_sitelink);

#ifdef MULTI_GPU
  for(int i=0; i<4; i++){
    host_free(ghost_sitelink[i]);
    host_free(ghost_wlink[i]);
    for(int j=0;j <4; j++){
      if (i==j) continue;
      host_free(ghost_sitelink_diag[i*4+j]);
      host_free(ghost_wlink_diag[i*4+j]);
    }
  }
#endif

}


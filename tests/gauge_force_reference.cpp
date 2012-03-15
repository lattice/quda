#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "quda.h"
#include "test_util.h"
#include "misc.h"
#include "gauge_force_reference.h"

extern int Z[4];
extern int V;
extern int Vh;
extern int Vh_ex;
extern int E[4];


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

#define CONJG(a,b) { (b).real = (a).real; (b).imag = -(a).imag; }

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

typedef struct { 
    fcomplex m01,m02,m12; 
    float m00im,m11im,m22im; 
    float space; 
} fanti_hermitmat;

typedef struct { 
    dcomplex m01,m02,m12; 
    double m00im,m11im,m22im; 
    double space; 
} danti_hermitmat;

extern int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1);

template<typename su3_matrix>
void  su3_adjoint( su3_matrix *a, su3_matrix *b )
{
    int i,j;
    for(i=0;i<3;i++)for(j=0;j<3;j++){
	    CONJG( a->e[j][i], b->e[i][j] );
	}
}


template<typename su3_matrix, typename anti_hermitmat>
void
make_anti_hermitian( su3_matrix *m3, anti_hermitmat *ah3 ) 
{
    
    typeof(ah3->m00im) temp =
	(m3->e[0][0].imag + m3->e[1][1].imag + m3->e[2][2].imag)*0.33333333333333333;
    ah3->m00im = m3->e[0][0].imag - temp;
    ah3->m11im = m3->e[1][1].imag - temp;
    ah3->m22im = m3->e[2][2].imag - temp;
    ah3->m01.real = (m3->e[0][1].real - m3->e[1][0].real)*0.5;
    ah3->m02.real = (m3->e[0][2].real - m3->e[2][0].real)*0.5;
    ah3->m12.real = (m3->e[1][2].real - m3->e[2][1].real)*0.5;
    ah3->m01.imag = (m3->e[0][1].imag + m3->e[1][0].imag)*0.5;
    ah3->m02.imag = (m3->e[0][2].imag + m3->e[2][0].imag)*0.5;
    ah3->m12.imag = (m3->e[1][2].imag + m3->e[2][1].imag)*0.5;
    
}

template <typename anti_hermitmat, typename su3_matrix>
static void
uncompress_anti_hermitian(anti_hermitmat *mat_antihermit,
			  su3_matrix *mat_su3 )
{
    typeof(mat_antihermit->m00im) temp1;
    mat_su3->e[0][0].imag=mat_antihermit->m00im;
    mat_su3->e[0][0].real=0.;
    mat_su3->e[1][1].imag=mat_antihermit->m11im;
    mat_su3->e[1][1].real=0.;
    mat_su3->e[2][2].imag=mat_antihermit->m22im;
    mat_su3->e[2][2].real=0.;
    mat_su3->e[0][1].imag=mat_antihermit->m01.imag;
    temp1=mat_antihermit->m01.real;
    mat_su3->e[0][1].real=temp1;
    mat_su3->e[1][0].real= -temp1;
    mat_su3->e[1][0].imag=mat_antihermit->m01.imag;
    mat_su3->e[0][2].imag=mat_antihermit->m02.imag;
    temp1=mat_antihermit->m02.real;
    mat_su3->e[0][2].real=temp1;
    mat_su3->e[2][0].real= -temp1;
    mat_su3->e[2][0].imag=mat_antihermit->m02.imag;
    mat_su3->e[1][2].imag=mat_antihermit->m12.imag;
    temp1=mat_antihermit->m12.real;
    mat_su3->e[1][2].real=temp1;
    mat_su3->e[2][1].real= -temp1;
    mat_su3->e[2][1].imag=mat_antihermit->m12.imag;    
}

template <typename su3_matrix, typename Float>
static void
scalar_mult_sub_su3_matrix(su3_matrix *a,su3_matrix *b, Float s, su3_matrix *c)
{    
    int i,j;
    for(i=0;i<3;i++){
	for(j=0;j<3;j++){	    
	    c->e[i][j].real = a->e[i][j].real - s*b->e[i][j].real;
	    c->e[i][j].imag = a->e[i][j].imag - s*b->e[i][j].imag;
	}
    }
}

template <typename su3_matrix, typename Float>
static void
scalar_mult_add_su3_matrix(su3_matrix *a,su3_matrix *b, Float s, su3_matrix *c)
{
    int i,j;
    for(i=0;i<3;i++){
	for(j=0;j<3;j++){	    
	    c->e[i][j].real = a->e[i][j].real + s*b->e[i][j].real;
	    c->e[i][j].imag = a->e[i][j].imag + s*b->e[i][j].imag;
	}
    }    
}

template <typename su3_matrix>
static void
mult_su3_nn(su3_matrix* a, su3_matrix* b, su3_matrix* c)
{
    int i,j,k;
    typeof(a->e[0][0]) x,y;
    for(i=0;i<3;i++){
	for(j=0;j<3;j++){
	    x.real=x.imag=0.0;
	    for(k=0;k<3;k++){
		CMUL( a->e[i][k] , b->e[k][j] , y );
		CSUM( x , y );
	    }
	    c->e[i][j] = x;
	}
    }    
}
template<typename su3_matrix>
static void 
mult_su3_an( su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
    int i,j,k;
    typeof(a->e[0][0]) x,y;
    for(i=0;i<3;i++){
	for(j=0;j<3;j++){
	    x.real=x.imag=0.0;
	    for(k=0;k<3;k++){
		CMULJ_( a->e[k][i] , b->e[k][j], y );
		CSUM( x , y );
	    }
	    c->e[i][j] = x;
	}
    }
}

template<typename su3_matrix>
static void
mult_su3_na(  su3_matrix *a, su3_matrix *b, su3_matrix *c )
{
    int i,j,k;
    typeof(a->e[0][0]) x,y;
    for(i=0;i<3;i++){
	for(j=0;j<3;j++){
	    x.real=x.imag=0.0;
	    for(k=0;k<3;k++){
		CMUL_J( a->e[i][k] , b->e[j][k] , y );
		CSUM( x , y );
	    }
	    c->e[i][j] = x;
	}
    }
}

template < typename su3_matrix>
void
print_su3_matrix(su3_matrix *a)
{
    int i, j;
    for(i=0;i < 3; i++){
	for(j=0;j < 3;j++){
	    printf("(%f %f)\t", a->e[i][j].real, a->e[i][j].imag);
	}
	printf("\n");
    }
    
}


int
gf_neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1) 
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }
  int X1 = Z[0];  
  int X2 = Z[1];
  int X3 = Z[2];
  //int X4 = Z[3];
  int X1h =X1/2;

  int za = half_idx/X1h;
  int x1h = half_idx - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2*x1h + x1odd;

#ifdef MULTI_GPU
  x4 = x4+dx4;
  x3 = x3+dx3;
  x2 = x2+dx2;
  x1 = x1+dx1;
  
  int nbr_half_idx = ( (x4+2)*(E[2]*E[1]*E[0]) + (x3+2)*(E[1]*E[0]) + (x2+2)*(E[0]) + (x1+2)) / 2;
#else
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
 
  int nbr_half_idx = (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
#endif

  int oddBitChanged = (dx4+dx3+dx2+dx1)%2;
  if (oddBitChanged){
    oddBit = 1 - oddBit;
  }
  int ret = nbr_half_idx;
  if (oddBit){
#ifdef MULTI_GPU
    ret = Vh_ex + nbr_half_idx;    
#else
    ret = Vh + nbr_half_idx;
#endif
  }
    
  return ret;
}



//this functon compute one path for all lattice sites
template<typename su3_matrix, typename Float>
static void
compute_path_product(su3_matrix* staple, su3_matrix** sitelink, su3_matrix** sitelink_ex_2d,
		     int* path, int len, Float loop_coeff, int dir)
{
  int i, j;

    su3_matrix prev_matrix, curr_matrix, tmat;
    int dx[4];

    for(i=0;i<V;i++){
	memset(dx,0, sizeof(dx));
	memset(&curr_matrix, 0, sizeof(curr_matrix));
	
	curr_matrix.e[0][0].real = 1.0;
	curr_matrix.e[1][1].real = 1.0;
	curr_matrix.e[2][2].real = 1.0;
	
	dx[dir] =1;
	for(j=0; j < len;j++){
	    int lnkdir;

	    prev_matrix = curr_matrix;
	    if (GOES_FORWARDS(path[j])){
		//dx[path[j]] +=1;
		lnkdir = path[j];
	    }else{		
		dx[OPP_DIR(path[j])] -=1;
		lnkdir=OPP_DIR(path[j]);
	    }
	    
	    int nbr_idx = gf_neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
#ifdef MULTI_GPU
	    su3_matrix* lnk = sitelink_ex_2d[lnkdir] + nbr_idx;
#else	    
	    su3_matrix* lnk = sitelink[lnkdir]+ nbr_idx;
#endif
	    if (GOES_FORWARDS(path[j])){
		mult_su3_nn(&prev_matrix, lnk, &curr_matrix);		
	    }else{
		mult_su3_na(&prev_matrix, lnk, &curr_matrix);		
	    }

	    if (GOES_FORWARDS(path[j])){
		dx[path[j]] +=1;
	    }else{		
	      //we already substract one in the code above
	    }	    

	    
	}//j

	su3_adjoint(&curr_matrix, &tmat );
	
	scalar_mult_add_su3_matrix(staple + i , &tmat, loop_coeff, staple+i); 

    }//i
    
    return;
}


template <typename su3_matrix, typename anti_hermitmat, typename Float>
static void
update_mom(anti_hermitmat* momentum, int dir, su3_matrix** sitelink,
	   su3_matrix* staple, Float eb3)
{
    int i;
    for(i=0;i <V; i++){
	su3_matrix tmat1;
	su3_matrix tmat2;
	su3_matrix tmat3;

	su3_matrix* lnk = sitelink[dir] + i;
	su3_matrix* stp = staple + i;
	anti_hermitmat* mom = momentum + 4*i+dir;
	
	mult_su3_na(lnk, stp, &tmat1);
	uncompress_anti_hermitian(mom, &tmat2);
	
	scalar_mult_sub_su3_matrix(&tmat2, &tmat1, eb3, &tmat3);
	make_anti_hermitian(&tmat3, mom);
	
    }
    
}



/* This function only computes one direction @dir
 * 
 */

void
gauge_force_reference_dir(void* refMom, int dir, double eb3, void** sitelink, void** sitelink_ex_2d, QudaPrecision prec, 
			  int **path_dir, int* length, void* loop_coeff, int num_paths)
{
    int i;
    
    void* staple;
    int gSize =  prec;    

    staple = malloc(V* gaugeSiteSize* gSize);
    if (staple == NULL){
	fprintf(stderr, "ERROR: malloc failed for staple in functon %s\n", __FUNCTION__);
	exit(1);
    }
    
    memset(staple, 0,  V*gaugeSiteSize* gSize);
    
    for(i=0;i < num_paths; i++){	
	if (prec == QUDA_DOUBLE_PRECISION){
	    double* my_loop_coeff = (double*)loop_coeff;
	    compute_path_product((dsu3_matrix*)staple, (dsu3_matrix**)sitelink, (dsu3_matrix**)sitelink_ex_2d, 
				 path_dir[i], length[i], my_loop_coeff[i], dir);
	}else{
	    float* my_loop_coeff = (float*)loop_coeff;
	    compute_path_product((fsu3_matrix*)staple, (fsu3_matrix**)sitelink, (fsu3_matrix**)sitelink_ex_2d, 
				 path_dir[i], length[i], my_loop_coeff[i], dir);
	}	
    }        
    
    if (prec == QUDA_DOUBLE_PRECISION){
      update_mom((danti_hermitmat*) refMom, dir, (dsu3_matrix**)sitelink, (dsu3_matrix*)staple, (double)eb3);
    }else{
      update_mom((fanti_hermitmat*)refMom, dir, (fsu3_matrix**)sitelink, (fsu3_matrix*)staple, (float)eb3);
    }
    
    free(staple);
}


void
gauge_force_reference(void* refMom, double eb3, void** sitelink, void** sitelink_ex_2d, QudaPrecision prec, 
		      int ***path_dir, int* length, void* loop_coeff, int num_paths)
{
  for(int dir =0; dir < 4; dir++){
    gauge_force_reference_dir(refMom, dir, eb3, sitelink, sitelink_ex_2d, prec, path_dir[dir],
			      length, loop_coeff, num_paths);
    
  }
  
}

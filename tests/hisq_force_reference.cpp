#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "quda.h"
#include "test_util.h"
#include "misc.h"
#include "hisq_force_reference.h"

extern int Z[4];
extern int V;
extern int Vh;


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

typedef struct { fsu3_vector h[2]; } fhalf_wilson_vector;
typedef struct { dsu3_vector h[2]; } dhalf_wilson_vector;


template<typename su3_matrix>
su3_matrix* get_su3_matrix(int gauge_order, su3_matrix* p, int idx, int dir)
{
  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    return (p + 4*idx + dir);
  }else{ //QDP format
    su3_matrix* data = ((su3_matrix**)p)[dir];
    return data + idx;
  }
}

template<typename su3_matrix>
static void  
su3_adjoint( su3_matrix *a, su3_matrix *b )
{
    int i,j;
    for(i=0;i<3;i++)for(j=0;j<3;j++){
	    CONJG( a->e[j][i], b->e[i][j] );
	}
}

template<typename su3_matrix>
static void  
adjoint_su3_matrix( su3_matrix *a)
{
  su3_matrix b;
  int i,j;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      CONJG( a->e[j][i], b.e[i][j] );
    }

  *a = b;
}

template<typename su3_matrix, typename anti_hermitmat>
static void
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
template <typename su3_matrix, typename Float>
static void
scale_su3_matrix(su3_matrix *a,  Float s)
{
  int i,j;
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){	    
      a->e[i][j].real = a->e[i][j].real * s;
      a->e[i][j].imag = a->e[i][j].imag * s;
    }
  }    
}

template<typename su3_matrix, typename su3_vector>
static void
mult_su3_mat_vec( su3_matrix *a, su3_vector *b, su3_vector *c  )
{
    int i,j;
    typeof(a->e[0][0]) x,y;
    for(i=0;i<3;i++){
	x.real=x.imag=0.0;
	for(j=0;j<3;j++){
	    CMUL( a->e[i][j] , b->c[j] , y );
	    CSUM( x , y );
	}
	c->c[i]=x;
    }
}
template<typename su3_matrix, typename su3_vector>
static void
mult_adj_su3_mat_vec( su3_matrix *a, su3_vector *b, su3_vector *c )
{
    int i,j;
    typeof(a->e[0][0]) x,y,z;
    for(i=0;i<3;i++){
	x.real=x.imag=0.0;
	for(j=0;j<3;j++){
	    CONJG( a->e[j][i], z );
	    CMUL( z , b->c[j], y );
	    CSUM( x , y );
	}
	c->c[i] = x;
    }
}

template<typename su3_vector, typename su3_matrix>
static void
su3_projector( su3_vector *a, su3_vector *b, su3_matrix *c )
{
    int i,j;
    for(i=0;i<3;i++)for(j=0;j<3;j++){
	    CMUL_J( a->c[i], b->c[j], c->e[i][j] );
	}
}

template<typename su3_vector, typename Real>
static void 
scalar_mult_add_su3_vector(su3_vector *a, su3_vector *b, Real s,
			   su3_vector *c)
{    
    int i;
    for(i=0;i<3;i++){
	c->c[i].real = a->c[i].real + s*b->c[i].real;
	c->c[i].imag = a->c[i].imag + s*b->c[i].imag;
    }    
}



template < typename su3_matrix>
static void
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



// Add a matrix multiplication function
template<typename su3_matrix>
static void
matrix_mult_nn(su3_matrix* a, su3_matrix* b, su3_matrix* c){
  // c = a*b
  typeof(c->e[0][0]) x;
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){	
      c->e[i][j].real = 0.;
      c->e[i][j].imag = 0.;
      for(int k=0; k<3; k++){	
	CMUL(a->e[i][k],b->e[k][j],x);
	c->e[i][j].real += x.real;
	c->e[i][j].imag += x.imag;
      } 	
    }	
  }
  return;
}


template<typename su3_matrix>
static void
matrix_mult_an(su3_matrix* a, su3_matrix* b, su3_matrix* c){
  // c = (a^{\dagger})*b
  typeof(c->e[0][0]) x;
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){	
      c->e[i][j].real = 0.;
      c->e[i][j].imag = 0.;
      for(int k=0; k<3; k++){	
	CMULJ_(a->e[k][i],b->e[k][j],x);
	c->e[i][j].real += x.real;
	c->e[i][j].imag += x.imag;
      } 	
    }	
  }
  return;
}



template<typename su3_matrix>
static void
matrix_mult_na(su3_matrix* a, su3_matrix* b, su3_matrix* c){
  // c = a*b^{\dagger}
  typeof(c->e[0][0]) x;
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      c->e[i][j].real = 0.; c->e[i][j].imag = 0.;
      for(int k=0; k<3; k++){
        CMUL_J(a->e[i][k],b->e[j][k],x);
	c->e[i][j].real += x.real;
	c->e[i][j].imag += x.imag;
      }
    }
  }
  return;
}

template<typename su3_matrix>
static void
matrix_mult_aa(su3_matrix* a, su3_matrix* b, su3_matrix* c){
  
  su3_matrix a_adjoint;
  su3_adjoint(a, &a_adjoint);
  matrix_mult_na(&a_adjoint, b, c);
}

template <typename su3_matrix, typename anti_hermitmat, typename Float>
static void
update_mom(anti_hermitmat* momentum, int dir, su3_matrix* sitelink,
	   su3_matrix* staple, Float eb3)
{
    int i;
    for(i=0;i <V; i++){
	su3_matrix tmat1;
	su3_matrix tmat2;
	su3_matrix tmat3;

	su3_matrix* lnk = sitelink + 4*i+dir;
	su3_matrix* stp = staple + i;
	anti_hermitmat* mom = momentum + 4*i+dir;
	
	mult_su3_na(lnk, stp, &tmat1);
	uncompress_anti_hermitian(mom, &tmat2);
	
	scalar_mult_sub_su3_matrix(&tmat2, &tmat1, eb3, &tmat3);
	make_anti_hermitian(&tmat3, mom);
	
    }
    
}


template<typename half_wilson_vector, typename su3_matrix>
static void 
u_shift_hw(half_wilson_vector *src, half_wilson_vector *dest, int dir, su3_matrix* sitelink ) 
{
    int i ;
    int dx[4];
    
    dx[3]=dx[2]=dx[1]=dx[0]=0;
    
    if(GOES_FORWARDS(dir)){	
	dx[dir]=1;	
	for(i=0;i < V; i++){
	    int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
	    half_wilson_vector* hw = src + nbr_idx;
	    su3_matrix* link = sitelink + i*4 + dir;
	    mult_su3_mat_vec(link, &hw->h[0], &dest[i].h[0]);
	    mult_su3_mat_vec(link, &hw->h[1], &dest[i].h[1]);	    
	}	
    }else{
	dx[OPP_DIR(dir)]=-1;
	for(i=0;i < V; i++){
	    int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
	    half_wilson_vector* hw = src + nbr_idx;
	    su3_matrix* link = sitelink + nbr_idx*4 + OPP_DIR(dir);
	    mult_adj_su3_mat_vec(link, &hw->h[0], &dest[i].h[0]);
	    mult_adj_su3_mat_vec(link, &hw->h[1], &dest[i].h[1]);	   
	}	
    }
}



template<typename half_wilson_vector, typename su3_matrix>
static void 
shifted_outer_prod(half_wilson_vector *src, su3_matrix* dest, int dir)
{
    
    int i;
    int dx[4];
    
    dx[3]=dx[2]=dx[1]=dx[0]=0;
    
    if(GOES_FORWARDS(dir)){	
	dx[dir]=1;	
    }else{ dx[OPP_DIR(dir)]=-1; }

    for(i=0;i < V; i++){
      int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
      half_wilson_vector* hw = src + nbr_idx;
      su3_projector( &src[i].h[0], &(hw->h[0]), &dest[i]);
    }	
}


template<typename half_wilson_vector, typename su3_matrix>
static void 
forward_shifted_outer_prod(half_wilson_vector *src, su3_matrix* dest, int dir)
{

  int i;
  int dx[4];
    
  dx[3]=dx[2]=dx[1]=dx[0]=0;
    
  if(GOES_FORWARDS(dir)){	
    dx[dir]=1;	
  }else{ dx[OPP_DIR(dir)]=-1; }

  for(i=0;i < V; i++){
    int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
    half_wilson_vector* hw = src + nbr_idx;
    //su3_projector( &src[i].h[0], &(hw->h[0]), &dest[i]);
    su3_projector( &(hw->h[0]), &src[i].h[0], &dest[i]);
  }	

  return;
}



template <typename half_wilson_vector, typename su3_matrix>
static void
computeLinkOrderedOuterProduct(half_wilson_vector *src, su3_matrix* dest, int gauge_order)
{
  int dx[4];
  for(int i=0; i<V; ++i){
    for(int dir=0; dir<4; ++dir){
      dx[3]=dx[2]=dx[1]=dx[0]=0;
      dx[dir] = 1;
      int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
      half_wilson_vector* hw = src + nbr_idx;
      su3_matrix* p = get_su3_matrix(gauge_order, dest, i, dir);
      su3_projector( &(hw->h[0]), &src[i].h[0], p);
    } // dir
  } // i
  return;
}


template <typename half_wilson_vector, typename su3_matrix>
static void
computeLinkOrderedOuterProduct(half_wilson_vector *src, su3_matrix* dest, size_t nhops, int gauge_order)
{
  int dx[4];
  for(int i=0; i<V; ++i){
    for(int dir=0; dir<4; ++dir){
      dx[3]=dx[2]=dx[1]=dx[0]=0;
      dx[dir] = nhops;
      int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
      half_wilson_vector* hw = src + nbr_idx;
      su3_matrix* p = get_su3_matrix(gauge_order, dest, i, dir);
      su3_projector( &(hw->h[0]), &src[i].h[0], p);
    } // dir
  } // i
  return;
}




void computeLinkOrderedOuterProduct(void *src, void *dst, QudaPrecision precision, int gauge_order)
{
  if(precision == QUDA_SINGLE_PRECISION){
    computeLinkOrderedOuterProduct((fhalf_wilson_vector*)src,(fsu3_matrix*)dst, gauge_order);
  }else{
    computeLinkOrderedOuterProduct((dhalf_wilson_vector*)src,(dsu3_matrix*)dst, gauge_order);
  }
  return;
}

void computeLinkOrderedOuterProduct(void *src, void *dst, QudaPrecision precision, size_t nhops, int gauge_order)
{
  if(precision == QUDA_SINGLE_PRECISION){
    computeLinkOrderedOuterProduct((fhalf_wilson_vector*)src,(fsu3_matrix*)dst, nhops, gauge_order);
  }else{
    computeLinkOrderedOuterProduct((dhalf_wilson_vector*)src,(dsu3_matrix*)dst, nhops, gauge_order);
  }
  return;
}



template<typename half_wilson_vector, typename su3_matrix> 
static void shiftedOuterProduct(half_wilson_vector *src, su3_matrix* dest){
  for(int dir=0; dir<4; dir++){
    shifted_outer_prod(src, &dest[dir*V], OPP_DIR(dir));
  }
}

template<typename half_wilson_vector, typename su3_matrix> 
static void forwardShiftedOuterProduct(half_wilson_vector *src, su3_matrix* dest){
  for(int dir=0; dir<4; dir++){
    forward_shifted_outer_prod(src, &dest[dir*V], dir);
  }
}


void computeHisqOuterProduct(void* src, void* dest, QudaPrecision precision){
  if(precision == QUDA_SINGLE_PRECISION){
    forwardShiftedOuterProduct((fhalf_wilson_vector*)src,(fsu3_matrix*)dest);	
  }else{
    forwardShiftedOuterProduct((dhalf_wilson_vector*)src,(dsu3_matrix*)dest);	
  }
}


// Note that the hisq routines do not involve half-wilson vectors, 
// but instead require colour matrix shifts


template<typename su3_matrix> 
static void 
u_shift_mat(su3_matrix *src, su3_matrix *dest, int dir, su3_matrix* sitelink)
{
  int i;
  int dx[4];
  dx[3]=dx[2]=dx[1]=dx[0]=0;

  if(GOES_FORWARDS(dir)){
    dx[dir]=1;
    for(i=0; i<V; i++){
      int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
      su3_matrix* mat = src+nbr_idx; // No need for a factor of 4 here, the colour matrices do not have a Lorentz index
      su3_matrix* link = sitelink + i*4 + dir;
      matrix_mult_nn(link, mat, &dest[i]);	
    }	
  }else{
    dx[OPP_DIR(dir)]=-1;
    for(i=0; i<V; i++){
      int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
      su3_matrix* mat = src+nbr_idx; // No need for a factor of 4 here, the colour matrices do not have a Lorentz index
      su3_matrix* link = sitelink + nbr_idx*4 + OPP_DIR(dir);
      matrix_mult_an(link, mat, &dest[i]);
    }
  }
  return;
}







template <typename half_wilson_vector,
	  typename anti_hermitmat, typename Real>
static void 
add_3f_force_to_mom(half_wilson_vector *back, half_wilson_vector *forw, 
		    int dir, Real coeff[2], anti_hermitmat* momentum) 
{
    Real my_coeff[2] ;
    Real tmp_coeff[2] ;
    int mydir;
    int i;
    
    if(GOES_BACKWARDS(dir)){
	mydir = OPP_DIR(dir);
	my_coeff[0] = -coeff[0]; 
	my_coeff[1] = -coeff[1];
    }else{ 
	mydir = dir; 
	my_coeff[0] = coeff[0]; 
	my_coeff[1] = coeff[1]; 
    }
    
    for(i=0;i < V;i++){
	if (i < Vh){
	    tmp_coeff[0] = my_coeff[0];
	    tmp_coeff[1] = my_coeff[1];
	}else{
	    tmp_coeff[0] = -my_coeff[0];
	    tmp_coeff[1] = -my_coeff[1] ;	
	}
	
	if (sizeof(Real) == sizeof(float)){
	    fsu3_matrix tmat;
	    fsu3_matrix mom_matrix;
	    anti_hermitmat* mom = momentum+ 4* i + mydir;
	    uncompress_anti_hermitian(mom, &mom_matrix);
	    su3_projector( &back[i].h[0], &forw[i].h[0], &tmat);
	    scalar_mult_add_su3_matrix(&mom_matrix, &tmat,  tmp_coeff[0], &mom_matrix );
	    make_anti_hermitian(&mom_matrix, mom);
	}else{
	    dsu3_matrix tmat;
	    dsu3_matrix mom_matrix;
	    anti_hermitmat* mom = momentum+ 4* i + mydir;
	    uncompress_anti_hermitian(mom, &mom_matrix);
	    su3_projector( &back[i].h[0], &forw[i].h[0], &tmat);
	    scalar_mult_add_su3_matrix(&mom_matrix, &tmat,  tmp_coeff[0], &mom_matrix );
	    make_anti_hermitian(&mom_matrix, mom);
	}
    }    
}


template<typename Real, typename half_wilson_vector, typename anti_hermitmat>
  static void 
side_link_3f_force(int mu, int nu, Real coeff[2], half_wilson_vector *Path   , 
    half_wilson_vector *Path_nu, half_wilson_vector *Path_mu, 
    half_wilson_vector *Path_numu, anti_hermitmat* mom) 
{
  Real m_coeff[2] ;

  m_coeff[0] = -coeff[0] ;
  m_coeff[1] = -coeff[1] ;

  if(GOES_FORWARDS(mu)){
    if(GOES_FORWARDS(nu)){
      add_3f_force_to_mom(Path_numu, Path, mu, coeff, mom) ;
    }else{
      add_3f_force_to_mom(Path,Path_numu,OPP_DIR(mu),m_coeff, mom);
    }
  }
  else{
    if(GOES_FORWARDS(nu))
      add_3f_force_to_mom(Path_nu, Path_mu, mu, m_coeff, mom);
    else
      add_3f_force_to_mom(Path_mu, Path_nu, OPP_DIR(mu), coeff, mom) ;
  }
}


template<typename Real, typename su3_matrix, typename anti_hermitmat>
  static void
add_force_to_momentum(su3_matrix *back, su3_matrix *forw,
    int dir, Real coeff, anti_hermitmat* momentum)
{
  Real my_coeff;
  Real tmp_coeff;
  int mydir;
  int i;

  if(GOES_BACKWARDS(dir)){
    mydir = OPP_DIR(dir);
    my_coeff = -coeff;
  }else{
    mydir = dir;
    my_coeff = coeff;	
  }


  for(i=0; i<V; i++){
    if(i<Vh){ tmp_coeff = my_coeff; }
    else{ tmp_coeff = -my_coeff; }


    su3_matrix tmat;
    su3_matrix mom_matrix;
    anti_hermitmat* mom = momentum + 4*i + mydir;
    uncompress_anti_hermitian(mom, &mom_matrix);
    matrix_mult_na(&back[i], &forw[i], &tmat);
    scalar_mult_add_su3_matrix(&mom_matrix, &tmat, tmp_coeff, &mom_matrix);  

    make_anti_hermitian(&mom_matrix, mom);	
  }
  return;
}


template<typename Real, typename su3_matrix, typename anti_hermitmat>
static void 
side_link_force(int mu, int nu, Real coeff, su3_matrix *Path,
		su3_matrix *Path_nu, su3_matrix *Path_mu,
		su3_matrix *Path_numu, anti_hermitmat* mom)
{
  Real m_coeff;
  m_coeff = -coeff;

  if(GOES_FORWARDS(mu)){
    if(GOES_FORWARDS(nu)){
      add_force_to_momentum(Path_numu, Path, mu, coeff, mom); 
      // In the example below:
      // add_force_to_momentum(P7rho, Qnumu, rho, coeff, mom)
    }else{
      add_force_to_momentum(Path, Path_numu, OPP_DIR(mu), m_coeff, mom);	
      // add_force_to_momentum(Qnumu, P7rho, -Qnumu, rho, m_coeff, mom)
    }	
  }
  else { // if (GOES_BACKWARDS(mu))
    if(GOES_FORWARDS(nu)){
      add_force_to_momentum(Path_nu, Path_mu, mu, m_coeff, mom);
      // add_force_to_momentum(P7, Qrhonumu, rho, m_coeff, mom) 
    }else{
      add_force_to_momentum(Path_mu, Path_nu, OPP_DIR(mu), coeff, mom);	
      // add_force_to_momentum(Qrhonumu, P7, rho, coeff, mom)
    }
  }
  return; 
}




#define Pmu          tempmat[0]
#define Pnumu        tempmat[1]
#define Prhonumu     tempmat[2]
#define P7	     tempmat[3]
#define P7rho	     tempmat[4]
#define P5           tempmat[5]
#define P3           tempmat[6]
#define P5nu	     tempmat[3]
#define P3mu         tempmat[3]
#define Popmu        tempmat[4]
#define Pmumumu      tempmat[4]

#define Qmu          tempmat[7]
#define Qnumu        tempmat[8]
#define Qrhonumu     tempmat[2] // same as Prhonumu





template<typename su3_matrix>
static void set_identity_matrix(su3_matrix* mat)
{
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      mat->e[i][j].real=0;
      mat->e[i][j].imag=0;		
    }
    mat->e[i][i].real=1;		
  }
} 


template<typename su3_matrix> 
static void set_identity(su3_matrix* matrices, int num_dirs){
  for(int i=0; i<V*num_dirs; i++){
    set_identity_matrix(&matrices[i]);	
  }
}


template <typename Real, typename su3_matrix, typename anti_hermitmat>
void do_color_matrix_hisq_force_reference(Real eps, Real weight, 
			   su3_matrix* temp_xx, Real* act_path_coeff,
			   su3_matrix* sitelink, anti_hermitmat* mom)
{
  int i;
  int mu, nu, rho, sig;
  Real coeff;
  Real OneLink, Lepage, FiveSt, ThreeSt, SevenSt;
//  Real Naik;
  Real mLepage, mFiveSt, mThreeSt, mSevenSt;    

  su3_matrix* tempmat[9];
  int sites_on_node = V;
  //int DirectLinks[8] ;
 
  Real ferm_epsilon;
  ferm_epsilon = 2.0*weight*eps;
  OneLink = act_path_coeff[0]*ferm_epsilon ; 
//  Naik    = act_path_coeff[1]*ferm_epsilon ; mNaik    = -Naik;
  ThreeSt = act_path_coeff[2]*ferm_epsilon ; mThreeSt = -ThreeSt;
  FiveSt  = act_path_coeff[3]*ferm_epsilon ; mFiveSt  = -FiveSt;
  SevenSt = act_path_coeff[4]*ferm_epsilon ; mSevenSt = -SevenSt;
  Lepage  = act_path_coeff[5]*ferm_epsilon ; mLepage  = -Lepage;

//  for(mu=0;mu<8;mu++){
//    DirectLinks[mu] = 0 ;
//  }

  for(mu=0; mu<9; mu++){	
    tempmat[mu] = (su3_matrix *)malloc( sites_on_node*sizeof(su3_matrix) );  
  }


  su3_matrix* id;
  id = (su3_matrix *)malloc(sites_on_node*sizeof(su3_matrix) ); 
  //su3_matrix* id4;
  //id4 = (su3_matrix *)malloc(sites_on_node*4*sizeof(su3_matrix) ); 

 // su3_matrix* temp_mat;
//  temp_mat = (su3_matrix *)malloc(sites_on_node*sizeof(su3_matrix) ); 


  //  su3_matrix* temp_xx;
  //  temp_xx = (su3_matrix *)malloc(sites_on_node*8*sizeof(su3_matrix) );
  //  shiftedOuterProduct(temp_x, temp_xx);

  // initialise id so that it is the identity matrix on each lattice site
  set_identity(id,1);

  printf("Calling modified hisq\n");
  return;


  for(sig=0; sig < 8; sig++){
    // One-link term - don't have the savings here that we get when working with the 
    // half-wilson vectors
    if(GOES_FORWARDS(sig)){
      u_shift_mat(&temp_xx[OPP_DIR(sig)*V], Pmu, sig, sitelink);
      add_force_to_momentum(Pmu, id, sig, OneLink, mom); // I could optimise functions which 
      // involve id
    }



    for(mu = 0; mu < 8; mu++){
      if ( (mu == sig) || (mu == OPP_DIR(sig))){
        continue;
      }
      //     3 link path 
      //	 sig
      //    A  _______
      //      |       |
      //  mu /|\     \|/
      //      |       |
      //
      //
      u_shift_mat(&temp_xx[OPP_DIR(sig)*V], Pmu, OPP_DIR(mu), sitelink); // temp_xx[sig] stores |X(x)><X(x-sig)|
      u_shift_mat(id, Qmu, OPP_DIR(mu), sitelink); // This returns the path less the outer-product of quark fields at the end 
                                                   // Qmu = U[mu]	


      u_shift_mat(Pmu, P3, sig, sitelink); // P3 is U[sig](X)U[-mu](X+sig) temp_xx
      if (GOES_FORWARDS(sig)){
        // add contribution from middle link
        add_force_to_momentum(P3, Qmu, sig, mThreeSt, mom); 
        // matrix_mult_na(P3[x],Qmu[x],tmp);
        // mom[sig][x] += mThreeSt*tmp; 
      }
      for(nu=0; nu < 8; nu++){
        if (nu == sig || nu == OPP_DIR(sig)
            || nu == mu || nu == OPP_DIR(mu)){
          continue;
        }

        // 5 link path
        //
        //        sig
        //    A ________
        //     |        |
        //    /|\      \|/
        //     |        |
        //      \	 \
        //	 \        \

        u_shift_mat(Pmu, Pnumu, OPP_DIR(nu), sitelink);
        u_shift_mat(Qmu, Qnumu, OPP_DIR(nu), sitelink);

        u_shift_mat(Pnumu, P5, sig, sitelink);
        if (GOES_FORWARDS(sig)){
          add_force_to_momentum(P5, Qnumu, sig, FiveSt, mom);
        } // seems to do what I think it should
        for(rho =0; rho < 8; rho++){
          if (rho == sig || rho == OPP_DIR(sig)
              || rho == mu || rho == OPP_DIR(mu)
              || rho == nu || rho == OPP_DIR(nu)){
            continue;
          }

          //7 link path
          u_shift_mat(Pnumu, Prhonumu, OPP_DIR(rho), sitelink);
          u_shift_mat(Prhonumu, P7, sig, sitelink);

          // Prhonumu = tempmat[2] is not needed again
          // => store Qrhonumu in the same memory
          u_shift_mat(Qnumu, Qrhonumu, OPP_DIR(rho), sitelink);


          if(GOES_FORWARDS(sig)){
            add_force_to_momentum(P7, Qrhonumu, sig, mSevenSt, mom) ;	
          }
             u_shift_mat(P7, P7rho, rho, sitelink);
             side_link_force(rho, sig, SevenSt, Qnumu, P7, Qrhonumu, P7rho, mom);		    
             if(FiveSt != 0)coeff = SevenSt/FiveSt ; else coeff = 0;
             for(i=0; i<V; i++){
             scalar_mult_add_su3_matrix(&P5[i], &P7rho[i], coeff, &P5[i]);
             } // end loop over volume
        } // end loop over rho	


           u_shift_mat(P5, P5nu, nu, sitelink);	
           side_link_force(nu,sig,mFiveSt,Qmu,P5,Qnumu,P5nu,mom); // I believe this should do what I want it to
        // check this!
        if(ThreeSt != 0)coeff	= FiveSt/ThreeSt; else coeff = 0;

        for(i=0; i<V; i++){
        scalar_mult_add_su3_matrix(&P3[i], &P5nu[i], coeff, &P3[i]);
        } // end loop over volume
      } // end loop over nu

      // Lepage term 
      u_shift_mat(Pmu, Pnumu, OPP_DIR(mu), sitelink);
      u_shift_mat(Qmu, Qnumu, OPP_DIR(mu), sitelink);

      u_shift_mat(Pnumu, P5, sig, sitelink);
      if(GOES_FORWARDS(sig)){
      add_force_to_momentum(P5, Qnumu, sig, Lepage, mom); 
      }

      u_shift_mat(P5, P5nu, mu, sitelink);
      side_link_force(mu, sig, mLepage, Qmu, P5, Qnumu, P5nu, mom);


      if(ThreeSt != 0)coeff = Lepage/ThreeSt; else coeff = 0;

      for(i=0; i<V; i++){
      scalar_mult_add_su3_matrix(&P3[i], &P5nu[i], coeff, &P3[i]);
      }

      if(GOES_FORWARDS(mu)){
      u_shift_mat(P3, P3mu, mu, sitelink);
      }
      side_link_force(mu, sig, ThreeSt, id, P3, Qmu, P3mu, mom);
    } // end loop over mu
  } // end loop over sig
} // modified hisq_force_reference






// This version of the test routine uses 
// half-wilson vectors instead of color matrices.
template <typename Real, typename su3_matrix, typename anti_hermitmat, typename half_wilson_vector>
void do_halfwilson_hisq_force_reference(Real eps, Real weight, 
			   half_wilson_vector* temp_x, Real* act_path_coeff,
			   su3_matrix* sitelink, anti_hermitmat* mom)
{
  int i;
  int mu, nu, rho, sig;
  Real coeff;
  Real OneLink, Lepage, FiveSt, ThreeSt, SevenSt;
  //Real Naik;
  Real mLepage, mFiveSt, mThreeSt, mSevenSt;    

  su3_matrix* tempmat[9];
  int sites_on_node = V;
//  int DirectLinks[8] ;
 
  Real ferm_epsilon;
  ferm_epsilon = 2.0*weight*eps;
  OneLink = act_path_coeff[0]*ferm_epsilon ; 
//  Naik    = act_path_coeff[1]*ferm_epsilon ; 
  ThreeSt = act_path_coeff[2]*ferm_epsilon ; mThreeSt = -ThreeSt;
  FiveSt  = act_path_coeff[3]*ferm_epsilon ; mFiveSt  = -FiveSt;
  SevenSt = act_path_coeff[4]*ferm_epsilon ; mSevenSt = -SevenSt;
  Lepage  = act_path_coeff[5]*ferm_epsilon ; mLepage  = -Lepage;
   
 
//  for(mu=0;mu<8;mu++){
//    DirectLinks[mu] = 0 ;
//  }

  for(mu=0; mu<9; mu++){	
    tempmat[mu] = (su3_matrix *)malloc( sites_on_node*sizeof(su3_matrix) );  
  }


  su3_matrix* id;
  id = (su3_matrix *)malloc(sites_on_node*sizeof(su3_matrix) ); 
  su3_matrix* id4;
  id4 = (su3_matrix *)malloc(sites_on_node*4*sizeof(su3_matrix) ); 

  su3_matrix* temp_mat;
  temp_mat = (su3_matrix *)malloc(sites_on_node*sizeof(su3_matrix) ); 

  // initialise id so that it is the identity matrix on each lattice site
  set_identity(id,1);

  printf("Calling hisq reference routine\n");
  for(sig=0; sig < 8; sig++){
    shifted_outer_prod(temp_x, temp_mat, OPP_DIR(sig));

    // One-link term - don't have the savings here that we get when working with the 
    // half-wilson vectors
    if(GOES_FORWARDS(sig)){
      u_shift_mat(temp_mat, Pmu, sig, sitelink);
       add_force_to_momentum(Pmu, id, sig, OneLink, mom); 
    }

    for(mu = 0; mu < 8; mu++){
      if ( (mu == sig) || (mu == OPP_DIR(sig))){
		continue;
      }
      // 3 link path 
      //	 sig
      //    A  _______
      //      |       |
      //  mu /|\     \|/
      //      |       |
      //
      //
      u_shift_mat(temp_mat, Pmu, OPP_DIR(mu), sitelink); // temp_xx[sig] stores |X(x)><X(x-sig)|
      u_shift_mat(id, Qmu, OPP_DIR(mu), sitelink);       // This returns the path less the outer-product of quark fields at the end 
						         // Qmu = U[mu]	
	

      u_shift_mat(Pmu, P3, sig, sitelink); // P3 is U[sig](X)U[-mu](X+sig) temp_xx
      if (GOES_FORWARDS(sig)){
	  // add contribution from middle link
        add_force_to_momentum(P3, Qmu, sig, mThreeSt, mom); // matrix_mult_na(P3[x],Qmu[x],tmp);
							    // mom[sig][x] += mThreeSt*tmp; 
      }
      for(nu=0; nu < 8; nu++){
        if (nu == sig || nu == OPP_DIR(sig)
	 || nu == mu || nu == OPP_DIR(mu)){
	 continue;
        }
	
	// 5 link path
        //
	//        sig
	//    A ________
	//     |        |
	//    /|\      \|/
	//     |        |
	//      \        \
	//	 \        \

	u_shift_mat(Pmu, Pnumu, OPP_DIR(nu), sitelink);
	u_shift_mat(Qmu, Qnumu, OPP_DIR(nu), sitelink);

        u_shift_mat(Pnumu, P5, sig, sitelink);
        if (GOES_FORWARDS(sig)){
	  add_force_to_momentum(P5, Qnumu, sig, FiveSt, mom);
        } // seems to do what I think it should

	for(rho =0; rho < 8; rho++){
	  if (rho == sig || rho == OPP_DIR(sig)
	    || rho == mu || rho == OPP_DIR(mu)
	    || rho == nu || rho == OPP_DIR(nu)){
	       continue;
	   }
		    
           //7 link path
	  u_shift_mat(Pnumu, Prhonumu, OPP_DIR(rho), sitelink);
	  u_shift_mat(Prhonumu, P7, sig, sitelink);
	  
	  // Prhonumu = tempmat[2] is not needed again
	  // => store Qrhonumu in the same memory
	  u_shift_mat(Qnumu, Qrhonumu, OPP_DIR(rho), sitelink);
	

          if(GOES_FORWARDS(sig)){
	    add_force_to_momentum(P7, Qrhonumu, sig, mSevenSt, mom) ;	
	  }

	  u_shift_mat(P7, P7rho, rho, sitelink);
	  side_link_force(rho, sig, SevenSt, Qnumu, P7, Qrhonumu, P7rho, mom);		    
	  if(FiveSt != 0)coeff = SevenSt/FiveSt ; else coeff = 0;
	  for(i=0; i<V; i++){
	    scalar_mult_add_su3_matrix(&P5[i], &P7rho[i], coeff, &P5[i]);
	  } // end loop over volume
	} // end loop over rho	


        u_shift_mat(P5, P5nu, nu, sitelink);	
	side_link_force(nu,sig,mFiveSt,Qmu,P5,Qnumu,P5nu,mom); // I believe this should do what I want it to
							       // check this!
	if(ThreeSt != 0)coeff	= FiveSt/ThreeSt; else coeff = 0;
	
        for(i=0; i<V; i++){
	  scalar_mult_add_su3_matrix(&P3[i], &P5nu[i], coeff, &P3[i]);
	} // end loop over volume
      } // end loop over nu

      // Lepage term 
      u_shift_mat(Pmu, Pnumu, OPP_DIR(mu), sitelink);
      u_shift_mat(Qmu, Qnumu, OPP_DIR(mu), sitelink);
	
      u_shift_mat(Pnumu, P5, sig, sitelink);
      if(GOES_FORWARDS(sig)){
        add_force_to_momentum(P5, Qnumu, sig, Lepage, mom); 
      }

      u_shift_mat(P5, P5nu, mu, sitelink);
      side_link_force(mu, sig, mLepage, Qmu, P5, Qnumu, P5nu, mom);


      if(ThreeSt != 0)coeff = Lepage/ThreeSt; else coeff = 0;

      for(i=0; i<V; i++){
	scalar_mult_add_su3_matrix(&P3[i], &P5nu[i], coeff, &P3[i]);
      }
   
      if(GOES_FORWARDS(mu)){
	u_shift_mat(P3, P3mu, mu, sitelink);
      }
      side_link_force(mu, sig, ThreeSt, id, P3, Qmu, P3mu, mom);
    } // end loop over mu
  } // end loop over sig

  for(mu=0; mu<9; mu++){	
    free(tempmat[mu]);
  }
  
  free(id);
  free(id4);
  free(temp_mat);
}

#undef Pmu
#undef Pnumu
#undef Prhonumu
#undef P3
#undef P3mu
#undef P5
#undef P5nu
#undef P7
#undef P7rho
#undef P7rhonu

#undef Popmu
#undef Pmumumu

#undef Qmu
#undef Qnumu
#undef Qrhonumu



static
void set_identity(fsu3_matrix *sitelink){
  int tot = V*4;
  for(int i=0; i<tot; i++){ // loop over sites and directions
   for(int a=0; a<3; a++){
     for(int b=0; b<3; b++){
        sitelink[i].e[a][b].real = sitelink[i].e[a][b].imag = 0.;
     }
   }
   for(int a=0; a<3; a++){ // set the diagonal elements to unity
     sitelink[i].e[a][a].real = 1.;
   }
  }
  return;
}




void halfwilson_hisq_force_reference(float eps, float weight, 
			  void* act_path_coeff, void* temp_x,
			  void* sitelink, void* mom)
{
 do_halfwilson_hisq_force_reference((float)eps, (float)weight,
			  (fhalf_wilson_vector*) temp_x, (float*)act_path_coeff,
			  (fsu3_matrix*)sitelink, (fanti_hermitmat*)mom);
 return;
}



void halfwilson_hisq_force_reference(double eps, double weight, 
			  void* act_path_coeff, void* temp_x,
			  void* sitelink, void* mom)
{
 do_halfwilson_hisq_force_reference((double)eps, (double)weight,
			  (dhalf_wilson_vector*) temp_x, (double*)act_path_coeff,
			  (dsu3_matrix*)sitelink, (danti_hermitmat*)mom);
 return;
}



void color_matrix_hisq_force_reference(float eps, float weight, 
			  void* act_path_coeff, void* temp_xx,
			  void* sitelink, void* mom)
{
 do_color_matrix_hisq_force_reference((float)eps, (float)weight,
			  (fsu3_matrix*) temp_xx, (float*)act_path_coeff,
			  (fsu3_matrix*)sitelink, (fanti_hermitmat*)mom);
  return;
}




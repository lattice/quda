#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "quda.h"
#include "test_util.h"
#include "misc.h"
#include "fermion_force_reference.h"

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
static void  
su3_adjoint( su3_matrix *a, su3_matrix *b )
{
    int i,j;
    for(i=0;i<3;i++)for(j=0;j<3;j++){
	    CONJG( a->e[j][i], b->e[i][j] );
	}
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
	c->c[i] = x;
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
	    su3_projector( &back[i].h[1], &forw[i].h[1], &tmat);
	    scalar_mult_add_su3_matrix(&mom_matrix, &tmat,  tmp_coeff[1], &mom_matrix );	
	    make_anti_hermitian(&mom_matrix, mom);
	}else{
	    dsu3_matrix tmat;
	    dsu3_matrix mom_matrix;
	    anti_hermitmat* mom = momentum+ 4* i + mydir;
	    uncompress_anti_hermitian(mom, &mom_matrix);
	    su3_projector( &back[i].h[0], &forw[i].h[0], &tmat);
	    scalar_mult_add_su3_matrix(&mom_matrix, &tmat,  tmp_coeff[0], &mom_matrix );
	    su3_projector( &back[i].h[1], &forw[i].h[1], &tmat);
	    scalar_mult_add_su3_matrix(&mom_matrix, &tmat,  tmp_coeff[1], &mom_matrix );	
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



#define Pmu          tempvec[0] 
#define Pnumu        tempvec[1]
#define Prhonumu     tempvec[2]
#define P7           tempvec[3]
#define P7rho        tempvec[4]              
#define P7rhonu      tempvec[5]
#define P5           tempvec[6]
#define P3           tempvec[7]
#define P5nu         tempvec[3]
#define P3mu         tempvec[3]
#define Popmu        tempvec[4]
#define Pmumumu      tempvec[4]



template <typename Real, typename half_wilson_vector, typename anti_hermitmat, 
	  typename su3_matrix>
static void
do_fermion_force_reference(Real eps, Real weight1, Real weight2,
			   half_wilson_vector* temp_x, Real* act_path_coeff, 
			   su3_matrix* sitelink, anti_hermitmat* mom)    
{
    int i;
    int mu, nu, rho, sig;
    Real coeff[2];
    Real OneLink[2], Lepage[2], Naik[2], FiveSt[2], ThreeSt[2], SevenSt[2] ;
    Real mNaik[2], mLepage[2], mFiveSt[2], mThreeSt[2], mSevenSt[2];    
    half_wilson_vector *tempvec[8] ;
    int sites_on_node = V;
    int DirectLinks[8] ;
 
    Real ferm_epsilon;
    ferm_epsilon = 2.0*weight1*eps;
    OneLink[0] = act_path_coeff[0]*ferm_epsilon ; 
    Naik[0]    = act_path_coeff[1]*ferm_epsilon ; mNaik[0]    = -Naik[0];
    ThreeSt[0] = act_path_coeff[2]*ferm_epsilon ; mThreeSt[0] = -ThreeSt[0];
    FiveSt[0]  = act_path_coeff[3]*ferm_epsilon ; mFiveSt[0]  = -FiveSt[0];
    SevenSt[0] = act_path_coeff[4]*ferm_epsilon ; mSevenSt[0] = -SevenSt[0];
    Lepage[0]  = act_path_coeff[5]*ferm_epsilon ; mLepage[0]  = -Lepage[0];
    
    ferm_epsilon = 2.0*weight2*eps;
    OneLink[1] = act_path_coeff[0]*ferm_epsilon ; 
    Naik[1]    = act_path_coeff[1]*ferm_epsilon ; mNaik[1]    = -Naik[1];
    ThreeSt[1] = act_path_coeff[2]*ferm_epsilon ; mThreeSt[1] = -ThreeSt[1];
    FiveSt[1]  = act_path_coeff[3]*ferm_epsilon ; mFiveSt[1]  = -FiveSt[1];
    SevenSt[1] = act_path_coeff[4]*ferm_epsilon ; mSevenSt[1] = -SevenSt[1];
    Lepage[1]  = act_path_coeff[5]*ferm_epsilon ; mLepage[1]  = -Lepage[1];
    
    for(mu=0;mu<8;mu++){
	DirectLinks[mu] = 0 ;
    }
    
    for(mu=0;mu<8;mu++){
	tempvec[mu] = (half_wilson_vector *)malloc( sites_on_node*sizeof(half_wilson_vector) );  
    }
    
    for(sig=0; sig < 8; sig++){
	for(mu = 0; mu < 8; mu++){
	    if ( (mu == sig) || (mu == OPP_DIR(sig))){
		continue;
	    }

	    // 3 link path 
	    u_shift_hw(temp_x, Pmu, OPP_DIR(mu), sitelink);
	    u_shift_hw(Pmu, P3, sig, sitelink);
	    if (GOES_FORWARDS(sig)){
		add_3f_force_to_mom(P3, Pmu, sig, mThreeSt, mom);
	    }	    
	    for(nu=0; nu < 8; nu++){
		if (nu == sig || nu == OPP_DIR(sig)
		    || nu == mu || nu == OPP_DIR(mu)){
		    continue;
		}
		
		// 5 link path
		u_shift_hw(Pmu, Pnumu, OPP_DIR(nu), sitelink);
		u_shift_hw(Pnumu, P5, sig, sitelink);
		if (GOES_FORWARDS(sig)){
		    add_3f_force_to_mom(P5, Pnumu, sig, FiveSt, mom);
		}


		for(rho =0; rho < 8; rho++){
		    if (rho == sig || rho == OPP_DIR(sig)
			|| rho == mu || rho == OPP_DIR(mu)
			|| rho == nu || rho == OPP_DIR(nu)){
			continue;
		    }
		    
		    //7 link path
		    u_shift_hw(Pnumu, Prhonumu, OPP_DIR(rho), sitelink);
		    u_shift_hw(Prhonumu, P7,sig, sitelink);
		    if(GOES_FORWARDS(sig)){
			add_3f_force_to_mom(P7, Prhonumu, sig, mSevenSt, mom) ;	
		    }
		    u_shift_hw(P7, P7rho, rho, sitelink);
		    side_link_3f_force(rho,sig,SevenSt, Pnumu, P7, Prhonumu, P7rho, mom);		    
		    if(FiveSt[0] != 0)coeff[0] = SevenSt[0]/FiveSt[0] ; else coeff[0] = 0;
		    if(FiveSt[1] != 0)coeff[1] = SevenSt[1]/FiveSt[1] ; else coeff[1] = 0;		    
		    for(i=0;i < V; i++){
			scalar_mult_add_su3_vector(&(P5[i].h[0]),&(P7rho[i].h[0]), coeff[0],
						   &(P5[i].h[0]));
			scalar_mult_add_su3_vector(&(P5[i].h[1]),&(P7rho[i].h[1]), coeff[1],
						   &(P5[i].h[1]));
		    }		    

		}//rho	

		u_shift_hw(P5,P5nu, nu, sitelink);
		side_link_3f_force(nu,sig,mFiveSt,Pmu,P5, Pnumu,P5nu, mom);
		if(ThreeSt[0] != 0)coeff[0] = FiveSt[0]/ThreeSt[0] ; else coeff[0] = 0;
		if(ThreeSt[1] != 0)coeff[1] = FiveSt[1]/ThreeSt[1] ; else coeff[1] = 0;
		for(i=0; i < V; i++){
		    scalar_mult_add_su3_vector(&(P3[i].h[0]),&(P5nu[i].h[0]),coeff[0],&(P3[i].h[0]));
		    scalar_mult_add_su3_vector(&(P3[i].h[1]),&(P5nu[i].h[1]),coeff[1],&(P3[i].h[1]));
		}

	    }//nu

	    //Lepage term
	    u_shift_hw(Pmu, Pnumu, OPP_DIR(mu), sitelink);
	    u_shift_hw(Pnumu, P5, sig, sitelink);
	    if(GOES_FORWARDS(sig)){
		add_3f_force_to_mom(P5, Pnumu, sig, Lepage, mom);
	    }
	    
	    u_shift_hw(P5,P5nu, mu, sitelink);
	    side_link_3f_force(mu, sig, mLepage, Pmu, P5, Pnumu, P5nu, mom);
	    if(ThreeSt[0] != 0) coeff[0] = Lepage[0]/ThreeSt[0] ; else coeff[0] = 0;
	    if(ThreeSt[1] != 0) coeff[1] = Lepage[1]/ThreeSt[1] ; else coeff[1] = 0;
	    for(i=0;i < V;i++){
		scalar_mult_add_su3_vector(&(P3[i].h[0]),&(P5nu[i].h[0]),coeff[0],&(P3[i].h[0]));
		scalar_mult_add_su3_vector(&(P3[i].h[1]),&(P5nu[i].h[1]),coeff[1],&(P3[i].h[1]));
	    }
	    
	    if(GOES_FORWARDS(mu)) {
		u_shift_hw(P3,P3mu, mu, sitelink);
	    }
	    side_link_3f_force(mu, sig, ThreeSt, temp_x, P3, Pmu, P3mu, mom);   
	    
	    //One link term and Naik term
	    if( (!DirectLinks[mu]) ){
		DirectLinks[mu]=1 ;
		if(GOES_BACKWARDS(mu)){
		    //one link term
		    add_3f_force_to_mom(Pmu, temp_x, OPP_DIR(mu), OneLink, mom);
		    u_shift_hw(temp_x, Popmu, mu, sitelink);
		    add_3f_force_to_mom(Pnumu, Popmu, OPP_DIR(mu), mNaik, mom);
		    u_shift_hw(Pnumu, Pmumumu, OPP_DIR(mu), sitelink);
		    add_3f_force_to_mom(Pmumumu, temp_x, OPP_DIR(mu), Naik, mom);
		}else{
		    u_shift_hw(temp_x, Popmu, mu, sitelink);
		    add_3f_force_to_mom(Popmu, Pnumu, mu, Naik, mom);
		}		
	    }//if
	}//mu
    }//sig

    for(mu=0;mu<8;mu++){
      free(tempvec[mu]);
    }
    
}

#undef Pmu
#undef Pnumu
#undef Prhonumu
#undef P7
#undef P7rho
#undef P7rhonu
#undef P5
#undef P3
#undef P5nu
#undef P3mu
#undef Popmu
#undef Pmumumu


void
fermion_force_reference(float eps, float weight1, float weight2,
			void* act_path_coeff, void* temp_x, void* sitelink, void* mom)    
{
    do_fermion_force_reference((float)eps, (float)weight1, (float)weight2, 
			       (fhalf_wilson_vector*) temp_x, (float*)act_path_coeff,
			       (fsu3_matrix*)sitelink, (fanti_hermitmat*) mom);   
    
}

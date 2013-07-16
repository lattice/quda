#include <complex>
#define  cdouble complex<double>
#include <iostream>
// some params
#include "wilson_dslash_reference.h"
#include "misc.h"
#include "face_quda.h"
#include "test_util.h"
#include <string>
using namespace std;

int read_d(char *src,int *data);
int read_lf(char *src,double *data);
int read_c(char *src,char *data);

// A class of 3*3 complex matrix
class ColorMatrix{
private:
  cdouble u[3][3];

public:
  ColorMatrix(float* gauge){
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      u[i][j] = cdouble(*(gauge + i*6 + j*2 + 0), *(gauge + i*6 + j*2 + 1));
    }
  }

  ColorMatrix(double* gauge = NULL){
    if(gauge == NULL)
      for(int i=0; i<3; i++)
      for(int j=0; j<3; j++){
      u[i][j] = cdouble(0);
      }
    else
      for(int i=0; i<3; i++)
      for(int j=0; j<3; j++){
      u[i][j] = cdouble(*(gauge + i*6 + j*2 + 0), *(gauge+i*6+j*2+1));
      }
  }

  friend ColorMatrix operator * (ColorMatrix A, ColorMatrix B){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++){
      C.u[i][k] += A.u[i][j] * B.u[j][k];
    }
    return C;
  }

  friend ColorMatrix operator + (ColorMatrix A, ColorMatrix B){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = A.u[i][j] + B.u[i][j];
    }
    return C;
  }

  friend ColorMatrix operator - (ColorMatrix A, ColorMatrix B){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = A.u[i][j] - B.u[i][j];
    }
    return C;
  }

  friend ColorMatrix operator - (ColorMatrix A, cdouble B){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = A.u[i][j];
      if(i==j) C.u[i][j] -= B;
    }
    return C;
  }

  friend ColorMatrix operator + (ColorMatrix A, cdouble B){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = A.u[i][j];
      if(i==j) C.u[i][j] += B;
    }
    return C;
  }

  friend ColorMatrix operator + (cdouble B, ColorMatrix A){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = A.u[i][j];
      if(i==j) C.u[i][j] += B;
    }
    return C;
  }

  friend ColorMatrix operator * (cdouble A, ColorMatrix B){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = A * B.u[i][j];
    }
    return C;
  }

  ColorMatrix dagger(){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = conj(u[j][i]);
    }
    return C;
  }

  ColorMatrix Exp(){
    ColorMatrix C;
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      C.u[i][j] = exp(u[i][j]);
    }
    return C;
  }

  cdouble trace(){
    cdouble temp=0;
    for(int i=0; i<3; i++)
      temp += u[i][i];
    return temp;	
  }

  cdouble GetU(int i, int j){
    return u[i][j];
  }
};

// add by ybyang, Apr 29 class of input paramers
class global_param{
public:
  int ioflag;
  int nx,ny,nz,nt;
  double kappa;
  char ds_type[50]; 
  double mu;
  double csw;
  int if_anisotropic; 
  double ani_vlight;
  double ani_us;
  double ani_xsi;
  int multi_shift; //come soon,  useless now
  int num_offsets;
  double *offsets;
  int iflag_read_gauge;
  char latfile[50];
  int ifsmear;
  int iconf_st;
  int iconf_ed;
  int iconf_in;
  int gauge_format; //1 for quda order; 2 for kfc with big endian
  char prec_sloppy[50];
  char inversemethod[50];
  int indx[4];		//x y z t
  char prop_path[50];
  int tsource_st;
  int zsource_st;
  int ysource_st;
  int xsource_st;
  int tsource_in;
  int zsource_in;
  int ysource_in;
  int xsource_in;
  int source_type;
  int N_smear;
  double k_smear;
  int if_seq;
  int time_seq;
  int dr_st;
  int dr_ed;
  int p_st;
  int p_ed;
  int ifverbose;
  int ifC;
  int ifB;
  int myid;

  global_param(int myid)
    { 
      if(myid==0){
	ioflag=read_d("nx",&nx);
	ioflag=read_d("ny",&ny);
	ioflag=read_d("nz",&nz);
	ioflag=read_d("nt",&nt);
	ioflag=read_lf("kappa", &kappa); 
	ioflag=read_c("dslash_type", ds_type);
	if(!strcmp(ds_type,"TWISTED_MASS"))
	{
	   ioflag=read_lf("mu",&mu);
	}
	if(!strcmp(ds_type,"CLOVER"))
	{
	   ioflag=read_lf("csw",&csw);
	}
	ioflag=read_d("if_anisotropic",&if_anisotropic);
	if(if_anisotropic == 1)
	{
	   ioflag=read_lf("Vlight",&ani_vlight);
	   ioflag=read_lf("Us",&ani_us);
	   ioflag=read_lf("Xsi",&ani_xsi);
	}
	ioflag=read_d("multi_shift",&multi_shift);
	if(multi_shift==1)
	 {
	    ioflag=read_d("num_of_mass",&num_offsets);
	    offsets=(double*)malloc(num_offsets*sizeof(double));
	    for(int imass=0;imass<num_offsets;imass++)
	    ioflag=read_lf("offset",&offsets[imass]);
	 } //come soon,  useless now
	ioflag=read_d("iflag_read_gauge",&iflag_read_gauge);
	if(iflag_read_gauge==2)
	  {
	    ioflag=read_c("latfile",latfile);
	    ioflag=read_d("ifsmear",&ifsmear);
	    ioflag=read_d("iconf_st",&iconf_st);
	    ioflag=read_d("iconf_ed",&iconf_ed);
	    ioflag=read_d("iconf_in",&iconf_in);
	    ioflag=read_d("gauge_format",&gauge_format);
	  }
	ioflag=read_c("quda_prec_sloppy",prec_sloppy);
	ioflag=read_c("inversemethod",inversemethod);
	for(int i=0;i<4;i++){
	  ioflag=read_d("index",&indx[i]);
	}

	ioflag = read_c("prop_path",prop_path);
	ioflag=read_d("tsource_st",&tsource_st);
	ioflag=read_d("tsource_in",&tsource_in);
	ioflag=read_d("zsource_st",&zsource_st);
	ioflag=read_d("zsource_in",&zsource_in);
	ioflag=read_d("ysource_st",&ysource_st);
	ioflag=read_d("ysource_in",&ysource_in);
	ioflag=read_d("xsource_st",&xsource_st);
	ioflag=read_d("xsource_in",&xsource_in);
	ioflag=read_d("source_type",&source_type);

	ioflag=read_d("N_smear",&N_smear);
	ioflag=read_lf("k_smear",&k_smear);

	ioflag=read_d("if_seq",&if_seq);
	if(if_seq == 1)
	  ioflag=read_d("time_seq",&time_seq);
	
	ioflag=read_d("dr_st",&dr_st);
	ioflag=read_d("dr_ed",&dr_ed);
	ioflag=read_d("p_st",&p_st);
	ioflag=read_d("p_ed",&p_ed);
	ioflag=read_d("ifverbose",&ifverbose);
	ioflag=read_d("ifC",&ifC);
	ioflag=read_d("ifB",&ifB);

	if(ioflag==0){
	    printf("input error,exit...\n");
       }
      }
    }
};
enum SourceType_s {
   PointSource,
   WallSource,
   WallSourceWithP,
   SmearedSource,
   RandomSource,
};

// declarations
void ComplexMatrixProduct(cdouble* A, cdouble* B, const int dim, cdouble*C = NULL);
void GenerateNeighbourSitesTable(int *NeighbourSitesTable);
void construct_clover_term_from_gauge(void* clover_inv, void** gauge, QudaPrecision GaugePrecision, 
				      QudaPrecision CloverPrecision, double kappa, double Csw, 
				      int* NeighbourSitesTable, QudaGaugeParam *param);
void stout_smear(void** gauge, QudaPrecision GaugePrecision, double alpha, int *NeighbourSitesTable);
void rotate_spinor(void * spinor, int flag, int basis, QudaPrecision precision);
void ReadGaugeField(char* latfile, void **qauge, QudaPrecision precision, int format);
void apply_anisotropic_to_gauge(void **gauge, QudaGaugeParam *param); 
int smear_source(double *spinorIn0,int tsource,int zsource,int ysource,
	       int xsource,int isource,QudaGaugeParam *gauge_param,QudaInvertParam *inv_param,
	       int N,double k,void** u);
int get_index_quda(int t,int z,int y,int x,int sdim);
int get_xyzt_quda(int index,int sdim,int *x);
int get_index(int t,int z,int y,int x,int sdim);
int get_xyzt(int index,int sdim,int *x);
int init_from_in(global_param &in,QudaDslashType &dslash_type,bool &tune,int &xdim,int &ydim,
		 int &zdim,int &tdim,QudaReconstructType &link_recon,QudaPrecision &prec,
		 QudaReconstructType &link_recon_sloppy,QudaPrecision &prec_sloppy,
		 int myid,QudaGaugeParam &gauge_param,QudaInvertParam &inv_param);
int set_source(SourceType_s SourceType,void *spinorIn0,int tsource,int zsource,int ysource,
	       int xsource,int isource,QudaGaugeParam *gauge_param,QudaInvertParam *inv_param);
double random(double,double);
cdouble Z3(double i);
//==============================================
extern "C" {
    void two_point_(double*prop,double*u,int *ns,int*nmom,double*twopf,int*ntime);
    void three_point_(double*,double*,double*,int*,int*,double*,int*,int*);
    void two_point_b_(double*,double*,int*,int*,double*,int*);
}

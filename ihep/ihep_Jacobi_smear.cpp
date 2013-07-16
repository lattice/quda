/*===============================================
 *   Jacobi smearing
 *   PRD 56.5
 *   Jian Liang
 *   2013 July
===============================================*/
#include "ihep_utility.h"

int Jacobi_smear(cdouble **s,int sdim,QudaGaugeParam *gauge_param,
		 QudaInvertParam *inv_param,void** u,double k,int tsource)
{
  int index,index_n[6];
  int xx[4];
  int Vs = sdim*sdim*sdim;
  cdouble* s_t[Vs];
  ColorMatrix m[6];

  // tmp of color vector
  for(int i=0;i<Vs;i++)
    {
      s_t[i] = new cdouble[Vs*3];
    }

  for(int i=0;i<Vs;i++)
  for(int j=0;j<Vs;j++)
  for(int ic=0;ic<3;ic++)
    {
	s_t[i][j*3+ic] = cdouble(0,0);
    }

  // let s^(n-1) becomes s^(n)
  for(int i=0;i<Vs;i++)
    {
      get_xyzt(i,sdim,xx);
      int x=xx[0];
      int y=xx[1];
      int z=xx[2];
      int t=tsource;
      index      = get_index_quda(t,z,y,x,sdim);  // x 
      index_n[3] = get_index_quda(t,z,y,x-1,sdim);// xm
      index_n[4] = get_index_quda(t,z,y-1,x,sdim);// ym
      index_n[5] = get_index_quda(t,z-1,y,x,sdim);// zm

      m[0]=ColorMatrix(((double*)u[0]+index));   //xp 
      m[1]=ColorMatrix(((double*)u[1]+index));   //yp
      m[2]=ColorMatrix(((double*)u[2]+index));   //zp
      m[3]=ColorMatrix(((double*)u[0]+index_n[3]));//xm
      m[4]=ColorMatrix(((double*)u[1]+index_n[4]));//ym
      m[5]=ColorMatrix(((double*)u[2]+index_n[5]));//zm
      m[3] = m[3].dagger();
      m[4] = m[4].dagger();
      m[5] = m[5].dagger();
	
      t=0;
      index_n[0] = get_index(t,z,y,x+1,sdim);// xp
      index_n[1] = get_index(t,z,y+1,x,sdim);// yp
      index_n[2] = get_index(t,z+1,y,x,sdim);// zp
      index_n[3] = get_index(t,z,y,x-1,sdim);// xm
      index_n[4] = get_index(t,z,y-1,x,sdim);// ym
      index_n[5] = get_index(t,z-1,y,x,sdim);// zm

      for(int ix=0;ix<Vs;ix++)
      for(int j=0;j<6;j++)
      for(int ic=0;ic<3;ic++)
      for(int ic1=0;ic1<3;ic1++)
	s_t[i][ix*3+ic]+=m[j].GetU(ic,ic1)*s[index_n[j]][ix*3+ic1];
    }

  for(int i=0;i<Vs;i++)
  for(int j=0;j<Vs;j++)
  for(int ic=0;ic<3;ic++)
    {
	s[i][j*3+ic] = k*s_t[i][j*3+ic];
    }
  for(int i=0;i<Vs;i++)
    {
      delete[]s_t[i];
    }
  return 1;
}

int smear_source(double *spinorIn,int tsource,int zsource,int ysource,
	       int xsource,int color_source,QudaGaugeParam *gauge_param,QudaInvertParam *inv_param,
	       int N,double k,void** u)
{
  int i,j,index;
  int xdim = gauge_param->X[0];
  int ydim = gauge_param->X[1];
  int zdim = gauge_param->X[2];
  int Vs = xdim*ydim*zdim;
  for(i=0;i<Vs;i++)
  for(int ic=0;ic<6;ic++)
    spinorIn[i*6+ic] = 0.0;

  // no even-odd block here
  // no spin index here
  cdouble* s[Vs];
  for(i=0;i<Vs;i++)
    {
      s[i] = new cdouble[Vs*3];
    }

  // step 0
  for(i=0;i<Vs;i++)
  for(j=0;j<Vs;j++)
  for(int ic=0;ic<3;ic++)
    {
      if(j==i&&ic==color_source){
	s[i][j*3+ic] = cdouble(1,0);
      }
      else{
	s[i][j*3+ic] = cdouble(0,0);
      }
    }

  index = get_index(0,zsource,ysource,xsource,xdim);

  for(int ix=0;ix<Vs;ix++)
  for(int ic=0;ic<3;ic++){
    spinorIn[ix*6+ic*2+0] += real(s[index][ix*3+ic]);
    spinorIn[ix*6+ic*2+1] += imag(s[index][ix*3+ic]);
  }

  // main loop of smear times
  for(int n=1;n<=N;n++)
    {
      Jacobi_smear(s,xdim,gauge_param,inv_param,u,k,tsource);
      for(int ix=0;ix<Vs;ix++)
      for(int ic=0;ic<3;ic++){
	spinorIn[ix*6+ic*2+0] += real(s[index][ix*3+ic]);
	spinorIn[ix*6+ic*2+1] += imag(s[index][ix*3+ic]);
      }
    }

  for(int i=0;i<Vs;i++)
    {
      delete[]s[i];
    }
  return 1;
}

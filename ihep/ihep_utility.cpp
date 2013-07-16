#include "ihep_utility.h"

/********************************************************
             ybyang & lj
	     utilities
*********************************************************/

// C = A*B while C!=NULL or A = A*B when C=NULL
void ComplexMatrixProduct(cdouble* A, cdouble* B,  const int dim, cdouble* C)
{
  if(C == NULL){
    cdouble* T = new cdouble [dim*dim];
    for(int i=0; i<dim; i++)
    for(int j=0; j<dim; j++)
      T[i*dim+j] = 0;
    for(int i=0; i<dim; i++)
    for(int j=0; j<dim; j++)
    for(int k=0; k<dim; k++)
      T[i*dim+k] += A[i*dim+j]*B[j*dim+k];
    for(int i=0; i<dim; i++)
    for(int j=0; j<dim; j++)
      A[i*dim+j] = T[i*dim+j];
    delete[] T;
    T = NULL;
  }
  else{
    for(int i=0; i<dim; i++)
    for(int j=0; j<dim; j++)
      C[i*dim+j] = 0;
    for(int i=0; i<dim; i++)
    for(int j=0; j<dim; j++)
    for(int k=0; k<dim; k++)
      C[i*dim+k] += A[i*dim+j]*B[j*dim+k];
  }
}

void GenerateNeighbourSitesTable(int *NeighbourSitesTable){
// note here mu=0,...,3, where 0<->x, 1<->y, 2<->z, 3<->t).

  /*==========================================================
  the 7 neighbour needed is 1~7

      3(-1,1)	       2(1,0)		 
	  |------------------------------------|
	  |		 |                     |
	  |	U2       |          U1         |	
	  |		 |                     |1(1,0)
  4(-1,0) |-------------0(0,0)------------------------>mu
	  |		 |		       |
	  |	U3	 |	    U4	       |
	  |		 |		       |
	  |------------------------------------|
     5(-1,-1)	      6(0,-1)                7(1,-1)
  ==============================================================*/

  int realsite;
  int xdim = Z[0];
  int ydim = Z[1];
  int zdim = Z[2];
  int tdim = Z[3];
  int ix[4] = {0, 0, 0, 0};  
  int ixx[4] = {0, 0, 0, 0};  
  int dim[4] = {xdim, ydim, zdim, tdim};
  int offset[14] = {1, 0,    0, 1,    -1, 1,    -1, 0,    -1, -1,    0, -1,    1, -1};  
  int munu;
  int vi, vj;

  for(int sites=0; sites<V; sites++){
    realsite = (sites%Vh)*2;
    ixx[0] = realsite%xdim; 
    ixx[1] = (realsite/xdim)%ydim;
    ixx[2] = ((realsite/xdim)/ydim)%zdim;
    ixx[3] = ((realsite/xdim)/ydim)/zdim;
    if((ixx[0]+ixx[1]+ixx[2]+ixx[3])%2 != sites/Vh) ixx[0]++;
    munu = 0;
    for(int mu=0; mu<4; mu++)
    for(int nu=0; nu<4; nu++){
      if(nu == mu)	continue;	 
      for(int neighbor=0; neighbor<7; neighbor++){
	ix[0] = ixx[0]; 
	ix[1] = ixx[1]; 
	ix[2] = ixx[2]; 
	ix[3] = ixx[3]; 

	ix[mu] += offset[neighbor*2 + 0];
	if(ix[mu]==dim[mu] || ix[mu] ==-1)
	  ix[mu]  = (ix[mu] + dim[mu])%dim[mu];

	ix[nu] += offset[neighbor*2 + 1];
	if(ix[nu]==dim[nu] || ix[nu] ==-1)
	  ix[nu]  = (ix[nu] + dim[nu])%dim[nu];

	vi = ix[0] + dim[0]*(ix[1] + dim[1]*(ix[2] + dim[2]*ix[3]));
	vj = vi/2+((ix[0]+ix[1]+ix[2]+ix[3])%2)*V/2;
	NeighbourSitesTable[sites*84 + munu*7 +neighbor] = vj;
      }
      munu ++;
    }
  }
  cout<<"Neibhour is here "<<V*12*7*sizeof(int)<<endl;
  FILE * pfileOut;
  pfileOut=fopen("./NeighbourSitesTable","wb");
  fwrite((int*)NeighbourSitesTable, V*12*7*sizeof(int), 1, pfileOut);
  fflush(pfileOut);
  fclose(pfileOut);
}

template <typename Float>
void plaquette(Float* gauge[4], int* NeighbourSitesTable){
/*=======================================================
	nu			
(n+nu)	|----------2----------|
	|                     |
	3                     1	
	|                     |(n+mu)
	n ---------0------------->mu

plaquette(n) = link0*link1*link2^*link3^
=========================================================*/
int xdim = Z[0];
int ydim = Z[1];
int zdim = Z[2];
int tdim = Z[3];
int count = 0;
int munu[4][4];

for(int i=0; i<4; i++)
for(int j=0; j<4; j++)
  {
    if(i==j){
	munu[i][j] = 0;
    }else{
	munu = count;
	count++;
    }
  }

ColorMatrix link[4]; 
ColorMatrix planq;
double plaq; 
double aa = xdim*ydim*zdim*6;

for(int it=0; it<tdim; it++){
  for(int iz=0; iz<zdim; iz++)
  for(int iy=0; iy<ydim; iy++)
  for(int ix=0; ix<xdim; ix++)
  for(int mu=0; mu<3; mu++)
  for(int nu=mu+1; nu<4; nu++){
    int index = ix + xdim* (iy + ydim* (iz + zdim* it));
    index = index/2+((ix + iy + iz + it)%2)*(xdim*ydim*zdim*tdim)/2;
    link[0] = ColorMatrix( gauge[mu] + index*18);
    link[1] = ColorMatrix( gauge[nu] + NeighbourSitesTable[index*84 + munu[mu][nu]* 7 + 0]*18);
    link[2] = ColorMatrix( gauge[mu] + NeighbourSitesTable[index*84 + munu[mu][nu]* 7 + 1]*18);
    link[3] = ColorMatrix( gauge[nu] + index*18);
    planq = planq + link[0]*link[1]*link[2].dagger()*link[3].dagger();
  }
  if(it%4 == 0){
    plaq = real(planq.trace());
    plaq /= (aa*4);
    cout<<"t= "<<'\t'<<it<<" planquette="<<'\t'<<plaq<<endl;
    planq = planq - planq;
  }
}
}

/********************************************************
                    added by ybyang
		    moved by JianLiang
*********************************************************/
//input reading~
int read_d(char *src,int *data){

  char checkname[60];
  scanf("%s%d",checkname,data);
  if(!strcmp(checkname,src)) {
     printf("%-40.40s%d\n",checkname,*data);
     return 1;}
  else {
       printf("%-40.40s%d wrong \n",checkname,*data);
      return 0;}
}
int read_lf(char *src,double *data){
   char checkname[60];
   scanf("%s%lf",checkname,data);
   if(!strcmp(checkname,src)){
      printf("%-40.40s%lf\n",checkname,*data);
      return 1;}
   else return 0;
}
int read_c(char *src,char *data){
   char checkname[60];
   scanf("%s%s",checkname,data);
   if(!strcmp(checkname,src)){
      printf("%-40.40s%s\n",checkname,data);
     return 1;}
   else return 0;
}

/*********************************************/
template <typename Float>
void ReadQudaData(Float **res,int n,FILE *pfile){
     double *rfile;
     rfile= (double*)malloc(sizeof(double)*n);
     int dir,i;
     for(dir=0;dir<4;dir++){
       if(fread(rfile,sizeof(double),n,pfile)==0)
         printf("file size is incorrect!\n");
       for(i=0;i<n;i++)
         res[dir][i]=(Float)rfile[i];
     }
     free(rfile);
}

// reading data with format of kentucky university
template <typename Float>
void ReadKfcData(Float **res,int n,FILE *pfile){
  Float *rfile;
  rfile= (Float*)malloc(sizeof(Float)*n);
  int iDir,iT,iZ,iY,iX,iColorCol,iColorRow,iPart,m;
  int nx = Z[0];
  int ny = Z[1];
  int nz = Z[2];
  int nt = Z[3];
  unsigned char* pc1;
  unsigned char* pc2;

  for(iDir=0; iDir < 4; iDir++)
  {
    if(fread((Float*)rfile,sizeof(Float),n,pfile)==0)
      printf("file read error!\n");

    for(iT = 0; iT < nt; iT++)
    for(iZ = 0; iZ < nz; iZ++)
    for(iY = 0; iY < ny; iY++)
    for(iX = 0; iX < nx; iX++)
    {  
      int vi=iX+iY*nx+iZ*nx*ny+iT*nx*ny*nz;
      int vj=vi/2+((iX+iY+iZ+iT)%2)*V/2;
      for(iColorCol = 0; iColorCol < 3; iColorCol++)
      for(iColorRow = 0; iColorRow < 3; iColorRow++)
      for(iPart = 0; iPart < 2; iPart++)
        {  
           pc1=(unsigned char *)&rfile[(iColorRow*6+2*iColorCol+iPart)*V+vi];
           pc2=((unsigned char *)(res[iDir]+(vj*18+iColorCol*6+iColorRow*2+iPart)))+sizeof(Float)-1;
           for(m=0;m<(int)sizeof(Float);m++,pc1++,pc2--)
      	   (*pc2)=(*pc1);
           /*
           *(res[iDir]+(vj*18+iColorCol*6+iColorRow*2+iPart)) =
           rfile[(iColorRow*6+2*iColorCol+iPart)*V+vi];
           */
        } 
    }
  }
  free(rfile);
}

void ReadGaugeField(char* latfile,void **gauge, QudaPrecision precision, int format) {
  printf("%s%s\n","the latfile is ",latfile);
  FILE *pfile=fopen(latfile, "rb");
  if(pfile==NULL) printf("latfile do not exist!\n");
  else printf("begin to read file......");
  fflush(stdout);
  if(format == 1){
    if (precision == QUDA_DOUBLE_PRECISION) 
      ReadQudaData((double**)gauge,gaugeSiteSize*Vh*2,pfile);
    else  
      ReadQudaData((float**)gauge,gaugeSiteSize*Vh*2,pfile);
  }else if(format == 2){
    if (precision == QUDA_DOUBLE_PRECISION) 
      ReadKfcData((double**)gauge,gaugeSiteSize*Vh*2,pfile);
    else  
      ReadKfcData((float**)gauge,gaugeSiteSize*Vh*2,pfile);
  } 
}
/**********************************************/
// rotate gamma matrices
void set_rotate(double *rotate, double *inv_rotate, int basis){
  int i,j;
  for(i=0;i<4;i++)
    for(j=0;j<4;j++){rotate[4*i+j]=0.0;inv_rotate[4*i+j]=0.0;}
   
  if(basis==0){	//chiral to tmQCD
    rotate[4*0+3]=-1.0;
    rotate[4*1+2]= 1.0;
    rotate[4*2+1]= 1.0;
    rotate[4*3+0]=-1.0;
    inv_rotate[4*0+3]=-1.0;
    inv_rotate[4*1+2]= 1.0;
    inv_rotate[4*2+1]= 1.0;
    inv_rotate[4*3+0]=-1.0;
  }else if(basis==1){    //UKQCD to tmQCD
    rotate[4*0+0]=-1.0/sqrt(2);
    rotate[4*0+2]= 1.0/sqrt(2); 
    rotate[4*1+1]=-1.0/sqrt(2); 
    rotate[4*1+3]= 1.0/sqrt(2); 
    rotate[4*2+0]= 1.0/sqrt(2); 
    rotate[4*2+2]= 1.0/sqrt(2); 
    rotate[4*3+1]= 1.0/sqrt(2); 
    rotate[4*3+3]= 1.0/sqrt(2);
    
    inv_rotate[4*0+0]=-1.0/sqrt(2);     
    inv_rotate[4*0+2]= 1.0/sqrt(2);     
    inv_rotate[4*1+1]=-1.0/sqrt(2);     
    inv_rotate[4*1+3]= 1.0/sqrt(2);     
    inv_rotate[4*2+0]= 1.0/sqrt(2);     
    inv_rotate[4*2+2]= 1.0/sqrt(2);     
    inv_rotate[4*3+1]= 1.0/sqrt(2);     
    inv_rotate[4*3+3]= 1.0/sqrt(2);     
  }else if(basis==3){    //rotate is Jldg 2 Chrial
    rotate[4*0+1]=-1.0/sqrt(2);
    rotate[4*0+3]= 1.0/sqrt(2); 
    rotate[4*1+0]= 1.0/sqrt(2); 
    rotate[4*1+2]=-1.0/sqrt(2); 
    rotate[4*2+1]=-1.0/sqrt(2); 
    rotate[4*2+3]=-1.0/sqrt(2); 
    rotate[4*3+0]= 1.0/sqrt(2); 
    rotate[4*3+2]= 1.0/sqrt(2);
    
    inv_rotate[4*0+1]= 1.0/sqrt(2);     
    inv_rotate[4*0+3]= 1.0/sqrt(2);     
    inv_rotate[4*1+0]=-1.0/sqrt(2);     
    inv_rotate[4*1+2]=-1.0/sqrt(2);     
    inv_rotate[4*2+1]=-1.0/sqrt(2);     
    inv_rotate[4*2+3]= 1.0/sqrt(2);     
    inv_rotate[4*3+0]= 1.0/sqrt(2);     
    inv_rotate[4*3+2]=-1.0/sqrt(2);     
  }else printf("Invalid gamma basis to rotate...\n");
}   
template <typename Float>
void applyrotate(Float *spinor, double *rotate){
  int ix,is,id;
  double *temp=(double*)malloc(spinorSiteSize*sizeof(double));
  for(ix=0;ix<Vh*2;ix++)
  {   
    for(is=0;is<spinorSiteSize;is++) temp[is]=0.0;

    for(id=0;id<4;id++)
    for(is=0;is<spinorSiteSize;is++)
    {
      int ic=is%6,id0=is/6;
      temp[id*6+ic]+=(double)spinor[ix*spinorSiteSize+is]*rotate[id*4+id0];
    }
    for(is=0;is<spinorSiteSize;is++)
      spinor[ix*spinorSiteSize+is]=(Float)temp[is];
  }
  free(temp);
}

void rotate_spinor(void *spinor,int flag, int basis, QudaPrecision precision)
{
    double *rotate=(double*)malloc(16*sizeof(double));
    double *inv_rotate=(double*)malloc(16*sizeof(double));
    set_rotate(rotate,inv_rotate,basis);

    if (precision == QUDA_DOUBLE_PRECISION)
    {
      if(flag==1) applyrotate((double*)spinor, rotate);
      else applyrotate((double*)spinor, inv_rotate);
    }
    else  
    {
      if(flag==1) applyrotate((float*)spinor, rotate);
      else applyrotate((float*)spinor, inv_rotate);
    }

    free(rotate);
    free(inv_rotate);
}

// Apply anisotropic factors to links
void apply_anisotropic_to_gauge(void **gauge, QudaGaugeParam *param){
 if(param->cpu_prec == QUDA_DOUBLE_PRECISION){
   double eta_x = param->Vlight/param->Us;
   double eta_t = param->Xsi;
   cout<<"eta_x & eta_t = "<<eta_x<<'\t'<<eta_t<<endl;
   double eta = eta_t/eta_x;
   for (int i = 0; i < gaugeSiteSize*Vh*2; i++) {
     *((double*)gauge[3]+i) *= eta;
   }
 }else{
   float eta_x = param->Vlight/param->Us;
   float eta_t = param->Xsi;
   cout<<"eta_x & eta_t = "<<eta_x<<'\t'<<eta_t<<endl;
   float eta = eta_t/eta_x;
   for (int i = 0; i < gaugeSiteSize*Vh*2; i++) {
     *((float*)gauge[3]+i) *= eta;
   }
 }
}

// init gauge_param & inv_param using input params
int init_from_in(global_param &in,QudaDslashType &dslash_type,bool &tune,int &xdim,int &ydim,int &zdim,
		 int &tdim,QudaReconstructType &link_recon,QudaPrecision &prec,QudaReconstructType &link_recon_sloppy,
		 QudaPrecision &prec_sloppy,int myid,QudaGaugeParam &gauge_param,QudaInvertParam &inv_param)
{
  // external params
  if(!strcmp(in.ds_type,"TWISTED_MASS")) dslash_type = QUDA_TWISTED_MASS_DSLASH;
  else if(!strcmp(in.ds_type,"WILSON")) dslash_type = QUDA_WILSON_DSLASH;
  else if(!strcmp(in.ds_type,"CLOVER")) dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  else if(!strcmp(in.ds_type,"DOMAIN_WALL")) dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }
  xdim = in.nx;
  ydim = in.ny;
  zdim = in.nz;
  tdim = in.nt;

  if (prec_sloppy == QUDA_INVALID_PRECISION){
    //default prec_sloppy is single 
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    //default link_recon_sloppy is 12
    link_recon_sloppy = link_recon;
  }
  //link_recon = QUDA_RECONSTRUCT_NO;
  //link_recon_sloppy = link_recon;

  if(myid==0)cout<<"dslash_type = "<<'\t'<<" "<<'\t'<<get_dslash_type_str(dslash_type)<<endl;
  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = QUDA_HALF_PRECISION;
  //QudaPrecision cuda_prec_precondition = QUDA_SINGLE_PRECISION;
  if(!strcmp(in.prec_sloppy,"DOUBLE"))
    {cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;}
  else if(!strcmp(in.prec_sloppy,"SINGLE"))
    {cuda_prec_sloppy = QUDA_SINGLE_PRECISION;}
  else if(!strcmp(in.prec_sloppy,"HALF"))
    {cuda_prec_sloppy = QUDA_HALF_PRECISION;}
  if(myid==0)cout<<"cpu_prec ="<<'\t'<<'\t'<<get_prec_str(cpu_prec)<<endl;
  if(myid==0)cout<<"cuda_prec ="<<'\t'<<'\t'<<get_prec_str(cuda_prec)<<endl;;
  if(myid==0)cout<<"cuda_prec_sloppy ="<<'\t'<<get_prec_str(cuda_prec_sloppy)<<endl;
  if(myid==0)cout<<"cuda_prec_precondition ="<<get_prec_str(cuda_prec_precondition)<<endl;
  //================================================
  
  // gauge_params
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  // *** QUDA parameters begin here.
  //int multi_shift = 0; // whether to test multi-shift or standard solver

  gauge_param.anisotropy = 1;
  //gauge_param.anisotropy = 2.38;
  //anisotropic params
  gauge_param.Xsi = in.ani_xsi;
  gauge_param.Us  = in.ani_us;
  gauge_param.Vlight = in.ani_vlight;

  gauge_param.type = QUDA_WILSON_LINKS;
  //gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  //gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  //gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.ga_pad = 0; // 24*24*24/2;
  //================================================

  // invert_params
  int myLs = 1; // FIXME
  inv_param.Ls = (dslash_type == QUDA_DOMAIN_WALL_DSLASH) ? myLs : 1;
  inv_param.dslash_type = dslash_type;

  //double mass = -0.4125;
  //inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));
  inv_param.kappa = in.kappa;
  double eta_x = gauge_param.Vlight/gauge_param.Us;
  if(in.if_anisotropic) 
  inv_param.kappa *= eta_x;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.mu = in.mu;
    inv_param.twist_flavor = QUDA_TWIST_MINUS;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    //not been set yet
    inv_param.mass = 0.02;
    inv_param.m5 = -1.8;
  }

  // offsets used only by multi-shift solver
  inv_param.num_offset = 4;
  double offset[QUDA_MAX_MULTI_SHIFT] = {0.01, 0.02, 0.03, 0.04};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];

  //inv_param.solution_type = QUDA_MATPC_SOLUTION; //solve half of the system, see the email from clark.
  inv_param.solution_type = QUDA_MAT_SOLUTION;  //solve the full system, while preconditon solver is still used.
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  if(!strcmp(in.inversemethod,"CG"))
    {inv_param.inv_type = QUDA_CG_INVERTER;}
  else if(!strcmp(in.inversemethod,"GCR"))
    {inv_param.inv_type = QUDA_GCR_INVERTER;}
  else if(!strcmp(in.inversemethod,"MR"))
    {inv_param.inv_type = QUDA_MR_INVERTER;}  
  else
    {inv_param.inv_type = QUDA_BICGSTAB_INVERTER;}

  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
    inv_param.inv_type = QUDA_CG_INVERTER;
  } else {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
  }

  // inv_param.solve_type = QUDA_NORMOP_SOLVE;
  // inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  // inv_param.solve_type = QUDA_DIRECT_SOLVE;
  // inv_param.inv_type = QUDA_CG_INVERTER;

  char* solution_str[] = {"QUDA_MAT_SOLUTION",
    			  "QUDA_MATDAG_MAT_SOLUTION",
    			  "QUDA_MATPC_SOLUTION",
    			  "QUDA_MATPCDAG_MATPC_SOLUTION"};
  char* solve_str[] = {"QUDA_DIRECT_SOLVE",
    		       "QUDA_NORMOP_SOLVE",
    		       "QUDA_DIRECT_PC_SOLVE",
    		       "QUDA_NORMOP_PC_SOLVE"};
  char* inv_str[] = {   "QUDA_CG_INVERTER",
    			"QUDA_BICGSTAB_INVERTER",
    			"QUDA_GCR_INVERTER",
    			"QUDA_MR_INVERTER"};

  if(myid==0)cout<<"solution_type ="<<'\t'<<'\t'<<solution_str[inv_param.solution_type]<<endl;
  if(myid==0)cout<<"solve_type ="<<'\t'<<'\t'<<solve_str[inv_param.solve_type]<<endl;
  if(myid==0)cout<<"inv_type ="<<'\t'<<'\t'<<inv_str[inv_param.inv_type]<<endl;
  /* 
  //===default settings===
  inv_param.gcrNkrylov = 10;
  inv_param.tol = 5e-7;
  //inv_param.residual_type = QUDA_HEAVY_QUARK_RESIDUAL;
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) inv_param.tol_offset[i] = inv_param.tol;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 1e-1; // ignored by multi-shift solver
  */
  //===our settings===
  inv_param.gcrNkrylov = 30;
  inv_param.tol = 5e-8;
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) inv_param.tol_offset[i] = inv_param.tol;
  inv_param.maxiter = 100000;
  inv_param.reliable_delta = 1e-2; // ignored by multi-shift solver

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = QUDA_INVALID_INVERTER;
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  //inv_param.prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  //inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  if(in.ifverbose==1)
    inv_param.verbosity = QUDA_VERBOSE;
  else
    inv_param.verbosity = QUDA_SILENT;

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
    Ls = 1;
  }
  return 1;
}

// set source vector for inversion
int set_source(SourceType_s SourceType,void *spinorIn0,int tsource,int zsource,int ysource,
	       int xsource,int isource,QudaGaugeParam *gauge_param,QudaInvertParam *inv_param)
{
  int i,ispin,index;
  int xdim = gauge_param->X[0];
  int ydim = gauge_param->X[1];
  int zdim = gauge_param->X[2];
  int Vs = xdim*ydim*zdim/2;
  double Px = 2.0*3.1415926/xdim;

  for(i=0;i<Vs*gauge_param->X[3]*2;i++)
  for(ispin=0;ispin<24;ispin++)
    if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION){
      *((float*)spinorIn0+i*24+ispin) = 0.0;
    }else{
      *((double*)spinorIn0+i*24+ispin) = 0.0;
    }

  switch(SourceType){
    case PointSource:
      if(isource==0)
	cout<<endl<<"=-=-=-=-=-=  src is point src @ tzyx point "<<tsource<<" "<<zsource<<" "<<ysource<<" "<<xsource<<"    -=-=-=-=-="<<endl<<endl;
      // create a point source at xyzt source
      index = xsource+ xdim*(ysource+ ydim*(zsource+ zdim*tsource));
      index = index/2+((xsource+ysource+zsource+tsource)%2)*Vh;
      if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION)
	*((float*)spinorIn0+index*24+isource*2) = 1.0;
      else
	*((double*)spinorIn0+index*24+isource*2) = 1.0;
    break;
 
    case WallSource:
      if(isource==0)
	cout<<endl<<"=-=-=-=-=-=-=-   src is wall src @ "<<tsource<< " time slice   -=-=-=-=-=-=-="<<endl<<endl;
      // create wall source without momentum
      if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION)
	for(i=tsource*Vs;i<(tsource+1)*Vs;i++){
	  *((float*)spinorIn0+i*24+isource*2) = 1.0;
	  *((float*)spinorIn0+(i+Vh)*24+isource*2) = 1.0;
	}
      else
	for(i=tsource*Vs;i<(tsource+1)*Vs;i++){
	  *((double*)spinorIn0+i*24+isource*2) = 1.0;
	  *((double*)spinorIn0+(i+Vh)*24+isource*2) = 1.0;
	}
    break;
 
    case WallSourceWithP:
      if(isource==0)
	cout<<endl<<"=-=-=-=-=-   src is wall src with P @ "<<tsource<<" time slice   -=-=-=-=-=-"<<endl<<endl;
      // create wall source with momentum 
      for(int it=tsource; it<tsource+1; it++)
      for(int iz=0; iz<zdim; iz++)
      for(int iy=0; iy<ydim; iy++)
      for(int ix=0; ix<xdim; ix++){
	index = ix + xdim* (iy + ydim* (iz + zdim* it));
	index = index/2+((ix+iy+iz+it)%2)*Vh;
	if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION){
	  *((float*)spinorIn0+index*24+isource*2) = cos(Px*ix);
	  *((float*)spinorIn0+index*24+isource*2+1) = -sin(Px*ix);
	}
	else{
	  *((double*)spinorIn0+index*24+isource*2) = cos(Px*ix);
	  *((double*)spinorIn0+index*24+isource*2+1) = -sin(Px*ix);
	}
      }
      break;

    default:
      cout<<endl<<"=-=-=-=-=-=-=  src type is invalid!    =-=-=-=-=-=-="<<endl<<endl;
      exit(1);
  }
  return 1;
}

// return index using quda even-odd block
int get_index_quda(int t,int z,int y,int x,int sdim)
{
  int index;
  x=(x+sdim)%sdim;
  y=(y+sdim)%sdim;
  z=(z+sdim)%sdim;
  index = x+ sdim*(y+ sdim*(z+ sdim*t));
  index = index/2+((x+y+z+t)%2)*Vh;
  return index;
}

// return xyzt index
int get_xyzt_quda(int index,int sdim,int *x)
{
  int realindex = (index%Vh)*2+(index/Vh);
  x[0] = realindex%sdim; 
  x[1] = (realindex/sdim)%sdim;
  x[2] = ((realindex/sdim)/sdim)%sdim;
  x[3] = ((realindex/sdim)/sdim)/sdim;
  return 0;
}

// return index no even-odd block
int get_index(int t,int z,int y,int x,int sdim)
{
  int index;
  x=(x+sdim)%sdim;
  y=(y+sdim)%sdim;
  z=(z+sdim)%sdim;
  index = x+ sdim*(y+ sdim*(z+ sdim*t));
  return index;
}

// return xyzt  no even-odd block
int get_xyzt(int index,int sdim,int *x)
{
  int realindex = index;
  x[0] = realindex%sdim; 
  x[1] = (realindex/sdim)%sdim;
  x[2] = ((realindex/sdim)/sdim)%sdim;
  x[3] = ((realindex/sdim)/sdim)/sdim;
  return 0;
}

double random(double min, double max)
{
  return min+(max-min)*rand()/(RAND_MAX + 1.0);
}

cdouble Z3(double i)
{
  switch(int(i))
    {
    case 1:
      return cdouble(1,0);
      break;
    case 2:
      return cdouble(-0.5,-sqrt(3)/2);
      break;
    case 3:
      return cdouble(-0.5,sqrt(3)/2);
      break;
    default:
      cout<<"Z3 noise error..."<<endl;
      return 0;
    }
}

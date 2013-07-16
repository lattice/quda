//=============================================
//   Jian Liang
//   20130503
//   20130613 
//=============================================
#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>

// lj =========================================
// set some core constants(anisotropic)
#include <dslash_quda.h> 
// std c heads
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include "ihep_utility.h"
#define mpi
#ifdef mpi
#include <mpi.h>
#endif
//=============================================

#ifdef QMP_COMMS
#include <qmp.h>
#endif
#include <gauge_qio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
//In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

using namespace std;

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;
extern char latfile[];
extern void usage(char** );

void display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d \n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim);     

  printfQuda("Grid partition info):     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  return;
}

int main(int argc, char **argv)
{
  //=================  MPI HEAD  ==================
  int myid=0;
  int ncores=1;
  MPI_Status mpi_error;
#ifdef mpi
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&ncores);
  if (ncores!=12) {cout<<"should use all cores on one node!"<<endl; return 1;}
#endif

  //=================  Read input file  ==================
  if(myid==0)cout<<endl<<"=-=-=-=-=-=-=-=-=   Reading Input File   =-=-=-=-=-=-=-=-=-=-=-="<<endl<<endl;
  global_param in(myid);
  if(myid==0){
    if(in.ioflag==0){
      printf("input error,exit...\n");
      return 1;
    }
    cout<<endl<<"=-=-=-=-=-=-=-=-=   Read Input File Done  =-=-=-=-=-=-=-=-=-=-=-="<<endl<<endl;
  }

#ifdef mpi
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&in,sizeof(class global_param),MPI_CHAR,0,MPI_COMM_WORLD);
#endif
  // things that all threads have to do
  in.ani_us = sqrt(sqrt(in.ani_us));

  // *** QUDA parameters begin here.
  // int multi_shift = 0; // whether to test multi-shift or standard solver
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  if(myid==0)cout<<endl<<"=-=-=-=-=-=-=-=-=-=-=   some params   =-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl<<endl;
  init_from_in(in,dslash_type,tune,xdim,ydim,zdim,tdim,link_recon,
	   prec,link_recon_sloppy,prec_sloppy,myid,gauge_param,inv_param);

  int Vs = xdim*ydim*zdim/2; // half of the space volume
  int Vh = tdim*Vs;  // half of the whole lattice volume 
  int np = in.p_ed-in.p_st+1; // No. of momentums
  int nr = in.dr_ed-in.dr_st+1; // No. of spacial dis.
  int nsp[np];
  int nsf = 0;
  for(int i=0;i<np;i++)
    nsp[i] = 0;
  for(int i=0;i<np;i++){
    int p = i + in.p_st;
    for(int p1=0;p1<=p;p1++)
    for(int p2=0;p2<=p;p2++)
    for(int p3=0;p3<=p;p3++)
      if(p1*p1+p2*p2+p3*p3==p){
	nsp[i]++;
	nsf++;
      }
  }

  if(myid==0)cout<<"np="<<'\t'<<'\t'<<'\t'<<np<<endl;
  if(myid==0)cout<<"nr="<<'\t'<<'\t'<<'\t'<<nr<<endl;
  if(myid==0)for(int i=0;i<np;i++)cout<<"nsp["<<i<<"]="<<'\t'<<'\t'<<'\t'<<nsp[i]<<endl;
  if(myid==0)cout<<"nsf="<<'\t'<<'\t'<<'\t'<<nsf<<endl;
  if(myid==0)cout<<endl<<"=-=-=-=-=-=-=-=-=-=-=   params done   -=-=-=-=-=-=-=-=-=-=-=-=-="<<endl<<endl;
  //======================================================

  //set source type
  SourceType_s SourceType;
  SourceType = (enum SourceType_s)in.source_type; // force transfer

  //set some kernel constants for anisotropic lattice
  quda::setL_us(in.ani_us);
  quda::setL_Vlight(in.ani_vlight);
  quda::setL_Xsi(in.ani_xsi);
  //cout<<quda::getL_us()<<" "<<quda::getL_Vlight()<<" "<<quda::getL_Xsi()<<endl; 

  double kappa5;
  kappa5 = 0.5/(5 + inv_param.m5);  
  if(myid==0)display_test_info();
  if(myid==0)cout<<endl<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-"<<endl<<endl;

  setSpinorSiteSize(24);
  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover_inv=0, *clover=0;
  if(myid==0){
    for (int dir = 0; dir < 4; dir++) {
      gauge[dir] = malloc(V*gaugeSiteSize*gSize);
    }
  }

  int ntime = tdim/ncores;
  // propagators
  double *sf[nsf+1];
  for(int i=0;i<=nsf;i++)
    if(myid==0){
      sf[i]= (double*)malloc(V*spinorSiteSize*sSize*12);
    }else{
      sf[i]= (double*)malloc(ntime*Vs*2*spinorSiteSize*sSize*12);
    }

  MPI_Barrier(MPI_COMM_WORLD);
  char GaugeFile[100];

  if(myid==0){
    // declare the dimensions of the communication grid
    initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);
    // initialize the QUDA library
    initQuda(device);
  }

  // main loop
  for(int icfg=in.iconf_st;icfg<in.iconf_ed+1;icfg+=in.iconf_in){ 
    // start the timer
    double time0 = -((double)clock());
    if(myid==0){
      sprintf(GaugeFile, "%s%04d.dat", in.latfile, icfg);
      /* 
      if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
	read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
	construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
      } else { // else generate a random SU(3) field
	construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
      }
      */
      int * NeighbourSitesTable = new int[V*12*7];
      if(in.iflag_read_gauge==2){
	ReadGaugeField(GaugeFile, gauge, gauge_param.cpu_prec, in.gauge_format);
	printf("read file done! \n");
	fflush(stdout);

	if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || in.ifsmear == 1){
	  // the Neighboursitestable stores a site's 7 neigbours in an arbitriy plane
	  // 12 planes 7 neigbours needed per plane. So V*12*7
	  FILE * pfileIn;
	  if((pfileIn = fopen("./NeighbourSitesTable","rb")) == NULL){
	     GenerateNeighbourSitesTable(NeighbourSitesTable);
	     printf("NeighbourSitesTable generated\n");
	     fflush(stdout);}
	  else{
	     fflush(pfileIn);
	     fread((int*)NeighbourSitesTable, V*12*7*sizeof(int), 1, pfileIn);
	     printf("NeighbourSitesTable read from file\n");
	     fflush(stdout);
	     fclose(pfileIn);}
	}
	// for istropic lattice and periodic BC, construct_gauge_field do nothing
	construct_gauge_field(gauge, in.iflag_read_gauge, gauge_param.cpu_prec, &gauge_param);
      }
      else{ // else generate a random SU(3) field
	construct_gauge_field(gauge, in.iflag_read_gauge, gauge_param.cpu_prec, &gauge_param);
      }

      if(dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	//double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
	//double diag = 1.0; // constant added to the diagonal
	
	size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
	clover_inv = malloc(V*cloverSiteSize*cSize);
	//construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);
	construct_clover_term_from_gauge(clover_inv,  gauge, gauge_param.cpu_prec, inv_param.clover_cpu_prec,
					 in.kappa, in.csw, NeighbourSitesTable, &gauge_param);

	//=================  stout smear  ===================
	if (in.ifsmear == 1){
	 double alpha = 0.1;
	 stout_smear(gauge, gauge_param.cpu_prec, alpha, NeighbourSitesTable);
	}
      
	if(NeighbourSitesTable != NULL){
	  delete [] NeighbourSitesTable; 
	  NeighbourSitesTable = NULL;
	}

	// The uninverted clover term is only needed when solving the unpreconditioned
	// system or when using "asymmetric" even/odd preconditioning.
	// default is QUDA_DIRECT_PC_SOLVE & QUDAS_MATPC_EVEN_EVEN, So preconditioned is true while asymmetric is false
	// So in this case we only need clover_inv
	int preconditioned = (inv_param.solve_type == QUDA_DIRECT_PC_SOLVE ||
			      inv_param.solve_type == QUDA_NORMOP_PC_SOLVE);
	int asymmetric = preconditioned &&
			     (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
			      inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);
	if (!preconditioned) {
	  clover = clover_inv;
	  clover_inv = NULL;
	} else if (asymmetric) { // fake it by using the same random matrix
	  clover = clover_inv;   // for both clover and clover_inv
	} else {
	  clover = NULL;
	}
      }

      // apply \eta_x & \eta_t to gauge links
      if(in.if_anisotropic){ 
	apply_anisotropic_to_gauge(gauge, &gauge_param);
      }
      cout<<"inv_kappa = "<<inv_param.kappa<<endl;
      cout<<"rescale_kappa = "<<in.kappa<<endl;
      cout<<endl<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-="<<endl<<endl;
    }

    void *spinorIn0 = 0;
    void *spinorInP = 0;
    void *spinorCheck = 0;
    void *randomVector = 0;
    if(myid==0){
      spinorIn0 = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
      spinorInP = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
      spinorCheck = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
      randomVector = malloc(Vs*2*2);
    }
    /*
    void *spinorOut = NULL, **spinorOutMulti = NULL;
    if (multi_shift) {
      spinorOutMulti = (void**)malloc(inv_param.num_offset*sizeof(void *));
      for (int i=0; i<inv_param.num_offset; i++) {
        spinorOutMulti[i] = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
      }
    } else {
      spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
    }
    */

    void *spinorOut0 = NULL, *spinorOutP = NULL;
    void **spinorOutMulti = NULL;

    if(myid==0){
      if(in.multi_shift){
	spinorOutMulti = (void**)malloc(in.num_offsets*sizeof(void *));
	for (int imass=0; imass<in.num_offsets; imass++) {
	  spinorOutMulti[imass] = malloc(V*spinorSiteSize*sSize);
	}
      }else{
	spinorOut0 = malloc(V*spinorSiteSize*sSize);
	spinorOutP = malloc(V*spinorSiteSize*sSize);
      }

      /* 
      // create a point source at 0 (in each subvolume...  FIXME)
      if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) *((float*)spinorIn) = 1.0;
      else *((double*)spinorIn) = 1.0;
      */

      // start the timer
      //double time0 = -((double)clock());

      // initialize the QUDA library
      // initQuda(device);

      // load the gauge field
      loadGaugeQuda((void*)gauge, &gauge_param);

      // load the clover term, if desired
      if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);
    }

    //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-
    int i,j,k,ispin;
    i=j=k=0;
    int index=0;
    double *sum=0; // effective mass
    if(myid==0){
      sum=(double *) malloc(gauge_param.X[3]*sizeof(double));
      for(i=0;i<gauge_param.X[3];i++) sum[i]=0.0;
    
      //complex<double>* prop_0 = (complex<double>*) malloc(Nv*144*sizeof(complex<double>));
      //complex<double>* prop_p = (complex<double>*) malloc(Nv*144*sizeof(complex<double>));

      cout<<"=-=-=-=-=-   preparation is done，begin to set src    -=-=-=-=-="<<endl;
    }
    for(int tsource=in.tsource_st; tsource<tdim; tsource+=in.tsource_in)
    for(int zsource=in.zsource_st; zsource<zdim; zsource+=in.zsource_in)
    for(int ysource=in.ysource_st; ysource<ydim; ysource+=in.ysource_in)
    for(int xsource=in.xsource_st; xsource<xdim; xsource+=in.xsource_in){
      if(myid==0){
	double *spinorIn_smear[3];
	if(SourceType==SmearedSource){
	  for(int ic=0;ic<3;ic++){
	    spinorIn_smear[ic] = new double[Vs*2*3*2];
	    smear_source(spinorIn_smear[ic],tsource,zsource,ysource,xsource,ic,&gauge_param,
			 &inv_param,in.N_smear,in.k_smear,gauge);
	  }
	}
	if(SourceType==RandomSource){
	    srand(unsigned(time(0)));
	    for(i=0;i<Vs*2;i++)
	      ((cdouble*)randomVector)[i] = Z3(random(1,3));
	}
	for(int isource=0;isource<12;isource++) //4spin3color
	{
	  if(SourceType==SmearedSource){
	    for(i=0;i<Vs*gauge_param.X[3]*2;i++)
	    for(ispin=0;ispin<24;ispin++)
	      if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
		*((float*)spinorIn0+i*24+ispin) = 0.0;
	      }else{
		*((double*)spinorIn0+i*24+ispin) = 0.0;
	      }

	    int color_source=isource%3;
	    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION)
	      for(i=0;i<Vs*2;i++)
	      for(int ic=0;ic<6;ic++){
		int xx[4];
		get_xyzt(i,xdim,xx);
		int ix = get_index_quda(tsource,xx[2],xx[1],xx[0],xdim);
		*((float*)spinorIn0+ix*24+isource/3*6+ic) = spinorIn_smear[color_source][i*6+ic];
	      }
	    else
	      for(i=0;i<Vs*2;i++)
	      for(int ic=0;ic<6;ic++){
		int xx[4];
		get_xyzt(i,xdim,xx);
		int ix = get_index_quda(tsource,xx[2],xx[1],xx[0],xdim);
		*((double*)spinorIn0+ix*24+isource/3*6+ic) = spinorIn_smear[color_source][i*6+ic];
	      }
	    if(isource==0&&tsource==in.tsource_st){
	      cout<<"=-=-=--=-=-=-   src is smeared src @ "<<tsource<< " time slice  =-=-=-=-=-=-="<<endl<<endl;
              cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-="<<endl<<endl;
	    }
	  }else if(SourceType==RandomSource){
	    if(inv_param.cpu_prec == QUDA_SINGLE_PRECISION)
	      for(i=tsource*Vs;i<(tsource+1)*Vs;i++){
		j = (i-tsource*Vs);
		*((float*)spinorIn0+i*24+isource*2) = ((double*)randomVector)[2*j];
		*((float*)spinorIn0+i*24+isource*2+1) = ((double*)randomVector)[2*j+1];
		j = (i-tsource*Vs)+Vs;
		*((float*)spinorIn0+(i+Vh)*24+isource*2) = ((double*)randomVector)[2*j];
		*((float*)spinorIn0+(i+Vh)*24+isource*2+1) = ((double*)randomVector)[2*j+1];
	      }
	    else
	      for(i=tsource*Vs;i<(tsource+1)*Vs;i++){
		j = (i-tsource*Vs);
		*((double*)spinorIn0+i*24+isource*2) = ((double*)randomVector)[2*j];
		*((double*)spinorIn0+i*24+isource*2+1) = ((double*)randomVector)[2*j+1];
		j = (i-tsource*Vs)+Vs;
		*((double*)spinorIn0+(i+Vh)*24+isource*2) = ((double*)randomVector)[2*j];
		*((double*)spinorIn0+(i+Vh)*24+isource*2+1) = ((double*)randomVector)[2*j+1];
	      }
	  }else{
	    set_source(SourceType,spinorIn0,tsource,zsource,ysource,xsource,isource,&gauge_param,&inv_param);
	  }

	  //rotate source
	  rotate_spinor(spinorIn0,1,0,inv_param.cpu_prec);

	  // perform the inversion
	  if (in.multi_shift) { 
	   invertMultiShiftQuda(spinorOutMulti, spinorIn0, &inv_param);
	  } else {
	    invertQuda(spinorOut0, spinorIn0, &inv_param);
	  }

	  // specify the output point 
	  int indx = in.indx[0]+in.indx[1]*xdim+in.indx[2]*xdim*ydim+in.indx[3]*xdim*ydim*zdim;
	  indx = indx/2+((in.indx[0]+in.indx[1]+in.indx[2]+in.indx[3])%2)*(xdim*ydim*zdim*tdim)/2;

	  // to be modified!
	  if(in.multi_shift)
	    {
	      //.......
	    } 
	  else{
	    //rotate sink
	    rotate_spinor(spinorOut0,-1,0,inv_param.cpu_prec);

	    double *prop0=(double*)spinorOut0;

	    // rescale
	    for(i=0;i<Vs*gauge_param.X[3]*2;i++)
	    for(ispin=0;ispin<24;ispin++){
		 prop0[i*24+ispin]*=2*in.kappa;
		 //propP[i*24+ispin]*=2*kappa;
	    }

	    if(in.ifverbose)
	      {
		for(i=0;i<Vs*gauge_param.X[3];i++){
		  int it=i/Vs;
		  for(ispin=0;ispin<24;ispin++)
		    sum[it]+=prop0[i*24+ispin]*prop0[i*24+ispin]
			    +prop0[Vs*gauge_param.X[3]*24+i*24+ispin]*
			     prop0[Vs*gauge_param.X[3]*24+i*24+ispin];
		}

		//output props of site indx
		printf("=======================================\n");
		printf("props of site: %4d %4d %4d %4d \n",in.indx[0],in.indx[1],in.indx[2],in.indx[3]);
		for(i=0;i<12;i++)
		  printf("%4d %15.7e %15.7e \n",i,
		         prop0[indx*24+2*i], prop0[indx*24+2*i+1]);
		//output PS twopt funcs & mass_eff for test use
		printf("=======================================\n");
		printf("two point function and effective mass\n");
		printf("%4d,%15.7e\n",0,sum[0]);
		for(i=1;i<gauge_param.X[3];i++)
		  printf("%4d,%15.7e,%15.7e\n",i,sum[i],log(sum[i-1]/sum[i]));
		printf("=======================================\n");
		fflush(stdout);
	      }

	    //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	    // resort the order and store
	    // prop[isite, is, ic, 2]; isource:is,ic
	    index = 0;
	    for(int it=0; it<tdim; it++)
	    for(int iz=0; iz<zdim; iz++)
	    for(int iy=0; iy<ydim; iy++)
	    for(int ix=0; ix<xdim; ix++){
	      int index_quda = (index)/2+(ix+iy+iz+it)%2*Vh;
	      for(k=0; k<12; k++){
	        // fortran: S_f[ic_sink,ic_source,is_sink,is_source,isite]
	        // here: S_f[isite,is_source,is_sink,ic_source,ic_sink]
	        *(sf[0]+index*288 +(isource/3)*72 +(k/3)*18 +(isource%3)*6 +(k%3)*2) = prop0[index_quda*24+2*k];
	        *(sf[0]+index*288 +(isource/3)*72 +(k/3)*18 +(isource%3)*6 +(k%3)*2+1) = prop0[index_quda*24+2*k+1];
	        // prop_0[isite,ic_sink,is_sink,ic_source,is_source]
	        // k/3: is_sink
	        // k%3: ic_sink
	        // isource/3: is_source
	        // isource%3: ic_source
	        // prop_0[j*144 +(k/3)*36 +(k%3)*12 +(isource/3)*3 +(isource%3)] = cdouble(prop0[j*24+2*k],prop0[j*24+2*k+1]);
	      }
	      index++;
	    }
	    //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	    /* 
	    //props output to a file result
	    char OutFile[100];
	    //sprintf(OutFile, "%sprop_%04d_psrc_%02d%02d%02d%02d_%02d", prop_path,icfg,tsource,zsource,ysource,xsource,isource);
	    sprintf(OutFile, "%sprop_%04d_psrc_%02d%02d%02d%02d", in.prop_path,icfg,tsource,zsource,ysource,xsource);
	    fstream myfile;

	    cout<<"the prop file is "<<OutFile<<endl;
	    if(isource==0)
	      myfile.open(OutFile,ios::binary|ios::out|ios::trunc);  
	    else
	      myfile.open(OutFile,ios::binary|ios::out|ios::app);  
	   
	    if(!myfile){
	      cerr<<"result file error!"<<endl;
	      exit(1);
	    }
	    //for(i=0;i<V*spinorSiteSize;i++)
	    //myfile.write((char*)&prop0[i],sizeof(double));
	    myfile.write((char*)prop0,sizeof(double)*V*spinorSiteSize);
	    myfile.close();

	    //------------------------------------------------ 
	    */
	    // create a seq source at time_seq
	    if(in.if_seq == 1){
	      cout<<"seq src begin!"<<endl;
	      /*
	      // specific mom of each direction
	      for(int ip=P_1; ip<=P_2; ip++)
	      for(int idr=dr_1; idr<=dr_2; idr++)
	      { 
		if(ip == 0) idr=dr_2; // 0 mom, 3 drs are equal.
		for(i=0;i<Vs*gauge_param.X[3]*2;i++)
		for(ispin=0;ispin<24;ispin++)
		  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
		    *((float*)spinorInP+i*24+ispin) = 0.0;
		  }
		  else{
		    *((double*)spinorInP+i*24+ispin) = 0.0;
		  }

		  for(int iz=0; iz<zdim; iz++)
		  for(int iy=0; iy<ydim; iy++)
		  for(int ix=0; ix<xdim; ix++)
		  for(int ics=0; ics<12; ics++){
		    int it=(time_seq+tsource)%tdim;
		    index = 24*(( ix+xdim*(iy+ydim*(iz+zdim* it)) )/2+(ix+iy+iz+it)%2*Vh);
		    double Px=0;
		    switch(idr){
		      case 1:{Px=ip*ix*2.0*3.14159265358979323/xdim;break;}
		      case 2:{Px=ip*iy*2.0*3.14159265358979323/ydim;break;}
		      case 3:{Px=ip*iz*2.0*3.14159265358979323/zdim;break;}
		    }
		    int ga5=(ics>5)?(-1):1;
		    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION)
		    {
		      *((float*)spinorInP+index+ics*2) = ga5*(prop0[index+ics*2]*cos(Px) + prop0[index+ics*2+1]*sin(Px));
		      *((float*)spinorInP+index+ics*2+1) = ga5*(-prop0[index+ics*2]*sin(Px) + prop0[index+ics*2+1]*cos(Px));
		    }
		    else
		    {
		      *((double*)spinorInP+index+ics*2) = ga5*(prop0[index+ics*2]*cos(Px) + prop0[index+ics*2+1]*sin(Px));
		      *((double*)spinorInP+index+ics*2+1) = ga5*(-prop0[index+ics*2]*sin(Px) + prop0[index+ics*2+1]*cos(Px));
		    }
		  }
	      }
	      */

	      // only care about the total mom

	      int icount = 0;
	      for(int ip=in.p_st;ip<=in.p_ed;ip++)
	      for(int ipx=0;ipx<=ip;ipx++)
	      for(int ipy=0;ipy<=ip;ipy++)
	      for(int ipz=0;ipz<=ip;ipz++)
	      if(ipx*ipx+ipy*ipy+ipz*ipz==ip){
		icount++;
		for(i=0;i<Vs*gauge_param.X[3]*2;i++)
		for(ispin=0;ispin<24;ispin++)
		if(inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
		  *((float*)spinorInP+i*24+ispin) = 0.0;
		}
		else{
		  *((double*)spinorInP+i*24+ispin) = 0.0;
		}

		for(int iz=0; iz<zdim; iz++)
		for(int iy=0; iy<ydim; iy++)
		for(int ix=0; ix<xdim; ix++)
		for(int ics=0; ics<12; ics++){
		  int it=(in.time_seq+tsource)%tdim;
		  index = 24*(( ix+xdim*(iy+ydim*(iz+zdim* it)) )/2+(ix+iy+iz+it)%2*Vh);
		  double Px=0;
		  Px=ipx*ix*2.0*3.14159265358979323/xdim
		    +ipy*iy*2.0*3.14159265358979323/ydim
		    +ipz*iz*2.0*3.14159265358979323/zdim;

		  //int ga5=(ics>5)?(-1):1;
		  int ga5=1;
		  if(inv_param.cpu_prec == QUDA_SINGLE_PRECISION)
		    {
		      *((float*)spinorInP+index+ics*2) = ga5*(prop0[index+ics*2]*cos(Px) + prop0[index+ics*2+1]*sin(Px));
		      *((float*)spinorInP+index+ics*2+1) = ga5*(-prop0[index+ics*2]*sin(Px) + prop0[index+ics*2+1]*cos(Px));
		    }
		    else
		    {
		      *((double*)spinorInP+index+ics*2) = ga5*(prop0[index+ics*2]*cos(Px) + prop0[index+ics*2+1]*sin(Px));
		      *((double*)spinorInP+index+ics*2+1) = ga5*(-prop0[index+ics*2]*sin(Px) + prop0[index+ics*2+1]*cos(Px));
		    }
		}
		
		//rotate source
		rotate_spinor(spinorInP,1,0,inv_param.cpu_prec);
		// perform the inversion
		invertQuda(spinorOutP, spinorInP, &inv_param);
		//rotate sink
		rotate_spinor(spinorOutP,-1,0,inv_param.cpu_prec);
	     
		double *propP=(double*)spinorOutP;
		for(i=0;i<Vs*gauge_param.X[3]*2;i++) for(ispin=0;ispin<24;ispin++) propP[i*24+ispin]*=2*in.kappa;

		if(in.ifverbose){
		    for(i=0;i<Vs*gauge_param.X[3];i++){
		    int it=i/Vs;
		    for(ispin=0;ispin<24;ispin++)
		      sum[it]+=propP[i*24+ispin]*propP[i*24+ispin]
			      +propP[Vs*gauge_param.X[3]*24+i*24+ispin]*
			       propP[Vs*gauge_param.X[3]*24+i*24+ispin];
		    }

		  //output props of site indx
		  printf("=======================================\n");
		  printf("props of site: %4d %4d %4d %4d \n",in.indx[0],in.indx[1],in.indx[2],in.indx[3]);
		  for(i=0;i<12;i++)
		    printf("%4d %15.7e %15.7e \n",i,
			   propP[indx*24+2*i], propP[indx*24+2*i+1]);
		  //output PS twopt funcs & mass_eff for test use
		   printf("=======================================\n");
		   printf("two point function and effective mass\n");
		   printf("%4d,%15.7e\n",0,sum[0]);
		   for(i=1;i<gauge_param.X[3];i++)
		     printf("%4d,%15.7e,%15.7e\n",i,sum[i],log(sum[i-1]/sum[i]));
		   printf("=======================================\n");
		   fflush(stdout);
		}

		//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
		// resort the order and store
		// prop[isite, is, ic, 2]; isource:is,ic
		index = 0;
		cout<<"icount="<<icount<<endl;
		for(int it=0; it<tdim; it++)
		for(int iz=0; iz<zdim; iz++)
		for(int iy=0; iy<ydim; iy++)
		for(int ix=0; ix<xdim; ix++){
		  int index_quda = (index)/2+(ix+iy+iz+it)%2*Vh;
		  for(k=0; k<12; k++){
		    // fortran: S_f[ic_sink,ic_source,is_sink,is_source,isite]
		    // here: S_f[isite,is_source,is_sink,ic_source,ic_sink]
		    *(sf[icount]+index*288 +(isource/3)*72 +(k/3)*18 +(isource%3)*6 +(k%3)*2) = propP[index_quda*24+2*k];
		    *(sf[icount]+index*288 +(isource/3)*72 +(k/3)*18 +(isource%3)*6 +(k%3)*2+1) = propP[index_quda*24+2*k+1];
		    // prop_0[isite,ic_sink,is_sink,ic_source,is_source]
		    // k/3: ic_sink
		    // k%3: is_sink
		    // isource/3: ic_source
		    // isource%3: is_source
		    // prop_0[j*144 +(k/3)*36 +(k%3)*12 +(isource/3)*3 +(isource%3)] = cdouble(prop0[j*24+2*k],prop0[j*24+2*k+1]);
		  }
		  index++;
		}
	        //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
		/*
		//sprintf(OutFile, "%sprop_%04d_seqsrc_%02d_%02d_%02d_%02d", prop_path,icfg,time_seq,idr,ip,isource);
		sprintf(OutFile, "%sprop_%04d_seqsrc_%02d_%1d%1d%1d", in.prop_path,icfg,in.time_seq,ipx,ipy,ipz);
		cout<<"the prop file is "<<OutFile<<endl;
		if(isource==0)
		  myfile.open(OutFile,ios::binary|ios::out|ios::trunc);  
		else
		  myfile.open(OutFile,ios::binary|ios::out|ios::app);  

		if(!myfile){cerr<<"result file error!"<<endl;exit(1);}
		//    for(i=0;i<V*spinorSiteSize;i++)
		//    myfile.write((char*)&propP[i],sizeof(double));
		myfile.write((char*)propP,sizeof(double)*V*spinorSiteSize);
		myfile.close();
		//props output to a file result
		*/
	      } // end of ip idr loop
	    } // end of if(if_seq == 1)
	    //------------------------------------------------
	    if(in.ifverbose)
	      cout<<"isource "<<isource<<" done!"<<endl;
	  } // else end of no multimass
	} //loop end of isource
     
        //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        /* 
        cout<<"begin to calc kernels..."<<endl;
        cdouble *kernel_1; // 0 momentum
        kernel_1 = (cdouble*) malloc(Nt*144*144*sizeof(cdouble));
        cdouble *kernel_2; // P momentum
        kernel_2 = (cdouble*) malloc(Nt*144*144*sizeof(cdouble));
        
        for(i=0; i<Nt*144*144; i++){
          kernel_1[i] = cdouble(0,0);
          kernel_2[i] = cdouble(0,0);
        }
        
        for(int it=0; it<Nt; it++)
        for(int iz=0; iz<Nz; iz++)
        for(int iy=0; iy<Ny; iy++)
        for(int ix=0; ix<Nx; ix++){
          int index = ix + Nx* (iy + Ny* (iz + Nz* it));
          index = index/2+((ix + iy + iz + it)%2)*(Ns*Ns*Ns*Nt)/2;
          for(int ics1=0; ics1<144; ics1++)
          for(int ics2=0; ics2<144; ics2++){
            kernel_1[it*144*144 + ics1*144 +ics2] += prop_0[index*144 + ics1] * conj(prop_0[index*144 + ics2]);
            //kernel_2[it*144*144 + ics1*144 +ics2] += prop_0[index*144 + ics1] *exp(cdouble(0,p*(ix-xsource)))* conj(prop_0[index*144 + ics2]);
          }
        }
        
        cout<<"kernels calced!"<<endl;
        
        fstream myfile;
        myfile.open(filename,ios::binary|ios::out|ios::trunc);  
        if(!myfile){
          cerr<<"result file error!"<<endl;
          exit(1);
        }
        for(i=0;i<Nt*144*144;i++)
        myfile.write((char*)&kernel_1[i],sizeof(cdouble));
        myfile.close();
        
        free(kernel_1);
        myfile.open(filename,ios::binary|ios::out|ios::trunc);  
        if(!myfile){
          cerr<<"result file error!"<<endl;
          exit(1);
        }
        for(i=0;i<Nt*144*144;i++)
        myfile.write((char*)&kernel_2[i],sizeof(cdouble));
        myfile.close();
        
        free(kernel_2);
        */
        cout<<"tsource "<<tsource<<" ";
        cout<<"zsource "<<zsource<<" ";
        cout<<"ysource "<<ysource<<" ";
        cout<<"xsource "<<xsource<<" done!"<<endl;
      } // end of myid==0
      MPI_Barrier(MPI_COMM_WORLD);

      //twopt:============================================================================
      double* u = 0;
      if(in.ifB==1){
	if(myid==0)
	  {
	    u = (double*)malloc(V*4*18*sizeof(double));
	  }
	else
	  {
	    u = (double*)malloc(ntime*Vs*2*4*18*sizeof(double));
	  }
	if(myid==0)
	  {
	    // ir_quda(0:3->xyzt) ir_new(0:3->txyz)
	    int index = 0;
	    for(int it=0; it<tdim; it++)
	    for(int iz=0; iz<zdim; iz++)
	    for(int iy=0; iy<ydim; iy++)
	    for(int ix=0; ix<xdim; ix++){
	      int index_quda = (index)/2+(ix+iy+iz+it)%2*Vh;
	      for(int ir=0; ir<4; ir++)
	      for(int ic=0; ic<18; ic++){
		int ic1=ic%6/2;
		int ic2=ic/6;
		int ic_n=ic%2+ic2*2+ic1*6;
		u[index*4*18 + (ir+1)%4*18 + ic_n] = ((double*)(gauge[ir]))[index_quda*18 + ic];
	      }
	      index++;
	    }
	  }
#ifdef mpi
	for(int it=ntime;it<tdim;it+=ntime){
	  if(myid==0){
	    MPI_Send(u+it*Vs*2*72,ntime*Vs*2*72,MPI_DOUBLE,it/ntime,0,MPI_COMM_WORLD);
	  }
	  if(myid==it/ntime){ 
	    MPI_Recv(u,ntime*Vs*2*72,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&mpi_error);
	  }
	}
#endif
      }

#ifdef mpi
      for(int it=ntime;it<tdim;it+=ntime){
	if(myid==0){
	  MPI_Send(sf[0]+it*Vs*2*288,ntime*Vs*2*288,MPI_DOUBLE,it/ntime,0,MPI_COMM_WORLD);
	}
	if(myid==it/ntime){ 
	  MPI_Recv(sf[0],ntime*Vs*2*288,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&mpi_error);
	}
      }
      MPI_Barrier(MPI_COMM_WORLD);
#endif

      char OutFile[100];
      fstream myfile;

      if(in.ifC==1){
	double*twopf = (double*)malloc(ntime*np*sizeof(double)*2);
	double*meson = 0;
	if(myid==0) meson = (double*)malloc(tdim*np*sizeof(double)*2);
	two_point_(sf[0],(double*)gauge[0],&xdim,&np,twopf,&ntime);
#ifdef mpi
	MPI_Gather(twopf,ntime*np*2,MPI_DOUBLE,meson,ntime*np*2,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	free(twopf);
	if(myid==0)
	  {
	    double*meson_eq = (double*)malloc(ncores*np*ntime*sizeof(double)*2);
	    for(int it=0;it<tdim;it++)
	      for(int ip=0;ip<np;ip++)
		{
		  int it_new = (it-tsource+tdim)%tdim;
		  meson_eq[ip*tdim*2+it_new*2]=meson[it*np*2+ip*2];
		  meson_eq[ip*tdim*2+it_new*2+1]=meson[it*np*2+ip*2+1];
		}
	    for(int ip=0;ip<np;ip++)
	      {
		meson[ip*tdim*2+0]=meson_eq[ip*tdim*2+0];
		meson[ip*tdim*2+1]=meson_eq[ip*tdim*2+1];
		meson[ip*tdim*2+96]=meson_eq[ip*tdim*2+48*2];
		meson[ip*tdim*2+97]=meson_eq[ip*tdim*2+48*2+1];
		for(int it=1;it<tdim/2;it++)
		  {
		    meson[ip*tdim*2+it*2]=meson[ip*tdim*2+(tdim-it)*2]=
		    (meson_eq[ip*tdim*2+it*2]+meson_eq[ip*tdim*2+(tdim-it)*2])/2;
		    meson[ip*tdim*2+it*2+1]=meson[ip*tdim*2+(tdim-it)*2+1]=
		    (meson_eq[ip*tdim*2+it*2+1]+meson_eq[ip*tdim*2+(tdim-it)*2+1])/2;
		  }
	      }
	    if(in.ifverbose){
	      for(int i=0;i<tdim/2;i++) cout<<i<<'\t'<<meson_eq[2*i]<<'\t'<<meson[2*i]<<endl;
	      cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	      for(int i=0;i<tdim;i++) cout<<i<<'\t'<<meson_eq[2*i+1]<<'\t'<<meson[2*i+1]<<endl;
	      cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	    }
	    sprintf(OutFile, "%stwopf.%04d", in.prop_path,icfg);
	    cout<<"the twopf file is "<<OutFile<<endl;
	    if(tsource==0){
	      myfile.open(OutFile,ios::binary|ios::out|ios::trunc);
	    }
	    else{
	      myfile.open(OutFile,ios::binary|ios::out|ios::app);
	    } 
	    if(!myfile){cerr<<"result file error!"<<endl;exit(1);}
	    myfile.write((char*)meson,sizeof(double)*np*tdim*2);
	    myfile.close();
	    free(meson_eq);
	    free(meson);
	  }
      }
      if(in.ifB==1){
	int r = 0;
	double*twopf_B = (double*)malloc(ntime*sizeof(double)*2);
	double*meson_B = 0;
	if(myid==0) meson_B = (double*)malloc(tdim*sizeof(double)*2);
	two_point_b_(sf[0],u,&xdim,&r,twopf_B,&ntime);
#ifdef mpi
	MPI_Gather(twopf_B,ntime*2,MPI_DOUBLE,meson_B,ntime*2,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	free(twopf_B);
	if(myid==0)
	  {
	    double*meson_eq = (double*)malloc(ncores*ntime*sizeof(double)*2);
	    for(int it=0;it<tdim;it++)
	      {
		int it_new = (it-tsource+tdim)%tdim;
		meson_eq[it_new*2]=meson_B[it*2];
		meson_eq[it_new*2+1]=meson_B[it*2+1];
	      }

	    meson_B[0]=meson_eq[0*2+0];
	    meson_B[1]=meson_eq[0*2+1];
	    meson_B[48*2]=meson_eq[48*2];
	    meson_B[48*2+1]=meson_eq[48*2+1];
	    for(int it=1;it<tdim/2;it++)
	      {
		meson_B[it*2]=meson_B[(tdim-it)*2]=
		(meson_eq[it*2]-meson_eq[(tdim-it)*2])/2;
		meson_B[it*2+1]=meson_B[(tdim-it)*2+1]=
		(meson_eq[it*2+1]-meson_eq[(tdim-it)*2+1])/2;
	      }

	    if(in.ifverbose){
	      for(int i=0;i<tdim/2;i++) cout<<i<<'\t'<<meson_B[2*i]<<endl;
	      for(int i=0;i<tdim/2;i++) cout<<i<<'\t'<<meson_B[2*i+1]<<endl;
	      cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	    }
	    sprintf(OutFile, "%stwopf_B.%04d", in.prop_path,icfg);
	    cout<<"the twopf_B file is "<<OutFile<<endl;
	    if(tsource==0){
	      myfile.open(OutFile,ios::binary|ios::out|ios::trunc);
	    }
	    else{
	      myfile.open(OutFile,ios::binary|ios::out|ios::app);
	    } 
	    if(!myfile){cerr<<"result file error!"<<endl;exit(1);}
	    myfile.write((char*)meson_B,sizeof(double)*tdim*2);
	    myfile.close();
	    free(meson_eq);
	    free(meson_B);
	  }
	  free(u);
      }

      //threept:==========================================================================
      if(in.if_seq==1)
	{
	  if(myid==0)
	    {
	      u = (double*)malloc(V*4*18*sizeof(double));
	    }
	  else
	    {
	      u = (double*)malloc(ntime*Vs*2*4*18*sizeof(double));
	    }
	  if(myid==0)
	    {
	      // ir_quda(0:3->xyzt) ir_new(0:3->txyz)
	      int index = 0;
	      for(int it=0; it<tdim; it++)
	      for(int iz=0; iz<zdim; iz++)
	      for(int iy=0; iy<ydim; iy++)
	      for(int ix=0; ix<xdim; ix++){
		int index_quda = (index)/2+(ix+iy+iz+it)%2*Vh;
		for(int ir=0; ir<4; ir++)
		for(int ic=0; ic<18; ic++){
		  int ic1=ic%6/2;
		  int ic2=ic/6;
		  int ic_n=ic%2+ic2*2+ic1*6;
		  u[index*4*18 + (ir+1)%4*18 + ic_n] = ((double*)(gauge[ir]))[index_quda*18 + ic];
		}
		index++;
	      }
	    }
#ifdef mpi
	  for(int it=ntime;it<tdim;it+=ntime){
	    if(myid==0){
	      MPI_Send(u+it*Vs*2*72,ntime*Vs*2*72,MPI_DOUBLE,it/ntime,0,MPI_COMM_WORLD);
	    }
	    if(myid==it/ntime){ 
	      MPI_Recv(u,ntime*Vs*2*72,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&mpi_error);
	    }
	  }
	  for(int icount=1;icount<=nsf;icount++)
	    for(int it=ntime;it<tdim;it+=ntime){
	      if(myid==0){
		MPI_Send(sf[icount]+it*Vs*2*288,ntime*Vs*2*288,MPI_DOUBLE,it/ntime,0,MPI_COMM_WORLD);
	      }
	      if(myid==it/ntime){ 
		MPI_Recv(sf[icount],ntime*Vs*2*288,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&mpi_error);
	      }
	    }
	  MPI_Barrier(MPI_COMM_WORLD);
#endif

	  double*threepf1 = (double*)malloc(ntime*np*nr*sizeof(double)*4);
	  double*threepf2 = (double*)malloc(ntime*np*nr*sizeof(double)*4);
	  for(int i=0;i<ntime*np*nr*4;i++) threepf1[i] = 0.0;
	  for(int i=0;i<ntime*np*nr*4;i++) threepf2[i] = 0.0;
	  double*hadron = 0;
	  if(myid==0) hadron = (double*)malloc(tdim*np*nr*sizeof(double)*4);

	  int ip=0;
	  int ir=0;
	  int ipx=0,ipy=0,ipz=0;
	  int icount1 = 0; 
	  int icount2 = 0; 
	  for(ip=in.p_st;ip<=in.p_ed;ip++)
	  {
	    icount1=0;
	    for(ipx=0;ipx<=ip;ipx++)
	    for(ipy=0;ipy<=ip;ipy++)
	    for(ipz=0;ipz<=ip;ipz++){
	      if(ipx*ipx +ipy*ipy +ipz*ipz == ip){
		icount1 ++;
		icount2 ++;
		for(ir=in.dr_st;ir<=in.dr_ed;ir++){
		  three_point_(sf[0],sf[icount2],u,&xdim,&ir,
		  threepf1+((ip-in.p_st)*nr+(ir-in.dr_st))*ntime*4,&ntime,&myid);
		}
		for(int ixx=0;ixx<nr*ntime*4;ixx++){ 
		  threepf2[(ip-in.p_st)*nr*ntime*4 +ixx]
		  += threepf1[(ip-in.p_st)*nr*ntime*4 +ixx];
		}
	      }
	    }
	    for(int ixx=0;ixx<nr*ntime*4;ixx++) 
	      threepf1[(ip-in.p_st)*nr*ntime*4 +ixx]=
	      threepf2[(ip-in.p_st)*nr*ntime*4 +ixx]/icount1;
	  }
#ifdef mpi
	  MPI_Gather(threepf1,ntime*np*nr*4,MPI_DOUBLE,hadron,ntime*np*nr*4,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	  free(threepf1);
	  free(threepf2);
	  if(myid==0)
	    {
	      double*hadron_eq = (double*)malloc(np*nr*tdim*sizeof(double)*4);
	      for(int id=0;id<ncores;id++)
	      for(int ip=0;ip<np;ip++)
	      for(int ir=0;ir<nr;ir++)
	      for(int it=0;it<ntime;it++)
	      for(int ik=0;ik<2;ik++)
		{
		  int it_new = ((id*ntime+it)-tsource+tdim)%tdim;
		  hadron_eq[(((ik*np+ip)*nr+ir)*tdim+it_new)*2+0]=hadron[(((id*np+ip)*nr+ir)*ntime+it)*4+ik*2+0];
		  hadron_eq[(((ik*np+ip)*nr+ir)*tdim+it_new)*2+1]=hadron[(((id*np+ip)*nr+ir)*ntime+it)*4+ik*2+1];
		}
	      if(in.ifverbose){
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[2*i]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[nr*tdim*2+2*i]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[2*nr*tdim*2+2*i]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[3*nr*tdim*2+2*i]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[tdim*2+2*i]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[np*nr*tdim*2+2*i+1]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[np*nr*tdim*2+nr*tdim*2+2*i+1]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[np*nr*tdim*2+2*nr*tdim*2+2*i+1]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[np*nr*tdim*2+3*nr*tdim*2+2*i+1]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	        for(int i=in.time_seq;i<tdim/2;i++) cout<<i<<'\t'<<hadron_eq[np*nr*tdim*2+tdim*2+2*i+1]<<endl;
	        cout<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="<<endl;
	      }
	      sprintf(OutFile, "%sthreepf.%04d", in.prop_path,icfg);
	      cout<<"the twopf file is "<<OutFile<<endl;
	      if(tsource==0){
		myfile.open(OutFile,ios::binary|ios::out|ios::trunc);
	      }
	      else{
		myfile.open(OutFile,ios::binary|ios::out|ios::app);
	      } 
	      if(!myfile){cerr<<"result file error!"<<endl;exit(1);}
	      myfile.write((char*)hadron_eq,sizeof(double)*nr*np*tdim*4);
	      myfile.close();
	      free(hadron_eq);
	      free(hadron);
	  }
	  free(u);
	}
    } // loop end of tzyx source
    //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    if(myid==0){
      cout<<endl<<"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-="<<endl<<endl;
      //free(prop_0);
      //free(prop_p);
      free(sum);
      // stop the timer
      time0 += clock();
      time0 /= CLOCKS_PER_SEC;

      printfQuda("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
		 inv_param.spinorGiB, gauge_param.gaugeGiB);
      if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) printfQuda("   Clover: %f GiB\n", inv_param.cloverGiB);
      printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	     inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

      if (in.multi_shift){
	void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

	printfQuda("Host residuum checks: \n");
	for(int i=0; i < inv_param.num_offset; i++) {
	  ax(0, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	  
	  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	    tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
		     inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	    tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
		     inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	  } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	    wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
		      inv_param.cpu_prec, gauge_param);
	    wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
		      inv_param.cpu_prec, gauge_param);
	  } else {
	    printfQuda("Domain wall not supported for multi-shift\n");
	    exit(-1);
	  }

	  axpy(inv_param.offset[i], spinorOutMulti[i], spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	  mxpy(spinorIn0, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	  double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	  double src2 = norm_2(spinorIn0, V*spinorSiteSize, inv_param.cpu_prec);
	  double l2r = sqrt(nrm2 / src2);

	  printfQuda("Shift i=%d residuals: requested %g; relative QUDA = %g, host = %g; heavy-quark QUDA = %g\n",
		     i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, 
		     inv_param.true_res_hq_offset[i]);
	}
	free(spinorTmp);
      } 
      else {
	if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

	  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	    tm_mat(spinorCheck, gauge, spinorOut0, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
		       0, inv_param.cpu_prec, gauge_param); 
	  } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	    wil_mat(spinorCheck, gauge, spinorOut0, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
	  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	    dw_mat(spinorCheck, gauge, spinorOut0, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
	  } else {
	    printfQuda("Unsupported dslash_type\n");
	    exit(-1);
	  }
	  if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	      ax(0.5/kappa5, spinorCheck, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
	    } else {
	      ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	    }
	  }

	} else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {

	  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	    tm_matpc(spinorCheck, gauge, spinorOut0, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
		     inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
	    wil_matpc(spinorCheck, gauge, spinorOut0, inv_param.kappa, inv_param.matpc_type, 0, 
		      inv_param.cpu_prec, gauge_param);
	  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	    dw_matpc(spinorCheck, gauge, spinorOut0, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
	  } else {
	    printfQuda("Unsupported dslash_type\n");
	    exit(-1);
	  }

	  if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	      ax(0.25/(kappa5*kappa5), spinorCheck, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
	    } else {
	      ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	  
	    }
	  }
	}
	mxpy(spinorIn0, spinorCheck, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
	double nrm2 = norm_2(spinorCheck, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
	double src2 = norm_2(spinorIn0, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
	double l2r = sqrt(nrm2 / src2);

	printfQuda("Residuals: requested %g; relative QUDA = %g, host = %g; heavy-quark QUDA = %g\n",
		   inv_param.tol, inv_param.true_res, l2r, inv_param.true_res_hq);
      }

      // finalize the QUDA library
      //endQuda();

      cout<<"cfg "<<icfg<<" done~~~~~~~"<<endl<<endl;
      free(clover_inv);
      free(spinorIn0);
      free(spinorInP);
      free(spinorCheck);
      free(spinorOut0);
      free(spinorOutP);
      free(randomVector);
      //~~~~~~~~
      freeGaugeQuda();
      freeCloverQuda();
    }
  } // loop end of icfg
  if(myid==0){
    // finalize the QUDA library
    endQuda();
  }

#ifdef mpi
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  if(myid == 0) cout<<"=========================================="<<endl;
  return 0;
}

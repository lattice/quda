#ifndef _TWIST_QUDA_CONTRACT
#define _TWIST_QUDA_CONTRACT

#define tmp_re tmp.x
#define tmp_im tmp.y

#define TOTAL_COMPONENTS 16 

#if (__COMPUTE_CAPABILITY__ >= 130)

#define READ_INTERMEDIATE_SPINOR_DOUBLE(spinor, stride, sp_idx, norm_idx)	   \
  double2 J0	 = spinor[sp_idx + 0*(stride)];   \
  double2 J1	 = spinor[sp_idx + 1*(stride)];   \
  double2 J2	 = spinor[sp_idx + 2*(stride)];   \
  double2 J3	 = spinor[sp_idx + 3*(stride)];   \
  double2 J4	 = spinor[sp_idx + 4*(stride)];   \
  double2 J5	 = spinor[sp_idx + 5*(stride)];   \
  double2 J6	 = spinor[sp_idx + 6*(stride)];   \
  double2 J7	 = spinor[sp_idx + 7*(stride)];   \
  double2 J8	 = spinor[sp_idx + 8*(stride)];   \
  double2 J9	 = spinor[sp_idx + 9*(stride)];   \
  double2 J10	 = spinor[sp_idx +10*(stride)];   \
  double2 J11	 = spinor[sp_idx +11*(stride)];

#define READ_INTERMEDIATE_SPINOR_DOUBLE_TEX(spinor, stride, sp_idx, norm_idx)	\
  double2 J0	 = fetch_double2((spinor), sp_idx + 0*(stride));   \
  double2 J1	 = fetch_double2((spinor), sp_idx + 1*(stride));   \
  double2 J2	 = fetch_double2((spinor), sp_idx + 2*(stride));   \
  double2 J3	 = fetch_double2((spinor), sp_idx + 3*(stride));   \
  double2 J4	 = fetch_double2((spinor), sp_idx + 4*(stride));   \
  double2 J5	 = fetch_double2((spinor), sp_idx + 5*(stride));   \
  double2 J6	 = fetch_double2((spinor), sp_idx + 6*(stride));   \
  double2 J7	 = fetch_double2((spinor), sp_idx + 7*(stride));   \
  double2 J8	 = fetch_double2((spinor), sp_idx + 8*(stride));   \
  double2 J9	 = fetch_double2((spinor), sp_idx + 9*(stride));   \
  double2 J10	 = fetch_double2((spinor), sp_idx +10*(stride));   \
  double2 J11	 = fetch_double2((spinor), sp_idx +11*(stride));

#ifdef DIRECT_ACCESS_WILSON_SPINOR
	#define READ_SPINOR READ_SPINOR_DOUBLE
	#define READ_INTERMEDIATE_SPINOR READ_INTERMEDIATE_SPINOR_DOUBLE
	#define SPINORTEX in
	#define INTERTEX in2
#else
	#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
	#define READ_INTERMEDIATE_SPINOR READ_INTERMEDIATE_SPINOR_DOUBLE_TEX

	#ifdef USE_TEXTURE_OBJECTS
		#define SPINORTEX param.inTex
		#define INTERTEX param.outTex
	#else
		#define SPINORTEX spinorTexDouble
		#define INTERTEX interTexDouble
	#endif	// USE_TEXTURE_OBJECTS

#define	SPINOR_HOP	12

#endif

__global__ void contractGamma5Kernel(double2 *out, double2 *in1, double2 *in2, int myStride, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= param.threads)
		return;

	volatile double2		tmp;
	extern __shared__ double	sm[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile double			*accum_re = sm + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile double			*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / X1;
	xCoord1		 = eutId - auxCoord1 * X1;
	auxCoord2	 = auxCoord1 / X2;
	xCoord2		 = auxCoord1 - auxCoord2 * X2;
	xCoord4		 = auxCoord2 / X3;
	xCoord3		 = auxCoord2 - xCoord4 * X3;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + X1*(xCoord2 + X2*(xCoord3 + X3*xCoord4));			//AQUI

	READ_SPINOR			(SPINORTEX, myStride, sid, sid);
	READ_INTERMEDIATE_SPINOR	(INTERTEX,  myStride, sid, sid);
	
	//compute in1^dag * gamma5:

	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I6.x;
	I0.y	 = -I6.y;
	I6.x	 = tmp_re;
	I6.y	 = tmp_im;	

	tmp_re	 = +I3.x;
	tmp_im	 = -I3.y;
	I3.x	 = +I9.x;
	I3.y	 = -I9.y;
	I9.x	 = tmp_re;
	I9.y	 = tmp_im;	

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I7.x;
	I1.y	 = -I7.y;
	I7.x	 = tmp_re;
	I7.y	 = tmp_im;	

	tmp_re	 = +I4.x;
	tmp_im	 = -I4.y;
	I4.x	 = +I10.x;
	I4.y	 = -I10.y;
	I10.x	 = tmp_re;
	I10.y	 = tmp_im;

	tmp_re	 = +I2.x;
	tmp_im	 = -I2.y;
	I2.x	 = +I8.x;
	I2.y	 = -I8.y;
	I8.x	 = tmp_re;
	I8.y	 = tmp_im;	

	tmp_re	 = +I5.x;
	tmp_im	 = -I5.y;
	I5.x	 = +I11.x;
	I5.y	 = -I11.y;
	I11.x	 = tmp_re;
	I11.y	 = tmp_im;	

	//do products for all color component here:
	//00 component:
	tmp_re	 =  I0.x *  J0.x -  I0.y *  J0.y;
	tmp_re	+=  I1.x *  J1.x -  I1.y *  J1.y;
	tmp_re	+=  I2.x *  J2.x -  I2.y *  J2.y;
	accum_re[0*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J0.y +  I0.y *  J0.x;	
	tmp_im	+=  I1.x *  J1.y +  I1.y *  J1.x;	
	tmp_im	+=  I2.x *  J2.y +  I2.y *  J2.x;	
	accum_im[0*blockDim.x]	= tmp_im; 	

	//01 component:
	tmp_re	 =  I0.x *  J3.x -  I0.y *  J3.y;
	tmp_re	+=  I1.x *  J4.x -  I1.y *  J4.y;
	tmp_re	+=  I2.x *  J5.x -  I2.y *  J5.y;
	accum_re[1*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J3.y +  I0.y *  J3.x;	
	tmp_im	+=  I1.x *  J4.y +  I1.y *  J4.x;	
	tmp_im	+=  I2.x *  J5.y +  I2.y *  J5.x;	
	accum_im[1*blockDim.x]	= tmp_im; 	

	//02 component:
	tmp_re	 =  I0.x *  J6.x -  I0.y *  J6.y;
	tmp_re	+=  I1.x *  J7.x -  I1.y *  J7.y;
	tmp_re	+=  I2.x *  J8.x -  I2.y *  J8.y;
	accum_re[2*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J6.y +  I0.y *  J6.x;	
	tmp_im	+=  I1.x *  J7.y +  I1.y *  J7.x;	
	tmp_im	+=  I2.x *  J8.y +  I2.y *  J8.x;	
	accum_im[2*blockDim.x]	= tmp_im;	

	//03 component:
	tmp_re	 =  I0.x *  J9.x -  I0.y *  J9.y;
	tmp_re	+=  I1.x * J10.x -  I1.y * J10.y;
	tmp_re	+=  I2.x * J11.x -  I2.y * J11.y;
	accum_re[3*blockDim.x]	= tmp_re;

	tmp_im	 =  I0.x *  J9.y +  I0.y *  J9.x;	
	tmp_im	+=  I1.x * J10.y +  I1.y * J10.x;	
	tmp_im	+=  I2.x * J11.y +  I2.y * J11.x;	
	accum_im[3*blockDim.x]	= tmp_im;	

	//10 component:
	tmp_re	 =  I3.x *  J0.x -  I3.y *  J0.y;
	tmp_re	+=  I4.x *  J1.x -  I4.y *  J1.y;
	tmp_re	+=  I5.x *  J2.x -  I5.y *  J2.y;
	accum_re[ 4*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J0.y +  I3.y *  J0.x;	
	tmp_im	+=  I4.x *  J1.y +  I4.y *  J1.x;	
	tmp_im	+=  I5.x *  J2.y +  I5.y *  J2.x;	
	accum_im[ 4*blockDim.x]	= tmp_im; 	

	//11 component:
	tmp_re	 =  I3.x *  J3.x -  I3.y *  J3.y;
	tmp_re	+=  I4.x *  J4.x -  I4.y *  J4.y;
	tmp_re	+=  I5.x *  J5.x -  I5.y *  J5.y;
	accum_re[ 5*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J3.y +  I3.y *  J3.x;	
	tmp_im	+=  I4.x *  J4.y +  I4.y *  J4.x;	
	tmp_im	+=  I5.x *  J5.y +  I5.y *  J5.x;	
	accum_im[ 5*blockDim.x]	= tmp_im; 	

	//12 component:
	tmp_re	 =  I3.x *  J6.x -  I3.y *  J6.y;
	tmp_re	+=  I4.x *  J7.x -  I4.y *  J7.y;
	tmp_re	+=  I5.x *  J8.x -  I5.y *  J8.y;
	accum_re[ 6*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J6.y +  I3.y *  J6.x;	
	tmp_im	+=  I4.x *  J7.y +  I4.y *  J7.x;	
	tmp_im	+=  I5.x *  J8.y +  I5.y *  J8.x;	
	accum_im[ 6*blockDim.x]	= tmp_im;	

	//13 component:
	tmp_re	 =  I3.x *  J9.x -  I3.y *  J9.y;
	tmp_re	+=  I4.x * J10.x -  I4.y * J10.y;
	tmp_re	+=  I5.x * J11.x -  I5.y * J11.y;
	accum_re[ 7*blockDim.x]	= tmp_re;

	tmp_im	 =  I3.x *  J9.y +  I3.y *  J9.x;	
	tmp_im	+=  I4.x * J10.y +  I4.y * J10.x;	
	tmp_im	+=  I5.x * J11.y +  I5.y * J11.x;	
	accum_im[ 7*blockDim.x]	= tmp_im;	

	//20 component:
	tmp_re	 =  I6.x *  J0.x -  I6.y *  J0.y;
	tmp_re	+=  I7.x *  J1.x -  I7.y *  J1.y;
	tmp_re	+=  I8.x *  J2.x -  I8.y *  J2.y;
	accum_re[ 8*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J0.y +  I6.y *  J0.x;	
	tmp_im	+=  I7.x *  J1.y +  I7.y *  J1.x;	
	tmp_im	+=  I8.x *  J2.y +  I8.y *  J2.x;	
	accum_im[ 8*blockDim.x]	= tmp_im; 	

	//21 component:
	tmp_re	 =  I6.x *  J3.x -  I6.y *  J3.y;
	tmp_re	+=  I7.x *  J4.x -  I7.y *  J4.y;
	tmp_re	+=  I8.x *  J5.x -  I8.y *  J5.y;
	accum_re[ 9*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J3.y +  I6.y *  J3.x;	
	tmp_im	+=  I7.x *  J4.y +  I7.y *  J4.x;	
	tmp_im	+=  I8.x *  J5.y +  I8.y *  J5.x;	
	accum_im[ 9*blockDim.x]	= tmp_im; 	

	//22 component:
	tmp_re	 =  I6.x *  J6.x -  I6.y *  J6.y;
	tmp_re	+=  I7.x *  J7.x -  I7.y *  J7.y;
	tmp_re	+=  I8.x *  J8.x -  I8.y *  J8.y;
	accum_re[10*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J6.y +  I6.y *  J6.x;	
	tmp_im	+=  I7.x *  J7.y +  I7.y *  J7.x;	
	tmp_im	+=  I8.x *  J8.y +  I8.y *  J8.x;	
	accum_im[10*blockDim.x]	= tmp_im;	

	//23 component:
	tmp_re	 =  I6.x *  J9.x -  I6.y *  J9.y;
	tmp_re	+=  I7.x * J10.x -  I7.y * J10.y;
	tmp_re	+=  I8.x * J11.x -  I8.y * J11.y;
	accum_re[11*blockDim.x]	= tmp_re;

	tmp_im	 =  I6.x *  J9.y +  I6.y *  J9.x;	
	tmp_im	+=  I7.x * J10.y +  I7.y * J10.x;	
	tmp_im	+=  I8.x * J11.y +  I8.y * J11.x;	
	accum_im[11*blockDim.x]	= tmp_im;	

	//30 component:
	tmp_re	 =  I9.x *  J0.x -  I9.y *  J0.y;
	tmp_re	+= I10.x *  J1.x - I10.y *  J1.y;
	tmp_re	+= I11.x *  J2.x - I11.y *  J2.y;
	accum_re[12*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J0.y +  I9.y *  J0.x;	
	tmp_im	+= I10.x *  J1.y + I10.y *  J1.x;	
	tmp_im	+= I11.x *  J2.y + I11.y *  J2.x;	
	accum_im[12*blockDim.x]	= tmp_im; 	

	//31 component:
	tmp_re	 =  I9.x *  J3.x -  I9.y *  J3.y;
	tmp_re	+= I10.x *  J4.x - I10.y *  J4.y;
	tmp_re	+= I11.x *  J5.x - I11.y *  J5.y;
	accum_re[13*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J3.y +  I9.y *  J3.x;	
	tmp_im	+= I10.x *  J4.y + I10.y *  J4.x;	
	tmp_im	+= I11.x *  J5.y + I11.y *  J5.x;	
	accum_im[13*blockDim.x]	= tmp_im; 	

	//32 component:
	tmp_re	 =  I9.x *  J6.x -  I9.y *  J6.y;
	tmp_re	+= I10.x *  J7.x - I10.y *  J7.y;
	tmp_re	+= I11.x *  J8.x - I11.y *  J8.y;
	accum_re[14*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J6.y +  I9.y *  J6.x;	
	tmp_im	+= I10.x *  J7.y + I10.y *  J7.x;	
	tmp_im	+= I11.x *  J8.y + I11.y *  J8.x;	
	accum_im[14*blockDim.x]	= tmp_im;	

	//33 component:
	tmp_re	 =  I9.x *  J9.x -  I9.y *  J9.y;
	tmp_re	+= I10.x * J10.x - I10.y * J10.y;
	tmp_re	+= I11.x * J11.x - I11.y * J11.y;
	accum_re[15*blockDim.x]	= tmp_re;

	tmp_im	 =  I9.x *  J9.y +  I9.y *  J9.x;	
	tmp_im	+= I10.x * J10.y + I10.y * J10.x;	
	tmp_im	+= I11.x * J11.y + I11.y * J11.x;	
	accum_im[15*blockDim.x]	= tmp_im;	

   //Store output back to global buffer:


/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *param.threads*2]	 = make_double2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *param.threads*2]	 = make_double2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *param.threads*2]	 = make_double2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *param.threads*2]	 = make_double2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *param.threads*2]	 = make_double2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *param.threads*2]	 = make_double2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *param.threads*2]	 = make_double2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *param.threads*2]	 = make_double2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *param.threads*2]	 = make_double2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *param.threads*2]	 = make_double2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*param.threads*2]	 = make_double2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*param.threads*2]	 = make_double2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*param.threads*2]	 = make_double2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*param.threads*2]	 = make_double2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*param.threads*2]	 = make_double2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*param.threads*2]	 = make_double2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	return;
}

//Perform trace in color space only and for a given tslice 
//since the file is included in dslash_quda.h, no need to add dslash_constants.h file here (for, e.g., Vsh)
__global__ void contractTsliceKernel(double2 *out, double2 *in1, double2 *in2, int myStride, const int Tslice, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;					//number of threads is equal to Tslice volume
												//Adjust sid to correct tslice (exe domain must be Tslice volume!)
	int	inId	 = sid + Vsh*Tslice;							//Vsh - 3d space volume for the parity spinor (equale to exe domain!)
	int	outId; 
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= param.threads)								//param.threads == tslice volume
		return;

	volatile double2		tmp;
	extern __shared__ double	sm[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile double			*accum_re = sm + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile double			*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

//The output only for a given tslice (for the full tslice content, i.e., both parities!):

	eutId		 = 2*inId;
	auxCoord1	 = eutId / X1;
	xCoord1		 = eutId - auxCoord1 * X1;
	auxCoord2	 = auxCoord1 / X2;
	xCoord2		 = auxCoord1 - auxCoord2 * X2;
	xCoord4		 = auxCoord2 / X3;
	xCoord3		 = auxCoord2 - xCoord4 * X3;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + X1*(xCoord2 + X2*xCoord3);					//AQUI

	READ_SPINOR			(SPINORTEX, myStride, sid, sid);
	READ_INTERMEDIATE_SPINOR	(INTERTEX,  myStride, sid, sid);
	
	//compute in1^dag:

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;
	I4.y	 = -I4.y;
	I5.y	 = -I5.y;
	I6.y	 = -I6.y;	
	I7.y	 = -I7.y;
	I8.y	 = -I8.y;
	I9.y	 = -I9.y;
	I10.y	 = -I10.y;
	I11.y	 = -I11.y;

	//do products for all color component here:
	//00 component:
	tmp_re	 =  I0.x *  J0.x -  I0.y *  J0.y;
	tmp_re	+=  I1.x *  J1.x -  I1.y *  J1.y;
	tmp_re	+=  I2.x *  J2.x -  I2.y *  J2.y;
	accum_re[0*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J0.y +  I0.y *  J0.x;	
	tmp_im	+=  I1.x *  J1.y +  I1.y *  J1.x;	
	tmp_im	+=  I2.x *  J2.y +  I2.y *  J2.x;	
	accum_im[0*blockDim.x]	= tmp_im; 	

	//01 component:
	tmp_re	 =  I0.x *  J3.x -  I0.y *  J3.y;
	tmp_re	+=  I1.x *  J4.x -  I1.y *  J4.y;
	tmp_re	+=  I2.x *  J5.x -  I2.y *  J5.y;
	accum_re[1*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J3.y +  I0.y *  J3.x;	
	tmp_im	+=  I1.x *  J4.y +  I1.y *  J4.x;	
	tmp_im	+=  I2.x *  J5.y +  I2.y *  J5.x;	
	accum_im[1*blockDim.x]	= tmp_im; 	

	//02 component:
	tmp_re	 =  I0.x *  J6.x -  I0.y *  J6.y;
	tmp_re	+=  I1.x *  J7.x -  I1.y *  J7.y;
	tmp_re	+=  I2.x *  J8.x -  I2.y *  J8.y;
	accum_re[2*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J6.y +  I0.y *  J6.x;	
	tmp_im	+=  I1.x *  J7.y +  I1.y *  J7.x;	
	tmp_im	+=  I2.x *  J8.y +  I2.y *  J8.x;	
	accum_im[2*blockDim.x]	= tmp_im;	

	//03 component:
	tmp_re	 =  I0.x *  J9.x -  I0.y *  J9.y;
	tmp_re	+=  I1.x * J10.x -  I1.y * J10.y;
	tmp_re	+=  I2.x * J11.x -  I2.y * J11.y;
	accum_re[3*blockDim.x]	= tmp_re;

	tmp_im	 =  I0.x *  J9.y +  I0.y *  J9.x;	
	tmp_im	+=  I1.x * J10.y +  I1.y * J10.x;	
	tmp_im	+=  I2.x * J11.y +  I2.y * J11.x;	
	accum_im[3*blockDim.x]	= tmp_im;	

	//10 component:
	tmp_re	 =  I3.x *  J0.x -  I3.y *  J0.y;
	tmp_re	+=  I4.x *  J1.x -  I4.y *  J1.y;
	tmp_re	+=  I5.x *  J2.x -  I5.y *  J2.y;
	accum_re[ 4*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J0.y +  I3.y *  J0.x;	
	tmp_im	+=  I4.x *  J1.y +  I4.y *  J1.x;	
	tmp_im	+=  I5.x *  J2.y +  I5.y *  J2.x;	
	accum_im[ 4*blockDim.x]	= tmp_im; 	

	//11 component:
	tmp_re	 =  I3.x *  J3.x -  I3.y *  J3.y;
	tmp_re	+=  I4.x *  J4.x -  I4.y *  J4.y;
	tmp_re	+=  I5.x *  J5.x -  I5.y *  J5.y;
	accum_re[ 5*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J3.y +  I3.y *  J3.x;	
	tmp_im	+=  I4.x *  J4.y +  I4.y *  J4.x;	
	tmp_im	+=  I5.x *  J5.y +  I5.y *  J5.x;	
	accum_im[ 5*blockDim.x]	= tmp_im; 	

	//12 component:
	tmp_re	 =  I3.x *  J6.x -  I3.y *  J6.y;
	tmp_re	+=  I4.x *  J7.x -  I4.y *  J7.y;
	tmp_re	+=  I5.x *  J8.x -  I5.y *  J8.y;
	accum_re[ 6*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J6.y +  I3.y *  J6.x;	
	tmp_im	+=  I4.x *  J7.y +  I4.y *  J7.x;	
	tmp_im	+=  I5.x *  J8.y +  I5.y *  J8.x;	
	accum_im[ 6*blockDim.x]	= tmp_im;	

	//13 component:
	tmp_re	 =  I3.x *  J9.x -  I3.y *  J9.y;
	tmp_re	+=  I4.x * J10.x -  I4.y * J10.y;
	tmp_re	+=  I5.x * J11.x -  I5.y * J11.y;
	accum_re[ 7*blockDim.x]	= tmp_re;

	tmp_im	 =  I3.x *  J9.y +  I3.y *  J9.x;	
	tmp_im	+=  I4.x * J10.y +  I4.y * J10.x;	
	tmp_im	+=  I5.x * J11.y +  I5.y * J11.x;	
	accum_im[ 7*blockDim.x]	= tmp_im;	

	//20 component:
	tmp_re	 =  I6.x *  J0.x -  I6.y *  J0.y;
	tmp_re	+=  I7.x *  J1.x -  I7.y *  J1.y;
	tmp_re	+=  I8.x *  J2.x -  I8.y *  J2.y;
	accum_re[ 8*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J0.y +  I6.y *  J0.x;	
	tmp_im	+=  I7.x *  J1.y +  I7.y *  J1.x;	
	tmp_im	+=  I8.x *  J2.y +  I8.y *  J2.x;	
	accum_im[ 8*blockDim.x]	= tmp_im; 	

	//21 component:
	tmp_re	 =  I6.x *  J3.x -  I6.y *  J3.y;
	tmp_re	+=  I7.x *  J4.x -  I7.y *  J4.y;
	tmp_re	+=  I8.x *  J5.x -  I8.y *  J5.y;
	accum_re[ 9*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J3.y +  I6.y *  J3.x;	
	tmp_im	+=  I7.x *  J4.y +  I7.y *  J4.x;	
	tmp_im	+=  I8.x *  J5.y +  I8.y *  J5.x;	
	accum_im[ 9*blockDim.x]	= tmp_im; 	

	//22 component:
	tmp_re	 =  I6.x *  J6.x -  I6.y *  J6.y;
	tmp_re	+=  I7.x *  J7.x -  I7.y *  J7.y;
	tmp_re	+=  I8.x *  J8.x -  I8.y *  J8.y;
	accum_re[10*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J6.y +  I6.y *  J6.x;	
	tmp_im	+=  I7.x *  J7.y +  I7.y *  J7.x;	
	tmp_im	+=  I8.x *  J8.y +  I8.y *  J8.x;	
	accum_im[10*blockDim.x]	= tmp_im;	

	//23 component:
	tmp_re	 =  I6.x *  J9.x -  I6.y *  J9.y;
	tmp_re	+=  I7.x * J10.x -  I7.y * J10.y;
	tmp_re	+=  I8.x * J11.x -  I8.y * J11.y;
	accum_re[11*blockDim.x]	= tmp_re;

	tmp_im	 =  I6.x *  J9.y +  I6.y *  J9.x;	
	tmp_im	+=  I7.x * J10.y +  I7.y * J10.x;	
	tmp_im	+=  I8.x * J11.y +  I8.y * J11.x;	
	accum_im[11*blockDim.x]	= tmp_im;	

	//30 component:
	tmp_re	 =  I9.x *  J0.x -  I9.y *  J0.y;
	tmp_re	+= I10.x *  J1.x - I10.y *  J1.y;
	tmp_re	+= I11.x *  J2.x - I11.y *  J2.y;
	accum_re[12*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J0.y +  I9.y *  J0.x;	
	tmp_im	+= I10.x *  J1.y + I10.y *  J1.x;	
	tmp_im	+= I11.x *  J2.y + I11.y *  J2.x;	
	accum_im[12*blockDim.x]	= tmp_im; 	

	//31 component:
	tmp_re	 =  I9.x *  J3.x -  I9.y *  J3.y;
	tmp_re	+= I10.x *  J4.x - I10.y *  J4.y;
	tmp_re	+= I11.x *  J5.x - I11.y *  J5.y;
	accum_re[13*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J3.y +  I9.y *  J3.x;	
	tmp_im	+= I10.x *  J4.y + I10.y *  J4.x;	
	tmp_im	+= I11.x *  J5.y + I11.y *  J5.x;	
	accum_im[13*blockDim.x]	= tmp_im; 	

	//32 component:
	tmp_re	 =  I9.x *  J6.x -  I9.y *  J6.y;
	tmp_re	+= I10.x *  J7.x - I10.y *  J7.y;
	tmp_re	+= I11.x *  J8.x - I11.y *  J8.y;
	accum_re[14*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J6.y +  I9.y *  J6.x;	
	tmp_im	+= I10.x *  J7.y + I10.y *  J7.x;	
	tmp_im	+= I11.x *  J8.y + I11.y *  J8.x;	
	accum_im[14*blockDim.x]	= tmp_im;	

	//33 component:
	tmp_re	 =  I9.x *  J9.x -  I9.y *  J9.y;
	tmp_re	+= I10.x * J10.x - I10.y * J10.y;
	tmp_re	+= I11.x * J11.x - I11.y * J11.y;
	accum_re[15*blockDim.x]	= tmp_re;

	tmp_im	 =  I9.x *  J9.y +  I9.y *  J9.x;	
	tmp_im	+= I10.x * J10.y + I10.y * J10.x;	
	tmp_im	+= I11.x * J11.y + I11.y * J11.x;	
	accum_im[15*blockDim.x]	= tmp_im;	

   //Store output back to global buffer:


/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *param.threads*2]	 = make_double2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *param.threads*2]	 = make_double2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *param.threads*2]	 = make_double2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *param.threads*2]	 = make_double2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *param.threads*2]	 = make_double2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *param.threads*2]	 = make_double2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *param.threads*2]	 = make_double2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *param.threads*2]	 = make_double2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *param.threads*2]	 = make_double2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *param.threads*2]	 = make_double2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*param.threads*2]	 = make_double2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]);
	out[outId + 11*param.threads*2]	 = make_double2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]);
	out[outId + 12*param.threads*2]	 = make_double2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]);
	out[outId + 13*param.threads*2]	 = make_double2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]);
	out[outId + 14*param.threads*2]	 = make_double2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]);
	out[outId + 15*param.threads*2]	 = make_double2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	return;
}

__global__ void contractKernel		(double2 *out, double2 *in1, double2 *in2, int myStride, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= param.threads)
		return;

	volatile double2		tmp;
	extern __shared__ double	sm[];								//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile double			*accum_re	 = sm + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile double			*accum_im	 = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / X1;
	xCoord1		 = eutId - auxCoord1 * X1;
	auxCoord2	 = auxCoord1 / X2;
	xCoord2		 = auxCoord1 - auxCoord2 * X2;
	xCoord4		 = auxCoord2 / X3;
	xCoord3		 = auxCoord2 - xCoord4 * X3;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + X1*(xCoord2 + X2*(xCoord3 + X3*xCoord4));				//AQUI

	READ_SPINOR			(SPINORTEX, myStride, sid, sid);
	READ_INTERMEDIATE_SPINOR	(INTERTEX,  myStride, sid, sid);
	
	//compute in1^dag:

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;
	I4.y	 = -I4.y;
	I5.y	 = -I5.y;
	I6.y	 = -I6.y;	
	I7.y	 = -I7.y;
	I8.y	 = -I8.y;
	I9.y	 = -I9.y;
	I10.y	 = -I10.y;
	I11.y	 = -I11.y;

	//do products for all color component here:
	//00 component:
	tmp_re	 =  I0.x *  J0.x -  I0.y *  J0.y;
	tmp_re	+=  I1.x *  J1.x -  I1.y *  J1.y;
	tmp_re	+=  I2.x *  J2.x -  I2.y *  J2.y;
	accum_re[0*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J0.y +  I0.y *  J0.x;	
	tmp_im	+=  I1.x *  J1.y +  I1.y *  J1.x;	
	tmp_im	+=  I2.x *  J2.y +  I2.y *  J2.x;	
	accum_im[0*blockDim.x]	= tmp_im; 	

	//01 component:
	tmp_re	 =  I0.x *  J3.x -  I0.y *  J3.y;
	tmp_re	+=  I1.x *  J4.x -  I1.y *  J4.y;
	tmp_re	+=  I2.x *  J5.x -  I2.y *  J5.y;
	accum_re[1*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J3.y +  I0.y *  J3.x;	
	tmp_im	+=  I1.x *  J4.y +  I1.y *  J4.x;	
	tmp_im	+=  I2.x *  J5.y +  I2.y *  J5.x;	
	accum_im[1*blockDim.x]	= tmp_im; 	

	//02 component:
	tmp_re	 =  I0.x *  J6.x -  I0.y *  J6.y;
	tmp_re	+=  I1.x *  J7.x -  I1.y *  J7.y;
	tmp_re	+=  I2.x *  J8.x -  I2.y *  J8.y;
	accum_re[2*blockDim.x]	= tmp_re; 

	tmp_im	 =  I0.x *  J6.y +  I0.y *  J6.x;	
	tmp_im	+=  I1.x *  J7.y +  I1.y *  J7.x;	
	tmp_im	+=  I2.x *  J8.y +  I2.y *  J8.x;	
	accum_im[2*blockDim.x]	= tmp_im;	

	//03 component:
	tmp_re	 =  I0.x *  J9.x -  I0.y *  J9.y;
	tmp_re	+=  I1.x * J10.x -  I1.y * J10.y;
	tmp_re	+=  I2.x * J11.x -  I2.y * J11.y;
	accum_re[3*blockDim.x]	= tmp_re;

	tmp_im	 =  I0.x *  J9.y +  I0.y *  J9.x;	
	tmp_im	+=  I1.x * J10.y +  I1.y * J10.x;	
	tmp_im	+=  I2.x * J11.y +  I2.y * J11.x;	
	accum_im[3*blockDim.x]	= tmp_im;	

	//10 component:
	tmp_re	 =  I3.x *  J0.x -  I3.y *  J0.y;
	tmp_re	+=  I4.x *  J1.x -  I4.y *  J1.y;
	tmp_re	+=  I5.x *  J2.x -  I5.y *  J2.y;
	accum_re[ 4*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J0.y +  I3.y *  J0.x;	
	tmp_im	+=  I4.x *  J1.y +  I4.y *  J1.x;	
	tmp_im	+=  I5.x *  J2.y +  I5.y *  J2.x;	
	accum_im[ 4*blockDim.x]	= tmp_im; 	

	//11 component:
	tmp_re	 =  I3.x *  J3.x -  I3.y *  J3.y;
	tmp_re	+=  I4.x *  J4.x -  I4.y *  J4.y;
	tmp_re	+=  I5.x *  J5.x -  I5.y *  J5.y;
	accum_re[ 5*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J3.y +  I3.y *  J3.x;	
	tmp_im	+=  I4.x *  J4.y +  I4.y *  J4.x;	
	tmp_im	+=  I5.x *  J5.y +  I5.y *  J5.x;	
	accum_im[ 5*blockDim.x]	= tmp_im; 	

	//12 component:
	tmp_re	 =  I3.x *  J6.x -  I3.y *  J6.y;
	tmp_re	+=  I4.x *  J7.x -  I4.y *  J7.y;
	tmp_re	+=  I5.x *  J8.x -  I5.y *  J8.y;
	accum_re[ 6*blockDim.x]	= tmp_re; 

	tmp_im	 =  I3.x *  J6.y +  I3.y *  J6.x;	
	tmp_im	+=  I4.x *  J7.y +  I4.y *  J7.x;	
	tmp_im	+=  I5.x *  J8.y +  I5.y *  J8.x;	
	accum_im[ 6*blockDim.x]	= tmp_im;	

	//13 component:
	tmp_re	 =  I3.x *  J9.x -  I3.y *  J9.y;
	tmp_re	+=  I4.x * J10.x -  I4.y * J10.y;
	tmp_re	+=  I5.x * J11.x -  I5.y * J11.y;
	accum_re[ 7*blockDim.x]	= tmp_re;

	tmp_im	 =  I3.x *  J9.y +  I3.y *  J9.x;	
	tmp_im	+=  I4.x * J10.y +  I4.y * J10.x;	
	tmp_im	+=  I5.x * J11.y +  I5.y * J11.x;	
	accum_im[ 7*blockDim.x]	= tmp_im;	

	//20 component:
	tmp_re	 =  I6.x *  J0.x -  I6.y *  J0.y;
	tmp_re	+=  I7.x *  J1.x -  I7.y *  J1.y;
	tmp_re	+=  I8.x *  J2.x -  I8.y *  J2.y;
	accum_re[ 8*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J0.y +  I6.y *  J0.x;	
	tmp_im	+=  I7.x *  J1.y +  I7.y *  J1.x;	
	tmp_im	+=  I8.x *  J2.y +  I8.y *  J2.x;	
	accum_im[ 8*blockDim.x]	= tmp_im; 	

	//21 component:
	tmp_re	 =  I6.x *  J3.x -  I6.y *  J3.y;
	tmp_re	+=  I7.x *  J4.x -  I7.y *  J4.y;
	tmp_re	+=  I8.x *  J5.x -  I8.y *  J5.y;
	accum_re[ 9*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J3.y +  I6.y *  J3.x;	
	tmp_im	+=  I7.x *  J4.y +  I7.y *  J4.x;	
	tmp_im	+=  I8.x *  J5.y +  I8.y *  J5.x;	
	accum_im[ 9*blockDim.x]	= tmp_im; 	

	//22 component:
	tmp_re	 =  I6.x *  J6.x -  I6.y *  J6.y;
	tmp_re	+=  I7.x *  J7.x -  I7.y *  J7.y;
	tmp_re	+=  I8.x *  J8.x -  I8.y *  J8.y;
	accum_re[10*blockDim.x]	= tmp_re; 

	tmp_im	 =  I6.x *  J6.y +  I6.y *  J6.x;	
	tmp_im	+=  I7.x *  J7.y +  I7.y *  J7.x;	
	tmp_im	+=  I8.x *  J8.y +  I8.y *  J8.x;	
	accum_im[10*blockDim.x]	= tmp_im;	

	//23 component:
	tmp_re	 =  I6.x *  J9.x -  I6.y *  J9.y;
	tmp_re	+=  I7.x * J10.x -  I7.y * J10.y;
	tmp_re	+=  I8.x * J11.x -  I8.y * J11.y;
	accum_re[11*blockDim.x]	= tmp_re;

	tmp_im	 =  I6.x *  J9.y +  I6.y *  J9.x;	
	tmp_im	+=  I7.x * J10.y +  I7.y * J10.x;	
	tmp_im	+=  I8.x * J11.y +  I8.y * J11.x;	
	accum_im[11*blockDim.x]	= tmp_im;	

	//30 component:
	tmp_re	 =  I9.x *  J0.x -  I9.y *  J0.y;
	tmp_re	+= I10.x *  J1.x - I10.y *  J1.y;
	tmp_re	+= I11.x *  J2.x - I11.y *  J2.y;
	accum_re[12*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J0.y +  I9.y *  J0.x;	
	tmp_im	+= I10.x *  J1.y + I10.y *  J1.x;	
	tmp_im	+= I11.x *  J2.y + I11.y *  J2.x;	
	accum_im[12*blockDim.x]	= tmp_im; 	

	//31 component:
	tmp_re	 =  I9.x *  J3.x -  I9.y *  J3.y;
	tmp_re	+= I10.x *  J4.x - I10.y *  J4.y;
	tmp_re	+= I11.x *  J5.x - I11.y *  J5.y;
	accum_re[13*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J3.y +  I9.y *  J3.x;	
	tmp_im	+= I10.x *  J4.y + I10.y *  J4.x;	
	tmp_im	+= I11.x *  J5.y + I11.y *  J5.x;	
	accum_im[13*blockDim.x]	= tmp_im; 	

	//32 component:
	tmp_re	 =  I9.x *  J6.x -  I9.y *  J6.y;
	tmp_re	+= I10.x *  J7.x - I10.y *  J7.y;
	tmp_re	+= I11.x *  J8.x - I11.y *  J8.y;
	accum_re[14*blockDim.x]	= tmp_re; 

	tmp_im	 =  I9.x *  J6.y +  I9.y *  J6.x;	
	tmp_im	+= I10.x *  J7.y + I10.y *  J7.x;	
	tmp_im	+= I11.x *  J8.y + I11.y *  J8.x;	
	accum_im[14*blockDim.x]	= tmp_im;	

	//33 component:
	tmp_re	 =  I9.x *  J9.x -  I9.y *  J9.y;
	tmp_re	+= I10.x * J10.x - I10.y * J10.y;
	tmp_re	+= I11.x * J11.x - I11.y * J11.y;
	accum_re[15*blockDim.x]	= tmp_re;

	tmp_im	 =  I9.x *  J9.y +  I9.y *  J9.x;	
	tmp_im	+= I10.x * J10.y + I10.y * J10.x;	
	tmp_im	+= I11.x * J11.y + I11.y * J11.x;	
	accum_im[15*blockDim.x]	= tmp_im;	

/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *param.threads*2]	 = make_double2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *param.threads*2]	 = make_double2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *param.threads*2]	 = make_double2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *param.threads*2]	 = make_double2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *param.threads*2]	 = make_double2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *param.threads*2]	 = make_double2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *param.threads*2]	 = make_double2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *param.threads*2]	 = make_double2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *param.threads*2]	 = make_double2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *param.threads*2]	 = make_double2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*param.threads*2]	 = make_double2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*param.threads*2]	 = make_double2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*param.threads*2]	 = make_double2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*param.threads*2]	 = make_double2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*param.threads*2]	 = make_double2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*param.threads*2]	 = make_double2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	return;
}

#undef SPINORTEX
#undef INTERTEX

#undef	READ_SPINOR
#undef	READ_INTERMEDIATE_SPINOR

#undef	SPINOR_HOP

#endif // (__CUDA_ARCH__ >= 130)


#define READ_SPINOR_SINGLE(spinor, stride, sp_idx, norm_idx)	   \
  float4 I0 = spinor[sp_idx + 0*(stride)];   \
  float4 I1 = spinor[sp_idx + 1*(stride)];   \
  float4 I2 = spinor[sp_idx + 2*(stride)];   \
  float4 I3 = spinor[sp_idx + 3*(stride)];   \
  float4 I4 = spinor[sp_idx + 4*(stride)];   \
  float4 I5 = spinor[sp_idx + 5*(stride)];


#define READ_SPINOR_SINGLE_TEX(spinor, stride, sp_idx, norm_idx)	\
  float4 I0 = TEX1DFETCH(float4, (spinor), sp_idx + 0*(stride));	\
  float4 I1 = TEX1DFETCH(float4, (spinor), sp_idx + 1*(stride));	\
  float4 I2 = TEX1DFETCH(float4, (spinor), sp_idx + 2*(stride));	\
  float4 I3 = TEX1DFETCH(float4, (spinor), sp_idx + 3*(stride));	\
  float4 I4 = TEX1DFETCH(float4, (spinor), sp_idx + 4*(stride));	\
  float4 I5 = TEX1DFETCH(float4, (spinor), sp_idx + 5*(stride));

#define READ_INTERMEDIATE_SPINOR_SINGLE(spinor, stride, sp_idx, norm_idx)	   \
  float4 J0 = spinor[sp_idx + 0*(stride)];   \
  float4 J1 = spinor[sp_idx + 1*(stride)];   \
  float4 J2 = spinor[sp_idx + 2*(stride)];   \
  float4 J3 = spinor[sp_idx + 3*(stride)];   \
  float4 J4 = spinor[sp_idx + 4*(stride)];   \
  float4 J5 = spinor[sp_idx + 5*(stride)];

#define READ_INTERMEDIATE_SPINOR_SINGLE_TEX(spinor, stride, sp_idx, norm_idx)	\
  float4 J0 = TEX1DFETCH(float4, (spinor), sp_idx + 0*(stride));	\
  float4 J1 = TEX1DFETCH(float4, (spinor), sp_idx + 1*(stride));	\
  float4 J2 = TEX1DFETCH(float4, (spinor), sp_idx + 2*(stride));	\
  float4 J3 = TEX1DFETCH(float4, (spinor), sp_idx + 3*(stride));	\
  float4 J4 = TEX1DFETCH(float4, (spinor), sp_idx + 4*(stride));	\
  float4 J5 = TEX1DFETCH(float4, (spinor), sp_idx + 5*(stride));


#ifdef DIRECT_ACCESS_WILSON_SPINOR
	#define READ_SPINOR READ_SPINOR_SINGLE
	#define SPINORTEX in
#else
	#define READ_SPINOR READ_SPINOR_SINGLE_TEX

	#ifdef USE_TEXTURE_OBJECTS
		#define SPINORTEX param.inTex
	#else
		#define SPINORTEX spinorTexSingle
	#endif	// USE_TEXTURE_OBJECTS
#endif

#ifdef DIRECT_ACCESS_WILSON_INTER
	#define READ_INTERMEDIATE_SPINOR READ_INTERMEDIATE_SPINOR_SINGLE
	#define INTERTEX in2
#else
	#define READ_INTERMEDIATE_SPINOR READ_INTERMEDIATE_SPINOR_SINGLE_TEX

	#ifdef USE_TEXTURE_OBJECTS
		#define INTERTEX param.outTex
	#else
		#define INTERTEX interTexSingle
	#endif	// USE_TEXTURE_OBJECTS
#endif

#define SPINOR_HOP 6

__global__ void contractGamma5Kernel	(float2 *out, float4 *in1, float4 *in2, int myStride, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= param.threads)
		return;

	volatile float2		tmp;
	extern __shared__ float	sms[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile float		*accum_re = sms + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile float		*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / X1;
	xCoord1		 = eutId - auxCoord1 * X1;
	auxCoord2	 = auxCoord1 / X2;
	xCoord2		 = auxCoord1 - auxCoord2 * X2;
	xCoord4		 = auxCoord2 / X3;
	xCoord3		 = auxCoord2 - xCoord4 * X3;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + X1*(xCoord2 + X2*(xCoord3 + X3*xCoord4));			//AQUI

	//Load the full input spinors:

	READ_SPINOR			(SPINORTEX, myStride, sid, sid);
	READ_INTERMEDIATE_SPINOR	(INTERTEX,  myStride, sid, sid);

	//compute in1^dag * gamma5:

	//First color component

	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I3.x;
	I0.y	 = -I3.y;
	I3.x	 = tmp_re;
	I3.y	 = tmp_im;	

	tmp_re	 = +I1.z;
	tmp_im	 = -I1.w;
	I1.z	 = +I4.z;
	I1.w	 = -I4.w;
	I4.z	 = tmp_re;
	I4.w	 = tmp_im;	

	//Second color component

	tmp_re	 = +I0.z;
	tmp_im	 = -I0.w;
	I0.z	 = +I3.z;
	I0.w	 = -I3.w;
	I3.z	 = tmp_re;
	I3.w	 = tmp_im;	

	tmp_re	 = +I2.x;
	tmp_im	 = -I2.y;
	I2.x	 = +I5.x;
	I2.y	 = -I5.y;
	I5.x	 = tmp_re;
	I5.y	 = tmp_im;	

	//Third color component

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I4.x;
	I1.y	 = -I4.y;
	I4.x	 = tmp_re;
	I4.y	 = tmp_im;	

	tmp_re	 = +I2.z;
	tmp_im	 = -I2.w;
	I2.z	 = +I5.z;
	I2.w	 = -I5.w;
	I5.z	 = tmp_re;
	I5.w	 = tmp_im;	

	//do products for first color component here:

	//00 component:
	tmp_re	 = I0.x * J0.x - I0.y * J0.y;
	tmp_re	+= I0.z * J0.z - I0.w * J0.w;
	tmp_re	+= I1.x * J1.x - I1.y * J1.y;
	accum_re[0*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J0.y + I0.y * J0.x;	
	tmp_im	+= I0.z * J0.w + I0.w * J0.z;	
	tmp_im	+= I1.x * J1.y + I1.y * J1.x;	
	accum_im[0*blockDim.x]	= tmp_im;	
	
	//01 component:
	tmp_re	 = I0.x * J1.z - I0.y * J1.w;
	tmp_re	+= I0.z * J2.x - I0.w * J2.y;
	tmp_re	+= I1.x * J2.z - I1.y * J2.w;
	accum_re[1*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J1.w + I0.y * J1.z;	
	tmp_im	+= I0.z * J2.y + I0.w * J2.x;	
	tmp_im	+= I1.x * J2.w + I1.y * J2.z;	
	accum_im[1*blockDim.x]	= tmp_im;	

	//02 component:
	tmp_re	 = I0.x * J3.x - I0.y * J3.y;
	tmp_re	+= I0.z * J3.z - I0.w * J3.w;
	tmp_re	+= I1.x * J4.x - I1.y * J4.y;
	accum_re[2*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J3.y + I0.y * J3.x;	
	tmp_im	+= I0.z * J3.w + I0.w * J3.z;	
	tmp_im	+= I1.x * J4.y + I1.y * J4.x;	
	accum_im[2*blockDim.x]	= tmp_im;	
      
	//03 component:
	tmp_re	 = I0.x * J4.z - I0.y * J4.w;
	tmp_re	+= I0.z * J5.x - I0.w * J5.x;
	tmp_re	+= I1.x * J5.z - I1.y * J5.w;
      
	accum_re[3*blockDim.x]	= tmp_re;
      
	tmp_im	 = I0.x * J4.w + I0.y * J4.z;	
	tmp_im	+= I0.z * J5.y + I0.w * J5.y;	
	tmp_im	+= I1.x * J5.w + I1.y * J5.z;	
	accum_im[3*blockDim.x]	= tmp_im;	

	//10 component:
	tmp_re	 = I1.z * J0.x - I1.w * J0.y;
	tmp_re	+= I2.x * J0.z - I2.y * J0.w;
	tmp_re	+= I2.z * J1.x - I2.w * J1.y;
	accum_re[4*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J0.y + I1.w * J0.x;	
	tmp_im	+= I2.x * J0.w + I2.y * J0.z;	
	tmp_im	+= I2.z * J1.y + I2.w * J1.x;	
	accum_im[4*blockDim.x]	= tmp_im;	

	//11 component:
	tmp_re	 = I1.z * J1.z - I1.w * J1.w;
	tmp_re	+= I2.x * J2.x - I2.y * J2.y;
	tmp_re	+= I2.z * J2.z - I2.w * J2.w;
	accum_re[5*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J1.w + I1.w * J1.z;	
	tmp_im	+= I2.x * J2.y + I2.y * J2.x;	
	tmp_im	+= I2.z * J2.w + I2.w * J2.z;	
	accum_im[5*blockDim.x]	= tmp_im;	

	//12 component:
	tmp_re	 = I1.z * J3.x - I1.w * J3.y;
	tmp_re	+= I2.x * J3.z - I2.y * J3.w;
	tmp_re	+= I2.z * J4.x - I2.w * J4.y;
	accum_re[6*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J3.y + I1.w * J3.x;	
	tmp_im	+= I2.x * J3.w + I2.y * J3.z;	
	tmp_im	+= I2.z * J4.y + I2.w * J4.x;	
	accum_im[6*blockDim.x]	= tmp_im;	

	//13 component:
	tmp_re	 = I1.z * J4.z - I1.w * J4.w;
	tmp_re	+= I2.x * J5.x - I2.y * J5.y;
	tmp_re	+= I2.z * J5.z - I2.w * J5.w;
	accum_re[7*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J4.w + I1.w * J4.z;	
	tmp_im	+= I2.x * J5.y + I2.y * J5.x;	
	tmp_im	+= I2.z * J5.w + I2.w * J5.z;	
	accum_im[7*blockDim.x]	= tmp_im;	

	//20 component:
	tmp_re	 = I3.x * J0.x - I3.y * J0.y;
	tmp_re	+= I3.z * J0.z - I3.w * J0.w;
	tmp_re	+= I4.x * J1.x - I4.y * J1.y;
	accum_re[8*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J0.y + I3.y * J0.x;	
	tmp_im	+= I3.z * J0.w + I3.w * J0.z;	
	tmp_im	+= I4.x * J1.y + I4.y * J1.x;	
	accum_im[8*blockDim.x]	= tmp_im;	

	//21 component:
	tmp_re	 = I3.x * J1.z - I3.y * J1.w;
	tmp_re	+= I3.z * J2.x - I3.w * J2.y;
	tmp_re	+= I4.x * J2.z - I4.y * J2.w;
	accum_re[9*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J1.w + I3.y * J1.z;	
	tmp_im	+= I3.z * J2.y + I3.w * J2.x;	
	tmp_im	+= I4.x * J2.w + I4.y * J2.z;	
	accum_im[9*blockDim.x]	= tmp_im;	

	//22 component:
	tmp_re	 = I3.x * J3.x - I3.y * J3.y;
	tmp_re	+= I3.z * J3.z - I3.w * J3.w;
	tmp_re	+= I4.x * J4.x - I4.y * J4.y;
	accum_re[10*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J3.y + I3.y * J3.x;	
	tmp_im	+= I3.z * J3.w + I3.w * J3.z;	
	tmp_im	+= I4.x * J4.y + I4.y * J4.x;	
	accum_im[10*blockDim.x]	= tmp_im;	

	//23 component:
	tmp_re	 = I3.x * J4.z - I3.y * J4.w;
	tmp_re	+= I3.z * J5.x - I3.w * J5.y;
	tmp_re	+= I4.x * J5.z - I4.y * J5.w;
	accum_re[11*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J4.w + I3.y * J4.z;	
	tmp_im	+= I3.z * J5.y + I3.w * J5.x;	
	tmp_im	+= I4.x * J5.w + I4.y * J5.z;	
	accum_im[11*blockDim.x]	= tmp_im;	

	//30 component:
	tmp_re	 = I4.z * J0.x - I4.w * J0.y;
	tmp_re	+= I5.x * J0.z - I5.y * J0.w;
	tmp_re	+= I5.z * J1.x - I5.w * J1.y;
	accum_re[12*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J0.y + I4.w * J0.x;	
	tmp_im	+= I5.x * J0.w + I5.y * J0.z;	
	tmp_im	+= I5.z * J1.y + I5.w * J1.x;	
	accum_im[12*blockDim.x]	= tmp_im;	

	//31 component:
	tmp_re	 = I4.z * J1.z - I4.w * J1.w;
	tmp_re	+= I5.x * J2.x - I5.y * J2.y;
	tmp_re	+= I5.z * J2.z - I5.w * J2.w;
	accum_re[13*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J1.w + I4.w * J1.z;	
	tmp_im	+= I5.x * J2.y + I5.y * J2.x;	
	tmp_im	+= I5.z * J2.w + I5.w * J2.z;	
	accum_im[13*blockDim.x]	= tmp_im;	

	//32 component:
	tmp_re	 = I4.z * J3.x - I4.w * J3.y;
	tmp_re	+= I5.x * J3.z - I5.y * J3.w;
	tmp_re	+= I5.z * J4.x - I5.w * J4.y;
	accum_re[14*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J3.y + I4.w * J3.x;	
	tmp_im	+= I5.x * J3.w + I5.y * J3.z;	
	tmp_im	+= I5.z * J4.y + I5.w * J4.x;	
	accum_im[14*blockDim.x]	= tmp_im;	

	//33 component:
	tmp_re	 = I4.z * J4.z - I4.w * J4.w;
	tmp_re	+= I5.x * J5.x - I5.y * J5.y;
	tmp_re	+= I5.z * J5.z - I5.w * J5.w;
	accum_re[15*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J4.w + I4.w * J4.z;	
	tmp_im	+= I5.x * J5.y + I5.y * J5.x;	
	tmp_im	+= I5.z * J5.w + I5.w * J5.z;	
	accum_im[15*blockDim.x]	= tmp_im;	



	//Store output back to global buffer:


	/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *param.threads*2]	 = make_float2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *param.threads*2]	 = make_float2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *param.threads*2]	 = make_float2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *param.threads*2]	 = make_float2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *param.threads*2]	 = make_float2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *param.threads*2]	 = make_float2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *param.threads*2]	 = make_float2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *param.threads*2]	 = make_float2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *param.threads*2]	 = make_float2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *param.threads*2]	 = make_float2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*param.threads*2]	 = make_float2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*param.threads*2]	 = make_float2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*param.threads*2]	 = make_float2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*param.threads*2]	 = make_float2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*param.threads*2]	 = make_float2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*param.threads*2]	 = make_float2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	return;
}

//Perform trace in color space only and for a given tslice 
//since the file is included in dslash_quda.h, no need to add dslash_constants.h file here (for, e.g., Vsh)
__global__ void contractTsliceKernel	(float2 *out, float4 *in1, float4 *in2, int myStride, const int Tslice, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;					//number of threads is equal to Tslice volume
												//Adjust sid to correct tslice (exe domain must be Tslice volume!)
	int	inId	 = sid + Vsh*Tslice;							//Vsh - 3d space volume for the parity spinor (equale to exe domain!)
	int	outId; 
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= param.threads)								//param.threads == tslice volume
		return;

	volatile float2		tmp;
	extern __shared__ float	sms[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile float		*accum_re = sms + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile float		*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

//The output only for a given tslice (for the full tslice content, i.e., both parities!):

	eutId		 = 2*inId;
	auxCoord1	 = eutId / X1;
	xCoord1		 = eutId - auxCoord1 * X1;
	auxCoord2	 = auxCoord1 / X2;
	xCoord2		 = auxCoord1 - auxCoord2 * X2;
	xCoord4		 = auxCoord2 / X3;

//	if	(Tslice != xCoord4)
//		return;

	xCoord3		 = auxCoord2 - xCoord4 * X3;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + X1*(xCoord2 + X2*xCoord3);					//AQUI

	//Load the full input spinors:

	READ_SPINOR			(SPINORTEX, myStride, sid, sid);
	READ_INTERMEDIATE_SPINOR	(INTERTEX,  myStride, sid, sid);

	//compute in1^dag:

	I0.y	 = -I0.y;
	I0.w	 = -I0.w;
	I1.y	 = -I1.y;
	I1.w	 = -I1.w;
	I2.y	 = -I2.y;
	I2.w	 = -I2.w;	
	I3.y	 = -I3.y;
	I3.w	 = -I3.w;
	I4.y	 = -I4.y;
	I4.w	 = -I4.w;
	I5.y	 = -I5.y;
	I5.w	 = -I5.w;	

	//do products for first color component here:
	//00 component:
	tmp_re	 = I0.x * J0.x - I0.y * J0.y;
	tmp_re	+= I0.z * J0.z - I0.w * J0.w;
	tmp_re	+= I1.x * J1.x - I1.y * J1.y;
	accum_re[0*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J0.y + I0.y * J0.x;	
	tmp_im	+= I0.z * J0.w + I0.w * J0.z;	
	tmp_im	+= I1.x * J1.y + I1.y * J1.x;	
	accum_im[0*blockDim.x]	= tmp_im;	
	
	//01 component:
	tmp_re	 = I0.x * J1.z - I0.y * J1.w;
	tmp_re	+= I0.z * J2.x - I0.w * J2.y;
	tmp_re	+= I1.x * J2.z - I1.y * J2.w;
	accum_re[1*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J1.w + I0.y * J1.z;	
	tmp_im	+= I0.z * J2.y + I0.w * J2.x;	
	tmp_im	+= I1.x * J2.w + I1.y * J2.z;	
	accum_im[1*blockDim.x]	= tmp_im;	

	//02 component:
	tmp_re	 = I0.x * J3.x - I0.y * J3.y;
	tmp_re	+= I0.z * J3.z - I0.w * J3.w;
	tmp_re	+= I1.x * J4.x - I1.y * J4.y;
	accum_re[2*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J3.y + I0.y * J3.x;	
	tmp_im	+= I0.z * J3.w + I0.w * J3.z;	
	tmp_im	+= I1.x * J4.y + I1.y * J4.x;	
	accum_im[2*blockDim.x]	= tmp_im;	
      
	//03 component:
	tmp_re	 = I0.x * J4.z - I0.y * J4.w;
	tmp_re	+= I0.z * J5.x - I0.w * J5.x;
	tmp_re	+= I1.x * J5.z - I1.y * J5.w;
      
	accum_re[3*blockDim.x]	= tmp_re;
      
	tmp_im	 = I0.x * J4.w + I0.y * J4.z;	
	tmp_im	+= I0.z * J5.y + I0.w * J5.y;	
	tmp_im	+= I1.x * J5.w + I1.y * J5.z;	
	accum_im[3*blockDim.x]	= tmp_im;	

	//10 component:
	tmp_re	 = I1.z * J0.x - I1.w * J0.y;
	tmp_re	+= I2.x * J0.z - I2.y * J0.w;
	tmp_re	+= I2.z * J1.x - I2.w * J1.y;
	accum_re[4*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J0.y + I1.w * J0.x;	
	tmp_im	+= I2.x * J0.w + I2.y * J0.z;	
	tmp_im	+= I2.z * J1.y + I2.w * J1.x;	
	accum_im[4*blockDim.x]	= tmp_im;	

	//11 component:
	tmp_re	 = I1.z * J1.z - I1.w * J1.w;
	tmp_re	+= I2.x * J2.x - I2.y * J2.y;
	tmp_re	+= I2.z * J2.z - I2.w * J2.w;
	accum_re[5*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J1.w + I1.w * J1.z;	
	tmp_im	+= I2.x * J2.y + I2.y * J2.x;	
	tmp_im	+= I2.z * J2.w + I2.w * J2.z;	
	accum_im[5*blockDim.x]	= tmp_im;	

	//12 component:
	tmp_re	 = I1.z * J3.x - I1.w * J3.y;
	tmp_re	+= I2.x * J3.z - I2.y * J3.w;
	tmp_re	+= I2.z * J4.x - I2.w * J4.y;
	accum_re[6*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J3.y + I1.w * J3.x;	
	tmp_im	+= I2.x * J3.w + I2.y * J3.z;	
	tmp_im	+= I2.z * J4.y + I2.w * J4.x;	
	accum_im[6*blockDim.x]	= tmp_im;	

	//13 component:
	tmp_re	 = I1.z * J4.z - I1.w * J4.w;
	tmp_re	+= I2.x * J5.x - I2.y * J5.y;
	tmp_re	+= I2.z * J5.z - I2.w * J5.w;
	accum_re[7*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J4.w + I1.w * J4.z;	
	tmp_im	+= I2.x * J5.y + I2.y * J5.x;	
	tmp_im	+= I2.z * J5.w + I2.w * J5.z;	
	accum_im[7*blockDim.x]	= tmp_im;	

	//20 component:
	tmp_re	 = I3.x * J0.x - I3.y * J0.y;
	tmp_re	+= I3.z * J0.z - I3.w * J0.w;
	tmp_re	+= I4.x * J1.x - I4.y * J1.y;
	accum_re[8*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J0.y + I3.y * J0.x;	
	tmp_im	+= I3.z * J0.w + I3.w * J0.z;	
	tmp_im	+= I4.x * J1.y + I4.y * J1.x;	
	accum_im[8*blockDim.x]	= tmp_im;	

	//21 component:
	tmp_re	 = I3.x * J1.z - I3.y * J1.w;
	tmp_re	+= I3.z * J2.x - I3.w * J2.y;
	tmp_re	+= I4.x * J2.z - I4.y * J2.w;
	accum_re[9*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J1.w + I3.y * J1.z;	
	tmp_im	+= I3.z * J2.y + I3.w * J2.x;	
	tmp_im	+= I4.x * J2.w + I4.y * J2.z;	
	accum_im[9*blockDim.x]	= tmp_im;	

	//22 component:
	tmp_re	 = I3.x * J3.x - I3.y * J3.y;
	tmp_re	+= I3.z * J3.z - I3.w * J3.w;
	tmp_re	+= I4.x * J4.x - I4.y * J4.y;
	accum_re[10*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J3.y + I3.y * J3.x;	
	tmp_im	+= I3.z * J3.w + I3.w * J3.z;	
	tmp_im	+= I4.x * J4.y + I4.y * J4.x;	
	accum_im[10*blockDim.x]	= tmp_im;	

	//23 component:
	tmp_re	 = I3.x * J4.z - I3.y * J4.w;
	tmp_re	+= I3.z * J5.x - I3.w * J5.y;
	tmp_re	+= I4.x * J5.z - I4.y * J5.w;
	accum_re[11*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J4.w + I3.y * J4.z;	
	tmp_im	+= I3.z * J5.y + I3.w * J5.x;	
	tmp_im	+= I4.x * J5.w + I4.y * J5.z;	
	accum_im[11*blockDim.x]	= tmp_im;	

	//30 component:
	tmp_re	 = I4.z * J0.x - I4.w * J0.y;
	tmp_re	+= I5.x * J0.z - I5.y * J0.w;
	tmp_re	+= I5.z * J1.x - I5.w * J1.y;
	accum_re[12*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J0.y + I4.w * J0.x;	
	tmp_im	+= I5.x * J0.w + I5.y * J0.z;	
	tmp_im	+= I5.z * J1.y + I5.w * J1.x;	
	accum_im[12*blockDim.x]	= tmp_im;	

	//31 component:
	tmp_re	 = I4.z * J1.z - I4.w * J1.w;
	tmp_re	+= I5.x * J2.x - I5.y * J2.y;
	tmp_re	+= I5.z * J2.z - I5.w * J2.w;
	accum_re[13*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J1.w + I4.w * J1.z;	
	tmp_im	+= I5.x * J2.y + I5.y * J2.x;	
	tmp_im	+= I5.z * J2.w + I5.w * J2.z;	
	accum_im[13*blockDim.x]	= tmp_im;	

	//32 component:
	tmp_re	 = I4.z * J3.x - I4.w * J3.y;
	tmp_re	+= I5.x * J3.z - I5.y * J3.w;
	tmp_re	+= I5.z * J4.x - I5.w * J4.y;
	accum_re[14*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J3.y + I4.w * J3.x;	
	tmp_im	+= I5.x * J3.w + I5.y * J3.z;	
	tmp_im	+= I5.z * J4.y + I5.w * J4.x;	
	accum_im[14*blockDim.x]	= tmp_im;	

	//33 component:
	tmp_re	 = I4.z * J4.z - I4.w * J4.w;
	tmp_re	+= I5.x * J5.x - I5.y * J5.y;
	tmp_re	+= I5.z * J5.z - I5.w * J5.w;
	accum_re[15*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J4.w + I4.w * J4.z;	
	tmp_im	+= I5.x * J5.y + I5.y * J5.x;	
	tmp_im	+= I5.z * J5.w + I5.w * J5.z;	
	accum_im[15*blockDim.x]	= tmp_im;	

	//Store output back to global buffer:


	/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *param.threads*2]	 = make_float2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *param.threads*2]	 = make_float2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *param.threads*2]	 = make_float2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *param.threads*2]	 = make_float2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *param.threads*2]	 = make_float2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *param.threads*2]	 = make_float2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *param.threads*2]	 = make_float2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *param.threads*2]	 = make_float2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *param.threads*2]	 = make_float2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *param.threads*2]	 = make_float2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*param.threads*2]	 = make_float2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]);
	out[outId + 11*param.threads*2]	 = make_float2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]);
	out[outId + 12*param.threads*2]	 = make_float2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]);
	out[outId + 13*param.threads*2]	 = make_float2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]);
	out[outId + 14*param.threads*2]	 = make_float2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]);
	out[outId + 15*param.threads*2]	 = make_float2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	return;
}

__global__ void contractKernel		(float2 *out, float4 *in1, float4 *in2, int myStride, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= param.threads)
		return;

	volatile float2		tmp;
	extern __shared__ float	sms[];								//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile float			*accum_re	 = sms + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile float			*accum_im	 = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / X1;
	xCoord1		 = eutId - auxCoord1 * X1;
	auxCoord2	 = auxCoord1 / X2;
	xCoord2		 = auxCoord1 - auxCoord2 * X2;
	xCoord4		 = auxCoord2 / X3;
	xCoord3		 = auxCoord2 - xCoord4 * X3;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + X1*(xCoord2 + X2*(xCoord3 + X3*xCoord4));				//AQUI

	//Load the full input spinors:

	READ_SPINOR			(SPINORTEX, myStride, sid, sid);
	READ_INTERMEDIATE_SPINOR	(INTERTEX,  myStride, sid, sid);

	//compute in1^dag:

	I0.y	 = -I0.y;
	I0.w	 = -I0.w;
	I1.y	 = -I1.y;
	I1.w	 = -I1.w;
	I2.y	 = -I2.y;
	I2.w	 = -I2.w;	
	I3.y	 = -I3.y;
	I3.w	 = -I3.w;
	I4.y	 = -I4.y;
	I4.w	 = -I4.w;
	I5.y	 = -I5.y;
	I5.w	 = -I5.w;	

	//do products for first color component here:
	//00 component:
	tmp_re	 = I0.x * J0.x - I0.y * J0.y;
	tmp_re	+= I0.z * J0.z - I0.w * J0.w;
	tmp_re	+= I1.x * J1.x - I1.y * J1.y;
	accum_re[0*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J0.y + I0.y * J0.x;	
	tmp_im	+= I0.z * J0.w + I0.w * J0.z;	
	tmp_im	+= I1.x * J1.y + I1.y * J1.x;	
	accum_im[0*blockDim.x]	= tmp_im;	
	
	//01 component:
	tmp_re	 = I0.x * J1.z - I0.y * J1.w;
	tmp_re	+= I0.z * J2.x - I0.w * J2.y;
	tmp_re	+= I1.x * J2.z - I1.y * J2.w;
	accum_re[1*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J1.w + I0.y * J1.z;	
	tmp_im	+= I0.z * J2.y + I0.w * J2.x;	
	tmp_im	+= I1.x * J2.w + I1.y * J2.z;	
	accum_im[1*blockDim.x]	= tmp_im;	

	//02 component:
	tmp_re	 = I0.x * J3.x - I0.y * J3.y;
	tmp_re	+= I0.z * J3.z - I0.w * J3.w;
	tmp_re	+= I1.x * J4.x - I1.y * J4.y;
	accum_re[2*blockDim.x]	= tmp_re;

	tmp_im	 = I0.x * J3.y + I0.y * J3.x;	
	tmp_im	+= I0.z * J3.w + I0.w * J3.z;	
	tmp_im	+= I1.x * J4.y + I1.y * J4.x;	
	accum_im[2*blockDim.x]	= tmp_im;	
      
	//03 component:
	tmp_re	 = I0.x * J4.z - I0.y * J4.w;
	tmp_re	+= I0.z * J5.x - I0.w * J5.x;
	tmp_re	+= I1.x * J5.z - I1.y * J5.w;
      
	accum_re[3*blockDim.x]	= tmp_re;
      
	tmp_im	 = I0.x * J4.w + I0.y * J4.z;	
	tmp_im	+= I0.z * J5.y + I0.w * J5.y;	
	tmp_im	+= I1.x * J5.w + I1.y * J5.z;	
	accum_im[3*blockDim.x]	= tmp_im;	

	//10 component:
	tmp_re	 = I1.z * J0.x - I1.w * J0.y;
	tmp_re	+= I2.x * J0.z - I2.y * J0.w;
	tmp_re	+= I2.z * J1.x - I2.w * J1.y;
	accum_re[4*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J0.y + I1.w * J0.x;	
	tmp_im	+= I2.x * J0.w + I2.y * J0.z;	
	tmp_im	+= I2.z * J1.y + I2.w * J1.x;	
	accum_im[4*blockDim.x]	= tmp_im;	

	//11 component:
	tmp_re	 = I1.z * J1.z - I1.w * J1.w;
	tmp_re	+= I2.x * J2.x - I2.y * J2.y;
	tmp_re	+= I2.z * J2.z - I2.w * J2.w;
	accum_re[5*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J1.w + I1.w * J1.z;	
	tmp_im	+= I2.x * J2.y + I2.y * J2.x;	
	tmp_im	+= I2.z * J2.w + I2.w * J2.z;	
	accum_im[5*blockDim.x]	= tmp_im;	

	//12 component:
	tmp_re	 = I1.z * J3.x - I1.w * J3.y;
	tmp_re	+= I2.x * J3.z - I2.y * J3.w;
	tmp_re	+= I2.z * J4.x - I2.w * J4.y;
	accum_re[6*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J3.y + I1.w * J3.x;	
	tmp_im	+= I2.x * J3.w + I2.y * J3.z;	
	tmp_im	+= I2.z * J4.y + I2.w * J4.x;	
	accum_im[6*blockDim.x]	= tmp_im;	

	//13 component:
	tmp_re	 = I1.z * J4.z - I1.w * J4.w;
	tmp_re	+= I2.x * J5.x - I2.y * J5.y;
	tmp_re	+= I2.z * J5.z - I2.w * J5.w;
	accum_re[7*blockDim.x]	= tmp_re;

	tmp_im	 = I1.z * J4.w + I1.w * J4.z;	
	tmp_im	+= I2.x * J5.y + I2.y * J5.x;	
	tmp_im	+= I2.z * J5.w + I2.w * J5.z;	
	accum_im[7*blockDim.x]	= tmp_im;	

	//20 component:
	tmp_re	 = I3.x * J0.x - I3.y * J0.y;
	tmp_re	+= I3.z * J0.z - I3.w * J0.w;
	tmp_re	+= I4.x * J1.x - I4.y * J1.y;
	accum_re[8*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J0.y + I3.y * J0.x;	
	tmp_im	+= I3.z * J0.w + I3.w * J0.z;	
	tmp_im	+= I4.x * J1.y + I4.y * J1.x;	
	accum_im[8*blockDim.x]	= tmp_im;	

	//21 component:
	tmp_re	 = I3.x * J1.z - I3.y * J1.w;
	tmp_re	+= I3.z * J2.x - I3.w * J2.y;
	tmp_re	+= I4.x * J2.z - I4.y * J2.w;
	accum_re[9*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J1.w + I3.y * J1.z;	
	tmp_im	+= I3.z * J2.y + I3.w * J2.x;	
	tmp_im	+= I4.x * J2.w + I4.y * J2.z;	
	accum_im[9*blockDim.x]	= tmp_im;	

	//22 component:
	tmp_re	 = I3.x * J3.x - I3.y * J3.y;
	tmp_re	+= I3.z * J3.z - I3.w * J3.w;
	tmp_re	+= I4.x * J4.x - I4.y * J4.y;
	accum_re[10*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J3.y + I3.y * J3.x;	
	tmp_im	+= I3.z * J3.w + I3.w * J3.z;	
	tmp_im	+= I4.x * J4.y + I4.y * J4.x;	
	accum_im[10*blockDim.x]	= tmp_im;	

	//23 component:
	tmp_re	 = I3.x * J4.z - I3.y * J4.w;
	tmp_re	+= I3.z * J5.x - I3.w * J5.y;
	tmp_re	+= I4.x * J5.z - I4.y * J5.w;
	accum_re[11*blockDim.x]	= tmp_re;

	tmp_im	 = I3.x * J4.w + I3.y * J4.z;	
	tmp_im	+= I3.z * J5.y + I3.w * J5.x;	
	tmp_im	+= I4.x * J5.w + I4.y * J5.z;	
	accum_im[11*blockDim.x]	= tmp_im;	

	//30 component:
	tmp_re	 = I4.z * J0.x - I4.w * J0.y;
	tmp_re	+= I5.x * J0.z - I5.y * J0.w;
	tmp_re	+= I5.z * J1.x - I5.w * J1.y;
	accum_re[12*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J0.y + I4.w * J0.x;	
	tmp_im	+= I5.x * J0.w + I5.y * J0.z;	
	tmp_im	+= I5.z * J1.y + I5.w * J1.x;	
	accum_im[12*blockDim.x]	= tmp_im;	

	//31 component:
	tmp_re	 = I4.z * J1.z - I4.w * J1.w;
	tmp_re	+= I5.x * J2.x - I5.y * J2.y;
	tmp_re	+= I5.z * J2.z - I5.w * J2.w;
	accum_re[13*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J1.w + I4.w * J1.z;	
	tmp_im	+= I5.x * J2.y + I5.y * J2.x;	
	tmp_im	+= I5.z * J2.w + I5.w * J2.z;	
	accum_im[13*blockDim.x]	= tmp_im;	

	//32 component:
	tmp_re	 = I4.z * J3.x - I4.w * J3.y;
	tmp_re	+= I5.x * J3.z - I5.y * J3.w;
	tmp_re	+= I5.z * J4.x - I5.w * J4.y;
	accum_re[14*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J3.y + I4.w * J3.x;	
	tmp_im	+= I5.x * J3.w + I5.y * J3.z;	
	tmp_im	+= I5.z * J4.y + I5.w * J4.x;	
	accum_im[14*blockDim.x]	= tmp_im;	

	//33 component:
	tmp_re	 = I4.z * J4.z - I4.w * J4.w;
	tmp_re	+= I5.x * J5.x - I5.y * J5.y;
	tmp_re	+= I5.z * J5.z - I5.w * J5.w;
	accum_re[15*blockDim.x]	= tmp_re;

	tmp_im	 = I4.z * J4.w + I4.w * J4.z;	
	tmp_im	+= I5.x * J5.y + I5.y * J5.x;	
	tmp_im	+= I5.z * J5.w + I5.w * J5.z;	
	accum_im[15*blockDim.x]	= tmp_im;	

	//Store output back to global buffer:

	/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *param.threads*2]	 = make_float2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *param.threads*2]	 = make_float2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *param.threads*2]	 = make_float2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *param.threads*2]	 = make_float2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *param.threads*2]	 = make_float2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *param.threads*2]	 = make_float2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *param.threads*2]	 = make_float2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *param.threads*2]	 = make_float2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *param.threads*2]	 = make_float2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *param.threads*2]	 = make_float2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*param.threads*2]	 = make_float2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*param.threads*2]	 = make_float2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*param.threads*2]	 = make_float2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*param.threads*2]	 = make_float2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*param.threads*2]	 = make_float2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*param.threads*2]	 = make_float2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	return;
}

#undef SPINORTEX
#undef INTERTEX

#undef tmp_re
#undef tmp_im

#undef	READ_SPINOR
#undef	READ_INTERMEDIATE_SPINOR

#undef	SPINOR_HOP

#endif //_TWIST_QUDA_CONTRACT

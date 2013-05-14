#ifndef _TWIST_QUDA_CONTRACT
#define _TWIST_QUDA_CONTRACT

#define tmp_re tmp.x
#define tmp_im tmp.y

#define TOTAL_COMPONENTS 16 

#if (__COMPUTE_CAPABILITY__ >= 130)
__global__ void contractGamma5KernelD(double2 *out, double2 *in1, double2 *in2, int maxThreads, int myStride, const int XL, const int YL, const int ZL, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= maxThreads)
		return;

	#ifndef USE_TEXTURE_OBJECTS
		#define SPINORTEX	spinorTexDouble
		#define INTERTEX	interTexDouble
	#else
		#define SPINORTEX	param.inTex
		#define INTERTEX	param.xTex
	#endif

	volatile double2		tmp;
	extern __shared__ double	sm[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile double			*accum_re = sm + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile double			*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / XL;
	xCoord1		 = eutId - auxCoord1 * XL;
	auxCoord2	 = auxCoord1 / YL;
	xCoord2		 = auxCoord1 - auxCoord2 * YL;
	xCoord4		 = auxCoord2 / ZL;
	xCoord3		 = auxCoord2 - xCoord4 * ZL;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + XL*(xCoord2 + YL*(xCoord3 + ZL*xCoord4));			//AQUI

	//Load the first color component for each input spinor:
	{

	double2 I0	 = fetch_double2(SPINORTEX, sid + 0 * myStride);   
	double2 I1	 = fetch_double2(SPINORTEX, sid + 3 * myStride);   
	double2 I2	 = fetch_double2(SPINORTEX, sid + 6 * myStride);   
	double2 I3	 = fetch_double2(SPINORTEX, sid + 9 * myStride);   
	double2 J0	 = fetch_double2(INTERTEX,  sid + 0 * myStride);   
	double2 J1	 = fetch_double2(INTERTEX,  sid + 3 * myStride);   
	double2 J2	 = fetch_double2(INTERTEX,  sid + 6 * myStride);   
	double2 J3	 = fetch_double2(INTERTEX,  sid + 9 * myStride);   
	
	//compute in1^dag * gamma5:

	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I2.x;
	I0.y	 = -I2.y;
	I2.x	 = tmp_re;
	I2.y	 = tmp_im;	

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I3.x;
	I1.y	 = -I3.y;
	I3.x	 = tmp_re;
	I3.y	 = tmp_im;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] = tmp_re;
	accum_im[0*blockDim.x] = tmp_im;	
	
	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] = tmp_re;
	accum_im[1*blockDim.x] = tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] = tmp_re;
	accum_im[2*blockDim.x] = tmp_im;	
      
	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] = tmp_re;
	accum_im[3*blockDim.x] = tmp_im;	
      
	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] = tmp_re;
	accum_im[4*blockDim.x] = tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] = tmp_re;
	accum_im[5*blockDim.x] = tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] = tmp_re;
	accum_im[6*blockDim.x] = tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] = tmp_re;
	accum_im[7*blockDim.x] = tmp_im;	

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] = tmp_re;
	accum_im[8*blockDim.x] = tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] = tmp_re;
	accum_im[9*blockDim.x] = tmp_im;	

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] = tmp_re;
	accum_im[10*blockDim.x] = tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] = tmp_re;
	accum_im[11*blockDim.x] = tmp_im;	

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] = tmp_re;
	accum_im[12*blockDim.x] = tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] = tmp_re;
	accum_im[13*blockDim.x] = tmp_im;	

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] = tmp_re;
	accum_im[14*blockDim.x] = tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] = tmp_re;
	accum_im[15*blockDim.x] = tmp_im;	
	}

	//Load the second color component for each input spinor:
	{

	double2 I0	 = fetch_double2(SPINORTEX, sid + 1 * myStride);   
	double2 I1	 = fetch_double2(SPINORTEX, sid + 4 * myStride);   
	double2 I2	 = fetch_double2(SPINORTEX, sid + 7 * myStride);   
	double2 I3	 = fetch_double2(SPINORTEX, sid + 10* myStride);   
	double2 J0	 = fetch_double2(INTERTEX,  sid + 1 * myStride);   
	double2 J1	 = fetch_double2(INTERTEX,  sid + 4 * myStride);   
	double2 J2	 = fetch_double2(INTERTEX,  sid + 7 * myStride);   
	double2 J3	 = fetch_double2(INTERTEX,  sid + 10* myStride);   

	//compute in1^dag * gamma5:
	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I2.x;
	I0.y	 = -I2.y;
	I2.x	 = tmp_re;
	I2.y	 = tmp_im;	

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I3.x;
	I1.y	 = -I3.y;
	I3.x	 = tmp_re;
	I3.y	 = tmp_im;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	
       
	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	
       
	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;	
       
	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;	
       
	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;	
       
	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;	
       
	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;	
	}

	//Load the third color component for each input spinor:
	{

        double2 I0  = fetch_double2(SPINORTEX, sid + 2 * myStride);   
	double2 I1  = fetch_double2(SPINORTEX, sid + 5 * myStride);   
	double2 I2  = fetch_double2(SPINORTEX, sid + 8 * myStride);   
	double2 I3  = fetch_double2(SPINORTEX, sid + 11* myStride);   
	double2 J0  = fetch_double2(INTERTEX,  sid + 2 * myStride);   
	double2 J1  = fetch_double2(INTERTEX,  sid + 5 * myStride);   
	double2 J2  = fetch_double2(INTERTEX,  sid + 8 * myStride);   
	double2 J3  = fetch_double2(INTERTEX,  sid + 11* myStride);   

	//compute in1^dag * gamma5:
	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I2.x;
	I0.y	 = -I2.y;
	I2.x	 = tmp_re;
	I2.y	 = tmp_im;	

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I3.x;
	I1.y	 = -I3.y;
	I3.x	 = tmp_re;
	I3.y	 = tmp_im;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}

   //Store output back to global buffer:


/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *maxThreads*2]	 = make_double2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *maxThreads*2]	 = make_double2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *maxThreads*2]	 = make_double2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *maxThreads*2]	 = make_double2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *maxThreads*2]	 = make_double2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *maxThreads*2]	 = make_double2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *maxThreads*2]	 = make_double2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *maxThreads*2]	 = make_double2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *maxThreads*2]	 = make_double2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *maxThreads*2]	 = make_double2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*maxThreads*2]	 = make_double2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*maxThreads*2]	 = make_double2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*maxThreads*2]	 = make_double2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*maxThreads*2]	 = make_double2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*maxThreads*2]	 = make_double2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*maxThreads*2]	 = make_double2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	#undef SPINORTEX
	#undef INTERTEX

	return;
}

//Perform trace in color space only and for a given tslice 
//since the file is included in dslash_quda.h, no need to add dslash_constants.h file here (for, e.g., Vsh)
__global__ void contractTsliceKernelD(double2 *out, double2 *in1, double2 *in2, int maxThreads, int myStride, const int XL, const int YL, const int ZL, const int Tslice, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;					//number of threads is equal to Tslice volume
												//Adjust sid to correct tslice (exe domain must be Tslice volume!)
	int	inId	 = sid + Vsh*Tslice;							//Vsh - 3d space volume for the parity spinor (equale to exe domain!)
	int	outId; 
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= maxThreads)								//maxThreads == tslice volume
		return;

	#ifndef USE_TEXTURE_OBJECTS
		#define SPINORTEX	spinorTexDouble
		#define INTERTEX	interTexDouble
	#else
		#define SPINORTEX	param.inTex
		#define INTERTEX	param.xTex
	#endif

	volatile double2		tmp;
	extern __shared__ double	sm[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile double			*accum_re = sm + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile double			*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

//The output only for a given tslice (for the full tslice content, i.e., both parities!):

	eutId		 = 2*inId;
	auxCoord1	 = eutId / XL;
	xCoord1		 = eutId - auxCoord1 * XL;
	auxCoord2	 = auxCoord1 / YL;
	xCoord2		 = auxCoord1 - auxCoord2 * YL;
	xCoord4		 = auxCoord2 / ZL;

//	if	(Tslice != xCoord4)
//		return;

	xCoord3		 = auxCoord2 - xCoord4 * ZL;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + XL*(xCoord2 + YL*xCoord3);					//AQUI

	//Load the first color component for each input spinor:
	{

	double2 I0	 = fetch_double2(SPINORTEX, inId + 0 * myStride);   
	double2 I1	 = fetch_double2(SPINORTEX, inId + 3 * myStride);   
	double2 I2	 = fetch_double2(SPINORTEX, inId + 6 * myStride);   
	double2 I3	 = fetch_double2(SPINORTEX, inId + 9 * myStride);   
	double2 J0	 = fetch_double2(INTERTEX,  inId + 0 * myStride);   
	double2 J1	 = fetch_double2(INTERTEX,  inId + 3 * myStride);   
	double2 J2	 = fetch_double2(INTERTEX,  inId + 6 * myStride);   
	double2 J3	 = fetch_double2(INTERTEX,  inId + 9 * myStride);   
	
	//compute in1^dag:

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;	
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;	
	
	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] = tmp_re;
	accum_im[0*blockDim.x] = tmp_im;	
	
	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] = tmp_re;
	accum_im[1*blockDim.x] = tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] = tmp_re;
	accum_im[2*blockDim.x] = tmp_im;	
      
	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] = tmp_re;
	accum_im[3*blockDim.x] = tmp_im;	
      
	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] = tmp_re;
	accum_im[4*blockDim.x] = tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] = tmp_re;
	accum_im[5*blockDim.x] = tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] = tmp_re;
	accum_im[6*blockDim.x] = tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] = tmp_re;
	accum_im[7*blockDim.x] = tmp_im;	

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] = tmp_re;
	accum_im[8*blockDim.x] = tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] = tmp_re;
	accum_im[9*blockDim.x] = tmp_im;	

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] = tmp_re;
	accum_im[10*blockDim.x] = tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] = tmp_re;
	accum_im[11*blockDim.x] = tmp_im;	

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] = tmp_re;
	accum_im[12*blockDim.x] = tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] = tmp_re;
	accum_im[13*blockDim.x] = tmp_im;	

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] = tmp_re;
	accum_im[14*blockDim.x] = tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] = tmp_re;
	accum_im[15*blockDim.x] = tmp_im;
	}

	//Load the second color component for each input spinor:
	{

	double2 I0	 = fetch_double2(SPINORTEX, inId + 1 * myStride);   
	double2 I1	 = fetch_double2(SPINORTEX, inId + 4 * myStride);   
	double2 I2	 = fetch_double2(SPINORTEX, inId + 7 * myStride);   
	double2 I3	 = fetch_double2(SPINORTEX, inId + 10* myStride);   
	double2 J0	 = fetch_double2(INTERTEX,  inId + 1 * myStride);   
	double2 J1	 = fetch_double2(INTERTEX,  inId + 4 * myStride);   
	double2 J2	 = fetch_double2(INTERTEX,  inId + 7 * myStride);   
	double2 J3	 = fetch_double2(INTERTEX,  inId + 10* myStride);   

	//compute in^dag
	I0.y	 = -I0.y;
	I1.y	 = -I1.y;	
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	
       
	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	
       
	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;	
       
	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;	
       
	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;
       
	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;
       
	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}

	//Load the third color component for each input spinor:
	{

        double2 I0  = fetch_double2(SPINORTEX, inId + 2 * myStride);   
	double2 I1  = fetch_double2(SPINORTEX, inId + 5 * myStride);   
	double2 I2  = fetch_double2(SPINORTEX, inId + 8 * myStride);   
	double2 I3  = fetch_double2(SPINORTEX, inId + 11* myStride);   
	double2 J0  = fetch_double2(INTERTEX,  inId + 2 * myStride);   
	double2 J1  = fetch_double2(INTERTEX,  inId + 5 * myStride);   
	double2 J2  = fetch_double2(INTERTEX,  inId + 8 * myStride);   
	double2 J3  = fetch_double2(INTERTEX,  inId + 11* myStride);   

	//compute in^dag
	I0.y	 = -I0.y;
	I1.y	 = -I1.y;	
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}


   //Store output back to global buffer:


/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *maxThreads*2]	 = make_double2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *maxThreads*2]	 = make_double2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *maxThreads*2]	 = make_double2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *maxThreads*2]	 = make_double2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *maxThreads*2]	 = make_double2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *maxThreads*2]	 = make_double2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *maxThreads*2]	 = make_double2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *maxThreads*2]	 = make_double2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *maxThreads*2]	 = make_double2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *maxThreads*2]	 = make_double2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*maxThreads*2]	 = make_double2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]);
	out[outId + 11*maxThreads*2]	 = make_double2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]);
	out[outId + 12*maxThreads*2]	 = make_double2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]);
	out[outId + 13*maxThreads*2]	 = make_double2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]);
	out[outId + 14*maxThreads*2]	 = make_double2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]);
	out[outId + 15*maxThreads*2]	 = make_double2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	#undef SPINORTEX
	#undef INTERTEX

	return;
}

__global__ void contractKernelD		(double2 *out, double2 *in1, double2 *in2, int maxThreads, int myStride, const int XL, const int YL, const int ZL, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= maxThreads)
		return;

	#ifndef USE_TEXTURE_OBJECTS
		#define SPINORTEX	spinorTexDouble
		#define INTERTEX	interTexDouble
	#else
		#define SPINORTEX	param.inTex
		#define INTERTEX	param.xTex
	#endif

	volatile double2		tmp;
	extern __shared__ double	sm[];								//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile double			*accum_re	 = sm + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile double			*accum_im	 = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / XL;
	xCoord1		 = eutId - auxCoord1 * XL;
	auxCoord2	 = auxCoord1 / YL;
	xCoord2		 = auxCoord1 - auxCoord2 * YL;
	xCoord4		 = auxCoord2 / ZL;
	xCoord3		 = auxCoord2 - xCoord4 * ZL;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + XL*(xCoord2 + YL*(xCoord3 + ZL*xCoord4));				//AQUI

	//Load the first color component for each input spinor:
	{

	double2 I0	 = fetch_double2(SPINORTEX, sid + 0 * myStride);   
	double2 I1	 = fetch_double2(SPINORTEX, sid + 3 * myStride);   
	double2 I2	 = fetch_double2(SPINORTEX, sid + 6 * myStride);   
	double2 I3	 = fetch_double2(SPINORTEX, sid + 9 * myStride);   
	double2 J0	 = fetch_double2(INTERTEX,  sid + 0 * myStride);   
	double2 J1	 = fetch_double2(INTERTEX,  sid + 3 * myStride);   
	double2 J2	 = fetch_double2(INTERTEX,  sid + 6 * myStride);   
	double2 J3	 = fetch_double2(INTERTEX,  sid + 9 * myStride);   
	
	//compute in1^dag

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] = tmp_re;
	accum_im[0*blockDim.x] = tmp_im;	
	
	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] = tmp_re;
	accum_im[1*blockDim.x] = tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] = tmp_re;
	accum_im[2*blockDim.x] = tmp_im;	
      
	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] = tmp_re;
	accum_im[3*blockDim.x] = tmp_im;	
      
	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] = tmp_re;
	accum_im[4*blockDim.x] = tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] = tmp_re;
	accum_im[5*blockDim.x] = tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] = tmp_re;
	accum_im[6*blockDim.x] = tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] = tmp_re;
	accum_im[7*blockDim.x] = tmp_im;	

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] = tmp_re;
	accum_im[8*blockDim.x] = tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] = tmp_re;
	accum_im[9*blockDim.x] = tmp_im;	

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] = tmp_re;
	accum_im[10*blockDim.x] = tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] = tmp_re;
	accum_im[11*blockDim.x] = tmp_im;	

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] = tmp_re;
	accum_im[12*blockDim.x] = tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] = tmp_re;
	accum_im[13*blockDim.x] = tmp_im;	

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] = tmp_re;
	accum_im[14*blockDim.x] = tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] = tmp_re;
	accum_im[15*blockDim.x] = tmp_im;	
	}

	//Load the second color component for each input spinor:
	{

	double2 I0	 = fetch_double2(SPINORTEX, sid + 1 * myStride);   
	double2 I1	 = fetch_double2(SPINORTEX, sid + 4 * myStride);   
	double2 I2	 = fetch_double2(SPINORTEX, sid + 7 * myStride);   
	double2 I3	 = fetch_double2(SPINORTEX, sid + 10* myStride);   
	double2 J0	 = fetch_double2(INTERTEX,  sid + 1 * myStride);   
	double2 J1	 = fetch_double2(INTERTEX,  sid + 4 * myStride);   
	double2 J2	 = fetch_double2(INTERTEX,  sid + 7 * myStride);   
	double2 J3	 = fetch_double2(INTERTEX,  sid + 10* myStride);   

	//compute in1^dag * gamma5:

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	
       
	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	
       
	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;	
       
	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;	
       
	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;	
       
	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;	
       
	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;	
	}

	//Load the third color component for each input spinor:
	{

        double2 I0  = fetch_double2(SPINORTEX, sid + 2 * myStride);   
	double2 I1  = fetch_double2(SPINORTEX, sid + 5 * myStride);   
	double2 I2  = fetch_double2(SPINORTEX, sid + 8 * myStride);   
	double2 I3  = fetch_double2(SPINORTEX, sid + 11* myStride);   
	double2 J0  = fetch_double2(INTERTEX,  sid + 2 * myStride);   
	double2 J1  = fetch_double2(INTERTEX,  sid + 5 * myStride);   
	double2 J2  = fetch_double2(INTERTEX,  sid + 8 * myStride);   
	double2 J3  = fetch_double2(INTERTEX,  sid + 11* myStride);   

	//compute in1^dag

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}

	//Store output back to global buffer:

/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *maxThreads*2]	 = make_double2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *maxThreads*2]	 = make_double2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *maxThreads*2]	 = make_double2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *maxThreads*2]	 = make_double2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *maxThreads*2]	 = make_double2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *maxThreads*2]	 = make_double2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *maxThreads*2]	 = make_double2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *maxThreads*2]	 = make_double2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *maxThreads*2]	 = make_double2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *maxThreads*2]	 = make_double2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*maxThreads*2]	 = make_double2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*maxThreads*2]	 = make_double2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*maxThreads*2]	 = make_double2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*maxThreads*2]	 = make_double2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*maxThreads*2]	 = make_double2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*maxThreads*2]	 = make_double2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	#undef SPINORTEX
	#undef INTERTEX

	return;
}

#endif // (__CUDA_ARCH__ >= 130)

__global__ void contractGamma5KernelS(float2 *out, float2 *in1, float2 *in2, int maxThreads, int myStride, const int XL, const int YL, const int ZL, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= maxThreads)
		return;

	#ifndef USE_TEXTURE_OBJECTS
		#define SPINORTEX	spinorTexSingle2
		#define INTERTEX	interTexSingle2
	#else
		#define SPINORTEX	param.inTex
		#define INTERTEX	param.xTex
	#endif

	volatile float2		tmp;
	extern __shared__ float	sms[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile float		*accum_re = sms + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile float		*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / XL;
	xCoord1		 = eutId - auxCoord1 * XL;
	auxCoord2	 = auxCoord1 / YL;
	xCoord2		 = auxCoord1 - auxCoord2 * YL;
	xCoord4		 = auxCoord2 / ZL;
	xCoord3		 = auxCoord2 - xCoord4 * ZL;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + XL*(xCoord2 + YL*(xCoord3 + ZL*xCoord4));			//AQUI

	//Load the first color component for each input spinor:
	{

	float2 I0	 = TEX1DFETCH(float2, SPINORTEX, sid + 0 * myStride);   
	float2 I1	 = TEX1DFETCH(float2, SPINORTEX, sid + 3 * myStride);   
	float2 I2	 = TEX1DFETCH(float2, SPINORTEX, sid + 6 * myStride);   
	float2 I3	 = TEX1DFETCH(float2, SPINORTEX, sid + 9 * myStride);   
	float2 J0	 = TEX1DFETCH(float2, INTERTEX,  sid + 0 * myStride);   
	float2 J1	 = TEX1DFETCH(float2, INTERTEX,  sid + 3 * myStride);   
	float2 J2	 = TEX1DFETCH(float2, INTERTEX,  sid + 6 * myStride);   
	float2 J3	 = TEX1DFETCH(float2, INTERTEX,  sid + 9 * myStride);   
	
	//compute in1^dag * gamma5:

	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I2.x;
	I0.y	 = -I2.y;
	I2.x	 = tmp_re;
	I2.y	 = tmp_im;	

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I3.x;
	I1.y	 = -I3.y;
	I3.x	 = tmp_re;
	I3.y	 = tmp_im;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] = tmp_re;
	accum_im[0*blockDim.x] = tmp_im;	
	
	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] = tmp_re;
	accum_im[1*blockDim.x] = tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] = tmp_re;
	accum_im[2*blockDim.x] = tmp_im;	
      
	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] = tmp_re;
	accum_im[3*blockDim.x] = tmp_im;	
      
	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] = tmp_re;
	accum_im[4*blockDim.x] = tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] = tmp_re;
	accum_im[5*blockDim.x] = tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] = tmp_re;
	accum_im[6*blockDim.x] = tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] = tmp_re;
	accum_im[7*blockDim.x] = tmp_im;	

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] = tmp_re;
	accum_im[8*blockDim.x] = tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] = tmp_re;
	accum_im[9*blockDim.x] = tmp_im;	

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] = tmp_re;
	accum_im[10*blockDim.x] = tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] = tmp_re;
	accum_im[11*blockDim.x] = tmp_im;	

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] = tmp_re;
	accum_im[12*blockDim.x] = tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] = tmp_re;
	accum_im[13*blockDim.x] = tmp_im;	

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] = tmp_re;
	accum_im[14*blockDim.x] = tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] = tmp_re;
	accum_im[15*blockDim.x] = tmp_im;	
	}

	//Load the second color component for each input spinor:
	{

	float2 I0	 = TEX1DFETCH(float2, SPINORTEX, sid + 1 * myStride);   
	float2 I1	 = TEX1DFETCH(float2, SPINORTEX, sid + 4 * myStride);   
	float2 I2	 = TEX1DFETCH(float2, SPINORTEX, sid + 7 * myStride);   
	float2 I3	 = TEX1DFETCH(float2, SPINORTEX, sid + 10* myStride);   
	float2 J0	 = TEX1DFETCH(float2, INTERTEX,  sid + 1 * myStride);   
	float2 J1	 = TEX1DFETCH(float2, INTERTEX,  sid + 4 * myStride);   
	float2 J2	 = TEX1DFETCH(float2, INTERTEX,  sid + 7 * myStride);   
	float2 J3	 = TEX1DFETCH(float2, INTERTEX,  sid + 10* myStride);   

	//compute in1^dag * gamma5:
	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I2.x;
	I0.y	 = -I2.y;
	I2.x	 = tmp_re;
	I2.y	 = tmp_im;	

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I3.x;
	I1.y	 = -I3.y;
	I3.x	 = tmp_re;
	I3.y	 = tmp_im;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	
       
	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	
       
	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;	
       
	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;	
       
	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;	
       
	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;	
       
	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;	
	}

	//Load the third color component for each input spinor:
	{

	float2 I0	 = TEX1DFETCH(float2, SPINORTEX, sid + 2 * myStride);   
	float2 I1	 = TEX1DFETCH(float2, SPINORTEX, sid + 5 * myStride);   
	float2 I2	 = TEX1DFETCH(float2, SPINORTEX, sid + 8 * myStride);   
	float2 I3	 = TEX1DFETCH(float2, SPINORTEX, sid + 11* myStride);   
	float2 J0	 = TEX1DFETCH(float2, INTERTEX,  sid + 2 * myStride);   
	float2 J1	 = TEX1DFETCH(float2, INTERTEX,  sid + 5 * myStride);   
	float2 J2	 = TEX1DFETCH(float2, INTERTEX,  sid + 8 * myStride);   
	float2 J3	 = TEX1DFETCH(float2, INTERTEX,  sid + 11* myStride);   

	//compute in1^dag * gamma5:
	tmp_re	 = +I0.x;
	tmp_im	 = -I0.y;
	I0.x	 = +I2.x;
	I0.y	 = -I2.y;
	I2.x	 = tmp_re;
	I2.y	 = tmp_im;	

	tmp_re	 = +I1.x;
	tmp_im	 = -I1.y;
	I1.x	 = +I3.x;
	I1.y	 = -I3.y;
	I3.x	 = tmp_re;
	I3.y	 = tmp_im;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}

   //Store output back to global buffer:


/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *maxThreads*2]	 = make_float2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *maxThreads*2]	 = make_float2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *maxThreads*2]	 = make_float2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *maxThreads*2]	 = make_float2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *maxThreads*2]	 = make_float2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *maxThreads*2]	 = make_float2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *maxThreads*2]	 = make_float2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *maxThreads*2]	 = make_float2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *maxThreads*2]	 = make_float2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *maxThreads*2]	 = make_float2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*maxThreads*2]	 = make_float2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*maxThreads*2]	 = make_float2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*maxThreads*2]	 = make_float2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*maxThreads*2]	 = make_float2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*maxThreads*2]	 = make_float2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*maxThreads*2]	 = make_float2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	#undef SPINORTEX
	#undef INTERTEX

	return;
}

//Perform trace in color space only and for a given tslice 
//since the file is included in dslash_quda.h, no need to add dslash_constants.h file here (for, e.g., Vsh)
__global__ void contractTsliceKernelS(float2 *out, float2 *in1, float2 *in2, int maxThreads, int myStride, const int XL, const int YL, const int ZL, const int Tslice, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;					//number of threads is equal to Tslice volume
												//Adjust sid to correct tslice (exe domain must be Tslice volume!)
	int	inId	 = sid + Vsh*Tslice;							//Vsh - 3d space volume for the parity spinor (equale to exe domain!)
	int	outId; 
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= maxThreads)								//maxThreads == tslice volume
		return;

	#ifndef USE_TEXTURE_OBJECTS
		#define SPINORTEX	spinorTexSingle2
		#define INTERTEX	interTexSingle2
	#else
		#define SPINORTEX	param.inTex
		#define INTERTEX	param.xTex
	#endif

	volatile float2		tmp;
	extern __shared__ float	sms[];							//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile float			*accum_re = sms + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile float			*accum_im = accum_re + TOTAL_COMPONENTS*blockDim.x;

//The output only for a given tslice (for the full tslice content, i.e., both parities!):

	eutId		 = 2*inId;
	auxCoord1	 = eutId / XL;
	xCoord1		 = eutId - auxCoord1 * XL;
	auxCoord2	 = auxCoord1 / YL;
	xCoord2		 = auxCoord1 - auxCoord2 * YL;
	xCoord4		 = auxCoord2 / ZL;

//	if	(Tslice != xCoord4)
//		return;

	xCoord3		 = auxCoord2 - xCoord4 * ZL;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + XL*(xCoord2 + YL*xCoord3);					//AQUI

	//Load the first color component for each input spinor:
	{

	float2 I0  = TEX1DFETCH(float2, SPINORTEX, inId + 0 * myStride);   
	float2 I1  = TEX1DFETCH(float2, SPINORTEX, inId + 3 * myStride);   
	float2 I2  = TEX1DFETCH(float2, SPINORTEX, inId + 6 * myStride);   
	float2 I3  = TEX1DFETCH(float2, SPINORTEX, inId + 9 * myStride);   
	float2 J0  = TEX1DFETCH(float2, INTERTEX,  inId + 0 * myStride);   
	float2 J1  = TEX1DFETCH(float2, INTERTEX,  inId + 3 * myStride);   
	float2 J2  = TEX1DFETCH(float2, INTERTEX,  inId + 6 * myStride);   
	float2 J3  = TEX1DFETCH(float2, INTERTEX,  inId + 9 * myStride);   
	
	//compute in1^dag:

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;	
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;	
	
	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] = tmp_re;
	accum_im[0*blockDim.x] = tmp_im;	
	
	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] = tmp_re;
	accum_im[1*blockDim.x] = tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] = tmp_re;
	accum_im[2*blockDim.x] = tmp_im;	
      
	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] = tmp_re;
	accum_im[3*blockDim.x] = tmp_im;	
      
	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] = tmp_re;
	accum_im[4*blockDim.x] = tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] = tmp_re;
	accum_im[5*blockDim.x] = tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] = tmp_re;
	accum_im[6*blockDim.x] = tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] = tmp_re;
	accum_im[7*blockDim.x] = tmp_im;	

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] = tmp_re;
	accum_im[8*blockDim.x] = tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] = tmp_re;
	accum_im[9*blockDim.x] = tmp_im;	

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] = tmp_re;
	accum_im[10*blockDim.x] = tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] = tmp_re;
	accum_im[11*blockDim.x] = tmp_im;	

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] = tmp_re;
	accum_im[12*blockDim.x] = tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] = tmp_re;
	accum_im[13*blockDim.x] = tmp_im;	

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] = tmp_re;
	accum_im[14*blockDim.x] = tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] = tmp_re;
	accum_im[15*blockDim.x] = tmp_im;
	}

	//Load the second color component for each input spinor:
	{

	float2 I0  = TEX1DFETCH(float2, SPINORTEX, inId + 1 * myStride);   
	float2 I1  = TEX1DFETCH(float2, SPINORTEX, inId + 4 * myStride);   
	float2 I2  = TEX1DFETCH(float2, SPINORTEX, inId + 7 * myStride);   
	float2 I3  = TEX1DFETCH(float2, SPINORTEX, inId + 10* myStride);   
	float2 J0  = TEX1DFETCH(float2, INTERTEX,  inId + 1 * myStride);   
	float2 J1  = TEX1DFETCH(float2, INTERTEX,  inId + 4 * myStride);   
	float2 J2  = TEX1DFETCH(float2, INTERTEX,  inId + 7 * myStride);   
	float2 J3  = TEX1DFETCH(float2, INTERTEX,  inId + 10* myStride);   

	//compute in^dag
	I0.y	 = -I0.y;
	I1.y	 = -I1.y;	
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	
       
	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	
       
	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;	
       
	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;	
       
	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;
       
	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;
       
	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}

	//Load the third color component for each input spinor:
	{

	float2 I0  = TEX1DFETCH(float2, SPINORTEX, inId + 2 * myStride);   
	float2 I1  = TEX1DFETCH(float2, SPINORTEX, inId + 5 * myStride);   
	float2 I2  = TEX1DFETCH(float2, SPINORTEX, inId + 8 * myStride);   
	float2 I3  = TEX1DFETCH(float2, SPINORTEX, inId + 11* myStride);   
	float2 J0  = TEX1DFETCH(float2, INTERTEX,  inId + 2 * myStride);   
	float2 J1  = TEX1DFETCH(float2, INTERTEX,  inId + 5 * myStride);   
	float2 J2  = TEX1DFETCH(float2, INTERTEX,  inId + 8 * myStride);   
	float2 J3  = TEX1DFETCH(float2, INTERTEX,  inId + 11* myStride);   

	//compute in^dag
	I0.y	 = -I0.y;
	I1.y	 = -I1.y;	
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;	

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}


   //Store output back to global buffer:


/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *maxThreads*2]	 = make_float2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *maxThreads*2]	 = make_float2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *maxThreads*2]	 = make_float2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *maxThreads*2]	 = make_float2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *maxThreads*2]	 = make_float2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *maxThreads*2]	 = make_float2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *maxThreads*2]	 = make_float2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *maxThreads*2]	 = make_float2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *maxThreads*2]	 = make_float2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *maxThreads*2]	 = make_float2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*maxThreads*2]	 = make_float2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]);
	out[outId + 11*maxThreads*2]	 = make_float2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]);
	out[outId + 12*maxThreads*2]	 = make_float2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]);
	out[outId + 13*maxThreads*2]	 = make_float2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]);
	out[outId + 14*maxThreads*2]	 = make_float2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]);
	out[outId + 15*maxThreads*2]	 = make_float2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	#undef SPINORTEX
	#undef INTERTEX

	return;
}

__global__ void contractKernelS		(float2 *out, float2 *in1, float2 *in2, int maxThreads, int myStride, const int XL, const int YL, const int ZL, const int Parity, const DslashParam param)
{
	int	sid	 = blockIdx.x*blockDim.x + threadIdx.x;
	int	outId	 = sid;
	int	eutId, xCoord1, xCoord2, xCoord3, xCoord4, auxCoord1, auxCoord2;

	if	(sid >= maxThreads)
		return;

	#ifndef USE_TEXTURE_OBJECTS
		#define SPINORTEX	spinorTexSingle2
		#define INTERTEX	interTexSingle2
	#else
		#define SPINORTEX	param.inTex
		#define INTERTEX	param.xTex
	#endif

	volatile float2		tmp;
	extern __shared__ float	sms[];								//used for data accumulation: blockDim.x * 2 * 16 * sizeof(double)
   
	volatile float			*accum_re	 = sms + threadIdx.x;				//address it like idx*blockDim, where idx = 4*spinor_idx1 + spinor_idx2
	volatile float			*accum_im	 = accum_re + TOTAL_COMPONENTS*blockDim.x;

	eutId		 = 2*sid;
	auxCoord1	 = eutId / XL;
	xCoord1		 = eutId - auxCoord1 * XL;
	auxCoord2	 = auxCoord1 / YL;
	xCoord2		 = auxCoord1 - auxCoord2 * YL;
	xCoord4		 = auxCoord2 / ZL;
	xCoord3		 = auxCoord2 - xCoord4 * ZL;

	auxCoord1	 = (Parity + xCoord4 + xCoord3 + xCoord2) & 1;
	xCoord1		+= auxCoord1;
	outId		 = xCoord1 + XL*(xCoord2 + YL*(xCoord3 + ZL*xCoord4));				//AQUI

	//Load the first color component for each input spinor:
	{

	float2 I0	 = TEX1DFETCH(float2, SPINORTEX, sid + 0 * myStride);   
	float2 I1	 = TEX1DFETCH(float2, SPINORTEX, sid + 3 * myStride);   
	float2 I2	 = TEX1DFETCH(float2, SPINORTEX, sid + 6 * myStride);   
	float2 I3	 = TEX1DFETCH(float2, SPINORTEX, sid + 9 * myStride);   
	float2 J0	 = TEX1DFETCH(float2, INTERTEX,  sid + 0 * myStride);   
	float2 J1	 = TEX1DFETCH(float2, INTERTEX,  sid + 3 * myStride);   
	float2 J2	 = TEX1DFETCH(float2, INTERTEX,  sid + 6 * myStride);   
	float2 J3	 = TEX1DFETCH(float2, INTERTEX,  sid + 9 * myStride);   
	
	//compute in1^dag

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] = tmp_re;
	accum_im[0*blockDim.x] = tmp_im;	
	
	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] = tmp_re;
	accum_im[1*blockDim.x] = tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] = tmp_re;
	accum_im[2*blockDim.x] = tmp_im;	
      
	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] = tmp_re;
	accum_im[3*blockDim.x] = tmp_im;	
      
	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] = tmp_re;
	accum_im[4*blockDim.x] = tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] = tmp_re;
	accum_im[5*blockDim.x] = tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] = tmp_re;
	accum_im[6*blockDim.x] = tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] = tmp_re;
	accum_im[7*blockDim.x] = tmp_im;	

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] = tmp_re;
	accum_im[8*blockDim.x] = tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] = tmp_re;
	accum_im[9*blockDim.x] = tmp_im;	

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] = tmp_re;
	accum_im[10*blockDim.x] = tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] = tmp_re;
	accum_im[11*blockDim.x] = tmp_im;	

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] = tmp_re;
	accum_im[12*blockDim.x] = tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] = tmp_re;
	accum_im[13*blockDim.x] = tmp_im;	

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] = tmp_re;
	accum_im[14*blockDim.x] = tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] = tmp_re;
	accum_im[15*blockDim.x] = tmp_im;	
	}

	//Load the second color component for each input spinor:
	{

	float2 I0	 = TEX1DFETCH(float2, SPINORTEX, sid + 1 * myStride);
	float2 I1	 = TEX1DFETCH(float2, SPINORTEX, sid + 4 * myStride);
	float2 I2	 = TEX1DFETCH(float2, SPINORTEX, sid + 7 * myStride);
	float2 I3	 = TEX1DFETCH(float2, SPINORTEX, sid + 10* myStride);
	float2 J0	 = TEX1DFETCH(float2, INTERTEX,  sid + 1 * myStride);
	float2 J1	 = TEX1DFETCH(float2, INTERTEX,  sid + 4 * myStride);
	float2 J2	 = TEX1DFETCH(float2, INTERTEX,  sid + 7 * myStride);
	float2 J3	 = TEX1DFETCH(float2, INTERTEX,  sid + 10 * myStride);

	//compute in1^dag * gamma5:

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	
       
	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	
       
	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;	
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;	
       
	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;	
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;	

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;	
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;	
       
	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;	
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;	

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;	
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;	
       
	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;	
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;	

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;	
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;	
       
	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;	
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;	

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;	
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;	
	}

	//Load the third color component for each input spinor:
	{

	float2 I0	 = TEX1DFETCH(float2, SPINORTEX, sid + 2 * myStride);
	float2 I1	 = TEX1DFETCH(float2, SPINORTEX, sid + 5 * myStride);
	float2 I2	 = TEX1DFETCH(float2, SPINORTEX, sid + 8 * myStride);
	float2 I3	 = TEX1DFETCH(float2, SPINORTEX, sid + 11* myStride);
	float2 J0	 = TEX1DFETCH(float2, INTERTEX,  sid + 2 * myStride);
	float2 J1	 = TEX1DFETCH(float2, INTERTEX,  sid + 5 * myStride);
	float2 J2	 = TEX1DFETCH(float2, INTERTEX,  sid + 8 * myStride);
	float2 J3	 = TEX1DFETCH(float2, INTERTEX,  sid + 11 * myStride);

	//compute in1^dag

	I0.y	 = -I0.y;
	I1.y	 = -I1.y;
	I2.y	 = -I2.y;
	I3.y	 = -I3.y;

	//do products for first color component here:
	//00 component:
	tmp_re = I0.x * J0.x - I0.y * J0.y;
	tmp_im = I0.x * J0.y + I0.y * J0.x;	
	accum_re[0*blockDim.x] += tmp_re;
	accum_im[0*blockDim.x] += tmp_im;	

	//01 component:
	tmp_re = I0.x * J1.x - I0.y * J1.y;
	tmp_im = I0.x * J1.y + I0.y * J1.x;	
	accum_re[1*blockDim.x] += tmp_re;
	accum_im[1*blockDim.x] += tmp_im;	

	//02 component:
	tmp_re = I0.x * J2.x - I0.y * J2.y;
	tmp_im = I0.x * J2.y + I0.y * J2.x;	
	accum_re[2*blockDim.x] += tmp_re;
	accum_im[2*blockDim.x] += tmp_im;	

	//03 component:
	tmp_re = I0.x * J3.x - I0.y * J3.y;
	tmp_im = I0.x * J3.y + I0.y * J3.x;	
	accum_re[3*blockDim.x] += tmp_re;
	accum_im[3*blockDim.x] += tmp_im;	

	//10 component:
	tmp_re = I1.x * J0.x - I1.y * J0.y;
	tmp_im = I1.x * J0.y + I1.y * J0.x;	
	accum_re[4*blockDim.x] += tmp_re;
	accum_im[4*blockDim.x] += tmp_im;	

	//11 component:
	tmp_re = I1.x * J1.x - I1.y * J1.y;
	tmp_im = I1.x * J1.y + I1.y * J1.x;	
	accum_re[5*blockDim.x] += tmp_re;
	accum_im[5*blockDim.x] += tmp_im;	

	//12 component:
	tmp_re = I1.x * J2.x - I1.y * J2.y;
	tmp_im = I1.x * J2.y + I1.y * J2.x;	
	accum_re[6*blockDim.x] += tmp_re;
	accum_im[6*blockDim.x] += tmp_im;	

	//13 component:
	tmp_re = I1.x * J3.x - I1.y * J3.y;
	tmp_im = I1.x * J3.y + I1.y * J3.x;
	accum_re[7*blockDim.x] += tmp_re;
	accum_im[7*blockDim.x] += tmp_im;

	//20 component:
	tmp_re = I2.x * J0.x - I2.y * J0.y;
	tmp_im = I2.x * J0.y + I2.y * J0.x;
	accum_re[8*blockDim.x] += tmp_re;
	accum_im[8*blockDim.x] += tmp_im;

	//21 component:
	tmp_re = I2.x * J1.x - I2.y * J1.y;
	tmp_im = I2.x * J1.y + I2.y * J1.x;
	accum_re[9*blockDim.x] += tmp_re;
	accum_im[9*blockDim.x] += tmp_im;

	//22 component:
	tmp_re = I2.x * J2.x - I2.y * J2.y;
	tmp_im = I2.x * J2.y + I2.y * J2.x;
	accum_re[10*blockDim.x] += tmp_re;
	accum_im[10*blockDim.x] += tmp_im;

	//23 component:
	tmp_re = I2.x * J3.x - I2.y * J3.y;
	tmp_im = I2.x * J3.y + I2.y * J3.x;
	accum_re[11*blockDim.x] += tmp_re;
	accum_im[11*blockDim.x] += tmp_im;

	//30 component:
	tmp_re = I3.x * J0.x - I3.y * J0.y;
	tmp_im = I3.x * J0.y + I3.y * J0.x;
	accum_re[12*blockDim.x] += tmp_re;
	accum_im[12*blockDim.x] += tmp_im;

	//31 component:
	tmp_re = I3.x * J1.x - I3.y * J1.y;
	tmp_im = I3.x * J1.y + I3.y * J1.x;
	accum_re[13*blockDim.x] += tmp_re;
	accum_im[13*blockDim.x] += tmp_im;

	//32 component:
	tmp_re = I3.x * J2.x - I3.y * J2.y;
	tmp_im = I3.x * J2.y + I3.y * J2.x;
	accum_re[14*blockDim.x] += tmp_re;
	accum_im[14*blockDim.x] += tmp_im;

	//33 component:
	tmp_re = I3.x * J3.x - I3.y * J3.y;
	tmp_im = I3.x * J3.y + I3.y * J3.x;
	accum_re[15*blockDim.x] += tmp_re;
	accum_im[15*blockDim.x] += tmp_im;
	}

	//Store output back to global buffer:

/*	CONTRACTION FULL VOLUME		*/

	out[outId + 0 *maxThreads*2]	 = make_float2(accum_re[ 0*blockDim.x], accum_im[ 0*blockDim.x]);
	out[outId + 1 *maxThreads*2]	 = make_float2(accum_re[ 1*blockDim.x], accum_im[ 1*blockDim.x]);
	out[outId + 2 *maxThreads*2]	 = make_float2(accum_re[ 2*blockDim.x], accum_im[ 2*blockDim.x]);
	out[outId + 3 *maxThreads*2]	 = make_float2(accum_re[ 3*blockDim.x], accum_im[ 3*blockDim.x]);
	out[outId + 4 *maxThreads*2]	 = make_float2(accum_re[ 4*blockDim.x], accum_im[ 4*blockDim.x]);
	out[outId + 5 *maxThreads*2]	 = make_float2(accum_re[ 5*blockDim.x], accum_im[ 5*blockDim.x]);
	out[outId + 6 *maxThreads*2]	 = make_float2(accum_re[ 6*blockDim.x], accum_im[ 6*blockDim.x]);
	out[outId + 7 *maxThreads*2]	 = make_float2(accum_re[ 7*blockDim.x], accum_im[ 7*blockDim.x]);
	out[outId + 8 *maxThreads*2]	 = make_float2(accum_re[ 8*blockDim.x], accum_im[ 8*blockDim.x]);
	out[outId + 9 *maxThreads*2]	 = make_float2(accum_re[ 9*blockDim.x], accum_im[ 9*blockDim.x]);
	out[outId + 10*maxThreads*2]	 = make_float2(accum_re[10*blockDim.x], accum_im[10*blockDim.x]); 
	out[outId + 11*maxThreads*2]	 = make_float2(accum_re[11*blockDim.x], accum_im[11*blockDim.x]); 
	out[outId + 12*maxThreads*2]	 = make_float2(accum_re[12*blockDim.x], accum_im[12*blockDim.x]); 
	out[outId + 13*maxThreads*2]	 = make_float2(accum_re[13*blockDim.x], accum_im[13*blockDim.x]); 
	out[outId + 14*maxThreads*2]	 = make_float2(accum_re[14*blockDim.x], accum_im[14*blockDim.x]); 
	out[outId + 15*maxThreads*2]	 = make_float2(accum_re[15*blockDim.x], accum_im[15*blockDim.x]);

	#undef SPINORTEX
	#undef INTERTEX

	return;
}

#undef tmp_re
#undef tmp_im

#endif //_TWIST_QUDA_CONTRACT

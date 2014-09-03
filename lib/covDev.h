// DD_NAME_F = covDevM#
// DD_RECON_F = 8, 12, 18
// DD_DAG_F = Dagger, [blank]
// covDevM012() (no dagger)
//
// In addition, the kernels are templated on the precision of the
// fields (double, single, or half).

// initialize on first iteration

#ifndef DD_LOOP
	#define DD_LOOP
	#define DD_DAG 0
	#define DD_RECON 0
	#define	DD_PREC 0
#endif

// set options for current iteration

#if (DD_PREC == 0)
	#define DD_PARAM2 const double2 *gauge0, const double2 *gauge1
#else 
	#define DD_PARAM2 const float4  *gauge0, const float4  *gauge1
#endif

#if (DD_DAG==0) // no dagger
	#define DD_DAG_F
#else           // dagger
	#define DD_DAG_F Dagger
#endif

#define DD_PARAM4 const DslashParam param


#if (DD_RECON==0) // reconstruct from 8 reals
	#define DD_RECON_F 8

	#if (DD_PREC == 0)
		#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_DOUBLE

		#ifdef DIRECT_ACCESS_LINK
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_DOUBLE2
		#else 
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_DOUBLE2_TEX
		#endif // DIRECT_ACCESS_LINK
	#elif (DD_PREC == 1)
		#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_SINGLE

		#ifdef DIRECT_ACCESS_LINK
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_FLOAT4
		#else 
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_FLOAT4_TEX
		#endif
	#endif
#elif (DD_RECON==1) // reconstruct from 12 reals
	#define DD_RECON_F 12

	#if (DD_PREC == 0)
		#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_DOUBLE

		#ifdef DIRECT_ACCESS_LINK
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_DOUBLE2
		#else 
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_DOUBLE2_TEX
		#endif // DIRECT_ACCESS_LINK
	#elif (DD_PREC == 1)
		#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_SINGLE

		#ifdef DIRECT_ACCESS_LINK
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_FLOAT4
		#else 
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_FLOAT4_TEX
		#endif
	#endif
#else               // no reconstruct, load all components
	#define DD_RECON_F 18
	#define GAUGE_FLOAT2

	#if (DD_PREC == 0)
		#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_18_DOUBLE

		#ifdef DIRECT_ACCESS_LINK
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_18_DOUBLE2
		#else 
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_18_DOUBLE2_TEX
		#endif // DIRECT_ACCESS_LINK
	#elif (DD_PREC == 1)
		#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_18_SINGLE

		#ifdef DIRECT_ACCESS_LINK
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_18_FLOAT2
		#else 
			#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_18_FLOAT2_TEX
		#endif
	#endif
#endif


#if	(DD_PREC==0)

	#define TPROJSCALE tProjScale

	// double-precision gauge field
	#if (defined DIRECT_ACCESS_LINK) || (defined FERMI_NO_DBLE_TEX)
		#define GAUGE0TEX gauge0
		#define GAUGE1TEX gauge1
	#else
		#ifdef USE_TEXTURE_OBJECTS
			#define GAUGE0TEX param.gauge0Tex
			#define GAUGE1TEX param.gauge1Tex
		#else
			#define GAUGE0TEX gauge0TexDouble2
			#define GAUGE1TEX gauge1TexDouble2
		#endif
	#endif

	#define GAUGE_FLOAT2

	// double-precision spinor fields
	#define DD_PARAM1 double2* out
	#define DD_PARAM3 const double2* in

	#if (defined DIRECT_ACCESS_WILSON_SPINOR) || (defined FERMI_NO_DBLE_TEX)
		#define READ_SPINOR READ_SPINOR_DOUBLE
		#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
		#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
		#define SPINORTEX in
	#else
		#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
		#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
		#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX

		#ifdef USE_TEXTURE_OBJECTS
			#define SPINORTEX param.inTex
		#else
			#define SPINORTEX spinorTexDouble
		#endif	// USE_TEXTURE_OBJECTS
	#endif

	#if (defined DIRECT_ACCESS_WILSON_INTER) || (defined FERMI_NO_DBLE_TEX)
		#define READ_INTERMEDIATE_SPINOR READ_SPINOR_DOUBLE
		#define INTERTEX out
	#else
		#define READ_INTERMEDIATE_SPINOR READ_SPINOR_DOUBLE_TEX

		#ifdef USE_TEXTURE_OBJECTS
			#define INTERTEX param.outTex
		#else
			#define INTERTEX interTexDouble
		#endif
	#endif


	#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2_STR
	//#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2
	#define SPINOR_DOUBLE

	#define SPINOR_HOP 12

#else
	#define TPROJSCALE tProjScale_f

	// single-precision gauge field
	#ifdef DIRECT_ACCESS_LINK
		#define GAUGE0TEX gauge0
		#define GAUGE1TEX gauge1
	#else
		#ifdef USE_TEXTURE_OBJECTS
			#define GAUGE0TEX param.gauge0Tex
			#define GAUGE1TEX param.gauge1Tex
		#else
			#if (DD_RECON_F == 18)
				#define GAUGE0TEX gauge0TexSingle2
				#define GAUGE1TEX gauge1TexSingle2
			#else
				#define GAUGE0TEX gauge0TexSingle4
				#define GAUGE1TEX gauge1TexSingle4
			#endif
		#endif // USE_TEXTURE_OBJECTS
	#endif


	// single-precision spinor fields
	#define DD_PARAM1 float4* out
	#define DD_PARAM3 const float4* in

	#ifdef DIRECT_ACCESS_WILSON_SPINOR
		#define READ_SPINOR READ_SPINOR_SINGLE
		#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
		#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
		#define SPINORTEX in
	#else
		#define READ_SPINOR READ_SPINOR_SINGLE_TEX
		#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
		#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX

		#ifdef USE_TEXTURE_OBJECTS
			#define SPINORTEX param.inTex
		#else
			#define SPINORTEX spinorTexSingle
		#endif // USE_TEXTURE_OBJECTS
	#endif

	#ifdef DIRECT_ACCESS_WILSON_INTER
		#define READ_INTERMEDIATE_SPINOR READ_SPINOR_SINGLE
		#define INTERTEX out
	#else
		#define READ_INTERMEDIATE_SPINOR READ_SPINOR_SINGLE_TEX

		#ifdef USE_TEXTURE_OBJECTS
			#define INTERTEX param.outTex
		#else
			#define INTERTEX interTexSingle
		#endif // USE_TEXTURE_OBJECTS
	#endif

	#define WRITE_SPINOR WRITE_SPINOR_FLOAT4_STR

	#define SPINOR_HOP 6
#endif

// only build double precision if supported
#if !(__COMPUTE_CAPABILITY__ < 130 && DD_PREC == 0) 

	#define DD_CONCAT(n,r,d) n ## r ## d ## Kernel
	#define DD_FUNC(n,r,d) DD_CONCAT(n,r,d)

	// define the kernels
	#define DD_NAME_F covDevM0
	template <KernelType kernel_type>
	__global__ void	DD_FUNC(DD_NAME_F, DD_RECON_F, DD_DAG_F)	(DD_PARAM1, DD_PARAM2, DD_PARAM3, DD_PARAM4)
	{
		#if DD_DAG
			#include "covDev_mu0_dagger_core.h"
		#else
			#include "covDev_mu0_core.h"
		#endif
	}

	#undef DD_NAME_F
	#define DD_NAME_F covDevM1
	template <KernelType kernel_type>
	__global__ void	DD_FUNC(DD_NAME_F, DD_RECON_F, DD_DAG_F)	(DD_PARAM1, DD_PARAM2, DD_PARAM3, DD_PARAM4)
	{
		#if DD_DAG
			#include "covDev_mu1_dagger_core.h"
		#else
			#include "covDev_mu1_core.h"
		#endif
	}

	#undef DD_NAME_F
	#define DD_NAME_F covDevM2
	template <KernelType kernel_type>
	__global__ void	DD_FUNC(DD_NAME_F, DD_RECON_F, DD_DAG_F)	(DD_PARAM1, DD_PARAM2, DD_PARAM3, DD_PARAM4)
	{
		#if DD_DAG
			#include "covDev_mu2_dagger_core.h"
		#else
			#include "covDev_mu2_core.h"
		#endif
	}

	#undef DD_NAME_F
	#define DD_NAME_F covDevM3
	template <KernelType kernel_type>
	__global__ void	DD_FUNC(DD_NAME_F, DD_RECON_F, DD_DAG_F)	(DD_PARAM1, DD_PARAM2, DD_PARAM3, DD_PARAM4)
	{
		#if DD_DAG
			#include "covDev_mu3_dagger_core.h"
		#else
			#include "covDev_mu3_core.h"
		#endif
	}

	#undef DD_NAME_F
#endif

// clean up

#undef DD_NAME_F
#undef DD_RECON_F
#undef DD_DAG_F
#undef DD_PARAM1
#undef DD_PARAM2
#undef DD_PARAM3
#undef DD_PARAM4
#undef DD_CONCAT
#undef DD_FUNC

#undef DSLASH_XPAY
#undef READ_GAUGE_MATRIX
#undef RECONSTRUCT_GAUGE_MATRIX
#undef GAUGE0TEX
#undef GAUGE1TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef READ_INTERMEDIATE_SPINOR
#undef INTERTEX
#undef ACCUMTEX
#undef WRITE_SPINOR
#undef GAUGE_FLOAT2
#undef SPINOR_DOUBLE

#undef SPINOR_HOP

#undef TPROJSCALE

// prepare next set of options, or clean up after final iteration

#if (DD_PREC==0)
	#undef DD_PREC
	#define	DD_PREC 1
#else
	#undef DD_PREC
	#define	DD_PREC 0

	#if (DD_DAG==0)
		#undef DD_DAG
		#define DD_DAG 1
	#else
		#undef DD_DAG
		#define DD_DAG 0

		#if (DD_RECON==0)
			#undef DD_RECON
			#define DD_RECON 1
		#elif (DD_RECON==1)
			#undef DD_RECON
			#define DD_RECON 2
		#else
			#undef DD_LOOP
			#undef DD_DAG
			#undef DD_RECON
			#undef DD_PREC
		#endif	// DD_RECON
	#endif	// DD_DAG
#endif	// DD_PREC

#ifdef DD_LOOP
	#include "covDev.h"
#endif


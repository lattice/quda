// staggered_dslash_def.h - staggered Dslash kernel definitions
//
// See comments in wilson_dslash_def.h

// initialize on first iteration

#ifndef DD_LOOP
#define DD_LOOP
#define DD_AXPY 0
#define DD_FAT_RECON 8
#define DD_LONG_RECON 8
#define DD_PREC 0
#endif

// set options for current iteration

#if (DD_IMPROVED==1)
#define DD_FNAME improvedStaggeredDslash
#else
#ifndef TIFR
#define DD_FNAME staggeredDslash
#else
#define DD_FNAME staggeredDslashTIFR
#endif
#endif

#if (DD_DAG==0) // no dagger
#define DD_DAG_F
#else           // dagger
#define DD_DAG_F Dagger
#endif

#if (DD_AXPY==0) // no axpy
#define DD_AXPY_F 
#else            // axpy
#define DD_AXPY_F Axpy
#define DSLASH_AXPY
#endif

#if (DD_FAT_RECON==8)
#define DD_FAT_RECON_F 8
#elif (DD_FAT_RECON==9)
#define DD_FAT_RECON_F 9
#elif (DD_FAT_RECON==12)
#define DD_FAT_RECON_F 12
#elif (DD_FAT_RECON==13)
#define DD_FAT_RECON_F 13
#else 
#define DD_FAT_RECON_F 18
#endif

#define READ_LONG_PHASE(phase, dir, idx, stride) // May be a problem below with redefinitions

#if (DD_LONG_RECON==8) // reconstruct from 8 reals
#define DD_LONG_RECON_F 8

#if (DD_PREC==0) // DOUBLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE

#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, stride)
#endif
#else // texture access
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#endif // DD_FAT_RECON
#endif // DIRECT_ACCESS_FAT_LINK

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else 
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif // DD_FAT_RECON

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, stride)
#endif // DD_FAT_RECON
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, stride)
#endif // DD_FAT_RECON
#endif // DIRECT_ACCESS_FAT_LINK

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#else // HALF PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else 
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif //DD_FAT_RECON

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==18)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#endif // DIRECT_ACCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#endif // DD_PREC

#elif (DD_LONG_RECON == 9) // reconstruct from 9 reals

#define DD_LONG_RECON_F 9

#if (DD_PREC==0) // DOUBLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_9_DOUBLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#else 
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, stride)
#endif
#else // texture access
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#endif // DD_FAT_RECON
#endif // DIRECT_ACCESS_FAT_LINK
#undef READ_LONG_PHASE

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_DOUBLE(PHASE, phase, dir, idx, stride);
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_DOUBLE_TEX(PHASE, phase, dir, idx, stride);
#endif // DIRECT_ACCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_9_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, stride)
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, stride)
#endif
#endif // DIRECT_ACCESS_FAT_LINK
#undef READ_LONG_PHASE

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_FLOAT(PHASE, phase, dir, idx, stride);
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4_TEX(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_FLOAT_TEX(PHASE, phase, dir, idx, stride);
#endif // DIRECT_ACCESS_LONG_LINK

#else // HALF PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_9_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==18)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#endif // DIRECT_ACCESS_FAT_LINK
#undef READ_LONG_PHASE
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_SHORT(PHASE, phase, dir, idx, stride);
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4_TEX(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_SHORT_TEX(PHASE, phase, dir, idx, stride);
#endif // DIRECT_ACCESS_LONG_LINK

#endif // DD_PREC

#elif (DD_LONG_RECON == 12)// reconstruct from 12 reals

#define DD_LONG_RECON_F 12

#if (DD_PREC==0) // DOUBLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, stride)
#endif
#else // texture access
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#endif // DD_FAT_RECON
#endif // DIRECT_ACCESS_FAT_LINK

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else 
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, stride)
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, stride)
#endif
#endif // DIRECT_ACCESS_FAT_LINK

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#else // HALF PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#endif // DIRECT_ACCESS_FAT_LINK

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#endif // DD_PREC

#elif (DD_LONG_RECON == 13)
#define DD_LONG_RECON_F 13

#if (DD_PREC==0) // DOUBLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_13_DOUBLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#else 
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, stride)
#endif
#else // texture access
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#endif // DD_FAT_RECON
#endif // DIRECT_ACCESS_FAT_LINK

#undef READ_LONG_PHASE
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_DOUBLE(PHASE, phase, dir, idx, stride);
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_DOUBLE_TEX(PHASE, phase, dir, idx, stride);
#endif // DIRECT_ACCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_13_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, stride)
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, stride)
#endif
#endif // DIRECT_ACCESS_FAT_LINK

#undef READ_LONG_PHASE 
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_FLOAT(PHASE, phase, dir, idx, stride);
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4_TEX(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_FLOAT_TEX(PHASE, phase, dir, idx, stride);
#endif // DIRECT_ACCESS_LONG_LINK

#else // HALF PRECISION

#define RECONSTRUCT_LONG_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_13_SINGLE
#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#endif
#undef READ_LONG_PHASE
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_SHORT(PHASE, phase, dir, idx, stride);
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4_TEX(LONG, gauge, dir, idx, stride)
#define READ_LONG_PHASE(phase, dir, idx, stride) READ_GAUGE_PHASE_SHORT_TEX(PHASE, phase, dir, idx, stride);
#endif // DIRECT_ACCESS_LONG_LINK

#endif // DD_PREC

#else //18 reconstruct
#define DD_LONG_RECON_F 18
#define RECONSTRUCT_LONG_GAUGE_MATRIX(dir, gauge, idx, sign)

#if (DD_PREC==0) // DOUBLE PRECISION

#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, stride)
#endif
#else // texture access
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, stride)
#endif // DD_FAT_RECON
#endif // DIRECT_ACCESS_FAT_LINK

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION

#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, stride)
#endif
#else
#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_FLOAT4_TEX(FAT, gauge, dir, idx, stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, stride)
#endif
#endif // DIRECT_ACCESS_FAT_LINK
 
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_FLOAT2_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#else  // HALF PRECISION

#if (DD_FAT_RECON==8)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#elif (DD_FAT_RECON==12)
#define RECONSTRUCT_FAT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_FAT_GAUGE_MATRIX(dir, gauge, idx, sign)
#endif

#ifdef DIRECT_ACCESS_FAT_LINK

#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#else

#if (DD_FAT_RECON==8)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_8_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#elif (DD_FAT_RECON==12)
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_12_SHORT4_TEX(FAT, gauge, dir, idx, stride); RESCALE4(FAT, fat_link_max);
#else
#define READ_FAT_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, stride); RESCALE2(FAT, fat_link_max);
#endif
#endif

#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2(LONG, gauge, dir, idx, stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx, stride) READ_GAUGE_MATRIX_18_SHORT2_TEX(LONG, gauge, dir, idx, stride)
#endif // DIRECT_ACCESS_LONG_LINK

#endif // DD_PREC

#endif // DD_LONG_RECON

#if (DD_PREC==0) // double-precision fields

// gauge field
#define DD_PREC_F D
#if (defined DIRECT_ACCESS_FAT_LINK) || (defined FERMI_NO_DBLE_TEX)
#define FATLINK0TEX param.gauge0
#define FATLINK1TEX param.gauge1
#else
#ifdef USE_TEXTURE_OBJECTS
#define FATLINK0TEX param.gauge0Tex
#define FATLINK1TEX param.gauge1Tex
#else
#if (DD_IMPROVED == 1)
#define FATLINK0TEX fatGauge0TexDouble
#define FATLINK1TEX fatGauge1TexDouble
#else
#define FATLINK0TEX gauge0TexDouble2
#define FATLINK1TEX gauge1TexDouble2
#endif
#endif // USE_TEXTURE_OBJECTS
#endif

#if (defined DIRECT_ACCESS_LONG_LINK) || (defined FERMI_NO_DBLE_TEX)
#define LONGLINK0TEX param.longGauge0
#define LONGLINK1TEX param.longGauge1
#define LONGPHASE0TEX param.longPhase0
#define LONGPHASE1TEX param.longPhase1
#else
#ifdef USE_TEXTURE_OBJECTS
#define LONGLINK0TEX param.longGauge0Tex
#define LONGLINK1TEX param.longGauge1Tex
#define LONGPHASE0TEX param.longPhase0Tex
#define LONGPHASE1TEX param.longPhase1Tex
#else
#define LONGLINK0TEX longGauge0TexDouble
#define LONGLINK1TEX longGauge1TexDouble
#define LONGPHASE0TEX longPhase0TexDouble
#define LONGPHASE1TEX longPhase1TexDouble
#endif // USE_TEXTURE_OBJECTS
#endif

#define GAUGE_DOUBLE

// spinor fields
#if (defined DIRECT_ACCESS_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define SPINORTEX param.in
#define GHOSTSPINORTEX param.ghost
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_DOUBLE
#define READ_3RD_NBR_SPINOR READ_KS_NBR_SPINOR_DOUBLE
#define READ_1ST_NBR_SPINOR_GHOST READ_1ST_NBR_SPINOR_GHOST_DOUBLE
#define READ_3RD_NBR_SPINOR_GHOST READ_KS_NBR_SPINOR_GHOST_DOUBLE
#else
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#define GHOSTSPINORTEX param.ghostTex
#else
#define SPINORTEX spinorTexDouble
#define GHOSTSPINORTEX ghostSpinorTexDouble
#endif // USE_TEXTURE_OBJECTS
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_DOUBLE_TEX
#define READ_3RD_NBR_SPINOR READ_KS_NBR_SPINOR_DOUBLE_TEX
#define READ_1ST_NBR_SPINOR_GHOST READ_1ST_NBR_SPINOR_GHOST_DOUBLE_TEX
#define READ_3RD_NBR_SPINOR_GHOST READ_KS_NBR_SPINOR_GHOST_DOUBLE_TEX
#endif
#if (defined DIRECT_ACCESS_INTER) || (defined FERMI_NO_DBLE_TEX)
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR
#define INTERTEX param.out
#else
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_DOUBLE_TEX
#ifdef USE_TEXTURE_OBJECTS
#define INTERTEX param.outTex
#else
#define INTERTEX interTexDouble
#endif
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
#if (DD_AXPY==1)
#if (defined DIRECT_ACCESS_ACCUM) || (defined FERMI_NO_DBLE_TEX)
#define ACCUMTEX param.x
#define READ_ACCUM READ_ST_ACCUM_DOUBLE
#else
#ifdef USE_TEXTURE_OBJECTS
#define ACCUMTEX param.xTex
#else
#define ACCUMTEX accumTexDouble
#endif // USE_TEXTURE_OBJECTS
#define READ_ACCUM READ_ST_ACCUM_DOUBLE_TEX
#endif
#endif // DD_AXPY


#elif (DD_PREC==1) // single-precision fields

// gauge fields
#define DD_PREC_F S

#ifndef DIRECT_ACCESS_FAT_LINK
#ifdef USE_TEXTURE_OBJECTS
#define FATLINK0TEX param.gauge0Tex
#define FATLINK1TEX param.gauge1Tex
#else
#if (DD_IMPROVED == 1)
#define FATLINK0TEX fatGauge0TexSingle
#define FATLINK1TEX fatGauge1TexSingle
#else
#if (DD_FAT_RECON == 18)
#define FATLINK0TEX gauge0TexSingle2
#define FATLINK1TEX gauge1TexSingle2
#else
#define FATLINK0TEX gauge0TexSingle4
#define FATLINK1TEX gauge1TexSingle4
#endif
#endif // DD_IMPROVED
#endif
#else
#define FATLINK0TEX param.gauge0
#define FATLINK1TEX param.gauge1
#endif

#ifndef DIRECT_ACCESS_LONG_LINK //longlink access
#ifdef USE_TEXTURE_OBJECTS
#define LONGLINK0TEX param.longGauge0Tex
#define LONGLINK1TEX param.longGauge1Tex
#define LONGPHASE0TEX param.longPhase0Tex
#define LONGPHASE1TEX param.longPhase1Tex
#else
#if (DD_LONG_RECON ==18)
#define LONGLINK0TEX longGauge0TexSingle_norecon
#define LONGLINK1TEX longGauge1TexSingle_norecon
#else
#define LONGLINK0TEX longGauge0TexSingle
#define LONGLINK1TEX longGauge1TexSingle
#define LONGPHASE0TEX longPhase0TexSingle
#define LONGPHASE1TEX longPhase1TexSingle
#endif
#endif // USE_TEXTURE_OBJECTS
#else
#define LONGLINK0TEX param.longGauge0
#define LONGLINK1TEX param.longGauge1
#define LONGPHASE0TEX param.longPhase0
#define LONGPHASE1TEX param.longPhase1
#endif

// spinor fields
#ifndef DIRECT_ACCESS_SPINOR
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#define GHOSTSPINORTEX param.ghostTex
#else
#define SPINORTEX spinorTexSingle2
#define GHOSTSPINORTEX ghostSpinorTexSingle2
#endif // USE_TEXTURE_OBJECTS
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_SINGLE_TEX
#define READ_3RD_NBR_SPINOR READ_KS_NBR_SPINOR_SINGLE_TEX
#define READ_1ST_NBR_SPINOR_GHOST READ_1ST_NBR_SPINOR_GHOST_SINGLE_TEX
#define READ_3RD_NBR_SPINOR_GHOST READ_KS_NBR_SPINOR_GHOST_SINGLE_TEX
#else
#define SPINORTEX param.in
#define GHOSTSPINORTEX param.ghost
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_SINGLE
#define READ_3RD_NBR_SPINOR READ_KS_NBR_SPINOR_SINGLE
#define READ_1ST_NBR_SPINOR_GHOST READ_1ST_NBR_SPINOR_GHOST_SINGLE
#define READ_3RD_NBR_SPINOR_GHOST READ_KS_NBR_SPINOR_GHOST_SINGLE
#endif
#if (defined DIRECT_ACCESS_INTER)
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR
#define INTERTEX param.out
#else
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_SINGLE_TEX
#ifdef USE_TEXTURE_OBJECTS
#define INTERTEX param.outTex
#else
#define INTERTEX interTexSingle2
#endif // USE_TEXTURE_OBJECTS
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_FLOAT2
#if (DD_AXPY==1)
#if (defined DIRECT_ACCESS_ACCUM)
#define ACCUMTEX param.x
#define READ_ACCUM READ_ST_ACCUM_SINGLE
#else
#ifdef USE_TEXTURE_OBJECTS
#define ACCUMTEX param.xTex
#else
#define ACCUMTEX accumTexSingle2
#endif // USE_TEXTURE_OBJECTS
#define READ_ACCUM READ_ST_ACCUM_SINGLE_TEX
#endif
#endif // DD_AXPY


#else             // half-precision fields

// gauge fields
#define DD_PREC_F H

#ifndef DIRECT_ACCESS_FAT_LINK
#ifdef USE_TEXTURE_OBJECTS
#define FATLINK0TEX param.gauge0Tex
#define FATLINK1TEX param.gauge1Tex
#else
#if (DD_IMPROVED == 1)
#define FATLINK0TEX fatGauge0TexHalf
#define FATLINK1TEX fatGauge1TexHalf
#else
#if (DD_FAT_RECON == 18)
#define FATLINK0TEX gauge0TexHalf2
#define FATLINK1TEX gauge1TexHalf2
#else
#define FATLINK0TEX gauge0TexHalf4
#define FATLINK1TEX gauge1TexHalf4
#endif
#endif // DD_IMPROVED
#endif // USE_TEXTURE_OBJECTS
#else // DIRECT_ACCESS_FAT_LINK
#define FATLINK0TEX param.gauge0
#define FATLINK1TEX param.gauge1
#endif

#ifndef DIRECT_ACCESS_LONG_LINK
#ifdef USE_TEXTURE_OBJECTS
#define LONGLINK0TEX param.longGauge0Tex
#define LONGLINK1TEX param.longGauge1Tex
#define LONGPHASE0TEX param.longPhase0Tex
#define LONGPHASE1TEX param.longPhase1Tex
#else
#if (DD_LONG_RECON ==18)
#define LONGLINK0TEX longGauge0TexHalf_norecon
#define LONGLINK1TEX longGauge1TexHalf_norecon
#else
#define LONGLINK0TEX longGauge0TexHalf
#define LONGLINK1TEX longGauge1TexHalf
#define LONGPHASE0TEX longPhase0TexHalf
#define LONGPHASE1TEX longPhase1TexHalf
#endif
#endif // USE_TEXTURE_OBJECTS
#else  // DIRECT_ACCESS_LONG_LINK
#define LONGLINK0TEX param.longGauge0
#define LONGLINK1TEX param.longGauge1
#define LONGPHASE0TEX param.longPhase0
#define LONGPHASE1TEX param.longPhase1
#endif

#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_HALF_TEX
#define READ_3RD_NBR_SPINOR READ_KS_NBR_SPINOR_HALF_TEX
#define READ_1ST_NBR_SPINOR_GHOST READ_1ST_NBR_SPINOR_GHOST_HALF_TEX
#define READ_3RD_NBR_SPINOR_GHOST READ_KS_NBR_SPINOR_GHOST_HALF_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#define GHOSTSPINORTEX param.ghostTex
#else
#define SPINORTEX spinorTexHalf2
#define GHOSTSPINORTEX ghostSpinorTexHalf2
#endif // USE_TEXTURE_OBJECTS
#if (defined DIRECT_ACCESS_INTER)
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_HALF
#define INTERTEX param.out
#else
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_HALF_TEX
#ifdef USE_TEXTURE_OBJECTS
#define INTERTEX param.outTex
#else
#define INTERTEX interTexHalf2
#endif // USE_TEXTURE_OBJECTS
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_SHORT2
#if (DD_AXPY==1)
#ifdef USE_TEXTURE_OBJECTS
#define ACCUMTEX param.xTex
#else
#define ACCUMTEX accumTexHalf2
#endif // USE_TEXTURE_OBJECTS
#define READ_ACCUM READ_ST_ACCUM_HALF_TEX
#endif // DD_AXPY

#endif

#ifdef GPU_STAGGERED_DIRAC

// define the kernel

#if (DD_IMPROVED==1)

#define DD_CONCAT(n,p,r1,r2,d,x) n ## p ## r1 ## r2 ## d ## x ## Kernel
#define DD_FUNC(n,p,r1,r2,d,x) DD_CONCAT(n,p,r1,r2,d,x)

template <KernelType kernel_type>
__global__ void	DD_FUNC(DD_FNAME, DD_PREC_F, DD_FAT_RECON_F, DD_LONG_RECON_F, DD_DAG_F, DD_AXPY_F)(const DslashParam param) {
#if defined(GPU_STAGGERED_DIRAC) && DD_FAT_RECON == 18 // improved staggered only supports no reconstruct fat-links 
  #include "staggered_dslash_core.h"
#endif
}

#ifdef MULTI_GPU
template <>
__global__ void	DD_FUNC(DD_FNAME, DD_PREC_F, DD_FAT_RECON_F, DD_LONG_RECON_F, DD_DAG_F, DD_AXPY_F)<EXTERIOR_KERNEL_ALL>(const DslashParam param) {
#if defined(GPU_STAGGERED_DIRAC) && DD_FAT_RECON == 18 // improved staggered only supports no reconstruct fat-links 
  #include "staggered_fused_exterior_dslash_core.h"
#endif
}

#endif // MULTI_GPU

#else // naive staggered kernel

#undef READ_LONG_MATRIX
#define READ_LONG_MATRIX(gauge, dir, idx, stride)

#undef READ_LONG_PHASE
#define READ_LONG_PHASE(phase, dir, idx, stride)

#define DD_CONCAT(n,p,r,d,x) n ## p ## r ## d ## x ## Kernel
#define DD_FUNC(n,p,r,d,x) DD_CONCAT(n,p,r,d,x)

#if (DD_LONG_RECON == 18) // avoid kernel aliasing over non-existant long-links

template <KernelType kernel_type>
__global__ void	DD_FUNC(DD_FNAME, DD_PREC_F, DD_FAT_RECON_F, DD_DAG_F, DD_AXPY_F)(const DslashParam param) {
#if defined(GPU_STAGGERED_DIRAC) && DD_FAT_RECON != 9 && DD_FAT_RECON != 13
#include "staggered_dslash_core.h"
#endif
}

#ifdef MULTI_GPU
template <>
__global__ void	DD_FUNC(DD_FNAME, DD_PREC_F, DD_FAT_RECON_F, DD_DAG_F, DD_AXPY_F)<EXTERIOR_KERNEL_ALL>(const DslashParam param) {
#if defined(GPU_STAGGERED_DIRAC) && DD_FAT_RECON != 9 && DD_FAT_RECON != 13
#include "staggered_fused_exterior_dslash_core.h"
#endif
}
#endif // MULTI_GPU

#endif


#endif

#endif // ! GPU_STAGGERED_DIRAC

// clean up

#undef DD_PREC_F
#undef DD_FAT_RECON_F
#undef DD_LONG_RECON_F
#undef DD_DAG_F
#undef DD_AXPY_F
#undef DD_FNAME
#undef DD_CONCAT
#undef DD_FUNC

#undef DSLASH_AXPY
#undef READ_GAUGE_MATRIX
#undef RECONSTRUCT_FAT_GAUGE_MATRIX
#undef RECONSTRUCT_LONG_GAUGE_MATRIX
#undef FATLINK0TEX
#undef FATLINK1TEX
#undef LONGLINK0TEX
#undef LONGLINK1TEX
#undef LONGPHASE0TEX
#undef LONGPHASE1TEX
#undef SPINORTEX
#undef GHOSTSPINORTEX
#undef WRITE_SPINOR
#undef READ_AND_SUM_SPINOR
#undef INTERTEX
#undef ACCUMTEX
#undef READ_ACCUM
#undef CLOVERTEX
#undef READ_CLOVER
#undef DSLASH_CLOVER
#undef GAUGE_DOUBLE
#undef SPINOR_DOUBLE
#undef CLOVER_DOUBLE
#undef READ_FAT_MATRIX
#undef READ_LONG_MATRIX
#undef READ_LONG_PHASE
#undef READ_1ST_NBR_SPINOR
#undef READ_3RD_NBR_SPINOR
#undef READ_1ST_NBR_SPINOR_GHOST
#undef READ_3RD_NBR_SPINOR_GHOST


// prepare next set of options, or clean up after final iteration

#if (DD_AXPY==0)
#undef DD_AXPY
#define DD_AXPY 1
#else
#undef DD_AXPY
#define DD_AXPY 0

#if (DD_LONG_RECON==8)
#undef DD_LONG_RECON
#define DD_LONG_RECON 9
#elif (DD_LONG_RECON==9)
#undef DD_LONG_RECON
#define DD_LONG_RECON 12
#elif (DD_LONG_RECON==12)
#undef DD_LONG_RECON
#define DD_LONG_RECON 13
#elif (DD_LONG_RECON==13)
#undef DD_LONG_RECON
#define DD_LONG_RECON 18
#else
#undef DD_LONG_RECON

#define DD_LONG_RECON 8

#if (DD_FAT_RECON==8)
#undef DD_FAT_RECON
#define DD_FAT_RECON 9 // dummy
#elif (DD_FAT_RECON==9)
#undef DD_FAT_RECON
#define DD_FAT_RECON 12
#elif (DD_FAT_RECON==12)
#undef DD_FAT_RECON
#define DD_FAT_RECON 13 //dummy
#elif (DD_FAT_RECON==13)
#undef DD_FAT_RECON
#define DD_FAT_RECON 18
#else
#undef DD_FAT_RECON

#define DD_FAT_RECON 8

#if (DD_PREC==0)
#undef DD_PREC
#define DD_PREC 1
#elif (DD_PREC==1)
#undef DD_PREC
#define DD_PREC 2
#else

#undef DD_LOOP
#undef DD_AXPY
#undef DD_LONG_RECON
#undef DD_PREC

#endif // DD_PREC
#endif // DD_FAT_RECON
#endif // DD_LONG_RECON
#endif // DD_AXPY

#ifdef DD_LOOP
#include "staggered_dslash_def.h"
#endif

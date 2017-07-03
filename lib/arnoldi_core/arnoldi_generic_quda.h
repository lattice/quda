#ifndef ARNOLDI_GENERIC_QUDA_H_
#define ARNOLDI_GENERIC_QUDA_H_


/**** arnoldi_generic.h - Implicitly Restarted Arnoldi Method ********
 * David Weir, Joni Suorsa and Teemu Rantalaiho 2011-14              *
 *********************************************************************
 *
 *  Copyright 2011-2014 David Weir, Joni Suorsa and Teemu Rantalaiho
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *
 */


// This files contains implementation for a generic version of the Arnoldi method
// used by other methods - for instructions how to use this file directly consult
// to readme.txt or copy ideas from carnoldi.c/zarnoldi.c


// Print some of the intermediate steps
// #define EIG_DEBUG
// #define EIG_DEBUG_FULL
// #define GIVENS_DEBUG
// #define SHIFT
//#define EIG_INFO
// #define TRACE_STATISTICS

#ifdef TRACE_STATISTICS
static int s_nOPxs = 0;
static int s_nArnSteps = 0;
// Optional timing statistics - s_cputimef should be set properly
static double s_OPxTime = 0.0;
static double s_orthT = 0.0;
static double s_totArnStepT = 0.0;
static double s_parallelT = 0.0;
static double s_totArnT = 0.0;
static double s_GivensT = 0.0;
static double s_restartRotationT = 0.0;
static double s_deflateRotationT = 0.0;
static double s_deflateT = 0.0;
static double s_geevFunT = 0.0;
static int s_nCDotProds = 0;
static int s_nMADDs = 0;
static int s_nGivenss = 0;
static int s_nDefRotate = 0;
static int s_nShiftRotate = 0;
#endif


// TODO: Investigate when to use each codepath
#ifdef MULTI_GPU
#define USE_POST_ROTATE_CODEPATH 1
#else
#define USE_POST_ROTATE_CODEPATH 0
#endif

#include <stddef.h>

static int s_nStreams = 0;
static int s_curStreamIdx = -1;
static void** s_astreams = (void**)0;
static void setCurStream(int newStream);
static void waitStream(int stream);
static void** s_streamBufs = (void**)0;
static size_t* s_streamBufSizes = (size_t*)0;
static void   freeStreamBuffer(int streamID);


#ifdef MULTI_GPU
#ifdef CURRENT_STREAM
#undef CURRENT_STREAM
#endif
static void* getCurStream(void);
static void** getCurStreamBuf(void);
static size_t* getCurStreamBufSize(void);
#define CURRENT_STREAM()            (cudaStream_t)getCurStream()
#define CURRENT_STREAM_TMPBUF()     getCurStreamBuf()
#define CURRENT_STREAM_TMPBUFSIZE() getCurStreamBufSize()
#else
#define CURRENT_STREAM()
#endif

#include "apar_defs.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef MANGLE
#define MANGLE(X) X
#endif


#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef restrict
#define restrict
#endif


#if RADIX_SIZE == 64
#define radix double
#elif RADIX_SIZE == 32
#define radix float
#else
#error RADIX_SIZE not defined! Make it 32 or 64
#endif


// Important - you need to define type called: "fieldtype" and "fieldentry" together with the implementation to the following
// functions.


typedef struct lcomplex_s {
    radix real;
    radix imag;
} lcomplex;

// Define problem size with two components in order to enable advanced layouts (for example structure of arrays and meta-info between entries)
// This data is passed to get_fieldEntry(f, i, * res, j, stride) as i \in {0,.., size}, j \in {0,...,nMulti}, stride = stride.
typedef struct multisize_s {
    int size;
    int nMulti;
    int stride;
} multisize;


// Complex functions:


static inline __host__ __device__
lcomplex cmplxf( radix x, radix y )  {
    lcomplex c;
    c.real = x; c.imag = y;
    return(c);
}


static inline __host__ __device__
lcomplex caddf( const lcomplex * restrict a, const lcomplex * restrict b ) {
    lcomplex c;
    c.real = (*a).real + (*b).real;
    c.imag = (*a).imag + (*b).imag;
    return(c);
}

static inline __host__ __device__
lcomplex csubf( const lcomplex * restrict a, const lcomplex * restrict b ) {
    lcomplex c;
    c.real = (*a).real - (*b).real;
    c.imag = (*a).imag - (*b).imag;
    return(c);
}

static inline __host__ __device__
lcomplex ce_ithetaf( radix theta ){
    lcomplex c;
    c.real = (radix)cos( (double)theta );
    c.imag = (radix)sin( (double)theta );
    /* there must be a more efficient way */
    return( c );
}


static inline __host__ __device__
lcomplex mycsqrt( const lcomplex * restrict z ){
lcomplex c;
radix theta,r;
    //r = sqrt(hypot(z->real,z->imag));
    r = (radix)sqrt(hypot(z->real,z->imag));
    theta = (radix)(0.5*atan2(z->imag,z->real));
    c = ce_ithetaf(theta);
    c.real *=r; c.imag *= r;
    return(c);
}

static inline __host__ __device__
lcomplex cmulf( const lcomplex * restrict a, const lcomplex * restrict b) {
    lcomplex c;
    c.real = (*a).real * (*b).real - (*a).imag * (*b).imag;
    c.imag = (*a).imag * (*b).real + (*a).real * (*b).imag;
    return c;
}

static inline __host__ __device__
void cmaddf( const lcomplex * restrict a, const lcomplex * restrict b, lcomplex * c) {
    radix tmp;
    tmp = (*a).real * (*b).real - (*a).imag * (*b).imag;
    c->imag += (*a).imag * (*b).real + (*a).real * (*b).imag;
    c->real += tmp;
}

// As above, but compute c += (Conj a) b
static inline __host__ __device__
void cmaddf_an( const lcomplex * restrict a, const lcomplex * restrict b, lcomplex * c) {
    radix tmp;
    tmp = (*a).real * (*b).real + (*a).imag * (*b).imag;
    c->imag += (*a).real * (*b).imag - (*a).imag * (*b).real;
    c->real += tmp;
}



static inline __host__ __device__
radix cabs_sqf(const lcomplex* a) {
    return ( (*a).real*(*a).real + (*a).imag*(*a).imag );
}
static inline __host__ __device__
radix mycabs(const lcomplex* a) {
    radix res = (radix)sqrt(cabs_sqf(a));
    return res;
}
static inline __host__ __device__
radix CABSF(const lcomplex* a) {
    return mycabs(a);
}


static inline __host__ __device__
lcomplex cdivf( const lcomplex * restrict a, const lcomplex * restrict b ) {
    lcomplex c;
    radix scale;
    scale = (radix)1.0/cabs_sqf(b);
    c.real = scale*((*a).real*(*b).real + (*a).imag*(*b).imag);
    c.imag = scale*((*a).imag*(*b).real - (*a).real*(*b).imag);
    return(c);
}



static inline __host__ __device__
lcomplex conjgf( const lcomplex * a ){
    lcomplex c;
    c.real = (*a).real;
    c.imag = -(*a).imag;
    return(c);
}
/* c = ba     */
#define CMULREAL(a,b,c) do {(c).real = (b) * (a).real; (c).imag = (b)*(a).imag; } while(0)


//typedef wilson_vec_color fieldentry;

/*typedef struct fieldtype_s fieldtype;
typedef struct fieldentry_s fieldentry;*/


__host__ static fieldtype* new_fieldtype(int size, int nMulti);
__host__ static void free_fieldtype(fieldtype* f);
// Typical use-cases - example - vectors of 4 complex numbers:
// Array of structures layout: { [x0, x1, x2, x3]_0, [x0, x1, x2, x3]_1, ..., [x0, x1, x2, x3]_{N-1} }
//  -> nMultiComp = 4, stride = 1, get_field(f, i,j,stride) = f[i].x_j
// Structure of Arrays layout: { x0_0, x0_1, ..., x0_{N-1}, x1_0, x1_1, ..., x1_{N-1}, ... , x3_0, x3_1, ..., x3_{N-1} }
// -> nMultiComp = 4, stride = N, get_field(f, i, j, stride) = *(&f[0] + i + j * stride)
__device__ static inline void get_fieldEntry(const fieldtype* f, int i, fieldentry* result, int j, int stride);
__device__ static inline void set_fieldEntry(fieldtype* f, int i, const fieldentry* entry, int j, int stride);

__device__ static inline lcomplex fieldEntry_dot(const fieldentry* a, const fieldentry* b);
__device__ static inline radix fieldEntry_rdot(const fieldentry* a, const fieldentry* b);
// dst = scalar * a
__device__ static inline void fieldEntry_scalar_mult(const fieldentry* a, radix scalar, fieldentry* dst );
// dst = a + scalar * b
__device__ static inline void fieldEntry_scalar_madd(const fieldentry * restrict a, radix scalar, const fieldentry * restrict b, fieldentry * restrict dst );
// dst = scalar * a
__device__ static inline void fieldEntry_complex_mult(const fieldentry* a, lcomplex scalar, fieldentry* dst );
// dst = a + scalar * b
__device__ static inline void fieldEntry_complex_madd(const fieldentry * restrict a, lcomplex scalar, const fieldentry * restrict b, fieldentry * restrict dst );



/* Do reduction of val across all nodes - for single-node support, just leave this NULL */
typedef void (*mpi_reductionT) (double* val);
typedef void* (*mallocFunT)(size_t size);
typedef void (*freeFunT)(void* ptr);


// Optional stream API:
typedef void* (*getNewStreamfunT)(void);
typedef void  (*freeStreamfunT)(void*);
typedef void  (*waitStreamfunT)(void*);
// Optional timing statistics
typedef double (*cputimefunT)(void);
static cputimefunT s_cputimef = (cputimefunT)0;



// API: set these to what you want/need:
static DiracMatrix *s_mulf = nullptr;
static mpi_reductionT scalar_reduction_f  = nullptr;
static mpi_reductionT complex_reduction_f = nullptr;
// NOTE: change these into whatever you want... These are the host-side memory-management functions
static mallocFunT s_mallocf = malloc;
static freeFunT s_freef     = free;


#if defined(MULTI_GPU)
static getNewStreamfunT s_createStream = createCudaStream;
static freeStreamfunT s_destroyStream = destroyCudaStream;
static waitStreamfunT s_waitForStream = waitCudaStream;
#else
static getNewStreamfunT s_createStream = nullptr;
static freeStreamfunT s_destroyStream  = nullptr;
static waitStreamfunT s_waitForStream  = nullptr;
#endif


void setCurStream(int streamID){
    if (streamID >= s_nStreams)
        s_curStreamIdx = -1;
    else
        s_curStreamIdx = streamID;
}

void waitStream(int streamID){
    if (s_waitForStream && streamID < s_nStreams)
        s_waitForStream(s_astreams[streamID]);
}

#ifdef MULTI_GPU
void* getCurStream(void){
    void* res = (void*)0;
    if (s_curStreamIdx >= 0)
        res = s_astreams[s_curStreamIdx];
    return res;
}
void** getCurStreamBuf(void){
    void** res = (void**)0;
    if (s_curStreamIdx >= 0)
        res = &s_streamBufs[s_curStreamIdx];
    return res;
}
size_t* getCurStreamBufSize(void){
    size_t* res = (size_t*)0;
    if (s_curStreamIdx >= 0)
        res = &s_streamBufSizes[s_curStreamIdx];
    return res;
}
#endif
void freeStreamBuffer(int streamID){

#ifdef MULTI_GPU
#ifndef PRINT_FREE
#define PRINT_FREE(X)
#endif
    if (s_streamBufs[streamID] != NULL){
        cudaFree(s_streamBufs[streamID]);
        PRINT_FREE(s_streamBufs[streamID]);
    }
#endif
    s_streamBufs[streamID] = NULL;
    s_streamBufSizes[streamID] = 0;
}


typedef struct cmatrix_s {
  int n;
  int stride;
  lcomplex **e ;
} cmatrix;


// prototypes
static int numerical_eigs_dense(cmatrix *restrict A, cmatrix *restrict Q);

/*
#if defined(RADIX_F)
#define zgeev cgeev_
#else
#define zgeev zgeev_
#endif
*/

#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif


#if RADIX_SIZE == 64
#define geevfun zgeev_
#else
#define geevfun cgeev_
#endif

// NOTE: Need extern "C" { decl...} when linking C-functions from C++ code
EXTERN_C_BEGIN
void sgeev_(char jobvl, char jobvr, int n, float *a, int lda,
              float *wr,  float *wi,
              float *vl, int ldvl, float *vr, int ldvr, int *info);
void cgeev_(char *jobvl, char *jobvr, int *n, lcomplex *a,
               int *lda, lcomplex *w, lcomplex *vl,
               int *ldvl, lcomplex *vr, int *ldvr,
               lcomplex *work, int *lwork, radix *rwork,
               int *info);
void zgeev_(char *jobvl, char *jobvr, int *n, lcomplex *a,
               int *lda, lcomplex *w, lcomplex *vl,
               int *ldvl, lcomplex *vr, int *ldvr,
               lcomplex *work, int *lwork, radix *rwork,
               int *info);
EXTERN_C_END


// TODO: Align start of vectors in matrix?
// TODO: Check if this even works
//#define ALIGN_MAT   1
#define ALIGN_MAT   16

static
cmatrix cmat_new(int n){
  int i;
  int stride;
  cmatrix a;
  a.n=n;
  stride = n/ALIGN_MAT;
  if (stride * ALIGN_MAT < n) stride++;
  stride *= ALIGN_MAT;
  a.stride = stride;
  a.e= (lcomplex**) s_mallocf(n*sizeof(lcomplex*));
  if (a.e){
      a.e[0] = (lcomplex*) s_mallocf(n*stride*sizeof(lcomplex));
      if (a.e[0]){
          for (i=1;i<n;i++)  a.e[i] = a.e[i-1]+stride;
      } else {
          s_freef(a.e);
          a.e = NULL;
      }
  }
  return a;
}

static
void cmat_one(cmatrix* a)
{
  int i,j,n;
  n = a->n;
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      if (i==j)
        a->e[i][j].real = 1;
      else
        a->e[i][j].real = 0;
      a->e[i][j].imag = 0;
    }
  }
}

static
void cmat_zero (cmatrix* restrict a)
{
  int i,j;
  for (i=0;i<a->n;i++){
    for (j=0;j<a->n;j++){
      a->e[i][j].real = 0;
      a->e[i][j].imag = 0;
    }
  }
}

static
void cmat_adj(const cmatrix* a, cmatrix* b)
{
  int i,j;
  int n = a->n;
  for (i=0;i<n;i++){
    for (j=0;j<n;j++){
      b->e[i][j].real = a->e[j][i].real;
      b->e[i][j].imag = -a->e[j][i].imag;
    }
  }
}

static
void cmat_copy(const cmatrix * restrict src, cmatrix * restrict dst){
  int i,j;
  for (i=0;i<src->n;i++){
    for (j=0;j<src->n;j++){
      dst->e[i][j].real=src->e[i][j].real;
      dst->e[i][j].imag=src->e[i][j].imag;
    }
  }
}


static
void cmat_free(cmatrix* a){
  if (a->e){
      s_freef(a->e[0]);
      s_freef(a->e);
  }
}

/* Following routines are recycled or forked from (eg) w_eigen.c */


/* Bubble sort the eigenvectors according
 * to those which have the smallest magnitude real part
 */
static
void cmat_sortmag_eigenvectors(cmatrix * restrict a, cmatrix * restrict v, arnmode mode)
{
  int n;
  cmatrix tmp;

  // need volatile to prevent loop fusion!
  volatile radix *ds, *dr, *di, r;
  lcomplex* save_tmp0;
  lcomplex* cp;
  int i,j,k;

  int preferLarge = mode == arnmode_LM || mode == arnmode_LR;

  n= a->n;
  tmp = cmat_new(n);

  // Take adjoint, since then eigenvectors are rows
  cmat_adj(v,&tmp);

  ds = (radix *) s_mallocf(n*sizeof(radix));

  dr = (radix *) s_mallocf(n*sizeof(radix));
  di = (radix *) s_mallocf(n*sizeof(radix));


  /* Calls for separate
   * entries for the sort criterion
   * and real and imaginary parts
   */
  for (i=0;i<n;i++){

    // Examples:

    // Smallest magnitude:
    // ds[i]=fabs(a->e[i][i].real*a->e[i][i].real
    //         + a->e[i][i].imag*a->e[i][i].imag);

    // Smallest absolute imgainary part:
    // ds[i]=fabs(a->e[i][i].imag);

#ifdef SHIFT
#warning Have you checked your sort matches your shift?
    // Largest real:
    // ds[i] = -1.0*a->e[i][i].real;
    // Smallest real:
    ds[i] = a->e[i][i].real;
#else
    if (mode == arnmode_LM || mode == arnmode_SM)
        ds[i] = cabs_sqf(&a->e[i][i]);
    else
        ds[i] = a->e[i][i].real;
    if (preferLarge)
        ds[i] *= -1;
#endif

    dr[i]=a->e[i][i].real;
    di[i]=a->e[i][i].imag;

  }

  save_tmp0=tmp.e[0];
  for (i=0;i<n;i++){
    for (j=1;j<n-i;j++){
      k = j-1;
      if (ds[j]<ds[k]){ /*switch them*/

        r = dr[k];
        dr[k] = dr[j];
        dr[j] = r;



        r = di[k];
        di[k] = di[j];
        di[j] = r;

        r = ds[k];
        ds[k] = ds[j];
        ds[j] = r;

        cp = tmp.e[k];
        tmp.e[k] = tmp.e[j];
        tmp.e[j] = cp;
      }
    }
  }
  for (i=0;i<n;i++){
    a->e[i][i].real=dr[i];
    a->e[i][i].imag=di[i];
  }

  s_freef((void*)ds);

  s_freef((void*)dr);
  s_freef((void*)di);

  // Restore eigenvectors to columns
  cmat_adj(&tmp,v);
  tmp.e[0] = save_tmp0;
  cmat_free(&tmp);
}



typedef struct dotIn_t_s{
    const fieldtype* s;
    int stride;
} dotIn_t;

PARALLEL_REDUCE_BEGIN(fvec_magsq, dotIn_t, in, i, radix, sum, multiIdx)
{
    fieldentry v;
    get_fieldEntry(in.s, i, &v, multiIdx, in.stride);
    sum = fieldEntry_rdot(&v, &v);
}

PARALLEL_REDUCE_SUMFUN(fvec_magsq, sofar, sum, radix)
{
    sum += sofar;
}

PARALLEL_REDUCE_END(sum)

static
radix fieldvec_magsq_V( const fieldtype* s, multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif
  radix tmp;
  dotIn_t in;
  in.s = s;
  in.stride = size.stride;
  FOR_RANGE_REDUCE_KERNEL(fvec_magsq, in, &tmp, 0, size.size, size.nMulti, 0, 0);
  if (scalar_reduction_f) {
      double d_tmp = static_cast<double>(tmp);
      scalar_reduction_f(&d_tmp);
      tmp = d_tmp;
  }
#ifdef TRACE_STATISTICS
  if (s_cputimef)
      s_parallelT += s_cputimef() - t0;
#endif
  return tmp;
}

typedef struct two_fvec_s
{
  const fieldtype * f1;
  const fieldtype * f2;
  int stride;
} two_fvec;

PARALLEL_REDUCE_BEGIN(fvec_cdot, two_fvec, in, i, lcomplex, sum, multiIdx)
{
    fieldentry a,b;
    get_fieldEntry(in.f1, i, &a, multiIdx, in.stride);
    get_fieldEntry(in.f2, i, &b, multiIdx, in.stride);
    sum = fieldEntry_dot(&a, &b);
}
PARALLEL_REDUCE_SUMFUN(fvec_cdot, sofar, sum, lcomplex)
{
    sum.real += sofar.real;
    sum.imag += sofar.imag;
}
PARALLEL_REDUCE_END(sum)


static
void fieldvec_cdot_V_skipmpi( const fieldtype* a, const fieldtype* b, multisize size, lcomplex* result, int onDev)
{
  two_fvec in;
  in.f1 = a;
  in.f2 = b;
  in.stride = size.stride;
  FOR_RANGE_REDUCE_KERNEL(fvec_cdot, in, result, 0, size.size, size.nMulti, onDev, 0);
}

static
lcomplex fieldvec_cdot_V( const fieldtype* a, const fieldtype* b, multisize size)
{
  lcomplex result;
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nCDotProds++;
#endif
  fieldvec_cdot_V_skipmpi(a, b, size, &result, 0);
  if (complex_reduction_f) {
      double d_result[2] = { static_cast<double>(result.real), static_cast<double>(result.imag) };
      complex_reduction_f(d_result);
      result.real = d_result[0];
      result.imag = d_result[1];
  }
#ifdef TRACE_STATISTICS
  if (s_cputimef){
      s_parallelT += s_cputimef() - t0;
  }
#endif
  return result;
}


typedef struct wvec_inputout_s
{
  const fieldtype * restrict s1;
  const fieldtype * restrict s2;
  radix scalar;
  lcomplex cscalar;
  fieldtype * out;
  int stride;
} wvec_inputout;


PARALLEL_KERNEL_BEGIN(fvec_copy, wvec_inputout, input, i, multiIdx)
{
    //d_if_sf_site(i) // TODO: How to easily incorporate special sites to the scheme?
    {
      fieldentry s1;
      get_fieldEntry(input.s1, i, &s1, multiIdx, input.stride);
      set_fieldEntry(input.out, i, &s1, multiIdx, input.stride);
    }
}
PARALLEL_KERNEL_END()

static
void fieldvec_copy_V(
        const fieldtype * s1,
        fieldtype * s2,
        multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif
  wvec_inputout input;
  input.s1 = s1;
  input.out = s2;
  input.stride = size.stride;
  KERNEL_CALL(fvec_copy, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_parallelT += s_cputimef() - t0;
  }
#endif
}


PARALLEL_KERNEL_BEGIN(wvec_scale, wvec_inputout, input, i, multiIdx)
{
  fieldentry s1, res;
  get_fieldEntry(input.s1, i, &s1, multiIdx, input.stride);
  fieldEntry_scalar_mult(&s1, input.scalar, &res );
  set_fieldEntry(input.out, i, &res, multiIdx, input.stride);
}
PARALLEL_KERNEL_END()

static
void fieldvec_scalar_mult_V(
        const fieldtype * s1,
        radix scalar,
        fieldtype * s2,
        multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif
  wvec_inputout input;
  input.s1 = s1;
  input.scalar = scalar;
  input.out = s2;
  input.stride = size.stride;
  KERNEL_CALL(wvec_scale, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_parallelT += s_cputimef() - t0;
  }
#endif
}


PARALLEL_KERNEL_BEGIN(wvec_madd, wvec_inputout, input, i, multiIdx)
{
  fieldentry s1, s2, res;
  get_fieldEntry(input.s1, i, &s1, multiIdx, input.stride);
  get_fieldEntry(input.s2, i, &s2, multiIdx, input.stride);
  fieldEntry_scalar_madd(&s1, input.scalar, &s2, &res );
  set_fieldEntry(input.out, i, &res, multiIdx, input.stride);
}
PARALLEL_KERNEL_END()

// This is res = s1 + scalar * s2
static
void fieldvec_scalar_madd_V(
        const fieldtype * s1,
        radix scalar,
        fieldtype * s2,
        fieldtype * res,
        multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif
  wvec_inputout input;
  input.s1 = s1;
  input.s2 = s2;
  input.scalar = scalar;
  input.out = res;
  input.stride = size.stride;
  KERNEL_CALL(wvec_madd, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_parallelT += s_cputimef() - t0;
  }
#endif
}


PARALLEL_KERNEL_BEGIN(wvec_cscale, wvec_inputout, input, i, multiIdx)
{
    //d_if_sf_site(i) // TODO: How to easily incorporate special sites to the scheme?
    {
      fieldentry s1, res;
      get_fieldEntry(input.s1, i, &s1, multiIdx, input.stride);
      fieldEntry_complex_mult(&s1, input.cscalar, &res );
      set_fieldEntry(input.out, i, &res, multiIdx, input.stride);
    }
}
PARALLEL_KERNEL_END()

static
void fieldvec_complex_mult_V(
        const fieldtype* s1,
        lcomplex scalar,
        fieldtype* s2,
        multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif
  wvec_inputout input;
  input.s1 = s1;
  input.cscalar = scalar;
  input.out = s2;
  input.stride = size.stride;
  KERNEL_CALL(wvec_cscale, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_parallelT += s_cputimef() - t0;
  }
#endif
}

PARALLEL_KERNEL_BEGIN(wvec_cmadd, wvec_inputout, input, i, multiIdx)
{
    //d_if_sf_site(i) // TODO: How to easily incorporate special sites to the scheme?
    {
      fieldentry a, b, res;
      get_fieldEntry(input.s1, i, &a, multiIdx, input.stride);
      get_fieldEntry(input.s2, i, &b, multiIdx, input.stride);
      fieldEntry_complex_madd(&a, input.cscalar, &b, &res);
      set_fieldEntry(input.out, i, &res, multiIdx, input.stride);
    }
}
PARALLEL_KERNEL_END()


static
void fieldvec_complex_madd_V(
        const fieldtype * a ,
        lcomplex scalar,
        const fieldtype * b,
        fieldtype * dst,
        multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nMADDs++;
#endif
  wvec_inputout input;
  input.s1 = a;
  input.cscalar = scalar;
  input.s2 = b;
  input.out = dst;
  input.stride = size.stride;
  KERNEL_CALL(wvec_cmadd, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_parallelT += s_cputimef() - t0;
  }
#endif
}

static
void fieldvec_complex_msub_V(
        const fieldtype * a ,
        lcomplex scalar,
        const fieldtype * b,
        fieldtype * dst,
        multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nMADDs++;
#endif
  wvec_inputout input;
  input.s1 = a;
  input.cscalar.real = -scalar.real;
  input.cscalar.imag = -scalar.imag;
  input.s2 = b;
  input.out = dst;
  input.stride = size.stride;
  KERNEL_CALL(wvec_cmadd, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_parallelT += s_cputimef() - t0;
  }
#endif
}



/* Normalizes a Wilson vector to unit length*/
static
void fieldvec_normalize_V(fieldtype* vec, multisize size){
  radix norm;
  norm=sqrt(fieldvec_magsq_V(vec,size));
  fieldvec_scalar_mult_V(vec,(radix)1.0/norm,vec,size);
}

typedef struct rb_inputT_s{
    fieldtype **eig_vec;
    fieldtype *tmp;
    int nVecs;
    int start;
    int stride;
    lcomplex* mat;
    int matStride;
    int blockSize;
    int size;
} rb_inputT;

PARALLEL_KERNEL_BEGIN2D(rb_copyToTemp, rb_inputT, in, idx, vecIdx, multiIdx)
{
    fieldentry x;
    int getIdx = idx + in.start;
    /*if (getIdx < in.size)*/{
        get_fieldEntry(in.eig_vec[vecIdx], getIdx, &x, multiIdx, in.stride);
        set_fieldEntry(in.tmp, idx + vecIdx * in.blockSize, &x, multiIdx, in.stride);
    }
}
PARALLEL_KERNEL_END2D()

PARALLEL_KERNEL_BEGIN2D(rb_doRotate, rb_inputT, in, idx, vecIdx, multiIdx)
{
    /*if (idx + in.start < in.size)*/{
        fieldentry res, x;
        int myStart = idx;
        lcomplex* dotvec = &in.mat[vecIdx];
        int limit = idx + in.nVecs * in.blockSize;
        get_fieldEntry(in.tmp, myStart, &x, multiIdx, in.stride);
        myStart += in.blockSize;
        fieldEntry_complex_mult(&x, *dotvec, &res);
        dotvec += in.matStride;
        for (;myStart < limit; myStart += in.blockSize){
            get_fieldEntry(in.tmp, myStart, &x, multiIdx, in.stride);
            fieldEntry_complex_madd(&res, *dotvec, &x, &res);
            dotvec += in.matStride;
        }
        set_fieldEntry(in.eig_vec[vecIdx], in.start + idx, &res, multiIdx, in.stride);
    }
}
PARALLEL_KERNEL_END2D()

static
int newrotate_basis(fieldtype **eig_vec, cmatrix *v, multisize size, int nSubEigs)
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif
    fieldtype* temp;
    int n_eigs = v->n;
    int nblocks, stride, start;
    rb_inputT input;

    int maxsize = 1024*256;
    int tmpsize = size.size < maxsize ? size.size : maxsize;

    temp = new_fieldtype(size.size, size.nMulti);
    if (!temp)
        return -1;

#ifdef MULTI_GPU
    const int blocksize =  32;
    cudaMalloc(&input.eig_vec, sizeof(fieldtype*) * v->n);
    cudaMemcpy(input.eig_vec, eig_vec, sizeof(fieldtype*) * v->n, cudaMemcpyHostToDevice);
    cudaMalloc(&input.mat, sizeof(lcomplex) * v->stride * v->n);
    cudaMemcpy(input.mat, v->e[0], sizeof(lcomplex) * v->stride * v->n, cudaMemcpyHostToDevice);
#else
    const int blocksize =  16;
    input.mat = v->e[0];
    input.eig_vec = eig_vec;
#endif

    if (tmpsize <= blocksize * n_eigs){
        if (size.size >= 4 * blocksize * n_eigs){
            tmpsize = 4 * blocksize * n_eigs;
        } else if (size.size >= 2 * blocksize * n_eigs){
            tmpsize = 2 * blocksize * n_eigs;
        } else {
            tmpsize = size.size;
        }
    }

    nblocks = tmpsize / (blocksize * n_eigs);
    if (nblocks == 0){
        free_fieldtype(temp);
        //printf("OLD ROTATE!\n");
        return 1; // Go to old codepath if very small vectors wrt, amount of requested eigenvalues
    }
    stride = blocksize * nblocks;

    //printf("\nnewrotate_basis - nblocks = %d!\n", nblocks);

    input.blockSize = stride;

    input.matStride = v->stride;
    input.nVecs = n_eigs;
    input.size = size.size;
    input.stride = size.stride;
    input.tmp = temp;

    for (start = 0; start < size.size - stride; start += stride){
        input.start = start;
        KERNEL_CALL2D(rb_copyToTemp, input, 0, stride, 0, n_eigs, size.nMulti);
        // Then do: vecs[myVec_idx][i + start] = Sum_{j=0 to n_vecs-1} m_{i+start},j * temp[i + j * stride]
        KERNEL_CALL2D(rb_doRotate, input, 0, stride, 0, nSubEigs, size.nMulti);
    }
    input.start = start;
    stride = size.size - start;
    // First do: temp[i + vec_idx * stride] = vecs[vec_idx][i + start], i \in {0,.., stride-1}, vec_idx \in {0,..,n_vecs-1}
    KERNEL_CALL2D(rb_copyToTemp, input, 0, stride, 0, n_eigs, size.nMulti);
    // Then do: vecs[myVec_idx][i + start] = Sum_{j=0 to n_vecs-1} m_{i+start},j * temp[i + j * stride]
    KERNEL_CALL2D(rb_doRotate, input, 0, stride, 0, nSubEigs, size.nMulti);



    free_fieldtype(temp);
#ifdef MULTI_GPU
    cudaFree(input.mat);
    cudaFree(input.eig_vec);
#endif
#ifdef TRACE_STATISTICS
  if (s_cputimef)
      s_parallelT += s_cputimef() - t0;
#endif

    return 0;
}


#define DEBUG_ROTATE_BASIS  0
#if DEBUG_ROTATE_BASIS
static void printVecs(fieldtype **eig_vec, int nvecs, int size){
    lcomplex* data;
    int i,j;
#ifdef MULTI_GPU
    data = (lcomplex*)s_mallocf(sizeof(lcomplex) * size);
#endif
    for (i = 0; i < nvecs; i++){
  #ifdef MULTI_GPU
        cudaMemcpy(data, eig_vec[i], sizeof(lcomplex) * size, cudaMemcpyDeviceToHost);
  #else
        data = (lcomplex*)eig_vec[i];
  #endif
        printf("Eig_vec[%d] = \n\t [ (%.5f ; %.5f) ", i, data[0].real, data[0].imag);
            for (j = 1; j < size; j++)
                printf(", (%.5f ; %.5f) ", data[j].real, data[j].imag);
    }

  #ifdef MULTI_GPU
   s_freef(data);
  #endif

}
#endif


/* rotate the eigen vector basis by v
   w'_i = v_ji w_j,
   where v is the matrix of the column eigen vectors of
   M generated above in construct_array(..) function.
*/

static
int rotate_basis(fieldtype **eig_vec, cmatrix *v, multisize size, int nSubEigs)
{
  fieldtype **tmp;
  int n_eigs;
  int i,j;
  int error = 0;

#if DEBUG_ROTATE_BASIS
  printf("Debug Rotate basis:\n \t matrix dims: (%d, %d)\n\tEigenvectors:\n", nSubEigs, v->n);
  printVecs(eig_vec, v->n, size.size);
  printf("\n\n Matrix:\n");
  for (i = 0; i < v->n; i++){
      printf("\nm[%d] = [ (%.5f, %.5f)", i, v->e[i][0].real, v->e[i][0].imag);
      for (j = 1; j < nSubEigs; j++){
          printf(", (%.5f, %.5f)", v->e[i][j].real, v->e[i][j].imag);
      }
  }
#endif

#if 1
  error = newrotate_basis(eig_vec, v, size, nSubEigs);

  if (error <= 0){
#if DEBUG_ROTATE_BASIS
  printf("\n\nResult of Rotate basis:\n\tEigenvectors:\n");
  printVecs(eig_vec, nSubEigs, size.size);
#endif
      return error;
  }
#endif
  error = 0;

  // Fallback codepath:

  n_eigs = v->n;
  tmp = (fieldtype **) s_mallocf(nSubEigs*sizeof(fieldtype*));
  if (!tmp){
      error = -1;
      goto cleanup;
  }

  for (i=0;i<nSubEigs;i++){
    tmp[i] = new_fieldtype(size.size, size.nMulti);
    if (!tmp[i]){
        nSubEigs = i;
        error = -2;
        goto cleanup;
    }
  }
  for (i=0;i<nSubEigs;i++){
    //wvec_zero_V(tmp[i],parity);
    fieldvec_complex_mult_V(eig_vec[0], v->e[0][i], tmp[i], size);
    for (j=1;j<n_eigs;j++){
      /* tmp[i] = tmp[i] + v->e[j][i]*eig_vec[j] */
      fieldvec_complex_madd_V(tmp[i], v->e[j][i], eig_vec[j], tmp[i], size);
    }
  }
  for (i=0;i<nSubEigs;i++){
    fieldvec_copy_V(tmp[i], eig_vec[i], size);
  }

cleanup:
  if (tmp){
      for (i=0;i<nSubEigs;i++){
          free_fieldtype(tmp[i]);
      }
      s_freef(tmp);
  }
#if DEBUG_ROTATE_BASIS
  printf("\n\nResult of Rotate basis:\n\tEigenvectors:\n");
  printVecs(eig_vec, nSubEigs, size.size);
#endif
  return error;
}


/* End recycled code */

typedef struct shift_in_s
{
    fieldtype * restrict dest;
    fieldtype * restrict temp1;
    radix sigma;
    radix invr;
    int stride;
} shift_in;

PARALLEL_KERNEL_BEGIN(shift_kernel, shift_in, input, i, multiIdx)
{
    fieldentry dest, temp1, temp2;
    radix sigma = input.sigma;
    radix invr = input.invr;
    get_fieldEntry(input.dest, i, &dest, multiIdx, input.stride);
    get_fieldEntry(input.temp1, i, &temp1, multiIdx, input.stride);
    fieldEntry_scalar_mult(&dest, invr, &temp2);
    fieldEntry_scalar_madd(&temp2, sigma, &temp1, &dest);
    set_fieldEntry(input.dest, i, &dest, multiIdx, input.stride);
    set_fieldEntry(input.temp1, i, &dest, multiIdx, input.stride);

}
PARALLEL_KERNEL_END()
/*
 * Do a single peeling transformation shift.
 *
 * While src is untouched, dest is overwritten.
 */
#ifdef SHIFT
static
void shift_once(
        ColorSpinorField  *src_s,
        ColorSpinorField  *dest_s,
        multisize size, int N, radix sigma, radix r)
{

  fieldtype *temp1 = nullptr;
  int i;

  fieldtype *src  = static_cast<fieldtype *>(src_s->V());//fieldtype * restrict
  fieldtype *dest = static_cast<fieldtype *>(dest_s->V());//fieldtype * restrict

  ColorSpinorParam csParam(*src_s);
  ColorSpinorField *temp1_s = ColorSpinorField::Create(csParam);

  fieldtype *temp1 = static_cast<fieldtype*>(temp1_s->V());

  fieldvec_copy_V(src, temp1, size);

  for(i=0;i<N;i++) {
    (*s_mulf)(*temp1_s,*dest_s);
    {
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif
        shift_in input;
        input.temp1 = temp1;
        input.dest = dest;
        input.invr = (radix)1.0/r;
        input.sigma = sigma;
        input.stride = size.stride;
        KERNEL_CALL(shift_kernel, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
        if (s_cputimef){
#ifdef MULTI_GPU
            cudaDeviceSynchronize();
#endif
            s_parallelT += s_cputimef() - t0;
        }
#endif

    }
    /*fieldvec_scalar_mult_V(dest, 1.0/r, temp2, size);
    wvec_scalar_mult_add_V(temp2,temp1,sigma,dest,parity);
    wvec_copy_V(dest,temp1,parity);*/
  }
  delete temp1_s;
}


/* Carry out several peeling transformations.
 *
 * Separated from shift_once, above, for
 * clarity. Memory alloction causes negligible overhead.
 */
static
void shift_mult( ColorSpinorField *src, ColorSpinorField *dest, multisize size )
{
  //fieldtype *temp1, *temp2, *temp3;

  //temp1 = new_fieldtype(size);
  // temp2 = new_fieldtype(size);
  // temp3 = new_new_fieldtype(size);

  // r = -1 rather than 1 to reverse order, eg:
  // shift_once(src,dest,parity,1,0.0,-1.0);
  // -- then can use largest first and still get smallest

  shift_once(src, dest, size, 4,4.0,0.05);
  //  shift_once(temp1,dest,size,1,0.0,-1.0);
  //  shift_once(temp2,temp3,size,6,2.0,500.0);
  //  shift_once(temp3,dest,size,2,16.0,1000.0);


  //free_fieldtype(temp1);
  //  free_fieldtype(temp2);
  // free_fieldtype(temp3);
}
#endif

typedef struct multi_fvec_cdot_in_s{
       fieldtype** V;
       int stride;
       int nMulti;
} multi_fvec_cdot_in;


PARALLEL_MVECMUL_BEGIN(multi_fvec_cdot, multi_fvec_cdot_in, input, y, x, fieldtype*, z, lcomplex, res)
{
       fieldentry m_yx, a;
       int multiIdx = 0;
       get_fieldEntry(input.V[y], x, &m_yx, multiIdx, input.stride);
       get_fieldEntry(z, x, &a, multiIdx, input.stride);
       res = fieldEntry_dot(&m_yx, &a);
       for (multiIdx = 1; multiIdx < input.nMulti; multiIdx++){
           lcomplex tmp;
           get_fieldEntry(input.V[y], x, &m_yx, multiIdx, input.stride);
           get_fieldEntry(z, x, &a, multiIdx, input.stride);
           tmp = fieldEntry_dot(&m_yx, &a);
           res.real += tmp.real;
           res.imag += tmp.imag;
       }
}
PARALLEL_MVECMUL_SUMFUN(multi_fvec_cdot, res, b, lcomplex)
{
       res.real += b.real;
       res.imag += b.imag;
}
PARALLEL_MVECMUL_STOREFUN(multi_fvec_cdot, res, lcomplex, lcomplex*, dst, y)
{
       dst[y] = res;
}
PARALLEL_MVECMUL_END()


// Sometimes this one seems a tad faster, but on small vectors not (not enough unrolling here)
#ifndef USE_MATRIX_MUL_CODEPATH
#define USE_MATRIX_MUL_CODEPATH 0
#endif


static void multi_fvec_cdot_V(fieldtype **V, fieldtype *z, int n, int maxressize, lcomplex* results, multisize size){
    lcomplex* tmp = results;
#if USE_MATRIX_MUL_CODEPATH || defined(MULTI_GPU)
    fieldtype **tmpV = V;
#endif
    int j;
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nCDotProds += n;
#endif

#ifdef MULTI_GPU
    lcomplex* devRes = NULL;
    fieldtype **devV = NULL;
    cudaMalloc(&devRes, sizeof(lcomplex) * maxressize);
    tmp = devRes;
    cudaMalloc(&devV, sizeof(fieldtype*) * maxressize);
    cudaMemcpy(devV, V, sizeof(fieldtype*) * n, cudaMemcpyHostToDevice);
    tmpV = devV;
#else
    (void)maxressize;
#endif
#if USE_MATRIX_MUL_CODEPATH
    {
        multi_fvec_cdot_in input;
        input.V = tmpV;
        input.stride = size.stride;
        input.nMulti = size.nMulti;
        CALL_MVECMUL_KERNEL(multi_fvec_cdot, input, z, size.size, n, tmp, lcomplex);
    }
#else
    for(j=0; j < (n); j++) {
      if (s_nStreams > 0)
          setCurStream(j);
      fieldvec_cdot_V_skipmpi(V[j], z, size, &tmp[j], 1);
    }
      //wvec_dot_V(V[j],z,parity);
    // First wait for all dot-products to complete - this way we don't go changing z while we're
    // still waiting for dot product results
    for(j=0; j < n; j++) {
        waitStream(j);
    }
    setCurStream(-1);
#endif

#ifdef MULTI_GPU
    cudaMemcpy(results, devRes, sizeof(lcomplex) * n, cudaMemcpyDeviceToHost);
    cudaFree(devRes);
    cudaFree(devV);
#endif

#ifdef TRACE_STATISTICS
    if (s_cputimef){
      s_parallelT += s_cputimef() - t0;
    }
#endif

}
typedef struct superpos_sub_in_s{
    int n;
    fieldtype** V;
    const fieldtype* z;
    fieldtype* dst;
    lcomplex* scalars;
    int stride;
} superpos_sub_in;

PARALLEL_KERNEL_BEGIN(superpos_sub, superpos_sub_in, in, i, multiIdx)
{
    int j;
    fieldentry res;
    get_fieldEntry(in.V[0], i, &res, multiIdx, in.stride);
    fieldEntry_complex_mult(&res, in.scalars[0], &res);
    for (j = 1; j < in.n; j++){
        fieldentry x;
        lcomplex alpha = in.scalars[j];
        get_fieldEntry(in.V[j], i, &x, multiIdx, in.stride);
        fieldEntry_complex_madd(&res, alpha, &x, &res);
    }
    {
        fieldentry x, t;
        get_fieldEntry(in.z, i, &x, multiIdx, in.stride);
        fieldEntry_scalar_madd(&x, -1.0, &res, &t); // t = z - res
        set_fieldEntry(in.dst, i, &t, multiIdx, in.stride);
    }
}
PARALLEL_KERNEL_END()

static void
fieldvec_linearsuperpos_sub(fieldtype** V, const fieldtype* z, int n, lcomplex* scalars, fieldtype* dst, multisize size){
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nMADDs += n;
#endif
    lcomplex* tmp = scalars;
    fieldtype** tmpVecs = V;
    superpos_sub_in input;
#ifdef MULTI_GPU
    lcomplex* devScalars = NULL;
    fieldtype** devVecs = NULL;
    cudaMalloc(&devScalars, sizeof(lcomplex) * n);
    cudaMalloc(&devVecs, sizeof(fieldtype*) * n);
    tmp = devScalars;
    tmpVecs = devVecs;
    cudaMemcpy(devScalars, scalars, sizeof(lcomplex) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(devVecs, V, sizeof(fieldtype*) * n, cudaMemcpyHostToDevice);
#endif
    input.V = tmpVecs;
    input.dst = dst;
    input.n = n;
    input.scalars = tmp;
    input.stride = size.stride;
    input.z = z;
    KERNEL_CALL(superpos_sub, input, 0, size.size, size.nMulti);
#ifdef MULTI_GPU
    cudaFree(devScalars);
    cudaFree(devVecs);
#endif
#ifdef TRACE_STATISTICS
  if (s_cputimef)
      s_parallelT += s_cputimef() - t0;
#endif
}

typedef struct fvec_msub_dot_in_s{
    fieldtype* Vj;
    fieldtype* Vj1;
    fieldtype* z;
    lcomplex malpha;
    int stride;
} fvec_msub_dot_in;

PARALLEL_REDUCE_BEGIN(field_vec_msub_dot, fvec_msub_dot_in, in, i, lcomplex, a_out, multiIdx)
{
    fieldentry z, vj, vj1;
    get_fieldEntry(in.z, i, &z, multiIdx, in.stride);
    get_fieldEntry(in.Vj, i, &vj, multiIdx, in.stride);
    get_fieldEntry(in.Vj1, i, &vj1, multiIdx, in.stride);
    fieldEntry_complex_madd(&z, in.malpha, &vj, &z);
    a_out = fieldEntry_dot(&vj1, &z);
    set_fieldEntry(in.z, i, &z, multiIdx, in.stride);
}
PARALLEL_REDUCE_SUMFUN(field_vec_msub_dot, tmpres, a_out, lcomplex)
{
    a_out.real += tmpres.real;
    a_out.imag += tmpres.imag;
}
PARALLEL_REDUCE_END(a_out)

// Compute: z = z - alpha V[j], alpha = z^V[j+1]
static void field_vec_msub_dot_V(fieldtype* Vj, lcomplex* alpha, fieldtype* z, fieldtype* Vj1, multisize size)
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nCDotProds++;
    s_nMADDs++;
#endif
    fvec_msub_dot_in in;
    lcomplex res;
    in.Vj = Vj;
    in.Vj1 = Vj1;
    in.stride = size.stride;
    in.z = z;
    in.malpha.real = -alpha->real;
    in.malpha.imag = -alpha->imag;
    FOR_RANGE_REDUCE_KERNEL(field_vec_msub_dot, in, &res, 0, size.size, size.nMulti, 0, 0);
    *alpha = res;
    if (complex_reduction_f){
      double d_result[2] = { static_cast<double>(alpha->real), static_cast<double>(alpha->imag) };
      complex_reduction_f(d_result);
      alpha->real = d_result[0];
      alpha->imag = d_result[1];
    }

#ifdef TRACE_STATISTICS
    if (s_cputimef)
        s_parallelT += s_cputimef() - t0;
#endif
}

/*
 * Recompute the Arnoldi factorisation.
 *
 * Calculates out to a (k+p)*(k+p) Hessenberg given a k*k one
 *
 * It is widely discussed in linear algebra textbooks, or see
 * Sorensen's technical report (ICASE Report 96-40).
 *
 * This routine has nothing to do with the restart strategy.
 */
static
int arnoldi_step(
    int k, int p, ColorSpinorFieldSet *V_s, cmatrix *H,
    ColorSpinorField *f_s, multisize size)
{
#ifdef TRACE_STATISTICS
  s_nArnSteps++;
  double t0 = s_cputimef ? s_cputimef() : 0.0;
  double t1;
#endif

  int i, j;
  radix beta;
  lcomplex alpha, alpha2;
  int error = 0;

  fieldtype **V = (fieldtype**)s_mallocf(sizeof(fieldtype*) * V_s->CompositeDim());
  for(i = 0; i < V_s->CompositeDim(); i++)  fieldtype *V = static_cast<fieldtype *> (V_s->Component(i).V());

  fieldtype *f  = static_cast<fieldtype * > (f_s->V()); 

  // initialise v, z
  //fieldtype *v = new_fieldtype(size.size, size.nMulti);
  ColorSpinorParam csParam(V_s->Component(0));
  ColorSpinorField *z_s = ColorSpinorField::Create(csParam);
  fieldtype *z = static_cast<fieldtype *>(z_s->V());
  //@fieldtype *z = new_fieldtype(size.size, size.nMulti);//this has to be external!

  lcomplex* Vdotzs = (lcomplex*)s_mallocf(sizeof(lcomplex) * H->n);


  if (!(/*v &&*/ z && Vdotzs)){
      error = -1;
      goto cleanup;
  }

  /*
   * on entry, we have an Arnoldi factorisation:
   * A V = V H + [0,0,(k-1 times),f]
   *
   * We compute steps k to p of this iteration
   * so we assume the matrices are at least that big
   * and that V, H and f have been set up for us
   */


  // Zero rest of Hessenberg
  for(i=k;i<(k+p);i++) {
    for(j=0;j<(k+p);j++) {
      H->e[i][j].real = 0.0;
      H->e[i][j].imag = 0.0;

      H->e[j][i].real = 0.0;
      H->e[j][i].imag = 0.0;
    }
  }

  // On entry, V has i columns
  for(i=k;i<(k+p);i++) {

#ifdef EIG_DEBUG
    //fprintf0(outf,"Arnoldi step %d/%d\n",i,(k+p)-1);
      printf("Arnoldi step %d/%d\n",i,(k+p)-1);
#endif

    beta = sqrt(fieldvec_magsq_V(f, size));

    fieldvec_scalar_mult_V(f, (radix)(1.0/beta), V[i]/*v*/, size);

    //fieldvec_copy_V(v, V[i], size);

    H->e[i][i-1].real = beta;
    H->e[i][i-1].imag = 0.0;

#ifndef SHIFT
    // NOTE: Assumes Mat_mult does not change V[i]
    //Mat_mult(V[i]/*v*/,z);
    (*s_mulf)(V_s->Component(i)/*v*/,*z_s);
#else
    //shift_mult(V[i]/*v*/,z,size);
#endif

    multi_fvec_cdot_V(V, z, i+1, H->n, Vdotzs, size);
#ifdef MULTI_GPU
    for(j=0; j < (i+1); j++)
        if (complex_reduction_f) {
            //complex_reduction_f(&Vdotzs[j]);
            double d_result[2] = { static_cast<double>(Vdotzs[j].real), static_cast<double>(Vdotzs[j].imag) };
            complex_reduction_f(d_result);
            Vdotzs[j].real = d_result[0];
            Vdotzs[j].imag = d_result[1];
        }
    fieldvec_linearsuperpos_sub(V, z, i+1, Vdotzs, z, size);
    for(j=0; j < (i+1); j++) {
        H->e[j][i] = Vdotzs[j];
    }
#else
    for(j=0; j < (i+1); j++) {
        H->e[j][i] = Vdotzs[j];
    }

    for(j=0; j < (i+1); j++) {
      // Store the result in temp_z to avoid pointer aliasing
      // operation is diagonal and hence destination can be one of the sources
      if (complex_reduction_f){
          //complex_reduction_f(&H->e[j][i]);
          double d_result[2] = { static_cast<double>(H->e[j][i].real), static_cast<double>(H->e[j][i].imag) };
          complex_reduction_f(d_result);
          H->e[j][i].real = d_result[0];
          H->e[j][i].imag = d_result[1];
      }
      fieldvec_complex_msub_V(z, (H->e[j][i]), V[j], z, size);
    }
#endif
    // First, should determine *IF* reorthogonalisation is necessary
    // Also ought to deal with invariant subspaces.
    // Reorthogonalize. Only do one step; could do more.
    // Effectively a modified Gram-Schmidt step
    // NOTE: This correction fails due to finite machine precision stuffses.
    //multi_fvec_cdot_V(V, z, i+1, H->n, Vdotzs, size);
#ifdef TRACE_STATISTICS
    t1 = s_cputimef ? s_cputimef() : 0.0;
    s_nCDotProds++;
    s_nMADDs++;
#endif

#if 1
    alpha = fieldvec_cdot_V(V[0], z, size);
    for(j=0;j<i;j++) {
        alpha2 = H->e[j][i];
        H->e[j][i] = caddf(&alpha, &alpha2);
        field_vec_msub_dot_V(V[j], &alpha, z, V[j+1], size);
    }
    fieldvec_complex_msub_V(z, alpha, V[i], z, size);
#else
    for(j=0;j<(i+1);j++) {
      //  if (complex_reduction_f)
      //      complex_reduction_f(&Vdotzs[j]);
      //alpha = Vdotzs[j];
      alpha = fieldvec_cdot_V(V[j], z, size);
      //printf("alpha(%d)(%d) = %e + i %e\n", i, j , alpha.real, alpha.imag);

      fieldvec_complex_msub_V(z, alpha, V[j], z, size);

      alpha2 = H->e[j][i];
      H->e[j][i] = caddf(&alpha, &alpha2);
    }
#endif

    fieldvec_copy_V(z,f,size);
#ifdef TRACE_STATISTICS
    if (s_cputimef)
        s_orthT += s_cputimef() - t1;;
#endif

  }
#ifdef TRACE_STATISTICS
  if (s_cputimef)
      s_totArnStepT += s_cputimef() - t0;
#endif
cleanup:
  if (V) s_freef(V);
  //@if (z) free_fieldtype(z);
  if(z_s) delete z_s;
 
  if (Vdotzs) s_freef(Vdotzs);
  return error;

}



PARALLEL_KERNEL_BEGIN(wvec_subtwo, wvec_inputout, input, i, multiIdx)
{
    //d_if_sf_site(i) // TODO: How to easily incorporate special sites to the scheme?
    {
      fieldentry s1, s2, z;
      get_fieldEntry(input.out, i, &z, multiIdx, input.stride);
      get_fieldEntry(input.s2, i, &s2, multiIdx, input.stride);
      get_fieldEntry(input.s1, i, &s1, multiIdx, input.stride);
      fieldEntry_complex_madd(&z, input.cscalar, &s2, &z);
      fieldEntry_scalar_madd(&z, input.scalar, &s1, &z);
      set_fieldEntry(input.out, i, &z, multiIdx, input.stride);
    }
}
PARALLEL_KERNEL_END()


static
void fieldvec_subtwo(
        fieldtype * z,
        radix scalar,
        const fieldtype * s1,
        lcomplex cscalar,
        const fieldtype * s2,
        multisize size )
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nMADDs++;
#endif
  wvec_inputout input;
  input.s1 = s1;
  input.scalar = -scalar;
  input.cscalar.real = -cscalar.real;
  input.cscalar.imag = -cscalar.imag;
  input.s2 = s2;
  input.out = z;
  input.stride = size.stride;
  KERNEL_CALL(wvec_subtwo, input, 0, size.size, size.nMulti);
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_parallelT += s_cputimef() - t0;
  }
#endif
}


/*
 * Recompute the Lanczos factorisation.
 *
 * Calculates out to a (k+p)*(k+p) Tridiagonal given a k*k one
 *
 *
 * This routine has nothing to do with the restart strategy.
 */
static
int lanczos_step(
    int k, int p, ColorSpinorFieldSet *V_s, cmatrix *H,
    ColorSpinorFieldSet *f_s, multisize size)
{
#ifdef TRACE_STATISTICS
  s_nArnSteps++;
  double t0 = s_cputimef ? s_cputimef() : 0.0;
  double t1;
#endif

  int i, j;
  radix beta;
  lcomplex alpha, alpha2;
  int error = 0;

  // initialise v, z
  //@fieldtype *z = new_fieldtype(size.size, size.nMulti);
  ColorSpinorParam csParam(V_s->Component(0));
  ColorSpinorField *z_s = ColorSpinorField::Create(csParam);
  fieldtype *z = static_cast<fieldtype *>(z_s->V());

  fieldtype **V = (fieldtype**)s_mallocf(sizeof(fieldtype*) * V_s->CompositeDim());
  for(i = 0; i < V_s->CompositeDim(); i++)  fieldtype *V = static_cast<fieldtype *> (V_s->Component(i).V());

  fieldtype *f  = static_cast<fieldtype *>(f_s->V()); 


  //lcomplex* Vdotzs = (lcomplex*)s_mallocf(sizeof(lcomplex) * H->n);


  if (!z){
      error = -1;
      goto cleanup;
  }


  /*
   * on entry, we have an Lanczos factorisation:
   * A V = V H + [0,0,(k-1 times),f]
   *
   * We compute steps k to p of this iteration
   * so we assume the matrices are at least that big
   * and that V, H and f have been set up for us
   */


  // Zero rest of Tridiagonal - TODO: Optimize to sparse format?
  for(i=k;i<(k+p);i++) {
    for(j=0;j<(k+p);j++) {
      H->e[i][j].real = 0.0;
      H->e[i][j].imag = 0.0;

      H->e[j][i].real = 0.0;
      H->e[j][i].imag = 0.0;
    }
  }

  // On entry, V has i columns
  for(i=k;i<(k+p);i++) {

#ifdef EIG_DEBUG
      printf("Lanczos step %d/%d\n",i,(k+p)-1);
#endif

    beta = sqrt(fieldvec_magsq_V(f, size));

    fieldvec_scalar_mult_V(f, (radix)(1.0/beta), V[i]/*v*/, size);

    //fieldvec_copy_V(v, V[i], size);

    H->e[i][i-1].real = beta;
    H->e[i][i-1].imag = 0.0;
    H->e[i-1][i].real = beta;
    H->e[i-1][i].imag = 0.0;

#ifndef SHIFT
    // NOTE: Assumes Mat_mult does not change V[i]
    //Mat_mult(V[i]/*v*/,z);
    (*s_mulf)(V_s->Component(i)/*v*/,*z_s);
#else
    //shift_mult(V[i]/*v*/,z,size);
#endif

    alpha = fieldvec_cdot_V(V[i], z, size);
    H->e[i][i] = alpha;

    /*if (i > 0)
        fieldvec_scalar_madd_V(z, -beta, V[i-1], z, size);
    fieldvec_complex_msub_V(z, alpha, V[i], z, size);
    */
    fieldvec_subtwo(z, beta, V[i-1], alpha, V[i], size);

    // First, should determine *IF* reorthogonalisation is necessary
    // Also ought to deal with invariant subspaces.
    // Reorthogonalize. Only do one step; could do more.
    // Effectively a modified Gram-Schmidt step
    // NOTE: This correction fails due to finite machine precision stuffses.
    //multi_fvec_cdot_V(V, z, i+1, H->n, Vdotzs, size);
#ifdef TRACE_STATISTICS
    t1 = s_cputimef ? s_cputimef() : 0.0;
    s_nCDotProds++;
    s_nMADDs++;
#endif

    alpha = fieldvec_cdot_V(V[0], z, size);
    for(j=0;j<i;j++) {
        alpha2 = H->e[j][i];
        if (i==j || j==i-1 || j==i+1)
            H->e[j][i] = caddf(&alpha, &alpha2);
        field_vec_msub_dot_V(V[j], &alpha, z, V[j+1], size);
    }
    fieldvec_complex_msub_V(z, alpha, V[i], z, size);

    fieldvec_copy_V(z,f,size);
#ifdef TRACE_STATISTICS
    if (s_cputimef)
        s_orthT += s_cputimef() - t1;;
#endif

  }
#ifdef TRACE_STATISTICS
  if (s_cputimef)
      s_totArnStepT += s_cputimef() - t0;
#endif
cleanup:
  if (V) s_freef(V);
  if (z_s) delete z_s;

  return error;
}


/*
 * Compute the QR decomposition using Givens rotations: A = QR
 * the argument R is the matrix A on entry.
 *
 * As for the previous function, but only the
 * sine and cosine Givens parameters are stored;
 * the array givens is then of length 2*N
 * with each rotation stored columnwise.
 *
 * On exit, the R matrix will still have been
 * constructed explicitly.
 */
static
void QRfactor_hessenberg_givens(cmatrix *restrict R, lcomplex *restrict givens)
{

  const double smallnum = 1e-25;

  int k, l;
  lcomplex signf, top, c, s, sco, f, g, gco, s1, s2;
  double z1, f2, g2;


  for(k=0;k<((R->n) - 1);k++) {

    // Construct the Givens rotation (carefully)

    f = R->e[k][k];
    g = R->e[k+1][k];

    f2 = cabs_sqf(&f);
    g2 = cabs_sqf(&g);


    if(g2 < smallnum) {

      c.real = 1.0;
      c.imag = 0.0;
      s.real = 0.0;
      s.imag = 0.0;

    } else if(f2 < smallnum) {

      c.real = 0.0;
      c.imag = 0.0;

      s.real = g.real/(radix)sqrt(g2);
      s.imag = (radix)-1.0*g.imag/(radix)sqrt(g2);

    } else {

      z1 = sqrt(f2 + g2);
      signf.real = f.real/(radix)sqrt(f2);
      signf.imag = f.imag/(radix)sqrt(f2);

      c.real = (radix)sqrt(f2)/(radix)z1;
      c.imag = 0.0;

      gco = conjgf(&g);
      top = cmulf(&signf, &gco);

      s.real = (radix)top.real/(radix)z1;
      s.imag = (radix)top.imag/(radix)z1;
    }

#ifdef GIVENS_DEBUG
    fprintf0(outf, "GIVENS k=%d, f=%g + %gi, g = %g + %gi, c = %g + %gi, s= %g + %gi\n",
         k, f.real, f.imag, g.real, g.imag, c.real, c.imag, s.real, s.imag);
#endif


    givens[k] = c;
    givens[(R->n) + k] = s;


    for(l=0;l<(R->n);l++) {

      f = R->e[k][l];
      g = R->e[k+1][l];

      s1 = cmulf(&f, &c);
      s2 = cmulf(&g, &s);

      R->e[k][l] = caddf(&s1, &s2);

      sco = conjgf(&s);

      s1 = cmulf(&f, &sco);
      s2 = cmulf(&g, &c);

      R->e[k+1][l] = csubf(&s2, &s1);

    }
  }

  // R is upper triangular, let's clean it up
  for(k=0;k<((R->n) - 1);k++) {
    for(l=k+1;l<(R->n);l++) {
      R->e[l][k].real = 0.0;
      R->e[l][k].imag = 0.0;
    }
  }

}


/*
 * Using an array of givens rotations (computed above),
 * apply the conjugate transpose of the corresponding unitary
 * operator to a Hessenberg matrix.
 */
static
void apply_givens_hessenberg(lcomplex *restrict givens, cmatrix *restrict H) {
  lcomplex f, g, s1, s2, c, s, sco;
  int k, l;

  for(k=0;k<((H->n) - 1);k++) {

    c = givens[k];
    s = givens[(H->n) + k];

    for(l=0;l<(H->n);l++) {

      sco = conjgf(&s);

      // Now apply conjugate to Q
      f = H->e[k][l];
      g = H->e[k+1][l];

      s1 = cmulf(&f, &c);
      s2 = cmulf(&g, &s);

      H->e[k][l] = caddf(&s1, &s2);

      s1 = cmulf(&f, &sco);
      s2 = cmulf(&g, &c);

      H->e[k+1][l] = csubf(&s2, &s1);

    }
  }

}


/*
 * Postmultiply the matrix Q by the unitary operator
 * built from the supplied givens rotations
 * -- this is totally general, unlike the above function.
 */
static
void right_givens(cmatrix *restrict Q, lcomplex *restrict givens) {
  lcomplex f, g, s1, s2, cco, s, sco;
  int k, l;

  for(k=0;k<((Q->n) - 1);k++) {

    cco = conjgf(& (givens[k]));
    s = givens[(Q->n) + k];
    sco = conjgf(&s);

    for(l=0;l<(Q->n);l++) {



      // Now apply conjugate to Q
      f.real = Q->e[l][k].real;
      f.imag = Q->e[l][k].imag;
      g.real = Q->e[l][k+1].real;
      g.imag = Q->e[l][k+1].imag;

      s1 = cmulf(&f, &cco);
      s2 = cmulf(&g, &sco);

      Q->e[l][k] = caddf(&s1, &s2);

      s1 = cmulf(&f, &s);
      s2 = cmulf(&g, &cco);

      Q->e[l][k+1] = csubf(&s2, &s1);

    }


  }
}


typedef struct givens_in_s
{
    //fieldtype * restrict tmpi;
    //fieldtype * restrict tmpi_plus1;
    fieldtype ** restrict tmpis;
    lcomplex* givens;
    int n_eigs;

    //lcomplex cco;
    //lcomplex sco;
    //lcomplex ms;
    int stride;
} givens_in;

PARALLEL_KERNEL_BEGIN(givens, givens_in, input, j, multiIdx)
{
    //d_if_sf_site(j)
    int i;
    fieldentry newi1;
    get_fieldEntry(input.tmpis[0], j, &newi1, multiIdx, input.stride);
    for (i=0;i<input.n_eigs-1;i++){
        fieldentry tmpi, tmpi1;
        fieldentry f, g, s1;
        lcomplex cco, s, sco;
        lcomplex ms;

        cco = conjgf(& (input.givens[i]));
        s = input.givens[input.n_eigs + i];
        sco = conjgf(&s);
        ms.real = -s.real;
        ms.imag = -s.imag;

        tmpi = newi1;
        get_fieldEntry(input.tmpis[i+1], j, &tmpi1, multiIdx, input.stride);



        f = tmpi;
        g = tmpi1;

        fieldEntry_complex_mult(&f, cco, &s1);

        fieldEntry_complex_madd(&s1, sco, &g, &tmpi);
        //fieldEntry_complex_mult(&g, input.sco, &s2);
        //fieldEntry_add(&s1, &s2, &tmpi);
        set_fieldEntry(input.tmpis[i], j, &tmpi, multiIdx, input.stride);

        fieldEntry_complex_mult(&f, ms, &s1);
        fieldEntry_complex_madd(&s1, cco, &g, &tmpi1);
        //fieldEntry_complex_mult(&g, input.cco, &s2);
        //fieldEntry_sub(&s2, &s1, &tmpi1);
        //set_fieldEntry(input.tmpis[i+1], j, &tmpi1, multiIdx, input.stride);
        newi1 = tmpi1;
    }
}
PARALLEL_KERNEL_END()

/*
 * As with rotate_basis, transforms
 * eigenvectors to new basis, but
 * does so using Givens rotations
 * -- therefore uses O(eig_vec*length(wilson_vector)) iterations
 * rather than O(eig_vec*eig_vec*length(wilson_vector)).
 *
 * Replacing rotate_basis with givens_rotate significantly
 * improves performance (new execution time 60% shorter).
 */
void MANGLE(givens_rotate)(lcomplex *givens, fieldtype **eig_vec, multisize size, int n_eigs)
{
#ifdef TRACE_STATISTICS
    double t0 = s_cputimef ? s_cputimef() : 0.0;
    s_nGivenss++;
#endif
  //lcomplex cco, s, sco;
  //int i;
  fieldtype **tmp;
  lcomplex* tmp_givens;
#ifdef MULTI_GPU
  cudaMalloc(&tmp, sizeof(fieldtype*) * n_eigs);
  cudaMemcpy(tmp, eig_vec, sizeof(fieldtype*) * n_eigs, cudaMemcpyHostToDevice);
  cudaMalloc(&tmp_givens, sizeof(lcomplex) * (2*n_eigs));
  cudaMemcpy(tmp_givens, givens, sizeof(lcomplex) * (2*n_eigs), cudaMemcpyHostToDevice);
#else
  tmp = eig_vec;
  tmp_givens = givens;
#endif
  {
      givens_in input;
      input.givens = tmp_givens;
      input.n_eigs = n_eigs;
      input.stride = size.stride;
      input.tmpis = tmp;
      KERNEL_CALL(givens, input, 0, size.size, size.nMulti);
  }

#ifdef MULTI_GPU
  cudaFree(tmp);
  cudaFree(tmp_givens);
#endif
  //for (i=0;i<n_eigs;i++)
    //  fieldvec_copy_V(eig_vec[i], tmp[i], size);

#if 0
  for (i=0;i<n_eigs-1;i++){

    cco = conjgf(& (givens[i]));
    s = givens[n_eigs + i];
    sco = conjgf(&s);

    {
        lcomplex ms;
        ms.real = -s.real;
        ms.imag = -s.imag;
        givens_in input;
        input.cco = cco;
        input.ms = ms;
        input.sco  =sco;
        input.tmpi = tmp[i];
        input.tmpi_plus1 = tmp[i+1];
        input.stride = size.stride;
        KERNEL_CALL(givens, input, 0, size.size, size.nMulti);
    }
  }

  for (i=0;i<n_eigs;i++)
    fieldvec_copy_V(tmp[i],eig_vec[i],size);
#endif
#ifdef TRACE_STATISTICS
  if (s_cputimef){
      s_parallelT += s_cputimef() - t0;
      s_GivensT += s_cputimef() - t0;
  }
#endif
}


/*
 * Turn the givens rotations into a dense matrix.
 */
static
void givens_to_matrix(lcomplex *restrict givens, cmatrix *restrict Q) {
  cmat_one(Q);
  right_givens(Q,givens);
}


/*
 * C = A*B, dense cmatrices
 * ... and don't use Strassen's algorithm!
 *
 * In the longer term:
 *  - Implement Strassen
 *  - Specialise to Hessenberg
 */
static
void cmat_mult(cmatrix *restrict A, cmatrix *restrict B,
           cmatrix *restrict C){
  int i,j,k;

  lcomplex temp1, temp2;

  for (i=0;i<A->n;i++){
      for (k=0;k<A->n;k++){
    C->e[i][k].real = 0.0;
    C->e[i][k].imag = 0.0;
      }
  }

  // Loops swapped to allow compiler to choose stride
  for (i=0;i<A->n;i++){
    for (j=0;j<A->n;j++){
      for (k=0;k<A->n;k++){
          temp1 = cmulf(&(A->e[i][j]),&(B->e[j][k]));
          temp2 = caddf(&temp1,&(C->e[i][k]));

          C->e[i][k] = temp2;
      }
    }
  }
}




/* Takes (conjugate) transpose of a and stores it in b */
static
void cmat_ctrans(cmatrix *a,cmatrix *b){
  int i,j;
  int n = a->n;
  for (i=0;i<n;i++){
    for (j=0;j<n;j++){
      b->e[i][j].real = a->e[j][i].real;
      b->e[i][j].imag = (radix)-1.0*a->e[j][i].imag;
    }
  }
}



/* Assuming b is smaller matrix, put
 * the bottom right hand corner of a in b */
static
void cmat_submat(cmatrix *a, cmatrix *b) {
  int i, j;
  int n = a->n;
  int n0 = a->n - b->n;

  for(i=n0; i<n; i++) {
    for(j=n0; j<n; j++) {
      b->e[i-n0][j-n0].real = a->e[i][j].real;
      b->e[i-n0][j-n0].imag = a->e[i][j].imag;
    }
  }
}


/* Assuming b is a larger matrix, put
   a in the bottom right hand corner of b */
static
void cmat_popmat(cmatrix *a, cmatrix *b) {
  int i, j;
  int n = b->n;
  int n0 = b->n - a->n;
  for(i=n0; i<n; i++) {
    for(j=n0; j<n; j++) {
      b->e[i][j].real = a->e[i-n0][j-n0].real;
      b->e[i][j].imag = a->e[i-n0][j-n0].imag;
    }
  }
}

int numerical_eigs_dense(cmatrix *restrict A, cmatrix *restrict Q) {

  int error = 0;
  // LEFT eigenvectors because of how Fortran handles the matrix
  char jobvl = 'V';
  char jobvr = 'N';
  int LEN = A->n;
  int N = LEN;


  // eigenvalues
  lcomplex *evalues = (lcomplex *)s_mallocf(LEN*sizeof(lcomplex));

  cmatrix evectors_temp = cmat_new(N);


  int ldvl = 1;

  int lwork = 4*N;


//  complex *work = (complex *)memalloc(lwork*sizeof(complex));

  //radix *rwork = (radix *)memalloc(2*N*sizeof(radix));

  lcomplex *work =  (lcomplex *)s_mallocf(lwork*sizeof(lcomplex));

  radix *rwork = (radix*)s_mallocf(2*N*sizeof(radix));

  int info = 1111;

  //lcomplex wkopt;
  if (!(evalues && evectors_temp.e && work && rwork)){
      error = -1;
      goto cleanup;
  }

  int i, j;
  for(i=0;i<evectors_temp.n;i++) {
    for(j=0;j<evectors_temp.n;j++) {
      evectors_temp.e[i][j].real = 0.0;
      evectors_temp.e[i][j].imag = 0.0;
    }
  }
#ifdef TRACE_STATISTICS
  {
    double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif

  geevfun(&jobvl, &jobvr, &N, A->e[0], &A->stride, evalues, evectors_temp.e[0], &evectors_temp.stride, NULL, &ldvl, work, &lwork, rwork, &info);

#ifdef TRACE_STATISTICS
  if (s_cputimef)
      s_geevFunT += s_cputimef() - t0;
  }
#endif


  if (info != 0){
      error = -2 + info*10;
      goto cleanup;
  }
#ifdef EIG_DEBUG
  printf("zgeev finished; info = %d\n", info);
#endif


  for(i=0; i<LEN; i++) {
    for(j=0; j<LEN; j++) {
      if(i==j) {
        A->e[i][j].real = evalues[i].real;
        A->e[i][j].imag = evalues[i].imag;
      } else {
        A->e[i][j].real = 0;
        A->e[i][j].imag = 0;
      }
    }
  }


  /* Fortran matrices are stored
   * in the opposite order to C matrices */
  cmat_ctrans(&evectors_temp,Q);
  // cmat_copy(&evectors_temp,Q);

cleanup:
  if (evalues) s_freef(evalues);
  cmat_free(&evectors_temp);
  if (work) s_freef(work);
  if (rwork) s_freef(rwork);
  return error;
}







/* Given a vector y, construct the 'orthQ' orthogonal matrix and
 * put it in Q.
 *
 * Source: figure 4.1 (p6) of "Deflation for implicitly restarted
 * Arnoldi methods" by D.C. Sorensen (Rice CAAM TR98-12)
 */
static
void orthQ(cmatrix *restrict Q, lcomplex *y) {

  int i, j;

  lcomplex sigma, yjsq;
  lcomplex tau0, tau;
  lcomplex gamma, gammastar;

  radix norm = 0.0;
  // Wipe Q clean
  cmat_zero(Q);

  // assume y not normalised
  for(i=0; i<Q->n; i++) {
    //norm += cabs_sqf(&y[i]);
    norm += CABSF(&y[i])*CABSF(&y[i]);
  }
  norm = sqrt(norm);

  // Step 1, first column is just y
  for(i=0; i<Q->n; i++) {
    Q->e[i][0].real = y[i].real/norm;
    Q->e[i][0].imag = y[i].imag/norm;
  }


  tau0.real = CABSF(&y[0]);
  tau0.imag = 0.0;
  sigma = cmulf(&tau0,&tau0);

  // Loop (step 3)
  for(j=1; j<Q->n; j++) {

    yjsq.real = CABSF(&y[j])*CABSF(&y[j]);
    //yjsq.real = cabs_sqf(&y[j]);
    yjsq.imag = 0.0;
    sigma = caddf(&sigma,&yjsq);
    tau = mycsqrt(&sigma);

    // Condition (step 3.4)
    /* CAUTION:
     * If the RHS (1e-100 currently) is too big, your matrix will
     * end up non-orthogonal. This will manifest itself in your
     * ritz values being badly disrupted after deflation.
     */
    if(CABSF(&tau0) > 1e-100) {
      gamma = cdivf(&y[j],&tau);
      gamma = cdivf(&gamma,&tau0);

      for(i=0;i<j;i++) {
    gammastar = conjgf(&gamma);
    Q->e[i][j] = cmulf(&y[i],&gammastar);
    Q->e[i][j].real = (radix)-1.0*Q->e[i][j].real;
    Q->e[i][j].imag = (radix)-1.0*Q->e[i][j].imag;
      }
      Q->e[j][j] = cdivf(&tau0,&tau);

    } else {
      Q->e[j-1][j].real = (radix)1.0;
      Q->e[j-1][j].imag = (radix)0.0;
    }

    tau0 = tau;
  }
}



/* Given a vector y, construct the 'orthU' orthogonal Hessenberg
 * matrix and put it in U.
 *
 * Source: figure 4.2 (p7) of "Deflation for implicitly restarted
 * Arnoldi methods" by D.C. Sorensen (Rice CAAM TR98-12)
 */
static
void orthU(cmatrix *restrict U, lcomplex *y) {

  int i;
  cmatrix Q = cmat_new(U->n);

  orthQ(&Q,y);


  for(i=0;i<U->n;i++) {
    int j;
    for(j=0;j<U->n-1;j++) {
      U->e[i][j] = Q.e[i][j+1];
    }
    U->e[i][j] = Q.e[i][0];
  }

/*
  for(i=0;i<U->n;i++) {
    for(j=0;j<U->n;j++) {
      U->e[i][j] = Q.e[i][(j+1)%(U->n)];
    }
  }
*/
  cmat_free(&Q);
}


/* Given a vector y (corresponding to a converged ritz value theta of
 * the Hessenberg h), with Arnoldi factorization A*eig_vec = h*V + f,
 * (with V containing y), lock the eigenvalue of h to the first column,
 * do the corresponding adjustment in eig_vec, and update f. The matrix
 * corresponding to this similarity transformation is then Q.
 *
 * Source: figure 5.1 (p9) of "Deflation for implicitly restarted
 * Arnoldi methods" by D.C. Sorensen (Rice CAAM TR98-12)
 *
 * TODO: theta not used - remove it?
 *
 */
static
int deflate(fieldtype **eig_vec,
         cmatrix *h,
         fieldtype *f,
         lcomplex *y,
         lcomplex theta,
         multisize size, cmatrix *Q) {
#ifdef TRACE_STATISTICS
  double t0 = s_cputimef ? s_cputimef() : 0.0;
  double t1;
#endif
  int j, k, l, m;

  int error = 0;

  cmatrix Qadj = cmat_new(h->n);
  cmatrix ht = cmat_new(h->n);
  cmatrix qt = cmat_new(h->n);

  lcomplex *z = (lcomplex *)s_mallocf(h->n*sizeof(lcomplex));
  //lcomplex z[h->n];

  double tot;

  //lcomplex entry;

  cmatrix U;

  //fieldtype *ftemp = new_fieldtype(size.size, size.nMulti);

  (void)theta;

  if (!(/*ftemp &&*/ z && Qadj.e && ht.e && qt.e)){
      error = -1;
      goto cleanup;
  }

  // orthQ zeros Q for us anyway
  orthQ(Q, y);

  cmat_adj(Q, &Qadj);
  cmat_mult(&Qadj, h, &ht);
  cmat_mult(&ht, Q, h);




  for(j=((h->n)); j>3; j--) {
    // 3.1
    radix invsqrt_tot;
    tot = 0.0;
    for(k=2;k<j;k++) {
      z[k-2] = h->e[j-1][k-1];
      //tot += cabs_sqf(z[k-2]);
      tot += CABSF(&z[k-2])*CABSF(&z[k-2]);
    }

    invsqrt_tot = (radix)(1.0/sqrt(tot));

    // 3.2
    for(k=2;k<j;k++) {
      z[k-2].real = z[k-2].real * invsqrt_tot;
      z[k-2].imag = (radix)-1.0*z[k-2].imag * invsqrt_tot;
    }

    // 3.3
    U = cmat_new(j-2);

    orthU(&U,z);



    cmat_copy(h,&ht);

    // 3.4
    // H(:,2:j-1) = H(:,2:j-1)U
    // H(k,m) = H(k,l) U(l,m)
    for(k=0;k<(h->n);k++) {
      for(m=1;m<(j-1);m++) {
      ht.e[k][m].real = 0.0;
      ht.e[k][m].imag = 0.0;

      for(l=1;l<(j-1);l++) {
        cmaddf(&(h->e[k][l]),&(U.e[l-1][m-1]), &(ht.e[k][m]));
        //entry = cmulf(&(h->e[k][l]),&(U.e[l-1][m-1]));

        //          entry.real = 0.0;
        //          entry.imag = 0.0;
        //ht.e[k][m] = caddf(&(ht.e[k][m]),&entry);

      }
    }
    }

    cmat_copy(&ht,h);

    // ht is now current h
    // 3.5
    // H(2:j-1,:) = U' H(2:j-1,:)
    // H(k,m) = conjgf(U(l,k)) H(l,m)
    for(k=1;k<(j-1);k++) {
      for(m=0;m<(h->n);m++) {
      ht.e[k][m].real = 0.0;
      ht.e[k][m].imag = 0.0;

      for(l=1;l<(j-1);l++) {
        cmaddf_an(&(U.e[l-1][k-1]), &(h->e[l][m]), &ht.e[k][m]);
        //entry = conjgf(&(U.e[l-1][k-1]));
        //entry = cmulf(&entry,&(h->e[l][m]));

        //      entry.real = 0.0;
        //      entry.imag = 0.0;


        //ht.e[k][m] = caddf(&(ht.e[k][m]),&entry);

        }
    }
    }


    cmat_copy(&ht, h);

    cmat_copy(Q, &qt);


    for(k=0;k<(h->n);k++) {
      for(m=1;m<(j-1);m++) {
      qt.e[k][m].real = 0.0;
      qt.e[k][m].imag = 0.0;

      for(l=1;l<j-1;l++) {
        cmaddf(&(Q->e[k][l]),&(U.e[l-1][m-1]), &(qt.e[k][m]));
        //entry = cmulf(&(Q->e[k][l]),&(U.e[l-1][m-1]));

        //qt.e[k][m] = caddf(&(qt.e[k][m]),&entry);

      }
      }
    }

    cmat_copy(&qt,Q);

    cmat_free(&U);

  }

  cmat_copy(Q, &qt);

#ifdef TRACE_STATISTICS
  s_nDefRotate++;
  t1 = s_cputimef ? s_cputimef() : 0.0;
#endif

  error = rotate_basis(eig_vec, &qt, size, qt.n);

#ifdef TRACE_STATISTICS
  if (s_cputimef)
    s_deflateRotationT += s_cputimef() - t1;
#endif


  if (error != 0){
      error = 10 * error - 2;
      goto cleanup;
  }

  // Doesn't work, copy makes no difference
#ifdef EIG_DEBUG

  printf( "scaling f by %g+%gi\n",
       Q->e[h->n-1][h->n-1].real,
       Q->e[h->n-1][h->n-1].imag);
#endif
  fieldvec_complex_mult_V(f, (Q->e[h->n-1][h->n-1]), f, size);
  //fieldvec_copy_V(ftemp, f, size);
  //wvec_cmplx_mult_V(f, &(Q->e[h->n-1][h->n-1]), ftemp, parity);
  //wvec_copy_V(ftemp, f, parity);

cleanup:
#ifdef TRACE_STATISTICS
  if (s_cputimef){
#ifdef MULTI_GPU
      cudaDeviceSynchronize();
#endif
      s_deflateT += s_cputimef() - t0;
  }
#endif
  //if (ftemp) free_fieldtype(ftemp);

  cmat_free(&Qadj);
  cmat_free(&ht);
  cmat_free(&qt);

  if (z) s_freef(z);
  return error;
  //free(z);
}




/*
 * Implictly restarted Arnoldi method.(why not to use this directly?)
 */
static
int arnoldi(ColorSpinorFieldSet *eig_vec_s,// vectors
         lcomplex* eig_val,     // values
         int n_eigs,            // number of eigenvalues
         radix tolerance,       // max absolute error (in Arnoldi)
         int* arn_iters,        // how many arnoldi iterations max - on exit returns the number of iterations used
         int n_eigs_extend,     // eigenvectors recomputed at restart
         multisize size,        // Size of the whole system in unit of number of field-entries in fieldtype.
         arnmode mode)          // Mode (arnmode_LM, arnmode_LR, arnmode_SM, arnmode_SR)
{

  int error = 0;
  int j, l, m;
  int steps, converged;

#ifdef TRACE_STATISTICS
  double t0 = s_cputimef ? s_cputimef() : 0.0;
#endif


//  const int max_iters = 200;  // not used w. lapack

  fieldtype **eig_vec = (fieldtype**)s_mallocf(sizeof(fieldtype*) * eig_vec_s->CompositeDim());
  for(j = 0; j < eig_vec_s->CompositeDim(); j++)  fieldtype *eig_vec = static_cast<fieldtype *> (eig_vec_s->Component(j).V());

  fieldtype *f = nullptr, *wvec_tmp  = nullptr;
  cmatrix h, ritzvec, hdiag, R, Q, deflateQ, Hnew, Qnew;
  lcomplex temp/*, prod*/;


  // TODO: These have to come from stack - at least on cpu-versions
  // Otherwise (at least on some x86) givens_rotate suffers a lot as
  // the system struggles with cache-priorities
  //lcomplex q[n_eigs+n_eigs_extend];
  //lcomplex qtemp[n_eigs+n_eigs_extend];
  //lcomplex givens[2*(n_eigs + n_eigs_extend)];

  lcomplex* q = (lcomplex*)s_mallocf(sizeof(lcomplex) * (n_eigs+n_eigs_extend));
  lcomplex* qtemp = (lcomplex*)s_mallocf(sizeof(lcomplex) * (n_eigs+n_eigs_extend));
  lcomplex* givens = (lcomplex*)s_mallocf(sizeof(lcomplex) * 2 * (n_eigs+n_eigs_extend));

  /*lcomplex *q;
  lcomplex *qtemp;
  lcomplex *givens;*/

  cmatrix hsmall, rsmall;

  //lcomplex alpha;
  double magf;

  //lcomplex *y  = NULL;
  //lcomplex y[n_eigs+n_eigs_extend];
  lcomplex* y = (lcomplex*)s_mallocf(sizeof(lcomplex) * (n_eigs+n_eigs_extend));

  lcomplex theta;

  int deflated_this_time = 0;

#ifdef EIG_DEBUG
  double timer = 0.0;
#endif

#ifdef TRACE_STATISTICS
  s_nOPxs = 0;
  s_nArnSteps = 0;

  s_totArnT = 0.0;
  s_parallelT = 0.0;
  s_OPxTime = 0.0;
  s_totArnStepT = 0.0;
  s_orthT = 0.0;
  s_deflateT = 0.0;
  s_GivensT = 0.0;
  s_restartRotationT = 0.0;
  s_deflateRotationT = 0.0;
  s_geevFunT = 0.0;

  s_nCDotProds = 0;
  s_nMADDs = 0;
  s_nGivenss = 0;
  s_nDefRotate = 0;
  s_nShiftRotate = 0;

#endif // TRACE_STATISTICS

  converged = 0;


  /*q = (lcomplex*)s_mallocf(sizeof(lcomplex) * (n_eigs+n_eigs_extend));
  qtemp = (lcomplex*)s_mallocf(sizeof(lcomplex) * (n_eigs+n_eigs_extend));
  givens = (lcomplex*)s_mallocf(sizeof(lcomplex) * 2 *(n_eigs+n_eigs_extend));*/

  ColorSpinorParam csParam(eig_vec_s->Component(0));
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  //@f = new_fieldtype(size.size, size.nMulti);//just pointer assignement
  ColorSpinorFieldSet *f_s = ColorSpinorFieldSet::Create(csParam);
  f        = static_cast<fieldtype*>(f_s->V()); 
  //
  //@wvec_tmp = new_fieldtype(size.size, size.nMulti);//just pointer assignement
  ColorSpinorFieldSet *wvec_tmp_s = ColorSpinorFieldSet::Create(csParam);
  wvec_tmp = static_cast<fieldtype*>(wvec_tmp_s->V()); 

  hdiag = cmat_new(n_eigs+n_eigs_extend);
  ritzvec = cmat_new(n_eigs+n_eigs_extend);
  Q = cmat_new(n_eigs+n_eigs_extend);
  Qnew = cmat_new(n_eigs+n_eigs_extend);
  R = cmat_new(n_eigs+n_eigs_extend);

  h = cmat_new(n_eigs+n_eigs_extend);
  Hnew = cmat_new(n_eigs+n_eigs_extend);

  hsmall = cmat_new(h.n);
  rsmall = cmat_new(h.n);

  if (!q || !qtemp || !givens || !y)
      goto cleanup;

  // Check that allocations were successful
  if (!(f && wvec_tmp && hdiag.e && ritzvec.e && Q.e && R.e && h.e &&
        Hnew.e && hsmall.e && rsmall.e /*&& q && qtemp && givens*/)){
      error = -1;
      goto cleanup;
  }

  cmat_zero(&h);

  if (s_createStream && s_destroyStream && s_waitForStream){
      s_astreams = (void**)(s_mallocf(sizeof(void*) * (n_eigs+n_eigs_extend)));
      s_streamBufs = (void**)(s_mallocf(sizeof(void*) * (n_eigs+n_eigs_extend)));
      s_streamBufSizes = (size_t*)(s_mallocf(sizeof(size_t) * (n_eigs+n_eigs_extend)));
      for (j = 0; j < n_eigs+n_eigs_extend; j++){
          s_nStreams = j;
          s_astreams[j] = s_createStream();
          s_streamBufSizes[j] = 0;
          s_streamBufs[j] = NULL;
          if (!s_astreams[j]){
              error = -2;
              goto cleanup;
          }
      }
      s_nStreams = (n_eigs+n_eigs_extend);
  }

  // Step 2
  fieldvec_normalize_V(eig_vec[0], size);
  //wvec_normalize_V(eig_vec[0], parity);
  // Step 4
#ifndef SHIFT
  //Mat_mult(eig_vec[0],wvec_tmp);
  (*s_mulf)(eig_vec_s->Component(0), *wvec_tmp_s);
#else
  //shift_mult(eig_vec[0],wvec_tmp,size);//redo it
#endif

  // Step 6a
  h.e[0][0] = fieldvec_cdot_V(eig_vec[0],wvec_tmp,size);
  // Step 6b
  fieldvec_complex_msub_V(wvec_tmp, (h.e[0][0]), eig_vec[0], f, size);
  //wvec_cmplx_mult_sub_V(wvec_tmp, eig_vec[0], &(h.e[0][0]), f, parity);


  // Now we have one-column V, 1x1 H, and f is the residual

#ifdef EIG_DEBUG
  printf("Calling arnoldi_step wanting %d eigs\n",n_eigs);
#endif

#ifdef USE_LANCZOS//STOP HERE
  error = lanczos_step(1, n_eigs-1, eig_vec_s, &h, f_s, size);
#else
  error = arnoldi_step(1, n_eigs-1, eig_vec_s, &h, f_s, size);//?
#endif
  if (error != 0){
      error = 10 * error - 3;
      goto cleanup;
  }


  /*
   * A simple discussion of the restarts is algorithm 7 in
   * http://www.ecse.rpi.edu/~rjradke/papers/radkemathesis.pdf
   */
  for(steps=0; steps < *arn_iters; steps++) {
    magf = sqrt(fieldvec_magsq_V(f,size));
#ifdef EIG_DEBUG
    printf("|f|=%g\n", magf);
#endif
    /* Fragment of code that refreshes residual.
     * Should do this every time an eigenvalue converges??
     * No - probably makes things worse
     */
    /*
    if(deflated_this_time) {
      fprintf0(outf,"Replacing basis vector\n");

      gaussian_wvec_V(f, parity);
      wvec_normalize_V(f, parity);

      // Reorthogonalize the new residual
      for(j=0;j<n_eigs;j++) {

    alpha = wvec_dot_V(eig_vec[j],f,parity);

    wvec_cmplx_mult_sub_V(f, eig_vec[j], &alpha, wvec_tmp, parity);
    wvec_copy_V(wvec_tmp,f,parity);
      }
    }
    */


#ifdef EIG_DEBUG
    if (s_cputimef)
        timer = s_cputimef();

    printf("Arnoldi restart step %d\n",steps);
    printf("Extending to %d+%d factorisation\n",n_eigs,n_eigs_extend);
#endif

#ifdef USE_LANCZOS
     error = lanczos_step(n_eigs, n_eigs_extend, eig_vec_s, &h, f_s, size);
#else
     error = arnoldi_step(n_eigs, n_eigs_extend, eig_vec_s, &h, f_s, size);//??
#endif
     if (error != 0){
         error = 10 * error - 4;
         goto cleanup;
     }

#ifdef EIG_DEBUG
    printf("Done. Now calculating eigenvalues\n");
#endif


  redeflate:

    // Pop out the submatrix that needs diagonalised
    cmat_submat(&h,&hsmall);

    // ... and diagonalise it
    error = numerical_eigs_dense(&hsmall, &rsmall);
    if (error != 0){
        error = 10 * error - 5;
        goto cleanup;
    }
    cmat_sortmag_eigenvectors(&hsmall, &rsmall, mode);

    /* NB for testing with Hermitian matrices, the Hessenberg is
     * symmetric, then one can use:
     */
    // Jacobi(&hdiag,&ritzvec,JACOBI_TOL);




    /*
     * Test against convergence criteria; see ARPACK documentation.
     * We test that the product of the last component of Ritz vectors
     * with the residual (the subdiagonal term in the Hessenberg)
     * is sufficiently small, but there is no way to evaluate
     * the error for a nonsymmetric eigenvalue problem.
     */
#ifdef EIG_DEBUG
    {
        lcomplex hdet, ct;

        hdet.real = 1.0;
        hdet.imag = 1.0;

        for(j=0;j<(rsmall.n);j++) {
          ct = cmulf(&hdet,&hsmall.e[j][j]);
          hdet = ct;
        }

        printf("hdet is %g\n", CABSF(&hdet));
    }
#endif
    deflated_this_time = 0;

    for(j=0;j<(rsmall.n);j++) {
      // Removed hdet, not replaced with norm (norm difficult to calculate)
      if((CABSF(&rsmall.e[rsmall.n-1][j])*magf/1.0 < tolerance)) {

#ifdef EIG_DEBUG
    printf("deflating because eig j=%d (%g+%gi) converged\n",
         j, hsmall.e[j][j].real, hsmall.e[j][j].imag);
#endif


    //y = (lcomplex *)s_mallocf((h.n - converged)*sizeof(lcomplex));

    for(l=0;l<(h.n-converged);l++) {
      y[l].real = rsmall.e[l][j].real;
      y[l].imag = rsmall.e[l][j].imag;
    }

    theta = hsmall.e[j][j];


    cmat_submat(&h,&hsmall);

    deflateQ = cmat_new(hsmall.n);

    error = deflate(&eig_vec[converged], &hsmall, f, y, theta, size, &deflateQ);
    if (error != 0){
        error = 10 * error - 6;
        cmat_free(&deflateQ);
        //s_freef(y);
        goto cleanup;
    }


    cmat_copy(&h,&Hnew);
    cmat_popmat(&hsmall,&Hnew);

    // H[m][l] = H[m][p]*Q[p][l]
    // probably not necessary, may make it worse
    /*
    for(m=0;m<converged;m++) {
      for(l=0;l<h.n-converged;l++) {

        Hnew.e[m][l+converged].real = 0.0;
        Hnew.e[m][l+converged].imag = 0.0;
        for(p=0;p<h.n-converged;p++) {
          prod = cmulf(&h.e[m][p], &deflateQ.e[p][l]);
          prod.imag = prod.imag;
          Hnew.e[m][l+converged] = caddf(&Hnew.e[m][l+converged], &prod);


        }
      }
    }
    */

    cmat_free(&deflateQ);
    //s_freef(y);

    cmat_copy(&Hnew,&h);

    converged++;

    deflated_this_time = 1;

    cmat_free(&hsmall);
    cmat_free(&rsmall);

    hsmall = cmat_new(h.n - converged);
    rsmall = cmat_new(h.n - converged);

    if(converged == n_eigs)
      break;

    // Wuhuu!
    goto redeflate;

      }
    }


    cmat_copy(&h,&hdiag);
    error = numerical_eigs_dense(&hdiag, &ritzvec);
    if (error != 0){
        error = 10 * error - 6;
        goto cleanup;
    }

    cmat_sortmag_eigenvectors(&hdiag, &ritzvec, mode);


    if(converged == n_eigs)
      break;

#ifdef EIG_INFO
    printf("Arnoldi: step %d, converged: %d\n", steps, converged);
#endif




#ifdef EIG_DEBUG
    for(j = 0; j < n_eigs; j++) {

      // NB: removed hdet
      printf("Retained Ritz value %d: %g+%gi (convergence=%g)\n", j,
           hdiag.e[j][j].real, hdiag.e[j][j].imag,
           CABSF(&ritzvec.e[n_eigs+n_eigs_extend-1][j])*magf);

    }
    for(j = n_eigs; j < (n_eigs+n_eigs_extend); j++) {
      printf("Restart Ritz value %d: %g+%gi\n", j,
           hdiag.e[j][j].real, hdiag.e[j][j].imag);
    }
#endif

    // Reset vector q to (0,0,...,1)
    for(j=0;j<(n_eigs + n_eigs_extend - 1);j++) {
      q[j].real = 0.0;
      q[j].imag = 0.0;
    }


    q[n_eigs + n_eigs_extend - 1].real = 1.0;
    q[n_eigs + n_eigs_extend - 1].imag = 0.0;
    {
        /*fieldtype **tmp_givens = (fieldtype **) s_mallocf((n_eigs+n_eigs_extend)*sizeof(fieldtype*));
        int i;
        for (i=0;i<n_eigs+n_eigs_extend;i++)
            tmp_givens[i] = new_fieldtype(size.size, size.nMulti);
        */
        for(j=n_eigs;j<(n_eigs + n_eigs_extend);j++) {

          // Could incorporate into loop
          cmat_copy(&h, &R);

          /*
           * Implicit shift
           * H - \sigma_j*I
           * we just factor this matrix, we don't need it after this loop
           */

          for(l=0;l<(n_eigs+n_eigs_extend);l++)
              R.e[l][l] = csubf(&R.e[l][l], &hdiag.e[j][j]);

          /*
           * Factor H = QR
           *
           * Very fast, O(n_eigs + n_eigs_extend)
           *
           * It's about 1% of the CPU time for this loop.
           */
          QRfactor_hessenberg_givens(&R, givens);

          // Similarity transformation for h
          apply_givens_hessenberg(givens, &h);
          right_givens(&h, givens);

          /*
           * Same thing happens to our Ritz vectors
           * This step was incredibly slow on a single machine, when
           * we had to calculate:
           *
           * givens_to_matrix(givens,&Q);
           * rotate_basis(eig_vec, &Q, parity);
           *
           * Now pretty fast that givens_rotate(...) works.
           */
#if !USE_POST_ROTATE_CODEPATH
          MANGLE(givens_rotate)(givens, eig_vec, size, n_eigs+n_eigs_extend);
#endif

          /*
           * Vector q = qt Q
           *
           * < 1% CPU time for the loop.
           *
           * Unfortunately still need givens_to_matrix,
           * luckily it is not a huge burden (<< 1% execution time).
           * Something to optimise later.
           */
          givens_to_matrix(givens,&Q);
#if USE_POST_ROTATE_CODEPATH
          if (j == n_eigs){
              cmat_copy(&Q,&Qnew);
          }
          else
          {
              #if 1
                right_givens(&Qnew, givens);
              #else
                cmat_mult(&Qnew,&Q,&R);
                cmat_copy(&R,&Qnew);
              #endif
          }
#endif
          for(l = 0; l < (n_eigs + n_eigs_extend); l++) {
            // Should this not be: conjgf(&q[l]);
            qtemp[l] =  q[l];
          }

          for(l = 0; l < (n_eigs + n_eigs_extend); l++) {
              q[l].real = 0.0;
              q[l].imag = 0.0;

            for(m=0; m<(n_eigs+n_eigs_extend); m++) {
              temp = cmulf(&qtemp[m],&(Q.e[m][l]));
              q[l].real += temp.real;
              q[l].imag += temp.imag;
            }
          }
        }
        /*for (i=0;i<n_eigs+n_eigs_extend;i++)
          free_fieldtype(tmp_givens[i]);
        s_freef(tmp_givens);*/
    }
#if USE_POST_ROTATE_CODEPATH
    #ifdef TRACE_STATISTICS
        {
            double t2 = s_cputimef ? s_cputimef() : 0;
            s_nShiftRotate++;
    #endif // TRACE_STATISTICS

        // Do: eig_vecs[i] = sum_j eig_vecs[j] Q_ji
        rotate_basis(eig_vec, &Qnew, size, /*Qnew.n*/ n_eigs);


    #ifdef TRACE_STATISTICS
            if (s_cputimef)
                s_restartRotationT += s_cputimef() - t2;
        }
    #endif // TRACE_STATISTICS
#endif //USE_POST_ROTATE_CODEPATH
#undef USE_POST_ROTATE_CODEPATH

    /* Calculate:
     * f = V[n_eigs]*h[n_eigs][n_eigs-1] + f.q[n_eigs-1]
     */
    fieldvec_complex_mult_V(f, q[n_eigs-1], wvec_tmp, size);
    fieldvec_complex_madd_V(wvec_tmp, h.e[n_eigs][n_eigs-1], eig_vec[n_eigs], f, size);
    //wvec_cmplx_mult_V(f, &(q[n_eigs-1]), wvec_tmp, parity);
    /*wvec_cmplx_mult_add_V(wvec_tmp, eig_vec[n_eigs],
              &(h.e[n_eigs][n_eigs-1]), f, parity);*/

#ifdef EIG_DEBUG
    if (s_cputimef)
        printf("iteration took %lf\n", s_cputimef() - timer);
#endif

  }

#ifdef EIG_DEBUG
  printf("End of IRAM: took %d steps\n",steps);
#endif

  if(converged < n_eigs) {
    printf("WARNING! Arnoldi algorithm did not converge!\n");
  }
  else {
      *arn_iters = steps;
  }




#ifdef EIG_DEBUG
  printf("Number, ritz value (re,im) and convergence vs criterion\n");

  for(j=0;j<n_eigs;j++) {

    printf("%d %g %g %g %g\n", j, hdiag.e[j][j].real, hdiag.e[j][j].imag,
            CABSF(&ritzvec.e[n_eigs+n_eigs_extend-1][j])*CABSF(&h.e[j+1][j]),
         tolerance*CABSF(&hdiag.e[j][j]));


#ifdef EIG_DEBUG_FULL

    // Neither are needed...
    //  rotate_basis(eig_vec, &ritzvec, parity);
    //  givens_rotate(givens, eig_vec, parity, n_eigs+n_eigs_extend);

#ifndef SHIFT
    //Mat_mult(eig_vec[j], wvec_tmp);
    (*s_mult)(eig_vec_s->Component(j), *wvec_tmp_s);
#else
    //shift_mult(eig_vec[j], wvec_tmp, size);
#endif

    temp =  fieldvec_cdot_V(wvec_tmp, eig_vec[j], size);

    double norm = sqrt(fieldvec_magsq_V(eig_vec[j],size));

    printf("Eigenvector norm, %g, inner product: %g+%gi\n",
         norm, temp.real, temp.imag);

    for(l=0;l<j;l++) {

      temp =  fieldvec_cdot_V(eig_vec[j], eig_vec[l], size);
      printf("Inner product with vector %d: %g+%gi\n", l, temp.real, temp.imag);

    }
#endif


  }
#endif

#ifndef SHIFT
  for(j=0;j<(n_eigs + n_eigs_extend);j++) {
    eig_val[j] = hdiag.e[j][j];
  }
#else

  //printf("Deshifting.\n");
  for(j=0;j<n_eigs;j++) {
    //Mat_mult(eig_vec[j], wvec_tmp);
    (*s_mulf)(eig_vec_s->Component(j), *wvec_tmp_s);

    temp = fieldvec_cdot_V(wvec_tmp, eig_vec[j], size);

    eig_val[j] = temp;
  }

#endif

#ifdef TRACE_STATISTICS
  printf("Arnoldi statistics:\n\n");
  printf("Total number of arnoldi steps: %d\n", s_nArnSteps);
  printf("Total number of OP*x ops: %d\n\n", s_nOPxs);

  printf("Number of complex (vector) dot-products: %d\n", s_nCDotProds);
  printf("Number of complex (vector) multiply-adds: %d\n", s_nMADDs);
  printf("Number of Deflation basis rotations: %d\n", s_nDefRotate);
  printf("Number of Shift-apply basis rotations: %d\n", s_nShiftRotate);
  printf("Number of Givens rotations: %d\n", s_nGivenss);
  if (s_cputimef){
      s_totArnT += s_cputimef() - t0;
      printf("Arnoldi Timing data:\n");
      printf("Total time in Arnoldi method: %f\n", s_totArnT);
      // Count matrix-vector operation as a parallel call
      printf("Time in parallel calls: %f\n", s_parallelT + s_OPxTime);
      printf("Time in serial (and MPI parallel) calls: %f\n", s_totArnT - s_OPxTime - s_parallelT);
      printf("Time in Arnoldi-step calls: %f\n", s_totArnStepT);
      printf("Time in OP*x ops (matrix-vector mul) %f\n", s_OPxTime);
      printf("Time in reorthogonalizations: %f\n\n", s_orthT);
      printf("Time in Deflate method: %f\n", s_deflateT);
      printf("Time in Givens rotate method: %f\n", s_GivensT);
      printf("Time in Arnoldi restart rotations: %f\n", s_restartRotationT);
      printf("Time in Deflation rotations: %f\n", s_deflateRotationT);
      printf("Time in small, dense geev-function method: %f\n", s_geevFunT);

  }

#endif

  // Clean up
cleanup:
  //if (f) free_fieldtype(f);
  //if (wvec_tmp) free_fieldtype(wvec_tmp);
  if(f_s) delete f_s;
  if(wvec_tmp_s) delete wvec_tmp_s;
  if(eig_vec) s_freef(eig_vec);

  if (s_destroyStream && s_createStream && s_astreams && s_waitForStream){
      for (j = 0; j < n_eigs + n_eigs_extend; j++){
          if (s_astreams[j]){
              s_destroyStream(s_astreams[j]);
              freeStreamBuffer(j);
          }
      }
      if (s_astreams) s_freef(s_astreams);
      if (s_streamBufs) s_freef(s_streamBufs);
      if (s_streamBufSizes) s_freef(s_streamBufSizes);
  }

  if (q) s_freef(q);
  if (qtemp) s_freef(qtemp);
  if (givens) s_freef(givens);

  if (y) s_freef(y);

  cmat_free(&hdiag);
  cmat_free(&ritzvec);
  cmat_free(&Q);
  cmat_free(&Qnew);
  cmat_free(&R);
  cmat_free(&h);
  cmat_free(&Hnew);
  cmat_free(&hsmall);
  cmat_free(&rsmall);

  return error;
}



/*
 * Find eigenvalues using Arnoldi.
 * Note: For a simple vector use size = number of entries in vector, nMulti = 1, stride = 1 (or 0???)
 */
static
int
run_arnoldiabs(
        int n_eig, int n_extend, radix tolerance,
        ColorSpinorFieldSet* e_vecs_s, lcomplex* e_vals,
        int* maxiter, int size, int nMulti, int stride,
        arnmode mode)
{

  int i;
  int error = 0;

  /* The restarts begin after n_eig and continue to
   *  n_eig_tot = n_eig + n_eig_extend
   *  -- all dense operations are O(f(n_eig_tot))
   */

  int n_eig_tot = n_eig + n_extend;

  int startfree = n_eig_tot;

  multisize msize;

  msize.nMulti = nMulti;
  msize.size   = size;
  msize.stride = stride;

  if (!e_vecs_s){
      //printf("Error in parameters to run_arnoldi() e_vecs is NULL!\n");
      return -1;
  }
  if (!e_vecs_s){
      //printf("Error in parameters to run_arnoldi() e_vecs[0] has to be the non-zero starting vector to Arnoldi method!!\n");
      return -2;
  }
  if (!e_vals){
      //printf("Error in parameters to run_arnoldi() e_vals is NULL!\n");
      return -3;
  }

  if (error == 0) error = 10 * arnoldi(e_vecs_s, e_vals, n_eig, tolerance, maxiter, n_extend, msize, mode);

  return error;
}


#endif // ARNOLDI_GENERIC_H_


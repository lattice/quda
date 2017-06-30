/*
 * this is a modified version of zarnoldi.c file, original header zarnoldi.c:
 *
 *  Created on: 7.8.2013
 *      Author: Teemu Rantalaiho
 *
 *
 *  Copyright 2013 Teemu Rantalaiho
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

#include <quda_arpack_interface.h>

using namespace quda;

#define RADIX_SIZE 64

// For normal complex types: Float2
typedef struct fieldtype_s
{
   double re;
   double im;
} fieldtype;

typedef struct fieldentry_s
{
   double re;
   double im;
} fieldentry;

#define MANGLE(X) zarnoldi_##X
#include "arnoldi_generic_quda.h"

#ifndef PRINT_FREE
#define PRINT_FREE(X)
#endif

#ifndef PRINT_MALLOC
#define PRINT_MALLOC(X)
#endif

//static cputimefunT s_cputimef = (cputimefunT)0;

static arnoldi_abs_int s_functions;

fieldtype* new_fieldtype(int size, int nMulti){
    fieldtype* result;
    {
#ifdef MULTI_GPU
      cudaMalloc(&result, nMulti * size * sizeof(fieldentry));
      PRINT_MALLOC(result);
#else
      result = (fieldtype*)s_mallocf(nMulti * size * sizeof(fieldentry));
#endif
    }
    return result;
}
void free_fieldtype(fieldtype* f){
    {
#ifdef MULTI_GPU
      cudaFree(f);
      PRINT_FREE(f);
#else
      s_freef(f);
#endif
    }
}

__device__
void get_fieldEntry(const fieldtype* f, int i, fieldentry* result, int multiIdx, int stride){
    fieldtype tmp = f[i+multiIdx*stride];
    result->re = tmp.re;
    result->im = tmp.im;
}

__device__
void set_fieldEntry(fieldtype* f, int i, const fieldentry* entry, int multiIdx, int stride){
    fieldtype tmp;
    tmp.re = entry->re;
    tmp.im = entry->im;
    f[i+multiIdx*stride] = tmp;
}

__device__
lcomplex fieldEntry_dot(const fieldentry* a, const fieldentry* b){
    lcomplex res;
    // (z1*) * z2 = (x1 - iy1)*(x2 + iy2) = x1x2 + y1y2 + i(x1y2 - y1x2)
    res.real = a->re * b->re + a->im * b->im;
    res.imag = a->re * b->im - a->im * b->re;
    return res;
}

__device__
radix fieldEntry_rdot(const fieldentry* a, const fieldentry* b){
    return a->re * b->re + a->im * b->im;
}

// dst = scalar * a
__device__
void fieldEntry_scalar_mult(const fieldentry* a, radix scalar, fieldentry* dst ){
    dst->re = scalar * a->re;
    dst->im = scalar * a->im;
}

// dst = a + scalar * b
__device__
void fieldEntry_scalar_madd(const fieldentry * restrict a, radix scalar, const fieldentry * restrict b, fieldentry * restrict dst ){
    dst->re = a->re + scalar * b->re;
    dst->im = a->im + scalar * b->im;
}

// dst = scalar * a
__device__
void fieldEntry_complex_mult(const fieldentry* a, lcomplex scalar, fieldentry* dst ){
    dst->re = scalar.real * a->re - scalar.imag * a->im;
    dst->im = scalar.real * a->im + scalar.imag * a->re;
}

// dst = a + scalar * b
__device__
void fieldEntry_complex_madd(const fieldentry * restrict a, lcomplex scalar, const fieldentry * restrict b, fieldentry * restrict dst ){
    dst->re = a->re + scalar.real * b->re - scalar.imag * b->im;
    dst->im = a->im + scalar.real * b->im + scalar.imag * b->re;
}

typedef struct init_arn_vec_in_s {
    const fieldtype* src;
    fieldtype* dst;
    int stride;
} init_arn_vec_in;

PARALLEL_KERNEL_BEGIN(init_arn_vec, init_arn_vec_in, in, i, multiIdx)
{
    fieldentry x;
    if (in.src){
        get_fieldEntry(in.src, i, &x, multiIdx, in.stride);
    }
    else {
        if (i == 0)
            x.re = 1.0;
        else
            x.re = 0.0;
        x.im = 0.0;
    }
    set_fieldEntry(in.dst, i, &x, multiIdx, in.stride);
}
PARALLEL_KERNEL_END()

// Note - with a normal vector of complex numbers, use nMulti = 1, stride = 0
// use nMulti = 1, stride = 0, for QUDA_FLOAT2_FIELD_ORDER ?
int arnoldiGPUSolve(
        dcomplex_t* results, const ColorSpinorField* init_vec, ColorSpinorFieldSet* rvecs, int size, int nMulti, int stride,
        int n_eigs, int n_extend, double tolerance, int* maxIter,
        const arnoldi_abs_int* functions, arnmode mode)
{
    if( (rvecs->CompositeDim() < (n_eigs + n_extend)) ) errorQuda("Wrong eigenspace dimension.\n");
    if(nMulti != 1 && stride != 0) errorQuda("Unsupported option.\n");

    //@fieldtype** e_vecs = (fieldtype**)s_mallocf(sizeof(fieldtype*) * (n_eigs + n_extend));//move outside!!!!
    ColorSpinorFieldSet *e_vecs = rvecs;

    dcomplex_t* e_vals = (dcomplex_t*)s_mallocf(sizeof(dcomplex_t) * (n_eigs + n_extend));

    if (!e_vals) errorQuda("Could not allocate host memory.\n");

    {
       init_arn_vec_in in;
       in.dst = static_cast<fieldtype*>(e_vecs->V());
       in.src = static_cast <const fieldtype*>(init_vec->V());
       in.stride = stride;
       KERNEL_CALL(init_arn_vec, in, 0, size, nMulti);
    }

    // Set necessary function pointers
    s_mulf = functions->mvecmulFun;//DiracMatrix

    scalar_reduction_f  = functions->scalar_redFun;
    complex_reduction_f = functions->complex_redFun;

    int error = run_arnoldiabs(n_eigs, n_extend, tolerance, e_vecs, (lcomplex*)e_vals, maxIter, size, nMulti, stride, mode);

    for (int i = 0; i < n_eigs; i++) results[i] = e_vals[i];

    if (e_vals)   s_freef(e_vals);

    return error;
}



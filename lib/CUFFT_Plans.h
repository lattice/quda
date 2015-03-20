
#ifndef CUFFT_PLANS_H
#define CUFFT_PLANS_H

#include <quda_internal.h>
#include <quda_matrix.h>
#include <cufft.h>

/*-------------------------------------------------------------------------------*/
#define CUFFT_SAFE_CALL( call) {                                      \
    cufftResult err = call;                                         \
    if( CUFFT_SUCCESS != err) {                                     \
        fprintf(stderr, "CUFFT error in file '%s' in line %i.\n",   \
                __FILE__, __LINE__);                                \
        exit(EXIT_FAILURE);                                         \
    } }
/*-------------------------------------------------------------------------------*/




inline void ApplyFFT(cufftHandle &plan, float2 *data_in, float2 *data_out, int direction){

    CUFFT_SAFE_CALL(cufftExecC2C(plan, (cufftComplex *)data_in, (cufftComplex *)data_out, direction));
}

inline void ApplyFFT(cufftHandle &plan, double2 *data_in, double2 *data_out, int direction){

    CUFFT_SAFE_CALL(cufftExecZ2Z(plan, (cufftDoubleComplex *)data_in, (cufftDoubleComplex *)data_out, direction));
}



void SetPlanFFTMany( cufftHandle &plan, int4 size, int dim, float2 *data){
    switch(dim){
    case 1:		
    {int n[1] = {size.w};
            cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.x * size.y * size.z);
    }
    break;
    case 3:
    {int n[3] = {size.x, size.y, size.z};
        cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.w);
    }
    break;
    }
    //printf("Created %dD FFT Plan in Single Precision\n", dim);
}

inline void SetPlanFFTMany( cufftHandle &plan, int4 size, int dim, double2 *data){
    switch(dim){
    case 1:		
    {int n[1] = {size.w};
            cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.x * size.y * size.z);
    }
    break;
    case 3:
    {int n[3] = {size.x, size.y, size.z};
        cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.w);
    }
    break;
    }
    //printf("Created %dD FFT Plan in Double Precision\n", dim);
}


inline void SetPlanFFT2DMany( cufftHandle &plan, int4 size, int dim, float2 *data){
    switch(dim){
    case 0:		
    {int n[2] = {size.w, size.z};
            cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.x * size.y);
    }
    break;
    case 1:
    {int n[2] = {size.y, size.x};
        cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.z * size.w);
    }
    break;
    }
    //printf("Created 2D FFT Plan in Single Precision\n");
}

inline void SetPlanFFT2DMany( cufftHandle &plan, int4 size, int dim, double2 *data){
    switch(dim){
    case 0:		
    {int n[2] = {size.w, size.z};
            cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.x * size.y);
    }
    break;
    case 1:
    {int n[2] = {size.y, size.x};
        cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.z * size.w);
    }
    break;
    }
    //printf("Created 2D FFT Plan in Double Precision\n");
}


#endif



#include "float3.h"

// dst += prefactor * dot(a,b)
extern "C" __global__ void
projection(float* __restrict__ kx, float* __restrict__ ky, float* __restrict__ kz,
           float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
           int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float km = kx[i]*mx[i]+ky[i]*my[i]+kz[i]*mz[i];
        kx[i] = kx[i] - km*mx[i];
        ky[i] = ky[i] - km*my[i];
        kz[i] = kz[i] - km*mz[i];
    }
}


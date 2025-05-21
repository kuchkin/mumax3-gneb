#include "float3.h"

// dst += prefactor * dot(a,b)
extern "C" __global__ void
projection4D(float* __restrict__ kx, float* __restrict__ ky, float* __restrict__ kz, float* __restrict__ kw,
             float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz, float* __restrict__ mw,
           int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float km = kx[i]*mx[i]+ky[i]*my[i]+kz[i]*mz[i]+kw[i]*mw[i];
        kx[i] = kx[i] - km*mx[i];
        ky[i] = ky[i] - km*my[i];
        kz[i] = kz[i] - km*mz[i];
        kw[i] = kw[i] - km*mw[i];
    }
}
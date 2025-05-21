
#include "float3.h"

// dst += prefactor * dot(a,b)
extern "C" __global__ void
dotproduct1(float* __restrict__ dst, float prefactor,
           float* __restrict__ ax, float* __restrict__ bx, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        // dst[i] = 0.0;
        dst[i] += prefactor * ax[i] * bx[i];
    }
}
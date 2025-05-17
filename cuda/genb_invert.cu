
#include "float3.h"

// dst += prefactor * dot(a,b)
extern "C" __global__ void
invert(float* __restrict__ dst, float* __restrict__ src, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        if(src[i]==0.) {
            dst[i] = 0.0;
        }else{
            dst[i] = 1./src[i];
        }
    }
}


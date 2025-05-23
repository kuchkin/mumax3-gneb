
#include "float3.h"

// dst += prefactor * dot(a,b)
extern "C" __global__ void
dotproduct3(float* __restrict__ dst, float prefactor,
            float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az, float* __restrict__ aw,
            float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz, float* __restrict__ bw, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        // dst[i] = 0.0;
        float ab = ax[i]*bx[i] + ay[i]*by[i] + az[i]*bz[i] + aw[i]*bw[i];
        float tx = bx[i] - ab*ax[i];
        float ty = by[i] - ab*ay[i];
        float tz = bz[i] - ab*az[i];
        float tw = bw[i] - ab*aw[i];

        dst[i] = prefactor * (tx*tx+ty*ty+tz*tz+tw*tw);
    }
}
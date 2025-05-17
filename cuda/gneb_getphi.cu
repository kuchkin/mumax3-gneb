
#include "float3.h"


// dst += prefactor * dot(a,b)
extern "C" __global__ void
getphi(float* __restrict__ dst,float* __restrict__ src, float prefactor,
           float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
           float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
           int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};

        float3 vecnom = cross(A, B);
        float SIN = sqrt(vecnom.x*vecnom.x + vecnom.y*vecnom.y + vecnom.z*vecnom.z)/(len(A)*len(B));

        float at = atan2f(SIN, src[i]);

        dst[i] = at*at;
    }
}



#include "float3.h"

// dst += prefactor * dot(a,b)
extern "C" __global__ void
copy_to_subspace(float* __restrict__ v0x, float* __restrict__ v0y, float* __restrict__ v1x,
           float* __restrict__ v1y, float* __restrict__ w2x, float* __restrict__ w2y,
           float* __restrict__ hwx, float* __restrict__ hwy,
           int N, int id, float alpha, float beta) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        

        v1x[i] = w2x[i];
        v1y[i] = w2y[i];

        w2x[i] = hwx[i] - alpha*v1x[i];
        w2y[i] = hwy[i] - alpha*v1y[i];

        if(id>0){
            w2x[i] -= beta*v0x[i];
            w2y[i] -= beta*v0y[i];
        }
        v0x[i] = v1x[i];
        v0y[i] = v1y[i];
    }
}


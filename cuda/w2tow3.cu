#include <stdint.h>
#include "float3.h"



// generate u1 u2
extern "C" __global__ void
w2tow3(float* __restrict__ u1x,  float* __restrict__  u1y,  float* __restrict__ u1z,
         float* __restrict__ u2x,  float* __restrict__  u2y,  float* __restrict__ u2z,
         float* __restrict__ w2x,  float* __restrict__  w2y,
         float* __restrict__ w3x,  float* __restrict__  w3y,  float* __restrict__ w3z,  int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {

        w3x[i] = w2x[i]*u1x[i] + w2y[i]*u2x[i];
        w3y[i] = w2x[i]*u1y[i] + w2y[i]*u2y[i];
        w3z[i] = w2x[i]*u1z[i] + w2y[i]*u2z[i];


    }
}

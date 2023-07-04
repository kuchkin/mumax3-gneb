#include <stdint.h>
#include "float3.h"



// generate u1 u2
extern "C" __global__ void
get_hw(float* __restrict__ k0x,  float* __restrict__  k0y,  float* __restrict__ k0z,
    float* __restrict__ kx,  float* __restrict__  ky,  float* __restrict__ kz,
         float* __restrict__ u1x,  float* __restrict__  u1y,  float* __restrict__ u1z,
         float* __restrict__ u2x,  float* __restrict__  u2y,  float* __restrict__ u2z,
          float* __restrict__ hwx,  float* __restrict__  hwy,int N, float epsilon) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {


       float Hx = (-k0x[i]+kx[i])/epsilon;
       float Hy = (-k0y[i]+ky[i])/epsilon;
       float Hz = (-k0z[i]+kz[i])/epsilon;

       hwx[i] = u1x[i]*Hx + u1y[i]*Hy + u1z[i]*Hz;
       hwy[i] = u2x[i]*Hx + u2y[i]*Hy + u2z[i]*Hz;
    }
}

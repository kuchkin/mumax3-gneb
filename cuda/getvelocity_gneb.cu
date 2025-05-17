#include <stdint.h>
#include "float3.h"



// velocity rotate
extern "C" __global__ void
get_velocity(float* __restrict__ vx,  float* __restrict__  vy,  float* __restrict__ vz,
    float* __restrict__ kx,  float* __restrict__  ky,  float* __restrict__ kz,
         float* __restrict__ m1,  float* __restrict__  m2,
         int N, int Nz) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float vf = 1.0;
        if(m1[i]<0){
            vf = 0.0;
        }else{
            vf = m1[i]/m2[i];
        }
        vx[i] = vf*kx[i];
        vy[i] = vf*ky[i];
        vz[i] = vf*kz[i];


    }
}

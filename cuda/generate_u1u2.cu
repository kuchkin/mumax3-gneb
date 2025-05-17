#include <stdint.h>
#include "float3.h"



// generate u1 u2
extern "C" __global__ void
generate_u1u2(float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
         float* __restrict__ u1x,  float* __restrict__  u1y,  float* __restrict__ u1z,
         float* __restrict__ u2x,  float* __restrict__  u2y,  float* __restrict__ u2z,
          int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {


        if(abs(mz[i])<0.5){
            u1x[i] = -mz[i]*mx[i];
            u1y[i] = -mz[i]*my[i];
            u1z[i] = 1.0-mz[i]*mz[i];
        }else{
            u1x[i] = 1.0-mx[i]*mx[i];
            u1y[i] = -mx[i]*my[i];
            u1z[i] = -mx[i]*mz[i];
        }

        float norm = 1.0/sqrt(u1x[i]*u1x[i] + u1y[i]*u1y[i] + u1z[i]*u1z[i]);
        u1x[i] *= norm; u1y[i] *= norm; u1z[i] *= norm;

        u2x[i] = -my[i]*u1z[i] + mz[i]*u1y[i];
        u2y[i] = -mz[i]*u1x[i] + mx[i]*u1z[i];
        u2z[i] = -mx[i]*u1y[i] + my[i]*u1x[i];


    }
}

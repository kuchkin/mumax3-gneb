#include <stdint.h>
#include "float3.h"



// velocity rotate
extern "C" __global__ void
velocity4D(float* __restrict__ vx,  float* __restrict__  vy,  float* __restrict__ vz, float* __restrict__ vw,
        float* __restrict__ kx,  float* __restrict__  ky,  float* __restrict__ kz, float* __restrict__ kw,
         float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz, float* __restrict__ mw,
         float* __restrict__ m0x,  float* __restrict__  m0y,  float* __restrict__ m0z, float* __restrict__ m0w,
         int N, int Nz) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    
    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 k = {kx[i], ky[i], kz[i]};


        float SIN = dot(m,k)  + mw[i]*kw[i];
        float COS = dot(m,m0) + mw[i]*m0w[i];
        
        // k = k*COS;
        // k -= m0*SIN;

        vx[i] = kx[i]*COS-m0x[i]*SIN;
        vy[i] = ky[i]*COS-m0y[i]*SIN;
        vz[i] = kz[i]*COS-m0z[i]*SIN;
        vw[i] = kw[i]*COS-m0w[i]*SIN;
        

        // if((vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]+vw[i]*vw[i])==1){
        //     vx[i] = 100.0;
        //     vy[i] = 100.0;
        //     vz[i] = 100.0;
        //     vw[i] = 100.0;
        // }
    }
}
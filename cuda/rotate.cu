#include <stdint.h>
#include "float3.h"



// generate u1 u2
extern "C" __global__ void
rotate(float* __restrict__ sx,  float* __restrict__  sy,  float* __restrict__ sz,
         float* __restrict__ vx,  float* __restrict__  vy,  float* __restrict__ vz,
         float* __restrict__ sxt,  float* __restrict__  syt,  float* __restrict__ szt,
         float* __restrict__ wx,  float* __restrict__  wy,  float* __restrict__ wz,
         float epsilon, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {


        float kx = sy[i]*wz[i] - sz[i]*wy[i];
        float ky = sz[i]*wx[i] - sx[i]*wz[i];
        float kz = sx[i]*wy[i] - sy[i]*wx[i];

        float norm = 1.0/sqrt(kx*kx+ky*ky+kz*kz);
        kx *= norm; ky *= norm; kz *= norm;

        float bx = ky*vz[i]-vy[i]*kz;
        float by = kz*vx[i]-vz[i]*kx;
        float bz = kx*vy[i]-vx[i]*ky;
        float ww = sqrt(wx[i]*wx[i]+wy[i]*wy[i]+wz[i]*wz[i]);
        float COS = cos(epsilon*ww);
        float SIN = sin(epsilon*ww);
        float ks = kx*vx[i]+ky*vy[i]+kz*vz[i];

        sxt[i] = vx[i]*COS - bx*SIN + kx*ks*(1.-COS);
        syt[i] = vy[i]*COS - by*SIN + ky*ks*(1.-COS);
        szt[i] = vz[i]*COS - bz*SIN + kz*ks*(1.-COS);
    }
}

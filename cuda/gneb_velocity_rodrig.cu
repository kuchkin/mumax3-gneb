#include <stdint.h>
#include "float3.h"



// velocity rotate
extern "C" __global__ void
velocity_rodrig(float* __restrict__ vx,  float* __restrict__  vy,  float* __restrict__ vz,
    float* __restrict__ kx,  float* __restrict__  ky,  float* __restrict__ kz,
         float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
         float* __restrict__ m0x,  float* __restrict__  m0y,  float* __restrict__ m0z,
         int N, int Nz, float dt) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    
    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 k = {kx[i], ky[i], kz[i]};

        float3 ds = k*dt;
        float theta = len(ds);
        float3 w = cross(m, ds);
        float p1 = dot(ds,k);
        float p2 = dot(w,k);
        float pref1, pref2;
        if(theta>0.005){
            pref1 = 1.0-theta*theta*(1.0-theta*theta/20.0)/6.0;
            pref2 = 0.5-theta*theta*(1.0-theta*theta/30.0)/24.0;
        }else{
            pref1 = sin(theta)/theta;
            pref2 = (1.0-cos(theta))/(theta*theta);
        }

        

        vx[i] = k.x*cos(theta)-p1*mx[i]*pref1 + w.x*p2*pref2;
        vy[i] = k.y*cos(theta)-p1*my[i]*pref1 + w.y*p2*pref2;
        vz[i] = k.z*cos(theta)-p1*mz[i]*pref1 + w.z*p2*pref2;
        
        // if(abs(vx[i] -1.0)<1.0e-5) vx[i] -= 1.0;
        // if(abs(vy[i] -1.0)<1.0e-5) vy[i] -= 1.0;
        // if(abs(vz[i] -1.0)<1.0e-5) vz[i] -= 1.0;


    }
}

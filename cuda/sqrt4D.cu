#include "float3.h"

// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
extern "C" __global__ void
sqrt4D(float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ vol,float* __restrict__ v4, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        // float v = (vol == NULL? 1.0f: vol[i]);
        // float3 V = {v*vx[i], v*vy[i], v*vz[i]};
        // float norm = v/sqrt(V.x*V.x+V.y*V.y+V.z*V.z+v4[i]*v4[i]);
        // vx[i] *= norm;
        // vy[i] *= norm;
        // vz[i] *= norm;
        float temp = vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i];
        if (temp >= 1.0){
            v4[i] = 0.0;
        }else{
             v4[i] = sqrt(1.0-temp);
        }
       
    }
}


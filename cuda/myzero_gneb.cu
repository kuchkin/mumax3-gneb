
#include "float3.h"


// dst += prefactor * dot(a,b)
extern "C" __global__ void
myzero(float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
           int Nx, int Ny, int Nz) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < Nx*Ny*Nz) {

        if((i/(Nx*Ny))%3!=0) {
           
            ax[i] = 0.0;
            ay[i] = 0.0;
            az[i] = 0.0;
        }
    }
}


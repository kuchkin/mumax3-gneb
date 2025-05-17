#include <stdint.h>
#include "float3.h"


extern "C" __global__ void
copyBeff(float* __restrict__  Bx2,  float* __restrict__  By2,  float* __restrict__  Bz2,
        float* __restrict__  Bx,  float* __restrict__  By,  float* __restrict__  Bz,
        int noi, int image, int Nx, int Ny, int Nz){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }
    int I = (iz*Ny + iy)*Nx + ix;
    float pref = 1.0;
    // if(iz%3!=0) pref = 0.0;
    iz = iz + image*Nz;
    int II = (iz*Ny + iy)*Nx + ix;
    Bx[II] = pref*Bx2[I];
    By[II] = pref*By2[I];
    Bz[II] = pref*Bz2[I];

    // if(iz/(Nz/noi)==image){
    //     Mx2[I] = Mx[((iz%(Nz/noi))*Ny + iy)*Nx + ix];
    //     My2[I] = My[((iz%(Nz/noi))*Ny + iy)*Nx + ix];
    //     Mz2[I] = Mz[((iz%(Nz/noi))*Ny + iy)*Nx + ix];
    // }

}


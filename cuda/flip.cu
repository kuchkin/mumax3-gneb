#include "stencil.h"

// flips 3D AFM.
extern "C" __global__ void
flip(float* __restrict__  dst, float* __restrict__  src,
       int Nx,  int Ny,  int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        if((ix + iy + iz)%2 == 0){
            dst[idx(ix, iy, iz)] = -1.0*src[idx(ix, iy, iz)];
        }else{
            dst[idx(ix, iy, iz)] = src[idx(ix, iy, iz)];
        }
    }
}


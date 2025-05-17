#include "stencil.h"

// shift dst by shy cells (positive or negative) along Z-axis.
// new edge value is clampL at left edge or clampR at right edge.
extern "C" __global__ void
shiftmagz(float* __restrict__  dstX,float* __restrict__  dstY,float* __restrict__  dstZ,
          float* __restrict__  srcX, float* __restrict__  srcY, float* __restrict__  srcZ,
          int Nx,  int Ny,  int Nz, int shz, float clampL, float clampR) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iz2 = iz-shz;
        float3 newval;
        if (iz2 < 0) {
            newval.x = clampL;
            newval.y = clampL;
            newval.z = clampL;
        } else if (iz2 >= Nz) {
            newval.x = clampR;
            newval.y = clampR;
            newval.z = clampR;

        } else {
            newval.x = srcX[idx(ix, iy, iz2)];
            newval.y = srcY[idx(ix, iy, iz2)];
            newval.z = srcZ[idx(ix, iy, iz2)];
        }
        dstX[idx(ix, iy, iz)] = newval.x;
        dstY[idx(ix, iy, iz)] = newval.y;
        dstZ[idx(ix, iy, iz)] = newval.z;
    }
}


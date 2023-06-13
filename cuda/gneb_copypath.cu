#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

#define MAX(a, b) ( (a>b)? (a)  :  (b) )
#define MIN(a, b) ( (a<b)? (a)  :  (b) )
#define ABS(a) ( (a>0.0)? (a)  :  (-1.0*a) )

// See gneb.go for more details.




extern "C" __global__ void
copypath(float* __restrict__ Tx, float* __restrict__ Ty, float* __restrict__ Tz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            int Nx, int Ny, int Nz,int noi){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int I = idx(ix, iy, iz);
    if(iz/(Nz/noi) == 0 || iz/(Nz/noi) == (noi-1)){
        Tx[I] = 0.0;
        Ty[I] = 0.0;
        Tz[I] = 0.0;
    }else{
        Tx[I] = mx[I];
        Ty[I] = my[I];
        Tz[I] = mz[I];
    }

    
}



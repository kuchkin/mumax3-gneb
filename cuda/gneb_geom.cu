#include <stdint.h>
#include "float3.h"



// Descent energy minimizer
extern "C" __global__ void
geom_vpo(float* __restrict__ Bx,  float* __restrict__  By,  float* __restrict__ Bz,float* __restrict__ vol,
          int N, int Nz) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {


 
        Bx[i] *= vol[i];
        By[i] *= vol[i];
        Bz[i] *= vol[i];
        


    }
}

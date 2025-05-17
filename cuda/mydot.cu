#include <stdint.h>
#include "float3.h"



// Descent energy minimizer
extern "C" __global__ void
mydot(float* __restrict__ sm,float* __restrict__ ax,  float* __restrict__  ay,  float* __restrict__ az,
         float* __restrict__ bx,  float* __restrict__  by,  float* __restrict__ bz, int N, int Nz) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    float temp = 0.0, vv = 0.0;
    if (i == 0) {
        for(int j=0; j<N; j++){
            vv = ax[j]*bx[j] +ay[j]*by[j]+az[j]*bz[j];
            // if(vv!=1){
                temp += vv;
            // }
            
        }
        // if(temp>4095) temp = 100000.0;
        sm[0] = temp;
    }
}
